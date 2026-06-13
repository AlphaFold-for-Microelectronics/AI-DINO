import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json
import math
import py3Dmol
import xraydb

import torch
import torch.nn.functional as F

from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from typing import Tuple, List, Dict, Optional, Union
from torch import Tensor
from matplotlib.figure import Figure

from aidino.xray_utils import wavelength_to_energy
from aidino.plot_utils import props, format_axis

_AXIS_DIRECTIONS = {
    'x': np.array([1, 0, 0]),
    'y': np.array([0, 1, 0]),
    'z': np.array([0, 0, 1]),
}

class Crystal:
    """
    Class representing a crystalline sample.
    """
    
    def __init__(
        self,
        cif_path: str,
        crystal_size: Optional[Tuple[int, int, int]] = None,
        wavelength: float = None,
        include_anomalous: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize a Crystal object.

        Parameters:
        -----------
        cif_path: str
            Path to cif file
        crystal_size: tuple, optional
            Tuple of integers (n1, n2, n3) specifying crystal size in unit cells.
            If omitted, set it later or let a field loader infer it.
        wavelength: float
            X-ray wavelength in meters
        include_anomalous: bool
            Whether to include anomalous scattering corrections (default: False)
        dtype: torch.dtype
            torch data type
        device: str
            torch device ('cuda' or 'cpu')
        """

        self.cif_path = cif_path
        self._crystal_size = tuple(crystal_size) if crystal_size is not None else None
        self.wavelength = wavelength
        self.dtype = dtype
        self.device = device

        if self.wavelength is None and include_anomalous:
            self.include_anomalous = False
            print('X-ray wavelength must be provided to include anomalous',
                  'scattering corrections. Setting include_anomalous to False.')
        else:
            self.include_anomalous = include_anomalous

        # Initialize reflections attribute
        self.reflections = None
        
        # Parse cif file
        self._parse_cif_file()

        # Rotation matrix tracking: cumulative rotation from original to current orientation
        self._rotation_matrix = torch.eye(3, dtype=self.dtype, device=self.device)

        # Lazy caches for tensors derived from self.structure / self.original_structure
        self._cached_lattice_vectors = None
        self._cached_original_lattice_vectors = None
        self._cached_atom_positions = None
        self._cached_atom_frac_coords = None

        # Parse atom types, occupancies, and site indices. Each pymatgen site
        # may carry a Composition with multiple species at fractional
        # occupancies (substitutional disorder). Expanding such sites into one
        # entry per species naturally implements the Virtual Crystal
        # Approximation in the per-atom kinematic sum:
        #     F_site(q) = Σ_species occupancy_species · f_species(q)
        # `_site_index` records which pymatgen site each expanded entry came
        # from, so atom_positions / atom_frac_coords re-broadcast the site's
        # coordinates across its expanded entries.
        self.atom_types: list = []
        self._occupancies: list = []
        self._site_index: list = []
        for site_idx, site in enumerate(self.structure.sites):
            for sp, occu in site.species.items():
                try:
                    elem_str = str(sp.element)
                except AttributeError:
                    elem_str = str(sp)
                self.atom_types.append(elem_str)
                self._occupancies.append(float(occu))
                self._site_index.append(site_idx)
        self._site_index = np.asarray(self._site_index, dtype=np.int64)

        # Get atomic form factors
        self._get_atomic_form_factor_coefficients()

        # Per-atom occupancy used by calculate_form_factors. Ones for fully-
        # ordered crystals; fractional values for disordered sites.
        self.occupancies = torch.as_tensor(
            self._occupancies, dtype=self.dtype, device=self.device,
        )

        # Set global position (default is origin)
        self._position = torch.zeros(3, dtype=self.dtype, device=self.device)

    def _parse_cif_file(self):
        """
        Parse the CIF file and extract the structure object.
        """
        parser = CifParser(self.cif_path)
        self.structure = parser.parse_structures(primitive=False)[0]

    def _load_atomic_form_factor_coefficients(self):
        """
        Load atomic form factor coefficients from JSON file.
        
        Parameters:
        -----------
        filename : str
            Path to the JSON file containing atomic form factor coefficients
            
        Returns:
        --------
        dict : Dictionary with element symbols as keys and coefficients as list of values
               Each coefficient list contains [a1, b1, a2, b2, a3, b3, a4, b4, c]
        """
        with open(os.path.join(os.path.dirname(__file__), 'resources/atomic_form_factors.json'), 'r') as f:
            data = json.load(f)
        
        return data
        
    def _get_atomic_form_factor_coefficients(self):
        """
        Extract atomic form factor coefficients.
        Includes anomalous scattering if X-ray wavelength is provided
        and self.include_anomalous is True.
        """        
        coeffs = self._load_atomic_form_factor_coefficients()

        # Always get Cromer-Mann coefficients for f0(q)
        a_np = np.empty((self.n_atoms, 4))
        b_np = np.empty((self.n_atoms, 4))
        c_np = np.empty((self.n_atoms,))

        # Initialize anomalous scattering arrays
        if self.include_anomalous:
            fp_np  = np.empty((self.n_atoms,))
            fpp_np = np.empty((self.n_atoms,))
            energy = wavelength_to_energy(self.wavelength)

        for i, atom_type in enumerate(self.atom_types):
            # Get Cromer-Mann coefficients
            a_np[i] = coeffs[atom_type][0:-1:2]
            b_np[i] = coeffs[atom_type][1:-1:2]
            c_np[i] = coeffs[atom_type][-1]

            # Get anomalous scattering factors if enabled
            if self.include_anomalous:
                fp_np[i]  = xraydb.f1_chantler(atom_type, energy)
                fpp_np[i] = xraydb.f2_chantler(atom_type, energy)

        self.coeff_a = torch.as_tensor(a_np, dtype=self.dtype, device=self.device).unsqueeze(0)
        self.coeff_b = torch.as_tensor(b_np, dtype=self.dtype, device=self.device).unsqueeze(0)
        self.coeff_c = torch.as_tensor(c_np, dtype=self.dtype, device=self.device).unsqueeze(0)
        if self.include_anomalous:
            self.f_prime        = torch.as_tensor(fp_np,  dtype=self.dtype, device=self.device).unsqueeze(0)
            self.f_double_prime = torch.as_tensor(fpp_np, dtype=self.dtype, device=self.device).unsqueeze(0)

    def _resolve_wavelength(self, wavelength: Optional[float]) -> float:
        """Return the explicit wavelength or fall back to self.wavelength."""
        if wavelength is not None:
            return wavelength
        if self.wavelength is None:
            raise TypeError("The X-ray wavelength must be provided.")
        return self.wavelength

    def _update_rotation_matrix(self):
        """
        Recompute _rotation_matrix from the ratio of current to original lattice matrices.
        Called internally after any rotation is applied.
        R is defined such that: L_curr = L_orig @ R.T  →  R = (L_orig^{-T} @ L_curr^T).T
        """
        L_orig = torch.tensor(
            self.original_structure.lattice.matrix, dtype=torch.float64
        )
        L_curr = torch.tensor(
            self.structure.lattice.matrix, dtype=torch.float64
        )
        # Solve L_orig @ R.T = L_curr  →  R.T = L_orig^{-1} @ L_curr
        R = torch.linalg.solve(L_orig, L_curr).T
        self._rotation_matrix = R.to(dtype=self.dtype, device=self.device)

    @property
    def position(self) -> Tensor:
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @position.deleter
    def position(self):
        del self._position

    @property
    def crystal_size(self) -> tuple:
        if self._crystal_size is None:
            raise RuntimeError(
                "crystal_size is not set. Pass it to Crystal(...) or let "
                "ExodusMesh.resample_to_crystal_grid fit it to the simulation box."
            )
        return self._crystal_size

    @crystal_size.setter
    def crystal_size(self, value):
        self._crystal_size = tuple(value) if value is not None else None
        
    @property
    def lattice_vectors(self) -> Tensor:
        if self._cached_lattice_vectors is None:
            self._cached_lattice_vectors = torch.tensor(
                self.structure.lattice.matrix * 1e-10, dtype=self.dtype, device=self.device
            )
        return self._cached_lattice_vectors

    @property
    def original_lattice_vectors(self) -> Tensor:
        if self._cached_original_lattice_vectors is None:
            source = getattr(self, 'original_structure', self.structure)
            self._cached_original_lattice_vectors = torch.tensor(
                source.lattice.matrix * 1e-10, dtype=self.dtype, device=self.device
            )
        return self._cached_original_lattice_vectors

    @property
    def rotation_matrix(self) -> Tensor:
        """
        Cumulative rotation matrix R from original to current crystal orientation.
        With pymatgen's row-vector lattice convention, L_current = L_original @ R.T.
        Identity if no rotation has been applied.
        """
        return self._rotation_matrix
        
    @property
    def atom_positions(self) -> Tensor:
        if self._cached_atom_positions is None:
            coords = self.structure.cart_coords[self._site_index]
            self._cached_atom_positions = torch.tensor(
                coords * 1e-10, dtype=self.dtype, device=self.device
            )
        return self._cached_atom_positions

    @property
    def atom_frac_coords(self) -> Tensor:
        if self._cached_atom_frac_coords is None:
            coords = self.structure.frac_coords[self._site_index]
            self._cached_atom_frac_coords = torch.tensor(
                coords, dtype=self.dtype, device=self.device
            )
        return self._cached_atom_frac_coords

    def _invalidate_structure_caches(self):
        """Drop cached tensors derived from self.structure after a rotation."""
        self._cached_lattice_vectors = None
        self._cached_atom_positions = None
        self._cached_atom_frac_coords = None
    
    @property
    def n_atoms(self) -> int:
        return len(self.atom_types)
        
    @property
    def n_cells(self) -> int:
        return self.crystal_size[0] * self.crystal_size[1] * self.crystal_size[2]

    @property
    def cell_volume(self) -> float:
        return torch.det(self.lattice_vectors).abs()

    @property
    def crystal_volume(self) -> float:
        return self.cell_volume * self.n_cells
    
    def print_size(self):
        """Print the physical dimensions of the crystal in nanometers."""
        L = self.original_lattice_vectors.cpu().numpy()  # [3, 3], meters
        dims_nm = np.linalg.norm(L, axis=1) * np.array(self.crystal_size) / 1e-9
        print(f"Crystal size: {dims_nm[0]:.1f} nm x {dims_nm[1]:.1f} nm x {dims_nm[2]:.1f} nm")
    
    @staticmethod
    def structure_to_xyz(structure):
        """
        Convert pymatgen Structure to XYZ string for py3Dmol.
        """
        lines = []
    
        natoms = len(structure)
        lines.append(str(natoms))
        lines.append("Generated from pymatgen Structure")
    
        for site in structure:
            # Pick the majority species at each site (XYZ format carries no
            # occupancy, so disordered sites are rendered as the dominant element).
            sp = max(site.species.items(), key=lambda kv: kv[1])[0]
            el = sp.symbol if hasattr(sp, 'symbol') else str(sp)
            x, y, z = site.coords
            lines.append(f"{el} {x:.6f} {y:.6f} {z:.6f}")
    
        return "\n".join(lines)

    def visualize_structure(self, crystal_size=None):
        """
        Visualize the crystal structure using py3Dmol.
        
        This method creates an interactive 3D visualization of the crystal structure,
        with options to display supercells. The visualization includes axes and uses
        a sphere representation for atoms.
        
        Parameters
        ----------
        crystal_size : tuple of int, optional
            The size of the crystal in unit cells to visualize.
            Default is None, which sets the size to that of the full crystal.
            If the resulting visualization would contain more than 50,000 atoms, the size
            will be automatically reduced to stay within this limit while attempting
            to maintain the aspect ratio.
        
        Returns
        -------
        None
            Displays the interactive 3D visualization.
        
        Notes
        -----
        The visualization is automatically rotated to provide a good initial viewing angle.
        The atom limit is set to 50,000 to ensure reasonable performance.
        """
        # Get the number of atoms in the base structure
        num_atoms_base = len(self.structure)
        
        # Calculate total atoms in the supercell
        if crystal_size is None:
            crystal_size = self.crystal_size
            
        total_atoms = num_atoms_base * crystal_size[0] * crystal_size[1] * crystal_size[2]
        
        # Check if we exceed the 50,000 atom limit
        if total_atoms > 50000:
            # Calculate the scale factor needed to stay under the limit
            scale_factor = (50000 / (num_atoms_base * crystal_size[0] * crystal_size[1] * crystal_size[2])) ** (1/3)
            
            # Apply the scale factor to each dimension, ensuring at least 1
            new_size = tuple(max(1, int(scale_factor * dim)) for dim in crystal_size)
            
            # Double-check and adjust if still over the limit
            while num_atoms_base * new_size[0] * new_size[1] * new_size[2] > 50000 and sum(new_size) > 3:
                # Reduce the largest dimension
                max_idx = new_size.index(max(new_size))
                new_size = list(new_size)
                new_size[max_idx] = max(1, new_size[max_idx] - 1)
                new_size = tuple(new_size)
            
            crystal_size = new_size
            final_atoms = num_atoms_base * crystal_size[0] * crystal_size[1] * crystal_size[2]
            print(f"Display size reduced to {crystal_size} to stay within 50,000 atom display limit.")
            print(f"Visualizing crystal of size {crystal_size} ({final_atoms} atoms)")
        else:
            print(f"Visualizing crystal of size {crystal_size} ({total_atoms} atoms)")
        
        # Create the supercell and convert to xyz format
        xyz = Crystal.structure_to_xyz(self.structure * crystal_size)
        
        # Set up the 3D view
        view = py3Dmol.view(width=500, height=500)
        
        view.removeAllModels()
        view.addModel(xyz, "xyz")
        view.setStyle({
            "sphere": {"scale": 0.25}
        })
        view.setViewStyle({
            "style": "outline",
            "outlineWidth": 0.02
        })
        
        # Initial orientation
        view.rotate(-90, "y")
        view.rotate(-100, "x")
        view.rotate(60, "z")
        view.rotate(20, "x")
        
        view.zoomTo()
        view.show()
            
    def get_xrd_pattern(self, wavelength: Optional[float] = None):
        """
        Calculate the XRD pattern of the structure at the given wavelength.
        
        Parameters:
        -----------
        wavelength: float
            Wavelength in meters.
            
        Returns:
        --------
        matplotlib.figure.Figure:
            Figure of the resulting XRD pattern
        """

        wavelength = self._resolve_wavelength(wavelength)
        calc = XRDCalculator(wavelength=wavelength * 1e10)
        pattern = calc.get_pattern(self.structure)

        # Plot and annotate XRD pattern
        fig, ax = plt.subplots(figsize=(10,3.5))
        calc.show_plot(self.structure, ax=ax)
        format_axis(ax, xlabel=r'$2\theta$ (deg.)', ylabel='Intensity (a.u.)', title='', xbins=None, ybins=None)

        # Format annotations
        angles = []
        hkls = []
        for i, annotation in enumerate(fig.axes[0].texts):
            text = annotation.get_text()
            annotation.set_text(text + f" ({pattern.x[i]:.2f}" + r"$^{\circ}$)")
            annotation.set_fontproperties(props)

            miller_indices = text.split(',')
            for indices in miller_indices:
                angles.append(pattern.x[i].item())
                hkls.append(tuple([int(k) for k in indices]))
            
        # Save valid reflections as a list of (hkl, angle) pairs so duplicate
        # hkls (e.g. one peak with multiple equivalent indices) are preserved.
        self.reflections = list(zip(hkls, angles))
        
        return fig

    def list_reflections(self, sort_by='value'):
        """
        Pretty print the reflections dictionary.
        
        Parameters:
        -----------
        sort_by : str, optional
            Sort by 'value' (default), 'hkl', or None for no sorting
        """
        if not self.reflections:
            print("No reflections to display. Please get_xrd_pattern() before listing reflections.")
            return
        
        # Prepare data for printing
        if sort_by == 'value':
            items = sorted(self.reflections, key=lambda x: x[1])
        elif sort_by == 'hkl':
            items = sorted(self.reflections, key=lambda x: x[0])
        else:
            items = list(self.reflections)
        
        # Print header
        print(f"{'h':>3} {'k':>3} {'l':>3}  {'2θ (degrees)':>15}")
        print("-" * 35)
        
        # Print each reflection
        for (h, k, l), angle in items:
            print(f"{h:>3} {k:>3} {l:>3}  {angle:>15.6f}")
        
        print("-" * 35)
        print(f"Total reflections: {len(self.reflections)}")
    
    def align_miller_plane_to_axis(self, miller_indices, target_axis='x'):
        """
        Rotate a crystal structure so that the normal to a specified Miller plane
        is aligned with a target axis.
        
        Parameters:
        -----------
        miller_indices : tuple
            Miller indices (h, k, l) of the plane
        target_axis : str
            Target axis ('x', 'y', or 'z')
        """
        
        if target_axis not in _AXIS_DIRECTIONS:
            raise ValueError("target_axis must be 'x', 'y', or 'z'")

        target_direction = _AXIS_DIRECTIONS[target_axis]

        # Save the original structure
        if not hasattr(self, 'original_structure'):
            self.original_structure = self.structure.copy()
        structure = self.original_structure.copy()
        
        # Get the reciprocal lattice
        reciprocal_lattice = structure.lattice.reciprocal_lattice
        
        # Calculate the normal vector to the Miller plane
        # Normal to (h,k,l) plane = h*a* + k*b* + l*c* (reciprocal lattice vectors)
        h, k, l = miller_indices
        miller_normal = (h * reciprocal_lattice.matrix[0] + 
                         k * reciprocal_lattice.matrix[1] + 
                         l * reciprocal_lattice.matrix[2])
        
        # Normalize the normal vector
        miller_normal = miller_normal / np.linalg.norm(miller_normal)
        
        # Calculate the rotation axis (cross product)
        rotation_axis = np.linalg.cross(miller_normal, target_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-10:  # Already aligned or anti-aligned
            if np.dot(miller_normal, target_direction) > 0:
                # Already aligned, no rotation needed
                self.structure = structure
                self._invalidate_structure_caches()
                self._update_rotation_matrix()
                return
            else:
                # Anti-aligned, need 180° rotation
                # Choose a perpendicular axis for 180° rotation
                rotation_axis = np.roll(target_direction, shift = 1)
                rotation_angle = np.pi
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # Calculate rotation angle
            rotation_angle = np.arccos(np.clip(np.dot(miller_normal, target_direction), -1.0, 1.0))
        
        # Apply rotation transformation
        transformation = RotationTransformation(rotation_axis, rotation_angle, angle_in_radians=True)
        rotated_structure = transformation.apply_transformation(structure)

        self.structure = rotated_structure
        self._invalidate_structure_caches()
        self._update_rotation_matrix()

    def misalign_about_axis(self, rotation_angle=0., rotation_axis='y'):
        """
        Rotate a crystal structure to misalign it.
        
        Parameters:
        -----------
        rotation_angle : float
            Angle in degrees by which to misalign the structure
        rotation_axis : str
            Rotation axis ('x', 'y', or 'z')
        """

        rotation_axis = _AXIS_DIRECTIONS[rotation_axis]

        # Save the original structure
        if not hasattr(self, 'original_structure'):
            self.original_structure = self.structure.copy()

        # Apply rotation transformation
        transformation = RotationTransformation(rotation_axis, rotation_angle, angle_in_radians=False)
        self.structure = transformation.apply_transformation(self.structure)
        self._invalidate_structure_caches()
        self._update_rotation_matrix()

    def calculate_form_factors(self, q_magnitude: Tensor) -> Tensor:
        """
        Calculate the approximate atomic form factors as a sum of gaussians for each atom over the given q values.
        
        Parameters:
        -----------
        q_magnitude : torch.Tensor
            q vector magnitude as a tensor of shape [batch_size, 1]

        Returns:
        --------
        torch.Tensor:
            Approximated form factor amplitudes of shape [batch_size, n_atoms]
        """
        # Calculate f0(q) = Σ[ai * exp(-bi * s²)] + c
        s_squared = (q_magnitude / (4 * torch.pi)) ** 2
        
        gaussian_terms = self.coeff_a.view(1,-1) * torch.exp(-self.coeff_b.view(1,-1) * s_squared)
        f0 = gaussian_terms.view(-1, self.n_atoms, 4).sum(dim=-1) + self.coeff_c

        if self.include_anomalous:
            # f(q,E) = f0(q) + f'(E) + i*f''(E)
            f = torch.complex(f0 + self.f_prime, self.f_double_prime)
        else:
            f = f0

        # Per-atom occupancy weight (VCA): for disordered sites, the expanded
        # per-species entries each contribute their fractional occupancy.
        return self.occupancies.view(1, -1) * f
            
    def get_penetration_depth(self, wavelength: Optional[float] = None) -> float:
        """
        Calculate linear attenuation coefficient and penetration depth.
        Uses NIST XCOM data via xraydb. Works for any energy from ~1 keV to 1 MeV.
        
        Parameters
        ----------
        wavelength: float
                Wavelength in meters.
        
        Returns
        -------
        penetration_depth : float
            1/e penetration depth in meters
        """
    
        wavelength = self._resolve_wavelength(wavelength)

        # Use xraydb's material_mu which handles formula strings
        # material_mu returns mu in 1/cm when density is provided
        mu_cm = xraydb.material_mu(
            self.structure.formula,
            wavelength_to_energy(wavelength),
            density=self.structure.density
        )
        
        # Convert to m^-1
        mu = mu_cm * 100
        
        # Calculate penetration depth (1/e depth)
        penetration_depth = 1.0 / mu
        
        return penetration_depth

    def create_displacement_field(self, batch_size=1, supercell_size=(1,1,1), scaling_factor=0.01, displacement_dict=None):
        """
        Generate a displacement field for atoms in a crystal structure.

        Creates either a deterministic displacement field from a provided dictionary,
        or a random displacement field scaled by a given factor.

        All displacement values are in **unit-cell fractional coordinates** (multiples
        of the lattice vectors), matching the input convention of
        ``BraggCoherentDiffraction.calculate_supercell_scattering``. A value of
        ``(0.01, 0, 0)`` means "shift this atom by 1% of the **a** lattice vector".

        Parameters
        ----------
        batch_size : int, optional
            Number of displacement fields to generate in the batch. Default is 1.
        supercell_size : tuple of int, optional
            Size of the supercell in each dimension (x, y, z). The crystal size is
            divided by this value per axis to determine the field shape. Default is (1, 1, 1).
        scaling_factor : float, optional
            Maximum magnitude of random displacements, in **fractional lattice
            coordinates**. Random draws are uniform on ``[-scaling_factor,
            scaling_factor]`` per Cartesian component. Only used when
            ``displacement_dict`` is None. Default is 0.01 (1% of a lattice vector).
        displacement_dict : dict or None, optional
            Dictionary mapping element symbols (str) to displacement vectors **in
            fractional lattice coordinates**. If provided, each atom type listed
            in the dictionary is assigned its corresponding fixed displacement
            vector, and all other atoms receive zero displacement. If None,
            displacements are drawn randomly as above. Default is None.
    
        Returns
        -------
        torch.Tensor
            Displacement field tensor of shape:
            ``(batch_size, crystal_x, crystal_y, crystal_z, *atom_positions.shape)``
            where ``crystal_k = self.crystal_size[k] // supercell_size[k]``.
            The tensor uses ``self.dtype`` and is located on ``self.device``.
        """
    
        if displacement_dict is not None:
            # Initialize the displacement field to zero for all atoms
            displacement_field = torch.zeros(
                (batch_size,) + tuple([self.crystal_size[k] // supercell_size[k] for k in range(3)]) +
                self.atom_positions.shape,
                dtype=self.dtype,
                device=self.device
            )
    
            # Assign each element its specified displacement vector
            for element, displacement in displacement_dict.items():
                displacement_field[..., self.atom_types.index(element), :] = displacement
    
        else:
            # Generate a random displacement field uniform in [-1, 1], then scale it
            displacement_field = scaling_factor * (
                2 * torch.rand(
                    (batch_size,) + tuple([self.crystal_size[k] // supercell_size[k] for k in range(3)]) +
                    self.atom_positions.shape,
                    dtype=self.dtype,
                    device=self.device
                ) - 1
            )
    
        return displacement_field
        
    def create_spherical_mask(
        self,
        center: Optional[Tuple[float, float, float]] = None,
        radius: Optional[float] = None,
        supercell_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tensor:
        """
        Create a spherical mask for the crystal.
        Assigns a shape_mask attribute of the resulting binary mask
        of shape [1, n1_grid, n2_grid, n3_grid].
    
        Parameters
        ----------
        center : tuple of float, optional
            Center of sphere in unit-cell coordinates.
            If None, uses crystal center.
        radius : float, optional
            Radius in unit-cell units.
            If None, uses half the minimum crystal dimension.
        supercell_size : tuple of int, optional
            (s1, s2, s3) supercell scaling factors.
            Grid dimensions are crystal_size / supercell_size.
        """
    
        # Original crystal size in unit cells
        c1, c2, c3 = self.crystal_size
    
        if supercell_size is None:
            s1 = s2 = s3 = 1
        else:
            s1, s2, s3 = supercell_size
    
        # Grid dimensions
        n1 = c1 // s1
        n2 = c2 // s2
        n3 = c3 // s3
    
        # Default center and radius are defined in unit-cell space
        if center is None:
            center = (c1 / 2.0, c2 / 2.0, c3 / 2.0)
    
        if radius is None:
            radius = min(c1, c2, c3) / 2.0
    
        # Grid indices
        i = torch.arange(n1, device=self.device, dtype=self.dtype)
        j = torch.arange(n2, device=self.device, dtype=self.dtype)
        k = torch.arange(n3, device=self.device, dtype=self.dtype)
    
        ii, jj, kk = torch.meshgrid(i, j, k, indexing="ij")
    
        # Map grid indices to unit-cell coordinates
        ii_uc = ii * s1
        jj_uc = jj * s2
        kk_uc = kk * s3
    
        # Distance in unit-cell space
        distances = torch.sqrt(
            (ii_uc - center[0]) ** 2
            + (jj_uc - center[1]) ** 2
            + (kk_uc - center[2]) ** 2
        )
    
        mask = (distances <= radius).to(self.dtype)
        self.shape_mask = mask.unsqueeze(0)

    def rotation_matrix_x(self, angle: float) -> Tensor:
        """
        Create rotation matrix for rotation around X axis.
        
        Parameters:
        -----------
        angle : float
            Rotation angle in radians
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=self.dtype, device=self.device)

    def rotation_matrix_y(self, angle: float) -> Tensor:
        """
        Create rotation matrix for rotation around Y axis.
        
        Parameters:
        -----------
        angle : float
            Rotation angle in radians
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=self.dtype, device=self.device)

    def rotation_matrix_z(self, angle: float) -> Tensor:
        """
        Create rotation matrix for rotation around Z axis.
        
        Parameters:
        -----------
        angle : float
            Rotation angle in radians
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=self.dtype, device=self.device)

    def rotation_matrix_euler(
        self,
        alpha: float,
        beta: float, 
        gamma: float
    ) -> Tensor:
        """
        Create rotation matrix R = Rz @ Ry @ Rx from Euler angles.
        Applied to a column vector v, R @ v rotates about the original X first,
        then the original Y, then the original Z (extrinsic XYZ, equivalently
        intrinsic ZYX).

        Parameters:
        -----------
        alpha : float
            Rotation around Z axis (radians)
        beta : float
            Rotation around Y axis (radians)
        gamma : float
            Rotation around X axis (radians)

        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        Rz = self.rotation_matrix_z(alpha)
        Ry = self.rotation_matrix_y(beta)
        Rx = self.rotation_matrix_x(gamma)

        return Rz @ Ry @ Rx

    def rotation_matrix_axis_angle(
        self,
        axis: Union[Tensor, Tuple[float, float, float]],
        angle: float
    ) -> Tensor:
        """
        Create rotation matrix from axis-angle representation (Rodrigues' formula).
        
        Parameters:
        -----------
        axis : Tensor or tuple
            Rotation axis (will be normalized). Can be:
            - torch.Tensor of shape [3]
            - Tuple of (x, y, z)
        angle : float
            Rotation angle in radians
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        if isinstance(axis, tuple):
            axis = torch.tensor(axis, dtype=self.dtype, device=self.device)
        else:
            axis = axis.to(device=self.device, dtype=self.dtype)
        
        # Normalize axis
        axis = axis / torch.norm(axis)
        
        # Rodrigues' rotation formula
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        K = torch.stack([
            torch.stack([zero,     -axis[2],  axis[1]]),
            torch.stack([axis[2],   zero,    -axis[0]]),
            torch.stack([-axis[1],  axis[0],  zero  ]),
        ])
        
        I = torch.eye(3, dtype=self.dtype, device=self.device)
        
        R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        return R
    
    def random_rotation_matrix(self, max_angle: Optional[float] = None) -> Tensor:
        """
        Generate a random rotation matrix.
        
        Uses the algorithm from "Fast Random Rotation Matrices" by James Arvo.
        This generates uniformly distributed random rotations.
        
        Parameters:
        -----------
        max_angle : float, optional
            If provided, limits rotation to angles up to max_angle (in radians)
            from identity. Useful for small perturbations.
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        if max_angle is not None:
            # Generate small rotation: random axis, limited angle
            # Random axis (normalized)
            axis = torch.randn(3, device=self.device)
            axis = axis / torch.norm(axis)
            
            # Random angle up to max_angle
            angle = torch.rand(1, device=self.device).item() * max_angle
            
            return self.rotation_matrix_axis_angle(axis, angle)
        else:
            # Generate uniformly distributed random rotation using Euler angles
            alpha = torch.rand(1).item() * 2 * math.pi  # [0, 2π]
            beta = torch.acos(2 * torch.rand(1) - 1).item()  # arccos(uniform[-1,1])
            gamma = torch.rand(1).item() * 2 * math.pi  # [0, 2π]
            
            return self.rotation_matrix_euler(alpha, beta, gamma)

    def rotate_mask(
        self,
        rotation_matrix: Optional[Tensor] = None,
        angles: Optional[Tuple[float, float, float]] = None,
        center: Optional[Tuple[float, float, float]] = None,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros'
    ) -> Tensor:
        """
        Rotate the stored 3D mask by applying a rotation matrix.

        The rotation is always applied to the original mask, and the result
        replaces self.shape_mask. This allows consecutive rotations to be
        applied to the original rather than compounding.
    
        The rotation is performed around the center of the mask (or specified center)
        using trilinear interpolation by default.
        
        Parameters:
        -----------
        rotation_matrix : torch.Tensor, optional
            3x3 rotation matrix. If None, will use angles parameter.
        angles : Tuple[float, float, float], optional
            Euler angles (alpha, beta, gamma) in radians for Z, Y, X rotations.
            Used if rotation_matrix is None.
        center : Tuple[float, float, float], optional
            Center of rotation in voxel coordinates ordered (n1, n2, n3) to
            match crystal_size. If None, uses mask center.
        mode : str
            Interpolation mode: 'bilinear' (default, actually trilinear in 3D) or 'nearest'
            - 'bilinear': Smooth but may blur edges
            - 'nearest': Sharp edges but aliasing artifacts
        padding_mode : str
            How to handle out-of-bounds: 'zeros', 'border', or 'reflection'
            
        Raises:
        -------
        AttributeError
            If self.shape_mask has not been initialized
            
        Notes:
        ------
        - The function is fully differentiable (gradients flow through rotation)
        - Uses PyTorch's grid_sample for efficient GPU computation
        - The rotation is applied in the mask's coordinate system
        
        Examples:
        ---------
        # Example 1: Rotate by specific Euler angles
        rotated = self.rotate_mask(angles=(math.pi/4, math.pi/6, 0))
        
        # Example 2: Random rotation
        R = random_rotation_matrix()
        rotated = self.rotate_mask(rotation_matrix=R)
        
        # Example 3: Rotate around specific point
        rotated = self.rotate_mask(angles=(0, math.pi/2, 0), center=(10, 20, 15))
        """
        # Check if shape_mask exists
        if not hasattr(self, 'shape_mask') or self.shape_mask is None:
            raise AttributeError(
                "shape_mask has not been initialized. Please create a shape mask "
                "before calling rotate_mask()."
            )

        # Save original mask on first rotation
        if not hasattr(self, 'original_shape_mask') or self.original_shape_mask is None:
            self.original_shape_mask = self.shape_mask.clone()
        
        mask = self.original_shape_mask
        
        # Handle input dimensions
        if mask.ndim == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, n1, n2, n3]
            squeeze_output = True
        elif mask.ndim == 4:
            mask = mask.unsqueeze(1)  # [batch, 1, n1, n2, n3]
            squeeze_output = False
        else:
            raise ValueError(f"Mask must be 3D or 4D, got {mask.ndim}D")
        
        batch_size, _, n1, n2, n3 = mask.shape
        
        # Get rotation matrix
        if rotation_matrix is None:
            if angles is None:
                raise ValueError("Must provide either rotation_matrix or angles")
            rotation_matrix = self.rotation_matrix_euler(*angles)
        else:
            rotation_matrix = rotation_matrix.to(self.device)
        
        # Get center of rotation
        if center is None:
            center = ((n1 - 1) / 2.0, (n2 - 1) / 2.0, (n3 - 1) / 2.0)
        
        # Create coordinate grid in normalized coordinates [-1, 1]
        # PyTorch's grid_sample expects coordinates in [-1, 1] range
        z = torch.linspace(-1, 1, n1, device=self.device)
        y = torch.linspace(-1, 1, n2, device=self.device)
        x = torch.linspace(-1, 1, n3, device=self.device)
        
        # Create meshgrid
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        
        # Stack to get [n1, n2, n3, 3]
        grid = torch.stack([xx, yy, zz], dim=-1)
        
        # Convert rotation matrix to work with normalized coordinates
        # We need to: denormalize -> rotate -> normalize
        
        # Scale factors to convert from [-1, 1] to voxel coordinates
        scale = torch.tensor([n3 - 1, n2 - 1, n1 - 1], device=self.device, dtype=self.dtype) / 2.0
        
        # Translation to center
        translation = torch.tensor([center[2], center[1], center[0]], device=self.device, dtype=self.dtype)
        
        # Flatten grid for matrix multiplication
        grid_flat = grid.view(-1, 3)  # [n1*n2*n3, 3]
        
        # Convert to voxel coordinates
        grid_voxel = grid_flat * scale + scale  # Now in [0, n-1]
        
        # Translate to center
        grid_centered = grid_voxel - translation
        
        # Apply rotation
        grid_rotated = grid_centered @ rotation_matrix.T
        
        # Translate back
        grid_voxel_rotated = grid_rotated + translation
        
        # Convert back to normalized coordinates [-1, 1]
        grid_normalized = (grid_voxel_rotated - scale) / scale
        
        # Reshape back to [n1, n2, n3, 3]
        grid_final = grid_normalized.view(n1, n2, n3, 3)
        
        # Expand for batch
        grid_final = grid_final.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        # Apply rotation using grid_sample
        # Note: grid_sample expects [batch, channels, depth, height, width]
        # and grid of shape [batch, depth, height, width, 3]
        rotated_mask = F.grid_sample(
            mask,
            grid_final,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True
        )
        
        # Remove channel dimension and optionally batch dimension
        if squeeze_output:
            rotated_mask = rotated_mask.squeeze(0).squeeze(0)  # [n1, n2, n3]
        else:
            rotated_mask = rotated_mask.squeeze(1)  # [batch, n1, n2, n3]
        
        # Update the current shape mask with the rotated result
        self.shape_mask = rotated_mask

    def rotate_mask_random(
        self,
        max_angle: Optional[float] = None,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros'
    ) -> Tuple[Tensor, Tensor]:
        """
        Rotate mask by a random rotation.
        
        Parameters:
        -----------
        max_angle : float, optional
            If provided, limits rotation angles. Useful for small perturbations.
        mode : str
            Interpolation mode
        padding_mode : str
            Padding mode
        """
        R = self.random_rotation_matrix(max_angle=max_angle)
        self.rotate_mask(rotation_matrix=R, mode=mode, padding_mode=padding_mode)

    def rotate_mask_axis(
        self,
        axis: Union[Tensor, Tuple[float, float, float]],
        angle: float,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros'
    ) -> Tensor:
        """
        Rotate mask around a specific axis.
        
        Parameters:
        -----------
        axis : Tensor or tuple
            Rotation axis (will be normalized)
        angle : float
            Rotation angle in radians
        mode : str
            Interpolation mode
        padding_mode : str
            Padding mode
        """
        R = self.rotation_matrix_axis_angle(axis, angle)
        return self.rotate_mask(rotation_matrix=R, mode=mode, padding_mode=padding_mode)

    def reset_mask_rotation(self):
        """Reset shape_mask to the original unrotated state."""
        if hasattr(self, 'original_shape_mask') and self.original_shape_mask is not None:
            self.shape_mask = self.original_shape_mask.clone()
        else:
            raise AttributeError("No original mask to reset to.")

def load_displacement_npz(filename, dtype=torch.float32, device='cpu'):
    """
    Load a .npz displacement file into a dict of torch tensors.

    Parameters
    ----------
    filename : str
        Path to the .npz file.
    dtype : torch.dtype
        Output dtype for the tensors.
    device : str
        Output device for the tensors.

    Returns
    -------
    dict[str, torch.Tensor]
        One tensor per array in the .npz, keyed by its array name.
    """
    data = np.load(filename)
    return {k: torch.tensor(data[k], dtype=dtype, device=device) for k in data}