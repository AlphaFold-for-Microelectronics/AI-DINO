import torch

from typing import Tuple, List, Dict, Optional, Union
from torch import Tensor

from aidino.sample import Crystal

class BraggCoherentDiffraction:
    """
    Class for simulating Bragg coherent diffraction.
    """
    
    def __init__(
        self,
        crystal: Crystal,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize a Bragg coherent diffraction simulator.
        
        Parameters:
        -----------
        crystal: Crystal
            An instance of a Crystal object
        dtype: torch.dtype
            torch data type
        device: str
            torch device ('cuda' or 'cpu')
        """
        self.crystal = crystal
        self.dtype = dtype
        self.device = device

    def _prepare_supercell_data(
        self,
        supercell_size: Tuple[int, int, int],
        mask: Optional[Tensor],
        continuum_displacement: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Compute supercell positions and preprocess mask and continuum displacement
        into flat [batch_size, n_supercells, ...] form. Called once before the
        q-batch loop so that neither mask downsampling nor displacement reshaping
        is repeated on every iteration.

        Returns
        -------
        supercell_positions         : [n_supercells, 3]
        supercell_mask_flat         : [batch_size, n_supercells] or None
        continuum_displacement_flat : [batch_size, n_supercells, 3] or None
        """
        d1, d2, d3 = supercell_size
        n1, n2, n3 = self.crystal.crystal_size

        # Check if dimensions divide evenly
        assert (n1 % d1 == 0) and (n2 % d2 == 0) and (n3 % d3 == 0)
        n_sc1, n_sc2, n_sc3 = n1 // d1, n2 // d2, n3 // d3

        # Generate supercell indices
        i_indices = torch.arange(0, n1, d1, dtype=self.dtype, device=self.device)
        j_indices = torch.arange(0, n2, d2, dtype=self.dtype, device=self.device)
        k_indices = torch.arange(0, n3, d3, dtype=self.dtype, device=self.device)

        # Compute a grid of all indices
        i, j, k = torch.meshgrid(i_indices, j_indices, k_indices, indexing='ij')

        # Reshape to [n_supercells, 3]
        supercell_indices = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=-1)

        # Calculate positions in real space
        # Result shape: [n_supercells, 3]
        supercell_positions = torch.matmul(supercell_indices, self.crystal.lattice_vectors)

        # Preprocess mask to [batch_size, n_supercells]
        supercell_mask_flat = None
        if mask is not None:
            # Convert mask to supercell level if needed
            if mask.shape[-3:] == (n1, n2, n3):
                # Mask is at unit cell resolution - need to downsample to supercell resolution
                # Reshape mask to group unit cells into supercells
                mask_grouped = mask.view(-1, n_sc1, d1, n_sc2, d2, n_sc3, d3)

                # Average over supercell dimensions
                supercell_mask_flat = torch.mean(mask_grouped, dim=(2, 4, 6)).flatten(start_dim=1)

            elif mask.shape[-3:] == (n_sc1, n_sc2, n_sc3):
                # Mask is already at supercell resolution
                supercell_mask_flat = mask.flatten(start_dim=1)

            else:
                raise ValueError(
                    f"Mask shape {mask.shape[-3:]} is incompatible with crystal grid "
                    f"of size ({n1}, {n2}, {n3}) or supercell grid ({n_sc1}, {n_sc2}, {n_sc3})"
                )

        # Preprocess continuum displacement to [batch_size, n_supercells, 3]
        continuum_displacement_flat = None
        if continuum_displacement is not None:
            continuum_displacement_flat = continuum_displacement.reshape(
                continuum_displacement.shape[0], -1, 3
            )

        return supercell_positions, supercell_mask_flat, continuum_displacement_flat

    def _compute_supercell_phase_factors(
        self,
        q_batch: Tensor,
        supercell_positions: Tensor,
        continuum_displacement_flat: Optional[Tensor],
        supercell_mask_flat: Optional[Tensor],
    ) -> Tensor:
        """
        Compute per-supercell phase factors for a batch of q-vectors,
        incorporating continuum displacement and mask.

        Returns
        -------
        phase_factors : [batch_size, n_pixels, n_supercells]
            Always has a batch dimension (batch_size=1 when no batch inputs are provided).
        """
        # Calculate q·R for each supercell and each q-vector
        # Result shape: [n_pixels, n_supercells]
        q_dot_R = torch.matmul(q_batch, supercell_positions.T)

        # Calculate e^(-iq·(R+u)) for each supercell and each q-vector,
        # incorporating continuum displacement u if provided
        if continuum_displacement_flat is not None:
            # Calculate q·u for each supercell and each q-vector
            # continuum_displacement_flat shape: [batch_size, n_supercells, 3]
            # Result shape: [batch_size, n_pixels, n_supercells]
            q_dot_u = torch.einsum(
                'pi,bni->bpn', q_batch,
                continuum_displacement_flat
            )
            phase_factors = torch.exp(-1j * (q_dot_R.unsqueeze(0) + q_dot_u))
        else:
            # Result shape: [1, n_pixels, n_supercells]
            phase_factors = torch.exp(-1j * q_dot_R).unsqueeze(0)

        # Apply mask if provided (maintains differentiability)
        if supercell_mask_flat is not None:
            # supercell_mask_flat shape: [batch_size, n_supercells] -> [batch_size, 1, n_supercells]
            phase_factors = phase_factors * supercell_mask_flat.unsqueeze(1)

        return phase_factors

    def calculate_structure_factor(self, q_vectors: Tensor) -> Tensor:
        """
        Calculate the structure factor for a given set of q-vectors.
        
        Parameters:
        -----------
        q_vectors: torch.Tensor
            Tensor of shape [..., 3] containing q-vectors
            
        Returns:
        --------
        torch.Tensor
            Structure factor as a complex tensor of shape [...]
        """
        
        # Store original shape of q_vectors for later reshaping
        q_size_original = q_vectors.shape[:-1]
        
        # Reshape q_vectors to [n_pixels, 3] for matrix multiplication
        q_vectors_flat = q_vectors.view(-1, 3)
        n_pixels = q_vectors_flat.shape[0]
        
        # Calculate q·r for each atom in the unit cell and each q-vector
        # q_vectors_flat shape: [n_pixels, 3]
        # atom_positions shape: [n_atoms, 3]
        # Result shape: [n_pixels, n_atoms]
        q_dot_r = torch.matmul(q_vectors_flat, self.crystal.atom_positions.T)
        
        # Calculate e^(-iq·r) for each atom and each q-vector
        phase_factors = torch.exp(-1j * q_dot_r)

        # Calculate |q| for form factor (convert to Å for typical form factor formulas)
        # Result shape: [n_pixels, 1]
        q_magnitude = torch.norm(q_vectors_flat, dim=-1, keepdim=True) * 1e-10

        # Vectorized calculation of form factors for all atoms and all q values
        form_factors = self.crystal.calculate_form_factors(q_magnitude)
        
        # Multiply by form factors and sum over atoms
        # Result shape: [n_pixels]
        structure_factor = torch.sum(form_factors * phase_factors, dim=-1)
        
        # Reshape back to original q_vectors shape
        structure_factor = structure_factor.view(q_size_original)
        
        return structure_factor

    def calculate_supercell_scattering(
        self,
        q_vectors: Tensor,
        supercell_size: Tuple[int, int, int],
        continuum_displacement: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        q_batch_size: Optional[int] = None) -> Tensor:
        
        """
        Calculate scattering using the supercell approach from Mokhtar et al.
        
        Parameters:
        -----------
        q_vectors: torch.Tensor
            Tensor of shape [..., 3] containing q-vectors
        supercell_size: Tuple
            Size of supercells (d1, d2, d3) in unit cells
        continuum_displacement: torch.Tensor, optional
            Tensor of shape [batch_size, n_sc1, n_sc2, n_sc3, 3] or [batch_size, n_supercells, 3].
            Per-supercell rigid shift in Cartesian coordinates in units of meters.
        mask: torch.Tensor, optional
            Optional mask of shape [batch_size, n1, n2, n3] or [batch_size, n_sc1, n_sc2, n_sc3]
        q_batch_size: int, optional
            Number of q-vectors to process at once. If None, process all at once.

        Returns:
        --------
        torch.Tensor
            Scattering amplitude as a complex tensor
        """

        d1, d2, d3 = supercell_size
        n1, n2, n3 = self.crystal.crystal_size

        supercell_positions, supercell_mask_flat, continuum_displacement_flat = \
            self._prepare_supercell_data(supercell_size, mask, continuum_displacement)

        # Store original shape of q_vectors for later reshaping
        q_size_original = q_vectors.shape[:-1]
        
        # Reshape q_vectors to [n_pixels, 3] for matrix multiplication
        q_vectors_flat = q_vectors.view(-1, 3)
        n_pixels = q_vectors_flat.shape[0]

        # Determine q_vector batch size
        if q_batch_size is None:
            q_batch_size = n_pixels
        
        # Process scattering calculations in q_vector batches
        scattering_results = []

        for i in range(0, n_pixels, q_batch_size):
            q_batch = q_vectors_flat[i:i+q_batch_size]

            # Calculate e^(-iq·(R+u)) for each supercell and each q-vector,
            # incorporating continuum displacement and mask
            # Result shape: [batch_size, n_pixels, n_supercells]
            phase_factors = self._compute_supercell_phase_factors(
                q_batch, supercell_positions,
                continuum_displacement_flat, supercell_mask_flat,
            )
            
            # Calculate structure factor for the unit cell and add a batch dimension
            S_q = self.calculate_structure_factor(q_batch).unsqueeze(0)
            
            # Number of unit cells per supercell
            cells_per_supercell = d1 * d2 * d3
            
            # Multiply structure factor by sum of phase factors and scale by supercell size
            # Result shape: [batch_size, n_pixels]
            scattering_batch = S_q * torch.sum(phase_factors, dim=-1) * cells_per_supercell
    
            # Add global position phase shift
            # Calculate q·R_g for each q-vector
            # grain_position shape: [3]
            # q_vectors_flat shape: [n_pixels, 3]
            # Result shape: [n_pixels]
            q_dot_global_position = torch.matmul(q_batch, self.crystal.position)
            
            # Calculate e^(-iq·R_g) for each q-vector
            global_phase_factor = torch.exp(-1j * q_dot_global_position)
            
            # Apply global position phase shift
            scattering_batch = scattering_batch * global_phase_factor.unsqueeze(0)

            # Append scattering results
            scattering_results.append(scattering_batch)

        # Concatenate along q-vector dimension
        scattering_amplitude = torch.cat(scattering_results, dim=1)
    
        # Reshape back to original q_vector shape
        scattering_amplitude = scattering_amplitude.view(-1, *q_size_original)
        
        return scattering_amplitude

    def calculate_supercell_scattering_with_displacements(
        self,
        q_vectors: Tensor,
        supercell_size: Tuple[int, int, int],
        sublattice_displacement: Tensor,
        lattice_strain: Optional[Tensor] = None,
        continuum_displacement: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        q_batch_size: Optional[int] = None
    ) -> Tensor:
        """
        Calculate scattering using the supercell approach from Mokhtar et al. with averaged displacements for each supercell.
        
        Parameters:
        -----------
        q_vectors: torch.Tensor
            Tensor of shape [..., 3] containing q-vectors
        supercell_size: Tuple
            Size of supercells (d1, d2, d3) in unit cells
        sublattice_displacement: torch.Tensor
            Tensor of shape [batch_size, n1, n2, n3, n_atoms, 3] or [batch_size, n_sc1, n_sc2, n_sc3, n_atoms, 3]
            containing the unique or supercell-averaged displacement field for each atom, respectively.
            Expressed in fractional coordinates relative to the original crystal lattice vectors.
        lattice_strain: torch.Tensor, optional
            Tensor of shape [batch_size, n_sc1, n_sc2, n_sc3, 3, 3] or [batch_size, n_supercells, 3, 3].
            Per-supercell lattice perturbation δL in Cartesian coordinates in units of meters.
        continuum_displacement: torch.Tensor, optional
            Tensor of shape [batch_size, n_sc1, n_sc2, n_sc3, 3] or [batch_size, n_supercells, 3].
            Per-supercell rigid shift in Cartesian coordinates in units of meters.
        mask: torch.Tensor, optional
            Optional mask of shape [batch_size, n1, n2, n3] or [batch_size, n_sc1, n_sc2, n_sc3]
        q_batch_size: int, optional
            Number of q-vectors to process at once. If None, process all at once.
            
        Returns:
        --------
        torch.Tensor
            Scattering amplitude as a complex tensor
        """
        
        d1, d2, d3 = supercell_size
        n1, n2, n3 = self.crystal.crystal_size
        n_sc1, n_sc2, n_sc3 = n1 // d1, n2 // d2, n3 // d3

        supercell_positions, supercell_mask_flat, continuum_displacement_flat = \
            self._prepare_supercell_data(supercell_size, mask, continuum_displacement)

        # Store original shape of q_vectors for later reshaping
        q_size_original = q_vectors.shape[:-1]
        
        # Reshape q_vectors to [n_pixels, 3] for matrix multiplication
        q_vectors_flat = q_vectors.view(-1, 3)
        n_pixels = q_vectors_flat.shape[0]

        # Calculate averaged sublattice displacements
        batch_size = sublattice_displacement.shape[0]
        n_atoms = sublattice_displacement.shape[-2]

        # Check displacement field dimensions and convert to supercell level if needed
        if sublattice_displacement.shape[-5:-2] == (n1, n2, n3):
            # Displacement field is at unit cell resolution - need to downsample to supercell resolution
            # Reshape to group unit cells into supercells: [batch_size, n_sc1, d1, n_sc2, d2, n_sc3, d3, n_atoms, 3]
            grouped = sublattice_displacement.view(batch_size, n_sc1, d1, n_sc2, d2, n_sc3, d3, n_atoms, 3)
            
            # Average over supercell dimensions (2, 4, 6) -> [batch_size, n_sc1, n_sc2, n_sc3, n_atoms, 3]
            sublattice_displacement_avg = torch.mean(grouped, dim=(2, 4, 6))
            
        elif sublattice_displacement.shape[-5:-2] == (n_sc1, n_sc2, n_sc3):
            # Displacement field is already at supercell resolution
            sublattice_displacement_avg = sublattice_displacement
            
        else:
            raise ValueError(
                f"Displacement field shape {sublattice_displacement.shape[-5:-2]} is incompatible with crystal grid "
                f"of size ({n1}, {n2}, {n3}) or supercell grid ({n_sc1}, {n_sc2}, {n_sc3})"
            )
        
        # Flatten to [batch_size, n_supercells, n_atoms, 3]
        sublattice_displacement_flat = sublattice_displacement_avg.view(batch_size, -1, n_atoms, 3)

        # Convert displacements from crystal fractional coordinates to Cartesian lab frame
        sublattice_displacement_flat = torch.matmul(sublattice_displacement_flat, self.crystal.lattice_vectors)

        # Incorporate lattice strain into sublattice displacement
        # A local strain δL perturbs the Cartesian position of atom m in supercell n by
        # δr_m^strain(n) = f_m @ δL(n), where f_m are the fractional coordinates of atom m.
        # This is precomputed once and added to sublattice_displacement_flat.
        if lattice_strain is not None:
            # Check dimensions and convert to supercell level if needed
            if lattice_strain.shape[-5:-2] == (n1, n2, n3):
                # Lattice strain is at unit cell resolution - downsample to supercell resolution
                # Reshape to group unit cells into supercells: [batch_size, n_sc1, d1, n_sc2, d2, n_sc3, d3, 3, 3]
                grouped_strain = lattice_strain.view(batch_size, n_sc1, d1, n_sc2, d2, n_sc3, d3, 3, 3)

                # Average over supercell dimensions (2, 4, 6) -> [batch_size, n_sc1, n_sc2, n_sc3, 3, 3]
                lattice_strain_flat = torch.mean(grouped_strain, dim=(2, 4, 6)).view(batch_size, -1, 3, 3)

            elif lattice_strain.shape[-5:-2] == (n_sc1, n_sc2, n_sc3):
                # Lattice strain is already at supercell resolution
                lattice_strain_flat = lattice_strain.view(batch_size, -1, 3, 3)

            else:
                raise ValueError(
                    f"Lattice strain shape {lattice_strain.shape[-5:-2]} is incompatible with crystal grid "
                    f"of size ({n1}, {n2}, {n3}) or supercell grid ({n_sc1}, {n_sc2}, {n_sc3})"
                )

            # Fractional coordinates of atoms from CIF structure (frame-invariant, dimensionless)
            # atom_fracs shape: [n_atoms, 3]
            atom_fracs = self.crystal.atom_frac_coords

            # Strain displacement for each atom in each supercell: δr_m = f_m @ δL(n)
            # atom_fracs shape:          [n_atoms, 3]
            # lattice_strain_flat shape: [batch_size, n_supercells, 3, 3]
            # Result shape:              [batch_size, n_supercells, n_atoms, 3] Cartesian meters, lab frame
            strain_displacement_flat = torch.einsum('mi,bnij->bnmj', atom_fracs, lattice_strain_flat)

            # Add to sublattice displacement — both are Cartesian meters in the lab frame
            # sublattice_displacement_flat shape: [batch_size, n_supercells, n_atoms, 3]
            sublattice_displacement_flat = sublattice_displacement_flat + strain_displacement_flat
            
        # Determine batch size
        if q_batch_size is None:
            q_batch_size = n_pixels

        # Process in batches
        scattering_results = []
        
        for i in range(0, n_pixels, q_batch_size):
            q_batch = q_vectors_flat[i:i+q_batch_size]
            
            # Calculate q·r for each atom in the unit cell and each q-vector
            # q_vectors_flat shape: [n_pixels, 3]
            # atom_positions shape: [n_atoms, 3]
            # Result shape: [n_pixels, n_atoms]
            q_dot_r = torch.matmul(q_batch, self.crystal.atom_positions.T)
    
            # Calculate e^(-iq·r) for each atom and each q-vector
            basis_phase_factors = torch.exp(-1j * q_dot_r)
    
            # Calculate q·u_m for each supercell, each atom in the unit cell, and each q-vector
            # q_batch shape: [n_pixels, 3]
            # sublattice_displacement_flat shape: [batch_size, n_supercells, n_atoms, 3]
            # Result shape: [batch_size, n_pixels, n_supercells, n_atoms]
            q_dot_displacements = torch.einsum('pi,bnmi->bpnm', q_batch, sublattice_displacement_flat)
    
            # Calculate e^(-iq·u_m) for each supercell, each atom in the unit cell, and each q-vector
            displacement_phase_factors = torch.exp(-1j * q_dot_displacements)
    
            # Calculate total phase factors for all supercells
            # total phase = basis_phase * displacement_phase
            # basis_phase_factors shape: [n_pixels, n_atoms] -> [1, n_pixels, 1, n_atoms]  
            # displacement_phase_factors shape: [batch_size, n_pixels, n_supercells, n_atoms]
            # Result shape: [batch_size, n_pixels, n_supercells, n_atoms]
            total_phase_factors = basis_phase_factors.unsqueeze(1).unsqueeze(0) * displacement_phase_factors
    
            # Calculate |q| for form factor (convert to Å for typical form factor formulas)
            # Result shape: [n_pixels, 1]
            q_magnitude = torch.norm(q_batch, dim=-1, keepdim=True) * 1e-10
    
            # Vectorized calculation of form factors for all atoms and all q values
            form_factors = self.crystal.calculate_form_factors(q_magnitude)
            
            # Calculate modified structure factors for all supercells
            # form_factors shape: [n_pixels, n_atoms] -> [1, n_pixels, 1, n_atoms]
            # Result shape: [batch_size, n_pixels, n_supercells]
            modified_structure_factors = torch.sum(form_factors.unsqueeze(1).unsqueeze(0) * total_phase_factors, dim=-1)
    
            # Apply mask to modified structure factors if provided (maintains differentiability)
            # Mask is applied here rather than to the supercell phase factors so that it
            # correctly zeros out the per-supercell structure factor contribution.
            # supercell_mask_flat shape: [batch_size, n_supercells] -> [batch_size, 1, n_supercells]
            # modified_structure_factors shape: [batch_size, n_pixels, n_supercells]
            if supercell_mask_flat is not None:
                modified_structure_factors = modified_structure_factors * supercell_mask_flat.unsqueeze(1)

            # Calculate e^(-iq·(R+u)) for each supercell and each q-vector,
            # incorporating continuum displacement. Mask is passed as None here
            # since it has already been applied to modified_structure_factors above.
            # Result shape: [batch_size, n_pixels, n_supercells]
            supercell_phase_factors = self._compute_supercell_phase_factors(
                q_batch, supercell_positions,
                continuum_displacement_flat, supercell_mask_flat=None,
            )
    
            # Number of unit cells per supercell
            cells_per_supercell = d1 * d2 * d3
            
            # Multiply modified structure factor by sum of phase factors and scale by supercell size
            # modified_structure_factors shape: [batch_size, n_pixels, n_supercells]
            # supercell_phase_factors shape: [batch_size, n_pixels, n_supercells]
            # Result shape: [batch_size, n_pixels]
            scattering_batch = torch.sum(modified_structure_factors * supercell_phase_factors, dim=-1) * cells_per_supercell
    
            # Add global position phase shift
            # Calculate q·R_g for each q-vector
            # grain_position shape: [3]
            # q_vectors_flat shape: [n_pixels, 3]
            # Result shape: [n_pixels]
            q_dot_global_position = torch.matmul(q_batch, self.crystal.position)
            
            # Calculate e^(-iq·R_g) for each q-vector
            global_phase_factor = torch.exp(-1j * q_dot_global_position)
            
            # Apply global position phase shift
            scattering_batch = scattering_batch * global_phase_factor.unsqueeze(0)

            # Append scattering results
            scattering_results.append(scattering_batch)

        # Concatenate along q-vector dimension
        scattering_amplitude = torch.cat(scattering_results, dim=1)
    
        # Reshape back to original q_vector shape
        scattering_amplitude = scattering_amplitude.view((batch_size,) + q_size_original)
        
        return scattering_amplitude