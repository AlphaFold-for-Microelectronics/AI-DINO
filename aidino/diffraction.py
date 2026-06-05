import torch

from typing import Tuple, List, Dict, Optional, Union
from torch import Tensor

from aidino.sample import Crystal

# q magnitude unit conversion: meters^-1 to angstroms^-1, for form-factor lookups
_Q_M_TO_INV_ANG = 1e-10

class BraggCoherentDiffraction:
    """
    Class for simulating Bragg coherent diffraction.
    """
    
    def __init__(self, crystal: Crystal):
        """
        Initialize a Bragg coherent diffraction simulator.

        Parameters:
        -----------
        crystal: Crystal
            An instance of a Crystal object. The simulator inherits dtype and
            device from this crystal so that all subsequent arithmetic stays
            consistent with the crystal's tensors.
        """
        self.crystal = crystal

    def _downsample_to_supercell(
        self,
        field: Tensor,
        supercell_size: Tuple[int, int, int],
        n_trailing: int = 0,
    ) -> Tensor:
        """
        Downsample a field whose spatial axes (n1, n2, n3) sit at positions
        [-3 - n_trailing : -n_trailing] to supercell resolution (n_sc1, n_sc2, n_sc3)
        by averaging over (d1, d2, d3). Returns the field unchanged if it is
        already at supercell resolution. Raises ValueError otherwise.

        n_trailing is the number of non-spatial dims AFTER the spatial block:
        0 for a mask [B, n_*, n_*, n_*], 2 for a per-atom or 3×3 field
        [B, n_*, n_*, n_*, n_atoms, 3] or [B, n_*, n_*, n_*, 3, 3].
        """
        d1, d2, d3 = supercell_size
        n1, n2, n3 = self.crystal.crystal_size
        n_sc1, n_sc2, n_sc3 = n1 // d1, n2 // d2, n3 // d3

        expected_ndim = 4 + n_trailing  # batch + 3 spatial + n_trailing
        if field.ndim != expected_ndim:
            raise ValueError(
                f"Field has ndim {field.ndim}, expected {expected_ndim} "
                f"(batch + 3 spatial + {n_trailing} trailing dims)."
            )

        spatial_end = field.ndim - n_trailing if n_trailing else field.ndim
        spatial_start = spatial_end - 3
        spatial_shape = tuple(field.shape[spatial_start:spatial_end])

        if spatial_shape == (n_sc1, n_sc2, n_sc3):
            return field
        if spatial_shape != (n1, n2, n3):
            raise ValueError(
                f"Field spatial shape {spatial_shape} must be {(n1, n2, n3)} "
                f"or {(n_sc1, n_sc2, n_sc3)}."
            )

        leading  = field.shape[:spatial_start]
        trailing = field.shape[spatial_end:]
        grouped = field.view(leading + (n_sc1, d1, n_sc2, d2, n_sc3, d3) + trailing)
        return torch.mean(grouped, dim=(spatial_start + 1, spatial_start + 3, spatial_start + 5))

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

        if (n1 % d1) or (n2 % d2) or (n3 % d3):
            raise ValueError(
                f"crystal_size ({n1}, {n2}, {n3}) must divide evenly by "
                f"supercell_size ({d1}, {d2}, {d3})."
            )
        n_sc1, n_sc2, n_sc3 = n1 // d1, n2 // d2, n3 // d3

        # Generate supercell indices (match the crystal's dtype/device)
        dtype, device = self.crystal.dtype, self.crystal.device
        i_indices = torch.arange(0, n1, d1, dtype=dtype, device=device)
        j_indices = torch.arange(0, n2, d2, dtype=dtype, device=device)
        k_indices = torch.arange(0, n3, d3, dtype=dtype, device=device)

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
            mask_sc = self._downsample_to_supercell(mask, supercell_size, n_trailing=0)
            supercell_mask_flat = mask_sc.flatten(start_dim=1)

        # Preprocess continuum displacement to [batch_size, n_supercells, 3]
        continuum_displacement_flat = None
        if continuum_displacement is not None:
            if continuum_displacement.ndim not in (3, 5):
                raise ValueError(
                    f"continuum_displacement has ndim {continuum_displacement.ndim}, "
                    f"expected 5 ([B, n1, n2, n3, 3]) or 3 ([B, n_supercells, 3])."
                )
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

    def _run_q_batches(
        self,
        q_vectors: Tensor,
        q_batch_size: Optional[int],
        per_batch_fn,
    ) -> Tensor:
        """
        Flatten q_vectors, loop over q-batches, apply the per-batch core function,
        apply the global position phase shift exp(-i q · R_g), concatenate, and
        reshape back to the original q-vector layout.

        per_batch_fn(q_batch) must return a complex tensor of shape
        [batch_size, n_pixels_in_batch] (batch_size may be 1).
        """
        q_size_original = q_vectors.shape[:-1]
        q_vectors_flat = q_vectors.view(-1, 3)
        n_pixels = q_vectors_flat.shape[0]
        if q_batch_size is None:
            q_batch_size = n_pixels

        results = []
        for i in range(0, n_pixels, q_batch_size):
            q_batch = q_vectors_flat[i:i + q_batch_size]
            # scattering_batch shape: [batch_size, n_pixels_in_batch]
            scattering_batch = per_batch_fn(q_batch)
            # Global position phase shift e^(-iq·R_g).
            # crystal.position shape: [3]; q_batch shape: [n_pixels_in_batch, 3]
            # global_phase shape: [n_pixels_in_batch] -> [1, n_pixels_in_batch] for broadcast
            global_phase = torch.exp(-1j * torch.matmul(q_batch, self.crystal.position))
            results.append(scattering_batch * global_phase.unsqueeze(0))

        # Concatenate along the q dimension, then reshape to [batch_size, *q_vectors.shape[:-1]].
        return torch.cat(results, dim=1).view(-1, *q_size_original)

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
        q_magnitude = torch.norm(q_vectors_flat, dim=-1, keepdim=True) * _Q_M_TO_INV_ANG

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

        Use this method when each supercell has the *same* unit-cell structure
        factor (i.e., no per-atom displacement within the supercell, only an
        optional rigid continuum_displacement). For per-atom sublattice
        displacement or lattice strain, use
        calculate_supercell_scattering_with_displacements instead.

        Parameters:
        -----------
        q_vectors: torch.Tensor
            Tensor of shape [..., 3] containing q-vectors
        supercell_size: Tuple
            Size of supercells (d1, d2, d3) in unit cells
        continuum_displacement: torch.Tensor, optional
            Tensor of shape [batch_size, n_sc1, n_sc2, n_sc3, 3] or [batch_size, n_supercells, 3].
            Per-supercell rigid shift in Cartesian lab-frame meters.
        mask: torch.Tensor, optional
            Per-supercell weight mask of shape [batch_size, n1, n2, n3] or
            [batch_size, n_sc1, n_sc2, n_sc3]. When passed at unit-cell
            resolution, the mask is downsampled by averaging — binary masks
            therefore yield fractional fill-fraction weights for partially
            occupied supercells (a "soft" support, not a hard threshold).
            Pass at supercell resolution if you want exact 0/1 weights.
        q_batch_size: int, optional
            Number of q-vectors to process at once. If None, process all at once.

        Returns:
        --------
        torch.Tensor
            Scattering amplitude as a complex tensor of shape
            [batch_size, *q_vectors.shape[:-1]].
        """

        d1, d2, d3 = supercell_size
        cells_per_supercell = d1 * d2 * d3

        supercell_positions, supercell_mask_flat, continuum_displacement_flat = \
            self._prepare_supercell_data(supercell_size, mask, continuum_displacement)

        def per_batch(q_batch: Tensor) -> Tensor:
            # Per-supercell phase factors e^(-iq·(R + u_continuum)) with mask folded in.
            # Result shape: [batch_size, n_pixels, n_supercells]
            phase_factors = self._compute_supercell_phase_factors(
                q_batch, supercell_positions,
                continuum_displacement_flat, supercell_mask_flat,
            )
            # The unit-cell structure factor S(q) is identical across supercells, so
            # it factors out of the supercell sum.
            # S_q shape: [1, n_pixels]
            S_q = self.calculate_structure_factor(q_batch).unsqueeze(0)
            # Sum over supercells, scale by unit cells per supercell.
            # Result shape: [batch_size, n_pixels]
            return S_q * torch.sum(phase_factors, dim=-1) * cells_per_supercell

        return self._run_q_batches(q_vectors, q_batch_size, per_batch)

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
        Calculate scattering with per-atom displacements that vary per supercell.

        Use this method when sublattice_displacement (and optionally lattice_strain)
        make the per-supercell modified structure factor differ from one supercell
        to the next, so it cannot be factored out of the supercell sum as in
        calculate_supercell_scattering. continuum_displacement alone (a rigid
        per-supercell shift) does NOT require this method.

        Parameters:
        -----------
        q_vectors: torch.Tensor
            Tensor of shape [..., 3] containing q-vectors
        supercell_size: Tuple
            Size of supercells (d1, d2, d3) in unit cells
        sublattice_displacement: torch.Tensor
            Tensor of shape [batch_size, n1, n2, n3, n_atoms, 3] or
            [batch_size, n_sc1, n_sc2, n_sc3, n_atoms, 3]. Per-atom displacement
            field in unit-cell fractional coordinates. The conversion to lab-frame
            Cartesian uses the current (possibly rotated) lattice vectors, so the
            displacement rotates with the crystal — matching the lab-frame
            convention of the diffraction calculation.
        lattice_strain: torch.Tensor, optional
            Tensor of shape [batch_size, n1, n2, n3, 3, 3] or
            [batch_size, n_sc1, n_sc2, n_sc3, 3, 3]. Per-supercell lattice-vector
            perturbation matrix δL; rows are δa, δb, δc in Cartesian lab-frame meters.
            L_local = L_0 + δL.
        continuum_displacement: torch.Tensor, optional
            Tensor of shape [batch_size, n_sc1, n_sc2, n_sc3, 3] or [batch_size, n_supercells, 3].
            Per-supercell rigid shift in Cartesian lab-frame meters.
        mask: torch.Tensor, optional
            Per-supercell weight mask of shape [batch_size, n1, n2, n3] or
            [batch_size, n_sc1, n_sc2, n_sc3]. See calculate_supercell_scattering
            for the soft-fill-fraction semantics.
        q_batch_size: int, optional
            Number of q-vectors to process at once. If None, process all at once.

        Returns:
        --------
        torch.Tensor
            Scattering amplitude as a complex tensor of shape
            [batch_size, *q_vectors.shape[:-1]].
        """
        
        d1, d2, d3 = supercell_size
        cells_per_supercell = d1 * d2 * d3

        supercell_positions, supercell_mask_flat, continuum_displacement_flat = \
            self._prepare_supercell_data(supercell_size, mask, continuum_displacement)

        batch_size = sublattice_displacement.shape[0]
        n_atoms = sublattice_displacement.shape[-2]

        # Downsample to supercell resolution if necessary, then flatten and
        # convert from fractional unit-cell coords to Cartesian lab-frame meters.
        sublattice_displacement_sc = self._downsample_to_supercell(
            sublattice_displacement, supercell_size, n_trailing=2,
        )
        sublattice_displacement_flat = torch.matmul(
            sublattice_displacement_sc.view(batch_size, -1, n_atoms, 3),
            self.crystal.lattice_vectors,
        )

        # Add the lattice-strain contribution to each atom's displacement:
        # δr_m^strain(n) = f_m @ δL(n).
        if lattice_strain is not None:
            lattice_strain_sc = self._downsample_to_supercell(
                lattice_strain, supercell_size, n_trailing=2,
            )
            lattice_strain_flat = lattice_strain_sc.view(batch_size, -1, 3, 3)
            atom_fracs = self.crystal.atom_frac_coords  # [n_atoms, 3]
            strain_displacement_flat = torch.einsum(
                'mi,bnij->bnmj', atom_fracs, lattice_strain_flat,
            )
            sublattice_displacement_flat = sublattice_displacement_flat + strain_displacement_flat

        def per_batch(q_batch: Tensor) -> Tensor:
            # Per-atom unit-cell phase factors e^(-iq·r_m) (q-only, no n).
            # q_batch shape:           [n_pixels, 3]
            # atom_positions.T shape:  [3, n_atoms]
            # q_dot_r shape:           [n_pixels, n_atoms]
            q_dot_r = torch.matmul(q_batch, self.crystal.atom_positions.T)
            basis_phase_factors = torch.exp(-1j * q_dot_r)                    # [n_pixels, n_atoms]

            # Atomic form factors (q-only).
            # q_magnitude shape: [n_pixels, 1]
            # form_factors shape: [n_pixels, n_atoms]
            q_magnitude = torch.norm(q_batch, dim=-1, keepdim=True) * _Q_M_TO_INV_ANG
            form_factors = self.crystal.calculate_form_factors(q_magnitude)

            # Per-atom displacement phase factors e^(-iq·u_m(n)) for each supercell.
            # q_batch shape:                       [n_pixels, 3]
            # sublattice_displacement_flat shape:  [batch_size, n_supercells, n_atoms, 3]
            # q_dot_displacements shape:           [batch_size, n_pixels, n_supercells, n_atoms]
            q_dot_displacements = torch.einsum(
                'pi,bnmi->bpnm', q_batch, sublattice_displacement_flat,
            )
            displacement_phase_factors = torch.exp(-1j * q_dot_displacements)

            # Modified structure factor per supercell, n: Σ_m f_m(q) · e^(-iq·r_m) · e^(-iq·u_m(n)).
            # Fused einsum avoids materializing the [B, n_pix, n_sc, n_atoms] intermediate.
            # weighted_basis shape:              [n_pixels, n_atoms]
            # modified_structure_factors shape:  [batch_size, n_pixels, n_supercells]
            weighted_basis = basis_phase_factors * form_factors
            modified_structure_factors = torch.einsum(
                'pm,bpnm->bpn', weighted_basis, displacement_phase_factors,
            )

            # Mask multiplies the per-supercell modified structure factor (it varies
            # with n here, unlike in calculate_supercell_scattering where it factors out).
            # supercell_mask_flat shape: [batch_size, n_supercells] -> [batch_size, 1, n_supercells]
            if supercell_mask_flat is not None:
                modified_structure_factors = modified_structure_factors * supercell_mask_flat.unsqueeze(1)

            # Outer per-supercell phase factors e^(-iq·(R + u_continuum)). Mask already applied above.
            # supercell_phase_factors shape: [batch_size, n_pixels, n_supercells]
            supercell_phase_factors = self._compute_supercell_phase_factors(
                q_batch, supercell_positions,
                continuum_displacement_flat, supercell_mask_flat=None,
            )
            # Sum over supercells, scale by unit cells per supercell.
            # Result shape: [batch_size, n_pixels]
            return torch.sum(modified_structure_factors * supercell_phase_factors, dim=-1) * cells_per_supercell

        return self._run_q_batches(q_vectors, q_batch_size, per_batch)