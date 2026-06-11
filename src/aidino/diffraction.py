import torch
import torch.nn.functional as F

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

        # Preprocess continuum displacement to [batch_size, n_supercells, 3].
        # 5D unit-cell-resolution input is averaged into the supercell grid,
        # matching the behavior of mask and sublattice_displacement.
        continuum_displacement_flat = None
        if continuum_displacement is not None:
            if continuum_displacement.ndim == 5:
                continuum_displacement = self._downsample_to_supercell(
                    continuum_displacement, supercell_size, n_trailing=1,
                )
            elif continuum_displacement.ndim != 3:
                raise ValueError(
                    f"continuum_displacement has ndim {continuum_displacement.ndim}, "
                    f"expected 5 ([B, n1, n2, n3, 3] or [B, n_sc1, n_sc2, n_sc3, 3]) "
                    f"or 3 ([B, n_supercells, 3])."
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

    def _run_fft(
        self,
        q_vectors: Tensor,
        supercell_size: Tuple[int, int, int],
        supercell_positions: Tensor,
        bragg_vector: Tensor,
        F_s_at_G: Tensor,
        fft_oversampling: int,
    ) -> Tensor:
        """
        FFT-based scattering: take the per-supercell modified structure factor
        evaluated at the Bragg vector G only, compute a 3D FFT over supercell
        positions for the full diffraction pattern, then interpolate to the
        detector pixel q-vectors.

        Uses the approximation F_s(G + Δq) ≈ F_s(G), valid when |Δq| << |G|
        (typical BCDI detector geometry centered on the Bragg peak).

        Parameters
        ----------
        q_vectors : [..., 3] — detector pixel q-vectors in lab Cartesian (1/m).
        supercell_size : (d1, d2, d3).
        supercell_positions : [n_supercells, 3] in lab Cartesian (meters).
        bragg_vector : [3] — Bragg vector G in lab Cartesian (1/m).
        F_s_at_G : [batch_size, n_supercells] complex — per-supercell modified
            structure factor evaluated at q = G, with mask and continuum-phase
            already folded in by the caller.
        fft_oversampling : int — zero-pad the supercell grid by this factor per
            axis before the FFT so the bin spacing becomes M× finer in Δq.
            Required because bilinear interpolation can't resolve fringes
            narrower than one bin. Set M ≥ max(2·ceil(β), 8) where β is the
            BCDI oversampling ratio (Detector.calculate_oversampling_ratio):
            the 2·β term ensures FFT bins are at least as fine as detector
            pixels; the floor of 8 caps the bilinear interpolation error per
            fringe to a few percent regardless of β.

        Returns
        -------
        Scattering amplitude as a complex tensor of shape
        [batch_size, *q_vectors.shape[:-1]].
        """
        dtype, device = self.crystal.dtype, self.crystal.device
        d1, d2, d3 = supercell_size
        n1, n2, n3 = self.crystal.crystal_size
        n_sc1, n_sc2, n_sc3 = n1 // d1, n2 // d2, n3 // d3
        batch_size = F_s_at_G.shape[0]
        M = int(fft_oversampling)
        if M < 1:
            raise ValueError(f"fft_oversampling must be >= 1, got {fft_oversampling}.")

        # 1. O_G(R_s) = F_s_at_G · exp(-iG·R_s).
        # G_dot_Rs shape: [n_supercells]
        # O_G shape:     [batch_size, n_supercells]
        G_dot_Rs = torch.matmul(supercell_positions, bragg_vector)
        O_G = F_s_at_G * torch.exp(-1j * G_dot_Rs)

        # 2. Reshape to the supercell grid, optionally zero-pad by M per axis to
        # densify the FFT bin grid in Δq (Nyquist range unchanged; only bin
        # density grows, so bilinear interpolation can resolve speckles finer
        # than the unpadded fringe spacing). F.pad keeps the chain differentiable.
        O_G_grid = O_G.view(batch_size, n_sc1, n_sc2, n_sc3)
        if M > 1:
            # F.pad's pad tuple iterates last dim → first dim.
            O_G_grid = F.pad(
                O_G_grid,
                (0, (M - 1) * n_sc3,
                 0, (M - 1) * n_sc2,
                 0, (M - 1) * n_sc1),
            )

        # 3D FFT, DC bin centered via fftshift.
        # A_grid shape: [batch_size, M·n_sc1, M·n_sc2, M·n_sc3] complex
        A_grid = torch.fft.fftshift(
            torch.fft.fftn(O_G_grid, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
        )

        # 3. Map detector Δq = q_vectors - G to normalized FFT grid coords.
        # A_sc has rows = supercell lattice vectors d_i · a_i in lab Cartesian.
        # k_norm[i] = (A_sc[i] · Δq) / π gives the bin coord normalized so that
        # ±1 correspond to ±Nyquist; for an orthorhombic-aligned lattice this
        # decouples per axis (k_norm_i = Δq_i · d_i · |a_i| / π).
        d_vec = torch.tensor(supercell_size, dtype=dtype, device=device).view(3, 1)
        A_sc = d_vec * self.crystal.lattice_vectors                # [3, 3]
        delta_q = q_vectors - bragg_vector                          # [..., 3]
        k_norm = torch.matmul(delta_q, A_sc.T) / torch.pi           # [..., 3]

        # 4. Reorder to (x, y, z) = (k3, k2, k1) for grid_sample's 5D convention
        # (W axis ↔ x coord). grid_5d shape: [batch_size, 1, 1, n_pixels, 3].
        grid_coords = k_norm[..., [2, 1, 0]]
        n_pixels = grid_coords.shape[:-1].numel()
        grid_5d = grid_coords.reshape(1, 1, 1, n_pixels, 3).expand(batch_size, -1, -1, -1, -1)

        # 5. Bilinear interpolation. grid_sample doesn't accept complex tensors,
        # so split into real/imag as a 2-channel input, then recombine.
        # input shape: [batch_size, 2, n_sc1, n_sc2, n_sc3]
        # out   shape: [batch_size, 2, 1, 1, n_pixels]
        A_real_imag = torch.stack([A_grid.real, A_grid.imag], dim=1)
        A_sampled = F.grid_sample(
            A_real_imag, grid_5d,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )
        amplitude = torch.complex(
            A_sampled[:, 0, 0, 0, :], A_sampled[:, 1, 0, 0, :],
        )                                                           # [batch_size, n_pixels]

        # 6. Apply global position phase shift exp(-iq·R_g) per detector pixel.
        q_flat = q_vectors.reshape(-1, 3)
        global_phase = torch.exp(-1j * torch.matmul(q_flat, self.crystal.position))
        amplitude = amplitude * global_phase.unsqueeze(0)

        # 7. Reshape to [batch_size, *q_vectors.shape[:-1]].
        return amplitude.view(batch_size, *q_vectors.shape[:-1])

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
        sublattice_displacement: Optional[Tensor] = None,
        lattice_strain: Optional[Tensor] = None,
        continuum_displacement: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        q_batch_size: Optional[int] = None,
        *,
        method: str = 'direct',
        bragg_vector: Optional[Tensor] = None,
        fft_oversampling: Optional[int] = None,
    ) -> Tensor:
        """
        Calculate scattering with the supercell approach from Mokhtar et al.

        If sublattice_displacement (or lattice_strain) is given, the modified
        structure factor F_s varies per supercell and is computed per-supercell.
        Otherwise S_uc(q) factors out of the supercell sum.

        method='direct' (default) evaluates F_s(q) exactly at each detector
        pixel. method='fft' evaluates F_s once at the Bragg vector and FFTs
        over supercell positions, then interpolates to the detector pixel
        q-vectors — much faster for typical detector sizes, accurate when
        |Δq| << |G|.

        Parameters
        ----------
        q_vectors : torch.Tensor
            Tensor of shape [..., 3] containing q-vectors.
        supercell_size : tuple
            Size of supercells (d1, d2, d3) in unit cells.
        sublattice_displacement : torch.Tensor, optional
            Per-atom displacement field in unit-cell fractional coordinates,
            shape [batch_size, n1, n2, n3, n_atoms, 3] or [batch_size,
            n_sc1, n_sc2, n_sc3, n_atoms, 3]. Converted to lab-frame
            Cartesian via the current (possibly rotated) lattice vectors so
            the displacement rotates with the crystal. When given (or when
            lattice_strain is given), the per-supercell modified structure
            factor is computed.
        lattice_strain : torch.Tensor, optional
            Per-supercell lattice-vector perturbation matrix δL (rows are
            δa, δb, δc in Cartesian lab-frame meters), shape [batch_size,
            n1, n2, n3, 3, 3] or [batch_size, n_sc1, n_sc2, n_sc3, 3, 3].
            Combines with sublattice_displacement via δr_m^strain = f_m · δL.
        continuum_displacement : torch.Tensor, optional
            Per-supercell rigid Cartesian shift, shape [batch_size, n1, n2,
            n3, 3], [batch_size, n_sc1, n_sc2, n_sc3, 3], or [batch_size,
            n_supercells, 3]. Unit-cell-resolution input is averaged into the
            supercell grid. Does NOT trigger the per-supercell path (it's
            just an outer phase factor).
        mask : torch.Tensor, optional
            Per-supercell weight mask, shape [batch_size, n1, n2, n3] or
            [batch_size, n_sc1, n_sc2, n_sc3]. Unit-cell-resolution masks are
            downsampled by averaging (soft fill-fraction); pass at supercell
            resolution for exact 0/1 weights.
        q_batch_size : int, optional
            Number of q-vectors per batch (direct method only). Ignored when
            method='fft'.
        method : {'direct', 'fft'}
            Algorithm choice. 'direct' is exact within the supercell
            approximation. 'fft' adds the approximation F_s(G + Δq) ≈ F_s(G)
            — much faster but degrades as |Δq|/|G| grows.
        bragg_vector : torch.Tensor, optional
            Bragg vector G as a length-3 tensor in lab Cartesian (1/m). Only
            used in FFT mode. Defaults to the centroid of q_vectors — valid
            when q_vectors are symmetric around the Bragg peak; for asymmetric
            coverage pass an explicit value.
        fft_oversampling : int, **required when method='fft'**
            Zero-padding factor M for the supercell FFT grid. The FFT bin
            spacing in Δq becomes M× finer, letting bilinear interpolation
            resolve speckle fringes that would otherwise be lost between bins.

            Recommended: ``fft_oversampling = max(2 * ceil(beta), 8)``, where
            ``beta`` comes from
            ``Detector.calculate_oversampling_ratio(crystal.crystal_volume)``.
            The ``2 * beta`` term ensures the FFT bin spacing is at least as
            fine as the detector pixel spacing (β = pixels-per-fringe / 2, so
            2β = pixels per fringe); the floor of 8 keeps the bilinear
            interpolation error per fringe to a few percent regardless of β.
            Memory scales as M³.

        Returns
        -------
        torch.Tensor
            Scattering amplitude as a complex tensor of shape
            [batch_size, *q_vectors.shape[:-1]].
        """
        d1, d2, d3 = supercell_size
        cells_per_supercell = d1 * d2 * d3
        n_atoms = self.crystal.n_atoms

        supercell_positions, supercell_mask_flat, continuum_displacement_flat = \
            self._prepare_supercell_data(supercell_size, mask, continuum_displacement)

        has_per_atom = sublattice_displacement is not None or lattice_strain is not None

        # Infer batch_size from the first batched input that exists; default to 1.
        batch_inputs = [x for x in (sublattice_displacement, lattice_strain,
                                     continuum_displacement, mask) if x is not None]
        batch_size = batch_inputs[0].shape[0] if batch_inputs else 1

        # Combine sublattice_displacement (fractional → Cartesian) and
        # lattice_strain (Cartesian δL → per-atom shift via f_m @ δL) into a
        # single per-atom Cartesian displacement buffer. Scalar-0 init means
        # we don't allocate a zeros buffer for an absent contribution and
        # gradients flow back through the + to whichever input was given.
        sublattice_displacement_flat = 0
        if sublattice_displacement is not None:
            sublattice_displacement_flat = sublattice_displacement_flat + torch.matmul(
                self._downsample_to_supercell(sublattice_displacement, supercell_size, n_trailing=2)
                    .view(batch_size, -1, n_atoms, 3),
                self.crystal.lattice_vectors,
            )
        if lattice_strain is not None:
            strain_sc = self._downsample_to_supercell(lattice_strain, supercell_size, n_trailing=2)
            sublattice_displacement_flat = sublattice_displacement_flat + torch.einsum(
                'mi,bnij->bnmj',
                self.crystal.atom_frac_coords,
                strain_sc.view(batch_size, -1, 3, 3),
            )

        if method == 'direct':
            def per_batch(q_batch: Tensor) -> Tensor:
                if has_per_atom:
                    # Per-supercell modified F_s: structure factor depends on n.
                    # q_dot_r shape:                       [n_pixels, n_atoms]
                    # basis_phase_factors shape:           [n_pixels, n_atoms]
                    # form_factors shape:                  [n_pixels, n_atoms]
                    # q_dot_displacements shape:           [batch_size, n_pixels, n_supercells, n_atoms]
                    # modified_structure_factors shape:    [batch_size, n_pixels, n_supercells]
                    q_dot_r = torch.matmul(q_batch, self.crystal.atom_positions.T)
                    basis_phase_factors = torch.exp(-1j * q_dot_r)
                    q_magnitude = torch.norm(q_batch, dim=-1, keepdim=True) * _Q_M_TO_INV_ANG
                    form_factors = self.crystal.calculate_form_factors(q_magnitude)

                    q_dot_displacements = torch.einsum(
                        'pi,bnmi->bpnm', q_batch, sublattice_displacement_flat,
                    )
                    displacement_phase_factors = torch.exp(-1j * q_dot_displacements)
                    weighted_basis = basis_phase_factors * form_factors
                    modified_structure_factors = torch.einsum(
                        'pm,bpnm->bpn', weighted_basis, displacement_phase_factors,
                    )

                    # Mask multiplies the per-supercell modified structure factor
                    # (it varies with n, so it can't factor out as in the else branch).
                    if supercell_mask_flat is not None:
                        modified_structure_factors = modified_structure_factors * supercell_mask_flat.unsqueeze(1)

                    # Outer phase factors e^(-iq·(R + u_continuum)); mask already applied above.
                    supercell_phase_factors = self._compute_supercell_phase_factors(
                        q_batch, supercell_positions,
                        continuum_displacement_flat, supercell_mask_flat=None,
                    )
                    return torch.sum(
                        modified_structure_factors * supercell_phase_factors, dim=-1
                    ) * cells_per_supercell

                else:
                    # Factored: S_uc(q) is identical across supercells.
                    # phase_factors shape: [batch_size, n_pixels, n_supercells]
                    # S_q shape:           [1, n_pixels]
                    phase_factors = self._compute_supercell_phase_factors(
                        q_batch, supercell_positions,
                        continuum_displacement_flat, supercell_mask_flat,
                    )
                    S_q = self.calculate_structure_factor(q_batch).unsqueeze(0)
                    return S_q * torch.sum(phase_factors, dim=-1) * cells_per_supercell

            return self._run_q_batches(q_vectors, q_batch_size, per_batch)

        elif method == 'fft':
            if fft_oversampling is None:
                raise ValueError(
                    "fft_oversampling is required when method='fft'. Compute "
                    "beta = detector.calculate_oversampling_ratio(crystal.crystal_volume) "
                    "and pass int(max(2 * ceil(beta), 8)) — see the docstring for details."
                )
            # Resolve Bragg vector: explicit value, or centroid of q_vectors.
            G = bragg_vector if bragg_vector is not None else q_vectors.reshape(-1, 3).mean(dim=0)
            n_supercells = supercell_positions.shape[0]

            # Compute F_s(G) per supercell — the per-atom sum collapses to a
            # single q-vector (G), no q-batch loop needed.
            if has_per_atom:
                # Same math as the direct path's per-atom branch, with pixel dim collapsed.
                G_dot_r = torch.matmul(G.unsqueeze(0), self.crystal.atom_positions.T).squeeze(0)  # [n_atoms]
                basis_phase_at_G = torch.exp(-1j * G_dot_r)                                       # [n_atoms]
                G_magnitude = torch.norm(G).reshape(1, 1) * _Q_M_TO_INV_ANG                       # [1, 1]
                form_factors_at_G = self.crystal.calculate_form_factors(G_magnitude).squeeze(0)   # [n_atoms]
                # G_dot_u shape:                  [batch_size, n_supercells, n_atoms]
                # displacement_phase_at_G shape:  [batch_size, n_supercells, n_atoms]
                # F_s_at_G shape:                 [batch_size, n_supercells]
                G_dot_u = torch.einsum('i,bnmi->bnm', G, sublattice_displacement_flat)
                displacement_phase_at_G = torch.exp(-1j * G_dot_u)
                weighted_basis_at_G = basis_phase_at_G * form_factors_at_G
                F_s_at_G = torch.einsum('m,bnm->bn', weighted_basis_at_G, displacement_phase_at_G)
            else:
                # Factored: S_uc(G) broadcast across supercells.
                # S_uc_at_G shape: [1] complex; expand to [batch_size, n_supercells] as a view.
                S_uc_at_G = self.calculate_structure_factor(G.unsqueeze(0))
                F_s_at_G = S_uc_at_G.view(1, 1).expand(batch_size, n_supercells)

            # Same cells_per_supercell scaling as the direct path.
            F_s_at_G = F_s_at_G * cells_per_supercell

            # Apply mask in real space (multiplication ↔ convolution in Δq, the standard
            # windowed-crystal interpretation that also matches the direct path's mask handling).
            if supercell_mask_flat is not None:
                F_s_at_G = F_s_at_G * supercell_mask_flat

            # Continuum displacement → per-supercell phase exp(-iG·u_s), consistent with
            # the F_s(G + Δq) ≈ F_s(G) approximation that justifies the FFT path.
            if continuum_displacement_flat is not None:
                G_dot_u_continuum = torch.einsum('i,bni->bn', G, continuum_displacement_flat)
                F_s_at_G = F_s_at_G * torch.exp(-1j * G_dot_u_continuum)

            return self._run_fft(
                q_vectors, supercell_size, supercell_positions,
                G, F_s_at_G, fft_oversampling,
            )

        else:
            raise ValueError(f"method must be 'direct' or 'fft', got {method!r}")