import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
import torch
import math

from torch import Tensor
from matplotlib.figure import Figure

from aidino.plot_utils import format_axis, truncate_colormap, create_figure_with_colorbar

class Detector:
    """
    Class representing a detector for Bragg coherent diffraction simulation.
    """
    
    def __init__(
        self,
        num_pixels_i: int,
        num_pixels_j: int,
        pixel_size: float,
        distance: float,
        wavelength: float,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize a detector object.
        
        Parameters:
        -----------
        num_pixels_i: int
            Number of pixels in the vertical detector dimension
        num_pixels_j: int
            Number of pixels in the horizontal detector dimension
        pixel_size: float
            Size of each pixel in meters
        distance: float
            Sample-to-detector distance in meters
        wavelength: float
            X-ray wavelength in meters
        dtype: torch.dtype
            torch data type
        device: str
            torch device ('cuda' or 'cpu')
        """
        
        self.num_pixels_i = num_pixels_i
        self.num_pixels_j = num_pixels_j
        self.pixel_size = pixel_size
        self.distance = distance
        self.wavelength = wavelength
        self.dtype = dtype
        self.device = device
        
        # Calculate k vector magnitude |k| = 2π/λ
        self.k_magnitude = 2 * torch.pi / wavelength
        
        # Create pixel grid
        self._create_pixel_grid()

    @staticmethod
    def round_in_base(x: float, digits: int = 2, base: int = 10) -> float:
        """
        Round a number to the specified significant digits in the given base.
        
        Parameters:
        -----------
        x: float
            Number to round
        digits: int
            Number of significant digits to maintain when rounding.
            If 0, returns the nearest power of `base` (i.e. base**exponent),
            choosing whichever adjacent power is closer in absolute value.
        base: int
            Base in which to round to the nearest power
            
        Returns:
        --------
        float:
            Rounded number
        """
        
        if x == 0:
            return type(x)(0)
        else:
            exponent = np.floor(math.log(abs(x), base))

            if digits == 0:
                ax = abs(x)
                low = base ** exponent
                high = base ** (exponent + 1)
                power = low if (ax - low) <= (high - ax) else high
                return type(x)(math.copysign(power, x))

            scale = base ** (exponent - digits + 1)
            rounded = round(x / scale) * scale

            return type(x)(rounded)
        
    @staticmethod
    def _rodrigues(v: Tensor, axis: Tensor, sin_a: Tensor, cos_a: Tensor) -> Tensor:
        """
        Rodrigues' rotation formula for a unit ``axis``:
            v_rot = v cos α + (axis × v) sin α + axis (axis · v) (1 − cos α).

        ``axis`` must be a unit vector. ``sin_a`` / ``cos_a`` typically carry
        per-pixel angles with shape [..., 1] so the result broadcasts to that
        leading shape.
        """
        # torch.linalg.cross requires equal rank — pad axis with leading singleton dims.
        if axis.ndim < v.ndim:
            axis = axis.view(*([1] * (v.ndim - axis.ndim)), *axis.shape)
        cross = torch.linalg.cross(axis, v, dim=-1)
        dot   = (axis * v).sum(dim=-1, keepdim=True)
        return v * cos_a + cross * sin_a + axis * dot * (1 - cos_a)

    def _create_pixel_grid(self):
        """
        Create a grid of pixel positions and corresponding angles.
        """
        
        # Create pixel indices
        # Note: i (θ-direction) increases bottom-to-top,
        # but j (φ-direction) increases right-to-left following right-handed coordinate system convention
        i_range = torch.arange(self.num_pixels_i, dtype=self.dtype, device=self.device)
        j_range = torch.arange(self.num_pixels_j-1,-1,-1, dtype=self.dtype, device=self.device)
        
        # Adjust indices to center the origin
        i_centered = i_range - (self.num_pixels_i - 1) / 2
        j_centered = j_range - (self.num_pixels_j - 1) / 2
        
        # Create meshgrid of indices using 'ij' indexing
        # This means i_grid varies along rows (vertical direction) and j_grid along columns (horizontal direction)
        i_grid, j_grid = torch.meshgrid(i_centered, j_centered, indexing='ij')
        
        # Calculate angles α_i and α_j (Equation 6 of Mokhtar et al.)
        # α_i varies in the θ-direction (vertical), α_j varies in the φ-direction (horizontal)
        self.alpha_i = torch.arctan(i_grid * self.pixel_size / self.distance)
        self.alpha_j = torch.arctan(j_grid * self.pixel_size / self.distance)
    
    def calculate_q_vectors(
        self,
        incident_wavevector: Tensor,
        reflected_wavevector: Tensor
    ) -> Tensor:
        """
        Calculate q vectors for each detector pixel.

        Implements Mokhtar et al. eq. (7): rotate k_f around k_i × k_f by α_i
        (θ-direction, varies along axis 0), then around k_f − k_i by α_j
        (φ-direction, varies along axis 1). Axis 1 runs right-to-left
        (j increases from num_pixels_j-1 down to 0) so the (i, j) → angle map
        follows a right-handed coordinate system: q_vectors[i, 0, :]
        corresponds to the +α_j edge and q_vectors[i, -1, :] to the −α_j edge.

        Parameters:
        -----------
        incident_wavevector: torch.Tensor
            Wavevector of incident beam
        reflected_wavevector: torch.Tensor
            Wavevector of reflected beam

        Returns:
        --------
        torch.Tensor
            q vectors for each pixel as a tensor of shape [num_pixels_i, num_pixels_j, 3]
        """
        
        # Normalize input beams to ensure they are unit vectors
        k_i = incident_wavevector / torch.norm(incident_wavevector)
        k_f = reflected_wavevector / torch.norm(reflected_wavevector)
        
        # Convert to device and dtype
        k_i = k_i.to(device=self.device, dtype=self.dtype)
        k_f = k_f.to(device=self.device, dtype=self.dtype)
        
        # Calculate the scattering plane normal (rotation axis 1, in the θ-direction)
        scattering_plane_normal = torch.linalg.cross(k_i, k_f)
        scattering_plane_normal = scattering_plane_normal / torch.norm(scattering_plane_normal)
        
        # Calculate perpendicular direction in the scattering plane (rotation axis 2, in the φ-direction)
        in_plane_perp = k_f - k_i
        in_plane_perp = in_plane_perp / torch.norm(in_plane_perp)
        
        # Prepare sin and cos values for rotations
        sin_alpha_i = torch.sin(self.alpha_i).unsqueeze(-1)
        cos_alpha_i = torch.cos(self.alpha_i).unsqueeze(-1)
        sin_alpha_j = torch.sin(self.alpha_j).unsqueeze(-1)
        cos_alpha_j = torch.cos(self.alpha_j).unsqueeze(-1)

        # Two Rodrigues rotations per pixel (Mokhtar et al. eq. (7)):
        # first rotate k_f around the scattering plane normal by α_i, then
        # rotate the result around (k_f − k_i) by α_j.
        k_r   = self._rodrigues(k_f, scattering_plane_normal, sin_alpha_i, cos_alpha_i)
        k_f_p = self._rodrigues(k_r, in_plane_perp,           sin_alpha_j, cos_alpha_j)
        
        # Scale by k magnitude to get actual wave vectors
        k_i_scaled = k_i * self.k_magnitude
        k_f_p_scaled = k_f_p * self.k_magnitude
        
        # Calculate q = k_f,P - k_i for each pixel
        # Expand k_i for broadcasting
        k_i_expanded = k_i_scaled.view(1, 1, 3).expand_as(k_f_p_scaled)
        q_vectors = k_f_p_scaled - k_i_expanded
        
        return q_vectors

    def calculate_fringe_spacing(self, sample_volume: float) -> float:
        """
        Calculate the fringe spacing given the sample size.

        Uses w_f ≈ 2π / V^(1/3) (Mokhtar et al. §2.2.2), which takes V^(1/3) as
        the characteristic linear size and is therefore an isotropic
        approximation — exact for cubic samples, an average for elongated ones.

        Parameters:
        -----------
        sample_volume: float
            Volume of sample in meters^3

        Returns:
        --------
        float
            Fringe spacing in 1 / meters
        """

        return 2 * torch.pi / sample_volume ** (1/3)

    def calculate_oversampling_ratio(self, sample_volume: float) -> float:
        """
        Calculate the oversampling ratio β attained with the given experimental parameters and sample size.

        Uses Equation 8 of Mokhtar et al. with V^(1/3) as the characteristic
        linear size; see calculate_fringe_spacing for the isotropic assumption.

        Parameters:
        -----------
        sample_volume: float
            Volume of sample in meters^3

        Returns:
        --------
        float
            Oversampling ratio
        """

        fringe_spacing = self.calculate_fringe_spacing(sample_volume)
        oversampling_ratio = fringe_spacing * self.distance / (2 * self.pixel_size * self.k_magnitude)

        return oversampling_ratio
        
    def calculate_resolution(self) -> float:
        """
        Approximate the resolution of the measurement using experimental parameters.
            
        Returns:
        --------
        float
            Resolution in meters
        """
        
        # Calculate the resolution as λ = 2ΔXsin(θ_max) ≈ 2ΔX(D/2R) = ΔXD/R, where D is the detector height
        resolution = (self.wavelength * self.distance) / (self.num_pixels_i * self.pixel_size)

        return resolution

    def plot_q_vectors(self, q_vectors: Tensor) -> Figure:
        """
        Plot the q vector components for each detector pixel.
        
        Parameters:
        -----------
        q_vectors: torch.Tensor
            q vectors for each pixel as a tensor of shape [num_pixels_i, num_pixels_j, 3]
            
        Returns:
        --------
        matplotlib.figure.Figure:
            Figure of the resulting plot
        """
        
        # Reduce on device, then sync each axis range once to Python list.
        vmax = [Detector.round_in_base(v) for v in q_vectors.abs().amax(dim=(0,1)).cpu().tolist()]
        vmin = [Detector.round_in_base(v) for v in q_vectors.amin(dim=(0,1)).cpu().tolist()]

        cmap, norm, sm = [], [], []
        for i in range(3):
            if vmin[i] * vmax[i] < 0:
                cmap.append(cm.vik_r)
                norm.append(plt.Normalize(vmin=-vmax[i], vmax=vmax[i]))
            else:
                cmap.append(truncate_colormap(cm.vik_r, minval=0 if (vmax[i] < 0) else 0.5, maxval=0.5 if (vmax[i] < 0) else 1.))
                norm.append(plt.Normalize(vmin=vmin[i], vmax=vmax[i]))
            sm.append(mpl.cm.ScalarMappable(cmap=cmap[i], norm=norm[i]))

        num_rows, num_cols = 2, 3
        fig, ax = create_figure_with_colorbar(num_rows, num_cols, cbar_orientation='horizontal', sharey='row')
        
        alpha_i_bottom, alpha_i_top = torch.rad2deg(self.alpha_i[0,0]).item(), torch.rad2deg(self.alpha_i[-1,0]).item()
        alpha_j_left, alpha_j_right = torch.rad2deg(self.alpha_j[0,0]).item(), torch.rad2deg(self.alpha_j[0,-1]).item()
        component = [r'$Q_x\ (m^{-1})$', r'$Q_y\ (m^{-1})$', r'$Q_z\ (m^{-1})$']
        
        for i in range(3):
            ax[1,i].imshow(q_vectors[...,i].cpu(), extent=(alpha_j_left, alpha_j_right, alpha_i_bottom, alpha_i_top),
                           origin='lower', cmap=cmap[i], norm=norm[i])
            format_axis(ax[1,i], xlabel=r'$\alpha_j$ (deg.)', ylabel=r'$\alpha_i$ (deg.)' if i==0 else '', xbins=8, ybins=8)
            
            cbar = plt.colorbar(sm[i], cax=ax[0,i], orientation='horizontal')
            ax[0,i].tick_params(axis='x', which='both', top=True, labeltop=True, bottom=False, labelbottom=False)
            ax[0,i].xaxis.set_label_position('top')
            format_axis(ax[0,i], xlabel=component[i], xbins=6)
            
        return fig

    def plot_intensity(self, intensity: Tensor, q_vectors: Tensor = None, vmin: float = None, vmax: float = None) -> Figure:
        """
        Plot the intensity for each detector pixel.
        
        Parameters:
        -----------
        intensity: torch.Tensor
            Scattered intensity for each pixel as a tensor of shape [num_pixels_i, num_pixels_j]
        q_vectors: torch.Tensor
            q vectors for each pixel as a tensor of shape [num_pixels_i, num_pixels_j, 3]
        vmin: float
            Minimum of the data range covered by the colormap
        vmax: float
            Maximum of the data range covered by the colormap
            
        Returns:
        --------
        matplotlib.figure.Figure:
            Figure of the resulting plot
        """
        
        vmax = vmax if vmax is not None else Detector.round_in_base(intensity.max().item(), digits=1)
        vmin = vmin if vmin is not None else vmax / 1e4

        cmap = cm.lapaz
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

        num_rows, num_cols = 1, 2
        fig, ax = create_figure_with_colorbar(num_rows, num_cols)

        if q_vectors is not None:
            extent = [k.cpu() * 1e-9 for k in [q_vectors[0,0,1], q_vectors[0,-1,1], q_vectors[0,0,0], q_vectors[-1,0,0]]]
            xlabel, ylabel = r'$Q_y\ (nm^{-1})$', r'$Q_x\ (nm^{-1})$'
        else:
            extent = None
            xlabel, ylabel = 'Pixels', 'Pixels'
            
        ax[0].imshow(intensity.cpu(), origin='lower', cmap=cmap, norm=norm, extent=extent)
        plt.colorbar(sm, cax=ax[1])
        
        format_axis(ax[0], xlabel=xlabel, ylabel=ylabel, xbins=8)
        format_axis(ax[1], ylabel='Intensity')
            
        return fig