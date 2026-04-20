import torch
import torch.nn as nn
import numpy as np

from torchdiffeq import odeint, odeint_adjoint

class ODE(nn.Module):
    """
    A PyTorch neural network module for solving Ordinary Differential Equations (ODEs).
    
    This class provides a wrapper around the torchdiffeq library, enabling flexible ODE solving
    with support for adjoint methods for memory-efficient gradient computation.
    
    Attributes
    ----------
    method : str
        The numerical integration method to use for solving the ODE.
    adjoint : bool
        Whether to use the adjoint method for gradient computation.
    odeint : callable
        The ODE integration function (either odeint or odeint_adjoint).
    dtype : torch.dtype
        The data type for computations.
    """
    
    def __init__(self, method='dopri5', adjoint=False, requires_grad=True, dtype=torch.float32):
        """
        Initialize the ODE solver.
        
        Parameters
        ----------
        method : str, optional
            The numerical integration method. Defaults to 'dopri5'.
            Common options include 'dopri5', 'rk4', etc.
        adjoint : bool, optional
            Whether to use adjoint method for gradient computation. Defaults to False.
            Only used if requires_grad is True.
        requires_grad : bool, optional
            Whether gradients are required. If False, adjoint is automatically disabled. Defaults to True.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.
        """
        super(ODE, self).__init__()
        
        self.method = method
        self.adjoint = adjoint if requires_grad else False
        self.odeint = odeint_adjoint if self.adjoint else odeint
        self.dtype = dtype
    
    def solve(self, t, y0, device='cpu', rtol=1e-7, atol=1e-9):
        """
        Solve the ODE given time points and initial conditions.
        
        Parameters
        ----------
        t : torch.Tensor
            Time points at which to evaluate the solution.
            Shape: (T,) where T is the number of time points.
        y0 : torch.Tensor
            Initial conditions. Shape: (batch_size, state_dim).
        device : str, optional
            Device to run computations on. Defaults to 'cpu'.
        rtol : float, optional
            Relative tolerance for the solver. Defaults to 1e-7.
        atol : float, optional
            Absolute tolerance for the solver. Defaults to 1e-9.
        
        Returns
        -------
        torch.Tensor
            Solution of the ODE at the specified time points.
            Shape: (T, batch_size, state_dim).
        """
        
        if self.training:
            return self.odeint(self.to(device), y0.to(device), t.to(device), method=self.method,
                               rtol=rtol, atol=atol, options={'dtype': self.dtype})
        else:
            with torch.no_grad():
                return odeint(self.to(device), y0.to(device), t.to(device), method=self.method,
                              rtol=rtol, atol=atol, options={'dtype': self.dtype})
    
    def get_batch(self, t, y, batch_time, batch_size):
        """
        Create batches from trajectory data.
        
        This method samples random subsequences from the trajectory data to create training batches.
        When batch_time equals T, it returns full trajectories for randomly sampled initial conditions.
        
        Parameters
        ----------
        t : torch.Tensor
            Time points. Shape: (T,).
        y : torch.Tensor
            Trajectory data. Shape: (T, M, ...) where:
            - T: number of time points
            - M: number of trajectories
        batch_time : int
            Length of each batch sequence (number of time steps).
            If batch_time == T, returns full trajectories.
        batch_size : int
            Number of sequences in the batch.
        
        Returns
        -------
        t_batch : torch.Tensor
            Time points for the batch. Shape: (batch_time,).
        y0_batch : torch.Tensor
            Initial conditions for each batch sequence. Shape: (batch_size, ...).
        y_batch : torch.Tensor
            Batch trajectory data. Shape: (batch_time, batch_size, ...).
        """
        
        T, M = y.shape[:2]
        t_batch = t[:batch_time]
    
        if batch_time == T:
            # Return full trajectories for randomly sampled initial conditions
            replace = batch_size > M  # Allow replacement if we need more samples than available trajectories
            traj_indices = np.random.choice(M, batch_size, replace=replace)
            
            y0_batch = y[0, traj_indices]  # Initial conditions from selected trajectories
            y_batch = y[:, traj_indices]   # Full trajectories for selected trajectories
            
        else:
            # Sample subsequences of trajectories for randomly sampled initial conditions
            # Generate all possible (time_start, trajectory_index) combinations
            c = [[i, j] for i in range(T - batch_time) for j in range(M)]
            sampled_indices = np.random.choice(len(c), batch_size, replace=False)
            
            # Extract time and trajectory indices
            time_indices = [c[i][0] for i in sampled_indices]
            traj_indices = [c[i][1] for i in sampled_indices]
            
            # Extract initial conditions
            y0_batch = y[time_indices, traj_indices]
            
            # Extract trajectory subsequences
            batch_indices = torch.arange(batch_time)[:, None] + torch.tensor(time_indices)[None, :]
            y_batch = y[batch_indices, traj_indices]
        
        return t_batch, y0_batch, y_batch

class Kuramoto3D(ODE):
    """
    3D Kuramoto model for studying synchronization dynamics of spatially coupled oscillators.
    
    This class implements the Kuramoto model with various spatial coupling kernels, allowing
    for the study of pattern formation, synchronization, and wave dynamics in 3D systems.
    The model evolves phase oscillators according to:
    
    dθ_i/dt = K * Σ_j G(r_ij) * sin(θ_j - θ_i)
    
    where θ_i are the oscillator phases, K is the coupling strength, and G(r_ij) is the
    spatial coupling kernel.
    
    Attributes
    ----------
    Nx, Ny, Nz : int
        Grid sizes for the 3D spatial domain (Nx x Ny x Nz grid). Default is 50 for each.
    K : float
        Coupling strength.
    kernel_type : str
        Type of spatial coupling kernel.
    conv : nn.Conv3d
        Convolution layer implementing the spatial coupling.
    """
    
    def __init__(self, args, method='dopri5', dtype=torch.float32):
        """
        Initialize the 3D Kuramoto model with specified parameters.
        
        Parameters
        ----------
        args : dict
            Dictionary containing model parameters. Missing parameters will use defaults.
            Can specify either:
            - 'N' for uniform grid (N×N×N)
            - 'Nx', 'Ny', 'Nz' for non-uniform grid
        method : str, optional
            ODE integration method. Defaults to 'dopri5'.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.
            
        Notes
        -----
        Available kernel types and their specific parameters:
        - 'gaussian': Standard Gaussian coupling (gaussian_sigma)
        - 'laplacian_of_gaussian': Mexican hat kernel for stripe patterns (log_sigma)
        - 'ring': Ring-shaped coupling for spiral waves (ring_radius, ring_width)
        - 'power_law': Power-law decay coupling (power_alpha, power_scale)
        - 'exponential_decay': Exponential decay coupling (decay_length)
        - 'step_function': Sharp cutoff coupling (step_radius)
        - 'periodic': Periodic spatial coupling (period_length)
        - 'double_exponential': Multi-scale interactions (short_range, long_range, weights)
        """
        super(Kuramoto3D, self).__init__(method, adjoint=False, requires_grad=False, dtype=dtype)
        
        default_args = {
            # Basic parameters
            'N': 50,
            'K': 0.2,
            
            # Kernel selection
            # ('gaussian', 'laplacian_of_gaussian', 'ring', 'power_law', 'exponential_decay',
            # 'step_function', 'periodic', 'double_exponential')
            'kernel_type': 'laplacian_of_gaussian',  
            
            # Gaussian kernel parameters
            'gaussian_sigma': 1.0,          # Standard deviation for Gaussian kernel
            
            # Laplacian of Gaussian kernel parameters
            'log_sigma': 1.5,               # Standard deviation for LoG kernel
            
            # Ring kernel parameters
            'ring_radius': 2.0,             # Radius of coupling ring
            'ring_width': 0.5,              # Width of the ring
            
            # Power-law kernel parameters
            'power_alpha': 2.5,             # Power-law exponent
            'power_scale': 2.0,             # Characteristic scale for power-law cutoff
            
            # Exponential decay kernel parameters
            'decay_length': 1.5,            # Decay length λ
            
            # Step function kernel parameters
            'step_radius': 2.0,             # Step function radius
            
            # Periodic kernel parameters
            'period_length': 3.0,           # Spatial period L
            
            # Double exponential kernel parameters
            'short_range': 0.8,             # Short-range decay length
            'long_range': 3.0,              # Long-range decay length
            'short_weight': 1.0,            # Weight of short-range component
            'long_weight': -0.5,            # Weight of long-range component
            
            # General parameters
            'cutoff_factor': 3.0,           # General cutoff factor for kernel size
            'normalization': 'none'          # Kernel normalization: 'sum', 'max', 'none'
        }
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)
        
        # If N is specified, use it for all directions
        # Otherwise, use Nx, Ny, Nz
        if 'Nx' in args or 'Ny' in args or 'Nz' in args:
            self.Nx = args.get('Nx', self.N)
            self.Ny = args.get('Ny', self.N)
            self.Nz = args.get('Nz', self.N)
        else:
            self.Nx = self.Ny = self.Nz = self.N
        
        # Get the kernel based on kernel_type
        kernel = self.K * self._get_kernel()[None, None]
        
        # Convolution operation with kernel
        self.conv = nn.Conv3d(1, 1, kernel.shape[-1], bias=False, padding='same', padding_mode='circular')
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)
    
    def _get_kernel(self):
        """
        Get the appropriate spatial coupling kernel based on kernel_type.
        
        Returns
        -------
        torch.Tensor
            3D spatial coupling kernel of shape (2d+1, 2d+1, 2d+1) where d depends
            on the kernel type and parameters.
        """
        kernel_methods = {
            'gaussian': self.gaussian_kernel,
            'laplacian_of_gaussian': self.laplacian_of_gaussian_kernel,
            'ring': self.ring_kernel,
            'power_law': self.power_law_kernel,
            'exponential_decay': self.exponential_decay_kernel,
            'step_function': self.step_function_kernel,
            'periodic': self.periodic_kernel,
            'double_exponential': self.double_exponential_kernel
        }
        
        return kernel_methods[self.kernel_type]()
    
    def _create_spatial_grid(self, d):
        """
        Create standardized 3D spatial grid for kernel evaluation.
        
        Parameters
        ----------
        d : int
            Half-size of the kernel grid. Creates grid from -d to +d in each dimension.
        
        Returns
        -------
        x : torch.Tensor
            Coordinate tensor of shape (3, 2d+1, 2d+1, 2d+1) containing x, y, z coordinates.
        r : torch.Tensor
            Radial distance tensor of shape (2d+1, 2d+1, 2d+1) containing Euclidean
            distances from the center.
        """
        d_range = torch.arange(-d, d+1, dtype=self.dtype)
        x = torch.stack(torch.meshgrid(d_range, d_range, d_range, indexing='ij'))
        r = torch.sqrt((x**2).sum(dim=0))  # Radial distance
        return x, r
    
    def _normalize_kernel(self, kernel):
        """
        Apply standardized normalization to the coupling kernel.
        
        Parameters
        ----------
        kernel : torch.Tensor
            Input kernel to be normalized.
        
        Returns
        -------
        torch.Tensor
            Normalized kernel according to self.normalization setting.
            
        Notes
        -----
        Normalization options:
        - 'sum': Normalize so kernel elements sum to 1
        - 'max': Normalize so maximum kernel value is 1  
        - 'none': No normalization applied
        """
        if self.normalization == 'sum':
            return kernel / kernel.sum()
        elif self.normalization == 'max':
            return kernel / kernel.max()
        else:
            return kernel
    
    def gaussian_kernel(self):
        """
        Generate Gaussian coupling kernel for local synchronization.
        
        Creates a 3D Gaussian kernel: K(r) ∝ exp(-r²/2σ²)
        
        Returns
        -------
        torch.Tensor
            3D Gaussian kernel with self-coupling removed.
            
        Notes
        -----
        This kernel promotes local synchronization and is commonly used
        for studying clustering and local pattern formation.
        """
        d = int(np.ceil(self.cutoff_factor * self.gaussian_sigma))
        x, r = self._create_spatial_grid(d)
        
        kernel = torch.exp(-r**2 / (2 * self.gaussian_sigma**2))
        kernel[d, d, d] = 0.  # Remove self-coupling
        
        return self._normalize_kernel(kernel)
    
    def laplacian_of_gaussian_kernel(self):
        """
        Generate Laplacian of Gaussian (Mexican Hat) kernel for stripe patterns.
        
        Creates a LoG kernel: K(r) = -(1/σ²)[r²/σ² - n] * exp(-r²/2σ²)
        where n is the spatial dimension (3).
        
        Returns
        -------
        torch.Tensor
            3D Laplacian of Gaussian kernel with self-coupling removed.
            
        Notes
        -----
        This kernel has a positive center surrounded by negative values,
        promoting local synchronization while inhibiting distant coupling.
        Often generates stripe and labyrinthine patterns.
        """
        d = int(np.ceil(self.cutoff_factor * self.log_sigma))
        x, r = self._create_spatial_grid(d)
        n = x.shape[0]  # Spatial dimension (3D)
        
        # LoG formula: -(1/σ²)[r²/σ² - n] * exp(-r²/2σ²)
        gaussian = torch.exp(-r**2 / (2 * self.log_sigma**2))
        laplacian_term = -(r**2 / self.log_sigma**2 - n) / (np.sqrt((2 * np.pi)**n) * self.log_sigma**(n + 2))
        kernel = laplacian_term * gaussian
        kernel[d, d, d] = 0.  # Remove self-coupling
        
        return self._normalize_kernel(kernel)
    
    def ring_kernel(self):
        """
        Generate ring-shaped coupling kernel for spiral waves and target patterns.
        
        Creates coupling concentrated in a ring at specified radius with Gaussian profile.
        
        Returns
        -------
        torch.Tensor
            3D ring kernel with self-coupling removed.
            
        Notes
        -----
        Ring kernels can generate spiral waves, target patterns, and rotating
        wave solutions depending on initial conditions and parameters.
        """
        d = int(np.ceil(self.ring_radius + 2 * self.ring_width))
        x, r = self._create_spatial_grid(d)
        
        # Create ring using Gaussian profile centered at ring_radius
        kernel = torch.exp(-((r - self.ring_radius)**2) / (2 * self.ring_width**2))
        kernel[d, d, d] = 0.  # Remove self-coupling
        
        return self._normalize_kernel(kernel)
    
    def power_law_kernel(self):
        """
        Generate power-law coupling kernel for long-range interactions.
        
        Creates kernel with algebraic decay: K(r) ∝ r^(-α)
        
        Returns
        -------
        torch.Tensor
            3D power-law kernel with self-coupling removed.
            
        Notes
        -----
        Power-law kernels exhibit scale-free properties and can generate
        complex spatial patterns with long-range correlations.
        """
        d = int(np.ceil(self.cutoff_factor * self.power_scale))
        x, r = self._create_spatial_grid(d)
        
        # Avoid division by zero at r=0
        r_safe = torch.where(r > 0, r, torch.inf)
        kernel = torch.pow(r_safe, -self.power_alpha)
        kernel[r == 0] = 0  # Set center and avoid infinities
        kernel[torch.isinf(kernel)] = 0
        
        return self._normalize_kernel(kernel)
    
    def exponential_decay_kernel(self):
        """
        Generate exponentially decaying coupling kernel.
        
        Creates kernel with exponential decay: K(r) ∝ exp(-r/λ)
        
        Returns
        -------
        torch.Tensor
            3D exponential decay kernel with self-coupling removed.
            
        Notes
        -----
        Exponential kernels provide intermediate-range coupling that
        decays more slowly than Gaussian but faster than power-law.
        """
        d = int(np.ceil(self.cutoff_factor * self.decay_length))
        x, r = self._create_spatial_grid(d)
        
        kernel = torch.exp(-r / self.decay_length)
        kernel[d, d, d] = 0.  # Remove self-coupling
        
        return self._normalize_kernel(kernel)
    
    def step_function_kernel(self):
        """
        Generate step function coupling kernel for sharp domain boundaries.
        
        Creates uniform coupling within a sphere and zero coupling outside.
        
        Returns
        -------
        torch.Tensor
            3D step function kernel with self-coupling removed.
            
        Notes
        -----
        Step function kernels create sharp transitions between synchronized
        and desynchronized regions, often leading to domain formation.
        """
        d = int(np.ceil(self.step_radius + 1))
        x, r = self._create_spatial_grid(d)
        
        kernel = (r <= self.step_radius).float()
        kernel[d, d, d] = 0.  # Remove self-coupling
        
        return self._normalize_kernel(kernel)
    
    def periodic_kernel(self):
        """
        Generate periodic coupling kernel for lattice-like structures.
        
        Creates spatially periodic coupling modulated by a Gaussian envelope.
        
        Returns
        -------
        torch.Tensor
            3D periodic kernel with self-coupling removed.
            
        Notes
        -----
        Periodic kernels can generate regular lattice patterns and
        standing wave solutions with characteristic wavelengths.
        """
        d = int(np.ceil(self.cutoff_factor * self.period_length))
        x, r = self._create_spatial_grid(d)
        
        # Cosine modulated by Gaussian envelope to ensure decay
        cosine_term = torch.cos(2 * np.pi * r / self.period_length)
        envelope = torch.exp(-r**2 / (2 * (self.period_length/2)**2))
        kernel = cosine_term * envelope
        kernel[d, d, d] = 0.  # Remove self-coupling
        
        return self._normalize_kernel(kernel)
    
    def double_exponential_kernel(self):
        """
        Generate double exponential kernel for multi-scale interactions.
        
        Combines short-range and long-range exponential components with different weights.
        
        Returns
        -------
        torch.Tensor
            3D double exponential kernel with self-coupling removed.
            
        Notes
        -----
        Double exponential kernels can model competing interactions at different
        scales, such as local synchronization with long-range inhibition.
        """
        d = int(np.ceil(3 * self.long_range))
        x, r = self._create_spatial_grid(d)
        
        short_exp = self.short_weight * torch.exp(-r / self.short_range)
        long_exp = self.long_weight * torch.exp(-r / self.long_range)
        kernel = short_exp + long_exp
        kernel[d, d, d] = 0.  # Remove self-coupling
        
        return self._normalize_kernel(kernel)
        
    def init_state(self, M=1, seed=12):
        """
        Initialize random phase configuration for the oscillators.
        
        Parameters
        ----------
        M : int, optional
            Number of independent realizations/trajectories. Defaults to 1.
        seed : int, optional
            Random seed for reproducibility. Defaults to 12.
        
        Returns
        -------
        torch.Tensor
            Initial phase configuration of shape (M, 1, Nx*Ny*Nz) with phases
            uniformly distributed in [0, 2π].
        """
        torch.manual_seed(seed)
        return 2 * torch.pi * torch.rand((M, 1, self.Nx, self.Ny, self.Nz), dtype=self.dtype).flatten(start_dim=-3)
    
    def forward(self, t, y):
        """
        Compute the time derivative of the phase configuration (ODE right-hand side).
        
        Implements the Kuramoto dynamics:
        dθ/dt = K * Σ_j G(r_ij) * sin(θ_j - θ_i)
        
        Parameters
        ----------
        t : float
            Current time.
        y : torch.Tensor
            Current phase configuration of shape (M, 1, Nx*Ny*Nz) where M is batch size.
        
        Returns
        -------
        torch.Tensor
            Time derivatives of phases with same shape as input y.
            
        Notes
        -----
        The convolution operation efficiently computes the spatial coupling
        using the precomputed kernel. The computation uses the identity:
        sin(θ_j - θ_i) = sin(θ_j)cos(θ_i) - cos(θ_j)sin(θ_i)
        """
        y = y.view((-1, 1, self.Nx, self.Ny, self.Nz))
        cosy = torch.cos(y)
        siny = torch.sin(y)
        conv_cosy = self.conv(cosy)
        conv_siny = self.conv(siny)
        return (cosy * conv_siny - siny * conv_cosy).flatten(start_dim=-3)


class KuramotoQuasi2D(ODE):
    """
    Quasi-2D Kuramoto model where a 2D kernel is replicated along the z-dimension.
    
    This class implements the Kuramoto model as a stack of independent 2D problems
    with identical spatial coupling in the x-y plane. The model evolves phase 
    oscillators according to:
    
    dθ_i/dt = K * Σ_j G(r_ij) * sin(θ_j - θ_i)
    
    where the coupling kernel G is 2D (acts only in x-y plane) and is applied
    independently to each z-slice.
    
    Attributes
    ----------
    Nx, Ny, Nz : int
        Grid sizes for the 3D spatial domain (Nx x Ny x Nz grid).
    K : float
        Coupling strength.
    log_sigma : float
        Standard deviation for the Laplacian of Gaussian kernel.
    conv : nn.Conv3d
        Convolution layer implementing the 2D spatial coupling replicated in z.
    """

    def __init__(self, args, method='dopri5', dtype=torch.float32):
        """
        Initialize the quasi-2D Kuramoto model with specified parameters.
        
        Parameters
        ----------
        args : dict
            Dictionary containing model parameters. Missing parameters will use defaults.
        method : str, optional
            ODE integration method. Defaults to 'dopri5'.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.

        Notes
        -----
        Only the Laplacian of Gaussian kernel is available for this quasi-2D model,
        as coupling acts only in the x-y plane and is replicated across z-slices.
        Laplacian of Gaussian kernel parameters:
        - 'log_sigma': Standard deviation for LoG kernel (default: 1.0)
        """
        super(KuramotoQuasi2D, self).__init__(method, adjoint=False, requires_grad=False, dtype=dtype)

        default_args = {
            'Nx': 100,
            'Ny': 100,
            'Nz': 50,
            'K': 0.2,
            'log_sigma': 1.5,
            'cutoff_factor': 3.0,
            'normalization': 'none',
        }

        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)

        # Get the 2D LoG kernel and embed it into a 3D kernel with singleton z dimension
        kernel_2d = self.K * self._get_kernel()           # (2d+1, 2d+1)
        kernel_3d = kernel_2d[..., None][None, None, ...]  # (1, 1, 2d+1, 2d+1, 1)

        # Conv3d with kernel size 1 in z-direction applies 2D conv to each z-slice independently
        self.conv = nn.Conv3d(1, 1, kernel_3d.shape[-3:],
                              bias=False, padding='same', padding_mode='circular')
        self.conv.weight = nn.Parameter(kernel_3d, requires_grad=False)

    def _get_kernel(self):
        """
        Get the Laplacian of Gaussian spatial coupling kernel.

        Returns
        -------
        torch.Tensor
            2D Laplacian of Gaussian kernel of shape (2d+1, 2d+1) where d is
            determined by cutoff_factor and log_sigma.
        """
        return self.laplacian_of_gaussian_kernel()

    def _create_spatial_grid(self, d):
        """
        Create standardized 2D spatial grid for kernel evaluation.

        Parameters
        ----------
        d : int
            Half-size of the kernel grid. Creates grid from -d to +d in each dimension.

        Returns
        -------
        x : torch.Tensor
            Coordinate tensor of shape (2, 2d+1, 2d+1) containing x, y coordinates.
        r : torch.Tensor
            Radial distance tensor of shape (2d+1, 2d+1) containing Euclidean
            distances from the center.
        """
        d_range = torch.arange(-d, d + 1, dtype=self.dtype)
        x = torch.stack(torch.meshgrid(d_range, d_range, indexing='ij'))
        r = torch.sqrt((x ** 2).sum(dim=0))
        return x, r

    def _normalize_kernel(self, kernel):
        """
        Apply standardized normalization to the coupling kernel.

        Parameters
        ----------
        kernel : torch.Tensor
            Input kernel to be normalized.

        Returns
        -------
        torch.Tensor
            Normalized kernel according to self.normalization setting.

        Notes
        -----
        Normalization options:
        - 'sum': Normalize so kernel elements sum to 1
        - 'max': Normalize so maximum kernel value is 1
        - 'none': No normalization applied
        """
        if self.normalization == 'sum':
            return kernel / kernel.sum()
        elif self.normalization == 'max':
            return kernel / kernel.max()
        else:
            return kernel

    def laplacian_of_gaussian_kernel(self):
        """
        Generate 2D Laplacian of Gaussian (Mexican Hat) kernel for stripe patterns.

        Creates a LoG kernel: K(r) = -(1/σ²)[r²/σ² - n] * exp(-r²/2σ²)
        where n is the spatial dimension (2).

        Returns
        -------
        torch.Tensor
            2D Laplacian of Gaussian kernel with self-coupling removed.

        Notes
        -----
        This kernel has a positive center surrounded by negative values,
        promoting local synchronization while inhibiting distant coupling.
        Often generates stripe and labyrinthine patterns. The 2D kernel is
        applied identically and independently to each z-slice.
        """
        d = int(np.ceil(self.cutoff_factor * self.log_sigma))
        x, r = self._create_spatial_grid(d)
        n = x.shape[0]  # Spatial dimension (2D)

        # LoG formula: -(1/σ²)[r²/σ² - n] * exp(-r²/2σ²)
        gaussian = torch.exp(-r ** 2 / (2 * self.log_sigma ** 2))
        laplacian_term = -(r ** 2 / self.log_sigma ** 2 - n) / (np.sqrt((2 * np.pi) ** n) * self.log_sigma ** (n + 2))
        kernel = laplacian_term * gaussian
        kernel[d, d] = 0.  # Remove self-coupling

        return self._normalize_kernel(kernel)

    def init_state(self, M=1, seed=12):
        """
        Initialize random phase configuration with identical patterns across all z-slices.

        Parameters
        ----------
        M : int, optional
            Number of independent realizations/trajectories. Defaults to 1.
        seed : int, optional
            Random seed for reproducibility. Defaults to 12.

        Returns
        -------
        torch.Tensor
            Initial phase configuration of shape (M, 1, Nx*Ny*Nz) with phases
            uniformly distributed in [0, 2π], identical across all z-slices.
        """
        torch.manual_seed(seed)

        # Generate 2D initial state and replicate across z-dimension
        y0_2d = 2 * np.pi * torch.rand((M, 1, self.Nx, self.Ny), dtype=self.dtype)
        y0 = y0_2d.unsqueeze(-1).expand(-1, -1, -1, -1, self.Nz)
        return y0.flatten(start_dim=-3)

    def forward(self, t, y):
        """
        Compute the time derivative of the phase configuration.

        Implements the Kuramoto dynamics with 2D spatial coupling applied
        independently to each z-slice:
        dθ/dt = K * Σ_j G(r_ij) * sin(θ_j - θ_i)

        Parameters
        ----------
        t : float
            Current time.
        y : torch.Tensor
            Current phase configuration of shape (M, 1, Nx*Ny*Nz).

        Returns
        -------
        torch.Tensor
            Time derivatives of phases with same shape as input y.

        Notes
        -----
        Uses the trigonometric identity:
        sin(θ_j - θ_i) = sin(θ_j)cos(θ_i) - cos(θ_j)sin(θ_i)
        """
        y = y.view((-1, 1, self.Nx, self.Ny, self.Nz))
        cosy = torch.cos(y)
        siny = torch.sin(y)
        conv_cosy = self.conv(cosy)
        conv_siny = self.conv(siny)
        return (cosy * conv_siny - siny * conv_cosy).flatten(start_dim=-3)

        
class CahnHilliard3D(ODE):
    """
    The Cahn-Hilliard equation is a fourth-order partial differential equation
    that models phase separation in binary alloys and other systems. This implementation
    solves the equation on a 3D periodic domain using finite differences and 
    convolutional operations for spatial derivatives.
    
    The equation takes the form:
    dc/dt = D * ∇²(c³ - c - γ * ∇²c)
    
    where c represents the concentration field, D is the diffusion parameter,
    and ∇² is the Laplace operator.
    
    Attributes
    ----------
    Nx, Ny, Nz : int
        Grid sizes for the 3D spatial domain (Nx x Ny x Nz grid). Default is 64 for each.
    Lx, Ly, Lz : float
        Physical lengths of the domain in each direction. Default is 2.0 for each.
    D : float
        Diffusion coefficient controlling the rate of evolution. Default is 5e-5.
    g: float
        Gradient energy coefficient controlling the length of transition regions between the domains. Default is 1e-4.
    laplacian : torch.nn.Conv3d
        Convolutional layer implementing the discrete Laplace operator
        with periodic boundary conditions.
    """
    def __init__(self, args, method='dopri5', dtype=torch.float32):
        """
        Initialize the 3D Cahn Hilliard model with specified parameters.
        
        Parameters
        ----------
        args : dict
            Dictionary containing model parameters. Missing parameters will use defaults.
            Can specify either:
            - 'N' and 'L' for uniform grid
            - 'Nx', 'Ny', 'Nz' and 'Lx', 'Ly', 'Lz' for non-uniform grid
        method : str, optional
            ODE integration method. Defaults to 'dopri5'.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.
        """
        super(CahnHilliard3D, self).__init__(method, adjoint=False, requires_grad=False, dtype=dtype)
        
        default_args = {'N': 128,
                        'L': 2.,
                        'D': 1e-4,
                        'g': 1e-4
                       }
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)
        
        # If N and L are specified, use them for all directions
        # Otherwise, use Nx, Ny, Nz and Lx, Ly, Lz
        if 'Nx' in args or 'Ny' in args or 'Nz' in args:
            self.Nx = args.get('Nx', self.N)
            self.Ny = args.get('Ny', self.N)
            self.Nz = args.get('Nz', self.N)
        else:
            self.Nx = self.Ny = self.Nz = self.N
            
        if 'Lx' in args or 'Ly' in args or 'Lz' in args:
            self.Lx = args.get('Lx', self.L)
            self.Ly = args.get('Ly', self.L)
            self.Lz = args.get('Lz', self.L)
        else:
            self.Lx = self.Ly = self.Lz = self.L

        # Calculate grid spacings for each direction
        hx = self.Lx / self.Nx
        hy = self.Ly / self.Ny
        hz = self.Lz / self.Nz
        
        # Seven-point stencil for the 3D Laplace operator ∇²
        # With non-uniform grid spacing: ∇²c = (1/hx²)(c_{i+1} - 2c_i + c_{i-1}) + ...
        # The stencil is organized as [x, y, z] (width, height, depth)
        stencil = torch.tensor([[[[[0, 0, 0],
                                   [0, 1./hz**2, 0],
                                   [0, 0, 0]],
                                  [[0, 1./hy**2, 0],
                                   [1./hx**2, -2.*(1./hx**2 + 1./hy**2 + 1./hz**2), 1./hx**2],
                                   [0, 1./hy**2, 0]],
                                  [[0, 0, 0],
                                   [0, 1./hz**2, 0],
                                   [0, 0, 0]]]]], dtype=self.dtype)
        
        self.laplacian = nn.Conv3d(1, 1, stencil.shape[-1], bias=False, padding='same', padding_mode='circular')
        self.laplacian.weight = nn.Parameter(stencil, requires_grad=False)

    def init_state(self, M=1, seed=12, mean=0., sigma=0.01):
        """
        Initialize the state vector with random noise.
        
        Parameters
        ----------
        M : int, optional
            Batch size (number of initial conditions). Defaults to 1.
        seed : int, optional
            Random seed for reproducible initialization. Defaults to 12.
        sigma : float, optional
            Amplitude scaling factor for the random noise. Defaults to 0.01.
            
        Returns
        -------
        torch.Tensor
            Initial state tensor of shape (M, 1, Nx*Ny*Nz).
        """
        torch.manual_seed(seed)
        return mean + sigma * torch.randn((M, 1, self.Nx, self.Ny, self.Nz), dtype=self.dtype).flatten(start_dim=-3) 
        
    def forward(self, t, c):
        """
        Compute the time derivative for the Cahn-Hilliard equation.
        
        This method implements the 3D Cahn-Hilliard equation:
        dc/dt = D * ∇²(c³ - c - γ * ∇²c)
        
        Parameters
        ----------
        t : torch.Tensor
            Current time.
        c : torch.Tensor
            Current state vector of shape (M, 1, Nx*Ny*Nz) where M is batch size
            and Nx*Ny*Nz represents the flattened 3D spatial grid.
            
        Returns
        -------
        torch.Tensor
            Time derivative dc/dt with same shape as input c, representing
            the rate of change according to the Cahn-Hilliard dynamics.
        """
        c = c.view(-1, 1, self.Nx, self.Ny, self.Nz)
        return self.D * self.laplacian(c ** 3 - c - self.g * self.laplacian(c)).flatten(start_dim=-3)

        
class CahnHilliardQuasi2D(ODE):
    """
    Quasi-2D Cahn-Hilliard model where a 2D operator is replicated along the z-dimension.
    
    This class implements the Cahn-Hilliard equation as a stack of independent 2D problems
    with identical spatial operators in the x-y plane. The equation evolves concentration
    fields according to:
    
    dc/dt = D * ∇²(c³ - c - γ * ∇²c)
    
    where the Laplacian operator ∇² is 2D (acts only in x-y plane) and is applied
    independently to each z-slice.
    
    Attributes
    ----------
    Nx, Ny : int
        Grid sizes for the 2D spatial domain in x and y directions.
    Nz : int
        Number of independent z-slices.
    Lx, Ly : float
        Physical lengths of the domain in x and y directions.
    D : float
        Diffusion coefficient controlling the rate of evolution.
    g : float
        Gradient energy coefficient controlling the length of transition regions.
    laplacian : nn.Conv3d
        Convolution layer implementing the 2D Laplace operator replicated in z.
    """
    
    def __init__(self, args, method='dopri5', dtype=torch.float32):
        """
        Initialize the quasi-2D Cahn-Hilliard model with specified parameters.
        
        Parameters
        ----------
        args : dict
            Dictionary containing model parameters:
            - 'Nx' : x-dimension size (default: 128)
            - 'Ny' : y-dimension size (default: 128)
            - 'Nz' : z-dimension size (default: 50)
            - 'L'  : physical domain length for both x and y (default: 2.0)
            - 'Lx' : physical domain length in x (overrides 'L' if specified)
            - 'Ly' : physical domain length in y (overrides 'L' if specified)
            - 'D'  : diffusion coefficient (default: 1e-4)
            - 'g'  : gradient energy coefficient (default: 1e-4)
        method : str, optional
            ODE integration method. Defaults to 'dopri5'.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.
        """
        super(CahnHilliardQuasi2D, self).__init__(method, adjoint=False, requires_grad=False, dtype=dtype)
        
        default_args = {
            'Nx': 128,
            'Ny': 128,
            'Nz': 50,
            'L': 2.,
            'D': 1e-4,
            'g': 1e-4
        }
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)

        # If Lx/Ly are specified, use them; otherwise fall back to the shared L
        if 'Lx' in args or 'Ly' in args:
            self.Lx = args.get('Lx', self.L)
            self.Ly = args.get('Ly', self.L)
        else:
            self.Lx = self.Ly = self.L

        # Calculate grid spacings for each direction independently
        hx = self.Lx / self.Nx
        hy = self.Ly / self.Ny

        # Five-point stencil for the 2D Laplace operator ∇²
        # With non-uniform grid spacing: ∇²c = (1/hx²)(c_{i+1} - 2c_i + c_{i-1})
        #                                      + (1/hy²)(c_{j+1} - 2c_j + c_{j-1})
        stencil_2d = torch.tensor([[0, 1./hy**2, 0.],
                                   [1./hx**2, -2.*(1./hx**2 + 1./hy**2), 1./hx**2],
                                   [0, 1./hy**2, 0.]], dtype=self.dtype)
        
        # Convert to 3D kernel with singleton z dimension
        # Shape: (3, 3) -> (3, 3, 1) -> (1, 1, 3, 3, 1)
        kernel_3d = stencil_2d[..., None]  # Add z dimension: (3, 3, 1)
        kernel_3d = kernel_3d[None, None, ...]  # Add channel dims: (1, 1, 3, 3, 1)
        
        # Conv3d with kernel size 1 in z-direction applies 2D conv to each z-slice independently
        self.laplacian = nn.Conv3d(1, 1, kernel_3d.shape[-3:], 
                                   bias=False, padding='same', padding_mode='circular')
        self.laplacian.weight = nn.Parameter(kernel_3d, requires_grad=False)
            
    def init_state(self, M=1, seed=12, mean=0., sigma=0.01):
        """
        Initialize random concentration field with identical patterns across all z-slices.
        
        Parameters
        ----------
        M : int, optional
            Batch size (number of initial conditions). Defaults to 1.
        seed : int, optional
            Random seed for reproducible initialization. Defaults to 12.
        sigma : float, optional
            Amplitude scaling factor for the random noise. Defaults to 0.01.
            
        Returns
        -------
        torch.Tensor
            Initial state tensor of shape (M, Nx*Ny*Nz) with random noise,
            identical across all z-slices.
        """
        torch.manual_seed(seed)
        
        # Generate 2D initial state
        c0_2d = mean + sigma * torch.randn((M, 1, self.Nx, self.Ny), dtype=self.dtype)
        
        # Replicate across z-dimension
        c0 = c0_2d.unsqueeze(-1).expand(-1, -1, -1, -1, self.Nz)
        return c0.flatten(start_dim=-3)
        
    def forward(self, t, c):
        """
        Compute the time derivative for the quasi-2D Cahn-Hilliard equation.
        
        This method implements the Cahn-Hilliard equation with 2D spatial operators
        applied independently to each z-slice:
        dc/dt = D * ∇²(c³ - c - γ * ∇²c)
        
        Parameters
        ----------
        t : torch.Tensor
            Current time.
        c : torch.Tensor
            Current state vector of shape (M, Nx*Ny*Nz) where M is batch size.
            
        Returns
        -------
        torch.Tensor
            Time derivative dc/dt with same shape as input c, representing
            the rate of change according to the Cahn-Hilliard dynamics.
        """
        c = c.view(-1, 1, self.Nx, self.Ny, self.Nz)
        return self.D * self.laplacian(c ** 3 - c - self.g * self.laplacian(c)).flatten(start_dim=-3)