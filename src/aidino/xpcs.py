import torch
from matplotlib.patches import Circle

def calculate_two_time_correlation(I):
    """
    Compute the two-time intensity correlation matrix.

    Parameters
    ----------
    I : torch.Tensor of shape (T, D)
        Intensity data where T is the number of time steps and D is the
        number of pixels/spatial elements.

    Returns
    -------
    C : torch.Tensor of shape (T, T)
        Normalized two-time correlation matrix where C[t1, t2] is the
        correlation between the spatial intensity patterns at times t1 and t2.
    """
    
    T, D = I.shape

    # Compute the mean intensity at each time step (average over spatial dimension)
    I_avg = I.mean(dim=1)

    # Compute the mean of squared intensities at each time step
    I2_avg = (I**2).mean(dim=1)

    # Compute the standard deviation at each time step
    I_std = torch.sqrt(I2_avg - I_avg**2)

    # Outer product of mean intensities
    I2 = torch.outer(I_avg, I_avg)

    # Outer product of standard deviations
    Id = torch.outer(I_std, I_std)

    # Compute the two-time correlation matrix
    II = (I @ I.T) / D

    # Return the normalized two-time correlation
    return (II - I2) / Id

def create_annulus_mask(height, width, r_inner, thickness, center=None, device=None):
    """
    Create an annulus mask in pixel units.

    Parameters
    ----------
    height, width : int
        Detector shape
    r_inner : float
        Inner radius (pixels)
    thickness : float
        Annulus thickness (pixels)
    center : tuple or None
        (cy, cx). Defaults to detector center.
    device : torch.device or None

    Returns
    -------
    mask : torch.BoolTensor of shape (H, W)
    """
    if center is None:
        cy = (height - 1) / 2
        cx = (width - 1) / 2
    else:
        cy, cx = center

    y = torch.arange(height, device=device).float()
    x = torch.arange(width, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)

    r_outer = r_inner + thickness
    mask = (r >= r_inner) & (r < r_outer)

    return mask

def draw_annulus_mask(ax, r_inner, width, center, lw=1.0, color="white"):
    r_outer = r_inner + width

    for r in (r_inner, r_outer):
        circ = Circle(
            center,
            r,
            edgecolor=color,
            facecolor="none",
            linewidth=lw,
            alpha=1.0,
        )
        ax.add_patch(circ)