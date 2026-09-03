# ----------------------------------------------------------------------- #
# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: AI-DINO: AI for Dynamic Imaging of Nanoscale Objects
# By: Argonne National Laboratory
#
# BSD Open Source License
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# Disclaimer
# This software is provided by the copyright holders and contributors "as is"
# and any express or implied warranties, including, but not limited to, the
# implied warranties of merchantability and fitness for a particular purpose
# are disclaimed. In no event shall the copyright holder or contributors be
# liable for any direct, indirect, incidental, special, exemplary, or
# consequential damages (including, but not limited to, procurement of
# substitute goods or services; loss of use, data, or profits; or business
# interruption) however caused and on any theory of liability, whether in
# contract, strict liability, or tort (including negligence or otherwise)
# arising in any way out of the use of this software, even if advised of the
# possibility of such damage.
# ----------------------------------------------------------------------- #

import torch

def calculate_two_time_correlation(I):
    """
    Compute the two-time intensity correlation matrix using the Sutton normalization:

        C[t1, t2] = <I(p,t1) I(p,t2)>_p / ( <I(p,t1)>_p <I(p,t2)>_p )

    where <.>_p averages over the D pixels of equivalent q.
    I must be raw, non-negative intensity.

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
    mu = I.mean(dim=1)                       # <I(p,t)>_p, shape (T,)

    # Compute the two-time correlation matrix
    II = (I @ I.T) / D                       # <I(p,t1) I(p,t2)>_p, shape (T, T)
    
    # Outer product of mean intensities (normalization)
    denom = torch.outer(mu, mu)

    # Return the normalized two-time correlation
    return II / denom

def _delta_q_magnitude(q_vectors, q_reference=None):
    """
    Compute |q - q_reference| per detector pixel.

    Parameters
    ----------
    q_vectors : torch.Tensor of shape (H, W, 3)
        Per-pixel momentum-transfer vectors, e.g. from
        ``Detector.calculate_q_vectors``.
    q_reference : torch.Tensor of shape (3,) or None
        Origin in reciprocal space from which the magnitude is measured.
        Defaults to the q vector at the detector-center pixel (the Bragg peak),
        so equivalent-q bins are concentric about the peak. Pass a zero vector
        to measure absolute |q| instead.

    Returns
    -------
    dq : torch.Tensor of shape (H, W)
        Magnitude of the deviation from ``q_reference`` at each pixel.
    """
    if q_reference is None:
        height, width = q_vectors.shape[:2]
        q_reference = q_vectors[height // 2, width // 2]

    return (q_vectors - q_reference).norm(dim=-1)

def create_q_annulus_mask(q_vectors, q_center, width, q_reference=None):
    """
    Create a mask of detector pixels of equivalent momentum transfer.

    Pixels are selected by the magnitude of their deviation from a reference
    point in reciprocal space, |q - q_reference|, using the physical per-pixel
    q vectors. Because the true q vectors encode the full detector geometry, the
    selected region is automatically elliptical/curved on the detector whenever
    the geometry is anisotropic, i.e. it defines truly equivalent q without
    assuming circular symmetry.

    Parameters
    ----------
    q_vectors : torch.Tensor of shape (H, W, 3)
        Per-pixel momentum-transfer vectors, e.g. from
        ``Detector.calculate_q_vectors``.
    q_center : float
        Center of the q bin, in the units of ``q_vectors``.
    width : float
        Full width of the q bin, in the units of ``q_vectors``.
    q_reference : torch.Tensor of shape (3,) or None
        Origin from which |q - q_reference| is measured. Defaults to the
        detector-center pixel (the Bragg peak). Pass a zero vector to bin on
        absolute |q|.

    Returns
    -------
    mask : torch.BoolTensor of shape (H, W)
    """
    dq = _delta_q_magnitude(q_vectors, q_reference)

    q_lower = q_center - width / 2
    q_upper = q_center + width / 2
    mask = (dq >= q_lower) & (dq < q_upper)

    return mask

def q_bin_centers(q_vectors, n, q_reference=None, q_min=None, q_max=None):
    """
    Generate ``n`` equally spaced q-bin centers spanning the observed q range.

    Convenience for choosing the ``q_center`` values to pass to
    ``create_q_annulus_mask``. Uses the same |q - q_reference| coordinate and
    reference default. Consecutive bins tile the range when the bin ``width`` is
    set to the spacing between adjacent centers.

    Parameters
    ----------
    q_vectors : torch.Tensor of shape (H, W, 3)
        Per-pixel momentum-transfer vectors, e.g. from
        ``Detector.calculate_q_vectors``.
    n : int
        Number of bin centers to return.
    q_reference : torch.Tensor of shape (3,) or None
        Origin from which |q - q_reference| is measured. Defaults to the
        detector-center pixel (the Bragg peak).
    q_min, q_max : float or None
        Range of bin centers. Default to ``q_max / (n + 1)`` and the maximum
        observed |q - q_reference|, respectively.

    Returns
    -------
    centers : torch.Tensor of shape (n,)
    """
    dq = _delta_q_magnitude(q_vectors, q_reference)

    if q_max is None:
        q_max = dq.max()
    if q_min is None:
        q_min = q_max / (n + 1)

    return torch.linspace(float(q_min), float(q_max), n, device=q_vectors.device)

def draw_q_annulus_mask(ax, q_vectors, q_center, width, q_reference=None,
                        origin="lower", lw=1.0, color="white"):
    """
    Overlay the boundaries of a q annulus on an image axis.

    The boundaries are the two |q - q_reference| iso-contours bounding the bin,
    drawn with ``ax.contour`` so they correctly follow the elliptical/curved
    equivalent-q loci (rather than assuming circles).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which the corresponding image was drawn with ``imshow``.
    q_vectors : torch.Tensor of shape (H, W, 3)
        Per-pixel momentum-transfer vectors.
    q_center, width : float
        Center and full width of the q bin, matching ``create_q_annulus_mask``.
    q_reference : torch.Tensor of shape (3,) or None
        Origin from which |q - q_reference| is measured. Defaults to the
        detector-center pixel (the Bragg peak).
    origin : {"lower", "upper"}
        Must match the ``origin`` passed to the paired ``imshow`` call so the
        overlay aligns with the image.
    lw : float
        Contour line width.
    color : str
        Contour line color.
    """
    dq = _delta_q_magnitude(q_vectors, q_reference)

    levels = sorted([q_center - width / 2, q_center + width / 2])
    ax.contour(
        dq.detach().cpu().numpy(),
        levels=levels,
        colors=color,
        linewidths=lw,
        origin=origin,
    )