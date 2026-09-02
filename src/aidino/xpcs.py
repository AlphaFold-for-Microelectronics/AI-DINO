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
from matplotlib.patches import Circle

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