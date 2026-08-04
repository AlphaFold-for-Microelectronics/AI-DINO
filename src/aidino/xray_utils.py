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

def wavelength_to_energy(wavelength: float) -> float:
    """
    Convert wavelength to energy.
    
    Parameters
    ----------
    wavelength : float
        Wavelength in meters
    
    Returns
    -------
    float
        Energy in eV
    """
    # Fundamental constants
    h = 4.135667696e-15  # Planck constant in eV·s
    c = 299792458        # Speed of light in m/s
    
    energy = (h * c) / wavelength
    return energy

def energy_to_wavelength(energy: float) -> float:
    """
    Convert energy to wavelength.
    
    Parameters
    ----------
    energy : float
        Energy in eV
    
    Returns
    -------
    float
        Wavelength in meters
    """
    # Fundamental constants
    h = 4.135667696e-15  # Planck constant in eV·s
    c = 299792458        # Speed of light in m/s
    
    wavelength = (h * c) / energy
    return wavelength