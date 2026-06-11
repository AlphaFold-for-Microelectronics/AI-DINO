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