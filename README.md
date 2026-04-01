# AI-DINO
### AI for Dynamic Imaging of Nanoscale Objects

> ⚠️ This package is under active development.

AI-DINO is a PyTorch-based forward modeling framework for simulating Bragg Coherent Diffraction Imaging (BCDI) experiments from crystalline nanostructures. All scattering calculations are fully differentiable and GPU-accelerated, enabling gradient-based optimization and seamless integration with machine learning workflows.

## Overview

The scattering calculation is based on the supercell factorization approach of [Mokhtar et al., *J. Phys. Commun.* **6**, 055003 (2022)](https://doi.org/10.1088/2399-6528/ac6ab0), which avoids the fixed-grid constraint of Fourier-transform-based methods and reduces computation time by several orders of magnitude. This enables full differentiability with respect to arbitrary atomic and lattice displacement fields — a requirement for gradient-based optimization and phase retrieval.

AI-DINO extends the Mokhtar et al. framework in several ways:

- **Continuum displacement fields** — per-supercell rigid shifts (e.g. from phase-field simulations) are incorporated as additional supercell-level phase factors, capturing the BCDI phase signal without modifying the structure factor
- **Sublattice displacements** — per-atom, per-supercell shifts (e.g. ferroelectric off-centering from polarization fields via Born effective charges) enter the modified per-supercell structure factor
- **Lattice strain** — local unit cell distortions from a strain tensor field perturb atom positions within each supercell and further modify the structure factor
- **Exodus II / MOOSE integration** — the `exodus` module resamples phase-field simulation output (displacements, polarization, strain) directly onto the diffraction supercell grid
- **Crystal orientation handling** — arbitrary sample rotations with full rotation matrix tracking for consistent coordinate frame transformations across all displacement fields

## Modules

| Module | Description |
|---|---|
| `sample.py` | `Crystal` class — parses CIF files, manages lattice vectors, atom positions, form factors, and crystal orientation |
| `diffraction.py` | `BraggCoherentDiffraction` class — supercell scattering calculations with optional displacement, strain, and mask fields |
| `exodus.py` | `ExodusMesh` / `CrystalGrid` — Exodus II file parsing and resampling onto the diffraction supercell grid |
| `beam.py` | X-ray beam profile generation |
| `detector.py` | Detector geometry and q-vector calculation |
| `xpcs.py` | Two-time intensity correlation for XPCS analysis |
| `xray_utils.py` | Utility functions (wavelength/energy conversion, etc.) |

## Requirements

See `requirements.txt`. Key dependencies:

- `torch` — all scattering calculations are fully differentiable PyTorch operations
- `pymatgen` — CIF parsing and crystal structure handling
- `netCDF4` — Exodus II file I/O

## Reference

A H Mokhtar, D Serban and M C Newton, "Simulation of Bragg coherent diffraction imaging," *J. Phys. Commun.* **6**, 055003 (2022). https://doi.org/10.1088/2399-6528/ac6ab0