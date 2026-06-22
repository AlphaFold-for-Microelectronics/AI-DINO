"""
Benchmark utilities for the direct vs FFT BCDI scattering methods.

Designed to be imported from benchmark.ipynb, which builds a synthetic
BaTiO3 cubic crystal with random continuum + sublattice displacement
fields and a Gaussian-beam mask.

Tests provided:
    - test_method_vs_supercells          (time + memory, both methods)
    - test_method_vs_batch_size          (time + memory vs frames/call, both methods)
    - test_direct_vs_qbatch              (time + memory, direct method)
    - test_direct_error_vs_supercell     (time + memory + chi^2_0, direct)
    - test_fft_error_vs_oversampling     (time + memory + chi^2_0 + chi^2, FFT)
    - test_custom_backward_consistency   (gradient parity, direct method)

chi^2_0 is the error vs the ground-truth direct@(1,1,1) intensity (total
approximation error — supercell + FFT combined). chi^2 (test 4 only) is the
error vs direct@same-supercell — the pure FFT approximation error within
the chosen supercell. The gap between chi^2_0 and chi^2 at large oversampling
reveals the supercell-approximation floor.

Each test loops over `include_sublattice` (False/True) and `grad_mode`.
By default the grad_modes are (no_grad, fwd, bwd) paired with the
relevant input — continuum_displacement when include_sublattice=False,
sublattice_displacement when include_sublattice=True. OOM is trapped so
a partial sweep records `oom=True` instead of crashing. For tests 3 & 4,
chi^2 quantities are computed only under no_grad (independent of grad_mode).

Tests 1, 2, 3, and 5 also accept `use_custom_backward_options` to compare
the direct method's default autograd backward against its custom-gradient
path (which avoids retaining the large per-atom intermediates in the
backward graph). The default `(False,)` preserves existing behavior; pass
`(False, True)` to sweep both. FFT iterations are not duplicated since
the FFT method has no custom-backward variant.

Results serialize as JSON to a per-session subdirectory of
benchmark_results/, named '<YYYY-MM-DD>_<gpu_name>' so JSON files and any
figures saved via `figure_path` from the same session land together.
"""
from __future__ import annotations

import gc
import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

from aidino.beam import GaussianBeam
from aidino.detector import Detector
from aidino.diffraction import BraggCoherentDiffraction
from aidino.sample import Crystal


# -----------------------------------------------------------------------------
# Plot styling (tweak here for global effect)
# -----------------------------------------------------------------------------

# Color for the right (twin) y-axis used to overlay chi^2 on time/memory plots.
TWIN_COLOR = '#7B9FBA'
# Line styles used cyclically when multiple y_twin fields are plotted together.
_TWIN_LINESTYLES = ('-', '--', ':', '-.')

# Axes-grid padding (inches). Figure size = axes_size * grid + these, so axes
# stay the same physical size regardless of label / tick / legend widths.
_PAD_LEFT = 0.85          # y-label + tick labels
_PAD_RIGHT = 0.1          # margin when no twin axis
_PAD_RIGHT_TWIN = 0.8     # twin y-label + tick labels
_PAD_TOP = 0.25           # top margin without title
_PAD_TOP_TITLE = 0.8      # extra height for suptitle
_PAD_BOTTOM = 0.65        # x-label + tick labels
_GAP_W = 0.35             # horizontal gap between panel columns
_LEGEND_W = 1.85          # reserved width for the legend itself (prevents clipping)


# -----------------------------------------------------------------------------
# Result schema + serialization
# -----------------------------------------------------------------------------

VALID_GRAD_MODES = (
    'no_grad',
    'fwd_continuum',
    'fwd_sublattice',
    'bwd_continuum',
    'bwd_sublattice',
)


@dataclass
class BenchmarkResult:
    test_name: str
    swept_param: str
    swept_value: Any                        # JSON-serializable scalar/tuple
    method: str                             # 'direct' | 'fft'
    include_sublattice: bool
    grad_mode: str
    elapsed_time_s: float
    peak_memory_gb: float
    chi_squared_0: Optional[float] = None       # vs direct@(1,1,1) (ground truth)
    chi_squared: Optional[float] = None          # vs direct@same supercell
    n_supercells: Optional[int] = None
    fft_oversampling: Optional[int] = None
    use_custom_backward: bool = False        # direct method's custom-grad path (False for FFT)
    oom: bool = False
    extra: dict = field(default_factory=dict)


def _json_safe(value: Any) -> Any:
    """Recursively coerce dataclass values to JSON-serializable forms."""
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


# Module-level session-device override. Set once at the top of a notebook via
# set_session_device('cuda:2') and all subsequent save_results / figure_path /
# session_dir calls use it. Without it, helpers fall back to
# torch.cuda.current_device(), which is NOT changed by a bare `device = 'cuda:N'`
# assignment in user code — only by `torch.cuda.set_device(N)`.
_SESSION_DEVICE: 'str | int | torch.device | None' = None


def set_session_device(device: 'str | int | torch.device | None') -> None:
    """Set (or clear with None) the module-level device used to name session
    directories. Recommended once at the top of a benchmarking notebook:

        from aidino.benchmark import set_session_device
        device = 'cuda:2'
        set_session_device(device)

    After that, every save_results / figure_path / session_dir call uses this
    device for the folder name without needing an explicit `device=` kwarg.
    Per-call `device=` arguments still override this when given.
    """
    global _SESSION_DEVICE
    _SESSION_DEVICE = device


def _gpu_name(device: 'str | int | torch.device | None' = None) -> str:
    """Sanitized GPU name suitable for use in a directory name.

    Resolution order for the device:
        1. The `device` argument if explicitly given (not None).
        2. The module-level session device set via `set_session_device(...)`.
        3. `torch.cuda.current_device()` if CUDA is available.
        4. 'cpu' otherwise.

    `device` accepts the same forms as `torch.device(...)`: 'cuda:N' / 'cuda'
    string, an int N, or a `torch.device`.
    """
    if device is None:
        device = _SESSION_DEVICE
    if not torch.cuda.is_available():
        return 'cpu'
    if device is None:
        index = torch.cuda.current_device()
    else:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.type == 'cpu':
            return 'cpu'
        index = dev.index if dev.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    return re.sub(r'[^\w.-]+', '_', name).strip('_') or 'unknown'


def session_dir(output_dir: str = '../benchmark_results') -> Path:
    """Return (creating if needed) a shared session directory for results
    and figures, named '<YYYY-MM-DD>_<gpu_name>'.

    Multiple calls on the same day with the same GPU return the same
    directory, so all JSON files and figures from one benchmark session
    naturally land together. The GPU is taken from the module-level
    session device (see `set_session_device`); falls back to
    `torch.cuda.current_device()` if none is set.
    """
    date = datetime.now().strftime('%Y-%m-%d')
    path = Path(output_dir) / f'{date}_{_gpu_name()}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    results: List[BenchmarkResult],
    test_name: str,
    output_dir: str = '../benchmark_results',
) -> Path:
    """Serialize results to <session_dir>/<test_name>_<HHMMSS>.json.

    Results land in the per-session subdirectory of `output_dir` (see
    `session_dir`). The filename suffix is the wall-clock time so
    multiple runs of the same test on the same day are disambiguated.
    """
    out_dir = session_dir(output_dir)
    timestamp = datetime.now().strftime('%H%M%S')
    out_path = out_dir / f'{test_name}_{timestamp}.json'
    payload = [_json_safe(asdict(r)) for r in results]
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def figure_path(
    name: str,
    *,
    ext: str = 'png',
    output_dir: str = '../benchmark_results',
    timestamp: bool = False,
) -> Path:
    """Return a path inside today's session directory for saving a figure.

    Usage:
        fig.savefig(figure_path('direct_vs_qbatch_memory'))
        # → ../benchmark_results/2026-06-21_<gpu>/direct_vs_qbatch_memory.png

        fig.savefig(figure_path('direct_vs_qbatch_memory', ext='pdf', timestamp=True))
        # → ../benchmark_results/<session>/direct_vs_qbatch_memory_143052.pdf

    Parameters
    ----------
    name : str
        Filename stem; use underscores to separate tags
        (e.g. f'{test_name}_{y_field}').
    ext : str, default 'png'
        File extension. Whatever your `fig.savefig` understands.
    output_dir : str, default '../benchmark_results'
        Parent directory; the session subdirectory is created inside it.
    timestamp : bool, default False
        Append `_HHMMSS` to the stem so multiple saves of the same name
        within one session don't overwrite each other. Off by default so
        that re-running a notebook cell overwrites the existing figure
        (usually what you want during iteration).
    """
    out_dir = session_dir(output_dir)
    stem = name
    if timestamp:
        stem = f'{name}_{datetime.now().strftime("%H%M%S")}'
    return out_dir / f'{stem}.{ext}'


def load_results(path: str | Path) -> List[BenchmarkResult]:
    """Re-instantiate BenchmarkResult instances from a saved JSON file."""
    data = json.loads(Path(path).read_text())
    return [BenchmarkResult(**d) for d in data]


# -----------------------------------------------------------------------------
# Measurement primitive
# -----------------------------------------------------------------------------

def measure_call(
    call_fn: Callable[[], Tensor],
    *,
    backward: bool = False,
    no_grad: bool = False,
    device: torch.device,
    warmup: bool = True,
    n_repeats: int = 3,
) -> Tuple[float, float, Optional[Tensor]]:
    """
    Run `call_fn()`, compute intensity = |amplitude|^2, optionally .backward()
    through loss = intensity.sum(), and report wall time and peak additional
    GPU memory.

    The CUDA peak is taken from torch.cuda.max_memory_allocated minus the
    baseline taken right before the timed calls, so prior live tensors do not
    inflate the result.

    warmup=True (default) runs one untimed call matching the measured mode
    (forward+backward when backward=True) before measurement. This amortizes
    cuFFT plan caching, kernel JIT, and allocator pool growth — without it
    the first call in a sweep can be 10x slower than subsequent calls.

    n_repeats > 1 (default 3) runs the timed call N times and returns the
    **median** elapsed time. Median is robust to occasional jitter spikes
    (GC pauses, thermal events). Peak memory is the max across all repeats,
    which equals the per-call peak (the workload is identical).

    On CUDA OOM the function catches the error, empties the cache, and
    returns (nan, nan, None) so the test loop can record oom=True and move on.
    """
    def run():
        if no_grad:
            with torch.no_grad():
                amp = call_fn()
        else:
            amp = call_fn()
        intensity = amp.abs() ** 2
        if backward:
            intensity.sum().backward()
        return amp

    try:
        if warmup:
            # Don't bind the result: any retained autograd graph from the
            # warmup would inflate baseline and underreport measurement peak.
            run()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        elapsed_per_run: List[float] = []
        amplitude: Optional[Tensor] = None
        for _ in range(n_repeats):
            # Release the previous run's autograd graph before allocating the next.
            amplitude = None
            t_start = time.time()
            amplitude = run()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed_per_run.append(time.time() - t_start)
        elapsed = float(np.median(elapsed_per_run))

        if device.type == 'cuda':
            peak = torch.cuda.max_memory_allocated(device)
            peak_gb = (peak - baseline) / 1024 ** 3
        else:
            peak_gb = 0.0

        return elapsed, peak_gb, amplitude

    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return float('nan'), float('nan'), None


def chi_squared(intensity: Tensor, intensity_ref: Tensor) -> float:
    """
    Normalized L2 error on amplitudes (Mokhtar et al. 2022, eq. 15):

        chi^2 = sum_q (sqrt(I_q) - sqrt(I0_q))^2 / sum_q I0_q
    """
    with torch.no_grad():
        diff = torch.sqrt(intensity) - torch.sqrt(intensity_ref)
        return ((diff ** 2).sum() / intensity_ref.sum()).item()


# -----------------------------------------------------------------------------
# Scenario builder (synthetic BaTiO3 + random fields)
# -----------------------------------------------------------------------------

@dataclass
class BenchSetup:
    """Canonical setup reused across all tests in one notebook session."""
    detector: Detector
    crystal: Crystal
    beam: GaussianBeam
    simulator: BraggCoherentDiffraction
    q_vectors: Tensor
    k_i: Tensor
    k_f: Tensor
    bragg_vector: Tensor
    base_continuum: Tensor           # [B, n1, n2, n3, 3], unit-cell resolution
    base_sublattice: Tensor          # [B, n1, n2, n3, n_atoms, 3], frac coords
    base_mask: Tensor                # [1, n1, n2, n3], shared across the batch
    crystal_size: Tuple[int, int, int]
    supercell_size_default: Tuple[int, int, int]
    batch_size: int                  # leading batch dim of base_continuum/sublattice
    continuum_scale_m: float         # retained so tests can rebuild fields at other B
    sublattice_scale_frac: float
    device: torch.device
    dtype: torch.dtype


def build_setup(
    *,
    cif_path: str = 'cifs/BaTiO3.cif',
    crystal_size: Tuple[int, int, int] = (60, 60, 60),
    supercell_size: Tuple[int, int, int] = (2, 2, 2),
    miller_indices: Tuple[int, int, int] = (1, 1, 0),
    theta_B_deg: float = 31.74 / 2.0,
    n_pixels: int = 128,
    pixel_size: float = 75e-6,
    distance: float = 0.1,
    wavelength: float = 1.5406e-10,
    fwhm_beam: float = 20e-9,
    batch_size: int = 1,
    continuum_scale_m: float = 1e-11,   # ~0.1 angstrom per supercell, mild
    sublattice_scale_frac: float = 0.01,  # 1% of a unit cell, mild
    seed: int = 0,
    device: str | torch.device = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> BenchSetup:
    """
    Build a synthetic BaTiO3 scenario with random continuum + sublattice
    displacement fields at unit-cell resolution (downsampling lets every
    test pick its own supercell_size without rebuilding the fields).

    batch_size controls the leading "frame" dimension of the displacement
    fields. Set >1 to simulate processing N frames simultaneously per
    forward call (e.g., for trajectory batching).

    The displacements are seeded so different runs are bit-identical.
    """
    device = torch.device(device)

    detector = Detector(
        num_pixels_i=n_pixels, num_pixels_j=n_pixels,
        pixel_size=pixel_size, distance=distance, wavelength=wavelength,
        dtype=dtype, device=device,
    )

    crystal = Crystal(
        cif_path, crystal_size=crystal_size, wavelength=wavelength,
        dtype=dtype, device=device,
    )
    crystal.align_miller_plane_to_axis(miller_indices, target_axis='x')

    theta_B = torch.deg2rad(torch.tensor(theta_B_deg, dtype=dtype, device=device))
    k_i = torch.tensor(
        [torch.sin(-theta_B), 0.0, torch.cos(-theta_B)],
        dtype=dtype, device=device,
    )
    k_f = torch.tensor(
        [torch.sin(theta_B), 0.0, torch.cos(theta_B)],
        dtype=dtype, device=device,
    )
    q_vectors = detector.calculate_q_vectors(k_i, k_f)
    bragg_vector = detector.k_magnitude * (k_f - k_i)

    # Sample the beam profile at unit-cell resolution so aidino's mask
    # auto-downsampling adapts it to whatever supercell_size each test uses.
    beam = GaussianBeam(wavelength=wavelength, fwhm=fwhm_beam)
    beam.create_profile(crystal=crystal, supercell_size=(1, 1, 1), k_i=k_i)

    simulator = BraggCoherentDiffraction(crystal=crystal)

    n1, n2, n3 = crystal_size
    n_atoms = crystal.n_atoms

    # Build synthetic fields at unit-cell resolution so any supercell_size works.
    # Use a Generator so seeded runs are bit-identical across sessions.
    gen = torch.Generator(device=device).manual_seed(seed)
    base_continuum = continuum_scale_m * torch.randn(
        (batch_size, n1, n2, n3, 3), generator=gen, dtype=dtype, device=device,
    )
    base_sublattice = sublattice_scale_frac * torch.randn(
        (batch_size, n1, n2, n3, n_atoms, 3), generator=gen, dtype=dtype, device=device,
    )

    # Mask stays at batch=1 — beam profile is the same across frames, and
    # aidino broadcasts it over the inferred batch dim of the other inputs.
    base_mask = beam.profile

    return BenchSetup(
        detector=detector,
        crystal=crystal,
        beam=beam,
        simulator=simulator,
        q_vectors=q_vectors,
        k_i=k_i,
        k_f=k_f,
        bragg_vector=bragg_vector,
        base_continuum=base_continuum,
        base_sublattice=base_sublattice,
        base_mask=base_mask,
        crystal_size=crystal_size,
        supercell_size_default=supercell_size,
        batch_size=batch_size,
        continuum_scale_m=continuum_scale_m,
        sublattice_scale_frac=sublattice_scale_frac,
        device=device,
        dtype=dtype,
    )


# -----------------------------------------------------------------------------
# Per-call input assembler
# -----------------------------------------------------------------------------

def _parse_grad_mode(grad_mode: str) -> Tuple[Optional[str], bool]:
    """Map a grad_mode string to (grad_target, do_backward).

    grad_target: 'continuum' | 'sublattice' | None (no_grad)
    do_backward: True if a .backward() call is needed.
    """
    if grad_mode == 'no_grad':       return None,         False
    if grad_mode == 'fwd_continuum': return 'continuum',  False
    if grad_mode == 'fwd_sublattice':return 'sublattice', False
    if grad_mode == 'bwd_continuum': return 'continuum',  True
    if grad_mode == 'bwd_sublattice':return 'sublattice', True
    raise ValueError(
        f"Unknown grad_mode {grad_mode!r}; expected one of {VALID_GRAD_MODES}"
    )


def make_data_inputs(
    setup: BenchSetup,
    *,
    include_sublattice: bool,
    grad_mode: str,
    batch_size: Optional[int] = None,
) -> dict:
    """
    Build the data-input kwargs for calculate_supercell_scattering.

    When batch_size is None (default), clones the base tensors from setup
    as-is (so the batch dim matches setup.batch_size). When given, builds
    fresh random tensors at the requested batch_size — used by sweeps that
    vary the batch dimension. Raises if a sublattice grad mode is requested
    with include_sublattice=False.
    """
    grad_target, _ = _parse_grad_mode(grad_mode)
    if grad_target == 'sublattice' and not include_sublattice:
        raise ValueError(
            f"grad_mode={grad_mode!r} requires include_sublattice=True"
        )

    inputs: dict = {'mask': setup.base_mask}

    if batch_size is None:
        continuum = setup.base_continuum.clone()
    else:
        n1, n2, n3 = setup.crystal_size
        continuum = setup.continuum_scale_m * torch.randn(
            (batch_size, n1, n2, n3, 3),
            dtype=setup.dtype, device=setup.device,
        )
    if grad_target == 'continuum':
        continuum.requires_grad_(True)
    inputs['continuum_displacement'] = continuum

    if include_sublattice:
        if batch_size is None:
            sublattice = setup.base_sublattice.clone()
        else:
            n1, n2, n3 = setup.crystal_size
            n_atoms = setup.crystal.n_atoms
            sublattice = setup.sublattice_scale_frac * torch.randn(
                (batch_size, n1, n2, n3, n_atoms, 3),
                dtype=setup.dtype, device=setup.device,
            )
        if grad_target == 'sublattice':
            sublattice.requires_grad_(True)
        inputs['sublattice_displacement'] = sublattice

    return inputs


def _grad_modes_for(include_sublattice: bool) -> Tuple[str, ...]:
    """Default grad-mode tuple: pair the gradient target with the relevant input."""
    if include_sublattice:
        return ('no_grad', 'fwd_sublattice', 'bwd_sublattice')
    return ('no_grad', 'fwd_continuum', 'bwd_continuum')


def _cleanup(device: torch.device) -> None:
    """Best-effort release of intermediates between configs."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def _n_supercells(crystal_size: Tuple[int, int, int],
                  supercell_size: Tuple[int, int, int]) -> int:
    return int(
        (crystal_size[0] // supercell_size[0])
        * (crystal_size[1] // supercell_size[1])
        * (crystal_size[2] // supercell_size[2])
    )


def make_supercell_grid(
    crystal_size: Tuple[int, int, int],
    *,
    include_full: bool = False,
    max_k: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Generate (k, k, k) supercell sizes where k evenly divides every axis
    of crystal_size. Mirrors the workflow.ipynb convention.

    include_full=False (default) drops the entry that would put the entire
    crystal in a single supercell. max_k optionally caps k to keep sweeps
    short on large crystals.
    """
    g = math.gcd(math.gcd(int(crystal_size[0]), int(crystal_size[1])),
                 int(crystal_size[2]))
    factors = sorted({i for i in range(1, int(math.isqrt(g)) + 1)
                      if g % i == 0} | {g // i for i in range(1, int(math.isqrt(g)) + 1)
                                        if g % i == 0})
    if not include_full:
        factors = [k for k in factors if k != g]
    if max_k is not None:
        factors = [k for k in factors if k <= max_k]
    return [(k, k, k) for k in factors]


def recommend_fft_oversampling(setup: BenchSetup) -> int:
    """M = 2 * ceil(beta) using the BCDI oversampling ratio."""
    beta = setup.detector.calculate_oversampling_ratio(
        setup.crystal.crystal_volume
    ).item()
    return int(2 * math.ceil(beta))


def recommend_supercell_size(setup: BenchSetup) -> Tuple[int, int, int]:
    """
    Largest supercell_size = (d1, d2, d3) such that d_i * |a_i| < ΔX along
    every crystal axis, where ΔX is the detector real-space resolution.
    Coarser supercells violate the supercell approximation.
    """
    delta_x = setup.detector.calculate_resolution()
    a_norms = setup.crystal.lattice_vectors.norm(dim=-1).detach().cpu().numpy()
    d = tuple(max(1, int(delta_x / float(a))) for a in a_norms)
    return d


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_method_vs_supercells(
    setup: BenchSetup,
    supercell_size_grid: Sequence[Tuple[int, int, int]],
    *,
    methods: Sequence[str] = ('direct', 'fft'),
    include_sublattice_options: Sequence[bool] = (False, True),
    grad_modes: Optional[Sequence[str]] = None,
    q_batch_size: int = 64,
    fft_oversampling: Optional[int] = None,
    use_custom_backward_options: Sequence[bool] = (False,),
    n_repeats: int = 3,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """
    Sweep `supercell_size_grid` for both methods. Records time and peak
    additional GPU memory per (include_sublattice, grad_mode, method,
    use_custom_backward); no error metric. swept_value is
    supercell_size_product = d1*d2*d3.

    `use_custom_backward_options` (default `(False,)`) toggles the direct
    method's custom-backward path. FFT iterations only run once per other-
    axis combination since FFT has no custom-backward variant.
    """
    results: List[BenchmarkResult] = []
    M = fft_oversampling or recommend_fft_oversampling(setup)

    # FFT method has no custom-backward path; restrict its iterations to (False,).
    direct_opts = tuple(use_custom_backward_options)
    fft_opts = (False,) if False in direct_opts else direct_opts[:1]
    opts_for = lambda m: direct_opts if m == 'direct' else fft_opts

    total = sum(
        len(supercell_size_grid) *
        sum(len(opts_for(m)) for m in methods) *
        len(grad_modes if grad_modes is not None else _grad_modes_for(s))
        for s in include_sublattice_options
    )
    pbar = tqdm(total=total, disable=verbose,
                desc='method_vs_supercells', bar_format=_BAR_FORMAT)

    for include_sub in include_sublattice_options:
        modes = grad_modes if grad_modes is not None else _grad_modes_for(include_sub)
        for sc_size in supercell_size_grid:
            n_sc = _n_supercells(setup.crystal_size, sc_size)
            for method in methods:
                for grad_mode in modes:
                    for use_custom in opts_for(method):
                        pbar.update(1)
                        # Release the previous iteration's amplitude (+ any retained
                        # autograd graph) so it doesn't sit in GPU memory through
                        # the next config's measurement.
                        _amp = None
                        _, do_bwd = _parse_grad_mode(grad_mode)
                        _cleanup(setup.device)
                        try:
                            data = make_data_inputs(
                                setup, include_sublattice=include_sub, grad_mode=grad_mode,
                            )
                        except ValueError:
                            continue  # sublattice mode without sublattice — skip

                        call_kwargs = {**data}
                        if method == 'direct':
                            call_kwargs['q_batch_size'] = q_batch_size
                            call_kwargs['use_custom_backward'] = use_custom
                        elif method == 'fft':
                            call_kwargs['method'] = 'fft'
                            call_kwargs['bragg_vector'] = setup.bragg_vector
                            call_kwargs['fft_oversampling'] = M

                        def call_fn():
                            return setup.simulator.calculate_supercell_scattering(
                                setup.q_vectors, sc_size, **call_kwargs
                            )

                        elapsed, peak, _amp = measure_call(
                            call_fn,
                            backward=do_bwd,
                            no_grad=(grad_mode == 'no_grad'),
                            device=setup.device,
                            n_repeats=n_repeats,
                        )
                        oom = math.isnan(elapsed)
                        results.append(BenchmarkResult(
                            test_name='method_vs_supercells',
                            swept_param='supercell_size_product',
                            swept_value=int(sc_size[0] * sc_size[1] * sc_size[2]),
                            method=method,
                            include_sublattice=include_sub,
                            grad_mode=grad_mode,
                            elapsed_time_s=elapsed if not oom else float('nan'),
                            peak_memory_gb=peak if not oom else float('nan'),
                            n_supercells=n_sc,
                            fft_oversampling=M if method == 'fft' else None,
                            use_custom_backward=use_custom if method == 'direct' else False,
                            oom=oom,
                            extra={'supercell_size': list(sc_size),
                                   'q_batch_size': q_batch_size if method == 'direct' else None},
                        ))
                        if verbose:
                            tag = 'OOM' if oom else f't={elapsed:.3f}s, mem={peak:.3f}GB'
                            print(f"  [n_sc={n_sc}, {method}, sub={include_sub}, "
                                  f"{grad_mode}, custom_bwd={use_custom}] {tag}")

    pbar.close()
    return results


def test_method_vs_batch_size(
    setup: BenchSetup,
    batch_sizes: Sequence[int],
    *,
    methods: Sequence[str] = ('direct', 'fft'),
    include_sublattice_options: Sequence[bool] = (False, True),
    grad_modes: Optional[Sequence[str]] = None,
    supercell_size: Optional[Tuple[int, int, int]] = None,
    q_batch_size: int = 64,
    fft_oversampling: Optional[int] = None,
    use_custom_backward_options: Sequence[bool] = (False,),
    n_repeats: int = 3,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """
    Sweep `batch_sizes` (leading "frame" dimension) for both methods at
    fixed supercell_size. Records time and peak additional GPU memory per
    (include_sublattice, grad_mode, method, batch_size,
    use_custom_backward). Useful for sizing workloads where many frames
    (e.g., timesteps) are processed per call, and for seeing how GPU
    saturation shifts the optimal q_batch_size.

    `use_custom_backward_options` (default `(False,)`) toggles the direct
    method's custom-backward path. FFT iterations only run once per other-
    axis combination since FFT has no custom-backward variant.
    """
    sc_size = supercell_size or setup.supercell_size_default
    n_sc = _n_supercells(setup.crystal_size, sc_size)
    M = fft_oversampling or recommend_fft_oversampling(setup)
    results: List[BenchmarkResult] = []

    direct_opts = tuple(use_custom_backward_options)
    fft_opts = (False,) if False in direct_opts else direct_opts[:1]
    opts_for = lambda m: direct_opts if m == 'direct' else fft_opts

    total = sum(
        len(batch_sizes) *
        sum(len(opts_for(m)) for m in methods) *
        len(grad_modes if grad_modes is not None else _grad_modes_for(s))
        for s in include_sublattice_options
    )
    pbar = tqdm(total=total, disable=verbose,
                desc='method_vs_batch_size', bar_format=_BAR_FORMAT)

    for include_sub in include_sublattice_options:
        modes = grad_modes if grad_modes is not None else _grad_modes_for(include_sub)
        for B in batch_sizes:
            for method in methods:
                for grad_mode in modes:
                    for use_custom in opts_for(method):
                        pbar.update(1)
                        # Release the previous iteration's amplitude (+ any retained
                        # autograd graph) so it doesn't sit in GPU memory through
                        # the next config's measurement.
                        _amp = None
                        _, do_bwd = _parse_grad_mode(grad_mode)
                        _cleanup(setup.device)
                        try:
                            data = make_data_inputs(
                                setup, include_sublattice=include_sub,
                                grad_mode=grad_mode, batch_size=B,
                            )
                        except ValueError:
                            continue

                        call_kwargs = {**data}
                        if method == 'direct':
                            call_kwargs['q_batch_size'] = q_batch_size
                            call_kwargs['use_custom_backward'] = use_custom
                        elif method == 'fft':
                            call_kwargs['method'] = 'fft'
                            call_kwargs['bragg_vector'] = setup.bragg_vector
                            call_kwargs['fft_oversampling'] = M

                        def call_fn():
                            return setup.simulator.calculate_supercell_scattering(
                                setup.q_vectors, sc_size, **call_kwargs
                            )

                        elapsed, peak, _amp = measure_call(
                            call_fn,
                            backward=do_bwd,
                            no_grad=(grad_mode == 'no_grad'),
                            device=setup.device,
                            n_repeats=n_repeats,
                        )
                        oom = math.isnan(elapsed)
                        results.append(BenchmarkResult(
                            test_name='method_vs_batch_size',
                            swept_param='batch_size',
                            swept_value=int(B),
                            method=method,
                            include_sublattice=include_sub,
                            grad_mode=grad_mode,
                            elapsed_time_s=elapsed if not oom else float('nan'),
                            peak_memory_gb=peak if not oom else float('nan'),
                            n_supercells=n_sc,
                            fft_oversampling=M if method == 'fft' else None,
                            use_custom_backward=use_custom if method == 'direct' else False,
                            oom=oom,
                            extra={'supercell_size': list(sc_size),
                                   'q_batch_size': q_batch_size if method == 'direct' else None},
                        ))
                        if verbose:
                            tag = 'OOM' if oom else f't={elapsed:.3f}s, mem={peak:.3f}GB'
                            print(f"  [B={B}, {method}, sub={include_sub}, "
                                  f"{grad_mode}, custom_bwd={use_custom}] {tag}")

    pbar.close()
    return results


def test_direct_vs_qbatch(
    setup: BenchSetup,
    q_batch_sizes: Sequence[int],
    *,
    include_sublattice_options: Sequence[bool] = (False, True),
    grad_modes: Optional[Sequence[str]] = None,
    supercell_size: Optional[Tuple[int, int, int]] = None,
    use_custom_backward_options: Sequence[bool] = (False,),
    n_repeats: int = 3,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """Sweep `q_batch_size` for the direct method. Records time + memory
    per (include_sublattice, grad_mode, q_batch_size, use_custom_backward).

    `use_custom_backward_options` (default `(False,)`) toggles the direct
    method's custom-backward path. Pass `(False, True)` to compare both;
    each combination is recorded as a separate `BenchmarkResult` with the
    `use_custom_backward` field distinguishing them.
    """
    sc_size = supercell_size or setup.supercell_size_default
    n_sc = _n_supercells(setup.crystal_size, sc_size)
    results: List[BenchmarkResult] = []

    total = sum(
        len(q_batch_sizes) *
        len(grad_modes if grad_modes is not None else _grad_modes_for(s)) *
        len(use_custom_backward_options)
        for s in include_sublattice_options
    )
    pbar = tqdm(total=total, disable=verbose,
                desc='direct_vs_qbatch', bar_format=_BAR_FORMAT)

    for include_sub in include_sublattice_options:
        modes = grad_modes if grad_modes is not None else _grad_modes_for(include_sub)
        for qbs in q_batch_sizes:
            for grad_mode in modes:
                for use_custom in use_custom_backward_options:
                    pbar.update(1)
                    # Release the previous iteration's amplitude (+ any retained
                    # autograd graph) so it doesn't sit in GPU memory through
                    # the next config's measurement.
                    _amp = None
                    _, do_bwd = _parse_grad_mode(grad_mode)
                    _cleanup(setup.device)
                    try:
                        data = make_data_inputs(
                            setup, include_sublattice=include_sub, grad_mode=grad_mode,
                        )
                    except ValueError:
                        continue

                    def call_fn():
                        return setup.simulator.calculate_supercell_scattering(
                            setup.q_vectors, sc_size, q_batch_size=qbs,
                            use_custom_backward=use_custom, **data,
                        )

                    elapsed, peak, _amp = measure_call(
                        call_fn,
                        backward=do_bwd,
                        no_grad=(grad_mode == 'no_grad'),
                        device=setup.device,
                        n_repeats=n_repeats,
                    )
                    oom = math.isnan(elapsed)
                    results.append(BenchmarkResult(
                        test_name='direct_vs_qbatch',
                        swept_param='q_batch_size',
                        swept_value=qbs,
                        method='direct',
                        include_sublattice=include_sub,
                        grad_mode=grad_mode,
                        elapsed_time_s=elapsed if not oom else float('nan'),
                        peak_memory_gb=peak if not oom else float('nan'),
                        n_supercells=n_sc,
                        use_custom_backward=use_custom,
                        oom=oom,
                        extra={'supercell_size': list(sc_size)},
                    ))
                    if verbose:
                        tag = 'OOM' if oom else f't={elapsed:.3f}s, mem={peak:.3f}GB'
                        print(f"  [qbs={qbs}, sub={include_sub}, {grad_mode}, "
                              f"custom_bwd={use_custom}] {tag}")

    pbar.close()
    return results


def test_custom_backward_consistency(
    setup: BenchSetup,
    *,
    supercell_size: Optional[Tuple[int, int, int]] = None,
    q_batch_size: int = 64,
    cases: Optional[Sequence[Dict[str, bool]]] = None,
    grad_tol: float = 1e-4,
    fwd_tol: float = 1e-6,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Verify that ``use_custom_backward=True`` on the direct method produces
    forward outputs and input gradients matching the default autograd path.

    For each ``case`` (a dict over which inputs are present and require_grad),
    runs forward+backward both ways on identical inputs and reports
    ``max abs / max rel`` diff for the forward amplitude and each gradient.

    Returns a list of per-case result dicts.

    ``cases`` defaults to a representative subset of the 15 non-trivial
    combinations of ``{mask, continuum, sublattice, strain}`` — pass an
    explicit list to test other subsets (e.g. for memory benchmarking on
    just the per-atom configurations).
    """
    sc_size = supercell_size or setup.supercell_size_default

    if cases is None:
        # Default: each input alone + the everything-on case.
        cases = [
            {'mask': True,  'continuum': False, 'sublattice': False, 'strain': False},
            {'mask': False, 'continuum': True,  'sublattice': False, 'strain': False},
            {'mask': False, 'continuum': False, 'sublattice': True,  'strain': False},
            {'mask': False, 'continuum': False, 'sublattice': False, 'strain': True},
            {'mask': True,  'continuum': True,  'sublattice': True,  'strain': True},
        ]

    results: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(cases), disable=verbose,
                desc='custom_backward_consistency', bar_format=_BAR_FORMAT)
    for case in cases:
        pbar.update(1)
        # Build identical inputs for both paths, with requires_grad set on
        # the ones the case marks present (so we can compare their gradients).
        inputs_def: dict = {}
        inputs_cus: dict = {}

        if case.get('mask', False):
            base = setup.base_mask
            inputs_def['mask'] = base.clone().detach().requires_grad_(True)
            inputs_cus['mask'] = base.clone().detach().requires_grad_(True)
        if case.get('continuum', False):
            base = setup.base_continuum
            inputs_def['continuum_displacement'] = base.clone().detach().requires_grad_(True)
            inputs_cus['continuum_displacement'] = base.clone().detach().requires_grad_(True)
        if case.get('sublattice', False):
            base = setup.base_sublattice
            inputs_def['sublattice_displacement'] = base.clone().detach().requires_grad_(True)
            inputs_cus['sublattice_displacement'] = base.clone().detach().requires_grad_(True)
        if case.get('strain', False):
            n_sc1 = setup.crystal_size[0] // sc_size[0]
            n_sc2 = setup.crystal_size[1] // sc_size[1]
            n_sc3 = setup.crystal_size[2] // sc_size[2]
            base = 1e-12 * torch.randn(
                (setup.batch_size, n_sc1, n_sc2, n_sc3, 3, 3),
                generator=torch.Generator(device=setup.device).manual_seed(0),
                dtype=setup.dtype, device=setup.device,
            )
            inputs_def['lattice_strain'] = base.clone().detach().requires_grad_(True)
            inputs_cus['lattice_strain'] = base.clone().detach().requires_grad_(True)

        common = dict(
            q_vectors=setup.q_vectors,
            supercell_size=sc_size,
            q_batch_size=q_batch_size,
            method='direct',
        )
        amp_def = setup.simulator.calculate_supercell_scattering(
            **common, use_custom_backward=False, **inputs_def,
        )
        amp_cus = setup.simulator.calculate_supercell_scattering(
            **common, use_custom_backward=True,  **inputs_cus,
        )

        fwd_abs = (amp_def - amp_cus).abs().max().item()
        fwd_scale = amp_def.abs().max().item() + 1e-30
        fwd_rel = fwd_abs / fwd_scale

        # Real scalar loss for backward: sum of intensities (= sum |F|^2).
        (amp_def.abs() ** 2).sum().backward()
        (amp_cus.abs() ** 2).sum().backward()

        grad_diffs: Dict[str, Dict[str, float]] = {}
        for name in inputs_def:
            g_def = inputs_def[name].grad
            g_cus = inputs_cus[name].grad
            d = (g_def - g_cus).abs().max().item()
            scale = g_def.abs().max().item() + 1e-30
            grad_diffs[name] = {
                'max_abs_diff': d,
                'max_rel_diff': d / scale,
                'grad_max_abs': g_def.abs().max().item(),
            }

        fwd_ok = fwd_rel < fwd_tol
        grads_ok = all(g['max_rel_diff'] < grad_tol for g in grad_diffs.values())
        result = {
            'case': dict(case),
            'forward_max_abs_diff': fwd_abs,
            'forward_max_rel_diff': fwd_rel,
            'forward_ok': fwd_ok,
            'gradients': grad_diffs,
            'all_gradients_ok': grads_ok,
            'pass': fwd_ok and grads_ok,
        }
        results.append(result)

        if verbose:
            status = 'PASS' if result['pass'] else 'FAIL'
            present = ', '.join(k for k, v in case.items() if v) or 'none'
            print(f"  [{status}] case=[{present}]  fwd_rel={fwd_rel:.2e}")
            for name, d in grad_diffs.items():
                print(f"         grad/{name}: rel={d['max_rel_diff']:.2e}, "
                      f"abs={d['max_abs_diff']:.2e}, scale={d['grad_max_abs']:.2e}")

        # Detach gradients between cases (defensive, avoids accumulation if
        # the caller reuses the same setup.base_* tensors with grad enabled).
        for x in inputs_def.values():
            if x.grad is not None: x.grad = None
        for x in inputs_cus.values():
            if x.grad is not None: x.grad = None
        _cleanup(setup.device)

    pbar.close()
    return results


def _intensity_from_call(setup: BenchSetup, sc_size, call_kwargs) -> Tensor:
    """Helper: run one direct/FFT call (no_grad) and return intensity tensor."""
    with torch.no_grad():
        amp = setup.simulator.calculate_supercell_scattering(
            setup.q_vectors, sc_size, **call_kwargs,
        )
    return amp.abs() ** 2


def test_direct_error_vs_supercell(
    setup: BenchSetup,
    supercell_sizes: Sequence[Tuple[int, int, int]],
    *,
    include_sublattice_options: Sequence[bool] = (False, True),
    grad_modes: Optional[Sequence[str]] = None,
    q_batch_size: int = 64,
    use_custom_backward_options: Sequence[bool] = (False,),
    n_repeats: int = 3,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """
    Direct method only. For each include_sublattice, compute a reference
    intensity at supercell_size=(1,1,1) under no_grad, then sweep
    `supercell_sizes` recording time and memory per (grad_mode,
    use_custom_backward) plus chi^2 vs the reference (computed once per
    swept value under no_grad — chi^2 is independent of grad_mode and of
    use_custom_backward).

    The reference itself is the first entry — chi^2 should be ~0.

    `use_custom_backward_options` (default `(False,)`) toggles the
    direct method's custom-backward path.
    """
    if tuple(supercell_sizes[0]) != (1, 1, 1):
        raise ValueError("supercell_sizes[0] must be (1, 1, 1) — used as the reference.")

    results: List[BenchmarkResult] = []

    custom_opts = tuple(use_custom_backward_options)
    total = sum(
        len(supercell_sizes) *
        len(grad_modes if grad_modes is not None else _grad_modes_for(s)) *
        len(custom_opts)
        for s in include_sublattice_options
    )
    pbar = tqdm(total=total, disable=verbose,
                desc='direct_error_vs_supercell', bar_format=_BAR_FORMAT)

    for include_sub in include_sublattice_options:
        modes = grad_modes if grad_modes is not None else _grad_modes_for(include_sub)

        # Reference intensity (no_grad, identical inputs) per include_sublattice setting.
        _cleanup(setup.device)
        try:
            data_ref = make_data_inputs(
                setup, include_sublattice=include_sub, grad_mode='no_grad',
            )
        except ValueError:
            continue
        ref_kwargs = {**data_ref, 'q_batch_size': q_batch_size}
        try:
            intensity_ref = _intensity_from_call(setup, (1, 1, 1), ref_kwargs)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            tqdm.write(f"  reference OOM for include_sublattice={include_sub}; skipping")
            continue

        for sc_size in supercell_sizes:
            n_sc = _n_supercells(setup.crystal_size, sc_size)
            for grad_mode in modes:
                for use_custom in custom_opts:
                    pbar.update(1)
                    # Release the previous iteration's amplitude (+ any retained
                    # autograd graph) so it doesn't sit in GPU memory through
                    # the next config's measurement.
                    amp = None
                    _, do_bwd = _parse_grad_mode(grad_mode)
                    _cleanup(setup.device)
                    try:
                        data = make_data_inputs(
                            setup, include_sublattice=include_sub, grad_mode=grad_mode,
                        )
                    except ValueError:
                        continue

                    def call_fn():
                        return setup.simulator.calculate_supercell_scattering(
                            setup.q_vectors, sc_size, q_batch_size=q_batch_size,
                            use_custom_backward=use_custom, **data,
                        )

                    elapsed, peak, amp = measure_call(
                        call_fn,
                        backward=do_bwd,
                        no_grad=(grad_mode == 'no_grad'),
                        device=setup.device,
                        n_repeats=n_repeats,
                    )
                    oom = math.isnan(elapsed)
                    chi2_0: Optional[float] = None
                    # chi^2 is independent of grad_mode (depends only on the forward
                    # amplitude), so compute it once per swept value under no_grad.
                    # Skip the (1, 1, 1) reference itself since chi^2 vs itself is 0.
                    if (not oom and amp is not None and grad_mode == 'no_grad'
                            and tuple(sc_size) != (1, 1, 1)):
                        intensity = amp.abs() ** 2
                        chi2_0 = chi_squared(intensity.detach(), intensity_ref)

                    results.append(BenchmarkResult(
                        test_name='direct_error_vs_supercell',
                        swept_param='supercell_size_product',
                        swept_value=int(sc_size[0] * sc_size[1] * sc_size[2]),
                        method='direct',
                        include_sublattice=include_sub,
                        grad_mode=grad_mode,
                        elapsed_time_s=elapsed if not oom else float('nan'),
                        peak_memory_gb=peak if not oom else float('nan'),
                        chi_squared_0=chi2_0,
                        n_supercells=n_sc,
                        use_custom_backward=use_custom,
                        oom=oom,
                        extra={'supercell_size': list(sc_size),
                               'q_batch_size': q_batch_size},
                    ))
                    if verbose:
                        tag = 'OOM' if oom else f't={elapsed:.3f}s, mem={peak:.3f}GB, chi2_0={chi2_0:.2e}'
                        print(f"  [sc={sc_size}, sub={include_sub}, {grad_mode}, "
                              f"custom_bwd={use_custom}] {tag}")

        # Drop the reference before moving to next include_sub.
        del intensity_ref
        _cleanup(setup.device)

    pbar.close()
    return results


def test_fft_error_vs_oversampling(
    setup: BenchSetup,
    oversampling_factors: Sequence[int],
    *,
    include_sublattice_options: Sequence[bool] = (False, True),
    grad_modes: Optional[Sequence[str]] = None,
    supercell_size: Optional[Tuple[int, int, int]] = None,
    q_batch_size: int = 64,
    n_repeats: int = 3,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """
    FFT method only. For each include_sublattice, compute a direct-method
    reference intensity once (no_grad), then sweep `oversampling_factors`
    recording time and memory per grad_mode plus chi^2 vs the direct
    reference (computed once per M under no_grad — chi^2 is independent
    of grad_mode).
    """
    sc_size = supercell_size or setup.supercell_size_default
    n_sc = _n_supercells(setup.crystal_size, sc_size)
    results: List[BenchmarkResult] = []

    total = sum(
        len(oversampling_factors) *
        len(grad_modes if grad_modes is not None else _grad_modes_for(s))
        for s in include_sublattice_options
    )
    pbar = tqdm(total=total, disable=verbose,
                desc='fft_error_vs_oversampling', bar_format=_BAR_FORMAT)

    for include_sub in include_sublattice_options:
        modes = grad_modes if grad_modes is not None else _grad_modes_for(include_sub)

        _cleanup(setup.device)
        try:
            data_ref = make_data_inputs(
                setup, include_sublattice=include_sub, grad_mode='no_grad',
            )
        except ValueError:
            continue
        ref_kwargs = {**data_ref, 'q_batch_size': q_batch_size}
        try:
            intensity_ref = _intensity_from_call(setup, sc_size, ref_kwargs)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            tqdm.write(f"  direct@sc reference OOM for include_sublattice={include_sub}; skipping")
            continue
        # Ground-truth reference: direct@(1,1,1). chi^2 vs this is the total
        # error (supercell + FFT combined), independent of the supercell
        # approximation. Skipped on OOM with chi_squared_0 left as None.
        intensity_ref_0: Optional[Tensor] = None
        try:
            intensity_ref_0 = _intensity_from_call(setup, (1, 1, 1), ref_kwargs)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            tqdm.write(f"  direct@(1,1,1) reference OOM for include_sublattice={include_sub}; chi^2_0 will be None")

        for M in oversampling_factors:
            for grad_mode in modes:
                pbar.update(1)
                # Release the previous iteration's amplitude (+ any retained
                # autograd graph) so it doesn't sit in GPU memory through
                # the next config's measurement.
                amp = None
                _, do_bwd = _parse_grad_mode(grad_mode)
                _cleanup(setup.device)
                try:
                    data = make_data_inputs(
                        setup, include_sublattice=include_sub, grad_mode=grad_mode,
                    )
                except ValueError:
                    continue

                call_kwargs = {
                    **data,
                    'method': 'fft',
                    'bragg_vector': setup.bragg_vector,
                    'fft_oversampling': M,
                }

                def call_fn():
                    return setup.simulator.calculate_supercell_scattering(
                        setup.q_vectors, sc_size, **call_kwargs,
                    )

                elapsed, peak, amp = measure_call(
                    call_fn,
                    backward=do_bwd,
                    no_grad=(grad_mode == 'no_grad'),
                    device=setup.device,
                    n_repeats=n_repeats,
                )
                oom = math.isnan(elapsed)
                chi2: Optional[float] = None
                chi2_0: Optional[float] = None
                # chi^2 is independent of grad_mode (depends only on the forward
                # amplitude), so compute it once per swept value under no_grad.
                if not oom and amp is not None and grad_mode == 'no_grad':
                    intensity = amp.abs() ** 2
                    chi2 = chi_squared(intensity.detach(), intensity_ref)
                    if intensity_ref_0 is not None:
                        chi2_0 = chi_squared(intensity.detach(), intensity_ref_0)

                results.append(BenchmarkResult(
                    test_name='fft_error_vs_oversampling',
                    swept_param='fft_oversampling',
                    swept_value=int(M),
                    method='fft',
                    include_sublattice=include_sub,
                    grad_mode=grad_mode,
                    elapsed_time_s=elapsed if not oom else float('nan'),
                    peak_memory_gb=peak if not oom else float('nan'),
                    chi_squared=chi2,
                    chi_squared_0=chi2_0,
                    n_supercells=n_sc,
                    fft_oversampling=int(M),
                    oom=oom,
                    extra={'supercell_size': list(sc_size),
                           'q_batch_size': q_batch_size},
                ))
                if verbose:
                    if oom:
                        tag = 'OOM'
                    else:
                        chi2_str = f'{chi2:.2e}' if chi2 is not None else '-'
                        chi2_0_str = f'{chi2_0:.2e}' if chi2_0 is not None else '-'
                        tag = f't={elapsed:.3f}s, mem={peak:.3f}GB, chi2={chi2_str}, chi2_0={chi2_0_str}'
                    print(f"  [M={M}, sub={include_sub}, {grad_mode}] {tag}")

        del intensity_ref
        if intensity_ref_0 is not None:
            del intensity_ref_0
        _cleanup(setup.device)

    pbar.close()
    return results


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _filter(results: Iterable[BenchmarkResult], **predicates) -> List[BenchmarkResult]:
    out = []
    for r in results:
        if all(getattr(r, k) == v for k, v in predicates.items()):
            out.append(r)
    return out


_Y_LABELS = {
    'elapsed_time_s': 'Compute time (s)',
    'peak_memory_gb': 'Peak memory (GB)',
    'chi_squared_0': r'$\chi^2_0$',
    'chi_squared': r'$\chi^2$',
}

_X_LABELS = {
    'n_supercells': 'Number of supercells',
    'q_batch_size': r'$q$-batch size',
    'supercell_size_product': 'Unit cells per supercell',
    'fft_oversampling': r'FFT oversampling factor',
    'batch_size': 'Batch size (frames per call)',
}

# Grad-mode category for plotting. The continuum/sublattice distinction is
# implied by the panel (see _grad_modes_for); the legend just shows the
# direction of the gradient. The direct method's custom-gradient backward
# path is rendered in the same color as its default counterpart but with
# a dashed line — see plot_sweep.
_GRAD_CATEGORIES = ('No grad', 'Forward', 'Backward')
_GRAD_CATEGORY = {
    'no_grad': 'No grad',
    'fwd_continuum': 'Forward',
    'fwd_sublattice': 'Forward',
    'bwd_continuum': 'Backward',
    'bwd_sublattice': 'Backward',
}

_METHOD_LABELS = {'direct': 'Direct', 'fft': 'FFT'}

_FACET_TITLES = {
    'include_sublattice': {True: 'With sublattice', False: 'No sublattice'},
    'method': _METHOD_LABELS,
}

_MARKERS_BY_METHOD = {'direct': 'o', 'fft': 's'}


def _legend_props():
    """plot_utils.props with a smaller size for legend text."""
    from aidino.plot_utils import props
    p = props.copy()
    p.set_size('medium')
    return p


def _default_cmap():
    """Truncated cmcrameri batlow if available, else a viridis subset."""
    import matplotlib.pyplot as plt
    from aidino.plot_utils import truncate_colormap
    try:
        import cmcrameri.cm as cm
        return truncate_colormap(cm.batlow, 0.2, 0.7)
    except ImportError:
        return truncate_colormap(plt.cm.viridis, 0.2, 0.7)


def plot_sweep(
    results: List[BenchmarkResult],
    y: str = 'elapsed_time_s',
    *,
    y_twin: Optional[Union[str, Sequence[str]]] = None,
    group_by: str = 'method',
    facet_by: Optional[str] = 'include_sublattice',
    log_x: bool = True,
    log_y: bool = True,
    log_y_twin: bool = True,
    title: Optional[str] = None,
    axes_size: Tuple[float, float] = (2.5, 2.5),
    vlines: Optional[Sequence[Tuple]] = None,
):
    """
    Scatter `y` vs swept_value, optionally faceted by `facet_by` into
    side-by-side panels with a shared y-axis (y-label on the leftmost
    panel only).

    The figure uses manual axes positioning so each panel is exactly
    `axes_size` inches regardless of label / tick / legend widths, giving
    consistent panel sizes across calls. Tune `axes_size` (and the
    `_PAD_*` constants at the top of this module) to taste.

    A single shared legend lives to the right of the rightmost panel:
        - marker shape: method (Direct: circle, FFT: square)
        - line color:   grad-mode category (No grad, Forward, Backward)
        - line style:   custom-backward variant — solid is the default
                        autograd path; dashed is the direct method's
                        custom-gradient path (`use_custom_backward=True`).
                        Custom variants share their default's color so the
                        legend reads "Forward / Forward (custom)" etc.
                        Only shown when the result set contains custom rows.
    The displacement target the gradient is taken w.r.t. (continuum or
    sublattice) is implied by the panel — see `_grad_modes_for` for the
    pairing per `include_sublattice` setting. `no_grad` is genuinely
    identical for custom vs default (no autograd graph either way), so its
    records are deduplicated and plotted as a single line.

    Uses aidino.plot_utils.format_axis for typography and the cmcrameri
    `batlow` colormap (falls back to viridis if cmcrameri is unavailable).
    OOM points are skipped silently.

    y_twin: optional field (or list of fields) plotted on a shared right-side
    twin axis (colored TWIN_COLOR). Used for overlaying chi^2 on a
    time/memory plot. Only no_grad rows are plotted since chi^2 is
    independent of grad_mode. When given as a list, each field is drawn
    in the same color with a distinct linestyle (solid, dashed, dotted,
    dash-dot in order). Right spine and tick marks are colored on every
    panel; tick labels and y-label appear on the rightmost panel only
    (axes are shared).

    vlines: optional reference lines drawn as dashed grey verticals on every
    panel. Each entry is either (x_value, legend_label) or
    (x_value, legend_label, valid_side) — valid_side is 'left' or 'right'
    to mark which side of the line is the valid regime; the opposite side
    is faintly shaded as the invalid region (omit for no shading).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from aidino.plot_utils import format_axis, props

    cmap = _default_cmap()
    legend_props = _legend_props()

    # Normalize y_twin to a list (possibly empty) for uniform handling.
    if y_twin is None:
        y_twin_list: List[str] = []
    elif isinstance(y_twin, str):
        y_twin_list = [y_twin]
    else:
        y_twin_list = list(y_twin)

    n_cat = len(_GRAD_CATEGORIES)
    cat_color = {
        cat: cmap(i / max(n_cat - 1, 1))
        for i, cat in enumerate(_GRAD_CATEGORIES)
    }

    facet_values = (
        sorted({getattr(r, facet_by) for r in results}) if facet_by else [None]
    )
    n_facets = len(facet_values)

    # ---- Compute figure size and axes positions (inches) ----
    aw, ah = axes_size
    pad_right = _PAD_RIGHT_TWIN if y_twin_list else _PAD_RIGHT
    pad_top = _PAD_TOP_TITLE if title else _PAD_TOP
    fig_w = (_PAD_LEFT + aw * n_facets + _GAP_W * (n_facets - 1)
             + pad_right + _LEGEND_W)
    fig_h = pad_top + ah + _PAD_BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes: List = []
    for col in range(n_facets):
        left_in = _PAD_LEFT + col * (aw + _GAP_W)
        ax = fig.add_axes([
            left_in / fig_w,
            _PAD_BOTTOM / fig_h,
            aw / fig_w,
            ah / fig_h,
        ])
        axes.append(ax)
    # Share y across panels (sharex is irrelevant for a single row).
    for ax in axes[1:]:
        ax.sharey(axes[0])

    twin_axes: List = []
    if y_twin_list:
        for i, ax in enumerate(axes):
            ax_t = ax.twinx()
            if i > 0:
                ax_t.sharey(twin_axes[0])
            twin_axes.append(ax_t)

    methods_seen: set = set()
    # Tracks (category, is_custom) tuples actually drawn — used to build the
    # 5-entry legend (No grad, Forward[, Forward (custom)], Backward[,
    # Backward (custom)]) only including the variants that show up.
    variants_seen: set = set()
    twins_plotted: set = set()

    for panel_idx, (ax, facet_val) in enumerate(zip(axes, facet_values)):
        subset = _filter(results, **({facet_by: facet_val} if facet_by else {}))
        if not subset:
            continue
        group_vals = sorted({getattr(r, group_by) for r in subset})
        grad_modes_present = sorted(
            {r.grad_mode for r in subset}, key=VALID_GRAD_MODES.index,
        )

        for gv in group_vals:
            for gm in grad_modes_present:
                category = _GRAD_CATEGORY.get(gm, gm)
                # no_grad is genuinely identical under custom vs default (the
                # autograd graph isn't built either way), so dedupe and plot
                # one solid line. Forward and Backward both retain different
                # amounts in the graph depending on use_custom_backward, so
                # plot them as paired lines (solid=default, dashed=custom).
                if category == 'No grad':
                    custom_values: tuple = (None,)
                else:
                    custom_values = (False, True)

                for use_custom in custom_values:
                    if use_custom is None:
                        candidates = [
                            r for r in subset
                            if getattr(r, group_by) == gv and r.grad_mode == gm
                            and not r.oom and getattr(r, y) is not None
                        ]
                        # Dedupe by swept_value — multiple records with the
                        # same x are identical when use_custom is irrelevant.
                        seen_x: set = set()
                        pts: list = []
                        for r in candidates:
                            if r.swept_value not in seen_x:
                                pts.append(r)
                                seen_x.add(r.swept_value)
                    else:
                        pts = [
                            r for r in subset
                            if getattr(r, group_by) == gv and r.grad_mode == gm
                            and r.use_custom_backward == use_custom
                            and not r.oom and getattr(r, y) is not None
                        ]
                    if not pts:
                        continue
                    pts.sort(key=lambda r: r.swept_value)
                    xs = [r.swept_value for r in pts]
                    ys = [getattr(r, y) for r in pts]
                    color = cat_color.get(category, 'black')
                    linestyle = '--' if use_custom else '-'
                    marker = _MARKERS_BY_METHOD.get(gv, 'o') if group_by == 'method' else 'o'
                    if group_by == 'method':
                        methods_seen.add(gv)
                    variants_seen.add((category, bool(use_custom)))
                    ax.plot(
                        xs, ys,
                        marker=marker, color=color,
                        linestyle=linestyle, linewidth=1, markersize=5,
                    )

        if y_twin_list:
            ax_t = twin_axes[panel_idx]
            no_grad_subset = [r for r in subset
                              if r.grad_mode == 'no_grad' and not r.oom]
            for y_t_idx, y_t in enumerate(y_twin_list):
                linestyle = _TWIN_LINESTYLES[y_t_idx % len(_TWIN_LINESTYLES)]
                twin_data = [r for r in no_grad_subset
                             if getattr(r, y_t) is not None]
                twin_methods = (
                    sorted({getattr(r, group_by) for r in twin_data})
                    if group_by == 'method' else [None]
                )
                for gv in twin_methods:
                    pts = [
                        r for r in twin_data
                        if gv is None or getattr(r, group_by) == gv
                    ]
                    if not pts:
                        continue
                    pts.sort(key=lambda r: r.swept_value)
                    xs = [r.swept_value for r in pts]
                    ys = [getattr(r, y_t) for r in pts]
                    marker = _MARKERS_BY_METHOD.get(gv, 'o') if gv is not None else 'o'
                    ax_t.plot(
                        xs, ys,
                        marker=marker, color=TWIN_COLOR,
                        linestyle=linestyle, linewidth=1, markersize=5,
                    )
                    twins_plotted.add(y_t)

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        if vlines:
            # Use the data-driven xlim as the shading edge so the patch
            # doesn't extend the view. Lock the xlim afterwards so a later
            # autoscale can't re-include the patch bounds.
            xlim = ax.get_xlim()
            for vline in vlines:
                x_val = vline[0]
                valid_side = vline[2] if len(vline) > 2 else None
                if valid_side == 'left':
                    ax.axvspan(x_val, xlim[1], color='gray', alpha=0.08, zorder=0)
                elif valid_side == 'right':
                    ax.axvspan(xlim[0], x_val, color='gray', alpha=0.08, zorder=0)
                ax.axvline(x_val, color='gray', linestyle='--',
                           linewidth=1, alpha=0.7)
            ax.set_xlim(xlim)

        swept_param = subset[0].swept_param
        is_left = (panel_idx == 0)
        if facet_by and facet_val is not None:
            panel_title = _FACET_TITLES.get(facet_by, {}).get(
                facet_val, f'{facet_by}={facet_val}'
            )
        else:
            panel_title = ''
        format_axis(
            ax,
            xlabel=_X_LABELS.get(swept_param, swept_param),
            ylabel=_Y_LABELS.get(y, y) if is_left else '',
            title=panel_title,
        )
        if not is_left:
            # Hide both major and minor tick labels — matplotlib labels
            # minor ticks on log axes whose range spans less than a decade,
            # and those would otherwise leak through and crowd the panel gap.
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(minor=True), visible=False)

    # Format twin axes: color spine + major/minor ticks on EVERY panel;
    # show tick labels and y-label only on the rightmost (axes are shared).
    if y_twin_list and twin_axes:
        twin_ylabel = r'$\chi^2$ error'
        for i, ax_t in enumerate(twin_axes):
            is_rightmost = (i == n_facets - 1)
            if log_y_twin:
                ax_t.set_yscale('log')
            ax_t.spines['right'].set_color(TWIN_COLOR)
            ax_t.tick_params(axis='y', which='both', colors=TWIN_COLOR)
            if is_rightmost:
                ax_t.set_ylabel(
                    twin_ylabel, color=TWIN_COLOR, fontproperties=props,
                )
                for lbl in ax_t.get_yticklabels(which='both'):
                    lbl.set_fontproperties(props)
                ax_t.yaxis.offsetText.set_fontproperties(props)
                ax_t.yaxis.offsetText.set_color(TWIN_COLOR)
            else:
                plt.setp(ax_t.get_yticklabels(), visible=False)
                plt.setp(ax_t.get_yticklabels(minor=True), visible=False)

    # Build the shared legend: grad-category variants (color + linestyle),
    # then methods (shaped markers in neutral black), then twin, then vlines.
    # Variant order pairs default with custom: No grad, Forward,
    # Forward (custom), Backward, Backward (custom). Only seen variants
    # appear; custom variants get a dashed line in the same category color.
    proxies: list = []
    labels: list = []
    _variant_order = [
        ('No grad', False),
        ('Forward', False),
        ('Forward', True),
        ('Backward', False),
        ('Backward', True),
    ]
    for cat, is_custom in _variant_order:
        if (cat, is_custom) not in variants_seen:
            continue
        label = f'{cat} (custom)' if is_custom else cat
        linestyle = '--' if is_custom else '-'
        proxies.append(Line2D(
            [0], [0], color=cat_color[cat], linewidth=1.5, linestyle=linestyle,
        ))
        labels.append(label)
    if group_by == 'method' and len(methods_seen) > 1:
        # Only show method markers in the legend when more than one method
        # is plotted — otherwise the marker shape carries no information.
        for method in ('direct', 'fft'):
            if method in methods_seen:
                proxies.append(Line2D(
                    [0], [0], color='black', linestyle='',
                    marker=_MARKERS_BY_METHOD[method], markersize=5,
                ))
                labels.append(_METHOD_LABELS[method])
    for y_t_idx, y_t in enumerate(y_twin_list):
        if y_t in twins_plotted:
            linestyle = _TWIN_LINESTYLES[y_t_idx % len(_TWIN_LINESTYLES)]
            proxies.append(Line2D(
                [0], [0], color=TWIN_COLOR, linewidth=1.5,
                linestyle=linestyle,
            ))
            labels.append(_Y_LABELS.get(y_t, y_t))
    if vlines:
        for vline in vlines:
            vlabel = vline[1]
            proxies.append(Line2D(
                [0], [0], color='gray', linestyle='--',
                linewidth=1, alpha=0.7,
            ))
            labels.append(vlabel)
    if proxies:
        legend_left_in = fig_w - _LEGEND_W
        fig.legend(
            proxies, labels,
            loc='center left',
            bbox_to_anchor=(legend_left_in / fig_w, 0.5),
            bbox_transform=fig.transFigure,
            frameon=False, prop=legend_props,
        )

    if title:
        # Center over the axes grid (excluding right padding + legend column).
        axes_left_in = _PAD_LEFT
        axes_right_in = (_PAD_LEFT + aw * n_facets + _GAP_W * (n_facets - 1))
        suptitle_x = 0.5 * (axes_left_in + axes_right_in) / fig_w
        suptitle_y = 1 - (_PAD_TOP_TITLE / 2) / fig_h
        fig.suptitle(title, x=suptitle_x, y=suptitle_y, fontproperties=props)
    return fig
