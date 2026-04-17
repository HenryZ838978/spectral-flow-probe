"""Spectral Flow Probe v2 — Phased Array Radar for Transformer representation geometry.

Theory: Representation Bandwidth Economics.
    PR = f(model, query). Total channel capacity is invariant under RL.
    RL alignment is an isovolumetric rotation of the weight singular vectors —
    bandwidth is reallocated, not added or removed.
    Validated across Qwen2.5, Mistral, Yi (3 families, 3 architectures).

Five entry points:

    1. SpectralProbe.scan()           — 7-band radar fingerprint of one model
    2. RotationAnalyzer.compare()     — base vs instruct weight rotation analysis
    3. RotationAnalyzer.profile()     — single-model spectrum profile
    4. BandwidthDiagnostic            — RL data mix audit (the 照妖镜)
    5. SpectralCallback               — fixed-prompt training-time monitor
    +  spectral_pr_loss               — differentiable per-band PR regularizer

⚠️  v2.0 is a breaking change from v1.
    The v1 SpectralReport / random-token monitor / scalar-PR diagnosis are
    removed. v1 measurements were unreliable (CV=30% from random-token
    probes). See README for the full post-mortem.
"""

__version__ = "2.0.0"

from .bands import BANDS, BAND_KEYS, BAND_NAMES
from .fingerprint import BandwidthFingerprint, BandResult, BandwidthComparison
from .probe import SpectralProbe
from .monitor import SpectralCallback
from .rotation import RotationAnalyzer, RotationReport, SpectrumProfile
from .diagnostic import BandwidthDiagnostic, DataMixReport
from .regularizer import spectral_pr_loss, compute_pr_differentiable
from .plot import plot_radar, plot_comparison, plot_grid

__all__ = [
    # Core data structures
    "BandwidthFingerprint", "BandResult", "BandwidthComparison",
    "RotationReport", "SpectrumProfile", "DataMixReport",
    # Tools
    "SpectralProbe",
    "RotationAnalyzer",
    "BandwidthDiagnostic",
    "SpectralCallback",
    "spectral_pr_loss", "compute_pr_differentiable",
    # Plot
    "plot_radar", "plot_comparison", "plot_grid",
    # Constants
    "BANDS", "BAND_KEYS", "BAND_NAMES",
]
