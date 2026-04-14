"""Spectral Flow Probe (SFP) — Geometric diagnostic toolkit for Transformers.

Four entry points:
    1. Auditor:      SpectralProbe.run()          — 20-min post-hoc diagnosis
    2. Monitor:      SpectralCallback             — real-time RL training guard
    3. Planner:      BudgetPlanner.estimate()     — pre-architecture budget check
    4. Regularizer:  spectral_pr_loss()           — differentiable PR-preserving loss
"""

__version__ = "0.1.0"

from .probe import SpectralProbe
from .report import SpectralReport
from .monitor import SpectralCallback
from .planner import BudgetPlanner
from .regularizer import spectral_pr_loss, compute_pr_differentiable
from .plot import plot_diagnosis, plot_compare

__all__ = [
    "SpectralProbe",
    "SpectralReport",
    "SpectralCallback",
    "BudgetPlanner",
    "spectral_pr_loss",
    "compute_pr_differentiable",
    "plot_diagnosis",
    "plot_compare",
]
