"""Diagnostic plotting — 4-panel figure and comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .report import SpectralReport

__all__ = ["plot_diagnosis", "plot_compare"]


def plot_diagnosis(report: SpectralReport, save: str | Path | None = None,
                   dpi: int = 150) -> Figure:
    """4-panel diagnostic figure for a single model.

    Panels:
      [0,0] S(depth) curve
      [0,1] PR(depth) curve
      [1,0] Eigenvalue spectrum (first / mid / last layer)
      [1,1] Diagnosis summary text
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"SFP Diagnosis: {report.model_path}", fontsize=13, fontweight="bold")

    depths = list(range(report.n_layers))
    slopes = report.spectral_slope
    prs = report.pr_curve

    # Panel 0: S(depth)
    ax = axes[0, 0]
    ax.plot(depths, slopes, "o-", markersize=3, linewidth=1.5, color="#2563eb")
    ax.axhline(slopes[0], ls="--", alpha=0.3, color="gray")
    ax.axhline(slopes[-1], ls="--", alpha=0.3, color="gray")
    ax.set_xlabel("Layer")
    ax.set_ylabel("S (spectral slope)")
    ax.set_title("S(depth)")
    ax.grid(True, alpha=0.3)

    # Panel 1: PR(depth)
    ax = axes[0, 1]
    ax.plot(depths, prs, "s-", markersize=3, linewidth=1.5, color="#dc2626")
    ax.set_xlabel("Layer")
    ax.set_ylabel("PR (participation ratio)")
    ax.set_title("PR(depth)")
    ax.grid(True, alpha=0.3)

    # Panel 2: Eigenvalue spectrum at 3 depths
    ax = axes[1, 0]
    for idx, label, color in [
        (0, "First", "#16a34a"),
        (report.n_layers // 2, "Mid", "#ca8a04"),
        (report.n_layers - 1, "Last", "#9333ea"),
    ]:
        ev = report.layers[idx].eigenvalues
        if len(ev) > 0:
            ax.semilogy(range(len(ev)), ev, "o-", markersize=2, label=f"L{idx} ({label})",
                        color=color, linewidth=1.2)
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Eigenvalue (log)")
    ax.set_title("Eigenvalue Spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Text summary
    ax = axes[1, 1]
    ax.axis("off")
    dx = report.diagnose()
    lines = [
        f"Params: {report.n_params:.2f}B",
        f"Layers: {report.n_layers}",
        f"ΔS = {report.delta_s:.4f}",
        f"ΔS/layer = {report.delta_s_per_layer:.5f}",
        f"PR(last) = {report.pr_last:.2f}",
        f"RL intensity: {dx['rl_intensity']}",
        f"PR health: {dx['pr_health']}",
    ]
    if report.moe:
        lines.append(f"MoE ratio: {report.moe.ratio:.1f}x")
    lines.append(f"Time: {report.elapsed_sec:.0f}s")
    ax.text(0.1, 0.9, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f9ff", alpha=0.8))

    plt.tight_layout()
    if save:
        fig.savefig(str(save), dpi=dpi, bbox_inches="tight")
    return fig


def plot_compare(
    reports: Sequence[SpectralReport],
    labels: Sequence[str] | None = None,
    save: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Compare S(depth) and PR(depth) across multiple models."""
    if labels is None:
        labels = [r.model_path.split("/")[-1] for r in reports]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SFP Model Comparison", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(reports)))

    for i, (rpt, lbl) in enumerate(zip(reports, labels)):
        frac = np.linspace(0, 1, rpt.n_layers)
        ax1.plot(frac, rpt.spectral_slope, "o-", markersize=2, label=lbl,
                 color=colors[i], linewidth=1.2)
        ax2.plot(frac, rpt.pr_curve, "s-", markersize=2, label=lbl,
                 color=colors[i], linewidth=1.2)

    ax1.set_xlabel("Relative Depth")
    ax1.set_ylabel("S")
    ax1.set_title("S(depth)")
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Relative Depth")
    ax2.set_ylabel("PR")
    ax2.set_title("PR(depth)")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(str(save), dpi=dpi, bbox_inches="tight")
    return fig
