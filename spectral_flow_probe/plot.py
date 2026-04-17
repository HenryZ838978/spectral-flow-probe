"""Plotting — radar charts and comparison panels for BandwidthFingerprint."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .fingerprint import BandwidthFingerprint, BandwidthComparison

__all__ = ["plot_radar", "plot_comparison", "plot_grid"]


def plot_radar(
    fp: BandwidthFingerprint,
    save: str | Path | None = None,
    color: str = "#3498db",
    title: str | None = None,
    dpi: int = 150,
) -> Figure:
    """Single-model 7-band radar chart."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    n = len(fp.bands)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    vals = list(fp.pr_vector) + [fp.pr_vector[0]]
    ax.plot(angles, vals, "o-", linewidth=2.5, color=color)
    ax.fill(angles, vals, alpha=0.2, color=color)

    labels = [b.name.replace(" ", "\n") for b in fp.bands]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    title_text = title or f"Bandwidth Fingerprint\n{Path(fp.model_path).name}"
    ax.set_title(
        f"{title_text}\nMean PR: {fp.mean_pr:.2f}  |  BW ratio: {fp.bandwidth_ratio:.2f}",
        fontsize=11, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    if save:
        fig.savefig(str(save), dpi=dpi, bbox_inches="tight")
    return fig


def plot_comparison(
    cmp: BandwidthComparison,
    save: str | Path | None = None,
    color_a: str = "#3498db",
    color_b: str = "#e74c3c",
    dpi: int = 150,
) -> Figure:
    """Side-by-side radar overlay of two fingerprints."""
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    n = len(cmp.fp_a.bands)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    vals_a = list(cmp.fp_a.pr_vector) + [cmp.fp_a.pr_vector[0]]
    vals_b = list(cmp.fp_b.pr_vector) + [cmp.fp_b.pr_vector[0]]

    ax.plot(angles, vals_a, "o-", linewidth=2.5, color=color_a, label=cmp.label_a)
    ax.fill(angles, vals_a, alpha=0.15, color=color_a)
    ax.plot(angles, vals_b, "s-", linewidth=2.5, color=color_b, label=cmp.label_b)
    ax.fill(angles, vals_b, alpha=0.15, color=color_b)

    labels = [b.name.replace(" ", "\n") for b in cmp.fp_a.bands]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_title(
        f"{cmp.label_a}  vs  {cmp.label_b}\n"
        f"Mean PR: {cmp.fp_a.mean_pr:.2f} → {cmp.fp_b.mean_pr:.2f}  |  "
        f"BW ratio: {cmp.fp_a.bandwidth_ratio:.2f} → {cmp.fp_b.bandwidth_ratio:.2f}",
        fontsize=11, fontweight="bold", pad=25,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    if save:
        fig.savefig(str(save), dpi=dpi, bbox_inches="tight")
    return fig


def plot_grid(
    fingerprints: Sequence[BandwidthFingerprint],
    labels: Sequence[str] | None = None,
    save: str | Path | None = None,
    cols: int = 3,
    dpi: int = 150,
) -> Figure:
    """Grid of radar charts for multiple models."""
    n = len(fingerprints)
    if labels is None:
        labels = [Path(fp.model_path).name for fp in fingerprints]

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows),
                              subplot_kw=dict(polar=True))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    n_bands = len(fingerprints[0].bands)
    angles = np.linspace(0, 2 * np.pi, n_bands, endpoint=False).tolist()
    angles += angles[:1]
    band_labels = [b.name.replace(" ", "\n") for b in fingerprints[0].bands]

    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (fp, lbl) in enumerate(zip(fingerprints, labels)):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        vals = list(fp.pr_vector) + [fp.pr_vector[0]]
        ax.plot(angles, vals, "o-", linewidth=2, color=colors[i])
        ax.fill(angles, vals, alpha=0.2, color=colors[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(band_labels, fontsize=8)
        ax.set_title(f"{lbl}\nBW ratio: {fp.bandwidth_ratio:.2f}",
                     fontsize=10, fontweight="bold", pad=15)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    if save:
        fig.savefig(str(save), dpi=dpi, bbox_inches="tight")
    return fig
