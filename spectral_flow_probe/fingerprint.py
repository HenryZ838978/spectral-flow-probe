"""BandwidthFingerprint — the radar scan result.

A single PR number is meaningless because PR = f(model, query).
A BandwidthFingerprint is a 7-dimensional vector, one PR per band,
plus per-layer profiles. This is the minimum representation of what
a model's spectral geometry actually is.

Replaces the legacy SpectralReport (which treated PR as a scalar).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bands import BAND_KEYS, BAND_NAMES, BANDS

__all__ = ["BandwidthFingerprint", "BandResult"]


@dataclass
class BandResult:
    """PR measurement for a single frequency band."""

    band_key: str
    name: str
    channel: str
    pr: float
    n_samples: int
    top5_eigenvalues: list[float] = field(default_factory=list)
    depth_profile: list[float] | None = None  # PR per layer, optional

    def to_dict(self) -> dict:
        d = {
            "band_key": self.band_key,
            "name": self.name,
            "channel": self.channel,
            "pr": round(self.pr, 3),
            "n_samples": self.n_samples,
        }
        if self.top5_eigenvalues:
            d["top5_eigenvalues"] = [round(float(e), 4) for e in self.top5_eigenvalues]
        if self.depth_profile:
            d["depth_profile"] = [round(float(p), 3) for p in self.depth_profile]
        return d


@dataclass
class BandwidthFingerprint:
    """7-dimensional spectral fingerprint of a model.

    This is not a scalar. It's a radar signature. Compare two fingerprints
    to see how RL alignment redistributed bandwidth across functional channels.
    """

    model_path: str
    n_params_B: float
    n_layers: int
    hidden_size: int | None
    bands: list[BandResult]
    elapsed_sec: float = 0.0

    # ─── Vector access ──────────────────────────────────────────
    @property
    def pr_vector(self) -> np.ndarray:
        """The 7-dimensional PR vector — one value per band."""
        return np.array([b.pr for b in self.bands])

    @property
    def band_names(self) -> list[str]:
        return [b.name for b in self.bands]

    def pr(self, band_key: str) -> float:
        """Lookup PR for a specific band."""
        for b in self.bands:
            if b.band_key == band_key:
                return b.pr
        raise KeyError(f"Band '{band_key}' not found. Available: {[b.band_key for b in self.bands]}")

    # ─── Aggregate statistics ──────────────────────────────────
    @property
    def mean_pr(self) -> float:
        """Mean PR across all bands. Not a health indicator — just an aggregate."""
        v = self.pr_vector
        return float(v.mean()) if len(v) else 0.0

    @property
    def std_pr(self) -> float:
        """Standard deviation of PR across bands. High = imbalanced."""
        v = self.pr_vector
        return float(v.std()) if len(v) else 0.0

    @property
    def bandwidth_ratio(self) -> float:
        """max(PR) / min(PR) — lower is more uniform.

        Empirical observation: RL alignment typically *reduces* this ratio
        (makes bandwidth more uniform across bands). Base models tend to
        have higher ratios (uneven bandwidth).
        """
        v = self.pr_vector
        if len(v) == 0 or v.min() <= 0:
            return 0.0
        return float(v.max() / v.min())

    @property
    def weakest_band(self) -> BandResult:
        return min(self.bands, key=lambda b: b.pr)

    @property
    def strongest_band(self) -> BandResult:
        return max(self.bands, key=lambda b: b.pr)

    # ─── Export ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "n_params_B": round(self.n_params_B, 2),
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "bands": [b.to_dict() for b in self.bands],
            "aggregate": {
                "mean_pr": round(self.mean_pr, 3),
                "std_pr": round(self.std_pr, 3),
                "bandwidth_ratio": round(self.bandwidth_ratio, 3),
                "weakest_band": self.weakest_band.name,
                "strongest_band": self.strongest_band.name,
            },
            "elapsed_sec": round(self.elapsed_sec, 1),
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_path}  ({self.n_params_B:.2f}B params, {self.n_layers} layers)",
            f"",
            f"  {'Band':<28s} {'PR':>8s}  {'Channel':<25s}",
            f"  {'-' * 65}",
        ]
        for b in self.bands:
            bar = "█" * min(int(b.pr * 3), 30)
            lines.append(f"  {b.name:<28s} {b.pr:8.2f}  {b.channel:<25s} {bar}")
        lines.extend([
            f"",
            f"  Mean PR = {self.mean_pr:.2f}  |  Std = {self.std_pr:.2f}  |  "
            f"BW ratio = {self.bandwidth_ratio:.2f}",
            f"  Weakest: {self.weakest_band.name} (PR={self.weakest_band.pr:.2f})",
            f"  Strongest: {self.strongest_band.name} (PR={self.strongest_band.pr:.2f})",
        ])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ─── Comparison utilities ──────────────────────────────────────

@dataclass
class BandwidthComparison:
    """Side-by-side comparison of two fingerprints (e.g., base vs instruct)."""

    fp_a: BandwidthFingerprint
    fp_b: BandwidthFingerprint
    label_a: str = "Model A"
    label_b: str = "Model B"

    def delta_vector(self) -> np.ndarray:
        """B - A per band."""
        return self.fp_b.pr_vector - self.fp_a.pr_vector

    def delta_pct(self) -> np.ndarray:
        """(B - A) / A × 100 per band."""
        a = self.fp_a.pr_vector
        b = self.fp_b.pr_vector
        return np.where(a > 0, (b - a) / a * 100, 0)

    def lifted_bands(self, threshold_pct: float = 10.0) -> list[str]:
        """Bands where B's PR is significantly higher than A's."""
        deltas = self.delta_pct()
        return [
            self.fp_a.bands[i].name
            for i in range(len(self.fp_a.bands))
            if deltas[i] > threshold_pct
        ]

    def suppressed_bands(self, threshold_pct: float = 10.0) -> list[str]:
        """Bands where B's PR is significantly lower than A's."""
        deltas = self.delta_pct()
        return [
            self.fp_a.bands[i].name
            for i in range(len(self.fp_a.bands))
            if deltas[i] < -threshold_pct
        ]

    def summary(self) -> str:
        lines = [
            f"Comparison: {self.label_a}  vs  {self.label_b}",
            f"",
            f"  {'Band':<28s} {'A':>8s} {'B':>8s} {'Δ':>8s} {'Δ%':>7s}",
            f"  {'-' * 65}",
        ]
        deltas = self.delta_vector()
        delta_pcts = self.delta_pct()
        for i, band in enumerate(self.fp_a.bands):
            name = band.name
            pa = self.fp_a.pr_vector[i]
            pb = self.fp_b.pr_vector[i]
            d = deltas[i]
            dp = delta_pcts[i]
            marker = "▲" if dp > 10 else ("▼" if dp < -10 else "─")
            lines.append(f"  {name:<28s} {pa:8.2f} {pb:8.2f} {d:+8.2f} {dp:+6.1f}% {marker}")

        lines.extend([
            f"",
            f"  Mean:  {self.fp_a.mean_pr:.2f} → {self.fp_b.mean_pr:.2f}",
            f"  BW ratio:  {self.fp_a.bandwidth_ratio:.2f} → {self.fp_b.bandwidth_ratio:.2f}",
        ])

        lifted = self.lifted_bands()
        suppressed = self.suppressed_bands()
        if lifted:
            lines.append(f"  Lifted bands: {', '.join(lifted)}")
        if suppressed:
            lines.append(f"  Suppressed bands: {', '.join(suppressed)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "fingerprint_a": self.fp_a.to_dict(),
            "fingerprint_b": self.fp_b.to_dict(),
            "delta": self.delta_vector().tolist(),
            "delta_pct": self.delta_pct().tolist(),
            "lifted_bands": self.lifted_bands(),
            "suppressed_bands": self.suppressed_bands(),
        }

    def __repr__(self) -> str:
        return self.summary()
