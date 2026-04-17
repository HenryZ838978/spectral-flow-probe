"""BandwidthDiagnostic — the RL data mix mirror (照妖镜).

Three diagnostic modes:

    1. Baseline diagnosis:
         "Before you start training, which bands are weak in your base model?"
    
    2. Training audit:
         "You finished RL. Which bands did it actually move? Which did it miss?
          Which got suppressed as a side effect?"
    
    3. Data mix audit:
         "You labeled your training data by band. Does the mix match what
          your model actually needs?"

The premise: PR per band is a bandwidth-utilization measurement. Bands with
low PR are under-activated channels. If your training data has no samples
targeting a weak band, that weakness persists. If your data over-targets a
band that's already strong, you waste training budget.

This tool does not classify your data for you. You bring a dict
{band_name: sample_count} or {band_name: fraction}. Use an LLM to classify
if you don't want to label manually.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .bands import BANDS, BAND_KEYS
from .fingerprint import BandwidthFingerprint, BandwidthComparison

log = logging.getLogger("sfp")

__all__ = ["BandwidthDiagnostic", "DataMixReport"]


# ═══════════════════════════════════════════════════════════════
#  Data mix audit
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataMixReport:
    """Verdict on a training data mix given a model's baseline fingerprint."""
    fingerprint: BandwidthFingerprint
    data_distribution: dict[str, float]  # band_key → fraction

    underserved: list[dict] = field(default_factory=list)
    oversupplied: list[dict] = field(default_factory=list)
    balanced: list[dict] = field(default_factory=list)

    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_path": self.fingerprint.model_path,
            "data_distribution": self.data_distribution,
            "underserved": self.underserved,
            "oversupplied": self.oversupplied,
            "balanced": self.balanced,
            "recommendations": self.recommendations,
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def summary(self) -> str:
        fp = self.fingerprint
        lines = [
            f"Data Mix Diagnostic: {fp.model_path}",
            "",
            f"  {'Band':<28s}  {'PR':>6s}  {'Data %':>8s}  {'Verdict':<12s}",
            f"  {'-' * 65}",
        ]
        # Consolidate rows
        all_entries = []
        for entry in self.underserved:
            all_entries.append((entry, "UNDERSERVED"))
        for entry in self.oversupplied:
            all_entries.append((entry, "OVERSUPPLIED"))
        for entry in self.balanced:
            all_entries.append((entry, "balanced"))
        # Preserve band order
        band_order = {b.band_key: i for i, b in enumerate(fp.bands)}
        all_entries.sort(key=lambda x: band_order.get(x[0]["band_key"], 99))

        for entry, verdict in all_entries:
            frac = entry["data_fraction"] * 100
            marker = "⚠️" if verdict == "UNDERSERVED" else ("🔻" if verdict == "OVERSUPPLIED" else " ")
            lines.append(
                f"  {entry['band_name']:<28s}  {entry['pr']:6.2f}  {frac:7.2f}%  "
                f"{verdict:<12s} {marker}"
            )

        if self.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════

class BandwidthDiagnostic:
    """Diagnostic wrapper around BandwidthFingerprint — the RL data mirror.

    Usage — baseline diagnosis (before training):

        from spectral_flow_probe import SpectralProbe, BandwidthDiagnostic
        fp = SpectralProbe("meta-llama/Llama-3.1-8B").scan()

        diag = BandwidthDiagnostic(fp)
        report = diag.diagnose_baseline()
        print(report)

    Usage — training audit (after training):

        fp_before = SpectralProbe("base_model").scan()
        fp_after  = SpectralProbe("trained_model").scan()
        report = diag.audit_training(fp_before, fp_after)
        print(report)

    Usage — data mix audit:

        # You provide {band_key: fraction} from your LLM classifier
        data_mix = {
            "band1_factual": 0.15,
            "band2_instruction": 0.35,
            "band3_creative": 0.05,
            "band4_code": 0.25,
            "band5_dialogue": 0.10,
            "band6_counterfactual": 0.02,
            "band7_safety": 0.08,
        }
        report = diag.audit_data_mix(fp_before, data_mix)
        print(report)
    """

    def __init__(
        self,
        weak_threshold_pct: float = 25.0,   # band is "weak" if PR < (100-weak)% of max
        underserved_ratio: float = 0.5,     # data share < this × PR share → underserved
        oversupplied_ratio: float = 2.0,    # data share > this × PR share → oversupplied
        lift_threshold_pct: float = 10.0,   # a band is "lifted" if PR +this% during training
    ):
        self.weak_threshold_pct = weak_threshold_pct
        self.underserved_ratio = underserved_ratio
        self.oversupplied_ratio = oversupplied_ratio
        self.lift_threshold_pct = lift_threshold_pct

    # ─────────────────────────────────────────────────────────
    def diagnose_baseline(self, fp: BandwidthFingerprint) -> dict[str, Any]:
        """Identify weak bands in a baseline model — what RL data should target."""
        max_pr = max(b.pr for b in fp.bands) if fp.bands else 1.0
        threshold = max_pr * (1 - self.weak_threshold_pct / 100)

        weak = []
        strong = []
        for b in fp.bands:
            info = {
                "band_key": b.band_key,
                "band_name": b.name,
                "pr": round(b.pr, 3),
                "pr_relative_pct": round(b.pr / max_pr * 100, 1),
            }
            if b.pr < threshold:
                weak.append(info)
            else:
                strong.append(info)

        return {
            "model_path": fp.model_path,
            "bandwidth_ratio": round(fp.bandwidth_ratio, 3),
            "weak_bands": weak,
            "strong_bands": strong,
            "recommendations": self._baseline_recs(fp, weak),
        }

    def _baseline_recs(self, fp: BandwidthFingerprint, weak: list[dict]) -> list[str]:
        recs = []
        if fp.bandwidth_ratio > 2.0:
            recs.append(
                f"Bandwidth ratio {fp.bandwidth_ratio:.2f}x is high — this model has "
                f"uneven bandwidth across functional channels. RL training data should "
                f"prioritize weak bands to improve uniformity."
            )
        for w in weak:
            recs.append(
                f"{w['band_name']} is at {w['pr_relative_pct']:.0f}% of the strongest band — "
                f"consider including dedicated training samples for this channel."
            )
        if fp.bandwidth_ratio < 1.3 and not weak:
            recs.append(
                "Baseline is well-balanced. RL can focus on specialization rather than "
                "remediation."
            )
        return recs

    # ─────────────────────────────────────────────────────────
    def audit_training(
        self,
        fp_before: BandwidthFingerprint,
        fp_after: BandwidthFingerprint,
    ) -> BandwidthComparison:
        """Verify which bands RL actually moved. Did it work as intended?"""
        return BandwidthComparison(
            fp_a=fp_before, fp_b=fp_after,
            label_a="Before training", label_b="After training",
        )

    # ─────────────────────────────────────────────────────────
    def audit_data_mix(
        self,
        fp: BandwidthFingerprint,
        data_distribution: dict[str, float],
    ) -> DataMixReport:
        """Compare the data distribution to the model's bandwidth profile.

        Args:
            fp: A BandwidthFingerprint of the model being trained (usually base).
            data_distribution: Dict mapping band_key → fraction. Values should
                sum to ~1.0 (will be normalized if not).

        The logic:
            Each band's "need" is inversely proportional to its baseline PR —
            weak bands need more data. A band is "underserved" if the data
            fraction is much less than the bandwidth deficit warrants.
        """
        # Normalize data distribution
        total = sum(data_distribution.values())
        if total == 0:
            raise ValueError("Data distribution sums to zero")
        data_norm = {k: v / total for k, v in data_distribution.items()}

        # Fill in zeros for missing bands
        for bk in BAND_KEYS:
            data_norm.setdefault(bk, 0.0)

        # Compute "deficit share" — inverted PR, normalized
        max_pr = max(b.pr for b in fp.bands) if fp.bands else 1.0
        deficits = {}
        for b in fp.bands:
            # How much this band lacks relative to the strongest
            deficits[b.band_key] = max(max_pr - b.pr, 0.0)
        deficit_total = sum(deficits.values())
        if deficit_total > 0:
            deficit_share = {k: v / deficit_total for k, v in deficits.items()}
        else:
            # Model is perfectly balanced — every band should get equal share
            deficit_share = {k: 1.0 / len(BAND_KEYS) for k in BAND_KEYS}

        # Uniform expectation (if model were flat)
        uniform = 1.0 / len(BAND_KEYS)

        underserved, oversupplied, balanced = [], [], []
        for b in fp.bands:
            data_frac = data_norm.get(b.band_key, 0.0)
            # Expected share: weighted between uniform and deficit-weighted
            expected = 0.5 * uniform + 0.5 * deficit_share.get(b.band_key, 0.0)

            if expected > 0:
                ratio = data_frac / expected
            else:
                ratio = 0.0

            entry = {
                "band_key": b.band_key,
                "band_name": b.name,
                "pr": round(b.pr, 3),
                "data_fraction": round(data_frac, 4),
                "expected_fraction": round(expected, 4),
                "ratio": round(ratio, 3),
            }

            if ratio < self.underserved_ratio:
                underserved.append(entry)
            elif ratio > self.oversupplied_ratio:
                oversupplied.append(entry)
            else:
                balanced.append(entry)

        recs = self._data_mix_recs(underserved, oversupplied, fp)

        return DataMixReport(
            fingerprint=fp,
            data_distribution=data_norm,
            underserved=underserved,
            oversupplied=oversupplied,
            balanced=balanced,
            recommendations=recs,
        )

    def _data_mix_recs(
        self,
        underserved: list[dict],
        oversupplied: list[dict],
        fp: BandwidthFingerprint,
    ) -> list[str]:
        recs = []
        for u in underserved:
            recs.append(
                f"Increase share of {u['band_name']} data — currently "
                f"{u['data_fraction']*100:.1f}% of mix, but baseline PR={u['pr']:.2f} "
                f"suggests this channel needs ~{u['expected_fraction']*100:.1f}%."
            )
        for o in oversupplied:
            recs.append(
                f"Consider reducing {o['band_name']} data — currently "
                f"{o['data_fraction']*100:.1f}% of mix, "
                f"{o['ratio']:.1f}× the expected share. Likely wasted capacity."
            )
        if not underserved and not oversupplied:
            recs.append(
                "Data mix is well-aligned with the model's bandwidth profile."
            )
        return recs
