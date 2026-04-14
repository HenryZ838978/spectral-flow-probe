"""SpectralReport — structured result container with diagnosis."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SpectralReport", "LayerResult"]


@dataclass
class LayerResult:
    layer: int
    S: float
    r2: float
    pr: float
    pc01: float
    eigenvalues: np.ndarray

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "S": round(self.S, 6),
            "r2": round(self.r2, 4),
            "pr": round(self.pr, 3),
            "pc01": round(self.pc01, 3),
        }


@dataclass
class MoEReport:
    aggregate_pr: float
    per_path_mean_pr: float
    per_path_prs: dict[str, float]
    ratio: float  # aggregate / per_path_mean

    def to_dict(self) -> dict:
        return {
            "aggregate_pr": round(self.aggregate_pr, 2),
            "per_path_mean_pr": round(self.per_path_mean_pr, 2),
            "ratio": round(self.ratio, 2),
            "per_path_prs": {k: round(v, 2) for k, v in self.per_path_prs.items()},
        }


@dataclass
class SpectralReport:
    model_path: str
    n_params: float
    n_layers: int
    hidden_size: int | None
    n_prompts: int
    layers: list[LayerResult]
    moe: MoEReport | None = None
    elapsed_sec: float = 0.0
    _diagnosis: dict | None = field(default=None, repr=False)

    @property
    def spectral_slope(self) -> list[float]:
        """S(depth) curve."""
        return [lr.S for lr in self.layers]

    @property
    def pr_curve(self) -> list[float]:
        """PR(depth) curve."""
        return [lr.pr for lr in self.layers]

    @property
    def s_first(self) -> float:
        return self.layers[0].S if self.layers else 0.0

    @property
    def s_last(self) -> float:
        return self.layers[-1].S if self.layers else 0.0

    @property
    def delta_s(self) -> float:
        """Total spectral expansion ΔS = S(last) - S(first)."""
        return self.s_last - self.s_first

    @property
    def delta_s_per_layer(self) -> float:
        return self.delta_s / self.n_layers if self.n_layers > 0 else 0.0

    @property
    def pr_first(self) -> float:
        return self.layers[0].pr if self.layers else 0.0

    @property
    def pr_last(self) -> float:
        return self.layers[-1].pr if self.layers else 0.0

    def diagnose(self) -> dict[str, Any]:
        """Heuristic diagnosis based on spectral signature.

        Thresholds calibrated from 11-model empirical study.
        """
        if self._diagnosis is not None:
            return self._diagnosis

        d: dict[str, Any] = {}
        d["delta_s"] = self.delta_s
        d["delta_s_per_layer"] = self.delta_s_per_layer
        d["pr_last"] = self.pr_last

        # RL intensity indicator
        if self.delta_s < -0.05:
            d["rl_intensity"] = "extreme"
            d["rl_note"] = "Negative ΔS — likely heavy RL; representation collapse risk"
        elif self.delta_s_per_layer < 0.002:
            d["rl_intensity"] = "heavy"
            d["rl_note"] = "Very low ΔS/layer — strong alignment compression"
        elif self.delta_s_per_layer < 0.005:
            d["rl_intensity"] = "moderate"
            d["rl_note"] = "Moderate ΔS/layer — typical chat-tuned model"
        else:
            d["rl_intensity"] = "light"
            d["rl_note"] = "High ΔS/layer — light alignment or base model"

        # PR health
        if self.pr_last > 10:
            d["pr_health"] = "excellent"
        elif self.pr_last > 5:
            d["pr_health"] = "good"
        elif self.pr_last > 2:
            d["pr_health"] = "compressed"
        else:
            d["pr_health"] = "collapsed"

        # Isotropy gradient
        slopes = self.spectral_slope
        if len(slopes) > 4:
            first_q = np.mean(slopes[:len(slopes) // 4])
            last_q = np.mean(slopes[-(len(slopes) // 4):])
            d["isotropy_gradient"] = float(last_q - first_q)

        # MoE
        if self.moe is not None:
            d["moe_routing_diversity"] = self.moe.ratio
            if self.moe.ratio > 3:
                d["moe_note"] = "High routing diversity — manifold diversification"
            else:
                d["moe_note"] = "Low routing diversity"

        self._diagnosis = d
        return d

    def to_dict(self) -> dict:
        d = {
            "model_path": self.model_path,
            "n_params_B": round(self.n_params, 2),
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "n_prompts": self.n_prompts,
            "delta_s": round(self.delta_s, 6),
            "delta_s_per_layer": round(self.delta_s_per_layer, 6),
            "s_first": round(self.s_first, 6),
            "s_last": round(self.s_last, 6),
            "pr_first": round(self.pr_first, 3),
            "pr_last": round(self.pr_last, 3),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "layers": [lr.to_dict() for lr in self.layers],
            "diagnosis": self.diagnose(),
        }
        if self.moe is not None:
            d["moe"] = self.moe.to_dict()
        return d

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        dx = self.diagnose()
        lines = [
            f"Model: {self.model_path} ({self.n_params:.2f}B params, {self.n_layers} layers)",
            f"ΔS = {self.delta_s:.4f} (ΔS/layer = {self.delta_s_per_layer:.5f})",
            f"PR(last) = {self.pr_last:.2f}  |  S(first) = {self.s_first:.4f}  |  S(last) = {self.s_last:.4f}",
            f"RL intensity: {dx['rl_intensity']}  |  PR health: {dx['pr_health']}",
        ]
        if self.moe:
            lines.append(
                f"MoE: aggregate PR={self.moe.aggregate_pr:.1f}, "
                f"per-path mean PR={self.moe.per_path_mean_pr:.1f}, "
                f"ratio={self.moe.ratio:.1f}x"
            )
        lines.append(f"Elapsed: {self.elapsed_sec:.1f}s")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
