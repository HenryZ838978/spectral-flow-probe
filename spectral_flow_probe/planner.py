"""BudgetPlanner — pre-architecture spectral budget estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["BudgetPlanner", "BudgetEstimate"]

# Empirical reference data from 11-model study.
# (n_params_B, delta_s_per_layer, pr_last, rl_category)
_REFERENCE = [
    (0.6,  0.0060, 7.0,  "light"),
    (4.0,  0.0045, 5.5,  "moderate"),
    (7.0,  0.0052, 8.5,  "light"),
    (7.0,  0.0035, 6.8,  "moderate"),
    (8.0,  0.0040, 7.2,  "moderate"),
    (14.0, 0.0048, 9.5,  "moderate"),
    (14.0, 0.0018, 2.8,  "heavy"),
    (30.0, 0.0042, 10.0, "moderate"),
]


@dataclass
class BudgetEstimate:
    """Predicted spectral budget for a planned model."""
    n_params_B: float
    n_layers: int
    n_modalities: int
    rl_category: str
    predicted_delta_s_per_layer: float
    predicted_delta_s: float
    predicted_pr_last: float
    headroom: float
    recommendation: str

    def __repr__(self) -> str:
        lines = [
            f"BudgetEstimate({self.n_params_B:.1f}B, {self.n_layers}L, "
            f"{self.n_modalities} modalities, RL={self.rl_category})",
            f"  Predicted ΔS/layer: {self.predicted_delta_s_per_layer:.5f}",
            f"  Predicted ΔS:       {self.predicted_delta_s:.4f}",
            f"  Predicted PR(last): {self.predicted_pr_last:.1f}",
            f"  Headroom score:     {self.headroom:.2f}",
            f"  Recommendation:     {self.recommendation}",
        ]
        return "\n".join(lines)


class BudgetPlanner:
    """Estimate spectral budget for a planned architecture.

    Usage::

        from spectral_flow_probe import BudgetPlanner

        plan = BudgetPlanner.estimate(
            n_params_B=14,
            n_layers=40,
            n_modalities=1,
            rl_category="moderate",
        )
        print(plan)
    """

    # Fitted from empirical data: ΔS/layer ≈ a * log(N) + b
    _A = 0.0012
    _B = 0.0030

    # RL penalty factor on ΔS/layer
    _RL_PENALTY = {
        "none": 0.0,
        "light": 0.0005,
        "moderate": 0.0012,
        "heavy": 0.0025,
        "extreme": 0.0040,
    }

    # Modality tax per additional modality
    _MOD_TAX = 0.0005

    @classmethod
    def estimate(
        cls,
        n_params_B: float,
        n_layers: int,
        n_modalities: int = 1,
        rl_category: str = "moderate",
    ) -> BudgetEstimate:
        """Predict spectral budget for a planned model.

        Args:
            n_params_B: Total parameter count in billions.
            n_layers: Number of Transformer decoder layers.
            n_modalities: Number of input modalities (1=text-only).
            rl_category: Expected RL intensity — "none", "light", "moderate", "heavy", "extreme".
        """
        rl_cat = rl_category.lower()
        if rl_cat not in cls._RL_PENALTY:
            raise ValueError(f"Unknown rl_category: {rl_category}. "
                             f"Choose from {list(cls._RL_PENALTY)}")

        f_n = cls._A * math.log(max(n_params_B, 0.1)) + cls._B
        g_rl = cls._RL_PENALTY[rl_cat]
        h_mod = cls._MOD_TAX * max(n_modalities - 1, 0)

        ds_per_layer = max(f_n - g_rl - h_mod, 0.0001)
        ds_total = ds_per_layer * n_layers
        pr_last = _estimate_pr(n_params_B, ds_per_layer)

        # Headroom: ratio of remaining budget after alignment costs
        raw_budget = f_n * n_layers
        used = (g_rl + h_mod) * n_layers
        headroom = 1.0 - (used / raw_budget) if raw_budget > 0 else 0.0

        if headroom > 0.6:
            rec = "Good headroom. Model can absorb RL + modality costs."
        elif headroom > 0.3:
            rec = "Moderate headroom. Consider lighter RL or fewer modalities."
        elif headroom > 0.1:
            rec = "Tight budget. Risk of representation compression."
        else:
            rec = "Insufficient budget. Increase params or reduce alignment intensity."

        return BudgetEstimate(
            n_params_B=n_params_B,
            n_layers=n_layers,
            n_modalities=n_modalities,
            rl_category=rl_cat,
            predicted_delta_s_per_layer=ds_per_layer,
            predicted_delta_s=ds_total,
            predicted_pr_last=pr_last,
            headroom=headroom,
            recommendation=rec,
        )


def _estimate_pr(n_params_B: float, ds_per_layer: float) -> float:
    """Rough PR estimate from budget and params."""
    base_pr = 3.0 + 2.5 * math.log(max(n_params_B, 0.1))
    expansion = ds_per_layer * 1500
    return max(base_pr + expansion, 1.0)
