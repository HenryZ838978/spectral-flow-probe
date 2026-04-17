"""Differentiable spectral PR loss for RL training regularization.

Key insight: PR = ||H||_F^4 / ||H^T H||_F^2  is fully differentiable
using only matrix norms — no SVD needed in the backward pass.

⚠️  v2 reinterpretation:
    The v1 framing was "prevent PR collapse during RL". We now know RL
    (in practice, on real RLHF pipelines) does not collapse PR globally —
    it rotates the singular vectors per query type. This loss should be
    interpreted as "constrain bandwidth redistribution magnitude on a
    specific query band", NOT "prevent global PR drop".

    The mathematical content is unchanged. Use it band-by-band: feed it
    hidden states from a specific functional band (e.g., creative-band
    prompts) to keep RL from over-suppressing that band's bandwidth.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["spectral_pr_loss", "compute_pr_differentiable"]


def compute_pr_differentiable(H: torch.Tensor) -> torch.Tensor:
    """Differentiable Participation Ratio from hidden state matrix.

    Args:
        H: (N, d) hidden state tensor.  Must require grad or be part of a grad graph.

    Returns:
        Scalar PR value (differentiable).
    """
    H_centered = H - H.mean(dim=0, keepdim=True)
    fro_sq = (H_centered ** 2).sum()            # ||H||_F^2
    gram = H_centered.T @ H_centered            # (d, d) = H^T H
    gram_fro_sq = (gram ** 2).sum()             # ||H^T H||_F^2
    pr = (fro_sq ** 2) / (gram_fro_sq + 1e-8)
    return pr


def spectral_pr_loss(
    H: torch.Tensor,
    target_pr: float | torch.Tensor | None = None,
    *,
    mode: str = "floor",
    margin: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable loss to preserve or target a Participation Ratio.

    Args:
        H: (N, d) or (B, N, d) hidden states.
        target_pr: Target PR value.  If None, uses a heuristic floor.
        mode: "floor" — penalize only when PR < target (one-sided).
              "target" — penalize deviation in both directions (MSE).
        margin: Soft margin for floor mode (no penalty if PR > target - margin).
        reduction: "mean" | "sum" | "none".

    Returns:
        Scalar loss (or per-batch if reduction="none" and B > 1).
    """
    if H.dim() == 3:
        losses = torch.stack([
            _single_loss(H[i], target_pr, mode, margin)
            for i in range(H.shape[0])
        ])
        if reduction == "mean":
            return losses.mean()
        elif reduction == "sum":
            return losses.sum()
        return losses

    return _single_loss(H, target_pr, mode, margin)


def _single_loss(
    H: torch.Tensor,
    target_pr: float | torch.Tensor | None,
    mode: str,
    margin: float,
) -> torch.Tensor:
    pr = compute_pr_differentiable(H)

    if target_pr is None:
        target_pr = max(float(H.shape[1]) * 0.1, 3.0)
    target = torch.tensor(float(target_pr), device=H.device, dtype=H.dtype) \
        if not isinstance(target_pr, torch.Tensor) else target_pr

    if mode == "floor":
        diff = target - margin - pr
        return F.relu(diff) ** 2
    elif mode == "target":
        return (pr - target) ** 2
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'floor' or 'target'.")
