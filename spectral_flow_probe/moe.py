"""MoE auto-detection and per-routing-path PR measurement."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .core import run_pca_layer, compute_pr
from ._compat import encode_prompt
from .report import MoEReport

log = logging.getLogger("sfp")

__all__ = ["detect_moe", "MoEInfo", "measure_moe_routing_pr"]


class MoEInfo:
    """Detected MoE configuration."""
    __slots__ = ("n_experts", "top_k", "moe_layer_indices", "gate_attr")

    def __init__(self, n_experts: int, top_k: int, moe_layer_indices: list[int],
                 gate_attr: str):
        self.n_experts = n_experts
        self.top_k = top_k
        self.moe_layer_indices = moe_layer_indices
        self.gate_attr = gate_attr


def detect_moe(model: Any) -> MoEInfo | None:
    """Auto-detect MoE configuration from model structure."""
    from ._compat import find_decoder_layers
    _, layers, n_layers, _ = find_decoder_layers(model)

    moe_indices = []
    n_experts = 0
    top_k = 0
    gate_attr = ""

    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate", None)
        if gate is None:
            continue
        if hasattr(mlp, "experts"):
            n_exp = len(mlp.experts)
        elif hasattr(gate, "weight"):
            n_exp = gate.weight.shape[0]
        else:
            continue

        conf = getattr(model.config, "num_experts_per_tok", None) or \
               getattr(model.config, "num_selected_experts", None) or 2
        n_experts = n_exp
        top_k = conf
        moe_indices.append(li)
        gate_attr = "mlp.gate"

    if not moe_indices:
        return None

    log.info("MoE detected: %d experts, top-%d, %d MoE layers",
             n_experts, top_k, len(moe_indices))
    return MoEInfo(n_experts, top_k, moe_indices, gate_attr)


@torch.no_grad()
def measure_moe_routing_pr(
    model: Any,
    tokenizer: Any,
    layers: list,
    prompts: list[str],
    *,
    n_fit: int = 20,
    model_tag: str = "",
) -> MoEReport | None:
    """Measure aggregate vs per-routing-path PR at the last layer."""
    info = detect_moe(model)
    if info is None:
        return None

    last_moe = info.moe_layer_indices[-1]
    layer = layers[last_moe]
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    gate = getattr(mlp, "gate", None)
    if gate is None:
        return None

    all_vecs = []
    route_map: dict[str, list[int]] = defaultdict(list)
    captures: dict[str, Any] = {}

    def gate_hook(module, inp, out):
        if isinstance(out, tuple) and len(out) >= 3:
            captures["expert_idx"] = out[2].detach().cpu()

    def layer_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures["hidden"] = h[:, -1, :].detach().float().cpu().numpy()

    for pi, prompt in enumerate(prompts):
        enc = encode_prompt(tokenizer, prompt, model_tag=model_tag)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        captures.clear()
        h1 = gate.register_forward_hook(gate_hook)
        h2 = layers[last_moe].register_forward_hook(layer_hook)
        try:
            model(**enc)
        finally:
            h1.remove()
            h2.remove()

        if "hidden" not in captures:
            continue
        vec = captures["hidden"]
        all_vecs.append(vec)

        if "expert_idx" in captures:
            idx = captures["expert_idx"]
            if idx.dim() >= 2:
                experts = tuple(sorted(idx[-1, :].tolist()))
            else:
                experts = tuple(sorted(idx.tolist()))
            route_key = str(experts)
            route_map[route_key].append(pi)

    if len(all_vecs) < 5:
        return None

    agg_mat = np.vstack(all_vecs)
    agg_ls = run_pca_layer(agg_mat, n_fit=n_fit)
    agg_pr = agg_ls.pr if agg_ls else 0.0

    per_path_prs: dict[str, float] = {}
    for rk, indices in route_map.items():
        if len(indices) < 3:
            continue
        vecs = np.vstack([all_vecs[i] for i in indices])
        ls = run_pca_layer(vecs, n_fit=n_fit)
        if ls:
            per_path_prs[rk] = ls.pr

    if not per_path_prs:
        return MoEReport(
            aggregate_pr=agg_pr,
            per_path_mean_pr=agg_pr,
            per_path_prs={},
            ratio=1.0,
        )

    mean_pp = float(np.mean(list(per_path_prs.values())))
    ratio = agg_pr / mean_pp if mean_pp > 0 else 1.0

    log.info("MoE PR: aggregate=%.2f, per-path mean=%.2f, ratio=%.1fx",
             agg_pr, mean_pp, ratio)
    return MoEReport(
        aggregate_pr=agg_pr,
        per_path_mean_pr=mean_pp,
        per_path_prs=per_path_prs,
        ratio=ratio,
    )
