#!/usr/bin/env python3
"""
Exp 8: Weight-Space Anatomy — Look at the object, not through the cloth.
=========================================================================
Directly compare weight matrices between base model (step 0) and DPO-trained
model (step 800). No forward pass, no query dependence. Pure model properties.

Analysis:
1. Per-layer relative Frobenius norm: ||W_dpo - W_base||_F / ||W_base||_F
2. Per-layer SVD spectrum shift: compare singular value distributions
3. Layer-type breakdown: which component types (q_proj, k_proj, v_proj, o_proj,
   gate_proj, up_proj, down_proj, lm_head) changed the most?

Usage:
    python exp8_weight_anatomy.py
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp8_weight_anatomy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "1.7B": {
        "base_path": "/cache/zhangjing/models/Qwen3-1.7B",
        "dpo_path": str(
            Path(__file__).resolve().parent
            / "results"
            / "exp5b_fullft_1.7B"
            / "checkpoints"
            / "step_800"
        ),
    },
    "0.6B": {
        "base_path": "/cache/zhangjing/models/Qwen3-0.6B",
        "dpo_path": str(
            Path(__file__).resolve().parent
            / "results"
            / "exp5_fullft_causality"
            / "checkpoints"
            / "step_800"
        ),
    },
}


def load_state_dict_from_safetensors(model_path: str) -> dict[str, torch.Tensor]:
    """Load all safetensors files from a directory into a single state dict."""
    path = Path(model_path)
    st_files = sorted(path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    state_dict = {}
    for f in st_files:
        state_dict.update(load_file(str(f), device="cpu"))
    return state_dict


def classify_param(name: str) -> tuple[str, int | None, str]:
    """Classify a parameter name into (component_type, layer_index, full_type).

    Returns e.g. ('q_proj', 12, 'self_attn.q_proj') or ('lm_head', None, 'lm_head').
    """
    parts = name.split(".")

    if "lm_head" in name:
        return "lm_head", None, "lm_head"
    if "embed_tokens" in name:
        return "embed_tokens", None, "embed_tokens"
    if "norm" in name and "layers" not in name:
        return "final_norm", None, "final_norm"

    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass

    component = parts[-2] if len(parts) >= 2 else parts[-1]
    weight_type = parts[-1]

    if "q_proj" in name or "q_a_proj" in name or "q_b_proj" in name:
        comp = "q_proj"
    elif "k_proj" in name or "kv_a_proj" in name:
        comp = "k_proj"
    elif "v_proj" in name or "kv_b_proj" in name:
        comp = "v_proj"
    elif "o_proj" in name:
        comp = "o_proj"
    elif "gate_proj" in name:
        comp = "gate_proj"
    elif "up_proj" in name:
        comp = "up_proj"
    elif "down_proj" in name:
        comp = "down_proj"
    elif "layernorm" in name.lower() or "norm" in name.lower():
        comp = "layer_norm"
    else:
        comp = component

    return comp, layer_idx, f"layer_{layer_idx}.{comp}" if layer_idx is not None else comp


def compute_relative_change(w_base: torch.Tensor, w_dpo: torch.Tensor) -> dict:
    """Compute relative Frobenius norm change and cos similarity."""
    diff = (w_dpo.float() - w_base.float())
    base_norm = torch.norm(w_base.float()).item()
    diff_norm = torch.norm(diff).item()
    rel_change = diff_norm / base_norm if base_norm > 0 else 0.0

    cos_sim = torch.nn.functional.cosine_similarity(
        w_base.float().reshape(1, -1), w_dpo.float().reshape(1, -1)
    ).item()

    return {
        "rel_frobenius": rel_change,
        "cosine_similarity": cos_sim,
        "base_norm": base_norm,
        "diff_norm": diff_norm,
    }


def compute_svd_shift(w_base: torch.Tensor, w_dpo: torch.Tensor, top_k: int = 50) -> dict:
    """Compare singular value spectra of base vs DPO weight matrices."""
    if w_base.dim() < 2:
        return {}

    k = min(top_k, min(w_base.shape))

    sv_base = torch.linalg.svdvals(w_base.float())[:k].numpy()
    sv_dpo = torch.linalg.svdvals(w_dpo.float())[:k].numpy()

    sv_base_norm = sv_base / (sv_base.sum() + 1e-12)
    sv_dpo_norm = sv_dpo / (sv_dpo.sum() + 1e-12)

    spectral_diff = float(np.linalg.norm(sv_base_norm - sv_dpo_norm))

    pr_base = float((sv_base.sum() ** 2) / (sv_base ** 2).sum()) if (sv_base ** 2).sum() > 0 else 0
    pr_dpo = float((sv_dpo.sum() ** 2) / (sv_dpo ** 2).sum()) if (sv_dpo ** 2).sum() > 0 else 0

    return {
        "spectral_diff": spectral_diff,
        "pr_base": pr_base,
        "pr_dpo": pr_dpo,
        "pr_change_pct": (pr_dpo - pr_base) / pr_base * 100 if pr_base > 0 else 0,
        "top5_sv_base": sv_base[:5].tolist(),
        "top5_sv_dpo": sv_dpo[:5].tolist(),
    }


def analyze_model(model_size: str):
    """Full weight-space analysis for one model size."""
    cfg = MODEL_CONFIGS[model_size]
    print(f"\n{'='*70}")
    print(f"  Exp 8: Weight-Space Anatomy — {model_size}")
    print(f"{'='*70}")
    print(f"  Base:  {cfg['base_path']}")
    print(f"  DPO:   {cfg['dpo_path']}")

    t0 = time.time()
    print("\n  Loading base weights...")
    sd_base = load_state_dict_from_safetensors(cfg["base_path"])
    print(f"  Loaded {len(sd_base)} parameters in {time.time()-t0:.1f}s")

    t1 = time.time()
    print("  Loading DPO weights...")
    sd_dpo = load_state_dict_from_safetensors(cfg["dpo_path"])
    print(f"  Loaded {len(sd_dpo)} parameters in {time.time()-t1:.1f}s")

    common_keys = sorted(set(sd_base.keys()) & set(sd_dpo.keys()))
    print(f"  Common parameters: {len(common_keys)}")

    per_param = {}
    by_component = defaultdict(list)
    by_layer = defaultdict(list)
    svd_results = {}

    for i, key in enumerate(common_keys):
        w_b = sd_base[key]
        w_d = sd_dpo[key]

        if w_b.shape != w_d.shape:
            print(f"  WARN: shape mismatch for {key}: {w_b.shape} vs {w_d.shape}")
            continue

        comp, layer_idx, full_type = classify_param(key)
        change = compute_relative_change(w_b, w_d)

        per_param[key] = {
            "component": comp,
            "layer": layer_idx,
            **change,
        }

        by_component[comp].append(change["rel_frobenius"])
        if layer_idx is not None:
            by_layer[layer_idx].append(change["rel_frobenius"])

        is_big_matrix = w_b.dim() >= 2 and min(w_b.shape) >= 64
        is_interesting = comp in ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj", "lm_head")
        if is_big_matrix and is_interesting:
            svd = compute_svd_shift(w_b, w_d)
            if svd:
                svd_results[key] = {"component": comp, "layer": layer_idx, **svd}

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(common_keys)} params...")

    component_summary = {}
    for comp, vals in sorted(by_component.items()):
        arr = np.array(vals)
        component_summary[comp] = {
            "mean_rel_change": float(np.mean(arr)),
            "max_rel_change": float(np.max(arr)),
            "std_rel_change": float(np.std(arr)),
            "n_params": len(vals),
        }

    layer_summary = {}
    for layer, vals in sorted(by_layer.items()):
        arr = np.array(vals)
        layer_summary[int(layer)] = {
            "mean_rel_change": float(np.mean(arr)),
            "max_rel_change": float(np.max(arr)),
        }

    print(f"\n  --- Component-Level Summary ({model_size}) ---")
    sorted_comps = sorted(component_summary.items(), key=lambda x: -x[1]["mean_rel_change"])
    for comp, stats in sorted_comps:
        print(f"  {comp:15s}  mean={stats['mean_rel_change']:.6f}  "
              f"max={stats['max_rel_change']:.6f}  n={stats['n_params']}")

    n_layers = len(layer_summary)
    if n_layers > 0:
        print(f"\n  --- Layer-Level Summary ({model_size}, {n_layers} layers) ---")
        layers_sorted = sorted(layer_summary.items())
        first_third = [v["mean_rel_change"] for k, v in layers_sorted[:n_layers // 3]]
        mid_third = [v["mean_rel_change"] for k, v in layers_sorted[n_layers // 3: 2 * n_layers // 3]]
        last_third = [v["mean_rel_change"] for k, v in layers_sorted[2 * n_layers // 3:]]
        print(f"  Early layers  (0-{n_layers//3-1}):    mean change = {np.mean(first_third):.6f}")
        print(f"  Middle layers ({n_layers//3}-{2*n_layers//3-1}):   mean change = {np.mean(mid_third):.6f}")
        print(f"  Late layers   ({2*n_layers//3}-{n_layers-1}):   mean change = {np.mean(last_third):.6f}")

        print(f"\n  --- Per-Layer Detail ---")
        for layer_idx, stats in layers_sorted:
            bar_len = int(stats["mean_rel_change"] * 5000)
            bar = "█" * min(bar_len, 60)
            print(f"  L{layer_idx:3d}  {stats['mean_rel_change']:.6f}  {bar}")

    if svd_results:
        print(f"\n  --- SVD Spectrum Shift (selected matrices) ---")
        svd_by_comp = defaultdict(list)
        for key, r in svd_results.items():
            svd_by_comp[r["component"]].append(r)

        for comp, items in sorted(svd_by_comp.items()):
            diffs = [it["spectral_diff"] for it in items]
            pr_changes = [it["pr_change_pct"] for it in items]
            print(f"  {comp:15s}  spectral_diff={np.mean(diffs):.6f}  "
                  f"PR_shift={np.mean(pr_changes):+.2f}%  (n={len(items)} matrices)")

    total_base_norm = sum(torch.norm(sd_base[k].float()).item() ** 2 for k in common_keys) ** 0.5
    total_diff_norm = sum(
        torch.norm((sd_dpo[k].float() - sd_base[k].float())).item() ** 2
        for k in common_keys
        if sd_base[k].shape == sd_dpo[k].shape
    ) ** 0.5
    global_rel = total_diff_norm / total_base_norm if total_base_norm > 0 else 0

    print(f"\n  === GLOBAL SUMMARY ({model_size}) ===")
    print(f"  Total ||W_base||   = {total_base_norm:.2f}")
    print(f"  Total ||ΔW||       = {total_diff_norm:.4f}")
    print(f"  Global rel change  = {global_rel:.6f} ({global_rel*100:.4f}%)")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    results = {
        "model_size": model_size,
        "base_path": cfg["base_path"],
        "dpo_path": cfg["dpo_path"],
        "n_common_params": len(common_keys),
        "global_rel_change": global_rel,
        "component_summary": component_summary,
        "layer_summary": {str(k): v for k, v in layer_summary.items()},
        "svd_summary": {
            comp: {
                "mean_spectral_diff": float(np.mean([it["spectral_diff"] for it in items])),
                "mean_pr_change_pct": float(np.mean([it["pr_change_pct"] for it in items])),
            }
            for comp, items in svd_by_comp.items()
        } if svd_results else {},
        "elapsed_seconds": elapsed,
    }

    out_file = RESULTS_DIR / f"weight_anatomy_{model_size.replace('.', '')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_file}")

    del sd_base, sd_dpo
    return results


if __name__ == "__main__":
    print("Exp 8: Weight-Space Anatomy — Direct Weight Comparison")
    print("No forward pass. No query dependence. Pure model property.\n")

    all_results = {}
    for size in ["1.7B", "0.6B"]:
        all_results[size] = analyze_model(size)

    combined_file = RESULTS_DIR / "weight_anatomy_combined.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_file}")
