#!/usr/bin/env python3
"""
Exp 9B: Weight-Space SVD — Qwen2.5-7B base vs Instruct
========================================================
The real RLHF comparison. Not our 800-step toy DPO.

Compares weight matrices directly: per-layer Frobenius norm,
SVD spectrum shift, component breakdown.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp9_radar"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_PATH = "/cache/zhangjing/models/Qwen2.5-7B"
INSTRUCT_PATH = "/cache/zhangjing/models/Qwen2.5-7B-Instruct"


def load_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    path = Path(model_path)
    st_files = sorted(path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors in {model_path}")
    sd = {}
    for f in st_files:
        sd.update(load_file(str(f), device="cpu"))
    return sd


def classify_param(name: str) -> tuple[str, int | None]:
    if "lm_head" in name:
        return "lm_head", None
    if "embed_tokens" in name:
        return "embed_tokens", None
    if "norm" in name and "layers" not in name:
        return "final_norm", None

    layer_idx = None
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass

    for tag in ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]:
        if tag in name:
            return tag, layer_idx

    if "layernorm" in name.lower() or "norm" in name.lower():
        return "layer_norm", layer_idx

    return parts[-2] if len(parts) >= 2 else "unknown", layer_idx


def svd_spectrum(w: torch.Tensor, top_k: int = 100) -> np.ndarray:
    if w.dim() < 2:
        return np.array([])
    k = min(top_k, min(w.shape))
    return torch.linalg.svdvals(w.float())[:k].numpy()


def main():
    print("=" * 70)
    print("  Exp 9B: Weight SVD — Qwen2.5-7B base vs Instruct")
    print("=" * 70)

    t0 = time.time()
    print("\n  Loading base weights...")
    sd_base = load_state_dict(BASE_PATH)
    print(f"  {len(sd_base)} params in {time.time()-t0:.1f}s")

    t1 = time.time()
    print("  Loading instruct weights...")
    sd_inst = load_state_dict(INSTRUCT_PATH)
    print(f"  {len(sd_inst)} params in {time.time()-t1:.1f}s")

    common = sorted(set(sd_base) & set(sd_inst))
    print(f"  Common params: {len(common)}")

    by_component = defaultdict(list)
    by_layer = defaultdict(list)
    svd_shifts = {}

    for i, key in enumerate(common):
        wb = sd_base[key]
        wi = sd_inst[key]
        if wb.shape != wi.shape:
            continue

        comp, layer_idx = classify_param(key)
        diff_norm = torch.norm((wi.float() - wb.float())).item()
        base_norm = torch.norm(wb.float()).item()
        rel = diff_norm / base_norm if base_norm > 0 else 0

        cos = torch.nn.functional.cosine_similarity(
            wb.float().reshape(1, -1), wi.float().reshape(1, -1)
        ).item()

        by_component[comp].append(rel)
        if layer_idx is not None:
            by_layer[layer_idx].append(rel)

        if wb.dim() >= 2 and min(wb.shape) >= 64 and comp in (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ):
            sv_b = svd_spectrum(wb)
            sv_i = svd_spectrum(wi)
            k = min(len(sv_b), len(sv_i))
            if k > 0:
                sv_b_n = sv_b[:k] / (sv_b[:k].sum() + 1e-12)
                sv_i_n = sv_i[:k] / (sv_i[:k].sum() + 1e-12)
                spec_diff = float(np.linalg.norm(sv_b_n - sv_i_n))

                pr_b = float((sv_b[:k].sum()**2) / ((sv_b[:k]**2).sum() + 1e-12))
                pr_i = float((sv_i[:k].sum()**2) / ((sv_i[:k]**2).sum() + 1e-12))

                svd_shifts[key] = {
                    "comp": comp, "layer": layer_idx,
                    "rel_frob": rel, "cos_sim": cos,
                    "spectral_diff": spec_diff,
                    "pr_base": pr_b, "pr_instruct": pr_i,
                    "pr_change_pct": (pr_i - pr_b) / pr_b * 100 if pr_b > 0 else 0,
                    "top10_sv_base": sv_b[:10].tolist(),
                    "top10_sv_instruct": sv_i[:10].tolist(),
                }

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(common)}...")

    # Component summary
    print(f"\n  --- Component-Level Change ---")
    comp_summary = {}
    for comp, vals in sorted(by_component.items(), key=lambda x: -np.mean(x[1])):
        arr = np.array(vals)
        comp_summary[comp] = {
            "mean": float(np.mean(arr)), "max": float(np.max(arr)),
            "std": float(np.std(arr)), "n": len(vals),
        }
        print(f"  {comp:15s}  mean={np.mean(arr):.6f}  max={np.max(arr):.6f}  n={len(vals)}")

    # Layer summary
    n_layers = max(by_layer.keys()) + 1 if by_layer else 0
    layer_summary = {}
    print(f"\n  --- Per-Layer Change ({n_layers} layers) ---")
    for l in range(n_layers):
        vals = by_layer.get(l, [])
        if vals:
            m = float(np.mean(vals))
            layer_summary[l] = {"mean": m, "max": float(np.max(vals))}
            bar = "█" * int(m * 200)
            print(f"  L{l:3d}  {m:.6f}  {bar}")

    if n_layers > 0:
        thirds = n_layers // 3
        early = [layer_summary[l]["mean"] for l in range(thirds) if l in layer_summary]
        mid = [layer_summary[l]["mean"] for l in range(thirds, 2*thirds) if l in layer_summary]
        late = [layer_summary[l]["mean"] for l in range(2*thirds, n_layers) if l in layer_summary]
        print(f"\n  Early  (0-{thirds-1}):     mean={np.mean(early):.6f}")
        print(f"  Middle ({thirds}-{2*thirds-1}):    mean={np.mean(mid):.6f}")
        print(f"  Late   ({2*thirds}-{n_layers-1}):    mean={np.mean(late):.6f}")

    # SVD spectrum analysis
    if svd_shifts:
        print(f"\n  --- SVD Spectrum Shift ---")
        svd_by_comp = defaultdict(list)
        for k, v in svd_shifts.items():
            svd_by_comp[v["comp"]].append(v)

        svd_summary = {}
        for comp, items in sorted(svd_by_comp.items(),
                                   key=lambda x: -np.mean([i["spectral_diff"] for i in x[1]])):
            diffs = [it["spectral_diff"] for it in items]
            pr_changes = [it["pr_change_pct"] for it in items]
            frobs = [it["rel_frob"] for it in items]
            svd_summary[comp] = {
                "mean_spectral_diff": float(np.mean(diffs)),
                "mean_pr_change_pct": float(np.mean(pr_changes)),
                "mean_rel_frob": float(np.mean(frobs)),
                "n": len(items),
            }
            print(f"  {comp:15s}  spec_diff={np.mean(diffs):.6f}  "
                  f"PR_shift={np.mean(pr_changes):+.2f}%  "
                  f"frob={np.mean(frobs):.6f}  (n={len(items)})")

        # Find the layers with largest SVD shifts
        print(f"\n  --- Top 10 Most Changed Matrices (by spectral_diff) ---")
        top10 = sorted(svd_shifts.items(), key=lambda x: -x[1]["spectral_diff"])[:10]
        for key, info in top10:
            print(f"  {key}")
            print(f"    comp={info['comp']}  layer={info['layer']}  "
                  f"spec_diff={info['spectral_diff']:.6f}  "
                  f"PR: {info['pr_base']:.2f} → {info['pr_instruct']:.2f} "
                  f"({info['pr_change_pct']:+.1f}%)  "
                  f"frob={info['rel_frob']:.6f}")

    # Global
    total_base = sum(torch.norm(sd_base[k].float()).item()**2 for k in common)**0.5
    total_diff = sum(
        torch.norm((sd_inst[k].float() - sd_base[k].float())).item()**2
        for k in common if sd_base[k].shape == sd_inst[k].shape
    )**0.5
    global_rel = total_diff / total_base if total_base > 0 else 0

    print(f"\n  === GLOBAL ===")
    print(f"  ||W_base||     = {total_base:.2f}")
    print(f"  ||ΔW||         = {total_diff:.4f}")
    print(f"  Global change  = {global_rel:.6f} ({global_rel*100:.4f}%)")
    print(f"  Time: {time.time()-t0:.0f}s")

    results = {
        "base_path": BASE_PATH,
        "instruct_path": INSTRUCT_PATH,
        "global_rel_change": global_rel,
        "component_summary": comp_summary,
        "layer_summary": {str(k): v for k, v in layer_summary.items()},
        "svd_summary": svd_summary if svd_shifts else {},
        "top10_changed": [
            {"key": k, **v} for k, v in sorted(svd_shifts.items(), key=lambda x: -x[1]["spectral_diff"])[:10]
        ] if svd_shifts else [],
    }

    out = RESULTS_DIR / "weight_svd_qwen25_7b_base_vs_instruct.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
