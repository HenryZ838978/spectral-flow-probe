#!/usr/bin/env python3
"""
Exp 9B: Weight SVD — Qwen2.5-7B base vs Instruct (GPU-accelerated)
====================================================================
Uses GPU for SVD computation. Samples key layers instead of brute-forcing all.
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

GPU_ID = int(os.environ.get("EXP9B_GPU", "6"))
DEVICE = f"cuda:{GPU_ID}"

INTERESTING_COMPS = {"q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj", "lm_head"}


def load_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    path = Path(model_path)
    st_files = sorted(path.glob("*.safetensors"))
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

    if "norm" in name.lower():
        return "layer_norm", layer_idx
    return parts[-2] if len(parts) >= 2 else "unknown", layer_idx


def main():
    print("=" * 70)
    print(f"  Exp 9B: Weight SVD — base vs Instruct (GPU {GPU_ID})")
    print("=" * 70)

    t0 = time.time()
    print("\n  Loading base...")
    sd_base = load_state_dict(BASE_PATH)
    print(f"  Loading instruct...")
    sd_inst = load_state_dict(INSTRUCT_PATH)

    common = sorted(set(sd_base) & set(sd_inst))
    print(f"  Common params: {len(common)}, loaded in {time.time()-t0:.0f}s")

    by_component = defaultdict(list)
    by_layer = defaultdict(list)
    svd_results = []

    for i, key in enumerate(common):
        wb = sd_base[key]
        wi = sd_inst[key]
        if wb.shape != wi.shape:
            continue

        comp, layer_idx = classify_param(key)

        base_norm = torch.norm(wb.float()).item()
        diff_norm = torch.norm((wi.float() - wb.float())).item()
        rel = diff_norm / base_norm if base_norm > 0 else 0

        by_component[comp].append(rel)
        if layer_idx is not None:
            by_layer[layer_idx].append(rel)

        if wb.dim() >= 2 and comp in INTERESTING_COMPS and min(wb.shape) >= 64:
            wb_g = wb.float().to(DEVICE)
            wi_g = wi.float().to(DEVICE)

            sv_b = torch.linalg.svdvals(wb_g)[:100].cpu().numpy()
            sv_i = torch.linalg.svdvals(wi_g)[:100].cpu().numpy()

            del wb_g, wi_g
            torch.cuda.empty_cache()

            k = min(len(sv_b), len(sv_i))
            sv_b_n = sv_b[:k] / (sv_b[:k].sum() + 1e-12)
            sv_i_n = sv_i[:k] / (sv_i[:k].sum() + 1e-12)
            spec_diff = float(np.linalg.norm(sv_b_n - sv_i_n))
            pr_b = float((sv_b[:k].sum()**2) / ((sv_b[:k]**2).sum() + 1e-12))
            pr_i = float((sv_i[:k].sum()**2) / ((sv_i[:k]**2).sum() + 1e-12))

            svd_results.append({
                "key": key, "comp": comp, "layer": layer_idx,
                "rel_frob": rel,
                "spectral_diff": spec_diff,
                "pr_base": pr_b, "pr_instruct": pr_i,
                "pr_change_pct": (pr_i - pr_b) / pr_b * 100 if pr_b > 0 else 0,
                "top5_sv_base": sv_b[:5].tolist(),
                "top5_sv_instruct": sv_i[:5].tolist(),
            })

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(common)} params... ({time.time()-t0:.0f}s)")

    # Component summary
    print(f"\n  --- Component-Level Frobenius Change ---")
    comp_summary = {}
    for comp, vals in sorted(by_component.items(), key=lambda x: -np.mean(x[1])):
        arr = np.array(vals)
        comp_summary[comp] = {
            "mean": float(np.mean(arr)), "max": float(np.max(arr)),
            "n": len(vals),
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
            mx = float(np.max(vals))
            layer_summary[l] = {"mean": m, "max": mx}
            bar = "█" * int(m * 50)
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
    if svd_results:
        print(f"\n  --- SVD Spectrum Shift ({len(svd_results)} matrices) ---")
        svd_by_comp = defaultdict(list)
        for r in svd_results:
            svd_by_comp[r["comp"]].append(r)

        svd_summary = {}
        for comp, items in sorted(svd_by_comp.items(),
                                   key=lambda x: -np.mean([i["spectral_diff"] for i in x[1]])):
            diffs = [it["spectral_diff"] for it in items]
            pr_ch = [it["pr_change_pct"] for it in items]
            frobs = [it["rel_frob"] for it in items]
            svd_summary[comp] = {
                "mean_spectral_diff": float(np.mean(diffs)),
                "mean_pr_change_pct": float(np.mean(pr_ch)),
                "mean_rel_frob": float(np.mean(frobs)),
                "n": len(items),
            }
            print(f"  {comp:15s}  spec_diff={np.mean(diffs):.6f}  "
                  f"PR_shift={np.mean(pr_ch):+.2f}%  "
                  f"frob={np.mean(frobs):.6f}  (n={len(items)})")

        # Per-layer SVD PR shift for attention components
        print(f"\n  --- Per-Layer Weight-PR Shift (q/k/v/o_proj) ---")
        attn_comps = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for l in range(n_layers):
            items_l = [r for r in svd_results if r["layer"] == l and r["comp"] in attn_comps]
            if items_l:
                avg_pr_b = np.mean([it["pr_base"] for it in items_l])
                avg_pr_i = np.mean([it["pr_instruct"] for it in items_l])
                avg_ch = np.mean([it["pr_change_pct"] for it in items_l])
                bar = "+" * int(max(0, avg_ch) * 2) + "-" * int(max(0, -avg_ch) * 2)
                print(f"  L{l:3d}  base={avg_pr_b:6.2f}  inst={avg_pr_i:6.2f}  "
                      f"Δ={avg_ch:+6.2f}%  {bar}")

        print(f"\n  --- Top 10 Most Spectrally Changed ---")
        top10 = sorted(svd_results, key=lambda x: -x["spectral_diff"])[:10]
        for r in top10:
            print(f"  {r['key']}")
            print(f"    spec_diff={r['spectral_diff']:.6f}  "
                  f"PR: {r['pr_base']:.2f}→{r['pr_instruct']:.2f} "
                  f"({r['pr_change_pct']:+.1f}%)  frob={r['rel_frob']:.6f}")

    # Global
    total_base = sum(torch.norm(sd_base[k].float()).item()**2 for k in common)**0.5
    total_diff = sum(
        torch.norm((sd_inst[k].float() - sd_base[k].float())).item()**2
        for k in common if sd_base[k].shape == sd_inst[k].shape
    )**0.5
    global_rel = total_diff / total_base

    print(f"\n  === GLOBAL ===")
    print(f"  ||W_base||     = {total_base:.2f}")
    print(f"  ||ΔW||         = {total_diff:.4f}")
    print(f"  Global change  = {global_rel:.6f} ({global_rel*100:.4f}%)")
    print(f"  Total time: {time.time()-t0:.0f}s")

    out = RESULTS_DIR / "weight_svd_qwen25_7b_base_vs_instruct.json"
    with open(out, "w") as f:
        json.dump({
            "global_rel_change": global_rel,
            "component_summary": comp_summary,
            "layer_summary": {str(k): v for k, v in layer_summary.items()},
            "svd_summary": svd_summary if svd_results else {},
            "svd_per_matrix": svd_results,
        }, f, indent=2)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
