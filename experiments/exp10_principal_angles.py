#!/usr/bin/env python3
"""
Exp 10: Principal Angle Analysis — direct measurement of U/V rotation.
=======================================================================

Previous experiments (Exp 8, 9B) showed:
    - Weights drift 1-25% (Frobenius) across RL-aligned models
    - SVD spectrum (Σ) is preserved to < 0.1%

But these do NOT directly measure HOW MUCH the singular vectors (U, V)
rotated. Principal angles between top-k left-singular subspaces are the
canonical quantification of rotation magnitude.

This experiment:
    - Null baseline: Qwen3-1.7B base vs DPO-800 (expect ≈ 0° — no real training)
    - Qwen2.5-7B base vs Instruct (full RLHF)
    - Mistral-7B base vs Instruct (full RLHF)
    - Yi-1.5-6B base vs Chat (heavy alignment)

For each pair we compute principal angles on every interesting weight matrix
(q/k/v/o_proj + gate/up/down_proj + lm_head), top-k=32 singular vectors,
and report per-component / per-layer distributions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from spectral_flow_probe import RotationAnalyzer  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp10_principal_angles"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


PAIRS = [
    {
        "name": "Null_Qwen3_1.7B_DPO800",
        "desc": "NULL: Qwen3-1.7B base vs weak-DPO 800 steps",
        "base": "/cache/zhangjing/models/Qwen3-1.7B",
        "inst": "/cache/zhangjing/spectral-flow-probe/experiments/results/exp5b_fullft_1.7B/checkpoints/step_800",
    },
    {
        "name": "Qwen2.5_7B",
        "desc": "Qwen2.5-7B base vs Instruct (Alibaba, full RLHF)",
        "base": "/cache/zhangjing/models/Qwen2.5-7B",
        "inst": "/cache/zhangjing/models/Qwen2.5-7B-Instruct",
    },
    {
        "name": "Mistral_7B",
        "desc": "Mistral-7B base vs Instruct (Mistral AI, full alignment)",
        "base": "/cache/zhangjing/models/Mistral-7B-v0.1",
        "inst": "/cache/zhangjing/models/Mistral-7B-Instruct-v0.1",
    },
    {
        "name": "Yi_1.5_6B",
        "desc": "Yi-1.5-6B base vs Chat (01.AI, heavy alignment)",
        "base": "/cache/zhangjing/models/Yi-1.5-6B",
        "inst": "/cache/zhangjing/models/Yi-1.5-6B-Chat",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=str, default=None,
                    help="Name filter; if None, run all pairs matching argv[--index]")
    ap.add_argument("--index", type=int, default=None,
                    help="Run only the Nth pair (0-indexed)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=32)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [exp10] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("exp10")

    # Filter pairs
    pairs = list(PAIRS)
    if args.index is not None:
        pairs = [pairs[args.index]]
    elif args.pair is not None:
        pairs = [p for p in pairs if args.pair in p["name"]]

    ra = RotationAnalyzer(top_k_sv=100, top_k_angles=args.top_k)

    summary = []
    for p in pairs:
        log.info("=" * 70)
        log.info("  %s — %s", p["name"], p["desc"])
        log.info("=" * 70)

        try:
            report = ra.compare(p["base"], p["inst"], gpu_id=args.gpu, verbose=True)
        except Exception as e:
            log.error("  FAILED: %s", e)
            continue

        print("\n" + report.summary() + "\n")

        out = RESULTS_DIR / f"angles_{p['name']}.json"
        report.to_json(str(out))
        log.info("  Saved to %s", out)

        summary.append({
            "name": p["name"],
            "desc": p["desc"],
            "global_rel_change_pct": report.global_rel_change * 100,
            "mean_svd_pr_shift_pct": report.mean_svd_pr_shift_pct,
            "mean_angle_deg": report.mean_angle_deg,
            "elapsed_sec": report.elapsed_sec,
        })

    if summary:
        sum_path = RESULTS_DIR / "summary.json"
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info("Summary saved to %s", sum_path)

        print("\n" + "=" * 70)
        print("  Principal Angle Summary")
        print("=" * 70)
        print(f"  {'pair':<30s} {'ΔW%':>8s} {'ΣPR%':>8s} {'Angle°':>8s}")
        print(f"  {'-' * 60}")
        for s in summary:
            print(f"  {s['name']:<30s} {s['global_rel_change_pct']:8.3f} "
                  f"{s['mean_svd_pr_shift_pct']:8.3f} {s['mean_angle_deg']:8.3f}")


if __name__ == "__main__":
    main()
