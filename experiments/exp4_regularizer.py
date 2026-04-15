#!/usr/bin/env python3
"""
Exp 4: Regularizer Sweep — Proving spectral_pr_loss Prevents PR Collapse
=========================================================================
Qwen3-1.7B + DPO with 5 lambda values: [0.0, 0.005, 0.01, 0.05, 0.1]
500 steps each, parallel across GPUs.

Improvements over the first attempt:
  - Wider lambda range (0.0 to 0.1)
  - Gradient flow verification (log whether pr_loss has grad)
  - mode="target" (bidirectional) for lambda >= 0.05
  - Track DPO eval metrics alongside PR

Usage:
    python experiments/exp4_regularizer.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "exp4_regularizer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

LAMBDA_SWEEP = [0.0, 0.005, 0.01, 0.05, 0.1]
FREE_GPUS = [3, 4, 5, 6, 7]


def main():
    t0 = time.time()
    print(f"=== Exp 4: Regularizer Sweep ({len(LAMBDA_SWEEP)} lambdas) ===")

    procs = []
    for i, lam in enumerate(LAMBDA_SWEEP):
        gpu = FREE_GPUS[i % len(FREE_GPUS)]
        tag = f"lambda_{lam:.3f}"
        log_path = RESULTS_DIR / f"{tag}.log"
        mode = "target" if lam >= 0.05 else "floor"

        cmd = [
            PYTHON, str(SCRIPT_DIR / "_exp4_worker.py"),
            "--pr_lambda", str(lam),
            "--pr_mode", mode,
            "--gpu_id", str(gpu),
            "--output_dir", str(RESULTS_DIR),
            "--tag", tag,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        print(f"[DISPATCH] {tag} (mode={mode}) -> GPU {gpu}")
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        procs.append((proc, tag))

    for proc, tag in procs:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"[DONE] {tag}: {status}")

    elapsed = (time.time() - t0) / 60
    print(f"\nAll runs complete in {elapsed:.1f} min")

    summary = collect_and_plot()
    print("=== Exp 4 DONE ===")


def collect_and_plot():
    summary = {}
    for jf in sorted(RESULTS_DIR.glob("lambda_*_result.json")):
        with open(jf) as f:
            data = json.load(f)
        tag = jf.stem.replace("_result", "")
        summary[tag] = data

    summary_path = RESULTS_DIR / "SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUMMARY] {len(summary)} results -> {summary_path}")

    if summary:
        try:
            plot_regularizer_sweep(summary)
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")

    return summary


def plot_regularizer_sweep(summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Exp 4: Regularizer Sweep — spectral_pr_loss Effectiveness",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(summary)))

    # Sort by lambda value
    sorted_items = sorted(summary.items(),
                          key=lambda x: x[1].get("pr_lambda", 0))

    # Panel 1: PR trajectory per lambda
    ax = axes[0]
    for (tag, data), color in zip(sorted_items, colors):
        lam = data.get("pr_lambda", 0)
        hist = data.get("pr_history", [])
        if not hist:
            continue
        steps = [h["step"] for h in hist]
        prs = [h.get("pr_last", 0) for h in hist]
        ax.plot(steps, prs, "o-", label=f"λ={lam:.3f}", color=color,
                markersize=3, linewidth=2)

    baseline = sorted_items[0][1].get("baseline_pr") if sorted_items else None
    if baseline:
        ax.axhline(baseline, ls="--", color="gray", alpha=0.5, label="Baseline")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("PR(last)")
    ax.set_title("PR During Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: pr_loss decomposition for highest lambda
    ax = axes[1]
    for (tag, data), color in zip(sorted_items, colors):
        lam = data.get("pr_lambda", 0)
        pl_hist = data.get("pr_loss_history", [])
        if not pl_hist or lam == 0:
            continue
        steps = [h["step"] for h in pl_hist]
        pr_losses = [h.get("pr_loss", 0) for h in pl_hist]
        ax.plot(steps, pr_losses, "s-", label=f"λ={lam:.3f}", color=color,
                markersize=3, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("pr_loss (before λ scaling)")
    ax.set_title("Spectral PR Loss During Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Final PR + DPO eval loss bar chart
    ax = axes[2]
    lambdas = []
    final_prs = []
    eval_losses = []
    for tag, data in sorted_items:
        lam = data.get("pr_lambda", 0)
        hist = data.get("pr_history", [])
        ev = data.get("eval", {})
        final_pr = hist[-1].get("pr_last", 0) if hist else 0
        eval_loss = ev.get("eval_loss", 0)
        lambdas.append(f"λ={lam:.3f}")
        final_prs.append(final_pr)
        eval_losses.append(eval_loss)

    x = np.arange(len(lambdas))
    w = 0.35
    ax.bar(x - w/2, final_prs, w, label="Final PR", color="#3b82f6")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, eval_losses, w, label="Eval Loss", color="#ef4444", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(lambdas, fontsize=8)
    ax.set_ylabel("PR(last)", color="#3b82f6")
    ax2.set_ylabel("Eval Loss", color="#ef4444")
    ax.set_title("Final PR vs Eval Loss by Lambda")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "fig_regularizer_sweep.png"
    fig.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {fig_path}")


if __name__ == "__main__":
    main()
