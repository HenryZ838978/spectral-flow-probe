#!/usr/bin/env python3
"""
Exp 2: Universality Matrix — PR Collapse Across Models & RL Methods
====================================================================
3 models x 3 methods (DPO, KTO, GRPO), monitoring PR(last) every 25 steps.

Runs are dispatched across GPUs in parallel waves via subprocess.

Usage:
    python experiments/exp2_universality.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "exp2_universality"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

MODELS = {
    "Qwen3-0.6B":   "/cache/zhangjing/models/Qwen3-0.6B",
    "Qwen3-4B":     "/cache/zhangjing/models/Qwen3-4B",
    "Mistral-7B":   "/cache/zhangjing/models/Mistral-7B-v0.1",
}

METHODS = ["DPO", "KTO", "GRPO"]

# GRPO is expensive (generates completions); only run on the smallest model
GRPO_MODELS = {"Qwen3-0.6B"}

FREE_GPUS = [3, 4, 5, 6, 7]

MAX_STEPS = 200
PROBE_EVERY = 25


def build_job_list() -> list[dict]:
    jobs = []
    for model_name, model_path in MODELS.items():
        for method in METHODS:
            if method == "GRPO" and model_name not in GRPO_MODELS:
                continue
            jobs.append({
                "model_name": model_name,
                "model_path": model_path,
                "method": method,
                "tag": f"{model_name}_{method}",
            })
    return jobs


def run_single_job(job: dict, gpu_id: int):
    """Launch a single training job as a subprocess on the given GPU."""
    tag = job["tag"]
    log_path = RESULTS_DIR / f"{tag}.log"
    cmd = [
        PYTHON, str(SCRIPT_DIR / "_exp2_worker.py"),
        "--model_path", job["model_path"],
        "--model_name", job["model_name"],
        "--method", job["method"],
        "--gpu_id", str(gpu_id),
        "--max_steps", str(MAX_STEPS),
        "--probe_every", str(PROBE_EVERY),
        "--output_dir", str(RESULTS_DIR),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[DISPATCH] {tag} -> GPU {gpu_id}  (log: {log_path})")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
        )
    return proc, tag


def run_wave(jobs: list[dict], gpus: list[int]):
    """Run a wave of jobs in parallel, one per GPU."""
    procs = []
    for i, job in enumerate(jobs):
        gpu = gpus[i % len(gpus)]
        proc, tag = run_single_job(job, gpu)
        procs.append((proc, tag))

    for proc, tag in procs:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"[DONE] {tag}: {status}")


def collect_results():
    """Gather all per-run JSONs into a summary."""
    summary = {}
    for jf in sorted(RESULTS_DIR.glob("*_result.json")):
        with open(jf) as f:
            data = json.load(f)
        tag = jf.stem.replace("_result", "")
        summary[tag] = data

    summary_path = RESULTS_DIR / "SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUMMARY] {len(summary)} results -> {summary_path}")
    return summary


def plot_universality(summary: dict):
    """Generate grid figure of PR trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    fig.suptitle("Exp 2: PR Collapse Universality — Models x RL Methods",
                 fontsize=14, fontweight="bold")

    method_colors = {"DPO": "#ef4444", "KTO": "#3b82f6", "GRPO": "#22c55e"}
    model_list = list(MODELS.keys())

    for ax_idx, model_name in enumerate(model_list):
        ax = axes[ax_idx]
        ax.set_title(model_name, fontsize=12, fontweight="bold")

        for method in METHODS:
            tag = f"{model_name}_{method}"
            if tag not in summary:
                continue
            data = summary[tag]
            hist = data.get("pr_history", [])
            if not hist:
                continue
            steps = [h["step"] for h in hist]
            prs = [h["pr_last"] for h in hist]
            baseline = data.get("baseline_pr")
            ax.plot(steps, prs, "o-", label=method,
                    color=method_colors.get(method, "gray"),
                    markersize=4, linewidth=2)
            if baseline:
                ax.axhline(baseline, ls="--", color="gray", alpha=0.4)

        ax.set_xlabel("Training Step")
        if ax_idx == 0:
            ax.set_ylabel("PR(last)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "fig_universality_matrix.png"
    fig.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {fig_path}")


def main():
    t0 = time.time()
    jobs = build_job_list()
    print(f"=== Exp 2: Universality Matrix ({len(jobs)} runs) ===")
    for j in jobs:
        print(f"  {j['tag']}")

    # Wave 1: up to 5 jobs on GPUs 3-7
    wave1 = jobs[:len(FREE_GPUS)]
    wave2 = jobs[len(FREE_GPUS):]

    print(f"\n--- Wave 1: {len(wave1)} jobs ---")
    run_wave(wave1, FREE_GPUS)

    if wave2:
        print(f"\n--- Wave 2: {len(wave2)} jobs ---")
        run_wave(wave2, FREE_GPUS)

    elapsed = (time.time() - t0) / 60
    print(f"\nAll runs complete in {elapsed:.1f} min")

    summary = collect_results()
    if summary:
        try:
            plot_universality(summary)
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")

    print("=== Exp 2 DONE ===")


if __name__ == "__main__":
    main()
