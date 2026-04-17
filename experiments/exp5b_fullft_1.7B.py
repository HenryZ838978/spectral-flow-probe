#!/usr/bin/env python3
"""
Exp 5b: Full Fine-Tuning Causality — Qwen3-1.7B (8-bit Adam)
=============================================================
Same protocol as Exp 5 (0.6B) but on a larger model to confirm:
  1) PR collapse occurs at 1.7B scale under full FT
  2) Benchmark degradation correlates with PR drop

Memory strategy:
  - 8-bit Adam via bitsandbytes (halves optimizer memory)
  - Train model on GPU 0, ref model on GPU 1
  - With CUDA_VISIBLE_DEVICES=6,7: GPU 6 = train, GPU 7 = ref

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python experiments/exp5b_fullft_1.7B.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    setup_logging, load_dpo_data, SpectralProbeCallback,
    cleanup, save_result, HF_MIRROR, FALLBACK_CHAT_TEMPLATE,
)
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainerCallback, TrainerControl, TrainerState,
)
from trl import DPOTrainer, DPOConfig

log = setup_logging("exp5b_1.7B")

MODEL_PATH = "/cache/zhangjing/models/Qwen3-1.7B"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp5b_fullft_1.7B"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 800
CHECKPOINT_EVERY = 100
PROBE_EVERY = 25

BENCHMARK_TASKS = ["arc_easy", "hellaswag"]
BENCHMARK_BATCH_SIZE = 8


def load_model_full_ft(model_path: str, gpu_id: int = 0):
    log.info("Loading model for full FT: %s on GPU %d", model_path, gpu_id)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": gpu_id},
    )
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Full FT: %d / %d params trainable (%.1f%%)",
             trainable, total, 100.0 * trainable / total)
    return model, tok


class FullModelCheckpointCallback(TrainerCallback):
    def __init__(self, save_every: int, save_dir: Path, tokenizer):
        self.save_every = save_every
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.saved_steps: list[int] = []

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        if state.global_step % self.save_every != 0 or state.global_step == 0:
            return
        if model is None:
            return
        ckpt_path = self.save_dir / f"step_{state.global_step}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_path))
        self.tokenizer.save_pretrained(str(ckpt_path))
        self.saved_steps.append(state.global_step)
        log.info("Saved full checkpoint at step %d -> %s", state.global_step, ckpt_path)


def run_lm_eval(model, tokenizer, tasks: list[str],
                batch_size: int = 8) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    log.info("Running lm-eval on tasks: %s", tasks)
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        batch_size=batch_size,
        log_samples=False,
    )

    scores = {}
    if "results" in results:
        for task_name, task_res in results["results"].items():
            acc_key = "acc,none"
            acc_norm_key = "acc_norm,none"
            if acc_norm_key in task_res:
                scores[task_name] = task_res[acc_norm_key]
            elif acc_key in task_res:
                scores[task_name] = task_res[acc_key]
            else:
                for k, v in task_res.items():
                    if "acc" in k and isinstance(v, (int, float)):
                        scores[task_name] = v
                        break

    log.info("Benchmark scores: %s", scores)
    return scores


def measure_pr(model) -> float:
    cb = SpectralProbeCallback(every_n=1, n_samples=30, tag="eval_pr")
    state = TrainerState()
    state.global_step = 1
    control = TrainerControl()
    cb.on_step_end(None, state, control, model=model)
    if cb.history:
        return cb.history[0].get("pr_last", 0.0)
    return 0.0


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Exp 5b: Full FT Causality — Qwen3-1.7B (8-bit Adam)")
    log.info("Model: %s, Steps: %d, Checkpoint every: %d",
             MODEL_PATH, MAX_STEPS, CHECKPOINT_EVERY)
    log.info("Benchmarks: %s", BENCHMARK_TASKS)
    log.info("=" * 60)

    # ── Phase 0: Baseline ────────────────────────────────────────
    log.info("--- Phase 0: Baseline evaluation ---")
    model, tok = load_model_full_ft(MODEL_PATH, gpu_id=0)

    baseline_pr = measure_pr(model)
    log.info("Baseline PR: %.2f", baseline_pr)

    baseline_scores = run_lm_eval(model, tok, BENCHMARK_TASKS,
                                   batch_size=BENCHMARK_BATCH_SIZE)
    log.info("Baseline benchmark scores: %s", baseline_scores)

    # ── Phase 1: Full FT DPO (2-GPU, 8-bit Adam) ────────────────
    log.info("--- Phase 1: Full FT DPO Training (train=GPU0, ref=GPU1, optim=adamw_8bit) ---")
    train_ds, eval_ds = load_dpo_data(n_train=5000, n_eval=500)

    log.info("Loading ref model on GPU 1...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 1},
    )
    ref_model.eval()

    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_cb = FullModelCheckpointCallback(CHECKPOINT_EVERY, ckpt_dir, tok)
    probe_cb = SpectralProbeCallback(
        every_n=PROBE_EVERY, n_samples=30, tag="fullft_1.7B"
    )

    config = DPOConfig(
        output_dir=str(RESULTS_DIR / "training"),
        run_name="exp5b_fullft_1.7B",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-6,
        optim="adamw_8bit",
        beta=0.1,
        max_length=384,
        bf16=True,
        logging_steps=10,
        save_steps=9999,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=42,
    )

    trainer = DPOTrainer(
        model=model, ref_model=ref_model, args=config,
        train_dataset=train_ds,
        processing_class=tok, callbacks=[probe_cb, ckpt_cb],
    )
    trainer.train()

    pr_by_step = {0: baseline_pr}
    for h in probe_cb.history:
        pr_by_step[h["step"]] = h.get("pr_last", 0)

    del trainer, model, ref_model
    cleanup()

    # ── Phase 2: Checkpoint benchmark evaluation ─────────────────
    log.info("--- Phase 2: Checkpoint benchmark evaluation ---")
    all_checkpoints = [{"step": 0, "pr": baseline_pr, "benchmarks": baseline_scores}]

    for step in ckpt_cb.saved_steps:
        ckpt_path = ckpt_dir / f"step_{step}"
        log.info("Evaluating checkpoint step %d from %s...", step, ckpt_path)

        ckpt_model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        ckpt_tok = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True
        )
        if ckpt_tok.pad_token is None:
            ckpt_tok.pad_token = ckpt_tok.eos_token

        ckpt_pr = measure_pr(ckpt_model)
        log.info("[step=%d] PR = %.2f", step, ckpt_pr)

        bench_scores = run_lm_eval(ckpt_model, ckpt_tok, BENCHMARK_TASKS,
                                    batch_size=BENCHMARK_BATCH_SIZE)

        all_checkpoints.append({
            "step": step, "pr": ckpt_pr, "benchmarks": bench_scores,
        })
        log.info("[step=%d] PR=%.2f  Benchmarks: %s", step, ckpt_pr, bench_scores)

        del ckpt_model, ckpt_tok
        cleanup()

    final_step = MAX_STEPS
    if final_step not in [c["step"] for c in all_checkpoints]:
        final_ckpt = ckpt_dir / f"step_{final_step}"
        if final_ckpt.exists():
            log.info("Evaluating final model at step %d...", final_step)
            final_model = AutoModelForCausalLM.from_pretrained(
                str(final_ckpt), trust_remote_code=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                device_map={"": 0},
            )
            final_tok = AutoTokenizer.from_pretrained(
                str(final_ckpt), trust_remote_code=True
            )
            if final_tok.pad_token is None:
                final_tok.pad_token = final_tok.eos_token
            final_pr = measure_pr(final_model)
            final_bench = run_lm_eval(final_model, final_tok, BENCHMARK_TASKS,
                                       batch_size=BENCHMARK_BATCH_SIZE)
            all_checkpoints.append({
                "step": final_step, "pr": final_pr, "benchmarks": final_bench,
            })
            del final_model, final_tok
            cleanup()

    # ── Phase 3: Correlation analysis ────────────────────────────
    log.info("--- Phase 3: Correlation analysis ---")
    from scipy import stats

    correlations = {}
    for task in BENCHMARK_TASKS:
        task_prs = []
        task_scores = []
        for c in all_checkpoints:
            if task in c.get("benchmarks", {}):
                task_prs.append(c["pr"])
                task_scores.append(c["benchmarks"][task])

        if len(task_prs) >= 3:
            rho, p = stats.spearmanr(task_prs, task_scores)
            correlations[f"pr_vs_{task}"] = {
                "rho": float(rho), "pvalue": float(p),
                "n_points": len(task_prs),
            }
            log.info("  PR vs %s: rho=%.3f, p=%.4f (n=%d)",
                     task, rho, p, len(task_prs))

    # ── Save results ─────────────────────────────────────────────
    result = {
        "model": MODEL_PATH,
        "method": "Full FT DPO (8-bit Adam)",
        "max_steps": MAX_STEPS,
        "baseline_pr": baseline_pr,
        "baseline_benchmarks": baseline_scores,
        "checkpoints": all_checkpoints,
        "pr_history": probe_cb.history,
        "correlations": correlations,
        "train_eval": {},
        "elapsed_min": (time.time() - t0) / 60,
    }
    save_result(result, RESULTS_DIR / "fullft_1.7B_results.json")

    # ── Phase 4: Plot ────────────────────────────────────────────
    try:
        plot_results(all_checkpoints, correlations, probe_cb.history, baseline_pr)
    except Exception as e:
        log.error("Plot failed: %s", e, exc_info=True)

    elapsed = (time.time() - t0) / 60
    log.info("=== Exp 5b DONE in %.1f min ===", elapsed)


def plot_results(checkpoints, correlations, pr_history, baseline_pr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tasks = len(BENCHMARK_TASKS)
    fig, axes = plt.subplots(2, max(n_tasks, 2), figsize=(8 * max(n_tasks, 2), 10))
    fig.suptitle(
        "Exp 5b: Full FT Causality — PR Collapse → Benchmark Degradation\n"
        f"Model: Qwen3-1.7B | Method: Full FT DPO (8-bit Adam) | Steps: {MAX_STEPS}",
        fontsize=14, fontweight="bold"
    )

    steps = [c["step"] for c in checkpoints]
    prs = [c["pr"] for c in checkpoints]

    ax = axes[0, 0]
    if pr_history:
        hsteps = [h["step"] for h in pr_history]
        hprs = [h.get("pr_last", 0) for h in pr_history]
        ax.plot(hsteps, hprs, "o-", color="#ef4444", markersize=3,
                linewidth=1.5, label="PR (probe, every 25 steps)")
    ax.axhline(y=baseline_pr, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline PR={baseline_pr:.1f}")
    ax.scatter(steps, prs, c="#1d4ed8", s=80, zorder=5,
               edgecolors="black", label="PR at checkpoint")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("PR(last)")
    ax.set_title("PR Trajectory (Full FT, 1.7B)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    colors = ["#22c55e", "#3b82f6", "#f59e0b", "#8b5cf6"]
    for i, task in enumerate(BENCHMARK_TASKS):
        task_steps = []
        task_scores = []
        for c in checkpoints:
            if task in c.get("benchmarks", {}):
                task_steps.append(c["step"])
                task_scores.append(c["benchmarks"][task])
        if task_scores:
            ax.plot(task_steps, task_scores, "o-", color=colors[i % len(colors)],
                    markersize=5, linewidth=2, label=task)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Benchmark Scores Over Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for i, task in enumerate(BENCHMARK_TASKS):
        ax = axes[1, i]
        task_prs = []
        task_scores = []
        task_steps = []
        for c in checkpoints:
            if task in c.get("benchmarks", {}):
                task_prs.append(c["pr"])
                task_scores.append(c["benchmarks"][task])
                task_steps.append(c["step"])

        if task_scores:
            scatter = ax.scatter(task_prs, task_scores, c=task_steps,
                                 cmap="RdYlGn_r", s=80, edgecolors="black",
                                 zorder=5)
            for j, step in enumerate(task_steps):
                ax.annotate(f"s{step}", (task_prs[j], task_scores[j]),
                            fontsize=7, alpha=0.7,
                            xytext=(5, 5), textcoords="offset points")
            plt.colorbar(scatter, ax=ax, label="Step")

        corr_key = f"pr_vs_{task}"
        if corr_key in correlations:
            rho = correlations[corr_key]["rho"]
            p = correlations[corr_key]["pvalue"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.set_title(f"PR vs {task}\nrho={rho:.3f}, p={p:.4f} {sig}")
        else:
            ax.set_title(f"PR vs {task}")

        ax.set_xlabel("PR(last)")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "fig_fullft_1.7B_causality.png"
    fig.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", fig_path)


if __name__ == "__main__":
    main()
