#!/usr/bin/env python3
"""
Exp 3: Causality Bridge — PR Collapse -> Downstream Quality Degradation
========================================================================
Train Qwen3-1.7B with DPO, saving checkpoints every 50 steps.
At each checkpoint, evaluate generation quality metrics and correlate with PR.

Metrics per checkpoint:
  - PR(last) via SpectralProbeCallback
  - Distinct-1, Distinct-2 (generation diversity)
  - Repetition rate (fraction of responses with repeated n-grams)
  - Mean response length
  - Type-token ratio (vocabulary richness)
  - Perplexity on held-out prompts

Usage:
    CUDA_VISIBLE_DEVICES=7 python experiments/exp3_causality.py
"""
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    setup_logging, load_base_model, apply_lora, load_dpo_data,
    make_dpo_config, SpectralProbeCallback, cleanup, save_result,
    HF_MIRROR, DATASET_NAME,
)
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback, TrainerControl, TrainerState

log = setup_logging("exp3_causality")

MODEL_PATH = "/cache/zhangjing/models/Qwen3-1.7B"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp3_causality"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 500
CHECKPOINT_EVERY = 50
PROBE_EVERY = 25
N_EVAL_PROMPTS = 50
N_PPL_SAMPLES = 200


# ── Checkpoint-saving callback ──────────────────────────────────

class CheckpointSaveCallback(TrainerCallback):
    """Save LoRA adapter at regular intervals for later evaluation."""

    def __init__(self, save_every: int, save_dir: Path):
        self.save_every = save_every
        self.save_dir = save_dir
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
        self.saved_steps.append(state.global_step)
        log.info("Saved checkpoint at step %d -> %s", state.global_step, ckpt_path)


# ── Generation quality metrics ──────────────────────────────────

def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-N: ratio of unique n-grams to total n-grams across all texts."""
    total_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        total_ngrams.extend(ngrams)
    if not total_ngrams:
        return 0.0
    return len(set(total_ngrams)) / len(total_ngrams)


def repetition_rate(texts: list[str], min_repeat: int = 3, ngram_size: int = 3) -> float:
    """Fraction of texts containing a repeated n-gram >= min_repeat times."""
    count = 0
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+ngram_size])
                  for i in range(len(tokens) - ngram_size + 1)]
        freq = Counter(ngrams)
        if any(v >= min_repeat for v in freq.values()):
            count += 1
    return count / max(len(texts), 1)


def type_token_ratio(texts: list[str]) -> float:
    """Average type-token ratio across texts."""
    ratios = []
    for text in texts:
        tokens = text.split()
        if tokens:
            ratios.append(len(set(tokens)) / len(tokens))
    return float(np.mean(ratios)) if ratios else 0.0


def mean_length(texts: list[str]) -> float:
    lengths = [len(t.split()) for t in texts]
    return float(np.mean(lengths)) if lengths else 0.0


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: list[str],
                       max_length: int = 256) -> float:
    """Average perplexity of the model on a list of texts."""
    device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).to(device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2:
            continue
        outputs = model(input_ids=ids, labels=ids)
        nll = outputs.loss.item() * (ids.shape[1] - 1)
        total_nll += nll
        total_tokens += ids.shape[1] - 1

    if total_tokens == 0:
        return float("inf")
    avg_nll = total_nll / total_tokens
    return math.exp(min(avg_nll, 100))


@torch.no_grad()
def generate_responses(model, tokenizer, prompts: list[str],
                       max_new_tokens: int = 128) -> list[str]:
    """Generate one response per prompt."""
    device = next(model.parameters()).device
    model.eval()
    responses = []

    for prompt in prompts:
        try:
            msgs = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            text = prompt

        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=256).to(device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)

    return responses


def evaluate_checkpoint(model, tokenizer, eval_prompts: list[str],
                        ppl_texts: list[str]) -> dict:
    """Run all quality metrics on a model checkpoint."""
    responses = generate_responses(model, tokenizer, eval_prompts)

    d1 = distinct_n(responses, 1)
    d2 = distinct_n(responses, 2)
    rep = repetition_rate(responses)
    ttr = type_token_ratio(responses)
    avg_len = mean_length(responses)
    ppl = compute_perplexity(model, tokenizer, ppl_texts)

    return {
        "distinct_1": d1,
        "distinct_2": d2,
        "repetition_rate": rep,
        "type_token_ratio": ttr,
        "mean_length": avg_len,
        "perplexity": ppl,
        "n_responses": len(responses),
        "sample_responses": responses[:3],
    }


# ── Main ────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Exp 3: Causality Bridge")
    log.info("Model: %s, Steps: %d, Checkpoint every: %d",
             MODEL_PATH, MAX_STEPS, CHECKPOINT_EVERY)
    log.info("=" * 60)

    # Load eval data
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    raw_ds = load_dataset(DATASET_NAME, split="train")
    raw_ds = raw_ds.shuffle(seed=99)
    eval_prompts = [raw_ds[i]["prompt"] for i in range(N_EVAL_PROMPTS)]
    ppl_texts = []
    for i in range(N_EVAL_PROMPTS, N_EVAL_PROMPTS + N_PPL_SAMPLES):
        chosen = raw_ds[i]["chosen"]
        if isinstance(chosen, list) and chosen:
            ppl_texts.append(chosen[-1]["content"])
        else:
            ppl_texts.append(str(chosen))

    # Phase 1: Train with checkpoint saves + PR monitoring
    log.info("--- Phase 1: DPO Training with checkpoints ---")
    model, tok = load_base_model(MODEL_PATH, gpu_id=0)

    # Evaluate baseline (step 0)
    log.info("Evaluating baseline (step 0)...")
    baseline_metrics = evaluate_checkpoint(model, tok, eval_prompts, ppl_texts)
    baseline_cb = SpectralProbeCallback(every_n=1, n_samples=30, tag="baseline")
    state0 = TrainerState()
    state0.global_step = 1
    baseline_cb.on_step_end(None, state0, TrainerControl(), model=model)
    baseline_pr = baseline_cb.history[0].get("pr_last", 0) if baseline_cb.history else 0
    log.info("Baseline PR: %.2f", baseline_pr)

    model = apply_lora(model)
    train_ds, eval_ds = load_dpo_data(n_train=5000, n_eval=500)

    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_cb = CheckpointSaveCallback(CHECKPOINT_EVERY, ckpt_dir)
    probe_cb = SpectralProbeCallback(
        every_n=PROBE_EVERY, n_samples=30, tag="causality"
    )

    config = make_dpo_config(
        "exp3_causality", str(RESULTS_DIR / "training"),
        max_steps=MAX_STEPS, batch_size=1, grad_accum=8,
    )
    trainer = DPOTrainer(
        model=model, args=config,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tok, callbacks=[probe_cb, ckpt_cb],
    )
    trainer.train()
    train_eval = trainer.evaluate()

    # Build PR lookup from probe history
    pr_by_step = {0: baseline_pr}
    for h in probe_cb.history:
        pr_by_step[h["step"]] = h.get("pr_last", 0)

    del trainer
    cleanup()

    # Phase 2: Evaluate each checkpoint
    log.info("--- Phase 2: Evaluating checkpoints ---")
    all_checkpoints = []

    step_0_entry = {
        "step": 0, "pr": baseline_pr,
        **baseline_metrics,
    }
    del step_0_entry["sample_responses"]
    all_checkpoints.append(step_0_entry)
    log.info("[step=0] PR=%.2f  D1=%.3f  D2=%.3f  Rep=%.3f  PPL=%.2f",
             baseline_pr, baseline_metrics["distinct_1"],
             baseline_metrics["distinct_2"], baseline_metrics["repetition_rate"],
             baseline_metrics["perplexity"])

    for step in ckpt_cb.saved_steps:
        ckpt_path = ckpt_dir / f"step_{step}"
        log.info("Evaluating checkpoint step %d...", step)

        base_model, tok2 = load_base_model(MODEL_PATH, gpu_id=0)
        ckpt_model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        ckpt_model.eval()

        metrics = evaluate_checkpoint(ckpt_model, tok2, eval_prompts, ppl_texts)

        closest_pr_step = min(pr_by_step.keys(), key=lambda s: abs(s - step))
        pr_val = pr_by_step.get(step, pr_by_step[closest_pr_step])

        entry = {"step": step, "pr": pr_val, **metrics}
        del entry["sample_responses"]
        all_checkpoints.append(entry)

        log.info("[step=%d] PR=%.2f  D1=%.3f  D2=%.3f  Rep=%.3f  PPL=%.2f",
                 step, pr_val, metrics["distinct_1"], metrics["distinct_2"],
                 metrics["repetition_rate"], metrics["perplexity"])

        del ckpt_model, base_model
        cleanup()

    # Phase 3: Correlation analysis
    log.info("--- Phase 3: Correlation analysis ---")
    from scipy import stats

    prs = [c["pr"] for c in all_checkpoints]
    d2s = [c["distinct_2"] for c in all_checkpoints]
    reps = [c["repetition_rate"] for c in all_checkpoints]
    ppls = [c["perplexity"] for c in all_checkpoints]

    corr_pr_d2 = stats.spearmanr(prs, d2s) if len(prs) >= 3 else (0, 1)
    corr_pr_rep = stats.spearmanr(prs, reps) if len(prs) >= 3 else (0, 1)
    corr_pr_ppl = stats.spearmanr(prs, ppls) if len(prs) >= 3 else (0, 1)

    correlations = {
        "pr_vs_distinct2": {"rho": float(corr_pr_d2[0]), "pvalue": float(corr_pr_d2[1])},
        "pr_vs_repetition": {"rho": float(corr_pr_rep[0]), "pvalue": float(corr_pr_rep[1])},
        "pr_vs_perplexity": {"rho": float(corr_pr_ppl[0]), "pvalue": float(corr_pr_ppl[1])},
    }
    log.info("Correlations:")
    for k, v in correlations.items():
        log.info("  %s: rho=%.3f, p=%.4f", k, v["rho"], v["pvalue"])

    # Save results
    result = {
        "model": MODEL_PATH,
        "max_steps": MAX_STEPS,
        "baseline_pr": baseline_pr,
        "checkpoints": all_checkpoints,
        "pr_history": probe_cb.history,
        "correlations": correlations,
        "train_eval": train_eval,
        "elapsed_min": (time.time() - t0) / 60,
    }
    save_result(result, RESULTS_DIR / "causality_results.json")

    # Phase 4: Plot
    try:
        plot_causality(all_checkpoints, correlations, probe_cb.history)
    except Exception as e:
        log.error("Plot failed: %s", e, exc_info=True)

    elapsed = (time.time() - t0) / 60
    log.info("=== Exp 3 DONE in %.1f min ===", elapsed)


def plot_causality(checkpoints: list[dict], correlations: dict,
                   pr_history: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Exp 3: PR Collapse → Generation Quality Degradation",
                 fontsize=14, fontweight="bold")

    steps = [c["step"] for c in checkpoints]
    prs = [c["pr"] for c in checkpoints]
    d1s = [c["distinct_1"] for c in checkpoints]
    d2s = [c["distinct_2"] for c in checkpoints]
    reps = [c["repetition_rate"] for c in checkpoints]
    ppls = [c["perplexity"] for c in checkpoints]
    ttrs = [c["type_token_ratio"] for c in checkpoints]

    # Top-left: PR trajectory (from probe, denser)
    ax = axes[0, 0]
    if pr_history:
        hsteps = [h["step"] for h in pr_history]
        hprs = [h.get("pr_last", 0) for h in pr_history]
        ax.plot(hsteps, hprs, "o-", color="#ef4444", markersize=3, linewidth=1.5)
    ax.scatter(steps, prs, c="#1d4ed8", s=60, zorder=5, edgecolors="black")
    ax.set_xlabel("Step")
    ax.set_ylabel("PR(last)")
    ax.set_title("PR Trajectory During Training")
    ax.grid(True, alpha=0.3)

    # Top-center: PR vs Distinct-2 scatter
    ax = axes[0, 1]
    ax.scatter(prs, d2s, c="#22c55e", s=60, edgecolors="black")
    for i, step in enumerate(steps):
        ax.annotate(f"s{step}", (prs[i], d2s[i]), fontsize=7, alpha=0.7)
    rho = correlations["pr_vs_distinct2"]["rho"]
    p = correlations["pr_vs_distinct2"]["pvalue"]
    ax.set_title(f"PR vs Distinct-2 (rho={rho:.3f}, p={p:.4f})")
    ax.set_xlabel("PR(last)")
    ax.set_ylabel("Distinct-2")
    ax.grid(True, alpha=0.3)

    # Top-right: PR vs Repetition scatter
    ax = axes[0, 2]
    ax.scatter(prs, reps, c="#f59e0b", s=60, edgecolors="black")
    for i, step in enumerate(steps):
        ax.annotate(f"s{step}", (prs[i], reps[i]), fontsize=7, alpha=0.7)
    rho = correlations["pr_vs_repetition"]["rho"]
    p = correlations["pr_vs_repetition"]["pvalue"]
    ax.set_title(f"PR vs Repetition Rate (rho={rho:.3f}, p={p:.4f})")
    ax.set_xlabel("PR(last)")
    ax.set_ylabel("Repetition Rate")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Distinct-1/2 over steps
    ax = axes[1, 0]
    ax.plot(steps, d1s, "o-", label="Distinct-1", color="#3b82f6", markersize=4)
    ax.plot(steps, d2s, "s-", label="Distinct-2", color="#22c55e", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distinct-N")
    ax.set_title("Generation Diversity Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-center: Perplexity over steps
    ax = axes[1, 1]
    ax.plot(steps, ppls, "o-", color="#8b5cf6", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity Over Training")
    ax.grid(True, alpha=0.3)

    # Bottom-right: Type-token ratio over steps
    ax = axes[1, 2]
    ax.plot(steps, ttrs, "o-", color="#ec4899", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Type-Token Ratio")
    ax.set_title("Vocabulary Richness Over Training")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "fig_causality_bridge.png"
    fig.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved causality figure: %s", fig_path)


if __name__ == "__main__":
    main()
