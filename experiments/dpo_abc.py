#!/usr/bin/env python3
"""
SFP DPO A/B/C Experiment — Overnight Automated Script
=====================================================
Three DPO training conditions on Qwen3-1.7B:
  A: vanilla DPO                    → observe PR collapse
  B: DPO + SpectralMonitor early-stop → PR preserved by halting
  C: DPO + spectral_pr_loss          → PR preserved + reward kept?

Usage:
    nohup python experiments/dpo_abc.py > experiments/results/dpo_abc.log 2>&1 &
"""
import gc
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

# ── SFP imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spectral_flow_probe.core import run_pca_layer
from spectral_flow_probe._compat import find_decoder_layers
from spectral_flow_probe.regularizer import spectral_pr_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dpo_abc")

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
MODEL_PATH = "/cache/zhangjing/models/Qwen3-1.7B"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
DATASET_SPLIT = "train"
N_SAMPLES = 5000
EVAL_SAMPLES = 500
HF_MIRROR = "https://hf-mirror.com"

MAX_STEPS = 500
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 5e-7
BETA = 0.1
MAX_LENGTH = 384

LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "v_proj"]

BASELINE_PR = 9.48
PR_TARGET = 7.0            # floor for regularizer (some slack below baseline)
PR_HALT = 4.0              # early-stop threshold for Run B
PR_LAMBDA = 0.01           # weight for spectral_pr_loss in Run C
PROBE_EVERY = 25           # measure PR every N steps
N_PROBE_SAMPLES = 30       # random inputs for PR measurement

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════
# SPECTRAL PROBE CALLBACK
# ════════════════════════════════════════════════════════════
class SpectralProbeCallback(TrainerCallback):
    """Measure PR(last) + S(last) every N steps. Optionally early-stop."""

    def __init__(self, every_n: int = 25, n_samples: int = 30,
                 pr_halt: float | None = None, tag: str = ""):
        self.every_n = every_n
        self.n_samples = n_samples
        self.pr_halt = pr_halt
        self.tag = tag
        self.history: list[dict] = []
        self._layers = None
        self._n_layers = 0

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        if state.global_step % self.every_n != 0 or state.global_step == 0:
            return
        if model is None:
            return
        metrics = self._measure(model)
        metrics["step"] = state.global_step
        metrics["epoch"] = state.epoch
        self.history.append(metrics)
        pr = metrics.get("pr_last", 999)
        s = metrics.get("s_last", 0)
        log.info("[%s] step=%d  PR(last)=%.2f  S(last)=%.4f",
                 self.tag, state.global_step, pr, s)

        if self.pr_halt is not None and pr < self.pr_halt:
            log.warning("[%s] PR(last)=%.2f < halt=%.2f → EARLY STOP",
                        self.tag, pr, self.pr_halt)
            control.should_training_stop = True

    def _setup(self, model):
        try:
            _, layers, n_layers, _ = find_decoder_layers(model)
        except RuntimeError:
            base = getattr(model, "base_model", model)
            base = getattr(base, "model", base)
            _, layers, n_layers, _ = find_decoder_layers(base)
        self._layers = layers
        self._n_layers = n_layers

    @torch.no_grad()
    def _measure(self, model) -> dict:
        if self._layers is None:
            self._setup(model)
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        last_layer = self._layers[-1]
        captures = []

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captures.append(h[:, -1, :].detach().float().cpu().numpy())

        handle = last_layer.register_forward_hook(hook_fn)
        try:
            for _ in range(self.n_samples):
                ids = torch.randint(100, 30000, (1, 64), device=device)
                model(input_ids=ids)
        finally:
            handle.remove()

        if was_training:
            model.train()

        if len(captures) < 5:
            return {}
        mat = np.vstack(captures)
        ls = run_pca_layer(mat)
        if ls is None:
            return {}
        return {"pr_last": ls.pr, "s_last": ls.S, "r2": ls.r2}


# ════════════════════════════════════════════════════════════
# SPECTRAL DPO TRAINER (Run C)
# ════════════════════════════════════════════════════════════
class SpectralDPOTrainer(DPOTrainer):
    """DPOTrainer + spectral_pr_loss on last-layer hidden states."""

    def __init__(self, *args, pr_lambda: float = 0.01,
                 pr_target: float = 7.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pr_lambda = pr_lambda
        self.pr_target = pr_target
        self._last_layer = None
        self._captured_hidden = None
        self._hook_handle = None
        self.pr_loss_history: list[dict] = []

    def _ensure_hook(self):
        if self._hook_handle is not None:
            return
        try:
            _, layers, n_layers, _ = find_decoder_layers(self.model)
        except RuntimeError:
            base = getattr(self.model, "base_model", self.model)
            base = getattr(base, "model", base)
            _, layers, n_layers, _ = find_decoder_layers(base)
        self._last_layer = layers[-1]

        def capture_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._captured_hidden = h[:, -1, :]  # (B, d) last token

        self._hook_handle = self._last_layer.register_forward_hook(capture_hook)

    def _compute_loss(self, model, inputs, return_outputs=False):
        self._ensure_hook()
        self._captured_hidden = None

        result = super()._compute_loss(model, inputs, return_outputs)
        if return_outputs:
            dpo_loss, outputs = result
        else:
            dpo_loss = result
            outputs = None

        pr_loss_val = torch.tensor(0.0, device=dpo_loss.device)
        if self._captured_hidden is not None and self._captured_hidden.shape[0] >= 4:
            H = self._captured_hidden.float()
            pr_loss_val = spectral_pr_loss(H, target_pr=self.pr_target, mode="floor")

        total_loss = dpo_loss + self.pr_lambda * pr_loss_val

        if self.state.global_step % PROBE_EVERY == 0:
            self.pr_loss_history.append({
                "step": self.state.global_step,
                "dpo_loss": dpo_loss.item(),
                "pr_loss": pr_loss_val.item(),
                "total_loss": total_loss.item(),
            })

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
def load_base_model():
    log.info("Loading model: %s", MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    return model, tok


def apply_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_data(tok):
    log.info("Loading dataset: %s (via %s)", DATASET_NAME, HF_MIRROR)
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    def format_to_trl(example):
        prompt = example["prompt"]
        chosen_raw = example["chosen"]
        rejected_raw = example["rejected"]
        if isinstance(chosen_raw, list):
            chosen_text = chosen_raw[-1]["content"] if chosen_raw else ""
        else:
            chosen_text = str(chosen_raw)
        if isinstance(rejected_raw, list):
            rejected_text = rejected_raw[-1]["content"] if rejected_raw else ""
        else:
            rejected_text = str(rejected_raw)
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
        }

    ds = ds.map(format_to_trl, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=42)
    train_ds = ds.select(range(min(N_SAMPLES, len(ds))))
    eval_ds = ds.select(range(N_SAMPLES, min(N_SAMPLES + EVAL_SAMPLES, len(ds))))
    log.info("Train: %d, Eval: %d", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


def make_dpo_config(run_name: str, output_dir: str) -> DPOConfig:
    return DPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        beta=BETA,
        max_length=MAX_LENGTH,
        bf16=True,
        logging_steps=10,
        save_steps=9999,
        eval_strategy="steps",
        eval_steps=100,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False,
        seed=42,
    )


def sfp_full_scan(model, tok, tag: str) -> dict:
    """Run full SFP probe (fast, 20 prompts)."""
    from spectral_flow_probe import SpectralProbe
    probe = SpectralProbe(MODEL_PATH, model=model, tokenizer=tok)
    report = probe.run(prompts=probe.default_prompts[:20], check_moe=False)
    result = report.to_dict()
    result["tag"] = tag
    log.info("[%s] Full SFP: ΔS=%.4f, PR(last)=%.2f", tag, report.delta_s, report.pr_last)
    return result


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# RUNS
# ════════════════════════════════════════════════════════════
def run_a():
    """Run A: Vanilla DPO."""
    log.info("=" * 60)
    log.info("RUN A: Vanilla DPO")
    log.info("=" * 60)
    t0 = time.time()

    model, tok = load_base_model()

    pre_scan = sfp_full_scan(model, tok, "A_pre")

    model = apply_lora(model)
    train_ds, eval_ds = load_data(tok)

    probe_cb = SpectralProbeCallback(
        every_n=PROBE_EVERY, n_samples=N_PROBE_SAMPLES, tag="A"
    )

    config = make_dpo_config("run_A_vanilla", str(RESULTS_DIR / "run_A"))
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=[probe_cb],
    )
    trainer.train()

    eval_result = trainer.evaluate()
    post_scan = sfp_full_scan(model, tok, "A_post")

    result = {
        "run": "A_vanilla_dpo",
        "elapsed_min": (time.time() - t0) / 60,
        "pre_scan": pre_scan,
        "post_scan": post_scan,
        "pr_history": probe_cb.history,
        "eval": eval_result,
        "final_step": trainer.state.global_step,
    }
    save_result(result, "run_A.json")

    del trainer, model
    cleanup()
    return result


def run_b():
    """Run B: DPO + SpectralMonitor early-stop."""
    log.info("=" * 60)
    log.info("RUN B: DPO + SpectralMonitor Early Stop (pr_halt=%.1f)", PR_HALT)
    log.info("=" * 60)
    t0 = time.time()

    model, tok = load_base_model()
    pre_scan = sfp_full_scan(model, tok, "B_pre")

    model = apply_lora(model)
    train_ds, eval_ds = load_data(tok)

    probe_cb = SpectralProbeCallback(
        every_n=PROBE_EVERY, n_samples=N_PROBE_SAMPLES,
        pr_halt=PR_HALT, tag="B"
    )

    config = make_dpo_config("run_B_earlystop", str(RESULTS_DIR / "run_B"))
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=[probe_cb],
    )
    trainer.train()

    eval_result = trainer.evaluate()
    post_scan = sfp_full_scan(model, tok, "B_post")

    result = {
        "run": "B_earlystop_dpo",
        "pr_halt": PR_HALT,
        "elapsed_min": (time.time() - t0) / 60,
        "pre_scan": pre_scan,
        "post_scan": post_scan,
        "pr_history": probe_cb.history,
        "eval": eval_result,
        "stopped_early": trainer.state.global_step < MAX_STEPS,
        "final_step": trainer.state.global_step,
    }
    save_result(result, "run_B.json")

    del trainer, model
    cleanup()
    return result


def run_c():
    """Run C: DPO + spectral_pr_loss regularizer."""
    log.info("=" * 60)
    log.info("RUN C: DPO + spectral_pr_loss (lambda=%.3f, target_pr=%.1f)",
             PR_LAMBDA, PR_TARGET)
    log.info("=" * 60)
    t0 = time.time()

    model, tok = load_base_model()
    pre_scan = sfp_full_scan(model, tok, "C_pre")

    model = apply_lora(model)
    train_ds, eval_ds = load_data(tok)

    probe_cb = SpectralProbeCallback(
        every_n=PROBE_EVERY, n_samples=N_PROBE_SAMPLES, tag="C"
    )

    config = make_dpo_config("run_C_regularized", str(RESULTS_DIR / "run_C"))
    trainer = SpectralDPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=[probe_cb],
        pr_lambda=PR_LAMBDA,
        pr_target=PR_TARGET,
    )
    trainer.train()

    eval_result = trainer.evaluate()
    post_scan = sfp_full_scan(model, tok, "C_post")

    result = {
        "run": "C_regularized_dpo",
        "pr_lambda": PR_LAMBDA,
        "pr_target": PR_TARGET,
        "elapsed_min": (time.time() - t0) / 60,
        "pre_scan": pre_scan,
        "post_scan": post_scan,
        "pr_history": probe_cb.history,
        "pr_loss_history": trainer.pr_loss_history,
        "eval": eval_result,
        "final_step": trainer.state.global_step,
    }
    save_result(result, "run_C.json")

    del trainer, model
    cleanup()
    return result


# ════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════
def plot_abc(result_a, result_b, result_c):
    """Generate the hero 3-curve comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("SFP DPO Experiment: PR Preservation via Spectral Regularization",
                 fontsize=14, fontweight="bold")

    configs = [
        (result_a, "A: Vanilla DPO", "#ef4444"),
        (result_b, "B: DPO + Early Stop", "#eab308"),
        (result_c, "C: DPO + PR Loss", "#22c55e"),
    ]

    # Panel 1: PR(last) vs step
    ax = axes[0]
    for res, label, color in configs:
        steps = [h["step"] for h in res["pr_history"]]
        prs = [h["pr_last"] for h in res["pr_history"]]
        ax.plot(steps, prs, "o-", label=label, color=color, markersize=3, linewidth=2)
    ax.axhline(BASELINE_PR, ls="--", color="gray", alpha=0.5, label=f"Baseline PR={BASELINE_PR}")
    ax.axhline(PR_TARGET, ls=":", color="blue", alpha=0.4, label=f"PR target={PR_TARGET}")
    ax.axhline(PR_HALT, ls=":", color="red", alpha=0.4, label=f"PR halt={PR_HALT}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("PR(last)")
    ax.set_title("PR(last) During Training")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Panel 2: S(last) vs step
    ax = axes[1]
    for res, label, color in configs:
        steps = [h["step"] for h in res["pr_history"]]
        ss = [h["s_last"] for h in res["pr_history"]]
        ax.plot(steps, ss, "s-", label=label, color=color, markersize=3, linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("S(last)")
    ax.set_title("S(last) During Training")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Summary bar chart
    ax = axes[2]
    labels_bar = ["A: Vanilla", "B: Early Stop", "C: + PR Loss"]
    pr_pre = [r["pre_scan"]["pr_last"] for r in [result_a, result_b, result_c]]
    pr_post = [r["post_scan"]["pr_last"] for r in [result_a, result_b, result_c]]
    x = np.arange(len(labels_bar))
    w = 0.35
    bars1 = ax.bar(x - w / 2, pr_pre, w, label="Before DPO", color="#93c5fd")
    bars2 = ax.bar(x + w / 2, pr_post, w, label="After DPO", color=["#ef4444", "#eab308", "#22c55e"])
    ax.set_ylabel("PR(last)")
    ax.set_title("PR Before vs After")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = RESULTS_DIR / "fig_dpo_abc.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    log.info("Saved hero figure: %s", path)
    plt.close(fig)

    # Bonus: pr_loss decomposition for Run C
    if result_c.get("pr_loss_history"):
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        steps = [h["step"] for h in result_c["pr_loss_history"]]
        dpo_l = [h["dpo_loss"] for h in result_c["pr_loss_history"]]
        pr_l = [h["pr_loss"] for h in result_c["pr_loss_history"]]
        ax2.plot(steps, dpo_l, "o-", label="DPO loss", color="#2563eb", markersize=3)
        ax2.plot(steps, pr_l, "s-", label=f"PR loss (x{PR_LAMBDA})", color="#dc2626", markersize=3)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.set_title("Run C: Loss Decomposition")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        path2 = RESULTS_DIR / "fig_run_c_loss.png"
        fig2.savefig(str(path2), dpi=150, bbox_inches="tight")
        log.info("Saved loss decomposition: %s", path2)
        plt.close(fig2)


# ════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════
def save_result(data: dict, filename: str):
    path = RESULTS_DIR / filename

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)
    log.info("Saved: %s", path)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("SFP DPO A/B/C EXPERIMENT")
    log.info("Model: %s", MODEL_PATH)
    log.info("Steps: %d, BS: %d, GA: %d, LR: %s", MAX_STEPS, BATCH_SIZE, GRAD_ACCUM, LR)
    log.info("PR_TARGET: %.1f, PR_HALT: %.1f, PR_LAMBDA: %.3f", PR_TARGET, PR_HALT, PR_LAMBDA)
    log.info("=" * 60)

    t_total = time.time()
    all_results = {}

    # Run A
    try:
        all_results["A"] = run_a()
    except Exception as e:
        log.error("Run A FAILED: %s", e, exc_info=True)
        all_results["A"] = {"error": str(e)}
    cleanup()

    # Run B
    try:
        all_results["B"] = run_b()
    except Exception as e:
        log.error("Run B FAILED: %s", e, exc_info=True)
        all_results["B"] = {"error": str(e)}
    cleanup()

    # Run C
    try:
        all_results["C"] = run_c()
    except Exception as e:
        log.error("Run C FAILED: %s", e, exc_info=True)
        all_results["C"] = {"error": str(e)}
    cleanup()

    # Summary + Plot
    elapsed = (time.time() - t_total) / 60
    log.info("=" * 60)
    log.info("ALL RUNS COMPLETE — %.1f min total", elapsed)

    summary = {"total_elapsed_min": elapsed}
    for key in ["A", "B", "C"]:
        r = all_results[key]
        if "error" in r:
            summary[key] = {"status": "FAILED", "error": r["error"]}
        else:
            summary[key] = {
                "status": "OK",
                "final_step": r["final_step"],
                "pr_pre": r["pre_scan"]["pr_last"],
                "pr_post": r["post_scan"]["pr_last"],
                "elapsed_min": r["elapsed_min"],
            }
            log.info("  %s: step=%d, PR %.2f → %.2f (%.1f min)",
                     key, r["final_step"],
                     r["pre_scan"]["pr_last"], r["post_scan"]["pr_last"],
                     r["elapsed_min"])

    save_result(summary, "SUMMARY.json")

    # Plot if all three succeeded
    if all("error" not in all_results[k] for k in ["A", "B", "C"]):
        try:
            plot_abc(all_results["A"], all_results["B"], all_results["C"])
        except Exception as e:
            log.error("Plotting failed: %s", e, exc_info=True)

    log.info("DONE. Results in: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
