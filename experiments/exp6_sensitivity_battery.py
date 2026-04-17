#!/usr/bin/env python3
"""
Exp 6: Sensitivity Battery — Which metrics detect PR collapse?
==============================================================
We proved PR collapse is real (69% on 1.7B). We proved ARC-Easy/HellaSwag
can't see it. Now we test metrics that SHOULD be sensitive to geometric
degradation:

Tier 1: Generation Diversity (Distinct-N, TTR, repetition rate)
Tier 2: Hard Benchmarks (ARC-Challenge, GSM8K via lm-eval)
Tier 3: LoRA Plasticity (fine-tune collapsed vs healthy checkpoint)

Uses existing checkpoints from Exp 5b (1.7B) — no retraining needed.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python experiments/exp6_sensitivity_battery.py
"""
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import setup_logging, cleanup, save_result, HF_MIRROR
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_flow_probe.core import run_pca_layer
from spectral_flow_probe._compat import find_decoder_layers

log = setup_logging("exp6_battery")

MODEL_PATH = "/cache/zhangjing/models/Qwen3-1.7B"
CKPT_BASE = Path(__file__).resolve().parent / "results" / "exp5b_fullft_1.7B" / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp6_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_STEPS = [0, 100, 200, 400, 600, 800]

EVAL_PROMPTS = [
    "请解释什么是Transformer架构。",
    "如何用Python实现一个简单的HTTP服务器？",
    "什么是量子纠缠？用简单的话解释。",
    "Explain the difference between TCP and UDP.",
    "如何在Linux中查找大文件？",
    "什么是GAN？它的工作原理是什么？",
    "请解释分布式系统中的CAP定理。",
    "How does gradient descent work?",
    "什么是零知识证明？",
    "请解释Docker和虚拟机的区别。",
    "写一首关于AI的诗。",
    "用一个比喻来描述互联网。",
    "如果你是一只猫你会干什么？",
    "Write a short story about a robot learning to paint.",
    "请用三句话解释相对论。",
    "What are the pros and cons of microservices architecture?",
    "如何向一个5岁小孩解释区块链？",
    "Describe the process of photosynthesis.",
    "请解释什么是递归，并给出一个例子。",
    "What is the trolley problem and why is it important?",
    "写一段关于未来城市的描写。",
    "How would you design a recommendation system?",
    "请比较监督学习和无监督学习。",
    "Explain how a neural network learns.",
    "如果时间可以倒流你想做什么？",
    "What is the difference between correlation and causation?",
    "请解释什么是注意力机制。",
    "How does HTTPS protect data in transit?",
    "用通俗的话解释傅里叶变换。",
    "What would happen if the internet disappeared for a day?",
    "请解释什么是强化学习。",
    "Write a haiku about machine learning.",
    "如何设计一个高可用的分布式系统？",
    "What is the significance of the Turing test?",
    "请解释什么是哈希表及其时间复杂度。",
    "Describe how search engines rank web pages.",
    "如果你能和历史上任何人对话你会选谁？",
    "What is the difference between deep learning and machine learning?",
    "请解释什么是API以及RESTful API。",
    "How do vaccines work at a molecular level?",
    "写一个关于程序员的笑话。",
    "What is quantum computing and why does it matter?",
    "请解释MapReduce的工作原理。",
    "How would you explain gravity to an alien?",
    "什么是大语言模型的幻觉问题？",
    "Describe three ways to reduce carbon emissions.",
    "请比较TCP和UDP协议的优缺点。",
    "What makes a good leader?",
    "如何用一句话总结人工智能的本质？",
    "Explain the concept of entropy in information theory.",
]

PPL_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing.",
    "In the beginning, there was nothing. Then, there was everything. The universe expanded.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems.",
    "The weather today is sunny with a high of 75 degrees Fahrenheit and low humidity.",
    "Python is a versatile programming language used in web development, data science, and AI.",
    "The theory of relativity fundamentally changed our understanding of space and time.",
    "To be or not to be, that is the question. Whether it is nobler in the mind to suffer.",
    "The Internet has transformed how we communicate, work, and access information globally.",
    "Climate change poses significant challenges to ecosystems and human societies worldwide.",
    "Quantum mechanics describes the behavior of particles at the atomic and subatomic level.",
] * 5


# ═══════════════════════════════════════════════════════════════
#  Tier 1: Generation Diversity
# ═══════════════════════════════════════════════════════════════

def distinct_n(texts: list[str], n: int) -> float:
    total_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        total_ngrams.extend(ngrams)
    if not total_ngrams:
        return 0.0
    return len(set(total_ngrams)) / len(total_ngrams)


def repetition_rate(texts: list[str], min_repeat: int = 3, ngram_size: int = 3) -> float:
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
    return math.exp(min(total_nll / total_tokens, 100))


@torch.no_grad()
def generate_responses(model, tokenizer, prompts: list[str],
                       max_new_tokens: int = 150) -> list[str]:
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


def run_tier1_diversity(model, tokenizer, step: int) -> dict:
    log.info("[Tier1] Step %d: generating %d responses...", step, len(EVAL_PROMPTS))
    responses = generate_responses(model, tokenizer, EVAL_PROMPTS)

    d1 = distinct_n(responses, 1)
    d2 = distinct_n(responses, 2)
    d3 = distinct_n(responses, 3)
    rep = repetition_rate(responses)
    ttr = type_token_ratio(responses)
    avg_len = mean_length(responses)
    ppl = compute_perplexity(model, tokenizer, PPL_TEXTS)

    result = {
        "distinct_1": d1, "distinct_2": d2, "distinct_3": d3,
        "repetition_rate": rep, "type_token_ratio": ttr,
        "mean_length": avg_len, "perplexity": ppl,
        "sample_responses": responses[:5],
    }
    log.info("[Tier1] Step %d: D1=%.3f D2=%.3f D3=%.3f Rep=%.3f TTR=%.3f Len=%.1f PPL=%.1f",
             step, d1, d2, d3, rep, ttr, avg_len, ppl)
    return result


# ═══════════════════════════════════════════════════════════════
#  Tier 2: Hard Benchmarks (ARC-Challenge, GSM8K)
# ═══════════════════════════════════════════════════════════════

def run_tier2_hard_benchmarks(model, tokenizer, step: int) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = ["arc_challenge"]
    log.info("[Tier2] Step %d: running lm-eval on %s", step, tasks)
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)
    results = lm_eval.simple_evaluate(
        model=lm_obj, tasks=tasks, batch_size=8, log_samples=False,
    )

    scores = {}
    if "results" in results:
        for task_name, task_res in results["results"].items():
            for key in ["acc_norm,none", "acc,none"]:
                if key in task_res:
                    scores[task_name] = task_res[key]
                    break

    log.info("[Tier2] Step %d: %s", step, scores)
    return scores


# ═══════════════════════════════════════════════════════════════
#  Tier 3: LoRA Plasticity Test
# ═══════════════════════════════════════════════════════════════

def run_tier3_plasticity(model_path: str, tokenizer, step: int,
                         gpu_id: int = 0) -> dict:
    """Fine-tune a fresh LoRA on this checkpoint for 100 steps, measure convergence."""
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer
    from datasets import load_dataset

    log.info("[Tier3] Step %d: loading model for plasticity test...", step)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": gpu_id},
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    os.environ["HF_ENDPOINT"] = HF_MIRROR
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256,
                         padding="max_length")

    tok_ds = ds.filter(lambda x: len(x["text"].strip()) > 50).select(range(2000))
    tok_ds = tok_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tok_ds.set_format("torch")

    loss_history = []

    class LossRecorder(torch.nn.Module):
        pass

    from transformers import TrainerCallback

    class RecordLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_history.append({"step": state.global_step, "loss": logs["loss"]})

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / f"plasticity_step{step}"),
        max_steps=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=9999,
        report_to="none",
        remove_unused_columns=False,
    )

    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tok_ds,
        data_collator=collator,
        callbacks=[RecordLossCallback()],
    )
    trainer.train()

    initial_loss = loss_history[0]["loss"] if loss_history else None
    final_loss = loss_history[-1]["loss"] if loss_history else None
    loss_drop = (initial_loss - final_loss) if (initial_loss and final_loss) else None

    result = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop": loss_drop,
        "loss_history": loss_history,
    }
    log.info("[Tier3] Step %d: init_loss=%.4f final_loss=%.4f drop=%.4f",
             step, initial_loss or 0, final_loss or 0, loss_drop or 0)

    del trainer, model
    cleanup()
    return result


# ═══════════════════════════════════════════════════════════════
#  PR Measurement
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_pr(model) -> float:
    try:
        _, layers, n_layers, _ = find_decoder_layers(model)
    except RuntimeError:
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        _, layers, n_layers, _ = find_decoder_layers(base)

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    last_layer = layers[-1]
    captures = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures.append(h[:, -1, :].detach().float().cpu().numpy())

    handle = last_layer.register_forward_hook(hook_fn)
    try:
        for _ in range(30):
            ids = torch.randint(100, 30000, (1, 64), device=device)
            model(input_ids=ids)
    finally:
        handle.remove()

    if was_training:
        model.train()

    if len(captures) < 5:
        return 0.0
    mat = np.vstack(captures)
    ls = run_pca_layer(mat)
    return ls.pr if ls else 0.0


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def load_checkpoint(step: int, gpu_id: int = 0):
    if step == 0:
        path = MODEL_PATH
    else:
        path = str(CKPT_BASE / f"step_{step}")
    log.info("Loading checkpoint step %d from %s on GPU %d", step, path, gpu_id)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": gpu_id},
    )
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Exp 6: Sensitivity Battery")
    log.info("Model: Qwen3-1.7B | Checkpoints: %s", CHECKPOINT_STEPS)
    log.info("Tier 1: Diversity | Tier 2: Hard Benchmarks | Tier 3: Plasticity")
    log.info("=" * 60)

    all_results = []

    for step in CHECKPOINT_STEPS:
        log.info("━" * 50)
        log.info("CHECKPOINT step=%d", step)
        log.info("━" * 50)

        model, tok = load_checkpoint(step, gpu_id=0)
        pr = measure_pr(model)
        log.info("PR(last) = %.2f", pr)

        # Tier 1: Diversity
        tier1 = run_tier1_diversity(model, tok, step)

        # Tier 2: Hard benchmarks
        tier2 = run_tier2_hard_benchmarks(model, tok, step)

        del model
        cleanup()

        # Tier 3: Plasticity (only for baseline and most-collapsed)
        tier3 = {}
        if step in [0, 200, 800]:
            ckpt_path = MODEL_PATH if step == 0 else str(CKPT_BASE / f"step_{step}")
            tier3 = run_tier3_plasticity(ckpt_path, tok, step, gpu_id=0)

        entry = {
            "step": step, "pr": pr,
            "tier1_diversity": tier1,
            "tier2_benchmarks": tier2,
            "tier3_plasticity": tier3,
        }
        all_results.append(entry)
        save_result(entry, RESULTS_DIR / f"step_{step}.json")

        del tok
        cleanup()

    # ── Correlation analysis ─────────────────────────────────────
    log.info("=" * 60)
    log.info("CORRELATION ANALYSIS")
    log.info("=" * 60)
    from scipy import stats

    prs = [r["pr"] for r in all_results]
    correlations = {}

    diversity_keys = ["distinct_1", "distinct_2", "distinct_3",
                      "repetition_rate", "type_token_ratio", "mean_length"]
    for key in diversity_keys:
        vals = [r["tier1_diversity"].get(key, 0) for r in all_results]
        if len(vals) >= 3:
            rho, p = stats.spearmanr(prs, vals)
            correlations[f"pr_vs_{key}"] = {"rho": float(rho), "pvalue": float(p)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            log.info("  PR vs %s: rho=%.3f p=%.4f %s", key, rho, p, sig)

    for task in ["arc_challenge"]:
        vals = [r["tier2_benchmarks"].get(task, 0) for r in all_results]
        if len(vals) >= 3 and any(v > 0 for v in vals):
            rho, p = stats.spearmanr(prs, vals)
            correlations[f"pr_vs_{task}"] = {"rho": float(rho), "pvalue": float(p)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            log.info("  PR vs %s: rho=%.3f p=%.4f %s", task, rho, p, sig)

    # ── Summary ──────────────────────────────────────────────────
    summary = {
        "model": MODEL_PATH,
        "experiment": "Exp 6: Sensitivity Battery",
        "checkpoints": all_results,
        "correlations": correlations,
        "elapsed_min": (time.time() - t0) / 60,
    }
    save_result(summary, RESULTS_DIR / "exp6_sensitivity_results.json")

    # ── Plot ─────────────────────────────────────────────────────
    try:
        plot_sensitivity(all_results, correlations, prs)
    except Exception as e:
        log.error("Plot failed: %s", e, exc_info=True)

    elapsed = (time.time() - t0) / 60
    log.info("=== Exp 6 DONE in %.1f min ===", elapsed)


def plot_sensitivity(results, correlations, prs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in results]
    n = len(steps)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Exp 6: Which Metrics Detect PR Collapse?\n"
        "Qwen3-1.7B Full FT DPO — PR collapses 69% but ARC-Easy/HellaSwag are blind",
        fontsize=14, fontweight="bold"
    )

    # PR trajectory
    ax = axes[0, 0]
    ax.plot(steps, prs, "o-", color="#dc2626", markersize=8, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("PR(last)")
    ax.set_title("PR Trajectory")
    ax.grid(True, alpha=0.3)
    for i, (s, p) in enumerate(zip(steps, prs)):
        ax.annotate(f"{p:.1f}", (s, p), fontsize=8, textcoords="offset points",
                    xytext=(0, 8), ha="center")

    # Distinct-2
    ax = axes[0, 1]
    d2s = [r["tier1_diversity"]["distinct_2"] for r in results]
    ax.plot(steps, d2s, "s-", color="#22c55e", markersize=8, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distinct-2")
    corr = correlations.get("pr_vs_distinct_2", {})
    rho_str = f"ρ={corr.get('rho', 0):.3f}" if corr else ""
    p_str = f"p={corr.get('pvalue', 1):.4f}" if corr else ""
    ax.set_title(f"Distinct-2 (diversity)\n{rho_str} {p_str}")
    ax.grid(True, alpha=0.3)

    # Repetition rate
    ax = axes[0, 2]
    reps = [r["tier1_diversity"]["repetition_rate"] for r in results]
    ax.plot(steps, reps, "^-", color="#f59e0b", markersize=8, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Repetition Rate")
    corr = correlations.get("pr_vs_repetition_rate", {})
    rho_str = f"ρ={corr.get('rho', 0):.3f}" if corr else ""
    p_str = f"p={corr.get('pvalue', 1):.4f}" if corr else ""
    ax.set_title(f"Repetition Rate\n{rho_str} {p_str}")
    ax.grid(True, alpha=0.3)

    # ARC-Challenge
    ax = axes[1, 0]
    arc_c = [r["tier2_benchmarks"].get("arc_challenge", 0) for r in results]
    ax.plot(steps, arc_c, "D-", color="#8b5cf6", markersize=8, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("ARC-Challenge Acc")
    corr = correlations.get("pr_vs_arc_challenge", {})
    rho_str = f"ρ={corr.get('rho', 0):.3f}" if corr else ""
    p_str = f"p={corr.get('pvalue', 1):.4f}" if corr else ""
    ax.set_title(f"ARC-Challenge (harder MCQ)\n{rho_str} {p_str}")
    ax.grid(True, alpha=0.3)

    # PR vs Distinct-2 scatter
    ax = axes[1, 1]
    scatter = ax.scatter(prs, d2s, c=steps, cmap="RdYlGn_r", s=100,
                         edgecolors="black", zorder=5)
    for j, step in enumerate(steps):
        ax.annotate(f"s{step}", (prs[j], d2s[j]), fontsize=8, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points")
    plt.colorbar(scatter, ax=ax, label="Step")
    ax.set_xlabel("PR(last)")
    ax.set_ylabel("Distinct-2")
    ax.set_title("PR vs Distinct-2 (scatter)")
    ax.grid(True, alpha=0.3)

    # Plasticity comparison
    ax = axes[1, 2]
    plasticity_steps = [r["step"] for r in results if r.get("tier3_plasticity")]
    if plasticity_steps:
        for r in results:
            if r.get("tier3_plasticity") and r["tier3_plasticity"].get("loss_history"):
                hist = r["tier3_plasticity"]["loss_history"]
                lh_steps = [h["step"] for h in hist]
                lh_loss = [h["loss"] for h in hist]
                ax.plot(lh_steps, lh_loss, "o-", markersize=4, linewidth=1.5,
                        label=f"step {r['step']} (PR={r['pr']:.1f})")
        ax.set_xlabel("Fine-tune Step")
        ax.set_ylabel("Loss")
        ax.set_title("LoRA Plasticity: Loss Curves")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No plasticity data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("LoRA Plasticity")

    plt.tight_layout()
    fig_path = RESULTS_DIR / "fig_exp6_sensitivity.png"
    fig.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", fig_path)


if __name__ == "__main__":
    main()
