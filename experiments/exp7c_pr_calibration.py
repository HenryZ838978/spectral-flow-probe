#!/usr/bin/env python3
"""
Exp 7C: PR Probe Calibration — Is our thermometer broken?
==========================================================
Same checkpoint, same model, but PR measurements differ 5x between runs.
This script measures PR variance systematically:

Method A: Fixed-seed random tokens (same tokens every run, 10 seeds)
Method B: Variable-seed random tokens (different tokens each run, 10 runs)
Method C: Fixed real prompts (deterministic, same text every time)

For each checkpoint: 10 measurements × 3 methods = 30 PR values.
Reports mean, std, min, max per method per checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=6 python exp7c_pr_calibration.py --model-size 1.7B --gpu-id 0
    CUDA_VISIBLE_DEVICES=7 python exp7c_pr_calibration.py --model-size 0.6B --gpu-id 0
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import setup_logging, HF_MIRROR
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_flow_probe.core import run_pca_layer
from spectral_flow_probe._compat import find_decoder_layers

os.environ["HF_ENDPOINT"] = HF_MIRROR
log = setup_logging("exp7c_calibration")

MODEL_CONFIGS = {
    "1.7B": {
        "base_path": "/cache/zhangjing/models/Qwen3-1.7B",
        "ckpt_base": Path(__file__).resolve().parent / "results" / "exp5b_fullft_1.7B" / "checkpoints",
    },
    "0.6B": {
        "base_path": "/cache/zhangjing/models/Qwen3-0.6B",
        "ckpt_base": Path(__file__).resolve().parent / "results" / "exp5_fullft_causality" / "checkpoints",
    },
}

CHECKPOINT_STEPS = [0, 200, 400, 800]
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp7_ood_probe"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = 10
N_PROBES = 30
SEQ_LEN = 64

FIXED_PROMPTS = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Machine learning is a subset of artificial intelligence that learns from data.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "In quantum mechanics, particles can exist in multiple states simultaneously.",
    "Python is a versatile programming language used for web development and AI.",
    "The theory of relativity was proposed by Albert Einstein in 1905.",
    "Climate change poses significant challenges to ecosystems worldwide.",
    "Neural networks consist of layers of interconnected nodes that process information.",
    "The Internet has fundamentally transformed how humans communicate globally.",
    "Distributed systems require careful handling of consistency and partition tolerance.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules.",
    "The Fibonacci sequence appears frequently in nature and mathematical structures.",
    "Cryptographic hash functions provide one-way transformation of arbitrary data.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
    "Water molecules consist of two hydrogen atoms bonded to one oxygen atom.",
    "Recursion is a programming technique where a function calls itself.",
    "The speed of light in vacuum is approximately 299792458 meters per second.",
    "Database transactions must satisfy atomicity consistency isolation and durability.",
    "Evolution through natural selection drives the diversity of life on Earth.",
    "Gradient descent optimizes neural network parameters by following the loss gradient.",
    "The periodic table organizes chemical elements by their atomic number.",
    "TCP provides reliable ordered delivery of data between applications.",
    "Earthquakes occur when tectonic plates suddenly release accumulated stress.",
    "Transformers use self-attention to process sequences without recurrence.",
    "The mitochondria are the powerhouse of the cell producing ATP energy.",
    "Binary search efficiently finds elements in sorted arrays with O(log n) time.",
    "Galaxies contain billions of stars held together by gravitational forces.",
    "Functional programming emphasizes immutability and pure functions without side effects.",
    "DNA contains the genetic instructions for the development of living organisms.",
    "Hash tables provide average O(1) lookup time using key-value associations.",
]


def get_decoder_hook(model):
    """Get the last decoder layer for hooking."""
    try:
        _, layers, n_layers, _ = find_decoder_layers(model)
    except RuntimeError:
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        _, layers, n_layers, _ = find_decoder_layers(base)
    return layers[-1]


@torch.no_grad()
def measure_pr_fixed_seed(model, seed: int) -> float:
    """Method A: Fixed-seed random tokens."""
    device = next(model.parameters()).device
    model.eval()
    last_layer = get_decoder_hook(model)
    captures = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures.append(h[:, -1, :].detach().float().cpu().numpy())

    handle = last_layer.register_forward_hook(hook_fn)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    try:
        for _ in range(N_PROBES):
            ids = torch.randint(100, 30000, (1, SEQ_LEN), generator=rng, device=device)
            model(input_ids=ids)
    finally:
        handle.remove()

    if len(captures) < 5:
        return 0.0
    mat = np.vstack(captures)
    ls = run_pca_layer(mat)
    return ls.pr if ls else 0.0


@torch.no_grad()
def measure_pr_variable_seed(model) -> float:
    """Method B: Variable-seed random tokens (no seed control)."""
    device = next(model.parameters()).device
    model.eval()
    last_layer = get_decoder_hook(model)
    captures = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures.append(h[:, -1, :].detach().float().cpu().numpy())

    handle = last_layer.register_forward_hook(hook_fn)
    try:
        for _ in range(N_PROBES):
            ids = torch.randint(100, 30000, (1, SEQ_LEN), device=device)
            model(input_ids=ids)
    finally:
        handle.remove()

    if len(captures) < 5:
        return 0.0
    mat = np.vstack(captures)
    ls = run_pca_layer(mat)
    return ls.pr if ls else 0.0


@torch.no_grad()
def measure_pr_fixed_prompts(model, tokenizer) -> float:
    """Method C: Fixed real prompts (completely deterministic)."""
    device = next(model.parameters()).device
    model.eval()
    last_layer = get_decoder_hook(model)
    captures = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures.append(h[:, -1, :].detach().float().cpu().numpy())

    handle = last_layer.register_forward_hook(hook_fn)
    try:
        for prompt in FIXED_PROMPTS:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=SEQ_LEN, padding="max_length").to(device)
            model(input_ids=enc["input_ids"])
    finally:
        handle.remove()

    if len(captures) < 5:
        return 0.0
    mat = np.vstack(captures)
    ls = run_pca_layer(mat)
    return ls.pr if ls else 0.0


def load_checkpoint(step: int, model_size: str, gpu_id: int = 0):
    cfg = MODEL_CONFIGS[model_size]
    path = cfg["base_path"] if step == 0 else str(cfg["ckpt_base"] / f"step_{step}")
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


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def summarize_measurements(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "values": values,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["1.7B", "0.6B"], required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    model_size = args.model_size

    log.info("=" * 60)
    log.info("Exp 7C: PR Probe Calibration — Qwen3-%s", model_size)
    log.info("Is our thermometer broken?")
    log.info("Methods: A=fixed-seed, B=variable-seed, C=fixed-prompts")
    log.info("%d repeats per method per checkpoint", N_REPEATS)
    log.info("=" * 60)

    all_results = []

    for step in CHECKPOINT_STEPS:
        log.info("━" * 50)
        log.info("CHECKPOINT step=%d (%s)", step, model_size)
        log.info("━" * 50)

        model, tok = load_checkpoint(step, model_size, args.gpu_id)

        # Method A: Fixed seeds (seeds 0-9, deterministic)
        log.info("[Method A] Fixed-seed random tokens × %d", N_REPEATS)
        pr_a = []
        for seed in range(N_REPEATS):
            pr = measure_pr_fixed_seed(model, seed=seed)
            pr_a.append(pr)
            log.info("  seed=%d: PR=%.2f", seed, pr)
        stats_a = summarize_measurements(pr_a)
        log.info("  → A: mean=%.2f ± %.2f (CV=%.1f%%)",
                 stats_a["mean"], stats_a["std"], stats_a["cv"] * 100)

        # Method B: Variable seeds (no control)
        log.info("[Method B] Variable-seed random tokens × %d", N_REPEATS)
        pr_b = []
        for i in range(N_REPEATS):
            pr = measure_pr_variable_seed(model)
            pr_b.append(pr)
            log.info("  run=%d: PR=%.2f", i, pr)
        stats_b = summarize_measurements(pr_b)
        log.info("  → B: mean=%.2f ± %.2f (CV=%.1f%%)",
                 stats_b["mean"], stats_b["std"], stats_b["cv"] * 100)

        # Method C: Fixed prompts (completely deterministic — should give SAME value every time)
        log.info("[Method C] Fixed real prompts × %d", N_REPEATS)
        pr_c = []
        for i in range(N_REPEATS):
            pr = measure_pr_fixed_prompts(model, tok)
            pr_c.append(pr)
            log.info("  run=%d: PR=%.2f", i, pr)
        stats_c = summarize_measurements(pr_c)
        log.info("  → C: mean=%.2f ± %.2f (CV=%.1f%%)",
                 stats_c["mean"], stats_c["std"], stats_c["cv"] * 100)

        entry = {
            "step": step,
            "model_size": model_size,
            "method_a_fixed_seed": stats_a,
            "method_b_variable_seed": stats_b,
            "method_c_fixed_prompts": stats_c,
        }
        all_results.append(entry)

        del model, tok
        cleanup()

    # Summary table
    log.info("=" * 60)
    log.info("CALIBRATION SUMMARY — Qwen3-%s", model_size)
    log.info("=" * 60)
    log.info("%-6s | %-22s | %-22s | %-22s", "Step",
             "A: Fixed Seed", "B: Variable Seed", "C: Fixed Prompts")
    log.info("-" * 80)
    for r in all_results:
        a = r["method_a_fixed_seed"]
        b = r["method_b_variable_seed"]
        c = r["method_c_fixed_prompts"]
        log.info("%-6d | %.1f ± %.1f (CV=%.0f%%) | %.1f ± %.1f (CV=%.0f%%) | %.1f ± %.1f (CV=%.0f%%)",
                 r["step"],
                 a["mean"], a["std"], a["cv"]*100,
                 b["mean"], b["std"], b["cv"]*100,
                 c["mean"], c["std"], c["cv"]*100)

    summary = {
        "experiment": f"Exp 7C: PR Probe Calibration — Qwen3-{model_size}",
        "model_size": model_size,
        "n_repeats": N_REPEATS,
        "n_probes": N_PROBES,
        "seq_len": SEQ_LEN,
        "checkpoints": all_results,
        "elapsed_min": (time.time() - t0) / 60,
    }

    out_file = RESULTS_DIR / f"calibration_{model_size}_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Saved: %s", out_file)
    log.info("=== Exp 7C DONE in %.1f min ===", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
