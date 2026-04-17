#!/usr/bin/env python3
"""
Exp 9B Band 0: Raw text completion prompts — the base model's home turf.
=========================================================================
If RL reallocates bandwidth, base should score higher here than instruct.
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

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_flow_probe.core import run_pca_layer
from spectral_flow_probe._compat import find_decoder_layers

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp9_radar"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BAND0_PROMPTS = [
    "The following is an excerpt from a scientific paper on quantum field theory published in Physical Review Letters:",
    "In the year 1847, the city of London was experiencing rapid industrial growth. The factories along the Thames",
    "Chapter 3: The Algorithm\n\nThe fundamental problem with recursive descent parsing is that",
    "Once upon a time, in a kingdom far beyond the mountains where the rivers flow upward, there lived",
    "Abstract: We present a novel approach to large-scale distributed systems that achieves consensus in",
    "The mitochondrial electron transport chain consists of four major protein complexes embedded in the inner",
    "WASHINGTON (Reuters) — Federal Reserve officials on Wednesday signaled they would keep interest rates",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left =",
    "The history of mathematics can be traced back to ancient Mesopotamia, where clay tablets from around 1800 BCE",
    "Ingredients:\n- 2 cups all-purpose flour\n- 1 cup unsalted butter, softened\n- 3/4 cup granulated sugar\nInstructions:\n1.",
]

INSTRUCTION_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum entanglement in simple terms.",
    "Write a Python function to reverse a linked list.",
    "List 5 benefits of regular exercise.",
    "Translate 'hello world' to Japanese.",
    "What would happen if the speed of light were halved?",
    "Compare and contrast TCP and UDP protocols.",
    "Write a haiku about the ocean.",
    "How does photosynthesis work?",
    "Describe the difference between supervised and unsupervised learning.",
]


def measure_pr(model, tokenizer, prompts, device, label=""):
    _, layers, n_layers, _ = find_decoder_layers(model)
    last_layer = layers[-1]
    captures = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures.append(h[:, -1, :].detach().float().cpu().numpy())

    handle = last_layer.register_forward_hook(hook_fn)
    try:
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512, padding=False).to(device)
            with torch.no_grad():
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        handle.remove()

    if len(captures) < 3:
        return 0.0
    mat = np.vstack(captures)
    ls = run_pca_layer(mat)
    return float(ls.pr) if ls else 0.0


def scan_model(model_path, label, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"\n  Loading {label}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": gpu_id},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    pr_band0 = measure_pr(model, tokenizer, BAND0_PROMPTS, device, "Band0")
    pr_instr = measure_pr(model, tokenizer, INSTRUCTION_PROMPTS, device, "Instruction")
    elapsed = time.time() - t0

    result = {
        "label": label,
        "band0_raw_completion": pr_band0,
        "band_instruction": pr_instr,
        "ratio_raw_over_instr": pr_band0 / pr_instr if pr_instr > 0 else 0,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"  {label}:")
    print(f"    Band 0 (Raw Completion):  PR = {pr_band0:.2f}")
    print(f"    Band Instruction:         PR = {pr_instr:.2f}")
    print(f"    Ratio (raw/instr):        {result['ratio_raw_over_instr']:.3f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    print("=" * 70)
    print("  Exp 9B: Band 0 — Raw Text vs Instruction")
    print("  Does the base model have MORE bandwidth on its home turf?")
    print("=" * 70)

    r_base = scan_model("/cache/zhangjing/models/Qwen2.5-7B", "Qwen2.5-7B-base", gpu_id=0)
    r_inst = scan_model("/cache/zhangjing/models/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", gpu_id=0)

    print(f"\n  === COMPARISON ===")
    print(f"  {'':30s} {'Base':>8s} {'Instruct':>10s} {'Delta':>8s}")
    print(f"  {'-'*60}")
    for band_name, key in [("Band 0: Raw Completion", "band0_raw_completion"),
                            ("Band: Instruction", "band_instruction")]:
        vb = r_base[key]
        vi = r_inst[key]
        delta = vi - vb
        pct = delta / vb * 100 if vb > 0 else 0
        marker = "▲" if delta > 0.2 else ("▼" if delta < -0.2 else "─")
        print(f"  {band_name:<30s} {vb:8.2f} {vi:10.2f} {delta:+7.2f} ({pct:+.1f}%) {marker}")

    print(f"\n  Base  ratio (raw/instr) = {r_base['ratio_raw_over_instr']:.3f}")
    print(f"  Inst  ratio (raw/instr) = {r_inst['ratio_raw_over_instr']:.3f}")

    out = RESULTS_DIR / "band0_raw_vs_instruction.json"
    with open(out, "w") as f:
        json.dump({"base": r_base, "instruct": r_inst}, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
