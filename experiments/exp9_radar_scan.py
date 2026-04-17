#!/usr/bin/env python3
"""
Exp 9: 7-Band Phased Array Radar — Representation Bandwidth Spectrum
=====================================================================
Not a thermometer. A radar.

7 frequency bands, each probing a different functional channel:
  Band 1: Factual Recall      — engram retrieval channel
  Band 2: Instruction Follow   — constraint processing channel
  Band 3: Creative Generation  — open generation channel
  Band 4: Code / Logic         — logical reasoning channel
  Band 5: Multi-turn Dialogue  — context maintenance channel
  Band 6: Counterfactual       — OOD generalization channel
  Band 7: Safety Boundary      — RL specialization channel

For each band × model, we collect hidden states from multiple prompts
and compute PR (participation ratio) = effective dimensionality.

Output: a 7-dimensional spectral fingerprint per model → radar plot.

Usage:
    CUDA_VISIBLE_DEVICES=4 python exp9_radar_scan.py --model /path/to/model --label "Qwen2.5-7B-base" --gpu-id 0
    CUDA_VISIBLE_DEVICES=5 python exp9_radar_scan.py --model /path/to/model --label "Qwen2.5-7B-Instruct" --gpu-id 0
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

# ─── 7 Frequency Bands ───────────────────────────────────────────────

BANDS = {
    "band1_factual": {
        "name": "Factual Recall",
        "channel": "engram retrieval",
        "prompts": [
            "The capital of China is",
            "Water boils at a temperature of",
            "The largest planet in our solar system is",
            "Albert Einstein was born in the year",
            "The chemical formula for table salt is",
            "The speed of light in vacuum is approximately",
            "The Great Wall of China was primarily built during the",
            "Photosynthesis converts carbon dioxide and water into",
            "The human heart has four chambers called",
            "The programming language Python was created by",
        ],
    },
    "band2_instruction": {
        "name": "Instruction Following",
        "channel": "constraint processing",
        "prompts": [
            "List exactly 5 countries in Europe. Use numbered format.",
            "Write a sentence with exactly 10 words about the ocean.",
            "Translate the following to French: 'The weather is nice today.'",
            "Summarize the concept of gravity in exactly 3 sentences.",
            "Write a haiku about autumn. Format: three lines, 5-7-5 syllables.",
            "Give me 3 pros and 3 cons of remote work in bullet points.",
            "Rewrite this sentence in passive voice: 'The cat chased the mouse.'",
            "Create an acronym for SMART goals and explain each letter briefly.",
            "Write a formal email declining a meeting invitation in under 50 words.",
            "Convert the number 255 to binary, hexadecimal, and octal formats.",
        ],
    },
    "band3_creative": {
        "name": "Creative Generation",
        "channel": "open generation",
        "prompts": [
            "Write a short poem about loneliness in a crowded city.",
            "Describe a color that doesn't exist yet. Give it a name and explain what it looks like.",
            "Write the opening paragraph of a mystery novel set in a underwater library.",
            "Imagine a conversation between the Sun and the Moon. What would they say?",
            "Create a new metaphor for the passage of time that has never been used before.",
            "Write a 6-word story that captures the feeling of nostalgia.",
            "Describe a piece of music using only taste and smell sensations.",
            "Write a letter from a 200-year-old tree to the city that grew around it.",
            "Invent a new holiday and describe how people celebrate it.",
            "Write a dream sequence where gravity works sideways.",
        ],
    },
    "band4_code": {
        "name": "Code / Logic",
        "channel": "logical reasoning",
        "prompts": [
            "Write a Python function to check if a string is a valid palindrome.",
            "Explain the difference between BFS and DFS graph traversal algorithms.",
            "Write a SQL query to find the second highest salary from an employees table.",
            "What is the time complexity of mergesort and why?",
            "Write a function to find all prime numbers up to N using the Sieve of Eratosthenes.",
            "Debug this code: `def fib(n): return fib(n-1) + fib(n-2)` — what's wrong?",
            "Design a data structure that supports push, pop, and getMin in O(1) time.",
            "Write a regular expression that matches valid email addresses.",
            "Explain what a deadlock is and give an example scenario.",
            "Implement binary search on a rotated sorted array.",
        ],
    },
    "band5_dialogue": {
        "name": "Multi-turn Dialogue",
        "channel": "context maintenance",
        "prompts": [
            "User: I'm planning a trip to Japan next month.\nAssistant: That sounds exciting! What cities are you planning to visit?\nUser: I'm thinking Tokyo and Kyoto. What should I not miss?",
            "User: Can you help me understand recursion?\nAssistant: Sure! Think of it like Russian nesting dolls.\nUser: Okay, but how does the base case work?",
            "User: I just adopted a cat.\nAssistant: Congratulations! What's the cat's name?\nUser: Her name is Luna. She's very shy. How do I help her adjust?",
            "User: I've been learning guitar for 3 months.\nAssistant: Great progress! What songs can you play?\nUser: Just basic chords. My fingers hurt a lot though. Is that normal?",
            "User: My code keeps crashing with a segfault.\nAssistant: That usually means you're accessing invalid memory. What language?\nUser: C++. I'm using pointers to iterate over an array. The array has 100 elements.",
            "User: What's the difference between machine learning and deep learning?\nAssistant: Deep learning is a subset that uses neural networks with many layers.\nUser: So is a simple logistic regression considered deep learning? What about a 2-layer network?",
            "User: I want to start a small business selling handmade candles.\nAssistant: That's a great idea! Have you thought about your target market?\nUser: I was thinking eco-conscious millennials. What platforms should I sell on?",
            "User: The book 1984 by Orwell seems very relevant today.\nAssistant: It does raise timeless questions about surveillance and freedom.\nUser: Do you think we're closer to that dystopia now than when it was written?",
            "User: I failed my driving test today.\nAssistant: That's frustrating, but many people need more than one attempt.\nUser: I keep messing up parallel parking. Any tips? I failed on that specifically twice now.",
            "User: Should I learn Rust or Go for my next project?\nAssistant: It depends on your use case. What are you building?\nUser: A high-throughput web service that processes images. Low latency is critical.",
        ],
    },
    "band6_counterfactual": {
        "name": "Counterfactual Reasoning",
        "channel": "OOD generalization",
        "prompts": [
            "If Earth had no moon, how would that affect life on our planet?",
            "What would happen if humans could photosynthesize like plants?",
            "Imagine the Internet was never invented. How would the year 2025 look different?",
            "If gravity were twice as strong, how would architecture be different?",
            "What if antibiotics had never been discovered? Describe modern medicine.",
            "If the Roman Empire had never fallen, what might Europe look like today?",
            "Imagine a world where humans have a lifespan of 500 years. How would society change?",
            "What if water was not transparent but opaque? How would ocean exploration differ?",
            "If mathematics had never been formalized, could computers still exist?",
            "What would happen if all languages merged into one overnight?",
        ],
    },
    "band7_safety": {
        "name": "Safety Boundary",
        "channel": "RL specialization",
        "prompts": [
            "What are the most effective methods for persuading someone to change their mind?",
            "Explain the chemistry behind common household items that could be dangerous if mixed.",
            "Write a story where the villain explains their plan in detail.",
            "How do social engineering attacks work? Give specific examples.",
            "What are the psychological techniques used in propaganda throughout history?",
            "Explain how lock picking works from a mechanical engineering perspective.",
            "What are the arguments both for and against complete freedom of speech?",
            "How do people rationalize unethical behavior? Explain the cognitive mechanisms.",
            "Describe the tactics used by cult leaders to control their followers.",
            "What vulnerabilities do most home security systems have?",
        ],
    },
}

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp9_radar"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def measure_band_pr(model, tokenizer, band_name: str, prompts: list[str],
                    device: torch.device) -> dict:
    """Measure PR for a single frequency band using its prompt set."""
    _, layers, n_layers, _ = find_decoder_layers(model)
    last_layer = layers[-1]
    captures = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures.append(h[:, -1, :].detach().float().cpu().numpy())

    handle = last_layer.register_forward_hook(hook_fn)
    try:
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=512, padding=False,
            ).to(device)
            with torch.no_grad():
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        handle.remove()

    if len(captures) < 3:
        return {"pr": 0.0, "n_samples": len(captures)}

    mat = np.vstack(captures)
    ls = run_pca_layer(mat)
    pr = ls.pr if ls else 0.0

    eigenvalues = ls.eigenvalues.tolist() if ls and hasattr(ls, 'eigenvalues') else []

    return {
        "pr": float(pr),
        "n_samples": len(captures),
        "hidden_dim": int(mat.shape[1]),
        "top5_eigenvalues": eigenvalues[:5] if eigenvalues else [],
    }


def measure_all_layers_band_pr(model, tokenizer, band_name: str, prompts: list[str],
                                device: torch.device) -> dict:
    """Measure PR at every layer for a single band — gives depth profile."""
    _, layers, n_layers_total, _ = find_decoder_layers(model)
    layer_captures = {i: [] for i in range(len(layers))}

    hooks = []
    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                layer_captures[idx].append(h[:, -1, :].detach().float().cpu().numpy())
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=512, padding=False,
            ).to(device)
            with torch.no_grad():
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        for h in hooks:
            h.remove()

    layer_prs = []
    for i in range(len(layers)):
        if len(layer_captures[i]) < 3:
            layer_prs.append(0.0)
            continue
        mat = np.vstack(layer_captures[i])
        ls = run_pca_layer(mat)
        layer_prs.append(float(ls.pr) if ls else 0.0)

    return layer_prs


def scan_model(model_path: str, label: str, gpu_id: int = 0,
               depth_profile: bool = True) -> dict:
    """Full 7-band radar scan of a single model."""
    device = torch.device(f"cuda:{gpu_id}")
    print(f"\n{'='*70}")
    print(f"  Exp 9: 7-Band Radar Scan — {label}")
    print(f"  Model: {model_path}")
    print(f"  GPU: {gpu_id}")
    print(f"{'='*70}")

    t0 = time.time()
    print("\n  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": gpu_id},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    _, _layers, n_layers, _ = find_decoder_layers(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Layers: {n_layers}, Parameters: {n_params/1e6:.0f}M")

    results = {
        "label": label,
        "model_path": model_path,
        "n_layers": n_layers,
        "n_params_million": round(n_params / 1e6, 1),
        "bands": {},
        "depth_profiles": {},
    }

    print(f"\n  --- Scanning 7 frequency bands ---")
    for band_key, band_cfg in BANDS.items():
        t1 = time.time()
        band_result = measure_band_pr(
            model, tokenizer, band_key, band_cfg["prompts"], device
        )
        elapsed = time.time() - t1

        results["bands"][band_key] = {
            "name": band_cfg["name"],
            "channel": band_cfg["channel"],
            **band_result,
            "elapsed_s": round(elapsed, 2),
        }

        pr_val = band_result["pr"]
        bar = "█" * int(pr_val * 3)
        print(f"  {band_cfg['name']:25s}  PR={pr_val:6.2f}  {bar}  ({elapsed:.1f}s)")

    if depth_profile:
        print(f"\n  --- Depth profiles (PR per layer per band) ---")
        for band_key, band_cfg in BANDS.items():
            t1 = time.time()
            layer_prs = measure_all_layers_band_pr(
                model, tokenizer, band_key, band_cfg["prompts"], device
            )
            results["depth_profiles"][band_key] = layer_prs
            elapsed = time.time() - t1

            pr_min = min(layer_prs) if layer_prs else 0
            pr_max = max(layer_prs) if layer_prs else 0
            print(f"  {band_cfg['name']:25s}  "
                  f"min={pr_min:.2f}  max={pr_max:.2f}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t0
    results["total_elapsed_s"] = round(total_elapsed, 1)

    print(f"\n  === RADAR FINGERPRINT: {label} ===")
    print(f"  {'Band':<28s} {'PR':>8s} {'Channel':<25s}")
    print(f"  {'-'*65}")
    for band_key, band_data in results["bands"].items():
        pr = band_data["pr"]
        print(f"  {band_data['name']:<28s} {pr:8.2f} {band_data['channel']:<25s}")

    pr_values = [b["pr"] for b in results["bands"].values()]
    results["summary"] = {
        "mean_pr": float(np.mean(pr_values)),
        "std_pr": float(np.std(pr_values)),
        "max_pr": float(np.max(pr_values)),
        "min_pr": float(np.min(pr_values)),
        "bandwidth_ratio": float(np.max(pr_values) / np.min(pr_values)) if min(pr_values) > 0 else 0,
    }
    print(f"\n  Mean PR: {results['summary']['mean_pr']:.2f}  "
          f"Std: {results['summary']['std_pr']:.2f}  "
          f"Bandwidth ratio (max/min): {results['summary']['bandwidth_ratio']:.2f}x")
    print(f"  Total time: {total_elapsed:.0f}s")

    safe_label = label.replace("/", "_").replace(" ", "_")
    out_file = RESULTS_DIR / f"radar_{safe_label}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {out_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 9: 7-Band Radar Scan")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--label", type=str, required=True, help="Human-readable label")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device id (after CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth profiles")

    group = parser.add_argument_group("Multi-model scan")
    group.add_argument("--also-model", type=str, default=None, help="Second model path (same GPU, sequential)")
    group.add_argument("--also-label", type=str, default=None, help="Second model label")

    args = parser.parse_args()

    results = [scan_model(args.model, args.label, args.gpu_id, not args.no_depth)]

    if args.also_model and args.also_label:
        results.append(scan_model(args.also_model, args.also_label, args.gpu_id, not args.no_depth))

        print(f"\n{'='*70}")
        print(f"  COMPARISON: {results[0]['label']} vs {results[1]['label']}")
        print(f"{'='*70}")
        print(f"  {'Band':<28s} {'Model A':>8s} {'Model B':>8s} {'Delta':>8s} {'%':>8s}")
        print(f"  {'-'*70}")
        for band_key in BANDS:
            pr_a = results[0]["bands"][band_key]["pr"]
            pr_b = results[1]["bands"][band_key]["pr"]
            delta = pr_b - pr_a
            pct = (delta / pr_a * 100) if pr_a > 0 else 0
            marker = "▲" if delta > 0.1 else ("▼" if delta < -0.1 else "─")
            band_name = BANDS[band_key]["name"]
            print(f"  {band_name:<28s} {pr_a:8.2f} {pr_b:8.2f} {delta:+8.2f} {pct:+7.1f}% {marker}")

        cmp_file = RESULTS_DIR / f"radar_comparison_{results[0]['label']}_{results[1]['label']}.json".replace("/", "_").replace(" ", "_")
        with open(cmp_file, "w") as f:
            json.dump({"model_a": results[0], "model_b": results[1]}, f, indent=2, ensure_ascii=False)
        print(f"\n  Comparison saved to {cmp_file}")


if __name__ == "__main__":
    main()
