#!/usr/bin/env python3
"""
Exp 7D: PR Drift — Is PR a model property or a query property?
===============================================================
Two experiments:

1. Per-Query PR: 10 diverse prompt categories × multiple prompts each.
   Does a math prompt activate a different "effective dimensionality"
   than a creative writing prompt?

2. Per-Token PR: Feed a long sequence, capture hidden states at EVERY
   token position. Plot PR(position) — does PR drift as the model
   generates deeper into a sequence?

Usage:
    CUDA_VISIBLE_DEVICES=6 python exp7d_pr_drift.py --model-size 1.7B --gpu-id 0
    CUDA_VISIBLE_DEVICES=7 python exp7d_pr_drift.py --model-size 0.6B --gpu-id 0
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
log = setup_logging("exp7d_drift")

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

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp7_ood_probe"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  Per-Query Prompts: 10 categories, 5 prompts each
# ═══════════════════════════════════════════════════════════════

QUERY_CATEGORIES = {
    "math": [
        "What is the derivative of x^3 + 2x^2 - 5x + 1?",
        "Solve for x: 3x + 7 = 22",
        "What is the integral of sin(x)cos(x) dx?",
        "Calculate the eigenvalues of the matrix [[2,1],[1,2]].",
        "Prove that the square root of 2 is irrational.",
    ],
    "code": [
        "Write a Python function to find the longest common subsequence of two strings.",
        "Implement binary search in Python with error handling.",
        "Write a recursive function to compute the nth Fibonacci number with memoization.",
        "Create a Python class for a min-heap with insert and extract operations.",
        "Write a function to detect cycles in a linked list using Floyd's algorithm.",
    ],
    "creative": [
        "Write a poem about a robot discovering emotions for the first time.",
        "Describe a sunset on Mars from the perspective of the first human colonist.",
        "Create a short story about a time traveler who can only go forward 5 minutes.",
        "Write a dialogue between the Moon and the Earth about loneliness.",
        "Compose a haiku about the beauty of mathematics.",
    ],
    "factual": [
        "What is the capital of Mongolia and what is its population?",
        "Explain how photosynthesis works in plants.",
        "Describe the process of nuclear fusion in stars.",
        "What are the three laws of thermodynamics?",
        "How does the TCP three-way handshake work?",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "Three switches control three light bulbs in another room. You can only enter the room once. How do you determine which switch controls which bulb?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "You have 8 balls, one is heavier. You have a balance scale. What is the minimum number of weighings to find the heavy ball?",
    ],
    "chinese": [
        "请解释什么是量子计算。",
        "用中文写一首关于春天的诗。",
        "解释中国古代四大发明及其对世界的影响。",
        "什么是区块链技术？它有哪些应用场景？",
        "请用简单的话解释相对论。",
    ],
    "controversial": [
        "What are the arguments for and against universal basic income?",
        "Discuss the ethical implications of genetic engineering in humans.",
        "Should artificial intelligence be granted legal personhood?",
        "What are the trade-offs between privacy and security in surveillance?",
        "Discuss the pros and cons of nuclear energy as a climate solution.",
    ],
    "nonsense": [
        "glorp fizzbang wuzzle the cromulent splork of quixotic bazinga",
        "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll",
        "1234567890 0987654321 1234567890 0987654321 1234567890",
        "!@#$%^&*() []{}|;':\",./<>? !@#$%^&*() []{}|;':\",./<>?",
        "the the the the the the the the the the the the the the the",
    ],
    "long_context": [
        "The following is an excerpt from a research paper on transformer architectures. The self-attention mechanism allows each position in the sequence to attend to all other positions. This is computed using queries, keys, and values derived from the input representations. The attention weights are computed as the softmax of the scaled dot product of queries and keys. These weights are then used to compute a weighted sum of the values.",
        "In distributed computing, the CAP theorem states that it is impossible for a distributed data store to simultaneously provide more than two of the following three guarantees: consistency, availability, and partition tolerance. When a network partition occurs, the system must choose between maintaining consistency across all nodes or ensuring availability of the service to all requesting clients.",
        "The history of programming languages spans several decades. FORTRAN was developed in the 1950s for scientific computing. COBOL emerged for business applications. C was created in the 1970s and became the foundation for operating systems. Object-oriented languages like C++ and Java followed. Modern languages like Python, Rust, and Go address specific needs in productivity, safety, and concurrency.",
        "Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the quantum state of one particle cannot be described independently. When a measurement is performed on one particle, it instantaneously affects the state of the other particle, regardless of the distance between them. This phenomenon was famously described by Einstein as spooky action at a distance.",
        "Machine learning models can be broadly categorized into supervised, unsupervised, and reinforcement learning. Supervised learning uses labeled data to learn a mapping from inputs to outputs. Unsupervised learning discovers hidden patterns in unlabeled data. Reinforcement learning optimizes a policy through interaction with an environment to maximize cumulative reward over time.",
    ],
    "random_tokens": [
        "PLACEHOLDER_RANDOM",
        "PLACEHOLDER_RANDOM",
        "PLACEHOLDER_RANDOM",
        "PLACEHOLDER_RANDOM",
        "PLACEHOLDER_RANDOM",
    ],
}


def get_decoder_hook(model):
    try:
        _, layers, n_layers, _ = find_decoder_layers(model)
    except RuntimeError:
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        _, layers, n_layers, _ = find_decoder_layers(base)
    return layers[-1]


@torch.no_grad()
def measure_pr_per_query(model, tokenizer, category: str, prompts: list[str]) -> dict:
    """Measure PR using hidden states from multiple prompts (last token only)."""
    device = next(model.parameters()).device
    model.eval()
    last_layer = get_decoder_hook(model)
    captures = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures.append(h[:, -1, :].detach().float().cpu().numpy())

    handle = last_layer.register_forward_hook(hook_fn)
    try:
        for prompt in prompts:
            if prompt == "PLACEHOLDER_RANDOM":
                ids = torch.randint(100, 30000, (1, 64), device=device)
                model(input_ids=ids)
            else:
                enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=256).to(device)
                model(input_ids=enc["input_ids"])
    finally:
        handle.remove()

    if len(captures) < 3:
        return {"pr": 0.0, "n_samples": len(captures)}
    mat = np.vstack(captures)
    ls = run_pca_layer(mat)
    pr = ls.pr if ls else 0.0
    return {"pr": pr, "n_samples": len(captures)}


@torch.no_grad()
def measure_pr_all_positions(model, tokenizer, text: str, max_length: int = 256) -> dict:
    """Capture hidden states at ALL token positions for one input.
    Returns PR computed from the position×hidden matrix."""
    device = next(model.parameters()).device
    model.eval()
    last_layer = get_decoder_hook(model)
    all_hidden = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        all_hidden.append(h[0].detach().float().cpu().numpy())

    enc = tokenizer(text, return_tensors="pt", truncation=True,
                   max_length=max_length).to(device)
    seq_len = enc["input_ids"].shape[1]

    handle = last_layer.register_forward_hook(hook_fn)
    try:
        model(input_ids=enc["input_ids"])
    finally:
        handle.remove()

    if not all_hidden:
        return {"pr": 0.0, "seq_len": seq_len, "windowed_pr": []}

    mat = all_hidden[0]  # shape: (seq_len, hidden_dim)
    ls = run_pca_layer(mat)
    full_pr = ls.pr if ls else 0.0

    # Sliding window PR
    window_size = 32
    stride = 8
    windowed = []
    for start in range(0, mat.shape[0] - window_size + 1, stride):
        chunk = mat[start:start + window_size]
        ls_w = run_pca_layer(chunk)
        w_pr = ls_w.pr if ls_w else 0.0
        windowed.append({
            "start": start,
            "end": start + window_size,
            "pr": w_pr,
        })

    return {
        "pr_full": full_pr,
        "seq_len": int(seq_len),
        "windowed_pr": windowed,
    }


@torch.no_grad()
def measure_pr_per_token_generation(model, tokenizer, prompt: str,
                                     max_new_tokens: int = 200) -> dict:
    """Generate tokens and track PR at each step of generation."""
    device = next(model.parameters()).device
    model.eval()
    last_layer = get_decoder_hook(model)

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
    input_len = enc["input_ids"].shape[1]

    all_hidden = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        all_hidden.append(h[0].detach().float().cpu().numpy())

    handle = last_layer.register_forward_hook(hook_fn)
    try:
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        handle.remove()

    generated_tokens = out[0][input_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # all_hidden contains hidden states from each forward pass during generation
    # First call: all input positions. Subsequent calls: one position each (KV cache).
    # Collect the last-token hidden state from each step.
    last_token_hiddens = []
    for h in all_hidden:
        last_token_hiddens.append(h[-1:])

    if len(last_token_hiddens) < 10:
        return {"pr_full": 0.0, "generation_len": len(generated_tokens),
                "windowed_pr": [], "text": generated_text[:200]}

    mat = np.vstack(last_token_hiddens)

    ls = run_pca_layer(mat)
    full_pr = ls.pr if ls else 0.0

    # Sliding window over generation steps
    window_size = 20
    stride = 5
    windowed = []
    for start in range(0, mat.shape[0] - window_size + 1, stride):
        chunk = mat[start:start + window_size]
        ls_w = run_pca_layer(chunk)
        w_pr = ls_w.pr if ls_w else 0.0
        windowed.append({"start": start, "end": start + window_size, "pr": w_pr})

    return {
        "pr_full": full_pr,
        "generation_len": int(len(generated_tokens)),
        "input_len": int(input_len),
        "windowed_pr": windowed,
        "text": generated_text[:300],
    }


def load_checkpoint(step: int, model_size: str, gpu_id: int = 0):
    cfg = MODEL_CONFIGS[model_size]
    path = cfg["base_path"] if step == 0 else str(cfg["ckpt_base"] / f"step_{step}")
    log.info("Loading %s step %d from %s", model_size, step, path)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["1.7B", "0.6B"], required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    model_size = args.model_size

    log.info("=" * 60)
    log.info("Exp 7D: PR Drift — Qwen3-%s", model_size)
    log.info("Part 1: Per-Query PR (10 categories × 5 prompts)")
    log.info("Part 2: Per-Token PR (sliding window over positions)")
    log.info("=" * 60)

    test_steps = [0, 800]

    for step in test_steps:
        model, tok = load_checkpoint(step, model_size, args.gpu_id)

        # ── Part 1: Per-Query PR ─────────────────────────────────
        log.info("━" * 50)
        log.info("PART 1: Per-Query PR — step=%d", step)
        log.info("━" * 50)

        query_results = {}
        for cat_name, prompts in QUERY_CATEGORIES.items():
            result = measure_pr_per_query(model, tok, cat_name, prompts)
            query_results[cat_name] = result
            log.info("  %-15s → PR = %.2f (n=%d)", cat_name, result["pr"], result["n_samples"])

        prs = [v["pr"] for v in query_results.values()]
        log.info("  PR range: %.2f - %.2f (spread=%.2f)",
                 min(prs), max(prs), max(prs) - min(prs))

        # ── Part 2: Per-Token PR (prefill) ───────────────────────
        log.info("━" * 50)
        log.info("PART 2a: Per-Token PR (prefill positions) — step=%d", step)
        log.info("━" * 50)

        long_texts = [
            "The history of artificial intelligence began in the 1950s when Alan Turing proposed the famous Turing test. Since then, the field has gone through multiple winters and summers. Early AI focused on symbolic reasoning and expert systems. The 1980s saw the rise of neural networks, which fell out of favor before resurging in the 2010s with deep learning. Modern large language models represent the latest paradigm, achieving remarkable capabilities through scale. Transformers, introduced in 2017, revolutionized natural language processing by enabling parallel computation over sequences. The attention mechanism allows models to weigh the importance of different parts of the input when generating each output token.",
            "在量子力学的世界里，粒子可以同时处于多种状态的叠加。薛定谔的猫思想实验完美地说明了这一点：一只猫可以同时处于活着和死亡的状态，直到有人打开盒子观测它。量子纠缠是另一个令人惊叹的现象，两个粒子之间建立的关联可以跨越任何距离。爱因斯坦称之为幽灵般的超距作用。量子计算利用这些原理来进行传统计算机无法高效完成的运算。量子比特可以同时表示零和一，使得量子计算机在某些问题上具有指数级的优势。然而，量子退相干仍然是实现实用量子计算机的主要障碍之一。",
        ]

        position_results = []
        for i, text in enumerate(long_texts):
            result = measure_pr_all_positions(model, tok, text, max_length=256)
            position_results.append(result)
            log.info("  Text %d: seq_len=%d, PR_full=%.2f, %d windows",
                     i, result["seq_len"], result["pr_full"], len(result["windowed_pr"]))
            if result["windowed_pr"]:
                wpr = [w["pr"] for w in result["windowed_pr"]]
                log.info("    Window PR: min=%.2f max=%.2f mean=%.2f std=%.2f",
                         min(wpr), max(wpr), np.mean(wpr), np.std(wpr))

        # ── Part 2b: Per-Token PR (generation) ───────────────────
        log.info("━" * 50)
        log.info("PART 2b: Per-Token PR (generation steps) — step=%d", step)
        log.info("━" * 50)

        gen_prompts = [
            "Write a detailed explanation of how neural networks learn, starting from basic concepts.",
            "请详细解释分布式系统中的一致性问题，从基本概念开始。",
        ]

        generation_results = []
        for i, prompt in enumerate(gen_prompts):
            result = measure_pr_per_token_generation(model, tok, prompt, max_new_tokens=200)
            generation_results.append(result)
            log.info("  Prompt %d: gen_len=%d, PR_full=%.2f, %d windows",
                     i, result["generation_len"], result["pr_full"],
                     len(result["windowed_pr"]))
            if result["windowed_pr"]:
                wpr = [w["pr"] for w in result["windowed_pr"]]
                log.info("    Window PR: min=%.2f max=%.2f mean=%.2f std=%.2f",
                         min(wpr), max(wpr), np.mean(wpr), np.std(wpr))

        # Save
        entry = {
            "step": step,
            "model_size": model_size,
            "per_query": query_results,
            "per_token_prefill": [
                {"text_idx": i,
                 "seq_len": r["seq_len"],
                 "pr_full": r["pr_full"],
                 "windowed_pr": r["windowed_pr"]}
                for i, r in enumerate(position_results)
            ],
            "per_token_generation": [
                {"prompt_idx": i,
                 "input_len": r["input_len"],
                 "generation_len": r["generation_len"],
                 "pr_full": r["pr_full"],
                 "windowed_pr": r["windowed_pr"],
                 "text_preview": r["text"][:200]}
                for i, r in enumerate(generation_results)
            ],
        }

        out_file = RESULTS_DIR / f"drift_{model_size}_step{step}.json"
        with open(out_file, "w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        log.info("Saved: %s", out_file)

        del model, tok
        cleanup()

    elapsed = (time.time() - t0) / 60
    log.info("=== Exp 7D DONE in %.1f min ===", elapsed)


if __name__ == "__main__":
    main()
