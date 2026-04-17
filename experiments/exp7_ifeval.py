#!/usr/bin/env python3
"""
Exp 7A: IFEval — OOD Probe for PR Collapse Detection
=====================================================
IFEval tests instruction-following (formatting constraints, structural
requirements). This requires the model to actively use representation
bandwidth — not recall memorized answers.

Hypothesis: PR-collapsed models will fail more formatting constraints
because they lack the geometric capacity to simultaneously maintain
content generation + format compliance.

Each script instance handles ONE model size on ONE GPU.
Pass --model-size and --gpu-id as arguments.

Usage:
    CUDA_VISIBLE_DEVICES=4 python exp7_ifeval.py --model-size 1.7B --gpu-id 0
    CUDA_VISIBLE_DEVICES=5 python exp7_ifeval.py --model-size 0.6B --gpu-id 0
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

log = setup_logging("exp7_ifeval")

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


def run_ifeval(model, tokenizer, step: int, model_size: str) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    log.info("[IFEval] Step %d (%s): starting evaluation...", step, model_size)

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["ifeval"],
        batch_size=4,
        log_samples=False,
    )

    scores = {}
    if "results" in results:
        for task_name, task_res in results["results"].items():
            for key, val in task_res.items():
                if isinstance(val, (int, float)):
                    scores[f"{task_name}/{key}"] = val

    log.info("[IFEval] Step %d (%s): %s", step, model_size, scores)
    return scores


def load_checkpoint(step: int, model_size: str, gpu_id: int = 0):
    cfg = MODEL_CONFIGS[model_size]
    if step == 0:
        path = cfg["base_path"]
    else:
        path = str(cfg["ckpt_base"] / f"step_{step}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["1.7B", "0.6B"], required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    model_size = args.model_size
    gpu_id = args.gpu_id

    log.info("=" * 60)
    log.info("Exp 7A: IFEval OOD Probe — Qwen3-%s on GPU %d", model_size, gpu_id)
    log.info("Checkpoints: %s", CHECKPOINT_STEPS)
    log.info("=" * 60)

    all_results = []

    for step in CHECKPOINT_STEPS:
        log.info("━" * 50)
        log.info("CHECKPOINT step=%d (%s)", step, model_size)
        log.info("━" * 50)

        model, tok = load_checkpoint(step, model_size, gpu_id)
        pr = measure_pr(model)
        log.info("PR(last) = %.2f", pr)

        ifeval_scores = run_ifeval(model, tok, step, model_size)

        entry = {
            "step": step,
            "model_size": model_size,
            "pr": pr,
            "ifeval": ifeval_scores,
        }
        all_results.append(entry)

        result_file = RESULTS_DIR / f"ifeval_{model_size}_step{step}.json"
        with open(result_file, "w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        log.info("Saved: %s", result_file)

        del model, tok
        cleanup()

    # Correlation analysis
    from scipy import stats
    prs = [r["pr"] for r in all_results]
    correlations = {}

    ifeval_keys = set()
    for r in all_results:
        ifeval_keys.update(r["ifeval"].keys())

    for key in sorted(ifeval_keys):
        vals = [r["ifeval"].get(key, 0) for r in all_results]
        if len(vals) >= 3 and any(v != vals[0] for v in vals):
            rho, p = stats.spearmanr(prs, vals)
            correlations[f"pr_vs_{key}"] = {"rho": float(rho), "pvalue": float(p)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            log.info("  PR vs %s: rho=%.3f p=%.4f %s", key, rho, p, sig)

    summary = {
        "experiment": f"Exp 7A: IFEval OOD Probe — Qwen3-{model_size}",
        "model_size": model_size,
        "checkpoints": all_results,
        "correlations": correlations,
        "elapsed_min": (time.time() - t0) / 60,
    }

    summary_file = RESULTS_DIR / f"ifeval_{model_size}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Saved: %s", summary_file)

    elapsed = (time.time() - t0) / 60
    log.info("=== Exp 7A IFEval (%s) DONE in %.1f min ===", model_size, elapsed)

    for r in all_results:
        prompt_strict = r["ifeval"].get("ifeval/prompt_level_strict_acc,none", "?")
        inst_strict = r["ifeval"].get("ifeval/inst_level_strict_acc,none", "?")
        log.info("  Step %d | PR=%.2f | prompt_strict=%.4f | inst_strict=%.4f",
                 r["step"], r["pr"],
                 prompt_strict if isinstance(prompt_strict, float) else 0,
                 inst_strict if isinstance(inst_strict, float) else 0)


if __name__ == "__main__":
    main()
