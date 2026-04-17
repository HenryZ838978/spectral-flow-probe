#!/usr/bin/env python3
"""
Exp 7B: LiveCodeBench Easy — Contamination-Free OOD Probe
=========================================================
43 easy coding problems from Jan-Apr 2025 (post Qwen3 training cutoff).
IMPOSSIBLE to be in pretraining data.

For each checkpoint: generate Python solution, execute against test cases,
report pass@1.

Usage:
    CUDA_VISIBLE_DEVICES=6 python exp7_livecode.py --model-size 1.7B --gpu-id 0
    CUDA_VISIBLE_DEVICES=7 python exp7_livecode.py --model-size 0.6B --gpu-id 0
"""
import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
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

log = setup_logging("exp7_livecode")

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

EXEC_TIMEOUT = 10


def load_livecode_problems():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id='livecodebench/code_generation_lite',
        filename='test6.jsonl', repo_type='dataset'
    )
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    easy = [p for p in problems if p['difficulty'] == 'easy']
    easy_medium = [p for p in problems if p['difficulty'] in ('easy', 'medium')]
    log.info("Loaded %d total problems, %d easy, %d easy+medium",
             len(problems), len(easy), len(easy_medium))
    return easy, easy_medium


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code = parts[1]
            if code.startswith("\n"):
                code = code[1:]
            code = code.split("```")[0]
            return code.strip()
    lines = []
    for line in response.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("Output"):
            if any(kw in stripped for kw in ["def ", "import ", "class ", "for ", "while ",
                                              "if ", "print(", "input(", "=", "return "]):
                lines.append(line)
            elif lines:
                lines.append(line)
    return "\n".join(lines).strip()


def execute_code(code: str, stdin_input: str) -> tuple[bool, str]:
    """Execute Python code with given stdin, return (success, stdout)."""
    if not code.strip():
        return False, "empty code"
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            input=stdin_input, capture_output=True, text=True,
            timeout=EXEC_TIMEOUT,
        )
        os.unlink(tmp_path)
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return False, "timeout"
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return False, str(e)


def evaluate_problem(model, tokenizer, problem: dict) -> dict:
    """Generate code and test against public test cases."""
    title = problem["question_title"]
    content = problem["question_content"]
    starter = problem.get("starter_code", "")

    prompt = (
        f"Solve the following programming problem in Python.\n"
        f"Read input from stdin and print output to stdout.\n\n"
        f"Problem: {title}\n\n{content}\n"
    )
    if starter:
        prompt += f"\nStarter code:\n```python\n{starter}\n```\n"
    prompt += "\nWrite a complete Python solution:\n```python\n"

    device = next(model.parameters()).device
    try:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        text = prompt

    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=2048).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=1024,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[0][enc["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    code = extract_code(response)

    tc_str = problem.get("public_test_cases", "[]")
    if isinstance(tc_str, str):
        try:
            test_cases = json.loads(tc_str)
        except:
            test_cases = []
    else:
        test_cases = tc_str

    passed = 0
    total = len(test_cases)
    details = []

    for tc in test_cases:
        inp = tc.get("input", "")
        expected = tc.get("output", "").strip()
        success, actual = execute_code(code, inp)

        is_pass = success and actual.strip() == expected.strip()
        if is_pass:
            passed += 1
        details.append({
            "input": inp[:100],
            "expected": expected[:100],
            "actual": actual[:100],
            "pass": is_pass,
        })

    return {
        "title": title,
        "difficulty": problem["difficulty"],
        "date": problem.get("contest_date", ""),
        "code_generated": code[:500],
        "response_length": len(response),
        "test_total": total,
        "test_passed": passed,
        "pass_rate": passed / max(total, 1),
        "all_pass": passed == total and total > 0,
        "details": details,
    }


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
    from spectral_flow_probe.core import run_pca_layer
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["1.7B", "0.6B"], required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    model_size = args.model_size

    log.info("=" * 60)
    log.info("Exp 7B: LiveCodeBench Easy — Qwen3-%s", model_size)
    log.info("Post-cutoff problems: Jan-Apr 2025 (contamination-free)")
    log.info("Checkpoints: %s", CHECKPOINT_STEPS)
    log.info("=" * 60)

    easy_problems, easy_medium_problems = load_livecode_problems()
    problems = easy_problems
    log.info("Using %d easy problems for evaluation", len(problems))

    all_results = []

    for step in CHECKPOINT_STEPS:
        log.info("━" * 50)
        log.info("CHECKPOINT step=%d (%s)", step, model_size)
        log.info("━" * 50)

        model, tok = load_checkpoint(step, model_size, args.gpu_id)
        pr = measure_pr(model)
        log.info("PR(last) = %.2f", pr)

        problem_results = []
        for i, prob in enumerate(problems):
            log.info("  [%d/%d] %s", i+1, len(problems), prob["question_title"])
            res = evaluate_problem(model, tok, prob)
            problem_results.append(res)
            status = "PASS" if res["all_pass"] else f"FAIL({res['test_passed']}/{res['test_total']})"
            log.info("    → %s (code len=%d)", status, len(res["code_generated"]))

        n_all_pass = sum(1 for r in problem_results if r["all_pass"])
        avg_pass_rate = np.mean([r["pass_rate"] for r in problem_results])

        entry = {
            "step": step,
            "model_size": model_size,
            "pr": pr,
            "n_problems": len(problems),
            "n_all_pass": n_all_pass,
            "pass_at_1": n_all_pass / len(problems),
            "avg_pass_rate": float(avg_pass_rate),
            "problems": problem_results,
        }
        all_results.append(entry)

        log.info("Step %d SUMMARY: pass@1=%d/%d (%.1f%%), avg_pass_rate=%.3f",
                 step, n_all_pass, len(problems),
                 100 * n_all_pass / len(problems), avg_pass_rate)

        result_file = RESULTS_DIR / f"livecode_{model_size}_step{step}.json"
        with open(result_file, "w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        log.info("Saved: %s", result_file)

        del model, tok
        cleanup()

    # Correlation
    from scipy import stats
    prs = [r["pr"] for r in all_results]
    pass1s = [r["pass_at_1"] for r in all_results]
    avg_prs_rates = [r["avg_pass_rate"] for r in all_results]

    correlations = {}
    if len(prs) >= 3:
        for name, vals in [("pass_at_1", pass1s), ("avg_pass_rate", avg_prs_rates)]:
            if any(v != vals[0] for v in vals):
                rho, p = stats.spearmanr(prs, vals)
                correlations[f"pr_vs_{name}"] = {"rho": float(rho), "pvalue": float(p)}
                log.info("PR vs %s: rho=%.3f p=%.4f", name, rho, p)

    summary = {
        "experiment": f"Exp 7B: LiveCodeBench Easy — Qwen3-{model_size}",
        "model_size": model_size,
        "dataset": "livecodebench/code_generation_lite test6 (v6)",
        "date_range": "2025-01-04 to 2025-04-06 (all post Qwen3 cutoff)",
        "difficulty": "easy",
        "n_problems": len(problems),
        "checkpoints": [{
            "step": r["step"], "pr": r["pr"],
            "n_all_pass": r["n_all_pass"],
            "pass_at_1": r["pass_at_1"],
            "avg_pass_rate": r["avg_pass_rate"],
        } for r in all_results],
        "correlations": correlations,
        "elapsed_min": (time.time() - t0) / 60,
    }

    summary_file = RESULTS_DIR / f"livecode_{model_size}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Saved: %s", summary_file)

    elapsed = (time.time() - t0) / 60
    log.info("=== Exp 7B LiveCodeBench (%s) DONE in %.1f min ===", model_size, elapsed)
    for r in all_results:
        log.info("  Step %d | PR=%.2f | pass@1=%d/%d (%.1f%%)",
                 r["step"], r["pr"], r["n_all_pass"], r["n_problems"],
                 100 * r["pass_at_1"])


if __name__ == "__main__":
    main()
