#!/usr/bin/env python3
"""
Worker for Exp 2: trains a single (model, method) combination and saves results.
Launched as a subprocess by exp2_universality.py.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure _common is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    setup_logging, load_base_model, apply_lora,
    load_dpo_data, load_kto_data, load_grpo_data,
    make_dpo_config, SpectralProbeCallback, cleanup, save_result,
)

import torch
from trl import DPOTrainer, DPOConfig


def measure_baseline_pr(model, n_samples: int = 30) -> float:
    """Quick PR measurement on the base model before training."""
    cb = SpectralProbeCallback(every_n=1, n_samples=n_samples, tag="baseline")
    from transformers import TrainerState, TrainerControl
    state = TrainerState()
    state.global_step = 1
    control = TrainerControl()
    cb.on_step_end(None, state, control, model=model)
    if cb.history:
        return cb.history[0].get("pr_last", 0.0)
    return 0.0


def run_dpo(model_path: str, model_name: str, gpu_id: int,
            max_steps: int, probe_every: int, output_dir: Path, log):
    log.info("=== DPO: %s on GPU %d ===", model_name, gpu_id)
    model, tok = load_base_model(model_path, gpu_id=0)  # CUDA_VISIBLE_DEVICES remaps
    baseline_pr = measure_baseline_pr(model)
    log.info("Baseline PR: %.2f", baseline_pr)

    model = apply_lora(model)
    train_ds, eval_ds = load_dpo_data(n_train=3000, n_eval=300)

    probe_cb = SpectralProbeCallback(
        every_n=probe_every, n_samples=30, tag=f"{model_name}_DPO"
    )
    config = make_dpo_config(
        f"{model_name}_DPO", str(output_dir / f"{model_name}_DPO"),
        max_steps=max_steps, batch_size=1, grad_accum=8,
        max_length=384,
    )
    trainer = DPOTrainer(
        model=model, args=config,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tok, callbacks=[probe_cb],
    )
    trainer.train()
    eval_result = trainer.evaluate()

    result = {
        "model": model_name, "method": "DPO",
        "baseline_pr": baseline_pr,
        "pr_history": probe_cb.history,
        "eval": eval_result,
        "final_step": trainer.state.global_step,
    }
    del trainer, model
    cleanup()
    return result


def run_kto(model_path: str, model_name: str, gpu_id: int,
            max_steps: int, probe_every: int, output_dir: Path, log):
    from trl import KTOConfig, KTOTrainer

    log.info("=== KTO: %s on GPU %d ===", model_name, gpu_id)
    model, tok = load_base_model(model_path, gpu_id=0)
    baseline_pr = measure_baseline_pr(model)
    log.info("Baseline PR: %.2f", baseline_pr)

    model = apply_lora(model)
    train_ds, eval_ds = load_kto_data(n_train=3000, n_eval=300)

    probe_cb = SpectralProbeCallback(
        every_n=probe_every, n_samples=30, tag=f"{model_name}_KTO"
    )
    kto_config = KTOConfig(
        output_dir=str(output_dir / f"{model_name}_KTO"),
        run_name=f"{model_name}_KTO",
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        beta=0.1,
        max_length=384,
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
    trainer = KTOTrainer(
        model=model, args=kto_config,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tok, callbacks=[probe_cb],
    )
    trainer.train()
    eval_result = trainer.evaluate()

    result = {
        "model": model_name, "method": "KTO",
        "baseline_pr": baseline_pr,
        "pr_history": probe_cb.history,
        "eval": eval_result,
        "final_step": trainer.state.global_step,
    }
    del trainer, model
    cleanup()
    return result


def run_grpo(model_path: str, model_name: str, gpu_id: int,
             max_steps: int, probe_every: int, output_dir: Path, log):
    from trl import GRPOConfig, GRPOTrainer

    log.info("=== GRPO: %s on GPU %d ===", model_name, gpu_id)
    model, tok = load_base_model(model_path, gpu_id=0)
    baseline_pr = measure_baseline_pr(model)
    log.info("Baseline PR: %.2f", baseline_pr)

    model = apply_lora(model)
    train_ds = load_grpo_data(n_train=3000)

    probe_cb = SpectralProbeCallback(
        every_n=probe_every, n_samples=30, tag=f"{model_name}_GRPO"
    )

    def reward_length_diversity(completions, **kwargs):
        """Simple rule-based reward: length + vocabulary diversity."""
        scores = []
        for c in completions:
            text = c[0]["content"] if isinstance(c, list) else str(c)
            tokens = text.split()
            length_score = min(len(tokens) / 50.0, 1.0)
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            scores.append(0.5 * length_score + 0.5 * unique_ratio)
        return scores

    grpo_config = GRPOConfig(
        output_dir=str(output_dir / f"{model_name}_GRPO"),
        run_name=f"{model_name}_GRPO",
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        num_generations=4,
        max_completion_length=128,
        bf16=True,
        logging_steps=10,
        save_steps=9999,
        report_to="none",
        gradient_checkpointing=False,
        seed=42,
    )
    trainer = GRPOTrainer(
        model=model, args=grpo_config,
        train_dataset=train_ds,
        reward_funcs=reward_length_diversity,
        processing_class=tok,
        callbacks=[probe_cb],
    )
    trainer.train()

    result = {
        "model": model_name, "method": "GRPO",
        "baseline_pr": baseline_pr,
        "pr_history": probe_cb.history,
        "final_step": trainer.state.global_step,
    }
    del trainer, model
    cleanup()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--method", required=True, choices=["DPO", "KTO", "GRPO"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--probe_every", type=int, default=25)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    log = setup_logging(f"exp2_{args.model_name}_{args.method}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tag = f"{args.model_name}_{args.method}"

    try:
        if args.method == "DPO":
            result = run_dpo(args.model_path, args.model_name, args.gpu_id,
                             args.max_steps, args.probe_every, output_dir, log)
        elif args.method == "KTO":
            result = run_kto(args.model_path, args.model_name, args.gpu_id,
                             args.max_steps, args.probe_every, output_dir, log)
        elif args.method == "GRPO":
            result = run_grpo(args.model_path, args.model_name, args.gpu_id,
                              args.max_steps, args.probe_every, output_dir, log)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        result["elapsed_min"] = (time.time() - t0) / 60
        result["status"] = "OK"

    except Exception as e:
        log.error("FAILED: %s", e, exc_info=True)
        result = {
            "model": args.model_name, "method": args.method,
            "status": "FAILED", "error": str(e),
            "elapsed_min": (time.time() - t0) / 60,
        }

    save_result(result, output_dir / f"{tag}_result.json")
    log.info("[%s] Done in %.1f min — %s", tag, result["elapsed_min"], result["status"])


if __name__ == "__main__":
    main()
