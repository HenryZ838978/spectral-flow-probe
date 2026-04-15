#!/usr/bin/env python3
"""
Worker for Exp 4: single lambda run for the regularizer sweep.
Launched as a subprocess by exp4_regularizer.py.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    setup_logging, load_base_model, apply_lora, load_dpo_data,
    make_dpo_config, SpectralProbeCallback, SpectralDPOTrainer,
    cleanup, save_result,
)
from trl import DPOTrainer
from transformers import TrainerState, TrainerControl

MODEL_PATH = "/cache/zhangjing/models/Qwen3-1.7B"
MAX_STEPS = 500
PROBE_EVERY = 25
PR_TARGET = 7.0


def measure_baseline_pr(model, n_samples: int = 30) -> float:
    cb = SpectralProbeCallback(every_n=1, n_samples=n_samples, tag="baseline")
    state = TrainerState()
    state.global_step = 1
    cb.on_step_end(None, state, TrainerControl(), model=model)
    if cb.history:
        return cb.history[0].get("pr_last", 0.0)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr_lambda", type=float, required=True)
    parser.add_argument("--pr_mode", type=str, default="floor")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    log = setup_logging(f"exp4_{args.tag}")
    output_dir = Path(args.output_dir)
    t0 = time.time()

    log.info("=== Exp 4 Worker: lambda=%.3f, mode=%s, GPU=%d ===",
             args.pr_lambda, args.pr_mode, args.gpu_id)

    try:
        model, tok = load_base_model(MODEL_PATH, gpu_id=0)  # remapped via CUDA_VISIBLE_DEVICES
        baseline_pr = measure_baseline_pr(model)
        log.info("Baseline PR: %.2f", baseline_pr)

        model = apply_lora(model)
        train_ds, eval_ds = load_dpo_data(n_train=5000, n_eval=500)

        probe_cb = SpectralProbeCallback(
            every_n=PROBE_EVERY, n_samples=30, tag=args.tag
        )

        config = make_dpo_config(
            f"exp4_{args.tag}",
            str(output_dir / f"train_{args.tag}"),
            max_steps=MAX_STEPS, batch_size=1, grad_accum=8,
        )

        if args.pr_lambda > 0:
            trainer = SpectralDPOTrainer(
                model=model, args=config,
                train_dataset=train_ds, eval_dataset=eval_ds,
                processing_class=tok, callbacks=[probe_cb],
                pr_lambda=args.pr_lambda,
                pr_target=PR_TARGET,
                pr_mode=args.pr_mode,
                probe_every=PROBE_EVERY,
            )
        else:
            trainer = DPOTrainer(
                model=model, args=config,
                train_dataset=train_ds, eval_dataset=eval_ds,
                processing_class=tok, callbacks=[probe_cb],
            )

        trainer.train()
        eval_result = trainer.evaluate()

        pr_loss_history = []
        if hasattr(trainer, "pr_loss_history"):
            pr_loss_history = trainer.pr_loss_history

        result = {
            "pr_lambda": args.pr_lambda,
            "pr_mode": args.pr_mode,
            "pr_target": PR_TARGET,
            "model": MODEL_PATH,
            "baseline_pr": baseline_pr,
            "pr_history": probe_cb.history,
            "pr_loss_history": pr_loss_history,
            "eval": eval_result,
            "final_step": trainer.state.global_step,
            "elapsed_min": (time.time() - t0) / 60,
            "status": "OK",
        }

        # Compute aggregate stats
        if probe_cb.history:
            prs = [h.get("pr_last", 0) for h in probe_cb.history]
            result["mean_pr"] = float(sum(prs) / len(prs))
            result["min_pr"] = float(min(prs))
            result["max_pr"] = float(max(prs))
            result["final_pr"] = prs[-1]
            result["pr_std"] = float((sum((p - result["mean_pr"])**2 for p in prs) / len(prs))**0.5)

        if pr_loss_history:
            grad_flags = [h.get("pr_loss_has_grad", False) for h in pr_loss_history]
            result["grad_flow_pct"] = sum(1 for g in grad_flags if g) / max(len(grad_flags), 1)

        del trainer, model
        cleanup()

    except Exception as e:
        log.error("FAILED: %s", e, exc_info=True)
        result = {
            "pr_lambda": args.pr_lambda,
            "pr_mode": args.pr_mode,
            "status": "FAILED",
            "error": str(e),
            "elapsed_min": (time.time() - t0) / 60,
        }

    save_result(result, output_dir / f"{args.tag}_result.json")
    log.info("[%s] Done in %.1f min", args.tag, result["elapsed_min"])


if __name__ == "__main__":
    main()
