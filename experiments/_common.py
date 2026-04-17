"""Shared infrastructure for SFP experiments.

Extracted from dpo_abc.py — SpectralProbeCallback, SpectralDPOTrainer,
model/data/config helpers used by exp2, exp3, exp4.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
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
)
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spectral_flow_probe.core import run_pca_layer
from spectral_flow_probe._compat import find_decoder_layers
from spectral_flow_probe.regularizer import spectral_pr_loss

HF_MIRROR = "https://hf-mirror.com"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"

log = logging.getLogger("sfp_exp")


def setup_logging(name: str = "sfp_exp"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ── SpectralProbeCallback ───────────────────────────────────────

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
            log.warning("[%s] PR(last)=%.2f < halt=%.2f -> EARLY STOP",
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


# ── SpectralDPOTrainer ──────────────────────────────────────────

class SpectralDPOTrainer(DPOTrainer):
    """DPOTrainer + spectral_pr_loss on last-layer hidden states."""

    def __init__(self, *args, pr_lambda: float = 0.01,
                 pr_target: float = 7.0, pr_mode: str = "floor",
                 probe_every: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self.pr_lambda = pr_lambda
        self.pr_target = pr_target
        self.pr_mode = pr_mode
        self.probe_every = probe_every
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
            self._captured_hidden = h[:, -1, :]

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
            pr_loss_val = spectral_pr_loss(
                H, target_pr=self.pr_target, mode=self.pr_mode
            )

        total_loss = dpo_loss + self.pr_lambda * pr_loss_val

        if self.state.global_step % self.probe_every == 0:
            entry = {
                "step": self.state.global_step,
                "dpo_loss": dpo_loss.item(),
                "pr_loss": pr_loss_val.item(),
                "total_loss": total_loss.item(),
            }
            if pr_loss_val.requires_grad and pr_loss_val.grad_fn is not None:
                entry["pr_loss_has_grad"] = True
            else:
                entry["pr_loss_has_grad"] = pr_loss_val.item() > 0
            self.pr_loss_history.append(entry)

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ── Model / Data / Config helpers ───────────────────────────────

FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


def load_base_model(model_path: str, gpu_id: int = 0):
    log.info("Loading model: %s on GPU %d", model_path, gpu_id)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        log.info("No chat_template found — setting fallback template")
        tok.chat_template = FALLBACK_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": gpu_id},
    )
    return model, tok


def apply_lora(model, r: int = 16, alpha: int = 32,
               targets: list[str] | None = None):
    targets = targets or ["q_proj", "v_proj"]
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_dpo_data(n_train: int = 5000, n_eval: int = 500):
    log.info("Loading dataset: %s (via %s)", DATASET_NAME, HF_MIRROR)
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    ds = load_dataset(DATASET_NAME, split="train")

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
    train_ds = ds.select(range(min(n_train, len(ds))))
    eval_ds = ds.select(range(n_train, min(n_train + n_eval, len(ds))))
    log.info("Train: %d, Eval: %d", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


def load_kto_data(n_train: int = 5000, n_eval: int = 500):
    """Load data in KTO format: prompt + completion + label (bool)."""
    log.info("Loading KTO dataset: %s (via %s)", DATASET_NAME, HF_MIRROR)
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    ds = load_dataset(DATASET_NAME, split="train")

    rows = []
    for ex in ds:
        prompt = ex["prompt"]
        chosen_raw = ex["chosen"]
        rejected_raw = ex["rejected"]
        c_text = chosen_raw[-1]["content"] if isinstance(chosen_raw, list) and chosen_raw else str(chosen_raw)
        r_text = rejected_raw[-1]["content"] if isinstance(rejected_raw, list) and rejected_raw else str(rejected_raw)
        rows.append({
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": c_text}],
            "label": True,
        })
        rows.append({
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": r_text}],
            "label": False,
        })

    from datasets import Dataset
    kto_ds = Dataset.from_list(rows).shuffle(seed=42)
    n_total = min(n_train * 2, len(kto_ds))
    n_eval_total = min(n_eval * 2, len(kto_ds) - n_total)
    train_ds = kto_ds.select(range(n_total))
    eval_ds = kto_ds.select(range(n_total, n_total + n_eval_total))
    log.info("KTO Train: %d, Eval: %d", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


def load_grpo_data(n_train: int = 5000):
    """Load prompts for GRPO (only needs prompt column)."""
    log.info("Loading GRPO prompts: %s (via %s)", DATASET_NAME, HF_MIRROR)
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    ds = load_dataset(DATASET_NAME, split="train")

    def extract_prompt(example):
        return {"prompt": [{"role": "user", "content": example["prompt"]}]}

    ds = ds.map(extract_prompt, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=42)
    train_ds = ds.select(range(min(n_train, len(ds))))
    log.info("GRPO prompts: %d", len(train_ds))
    return train_ds


def make_dpo_config(run_name: str, output_dir: str,
                    max_steps: int = 500, batch_size: int = 1,
                    grad_accum: int = 8, lr: float = 5e-7,
                    beta: float = 0.1, max_length: int = 384) -> DPOConfig:
    return DPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        max_length=max_length,
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


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def save_result(data: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

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
