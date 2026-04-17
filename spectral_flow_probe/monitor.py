"""SpectralCallback — real-time bandwidth monitoring during training.

⚠️  v2.0 breaking change:
    The v1 SpectralCallback used `torch.randint` to generate random tokens
    as probe inputs. This produced PR measurements with 30% CV across runs.
    Calibration experiments (Exp 7C) proved that "PR collapse" observed with
    random-token probes was measurement noise, NOT model change.

    v2 uses fixed natural-language prompts, grouped into 7 functional bands.
    Measurements are deterministic (CV = 0%) and reproducible across runs.

Usage:

    from spectral_flow_probe import SpectralCallback

    cb = SpectralCallback(
        every_n_steps=100,
        bands=["band2_instruction", "band4_code"],  # subset or None = all
        logger="wandb",
    )
    trainer = Trainer(..., callbacks=[cb])
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .bands import BANDS, BAND_KEYS
from .core import run_pca_layer

log = logging.getLogger("sfp")

__all__ = ["SpectralCallback"]


class SpectralCallback:
    """HuggingFace Trainer callback for fixed-prompt bandwidth monitoring.

    Args:
        every_n_steps: Run a scan every N training steps. Defaults to 100.
        bands: Which bands to measure. None = all 7. Pass a subset (e.g.,
            ["band2_instruction", "band7_safety"]) to speed up monitoring.
        n_prompts_per_band: How many prompts to use per band. Defaults to 5
            (half the full band, for speed). Use 10 for max stability.
        max_length: Prompt truncation length.
        logger: "wandb" | "tensorboard" | None.
        drift_threshold: Warn if any band's PR changes by more than this
            fraction between consecutive measurements.

    The v1 kwargs (pr_floor, pr_halt, layer_indices) are removed because
    they relied on scalar PR thresholds. PR is query-dependent — use
    per-band drift tracking instead.
    """

    def __init__(
        self,
        every_n_steps: int = 100,
        bands: list[str] | None = None,
        n_prompts_per_band: int = 5,
        max_length: int = 256,
        logger: str | None = None,
        drift_threshold: float | None = 0.10,
    ):
        self.every_n_steps = every_n_steps
        self.band_keys = bands if bands is not None else BAND_KEYS
        for bk in self.band_keys:
            if bk not in BANDS:
                raise ValueError(f"Unknown band: {bk}. Available: {BAND_KEYS}")
        self.n_prompts_per_band = n_prompts_per_band
        self.max_length = max_length
        self.logger_type = logger
        self.drift_threshold = drift_threshold

        self.history: list[dict] = []
        self._tokenizer = None
        self._layers = None
        self._tb_logger = None

    # ─── HF Trainer hooks ────────────────────────────────────
    def on_init_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if tokenizer is not None:
            self._tokenizer = tokenizer
        if model is not None:
            self._setup(model)

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.every_n_steps != 0:
            return
        if model is None:
            return
        if self._tokenizer is None and tokenizer is not None:
            self._tokenizer = tokenizer
        if self._tokenizer is None:
            log.warning("SpectralCallback: no tokenizer available; skipping scan")
            return
        if self._layers is None:
            self._setup(model)

        metrics = self._measure(model)
        metrics["step"] = state.global_step
        self.history.append(metrics)
        self._log(metrics, state)
        self._check_drift(metrics)

    # ─── Setup ───────────────────────────────────────────────
    def _setup(self, model):
        from ._compat import find_decoder_layers
        _, layers, n_layers, _ = find_decoder_layers(model)
        self._layers = layers
        if self._tokenizer is not None and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # ─── Measurement ────────────────────────────────────────
    @torch.no_grad()
    def _measure(self, model) -> dict:
        """Run the band scan. Returns per-band PR dict."""
        device = next(model.parameters()).device
        last_layer = self._layers[-1]

        result: dict[str, Any] = {}

        for band_key in self.band_keys:
            prompts = BANDS[band_key]["prompts"][: self.n_prompts_per_band]
            captures = []

            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captures.append(h[:, -1, :].detach().float().cpu().numpy())

            handle = last_layer.register_forward_hook(hook_fn)
            try:
                for prompt in prompts:
                    enc = self._tokenizer(
                        prompt, return_tensors="pt", truncation=True,
                        max_length=self.max_length, padding=False,
                    ).to(device)
                    model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                    )
            finally:
                handle.remove()

            if len(captures) >= 3:
                mat = np.vstack(captures)
                ls = run_pca_layer(mat)
                pr = float(ls.pr) if ls else 0.0
            else:
                pr = 0.0
            result[f"pr_{band_key}"] = pr

        # Aggregate stats for convenience
        prs = np.array([result[f"pr_{bk}"] for bk in self.band_keys])
        result["pr_mean"] = float(prs.mean())
        result["pr_std"] = float(prs.std())
        if prs.min() > 0:
            result["bandwidth_ratio"] = float(prs.max() / prs.min())

        return result

    # ─── Drift detection ────────────────────────────────────
    def _check_drift(self, metrics: dict) -> None:
        if self.drift_threshold is None or len(self.history) < 2:
            return
        prev = self.history[-2]
        for bk in self.band_keys:
            key = f"pr_{bk}"
            v_prev = prev.get(key, 0)
            v_now = metrics.get(key, 0)
            if v_prev > 0:
                drift = abs(v_now - v_prev) / v_prev
                if drift > self.drift_threshold:
                    log.warning(
                        "SpectralCallback: %s PR shifted %.1f%% (%.2f → %.2f) "
                        "at step %d. Bandwidth reallocation detected.",
                        BANDS[bk]["name"], drift * 100, v_prev, v_now,
                        metrics.get("step", -1),
                    )

    # ─── Logging ────────────────────────────────────────────
    def _log(self, metrics: dict, state) -> None:
        if self.logger_type == "wandb":
            try:
                import wandb
                wandb.log({f"sfp/{k}": v for k, v in metrics.items()},
                          step=state.global_step)
            except ImportError:
                pass
        elif self.logger_type == "tensorboard":
            try:
                if self._tb_logger is None:
                    from torch.utils.tensorboard import SummaryWriter
                    self._tb_logger = SummaryWriter()
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self._tb_logger.add_scalar(f"sfp/{k}", v, state.global_step)
            except ImportError:
                pass
