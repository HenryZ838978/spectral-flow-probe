"""SpectralCallback — real-time spectral monitoring during training."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from .core import run_pca_layer

log = logging.getLogger("sfp")

__all__ = ["SpectralCallback"]


class SpectralCallback:
    """HuggingFace Trainer callback for spectral monitoring.

    Usage::

        from spectral_flow_probe import SpectralCallback

        cb = SpectralCallback(
            layer_indices=[-1],       # which layers to watch
            every_n_steps=50,
            pr_floor=3.0,             # warn if PR(last) drops below
            logger="wandb",           # "wandb" | "tensorboard" | None
        )
        trainer = Trainer(..., callbacks=[cb])
    """

    def __init__(
        self,
        layer_indices: list[int] | None = None,
        every_n_steps: int = 50,
        n_probe_tokens: int = 20,
        pr_floor: float | None = None,
        pr_halt: float | None = None,
        logger: str | None = None,
    ):
        self.layer_indices = layer_indices or [-1]
        self.every_n_steps = every_n_steps
        self.n_probe_tokens = n_probe_tokens
        self.pr_floor = pr_floor
        self.pr_halt = pr_halt
        self.logger_type = logger
        self.history: list[dict] = []
        self._layers = None
        self._logger = None

    def on_init_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._setup_layers(model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return
        if model is None:
            return
        if self._layers is None:
            self._setup_layers(model)

        metrics = self._measure(model)
        metrics["step"] = state.global_step
        self.history.append(metrics)

        self._log(metrics, state)

        if self.pr_halt is not None and metrics.get("pr_last", 999) < self.pr_halt:
            log.warning("PR(last)=%.2f < halt threshold %.2f — requesting stop",
                        metrics["pr_last"], self.pr_halt)
            control.should_training_stop = True

        if self.pr_floor is not None and metrics.get("pr_last", 999) < self.pr_floor:
            log.warning("PR(last)=%.2f < floor %.2f", metrics["pr_last"], self.pr_floor)

    def _setup_layers(self, model):
        from ._compat import find_decoder_layers
        _, layers, n_layers, _ = find_decoder_layers(model)
        resolved = []
        for idx in self.layer_indices:
            if idx < 0:
                idx = n_layers + idx
            if 0 <= idx < n_layers:
                resolved.append((idx, layers[idx]))
        self._layers = resolved

    @torch.no_grad()
    def _measure(self, model) -> dict:
        """Capture hidden states from random input and compute spectral metrics."""
        device = next(model.parameters()).device
        hs = getattr(model.config, "hidden_size", 768)
        rand_ids = torch.randint(100, 30000, (self.n_probe_tokens, 32), device=device)

        captures: dict[int, list] = {idx: [] for idx, _ in self._layers}
        hooks = []

        def _make_hook(li):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captures[li].append(h[:, -1, :].detach().float().cpu().numpy())
            return fn

        for li, layer in self._layers:
            hooks.append(layer.register_forward_hook(_make_hook(li)))

        try:
            for i in range(self.n_probe_tokens):
                model(input_ids=rand_ids[i:i+1])
        finally:
            for h in hooks:
                h.remove()

        result = {}
        for li, _ in self._layers:
            vecs = captures[li]
            if not vecs:
                continue
            mat = np.vstack(vecs)
            ls = run_pca_layer(mat)
            if ls:
                result[f"S_L{li}"] = ls.S
                result[f"PR_L{li}"] = ls.pr
                if li == self._layers[-1][0]:
                    result["pr_last"] = ls.pr
                    result["s_last"] = ls.S
        return result

    def _log(self, metrics: dict, state):
        if self.logger_type == "wandb":
            try:
                import wandb
                wandb.log({f"sfp/{k}": v for k, v in metrics.items()},
                          step=state.global_step)
            except ImportError:
                pass
        elif self.logger_type == "tensorboard":
            try:
                if self._logger is None:
                    from torch.utils.tensorboard import SummaryWriter
                    self._logger = SummaryWriter()
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self._logger.add_scalar(f"sfp/{k}", v, state.global_step)
            except ImportError:
                pass
