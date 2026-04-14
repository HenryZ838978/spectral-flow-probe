"""SpectralProbe — Auditor entry point.  20-min model diagnosis."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from .core import run_pca_layer
from ._compat import load_model, encode_prompt, find_decoder_layers
from .prompts import DEFAULT_PROMPTS
from .report import SpectralReport, LayerResult
from .moe import detect_moe, MoEInfo, measure_moe_routing_pr

log = logging.getLogger("sfp")

__all__ = ["SpectralProbe"]


class SpectralProbe:
    """Load a model and run spectral diagnosis in one call."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        model: Any = None,
        tokenizer: Any = None,
    ):
        self.model_path = model_path
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            _, layers, n_layers, hs = find_decoder_layers(model)
            self.layers = layers
            self.n_layers = n_layers
            self.hidden_size = hs
        else:
            self.model, self.tokenizer, self.layers, self.n_layers, self.hidden_size = \
                load_model(model_path, dtype=dtype, device_map=device_map,
                           trust_remote_code=trust_remote_code)
        self.n_params = sum(p.numel() for p in self.model.parameters()) / 1e9

    @property
    def default_prompts(self) -> list[str]:
        return DEFAULT_PROMPTS

    @torch.no_grad()
    def run(
        self,
        prompts: list[str] | None = None,
        *,
        n_components: int = 30,
        n_fit: int = 20,
        progress: bool = True,
        check_moe: bool = True,
    ) -> SpectralReport:
        """Run full spectral scan.  Returns a SpectralReport."""
        if prompts is None:
            prompts = DEFAULT_PROMPTS
        t0 = time.time()

        hidden_bank: dict[int, list[np.ndarray]] = {i: [] for i in range(self.n_layers)}
        tag = self.model_path.lower()

        for pi, prompt in enumerate(prompts):
            if progress and (pi + 1) % 10 == 0:
                log.info("  prompt %d/%d", pi + 1, len(prompts))
            enc = encode_prompt(self.tokenizer, prompt, model_tag=tag)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            hooks, captures = _install_hooks(self.layers, self.n_layers)
            try:
                self.model(**enc)
            finally:
                for h in hooks:
                    h.remove()

            for li in range(self.n_layers):
                if li in captures and captures[li] is not None:
                    hidden_bank[li].append(captures[li])

        layer_results: list[LayerResult] = []
        for li in range(self.n_layers):
            vecs = hidden_bank[li]
            if not vecs:
                layer_results.append(LayerResult(li, 0.0, 0.0, 0.0, 0.0, np.array([])))
                continue
            mat = np.vstack(vecs)
            ls = run_pca_layer(mat, n_components=n_components, n_fit=n_fit)
            if ls is None:
                layer_results.append(LayerResult(li, 0.0, 0.0, 0.0, 0.0, np.array([])))
            else:
                layer_results.append(LayerResult(
                    layer=li, S=ls.S, r2=ls.r2, pr=ls.pr,
                    pc01=ls.pc01, eigenvalues=ls.eigenvalues,
                ))

        moe_info = None
        if check_moe:
            moe_det = detect_moe(self.model)
            if moe_det is not None:
                moe_info = measure_moe_routing_pr(
                    self.model, self.tokenizer, self.layers,
                    prompts=prompts, n_fit=n_fit, model_tag=tag,
                )

        elapsed = time.time() - t0
        report = SpectralReport(
            model_path=self.model_path,
            n_params=self.n_params,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            n_prompts=len(prompts),
            layers=layer_results,
            moe=moe_info,
            elapsed_sec=elapsed,
        )
        log.info("Done in %.1fs — ΔS=%.4f, PR(last)=%.2f", elapsed,
                 report.delta_s, report.pr_last)
        return report


def _install_hooks(layers: list, n_layers: int):
    """Register forward hooks to capture last-token hidden state per layer."""
    captures: dict[int, np.ndarray | None] = {}
    hooks = []

    def _make_hook(li: int):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            vec = h[:, -1, :].detach().float().cpu().numpy()  # (1, d)
            captures[li] = vec
        return hook_fn

    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(_make_hook(li)))
    return hooks, captures
