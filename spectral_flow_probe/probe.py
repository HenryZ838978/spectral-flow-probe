"""SpectralProbe — the Phased Array Radar scanner.

Input: a model.
Output: a BandwidthFingerprint — a 7-dimensional PR vector, one per functional band.

This is NOT a scalar-returning tool. PR is f(model, query). A single PR number
tells you nothing. The only meaningful measurement is a vector across a fixed,
shared set of query bands.

Usage:

    from spectral_flow_probe import SpectralProbe

    probe = SpectralProbe("meta-llama/Llama-3.1-8B-Instruct")
    fingerprint = probe.scan()
    print(fingerprint)            # human-readable radar
    fingerprint.to_json("out.json")
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from .bands import BANDS, BAND_KEYS
from .core import run_pca_layer
from .fingerprint import BandwidthFingerprint, BandResult
from ._compat import load_model, find_decoder_layers

log = logging.getLogger("sfp")

__all__ = ["SpectralProbe"]


class SpectralProbe:
    """Scan a model and return a BandwidthFingerprint.

    Args:
        model_path: Path or HF ID of the model.
        dtype: Model dtype. Defaults to bfloat16.
        device_map: HF device_map. Defaults to "auto".
        trust_remote_code: Passed to HF. Defaults to True.
        model, tokenizer: If already loaded, pass them directly and skip loading.
    """

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
        self.n_params_B = sum(p.numel() for p in self.model.parameters()) / 1e9

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def scan(
        self,
        *,
        depth_profile: bool = False,
        max_length: int = 512,
        progress: bool = True,
    ) -> BandwidthFingerprint:
        """Run a full 7-band radar scan.

        Args:
            depth_profile: If True, also measure PR per layer per band (slower).
            max_length: Max tokens per prompt (truncation limit).
            progress: Log each band as it completes.

        Returns:
            BandwidthFingerprint — a 7-dimensional PR vector with metadata.
        """
        device = next(self.model.parameters()).device
        t0 = time.time()
        band_results = []

        for band_key, band_cfg in BANDS.items():
            t1 = time.time()
            pr, n_samples, eigs, profile = self._scan_band(
                prompts=band_cfg["prompts"],
                device=device,
                max_length=max_length,
                depth_profile=depth_profile,
            )
            elapsed_band = time.time() - t1
            br = BandResult(
                band_key=band_key,
                name=band_cfg["name"],
                channel=band_cfg["channel"],
                pr=pr,
                n_samples=n_samples,
                top5_eigenvalues=eigs[:5] if eigs else [],
                depth_profile=profile,
            )
            band_results.append(br)
            if progress:
                log.info("  %s  PR=%.2f  (%.1fs)", band_cfg["name"], pr, elapsed_band)

        return BandwidthFingerprint(
            model_path=self.model_path,
            n_params_B=self.n_params_B,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            bands=band_results,
            elapsed_sec=time.time() - t0,
        )

    # ─────────────────────────────────────────────────────────
    def _scan_band(
        self,
        prompts: list[str],
        device: torch.device,
        max_length: int,
        depth_profile: bool,
    ) -> tuple[float, int, list[float], list[float] | None]:
        """Measure PR for a single band's prompt set."""
        last_layer = self.layers[-1]
        captures_last = []

        # Optional full-depth capture
        depth_captures: dict[int, list] = {i: [] for i in range(self.n_layers)} if depth_profile else {}

        def hook_last(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captures_last.append(h[:, -1, :].detach().float().cpu().numpy())

        def make_depth_hook(li):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                depth_captures[li].append(h[:, -1, :].detach().float().cpu().numpy())
            return fn

        handles = [last_layer.register_forward_hook(hook_last)]
        if depth_profile:
            for li, layer in enumerate(self.layers):
                if li == self.n_layers - 1:
                    continue
                handles.append(layer.register_forward_hook(make_depth_hook(li)))

        try:
            for prompt in prompts:
                enc = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=max_length, padding=False,
                ).to(device)
                self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                )
        finally:
            for h in handles:
                h.remove()

        if len(captures_last) < 3:
            return 0.0, len(captures_last), [], None

        mat = np.vstack(captures_last)
        ls = run_pca_layer(mat)
        pr_last = float(ls.pr) if ls else 0.0
        eigs = ls.eigenvalues.tolist() if ls and hasattr(ls, "eigenvalues") else []

        profile = None
        if depth_profile:
            profile = []
            for li in range(self.n_layers):
                if li == self.n_layers - 1:
                    profile.append(pr_last)
                    continue
                vecs = depth_captures.get(li, [])
                if len(vecs) < 3:
                    profile.append(0.0)
                    continue
                m = np.vstack(vecs)
                lsl = run_pca_layer(m)
                profile.append(float(lsl.pr) if lsl else 0.0)

        return pr_last, len(captures_last), eigs, profile
