"""Model loading and architecture compatibility helpers."""
from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

log = logging.getLogger("sfp")

__all__ = ["find_decoder_layers", "load_model", "encode_prompt"]


def find_decoder_layers(model: Any) -> tuple[Any, list, int, int | None]:
    """Find decoder layer list in a HuggingFace model.

    Returns (parent_module, layers_list, n_layers, hidden_size).
    Handles: CausalLM, multimodal (language_model), MoE, and brute-force fallback.
    """
    candidates = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(("model.model.layers", model.model, model.model.layers))
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            candidates.append(("language_model.model.layers", lm.model, lm.model.layers))

    if not candidates:
        for name, mod in model.named_modules():
            if hasattr(mod, "__len__") and len(mod) > 5:
                cname = type(mod).__name__.lower()
                if "layer" in cname or "block" in cname or "decoder" in cname:
                    parent = model
                    for part in name.rsplit(".", 1)[:-1]:
                        parent = getattr(parent, part, parent)
                    candidates.append((name, parent, mod))

    if not candidates:
        raise RuntimeError(
            "Cannot find decoder layers. Supported: model.model.layers, "
            "language_model.model.layers, or any ModuleList with >5 elements."
        )

    path, parent, layers = candidates[0]
    n_layers = len(layers)
    hs = getattr(model.config, "hidden_size", None) or \
         getattr(getattr(model.config, "text_config", None), "hidden_size", None)
    log.info("Found %d decoder layers at %s (hidden=%s)", n_layers, path, hs)
    return parent, list(layers), n_layers, hs


def load_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    **kwargs,
) -> tuple[Any, Any, list, int, int | None]:
    """Load a HuggingFace model and return (model, tokenizer, layers, n_layers, hidden_size)."""
    log.info("Loading %s ...", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device_map,
        **kwargs,
    )
    model.eval()
    _, layers, n_layers, hs = find_decoder_layers(model)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("  %d layers, hidden=%s, %.2fB params", n_layers, hs, n_params)
    return model, tok, layers, n_layers, hs


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    model_tag: str = "",
    max_length: int = 512,
    thinking: bool = False,
) -> dict[str, torch.Tensor]:
    """Tokenize a prompt using chat template if available, else raw."""
    try:
        msgs = [{"role": "user", "content": prompt}]
        kwargs = {}
        if "qwen3" in model_tag.lower():
            kwargs["enable_thinking"] = thinking
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kwargs,
        )
    except Exception:
        text = prompt
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
