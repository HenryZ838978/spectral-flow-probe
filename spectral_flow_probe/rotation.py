"""RotationAnalyzer — diagnose RL as isovolumetric rotation.

Core finding (Exp 9B + Exp 10, universal across Qwen2.5, Mistral, Yi):

    When a base model is aligned (SFT + RLHF), its weights change
    substantially (Frobenius norm drift 1%-25%), BUT the singular value
    spectrum of each weight matrix is preserved (< 0.5% PR shift).

    What changes is the orientation of the left-singular subspace.
    Top-32 singular subspaces rotate by 0.9° (null DPO baseline) up to
    9.3° (heavy Yi RLHF), scaling monotonically with training intensity
    while Σ stays essentially unchanged.

    RL rotates U, V. It does not compress or expand Σ. This is an
    isovolumetric rotation — total channel capacity is conserved; the
    direction in which the channel points shifts.

The measurement-angle shift theorem (Exp 11):

    For any weight matrix W = U Σ V^T and fixed probe input Q:
        PR(W, Q) = PR(V Σ U^T Q)

    When Σ is preserved and U rotates (U -> U'), the observed PR under
    any fixed Q changes because U^T Q rotates to U'^T Q. The probe didn't
    move; the capacity didn't change; only the coefficients of Q in the
    singular basis did.

    Empirically (Exp 11, n=644 points across 3 families × 7 bands × ~30
    layers): per-layer rotation angle θ(L) is a significant predictor of
    |ΔPR(L, band)|, with global Pearson r = +0.28, p = 7e-13. All 7
    bands show positive correlation, strongest on instruction / code /
    creative / safety (r = 0.32 to 0.47).

Two modes:

    1. Single-model mode: analyze ONE model's internal spectrum structure
       — useful when you only have the instruct version (no base pair).

    2. Pair mode: compare base vs instruct weights to measure rotation
       magnitude (Frobenius, Σ shift, principal angles) and verify that
       capacity is conserved.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("sfp")

__all__ = ["RotationAnalyzer", "RotationReport", "SpectrumProfile"]


ATTN_COMPONENTS = {"q_proj", "k_proj", "v_proj", "o_proj"}
FFN_COMPONENTS = {"gate_proj", "up_proj", "down_proj"}
INTERESTING = ATTN_COMPONENTS | FFN_COMPONENTS | {"lm_head"}


# ═══════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class SpectrumProfile:
    """Single-model spectrum summary — no pair needed."""
    model_path: str
    per_component: dict[str, dict] = field(default_factory=dict)
    per_layer: dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "per_component": self.per_component,
            "per_layer": {str(k): v for k, v in self.per_layer.items()},
        }

    def summary(self) -> str:
        lines = [f"Spectrum Profile: {self.model_path}", ""]
        lines.append(f"  {'Component':<15s}  {'mean PR':>10s}  {'mean σ_max':>12s}  {'n':>4s}")
        lines.append(f"  {'-' * 50}")
        for comp, s in sorted(self.per_component.items()):
            lines.append(
                f"  {comp:<15s}  {s['mean_pr']:10.2f}  {s['mean_sigma_max']:12.4f}  {s['n']:4d}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class RotationReport:
    """Pair-mode report: base vs instruct weight comparison."""
    model_a_path: str
    model_b_path: str
    n_common_params: int

    global_rel_change: float  # ||ΔW|| / ||W||
    component_summary: dict[str, dict] = field(default_factory=dict)
    layer_summary: dict[int, dict] = field(default_factory=dict)
    svd_summary: dict[str, dict] = field(default_factory=dict)
    angle_summary: dict[str, dict] = field(default_factory=dict)
    per_matrix: list[dict] = field(default_factory=list)

    is_isovolumetric: bool = False
    isovolumetric_threshold: float = 0.5  # mean |PR shift %| < this → isovolumetric
    top_k_angles: int = 32

    elapsed_sec: float = 0.0

    @property
    def mean_svd_pr_shift_pct(self) -> float:
        """Mean absolute PR shift across all components (lower = more isovolumetric)."""
        if not self.svd_summary:
            return 0.0
        shifts = [abs(v.get("mean_pr_change_pct", 0.0)) for v in self.svd_summary.values()]
        return float(np.mean(shifts)) if shifts else 0.0

    @property
    def mean_angle_deg(self) -> float:
        """Mean principal angle (degrees) across all interesting matrices.

        0° = no rotation (U/V identical). 90° = maximally rotated subspaces.
        """
        if not self.angle_summary:
            return 0.0
        vals = [v.get("mean_median_angle_deg", 0.0) for v in self.angle_summary.values()]
        return float(np.mean(vals)) if vals else 0.0

    def verdict(self) -> str:
        """Plain-English verdict."""
        global_pct = self.global_rel_change * 100
        shift_pct = self.mean_svd_pr_shift_pct
        angle_deg = self.mean_angle_deg
        if self.is_isovolumetric:
            return (
                f"ISOVOLUMETRIC ROTATION CONFIRMED. "
                f"Weights drifted {global_pct:.2f}% (Frobenius), "
                f"SVD spectrum PR shift is only {shift_pct:.3f}%, "
                f"but top-{self.top_k_angles} singular subspaces rotated "
                f"{angle_deg:.2f}° on average. "
                f"Capacity conserved, beam rotated."
            )
        else:
            return (
                f"NON-ISOVOLUMETRIC: weights drifted {global_pct:.2f}%, "
                f"SVD spectrum PR shifted {shift_pct:.3f}% (above threshold), "
                f"subspaces rotated {angle_deg:.2f}°. "
                f"Capacity may have changed. Inspect svd_summary for details."
            )

    def to_dict(self) -> dict:
        return {
            "model_a_path": self.model_a_path,
            "model_b_path": self.model_b_path,
            "n_common_params": self.n_common_params,
            "global_rel_change": self.global_rel_change,
            "global_rel_change_pct": self.global_rel_change * 100,
            "component_summary": self.component_summary,
            "layer_summary": {str(k): v for k, v in self.layer_summary.items()},
            "svd_summary": self.svd_summary,
            "angle_summary": self.angle_summary,
            "top_k_angles": self.top_k_angles,
            "mean_svd_pr_shift_pct": self.mean_svd_pr_shift_pct,
            "mean_angle_deg": self.mean_angle_deg,
            "is_isovolumetric": self.is_isovolumetric,
            "verdict": self.verdict(),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "per_matrix": self.per_matrix,
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def summary(self) -> str:
        lines = [
            f"Rotation Analysis: {Path(self.model_a_path).name} → {Path(self.model_b_path).name}",
            "",
            f"  Global ||ΔW|| / ||W||:     {self.global_rel_change*100:.4f}%",
            f"  Mean SVD PR shift:          {self.mean_svd_pr_shift_pct:.4f}%",
            f"  Mean top-{self.top_k_angles} subspace angle:  {self.mean_angle_deg:.3f}°",
            f"  Isovolumetric:              {'YES ✓' if self.is_isovolumetric else 'NO ✗'}",
            "",
            f"  Component frobenius change:",
        ]
        for comp, s in sorted(self.component_summary.items(), key=lambda x: -x[1]["mean_rel_change"]):
            if s["mean_rel_change"] > 0.0001:
                lines.append(f"    {comp:<15s} mean={s['mean_rel_change']:.6f}  n={s['n_params']}")

        if self.svd_summary:
            lines.append("")
            lines.append(f"  SVD spectrum shift per component:")
            for comp, s in sorted(self.svd_summary.items(), key=lambda x: -x[1]["mean_spectral_diff"]):
                lines.append(
                    f"    {comp:<15s} spec_diff={s['mean_spectral_diff']:.6f}  "
                    f"PR shift={s['mean_pr_change_pct']:+.3f}%"
                )

        if self.angle_summary:
            lines.append("")
            lines.append(f"  Principal angle (top-{self.top_k_angles}) per component [degrees]:")
            for comp, s in sorted(self.angle_summary.items(), key=lambda x: -x[1]["mean_median_angle_deg"]):
                lines.append(
                    f"    {comp:<15s} median={s['mean_median_angle_deg']:6.2f}°  "
                    f"max={s['mean_max_angle_deg']:6.2f}°  min={s['mean_min_angle_deg']:6.2f}°"
                )

        lines.append("")
        lines.append(f"  Verdict: {self.verdict()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════
#  Analyzer
# ═══════════════════════════════════════════════════════════════

class RotationAnalyzer:
    """Weight-space analyzer for isovolumetric rotation.

    Usage — pair mode (base vs instruct):

        from spectral_flow_probe import RotationAnalyzer
        ra = RotationAnalyzer()
        report = ra.compare(
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            gpu_id=0,
        )
        print(report)

    Usage — single-model mode (no base pair available):

        profile = ra.profile("mistralai/Mistral-7B-Instruct-v0.1", gpu_id=0)
        print(profile)
    """

    def __init__(
        self,
        top_k_sv: int = 100,
        top_k_angles: int = 32,
        isovolumetric_threshold: float = 0.5,  # 0.5% absolute PR shift is a generous bound
    ):
        self.top_k_sv = top_k_sv
        self.top_k_angles = top_k_angles
        self.isovolumetric_threshold = isovolumetric_threshold

    # ─────────────────────────────────────────────────────────
    def compare(
        self,
        base_path: str,
        instruct_path: str,
        *,
        gpu_id: int | str = "cpu",
        verbose: bool = True,
    ) -> RotationReport:
        """Pair mode: compare two checkpoints' weight matrices."""
        t0 = time.time()
        device = f"cuda:{gpu_id}" if gpu_id != "cpu" and torch.cuda.is_available() else "cpu"

        if verbose:
            log.info("Loading %s", base_path)
        sd_a = _load_state_dict(base_path)
        if verbose:
            log.info("Loading %s", instruct_path)
        sd_b = _load_state_dict(instruct_path)

        common = sorted(set(sd_a) & set(sd_b))
        if verbose:
            log.info("  %d common parameters", len(common))

        by_component: dict[str, list[float]] = defaultdict(list)
        by_layer: dict[int, list[float]] = defaultdict(list)
        svd_items: list[dict] = []

        total_base_sq = 0.0
        total_diff_sq = 0.0

        for i, key in enumerate(common):
            wa = sd_a[key]
            wb = sd_b[key]
            if wa.shape != wb.shape:
                continue

            comp, layer_idx = _classify_param(key)

            base_norm = torch.norm(wa.float()).item()
            diff_norm = torch.norm((wb.float() - wa.float())).item()
            rel = diff_norm / base_norm if base_norm > 0 else 0.0

            total_base_sq += base_norm ** 2
            total_diff_sq += diff_norm ** 2

            by_component[comp].append(rel)
            if layer_idx is not None:
                by_layer[layer_idx].append(rel)

            if wa.dim() >= 2 and comp in INTERESTING and min(wa.shape) >= 64:
                wa_d = wa.float().to(device)
                wb_d = wb.float().to(device)
                k_ang = min(self.top_k_angles, min(wa.shape))
                try:
                    # Full SVD — we need U for principal angles
                    U_a, S_a, _Vh_a = torch.linalg.svd(wa_d, full_matrices=False)
                    U_b, S_b, _Vh_b = torch.linalg.svd(wb_d, full_matrices=False)

                    # Spectrum (reuse top_k_sv)
                    sv_a = S_a[: self.top_k_sv].cpu().numpy()
                    sv_b = S_b[: self.top_k_sv].cpu().numpy()

                    # Principal angles between top-k_ang left-singular subspaces
                    # cos(θ_i) = singular values of U_a[:, :k]^T @ U_b[:, :k]
                    Ua_k = U_a[:, :k_ang]
                    Ub_k = U_b[:, :k_ang]
                    cos_vals = torch.linalg.svdvals(Ua_k.T @ Ub_k)
                    cos_vals = torch.clamp(cos_vals, -1.0, 1.0)
                    angles_rad = torch.arccos(cos_vals).cpu().numpy()
                    angles_deg = angles_rad * 180.0 / np.pi
                finally:
                    del wa_d, wb_d
                    if 'U_a' in dir():
                        del U_a, U_b, S_a, S_b, _Vh_a, _Vh_b
                    if device != "cpu":
                        torch.cuda.empty_cache()

                k = min(len(sv_a), len(sv_b))
                if k == 0:
                    continue
                sv_a_n = sv_a[:k] / (sv_a[:k].sum() + 1e-12)
                sv_b_n = sv_b[:k] / (sv_b[:k].sum() + 1e-12)
                spec_diff = float(np.linalg.norm(sv_a_n - sv_b_n))
                pr_a = float((sv_a[:k].sum() ** 2) / ((sv_a[:k] ** 2).sum() + 1e-12))
                pr_b = float((sv_b[:k].sum() ** 2) / ((sv_b[:k] ** 2).sum() + 1e-12))
                svd_items.append({
                    "comp": comp, "layer": layer_idx,
                    "rel_frob": rel, "spectral_diff": spec_diff,
                    "pr_base": pr_a, "pr_instruct": pr_b,
                    "pr_change_pct": (pr_b - pr_a) / pr_a * 100 if pr_a > 0 else 0.0,
                    "principal_angles_deg": angles_deg.tolist(),
                    "median_angle_deg": float(np.median(angles_deg)),
                    "mean_angle_deg": float(np.mean(angles_deg)),
                    "min_angle_deg": float(np.min(angles_deg)),
                    "max_angle_deg": float(np.max(angles_deg)),
                })

            if verbose and (i + 1) % 50 == 0:
                log.info("  processed %d/%d params", i + 1, len(common))

        # Aggregate
        global_rel = (total_diff_sq ** 0.5) / (total_base_sq ** 0.5) if total_base_sq > 0 else 0.0

        component_summary = {
            c: {
                "mean_rel_change": float(np.mean(v)),
                "max_rel_change": float(np.max(v)),
                "std_rel_change": float(np.std(v)),
                "n_params": len(v),
            } for c, v in by_component.items()
        }
        layer_summary = {
            l: {
                "mean_rel_change": float(np.mean(v)),
                "max_rel_change": float(np.max(v)),
            } for l, v in by_layer.items()
        }

        svd_by_comp = defaultdict(list)
        for it in svd_items:
            svd_by_comp[it["comp"]].append(it)
        svd_summary = {
            c: {
                "mean_spectral_diff": float(np.mean([it["spectral_diff"] for it in items])),
                "mean_pr_change_pct": float(np.mean([it["pr_change_pct"] for it in items])),
                "mean_rel_frob": float(np.mean([it["rel_frob"] for it in items])),
                "n_matrices": len(items),
            } for c, items in svd_by_comp.items()
        }

        # Principal angle summary per component
        angle_summary = {
            c: {
                "mean_median_angle_deg": float(np.mean([it["median_angle_deg"] for it in items])),
                "mean_mean_angle_deg":   float(np.mean([it["mean_angle_deg"]   for it in items])),
                "mean_max_angle_deg":    float(np.mean([it["max_angle_deg"]    for it in items])),
                "mean_min_angle_deg":    float(np.mean([it["min_angle_deg"]    for it in items])),
                "std_median_angle_deg":  float(np.std( [it["median_angle_deg"] for it in items])),
                "n_matrices": len(items),
            } for c, items in svd_by_comp.items()
        }

        mean_shift = float(np.mean([abs(s["mean_pr_change_pct"]) for s in svd_summary.values()])) \
            if svd_summary else 0.0
        is_iso = mean_shift < self.isovolumetric_threshold

        return RotationReport(
            model_a_path=base_path,
            model_b_path=instruct_path,
            n_common_params=len(common),
            global_rel_change=global_rel,
            component_summary=component_summary,
            layer_summary=layer_summary,
            svd_summary=svd_summary,
            angle_summary=angle_summary,
            per_matrix=svd_items,
            is_isovolumetric=is_iso,
            isovolumetric_threshold=self.isovolumetric_threshold,
            top_k_angles=self.top_k_angles,
            elapsed_sec=time.time() - t0,
        )

    # ─────────────────────────────────────────────────────────
    def profile(
        self,
        model_path: str,
        *,
        gpu_id: int | str = "cpu",
        verbose: bool = True,
    ) -> SpectrumProfile:
        """Single-model mode: inspect one model's weight spectrum structure.

        Useful when you don't have a base/instruct pair (the common case for
        downstream users of a released model).
        """
        t0 = time.time()
        device = f"cuda:{gpu_id}" if gpu_id != "cpu" and torch.cuda.is_available() else "cpu"

        if verbose:
            log.info("Loading %s", model_path)
        sd = _load_state_dict(model_path)

        per_comp: dict[str, list[dict]] = defaultdict(list)

        for i, (key, w) in enumerate(sd.items()):
            if w.dim() < 2 or min(w.shape) < 64:
                continue
            comp, layer_idx = _classify_param(key)
            if comp not in INTERESTING:
                continue

            w_d = w.float().to(device)
            try:
                sv = torch.linalg.svdvals(w_d)[: self.top_k_sv].cpu().numpy()
            finally:
                del w_d
                if device != "cpu":
                    torch.cuda.empty_cache()

            if len(sv) == 0:
                continue
            pr = float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12))
            per_comp[comp].append({
                "layer": layer_idx,
                "pr": pr,
                "sigma_max": float(sv[0]),
                "sigma_mean": float(sv.mean()),
            })

            if verbose and (i + 1) % 50 == 0:
                log.info("  processed %d params", i + 1)

        per_component = {
            c: {
                "mean_pr": float(np.mean([it["pr"] for it in items])),
                "mean_sigma_max": float(np.mean([it["sigma_max"] for it in items])),
                "n": len(items),
            } for c, items in per_comp.items()
        }

        # Aggregate per-layer
        per_layer_items: dict[int, list[float]] = defaultdict(list)
        for comp, items in per_comp.items():
            for it in items:
                if it["layer"] is not None:
                    per_layer_items[it["layer"]].append(it["pr"])
        per_layer = {l: {"mean_pr": float(np.mean(v))} for l, v in per_layer_items.items()}

        profile = SpectrumProfile(
            model_path=model_path,
            per_component=per_component,
            per_layer=per_layer,
        )
        log.info("Profile done in %.1fs", time.time() - t0)
        return profile


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load all safetensors or pytorch bins from a local dir or HF cache path."""
    from safetensors.torch import load_file
    p = Path(path)
    if p.is_dir():
        sd = {}
        st_files = sorted(p.glob("*.safetensors"))
        if st_files:
            for f in st_files:
                sd.update(load_file(str(f), device="cpu"))
            return sd
        bin_files = sorted(p.glob("pytorch_model*.bin"))
        if bin_files:
            for f in bin_files:
                sd.update(torch.load(str(f), map_location="cpu", weights_only=True))
            return sd
        raise FileNotFoundError(f"No safetensors or .bin files in {path}")

    # Treat as HF repo ID — try to resolve via snapshot_download
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(path, allow_patterns=["*.safetensors", "*.bin",
                                                          "*.json", "*.model"])
        return _load_state_dict(local)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load state dict from {path}. "
            f"Pass a local directory or ensure HF access. Error: {e}"
        )


def _classify_param(name: str) -> tuple[str, int | None]:
    """Extract component type and layer index from a parameter name."""
    if "lm_head" in name:
        return "lm_head", None
    if "embed" in name:
        return "embed", None
    if "norm" in name and "layers" not in name:
        return "final_norm", None

    layer_idx = None
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass

    for tag in ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]:
        if tag in name:
            return tag, layer_idx

    if "norm" in name.lower():
        return "layer_norm", layer_idx
    return parts[-2] if len(parts) >= 2 else "unknown", layer_idx
