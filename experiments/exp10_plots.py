#!/usr/bin/env python3
"""
Exp 10 Plots: Principal Angle Distributions
============================================
Generates:
    - angles_per_component.png      : bar chart of median angle by component, 4 pairs
    - angles_per_layer.png          : per-layer median angle heatmap across pairs
    - angles_distribution.png       : violin/box plot of full angle distribution per pair
    - angles_vs_weight_change.png   : scatter (ΔW%, angle°), shows scaling with training strength
    - angles_vs_spectrum.png        : scatter (SVD shift %, angle°), shows 'rotation without stretch'
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

for p in ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    try:
        font_manager.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


RESULTS = Path(__file__).resolve().parent / "results" / "exp10_principal_angles"
OUT_PLOTS = RESULTS / "plots"
OUT_PLOTS.mkdir(exist_ok=True, parents=True)


PAIR_ORDER = [
    ("Null_Qwen3_1.7B_DPO800", "Qwen3-1.7B DPO800\n(null)", "#9ca3af"),
    ("Qwen2.5_7B", "Qwen2.5-7B\nbase→Instruct", "#3498db"),
    ("Mistral_7B", "Mistral-7B\nbase→Instruct", "#9b59b6"),
    ("Yi_1.5_6B", "Yi-1.5-6B\nbase→Chat", "#27ae60"),
]

COMP_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj", "lm_head"]


def load_all() -> dict[str, dict]:
    data = {}
    for pid, _, _ in PAIR_ORDER:
        p = RESULTS / f"angles_{pid}.json"
        if p.exists():
            data[pid] = json.loads(p.read_text())
    return data


def plot_per_component(data: dict):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Principal Angle Rotation by Component — top-32 singular subspaces",
                 fontsize=15, fontweight="bold", y=1.02)

    width = 0.22
    x = np.arange(len(COMP_ORDER))

    for i, (pid, label, color) in enumerate(PAIR_ORDER):
        if pid not in data:
            continue
        angles = data[pid].get("angle_summary", {})
        vals = [angles.get(c, {}).get("mean_median_angle_deg", 0.0) for c in COMP_ORDER]
        errs = [angles.get(c, {}).get("std_median_angle_deg", 0.0) for c in COMP_ORDER]
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=3,
               label=label.replace("\n", " "), color=color, edgecolor="black",
               linewidth=0.8, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(COMP_ORDER, fontsize=10, rotation=15)
    ax.set_ylabel("Median principal angle (degrees)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Higher bar = more rotation on that component family across layers",
                 fontsize=11, pad=6, color="#555")

    plt.tight_layout()
    fig.savefig(OUT_PLOTS / "angles_per_component.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  saved angles_per_component.png")


def plot_per_layer(data: dict):
    """Per-layer median angle. Aggregates across attention and FFN component groups."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Per-layer Principal Angles — rotation profile across depth",
                 fontsize=15, fontweight="bold", y=1.00)

    for idx, (pid, label, color) in enumerate(PAIR_ORDER):
        ax = axes[idx // 2, idx % 2]
        if pid not in data:
            ax.set_title(f"{label}: (no data)")
            continue

        per = data[pid].get("per_matrix", [])
        if not per:
            ax.set_title(f"{label}: (no matrix data)")
            continue

        # Group by layer
        by_layer_attn: dict[int, list[float]] = {}
        by_layer_ffn: dict[int, list[float]] = {}
        for it in per:
            lidx = it.get("layer")
            if lidx is None:
                continue
            c = it.get("comp", "")
            angle = it.get("median_angle_deg", 0.0)
            if c in ("q_proj", "k_proj", "v_proj", "o_proj"):
                by_layer_attn.setdefault(lidx, []).append(angle)
            elif c in ("gate_proj", "up_proj", "down_proj"):
                by_layer_ffn.setdefault(lidx, []).append(angle)

        if not by_layer_attn:
            continue
        layers = sorted(set(list(by_layer_attn.keys()) + list(by_layer_ffn.keys())))
        attn_vals = [np.mean(by_layer_attn.get(l, [0.0])) for l in layers]
        ffn_vals = [np.mean(by_layer_ffn.get(l, [0.0])) for l in layers]

        ax.plot(layers, attn_vals, "o-", label="attention (q/k/v/o)", color="#e74c3c",
                linewidth=2, markersize=5)
        ax.plot(layers, ffn_vals, "s-", label="FFN (gate/up/down)", color="#3498db",
                linewidth=2, markersize=5)
        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel("Median angle (°)", fontsize=10)
        ax.set_title(label.replace("\n", " "), fontsize=11, fontweight="bold",
                     color=color)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUT_PLOTS / "angles_per_layer.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  saved angles_per_layer.png")


def plot_distribution(data: dict):
    """Box plot of all per-matrix median angles per pair."""
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Principal angle distribution — every weight matrix, every layer",
                 fontsize=15, fontweight="bold", y=1.02)

    labels = []
    data_lists = []
    colors = []
    for pid, label, color in PAIR_ORDER:
        if pid not in data:
            continue
        per = data[pid].get("per_matrix", [])
        vals = [it.get("median_angle_deg", 0.0) for it in per]
        labels.append(label)
        data_lists.append(vals)
        colors.append(color)

    bp = ax.boxplot(data_lists, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops=dict(color="black", linewidth=2),
                    meanprops=dict(color="red", linewidth=1.5, linestyle="--"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    ax.set_ylabel("Median principal angle per matrix (°)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(
        "Null baseline at bottom — each full-RLHF pair shows a distinct rotation band",
        fontsize=11, color="#555", pad=6,
    )

    plt.tight_layout()
    fig.savefig(OUT_PLOTS / "angles_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  saved angles_distribution.png")


def plot_scaling(data: dict):
    """(ΔW%, angle) scatter, shows rotation scales with training amount."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Principal angles scale with training intensity, not spectrum drift",
                 fontsize=15, fontweight="bold", y=1.02)

    # Aggregated per pair
    for pid, label, color in PAIR_ORDER:
        if pid not in data:
            continue
        d = data[pid]
        dw = d.get("global_rel_change_pct", d["global_rel_change"] * 100)
        pr = d.get("mean_svd_pr_shift_pct", 0)
        ang = d.get("mean_angle_deg", 0)
        axes[0].scatter(dw, ang, s=250, color=color, edgecolors="black",
                        linewidths=2, label=label.replace("\n", " "), zorder=5)
        axes[0].annotate(label.replace("\n", " "), (dw, ang),
                         textcoords="offset points", xytext=(10, 10), fontsize=9)
        axes[1].scatter(pr, ang, s=250, color=color, edgecolors="black",
                        linewidths=2, label=label.replace("\n", " "), zorder=5)
        axes[1].annotate(label.replace("\n", " "), (pr, ang),
                         textcoords="offset points", xytext=(10, 10), fontsize=9)

    axes[0].set_xlabel("Global ||ΔW|| / ||W||  (%)", fontsize=11)
    axes[0].set_ylabel("Mean principal angle (°)", fontsize=11)
    axes[0].set_title("Rotation scales with weight drift", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Mean SVD spectrum PR shift  (%)", fontsize=11)
    axes[1].set_ylabel("Mean principal angle (°)", fontsize=11)
    axes[1].set_title("But NOT with spectrum shift (≈ 0 for all)",
                      fontsize=11, fontweight="bold")
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUT_PLOTS / "angles_scaling.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  saved angles_scaling.png")


def make_summary_md(data: dict):
    """Markdown table summary."""
    lines = ["# Exp 10: Principal Angle Summary", ""]
    lines.append("| Pair | ΔW (%) | Σ PR shift (%) | Mean angle (°) |")
    lines.append("|---|---|---|---|")
    for pid, label, _ in PAIR_ORDER:
        if pid not in data:
            continue
        d = data[pid]
        lines.append(
            f"| {label.replace(chr(10), ' ')} "
            f"| {d.get('global_rel_change_pct', 0):.3f} "
            f"| {d.get('mean_svd_pr_shift_pct', 0):.4f} "
            f"| {d.get('mean_angle_deg', 0):.3f} |"
        )
    path = RESULTS / "summary.md"
    path.write_text("\n".join(lines))
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating Exp 10 plots...")
    data = load_all()
    print(f"  loaded {len(data)}/{len(PAIR_ORDER)} pairs")
    if not data:
        print("  no data yet!")
        exit(1)
    plot_per_component(data)
    plot_per_layer(data)
    plot_distribution(data)
    plot_scaling(data)
    make_summary_md(data)
    print("Done.")
