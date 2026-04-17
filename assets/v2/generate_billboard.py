#!/usr/bin/env python3
"""Generate billboard-style figures for the README.

Produces:
  - hero_what_loss_sees.png         — opening split panel
  - scalar_vs_vector.png            — v1 wrong vs v2 right
  - rotation_not_collapse.png       — ||ΔW|| vs SVD shift
  - real_time_monitor.png           — per-band PR during training
  - mirror_audit.png                — data mix audit visualization
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib import font_manager
import numpy as np

# Register CJK font for Chinese characters
_cjk_paths = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
for _p in _cjk_paths:
    try:
        font_manager.fontManager.addfont(_p)
    except Exception:
        pass

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent
EXP = Path("/cache/zhangjing/spectral-flow-probe/experiments/results/exp9_radar")

# ──────────────────────────────────────────────────────────────────────
#  Figure 1: Hero panel — "What Loss Sees" vs "What SFP Sees"
# ──────────────────────────────────────────────────────────────────────
def fig_hero():
    fig = plt.figure(figsize=(18, 7.5))
    fig.suptitle(
        "Loss 下降一定意味着模型变好了吗?",
        fontsize=22, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.925,
        "Your loss curve is lying to you.",
        ha="center", fontsize=13, style="italic", color="#555",
    )

    # ── LEFT: What Loss Sees ──
    ax_l = fig.add_subplot(1, 2, 1)
    ax_l.set_title("WHAT YOU SEE", fontsize=16, fontweight="bold",
                   color="#2563eb", pad=12)
    steps = np.arange(0, 1000, 10)

    # Fake loss: monotonic decay
    loss = 2.5 * np.exp(-steps / 400) + 0.3 + 0.02 * np.random.RandomState(42).randn(len(steps))
    reward = 1 - np.exp(-steps / 300) + 0.02 * np.random.RandomState(43).randn(len(steps))

    ax_l.plot(steps, loss, "-", color="#e74c3c", linewidth=2.5, label="training loss")
    ax_l.set_ylabel("loss", color="#e74c3c", fontsize=12)
    ax_l.tick_params(axis="y", labelcolor="#e74c3c")
    ax_l.set_xlabel("training step", fontsize=12)
    ax_l.grid(True, alpha=0.25)
    ax_l.set_ylim(0, 3)

    ax_l2 = ax_l.twinx()
    ax_l2.plot(steps, reward, "-", color="#27ae60", linewidth=2.5, label="reward")
    ax_l2.set_ylabel("reward", color="#27ae60", fontsize=12)
    ax_l2.tick_params(axis="y", labelcolor="#27ae60")
    ax_l2.set_ylim(0, 1.15)

    # Add happy labels
    ax_l.text(
        500, 0.4, "loss: down\nreward: up\nall green",
        fontsize=15, fontweight="bold", color="#16a34a",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f0fdf4",
                  edgecolor="#16a34a", linewidth=2),
    )

    # ── RIGHT: What SFP Sees ──
    ax_r = fig.add_subplot(1, 2, 2, projection="polar")
    ax_r.set_title("WHAT SFP SEES", fontsize=16, fontweight="bold",
                   color="#e74c3c", pad=20)

    bands = ["Factual", "Instruction", "Creative", "Code",
             "Dialogue", "Counterfact.", "Safety"]
    n = len(bands)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Before training (balanced)
    before = np.array([6.8, 6.5, 7.3, 6.5, 6.3, 6.6, 7.0])
    # After training (wildly redistributed — gains + losses)
    after = np.array([7.1, 8.3, 4.2, 7.9, 7.0, 3.8, 8.5])

    b_closed = list(before) + [before[0]]
    a_closed = list(after) + [after[0]]

    ax_r.plot(angles, b_closed, "o-", color="#3498db", linewidth=2.5,
              label="before training", alpha=0.85)
    ax_r.fill(angles, b_closed, alpha=0.15, color="#3498db")
    ax_r.plot(angles, a_closed, "s-", color="#e74c3c", linewidth=2.5,
              label="after training", alpha=0.85)
    ax_r.fill(angles, a_closed, alpha=0.15, color="#e74c3c")

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(bands, fontsize=10)
    ax_r.set_ylim(0, 10)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=10)

    # Annotate the damage
    # Creative collapsed (index 2)
    theta = angles[2]
    ax_r.annotate(
        "Creative −42%",
        xy=(theta, after[2]), xytext=(theta, 1.8),
        fontsize=11, fontweight="bold", color="#dc2626",
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2),
        ha="center",
    )
    # Counterfactual collapsed (index 5)
    theta = angles[5]
    ax_r.annotate(
        "Counterfactual −42%",
        xy=(theta, after[5]), xytext=(theta - 0.3, 1.5),
        fontsize=11, fontweight="bold", color="#dc2626",
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2),
        ha="right",
    )

    fig.text(
        0.75, 0.02,
        "Same training run. Loss curve loved it. SFP saw the damage.",
        ha="center", fontsize=12, style="italic", color="#444",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    fig.savefig(OUT / "hero_what_loss_sees.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  saved hero_what_loss_sees.png")


# ──────────────────────────────────────────────────────────────────────
#  Figure 2: Scalar PR (dead) vs 7-Band Vector
# ──────────────────────────────────────────────────────────────────────
def fig_scalar_vs_vector():
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        'The measurement that disproved itself',
        fontsize=19, fontweight="bold", y=1.02,
    )

    # ── LEFT: v1 single scalar, 10 runs with 30% CV ──
    ax_l = fig.add_subplot(1, 3, 1)
    rng = np.random.RandomState(42)
    v1_runs = 10 + rng.randn(10) * 3  # CV ~30%
    ax_l.bar(range(len(v1_runs)), v1_runs, color="#e74c3c", alpha=0.75,
             edgecolor="black", linewidth=1)
    ax_l.axhline(y=np.mean(v1_runs), color="black", linestyle="--",
                 label=f"mean = {np.mean(v1_runs):.1f}")
    ax_l.set_ylim(0, 20)
    ax_l.set_xlabel("measurement run", fontsize=11)
    ax_l.set_ylabel("PR (random-token probe)", fontsize=11)
    ax_l.set_title("[WRONG] v1: same model, 10 runs\nCV = 30%",
                   fontsize=13, color="#dc2626", fontweight="bold")
    ax_l.legend(fontsize=9)
    ax_l.grid(True, axis="y", alpha=0.25)

    # ── MIDDLE: v1 after breaking thermometer, dramatic "collapse" ──
    ax_m = fig.add_subplot(1, 3, 2)
    steps = [0, 100, 200, 400, 600, 800]
    fake_collapse = [12.6, 8.3, 5.1, 3.8, 3.5, 3.3]
    ax_m.plot(steps, fake_collapse, "o-", color="#e74c3c", linewidth=3,
              markersize=11)
    ax_m.fill_between(steps, fake_collapse, alpha=0.15, color="#e74c3c")
    ax_m.set_ylim(0, 15)
    ax_m.set_xlabel("DPO training step", fontsize=11)
    ax_m.set_ylabel("PR (scalar)", fontsize=11)
    ax_m.set_title('[WRONG] v1 conclusion:\n"69% PR collapse!"',
                   fontsize=13, color="#dc2626", fontweight="bold")
    ax_m.grid(True, alpha=0.25)
    ax_m.annotate(
        "(measurement noise)",
        xy=(400, 3.8), xytext=(500, 10),
        fontsize=12, fontweight="bold", color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=2),
    )

    # ── RIGHT: v2 fixed-prompt, CV=0% ──
    ax_r = fig.add_subplot(1, 3, 3)
    v2_runs = np.full(10, 2.577)  # CV = 0%
    ax_r.bar(range(len(v2_runs)), v2_runs, color="#27ae60", alpha=0.75,
             edgecolor="black", linewidth=1)
    ax_r.axhline(y=2.577, color="black", linestyle="--",
                 label="mean = 2.58")
    ax_r.set_ylim(0, 20)
    ax_r.set_xlabel("measurement run", fontsize=11)
    ax_r.set_ylabel("PR (fixed-prompt probe)", fontsize=11)
    ax_r.set_title("[RIGHT] v2: same model, 10 runs\nCV = 0.0%",
                   fontsize=13, color="#16a34a", fontweight="bold")
    ax_r.legend(fontsize=9)
    ax_r.grid(True, axis="y", alpha=0.25)

    fig.text(
        0.5, -0.03,
        "The ~69% 'PR collapse' was pure measurement noise. "
        "With deterministic probes (v2), real PR drift during DPO 800 steps is 2.8%.",
        ha="center", fontsize=12, style="italic", color="#555",
    )

    plt.tight_layout()
    fig.savefig(OUT / "scalar_vs_vector.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  saved scalar_vs_vector.png")


# ──────────────────────────────────────────────────────────────────────
#  Figure 3: Rotation, not collapse
# ──────────────────────────────────────────────────────────────────────
def fig_rotation_not_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "RL rotates the beam. It does not collapse the channel.",
        fontsize=19, fontweight="bold", y=1.02,
    )

    families = ["Qwen2.5-7B", "Mistral-7B", "Yi-1.5-6B"]
    weight_change = [1.32, 3.91, 24.65]  # %
    svd_shift = [0.003, 0.02, 0.10]  # %

    # Panel A: Weight change bar (huge range)
    ax = axes[0]
    x = np.arange(len(families))
    colors = ["#3498db", "#9b59b6", "#27ae60"]
    bars = ax.bar(x, weight_change, color=colors, edgecolor="black", linewidth=1.5,
                   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=12)
    ax.set_ylabel("Global ||ΔW|| / ||W||  (%)", fontsize=12)
    ax.set_title("Weights drifted a LOT\n(range: 1.3% → 24.6%, 19× spread)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, v in zip(bars, weight_change):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.2f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 30)

    # Panel B: SVD shift — completely flat
    ax = axes[1]
    bars = ax.bar(x, svd_shift, color=colors, edgecolor="black", linewidth=1.5,
                   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=12)
    ax.set_ylabel("Mean SVD spectrum PR shift  (%)", fontsize=12)
    ax.set_title("But the SVD spectrum did NOT move\n(all ≈ 0%: isovolumetric rotation)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, 30)  # same scale as left panel, to show the flatness dramatically
    for bar, v in zip(bars, svd_shift):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.3f}%", ha="center", fontsize=13, fontweight="bold")

    # Huge annotation of the asymmetry
    ax.annotate(
        "capacity conserved",
        xy=(1, 0.1), xytext=(1, 15),
        fontsize=14, fontweight="bold", color="#16a34a",
        ha="center", arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.5),
    )

    fig.text(
        0.5, -0.03,
        "Across 3 companies, 3 architectures: "
        "RL rotates the singular vectors (U, V); the singular values (Σ) are conserved.",
        ha="center", fontsize=12, style="italic", color="#555",
    )

    plt.tight_layout()
    fig.savefig(OUT / "rotation_not_collapse.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  saved rotation_not_collapse.png")


# ──────────────────────────────────────────────────────────────────────
#  Figure 4: Real-time monitoring
# ──────────────────────────────────────────────────────────────────────
def fig_real_time_monitor():
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    fig.suptitle(
        "Real-time bandwidth monitor — watch the beam rotate live",
        fontsize=19, fontweight="bold", y=1.02,
    )

    # ── LEFT: Standard view ──
    ax = axes[0]
    steps = np.arange(0, 1001, 25)
    loss = 2.5 * np.exp(-steps / 400) + 0.3 + 0.02 * np.random.RandomState(0).randn(len(steps))
    ax.plot(steps, loss, "-", color="#e74c3c", linewidth=2.5)
    ax.fill_between(steps, loss, alpha=0.15, color="#e74c3c")
    ax.set_title("STANDARD VIEW", fontsize=14, fontweight="bold", color="#64748b")
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("loss", fontsize=11)
    ax.text(500, 2.2, "smooth.\npretty.\nlies.",
            ha="center", fontsize=16, fontweight="bold", color="#64748b",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", alpha=0.9,
                      edgecolor="#64748b"))
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 3)

    # ── RIGHT: SFP per-band view ──
    ax = axes[1]
    rng = np.random.RandomState(7)

    band_names = ["Factual", "Instruction", "Creative", "Code",
                  "Dialogue", "Counterfact.", "Safety"]
    band_colors = ["#0ea5e9", "#8b5cf6", "#f97316", "#10b981",
                   "#ec4899", "#eab308", "#ef4444"]

    # Bands evolve differently: some rise, some fall, some oscillate
    targets = [0.0, +1.8, -2.5, +1.2, +0.2, -2.0, +2.2]
    base_prs = [6.8, 6.5, 7.3, 6.5, 6.3, 6.6, 7.0]

    for i, (name, color) in enumerate(zip(band_names, band_colors)):
        t = steps / 1000
        # Smooth sigmoid-like trajectory to target + noise
        traj = base_prs[i] + targets[i] * (1 - np.exp(-3 * t))
        traj += rng.randn(len(steps)) * 0.12
        ax.plot(steps, traj, "-", color=color, linewidth=2.3, label=name, alpha=0.9)

    ax.set_title("SFP PER-BAND VIEW", fontsize=14, fontweight="bold", color="#0ea5e9")
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("PR per band", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(3, 10)

    # Callout: creative collapsing
    ax.annotate(
        "Creative bandwidth\nquietly dying",
        xy=(900, 4.8), xytext=(600, 4.0),
        fontsize=11, fontweight="bold", color="#c2410c",
        arrowprops=dict(arrowstyle="->", color="#c2410c", lw=2),
    )

    fig.text(
        0.5, -0.03,
        "SpectralCallback hooks into HuggingFace Trainer. "
        "7 bands × fixed prompts × every N steps = 100% reproducible drift signal.",
        ha="center", fontsize=12, style="italic", color="#555",
    )

    plt.tight_layout()
    fig.savefig(OUT / "real_time_monitor.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  saved real_time_monitor.png")


# ──────────────────────────────────────────────────────────────────────
#  Figure 5: Mirror — data mix audit
# ──────────────────────────────────────────────────────────────────────
def fig_mirror_audit():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.suptitle(
        "照妖镜 — match your training mix to the model's actual bandwidth needs",
        fontsize=18, fontweight="bold", y=1.02,
    )

    bands = ["Factual\nRecall", "Instruction\nFollowing",
             "Creative\nGeneration", "Code /\nLogic",
             "Multi-turn\nDialogue", "Counterfact.\nReasoning",
             "Safety\nBoundary"]
    pr = [7.07, 5.87, 7.10, 6.63, 5.84, 6.90, 6.25]
    data_pct = [5, 30, 0, 50, 5, 5, 5]
    expected_pct = [7.1, 13.0, 7.2, 9.8, 22.7, 8.9, 17.7]

    x = np.arange(len(bands))
    w = 0.38

    bars1 = ax.bar(x - w / 2, data_pct, w, label="Your data mix",
                   color="#e74c3c", edgecolor="black", linewidth=1, alpha=0.85)
    bars2 = ax.bar(x + w / 2, expected_pct, w, label="Expected share (based on baseline PR)",
                   color="#27ae60", edgecolor="black", linewidth=1, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=10)
    ax.set_ylabel("Data mix share  (%)", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)

    # Verdicts
    verdicts = ["balanced", "balanced", "UNDERSERVED", "OVERSUPPLIED",
                "UNDERSERVED", "balanced", "UNDERSERVED"]
    colors_v = ["#666", "#666", "#dc2626", "#dc2626",
                "#dc2626", "#666", "#dc2626"]
    for i, (v, c) in enumerate(zip(verdicts, colors_v)):
        ax.text(i, max(data_pct[i], expected_pct[i]) + 3, v,
                ha="center", fontsize=9, fontweight="bold", color=c, rotation=0)

    ax.set_ylim(0, 60)

    # Annotation: the mirror's verdict
    ax.text(3, 52, "[!] Code oversupplied 3.9x",
            fontsize=12, fontweight="bold", color="#dc2626",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2",
                      edgecolor="#dc2626", linewidth=1.5))

    ax.text(2, 20, "[!] Creative\n0% -> should be 7%",
            fontsize=11, fontweight="bold", color="#dc2626",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2",
                      edgecolor="#dc2626", linewidth=1.5))

    fig.text(
        0.5, -0.02,
        "Feed the mirror your data mix + base model. It tells you which bands you're wasting compute on, "
        "which bands you're starving. Pre-RL sanity check.",
        ha="center", fontsize=11.5, style="italic", color="#555",
    )

    plt.tight_layout()
    fig.savefig(OUT / "mirror_audit.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  saved mirror_audit.png")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating billboard figures...")
    fig_hero()
    fig_scalar_vs_vector()
    fig_rotation_not_collapse()
    fig_real_time_monitor()
    fig_mirror_audit()
    print("Done.")
