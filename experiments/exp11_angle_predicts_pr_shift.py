#!/usr/bin/env python3
"""
Exp 11: The Measurement-Angle Shift Theorem — empirical verification.
=====================================================================

Claim
-----
    Observed PR shifts between base and aligned models are caused by
    rotation of the weight singular vectors (U, V), NOT by capacity loss.
    Because Σ is conserved (Exp 8, 9B, 10), the only way a fixed probe
    query Q can see a different PR is through a change in U^T Q — i.e.,
    the projection of Q onto a rotated basis.

Testable prediction
-------------------
    At any given layer L, bigger rotation θ(L) should produce bigger
    PR shift |ΔPR(L, band)| when the same fixed band-prompt is used
    to probe both models.

Data (all pre-computed; no new GPU time):
    - Exp 9 radar (Qwen2.5-7B base vs Instruct), `depth_profiles`
      field: 7 bands × 28 layers of per-layer PR.
    - Exp 10 angles JSON: per-matrix principal angles with layer index.

Method:
    For each layer L:
        θ(L) = mean( median_angle_deg of all attention + FFN matrices at L )
    For each (L, band):
        ΔPR = PR_instruct(L, band) - PR_base(L, band)
    Scatter x = θ(L), y = |ΔPR(L, band)|.
    Compute Spearman + Pearson correlation.
    If monotonic positive correlation holds, the theorem is
    empirically supported.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from scipy import stats


for p in ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    try:
        font_manager.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


EXP = Path("/cache/zhangjing/spectral-flow-probe/experiments/results")
OUT = EXP / "exp11_theorem"
OUT.mkdir(exist_ok=True)
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)


# Component types that matter for the transformation at each layer
ATTN_COMPS = {"q_proj", "k_proj", "v_proj", "o_proj"}
FFN_COMPS = {"gate_proj", "up_proj", "down_proj"}
LAYER_COMPS = ATTN_COMPS | FFN_COMPS


FAMILIES = [
    {
        "name": "Qwen2.5-7B",
        "base": "radar_Qwen2.5-7B-base.json",
        "inst": "radar_Qwen2.5-7B-Instruct.json",
        "angles": "angles_Qwen2.5_7B.json",
        "color": "#3498db",
    },
    {
        "name": "Mistral-7B",
        "base": "radar_Mistral-7B-base.json",
        "inst": "radar_Mistral-7B-Instruct.json",
        "angles": "angles_Mistral_7B.json",
        "color": "#9b59b6",
    },
    {
        "name": "Yi-1.5-6B",
        "base": "radar_Yi-1.5-6B-base.json",
        "inst": "radar_Yi-1.5-6B-Chat.json",
        "angles": "angles_Yi_1.5_6B.json",
        "color": "#27ae60",
    },
]


def load_family(f):
    """Load one family's radar + angle data."""
    return {
        "name": f["name"],
        "color": f["color"],
        "radar_base": json.load(open(EXP / "exp9_radar" / f["base"])),
        "radar_inst": json.load(open(EXP / "exp9_radar" / f["inst"])),
        "angles": json.load(open(EXP / "exp10_principal_angles" / f["angles"])),
    }


def compute_layer_angles(angles: dict, subset: set[str] | None = None) -> dict[int, float]:
    """Aggregate per-matrix angles to per-layer mean angle."""
    if subset is None:
        subset = LAYER_COMPS

    per_layer: dict[int, list[float]] = {}
    for it in angles["per_matrix"]:
        layer = it.get("layer")
        comp = it.get("comp")
        if layer is None or comp not in subset:
            continue
        per_layer.setdefault(layer, []).append(it["median_angle_deg"])

    return {l: float(np.mean(v)) for l, v in per_layer.items()}


def compute_pr_shifts(radar_base: dict, radar_inst: dict) -> dict[str, list[float]]:
    """ΔPR per layer per band."""
    shifts: dict[str, list[float]] = {}
    bp = radar_base["depth_profiles"]
    ip = radar_inst["depth_profiles"]
    bands = list(bp.keys())
    for band in bands:
        base_profile = bp[band]
        inst_profile = ip[band]
        n = min(len(base_profile), len(inst_profile))
        shifts[band] = [
            float(inst_profile[i] - base_profile[i]) for i in range(n)
        ]
    return shifts


def band_label(bkey: str) -> str:
    return {
        "band1_factual": "Factual",
        "band2_instruction": "Instruction",
        "band3_creative": "Creative",
        "band4_code": "Code / Logic",
        "band5_dialogue": "Dialogue",
        "band6_counterfactual": "Counterfactual",
        "band7_safety": "Safety",
    }.get(bkey, bkey)


def main():
    print("Exp 11: Measurement-Angle Shift Theorem — direct verification")
    print("=" * 70)
    print()

    # Load all 3 families to get a wider θ range
    fam_data = [load_family(f) for f in FAMILIES]

    rows = []
    per_family_stats = []

    for fam in fam_data:
        print(f"\n── {fam['name']} ──")

        layer_theta = compute_layer_angles(fam["angles"], LAYER_COMPS)
        theta_attn = compute_layer_angles(fam["angles"], ATTN_COMPS)
        theta_ffn = compute_layer_angles(fam["angles"], FFN_COMPS)

        theta_min = min(layer_theta.values())
        theta_max = max(layer_theta.values())
        print(f"  Layers: {len(layer_theta)}   θ range: {theta_min:.2f} – {theta_max:.2f}°")

        # Cumulative θ: hidden state at layer L has propagated through all 0..L,
        # so its effective rotation is the mean of layer rotations up to L.
        cumulative_theta: dict[int, float] = {}
        running = []
        for l in sorted(layer_theta.keys()):
            running.append(layer_theta[l])
            cumulative_theta[l] = float(np.mean(running))

        pr_shifts = compute_pr_shifts(fam["radar_base"], fam["radar_inst"])

        for band, profile in pr_shifts.items():
            for layer_idx, delta in enumerate(profile):
                if layer_idx not in layer_theta:
                    continue
                rows.append({
                    "family": fam["name"],
                    "layer": layer_idx,
                    "band": band,
                    "band_label": band_label(band),
                    "theta": layer_theta[layer_idx],
                    "theta_cum": cumulative_theta[layer_idx],
                    "theta_attn": theta_attn.get(layer_idx, 0.0),
                    "theta_ffn": theta_ffn.get(layer_idx, 0.0),
                    "delta_pr": delta,
                    "abs_delta_pr": abs(delta),
                })
        per_family_stats.append({
            "name": fam["name"],
            "theta_min": float(theta_min),
            "theta_max": float(theta_max),
            "n_layers": len(layer_theta),
        })

    print(f"\nTotal data points: {len(rows)}")

    # Aggregate arrays
    thetas = np.array([r["theta"] for r in rows])
    thetas_cum = np.array([r["theta_cum"] for r in rows])
    abs_dpr = np.array([r["abs_delta_pr"] for r in rows])
    signed_dpr = np.array([r["delta_pr"] for r in rows])
    print(f"Global θ range: {thetas.min():.2f} – {thetas.max():.2f}°")
    print(f"Global θ_cum range: {thetas_cum.min():.2f} – {thetas_cum.max():.2f}°")

    # ── Correlation analysis ──
    def corr(x, y, name):
        pear_r, pear_p = stats.pearsonr(x, y)
        spear_r, spear_p = stats.spearmanr(x, y)
        print(f"  {name}:")
        print(f"    Pearson  r = {pear_r:+.4f}   p = {pear_p:.2e}")
        print(f"    Spearman ρ = {spear_r:+.4f}   p = {spear_p:.2e}")
        return pear_r, pear_p, spear_r, spear_p

    print("\nAggregate correlations (all 7 bands × all layers pooled):")
    pr_abs     = corr(thetas,     abs_dpr,    "θ (per-layer)     vs |ΔPR|")
    pr_abs_cum = corr(thetas_cum, abs_dpr,    "θ (cumulative)    vs |ΔPR|")
    pr_sign    = corr(thetas,     signed_dpr, "θ (per-layer)     vs ΔPR (signed)")

    # Per-band correlations
    print("\nPer-band Pearson r(θ, |ΔPR|):")
    band_corrs = {}
    for band in pr_shifts:
        band_rows = [r for r in rows if r["band"] == band]
        x = np.array([r["theta"] for r in band_rows])
        y = np.array([r["abs_delta_pr"] for r in band_rows])
        pr, pp = stats.pearsonr(x, y) if len(x) >= 3 else (0, 1)
        sr, sp = stats.spearmanr(x, y) if len(x) >= 3 else (0, 1)
        band_corrs[band] = {
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
        }
        print(f"  {band_label(band):<18s}  r = {pr:+.3f}  ρ = {sr:+.3f}")

    # ── Regression: abs_delta_pr ~ theta ──
    slope, intercept, r_value, p_value, stderr = stats.linregress(thetas, abs_dpr)
    print(f"\nLinear regression (θ per-layer):   |ΔPR| = {slope:.4f} × θ + {intercept:.4f}")
    print(f"  R² = {r_value**2:.4f}   p = {p_value:.2e}")

    slope_c, intercept_c, r_c, p_c, _ = stats.linregress(thetas_cum, abs_dpr)
    print(f"Linear regression (θ cumulative):  |ΔPR| = {slope_c:.4f} × θ + {intercept_c:.4f}")
    print(f"  R² = {r_c**2:.4f}   p = {p_c:.2e}")

    # ──────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────
    band_colors = {
        "band1_factual": "#0ea5e9",
        "band2_instruction": "#8b5cf6",
        "band3_creative": "#f97316",
        "band4_code": "#10b981",
        "band5_dialogue": "#ec4899",
        "band6_counterfactual": "#eab308",
        "band7_safety": "#ef4444",
    }

    # Plot 1: scatter θ (per-layer) vs |ΔPR|, colored by family (primary)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(
        "Exp 11: rotation θ predicts |ΔPR| — Measurement-Angle Shift theorem",
        fontsize=15, fontweight="bold", y=1.02,
    )

    ax = axes[0]
    for fam in fam_data:
        fam_rows = [r for r in rows if r["family"] == fam["name"]]
        xs = [r["theta"] for r in fam_rows]
        ys = [r["abs_delta_pr"] for r in fam_rows]
        ax.scatter(xs, ys, s=45, color=fam["color"], alpha=0.65,
                   edgecolors="black", linewidths=0.4, label=fam["name"])

    x_line = np.array([thetas.min(), thetas.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "k--", linewidth=2, alpha=0.7,
            label=f"global fit: y = {slope:.3f}x + {intercept:.3f}")

    ax.set_xlabel("Layer rotation θ  (degrees)", fontsize=11)
    ax.set_ylabel("|ΔPR|  =  |PR_instruct − PR_base|  (per layer, per band)", fontsize=11)
    ax.set_title(
        f"θ (per-layer) vs |ΔPR|   |   "
        f"Pearson r = {pr_abs[0]:+.3f} (p = {pr_abs[1]:.1e})   |   "
        f"R² = {r_value**2:.3f}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.25)

    # Plot 2: per-family regression — different rotation regimes
    ax = axes[1]
    for fam in fam_data:
        fam_rows = [r for r in rows if r["family"] == fam["name"]]
        xs = np.array([r["theta"] for r in fam_rows])
        ys = np.array([r["abs_delta_pr"] for r in fam_rows])
        if len(xs) >= 3:
            s_f, i_f, r_f, p_f, _ = stats.linregress(xs, ys)
            x_lin = np.array([xs.min(), xs.max()])
            ax.plot(x_lin, s_f * x_lin + i_f, "-", color=fam["color"],
                    linewidth=2.5, alpha=0.9,
                    label=f"{fam['name']}:  slope={s_f:.3f}, r={r_f:+.3f}")
            ax.scatter(xs, ys, s=25, color=fam["color"], alpha=0.4,
                       edgecolors="black", linewidths=0.3)

    ax.set_xlabel("Layer rotation θ  (degrees)", fontsize=11)
    ax.set_ylabel("|ΔPR|", fontsize=11)
    ax.set_title("Per-family regression — different rotation regimes",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(PLOTS / "theta_predicts_pr_shift.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  saved theta_predicts_pr_shift.png")

    # Plot 3: per-layer profile for each family
    fig, axes = plt.subplots(1, len(fam_data), figsize=(7 * len(fam_data), 5.5))
    if len(fam_data) == 1:
        axes = [axes]
    fig.suptitle("Per-layer angle θ vs mean |ΔPR| — each family has its own regime",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, fam in zip(axes, fam_data):
        fam_rows = [r for r in rows if r["family"] == fam["name"]]
        if not fam_rows:
            continue
        layers_sorted = sorted(set(r["layer"] for r in fam_rows))
        layer_theta_f = {}
        for r in fam_rows:
            layer_theta_f.setdefault(r["layer"], r["theta"])
        thetas_list = [layer_theta_f[l] for l in layers_sorted]
        mean_dpr = []
        std_dpr = []
        for l in layers_sorted:
            vals = [r["abs_delta_pr"] for r in fam_rows if r["layer"] == l]
            mean_dpr.append(np.mean(vals))
            std_dpr.append(np.std(vals))

        ax.plot(layers_sorted, thetas_list, "o-", color="#e74c3c", linewidth=2,
                markersize=5, label="θ  (°)")
        ax.set_ylabel("θ (degrees)", color="#e74c3c", fontsize=10)
        ax.tick_params(axis="y", labelcolor="#e74c3c")
        ax.set_xlabel("Layer index", fontsize=10)

        ax2 = ax.twinx()
        ax2.errorbar(layers_sorted, mean_dpr, yerr=std_dpr, fmt="s-",
                     color="#3498db", linewidth=2, markersize=5, capsize=3,
                     alpha=0.8, label="mean |ΔPR| ± std")
        ax2.set_ylabel("|ΔPR|", color="#3498db", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#3498db")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
        ax.set_title(fam["name"], fontsize=11, fontweight="bold", color=fam["color"])
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(PLOTS / "per_layer_profiles.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  saved per_layer_profiles.png")

    # ── Save summary ──
    summary = {
        "theorem": (
            "For same probe Q and Σ conserved, ΔPR depends on rotation angle θ "
            "between U_base and U_instruct. Prediction: |ΔPR(L,band)| correlates "
            "with θ(L)."
        ),
        "families_analysed": [f["name"] for f in fam_data],
        "per_family": per_family_stats,
        "n_data_points": len(rows),
        "theta_range_deg": [float(thetas.min()), float(thetas.max())],
        "abs_delta_pr_range": [float(abs_dpr.min()), float(abs_dpr.max())],
        "aggregate_correlation_theta_per_layer": {
            "pearson_r": float(pr_abs[0]),
            "pearson_p": float(pr_abs[1]),
            "spearman_r": float(pr_abs[2]),
            "spearman_p": float(pr_abs[3]),
        },
        "aggregate_correlation_theta_cumulative": {
            "pearson_r": float(pr_abs_cum[0]),
            "pearson_p": float(pr_abs_cum[1]),
            "spearman_r": float(pr_abs_cum[2]),
            "spearman_p": float(pr_abs_cum[3]),
        },
        "linear_regression_theta_per_layer": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
        },
        "linear_regression_theta_cumulative": {
            "slope": float(slope_c),
            "intercept": float(intercept_c),
            "r_squared": float(r_c ** 2),
            "p_value": float(p_c),
        },
        "per_band_correlations": {
            band_label(k): v for k, v in band_corrs.items()
        },
        "verdict": (
            "THEOREM SUPPORTED. θ and |ΔPR| are positively correlated "
            "(p < 0.001 expected). PR shifts are the measurement-angle signature "
            "of U rotation — not capacity collapse."
        ) if pr_abs[3] < 0.001 and pr_abs[2] > 0 else (
            "Correlation weak or non-monotonic; inspect per-band table."
        ),
    }
    out_path = OUT / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Summary saved to {out_path}")

    # Also dump the full rows as CSV
    import csv
    csv_path = OUT / "data_points.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Data saved to {csv_path}")

    print("\n" + "=" * 70)
    print(summary["verdict"])
    print("=" * 70)


if __name__ == "__main__":
    main()
