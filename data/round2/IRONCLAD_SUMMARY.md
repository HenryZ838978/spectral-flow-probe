# Spectral Flow — Round 2 Results

**Generated**: 2026-04-14 00:15:26  
**Protocol**: 50 prompts, all layers, bf16 / AWQ  
**Models tested**: 9 (Qwen3 scaling series, RL gradient, SDE)  
**Status**: See `Spectral_Flow_Theory.md` for consolidated results including Round 3.

---

## Exp 1: Qwen3 Scaling (controlling RL pipeline, varying N)

| Model | N (B) | Layers | S(first) | S(last) | ΔS | ΔS/layer | PR(last) |
|-------|-------|--------|----------|---------|------|----------|----------|
| qwen3-0.6B | 0.6 | 28 | −0.200 | −0.123 | +0.078 | 0.00277 | 12.2 |
| qwen3-1.7B | 1.7 | 28 | −0.181 | −0.121 | +0.060 | 0.00214 | 12.3 |
| qwen3-4B | 4.0 | 36 | −0.187 | −0.122 | +0.065 | 0.00180 | 9.8 |
| qwen3-8B | 8.0 | 36 | −0.178 | −0.128 | +0.050 | 0.00138 | 8.9 |
| qwen3-14B | 14.8 | 40 | −0.152 | −0.134 | +0.017 | 0.00043 | 7.3 |

- ΔS/layer vs log(N): r = −0.968, p = 0.007
- ΔS (raw) vs log(N): r = −0.876, p = 0.051

**ΔS/layer is the cleaner metric** — normalizes out differing layer counts. Strictly monotonic decrease.

## Exp 2: RL Intensity Gradient

| Model | N (B) | RL | ΔS | S(last) | PR(last) | distinct-2 |
|-------|-------|-----|------|---------|----------|------------|
| qwen25-7B-base | 7.6 | none | +0.038 | −0.110 | 13.3 | 0.736 |
| qwen25-7B | 7.6 | moderate | +0.036 | −0.110 | 12.6 | 0.790 |
| qwen3-8B | 8.0 | heavy | +0.050 | −0.128 | 8.9 | 0.750 |
| qwen3-14B | 14.8 | heavy | +0.017 | −0.134 | 7.3 | 0.756 |
| deepseek-r1-14B | 14.8 | extreme | −0.028 | −0.159 | 4.3 | 0.717 |

PR(last) monotonically tracks RL intensity: 13.3 → 12.6 → 8.9 → 7.3 → 4.3. Cross-architecture comparison is confounded by layer count; PR(last) is more robust than ΔS for this purpose.

## Exp 3: SDE (scale=0.0 on Qwen3-14B)

| Condition | ΔS | S(last) | PR(last) |
|-----------|------|---------|----------|
| Baseline | +0.017 | −0.134 | 7.3 |
| SDE | +0.006 | −0.146 | 4.2 |

Global damage (ΔS down, PR down). Local restoration at L23 (+0.019). See Round 3 for scale=0.3 confirmation.

---

See `/cache/zhangjing/spectral_flow_exp/results/plots/paper_fig1–6` for visualizations.
