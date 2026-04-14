# Spectral Flow — Round 1 Results (Pilot)

**Generated**: 2026-04-13 22:53:25  
**Protocol**: 135 prompts (45 × 3), 4 sampled layers, bf16  
**Status**: Pilot data; superseded by Rounds 2–3 (all-layer, 50 prompts). See `Spectral_Flow_Theory.md` for consolidated results.

---

## 1. S(depth) & ΔS (4 models, diverse architecture)

| Model | N_params | S(first) | S(mid) | S(last) | ΔS | PR(last) |
|-------|----------|----------|--------|---------|------|----------|
| gemma4-4B | 4.0B | −0.198 | −0.157 | −0.130 | +0.068 | 8.5 |
| qwen25-7B | 7.6B | −0.146 | −0.150 | −0.110 | +0.036 | 12.6 |
| qwen3-14B | 14.8B | −0.152 | −0.147 | −0.134 | +0.017 | 7.3 |
| deepseek-r1-14B | 14.8B | −0.131 | −0.132 | −0.159 | −0.028 | 4.3 |

ΔS vs log₁₀(N): r = −0.887, p = 0.113 — **not significant** at conventional thresholds. The negative direction and low p reflect RL intensity confounding: DeepSeek-R1 and Qwen3 have heavy RL that compresses ΔS, masking the size effect. Round 2 addressed this with controlled Qwen3 scaling.

## 2. FormatLock (S(last) vs distinct-2)

| Model | S(last) | distinct-2 |
|-------|---------|------------|
| deepseek-r1-14B | −0.159 | 0.717 |
| qwen3-14B | −0.134 | 0.756 |
| gemma4-4B | −0.130 | 0.748 |
| qwen25-7B | −0.110 | 0.790 |

r = 0.972, p = 0.028 (N = 4). Suggestive but tiny sample. distinct-2 is a weak proxy for format lock — it correlates across diverse architectures but fails within a model family (r = 0.52 in 8-model expansion).

## 3. Q/K/V Spectral

| Model | S_Q(first) | S_Q(last) | S_K(first) | S_K(last) | S_V(first) | S_V(last) |
|-------|------------|-----------|------------|-----------|------------|-----------|
| gemma4-4B | 0.000 | −0.186 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen25-7B | 0.000 | −0.204 | 0.000 | −0.175 | 0.000 | −0.129 |
| qwen3-14B | −0.118 | −0.173 | −0.103 | −0.179 | −0.150 | −0.132 |
| deepseek-r1-14B | 0.000 | −0.196 | 0.000 | −0.197 | 0.000 | −0.181 |

Zero values at first layer indicate insufficient variance at layer 0 for reliable slope fitting. Not analyzed further.

---

See `/cache/zhangjing/spectral_flow_exp/results/plots/01–06` for visualizations.
