# Spectral Flow Probe v2

> **A 7-band Phased Array Radar for any Transformer.**
> Watch what your RL training is *actually* doing to your model's representation geometry — in real time, during training, every step.

---

## Loss 下降一定意味着模型变好了吗？

### *Your loss curve is lying to you.*

![Hero: What Loss Sees vs What SFP Sees](assets/v2/hero_what_loss_sees.png)

You train a model. Your loss goes down. Your reward goes up. Everything is green.

**Meanwhile, inside the model:** the Creative Generation channel lost 42% of its bandwidth. The Counterfactual Reasoning channel lost 42%. The Safety Boundary channel gained 25%. None of this is visible in the loss curve.

**SFP is the second monitor your training dashboard is missing.** 7 fixed probes. Deterministic. Reproducible. Runs in milliseconds per training step.

```python
from spectral_flow_probe import SpectralCallback
trainer = Trainer(..., callbacks=[SpectralCallback(every_n_steps=100)])
```

That's it. Now you can see the damage.

---

## The four things we learned the hard way

### 1️⃣ "PR" was not a scalar — it was a lie.

![Scalar PR was broken](assets/v2/scalar_vs_vector.png)

v1 of this tool measured a single PR number using random-token probes. It claimed to detect "69% PR collapse during DPO". **Both the method and the conclusion were wrong.**

- **Left**: Same model, 10 runs with random tokens. CV = 30%. The thermometer itself was broken.
- **Middle**: The "69% collapse" was 100% measurement noise.
- **Right**: Same model, 10 runs with fixed deterministic prompts. CV = 0%. Every time.

**PR = f(model, query), not f(model).** A single PR number is like a single-pixel photo of a 7-megapixel scene. You need a vector.

### 2️⃣ RL does not collapse the channel. It rotates the beam.

![Rotation not collapse](assets/v2/rotation_not_collapse.png)

We compared base ↔ instruct weights across three unrelated model families — Alibaba's Qwen2.5, Mistral AI's Mistral-7B, 01.AI's Yi-1.5. The weight matrices drifted by **1.3%, 3.9%, and 24.6%** respectively — a 19× spread.

**But the SVD spectrum moved by ≈0% in all three families.**

This is *isovolumetric rotation*. The singular values (Σ, the channel capacity) are conserved. The singular vectors (U, V, the channel direction) rotate. RL doesn't make the pipe narrower — it aims the pipe somewhere new.

### 3️⃣ The rotation is quantifiable — and measurable in real time.

![Real-time monitor](assets/v2/real_time_monitor.png)

The same training run, seen two ways:

- **Left (Standard view)**: loss curve. Smooth, pretty, monotonic. Tells you nothing about which capabilities are being traded for which.
- **Right (SFP view)**: seven bands, each tracked every N training steps. You can literally *watch* the beam rotate. Creative bandwidth dying at step 600? You'll see it at step 600, not after $50K of compute is already spent.

The monitor uses fixed-prompt probes. Zero variance. 100% reproducible. Hooks into any HuggingFace Trainer callback chain.

### 4️⃣ Your RL data mix is a diagnostic signal — use it.

![Data mix mirror](assets/v2/mirror_audit.png)

Before you start training, ask yourself: *does my data mix match what my model actually needs?*

SFP's `BandwidthDiagnostic` takes your base model + your training data distribution and flags where you're wasting compute:

> ⚠️ Code data is 50% of your mix, but this model's code channel is already near-saturated — you're 3.9× oversupplied.
> ⚠️ Creative data is 0% — and the creative channel is wide open for optimization. You're wasting the opportunity.
> ⚠️ Multi-turn Dialogue needs ~23% of your data budget to move meaningfully. You gave it 5%.

**It's the照妖镜** — a pre-training sanity check that fits on a slide.

---

## The whole toolkit in one picture

![3-Family Radar](assets/v2/3family_radar.png)

*Three model families, one mechanism. Universal across architectures.*

---

## Install + use in 30 seconds

```bash
git clone https://github.com/HenryZ838978/spectral-flow-probe.git
cd spectral-flow-probe
pip install -e .
```

```python
from spectral_flow_probe import SpectralProbe, BandwidthDiagnostic, SpectralCallback

# 1. Scan a model → get a 7-dimensional fingerprint
fp = SpectralProbe("meta-llama/Llama-3.1-8B-Instruct").scan()
print(fp)
print(fp.pr_vector)             # 7-dim numpy array
print(fp.weakest_band.name)     # "Multi-turn Dialogue"

# 2. Audit your RL data mix BEFORE spending $50K on GPUs
diag = BandwidthDiagnostic()
report = diag.audit_data_mix(fp, your_data_distribution)
print(report)                    # ← the mirror

# 3. Monitor during training (drop-in HF Trainer callback)
cb = SpectralCallback(
    every_n_steps=100,
    bands=["band3_creative", "band7_safety"],  # or None for all 7
    drift_threshold=0.10,                        # alert if any band shifts >10%
    logger="wandb",
)
trainer = Trainer(..., callbacks=[cb])
```

CLI:

```bash
sfp scan meta-llama/Llama-3.1-8B-Instruct --plot radar.png -o fp.json
sfp compare base/path instruct/path --plot diff.png
sfp rotate  base/path instruct/path --gpu 0       # SVD-space verdict
sfp profile single-model                          # no base pair needed
```

---

## The seven bands

Each band is a fixed, deterministic prompt set that targets one functional channel:

| Band | Channel | What it measures |
|---|---|---|
| 1. Factual Recall       | engram retrieval       | Knowledge bandwidth |
| 2. Instruction Following | constraint processing  | Format/structural compliance |
| 3. Creative Generation  | open generation        | Open-ended bandwidth |
| 4. Code / Logic         | logical reasoning      | Symbolic bandwidth |
| 5. Multi-turn Dialogue  | context maintenance    | Memory bandwidth |
| 6. Counterfactual       | OOD generalization     | Off-distribution bandwidth |
| 7. Safety Boundary      | RL specialization      | RL-targeted bandwidth |

Prompts are committed to git. **Same model + same band = same PR, every single run.** That's the whole reason v2 exists.

---

## The five entry points

```python
from spectral_flow_probe import (
    SpectralProbe,        # 7-band radar scan
    RotationAnalyzer,     # weight-space SVD analysis (pair or single-model mode)
    BandwidthDiagnostic,  # the RL data mix mirror (照妖镜)
    SpectralCallback,     # training-time monitor (fixed-prompt, deterministic)
    spectral_pr_loss,     # differentiable per-band regularizer
)
```

---

## Receipts (all claims above are backed by experiments in `experiments/`)

![Calibration](assets/v2/calibration.png)

![Band 0 3-family](assets/v2/band0_3family.png)

![Weak DPO control](assets/v2/weak_dpo_radar.png)

| Claim | Experiment | Result |
|---|---|---|
| Random-token probe has 30% CV | Exp 7C | Same checkpoint, 10 runs, PR ranges 4.6 – 14.1 |
| Fixed-prompt probe has 0% CV | Exp 7C | Same checkpoint, 10 runs, PR identical to 6 decimals |
| PR is query-dependent | Exp 7D | Same model, 10 query types, PR varies 2× |
| Weak DPO doesn't move the model | Exp 8 | 800 steps → 0.02% weight change |
| Isovolumetric rotation is universal | Exp 9B | 3 families, weight drift 1.3–24.6%, SVD shift ~0% everywhere |
| OOD benchmarks don't detect bandwidth loss | Exp 7A/B | IFEval + LiveCodeBench flat across all DPO checkpoints |

Full experiment log: `experiments/` — 9 experiments, 4 days, one refuted hypothesis, one new theory.

---

## What's gone from v1 (and why)

| Removed | Why |
|---|---|
| `SpectralReport.diagnose()` scalar "pr_health" | PR is not a scalar; thresholds are meaningless |
| `SpectralCallback` random-token probe | 30% measurement CV → unusable |
| `BudgetPlanner` empirical reference data | Data was measured with the broken probe |
| `prompts.py` flat 50-prompt list | Replaced with structured 7-band prompts |

If you depended on v1: `git checkout v0.1.0`. We don't recommend it.

---

## Theory

This tool is the experimental face of a broader theoretical framework: **Representation Bandwidth Economics**. In short:

- PR = f(model, query). Total channel capacity is an architectural invariant.
- RL alignment = isovolumetric rotation of the weight singular vectors.
- Alignment reallocates bandwidth across functional channels; it does not create or destroy it.
- Therefore: every RL run is a *zero-sum bandwidth trade*. SFP shows you what's being traded.

Full derivation and experimental record: `spectral_flow_exp/updates/EXPERIMENT_LOG.md` and the companion paper (in prep).

---

## Citation

```bibtex
@article{zhang2026bandwidtheconomics,
  title   = {Representation Bandwidth Economics: RL Alignment as Isovolumetric
             Rotation of the Spectral Beam Pattern},
  author  = {Zhang, Jing},
  year    = {2026},
  doi     = {10.5281/zenodo.19585083},
  url     = {https://doi.org/10.5281/zenodo.19585083}
}
```

---

## License

MIT. Use it. Fork it. Tell us what you find.

---

## Acknowledgments

This tool exists because v1 was wrong and we noticed.
We expect v2 to be wrong in some way too. Tell us how.
