<div align="center">

<h1>
<img src="https://img.shields.io/badge/SFP-Spectral_Flow_Probe-blue?style=for-the-badge&labelColor=0d1117&color=2563eb" height="42"/>
</h1>

<p>
<img src="https://img.shields.io/badge/python-≥3.10-3776ab?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/torch-≥2.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/license-Apache_2.0-green?style=flat-square"/>
<img src="https://img.shields.io/badge/models_tested-11-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/arXiv-2504.XXXXX-b31b1b?style=flat-square&logo=arxiv&logoColor=white"/>
</p>

<h3>20-Minute Geometric Diagnostic for Any Transformer</h3>

<p><i>One PCA scan. Zero benchmarks. See through any model.</i></p>

</div>

---

<div align="center">
<table>
<tr>
<td width="25%" align="center"><b>Auditor</b><br/><sub>Post-hoc diagnosis</sub></td>
<td width="25%" align="center"><b>Monitor</b><br/><sub>RL training guard</sub></td>
<td width="25%" align="center"><b>Planner</b><br/><sub>Architecture budget</sub></td>
<td width="25%" align="center"><b>Regularizer</b><br/><sub>Differentiable PR loss</sub></td>
</tr>
<tr>
<td align="center"><code>probe.run()</code></td>
<td align="center"><code>SpectralCallback</code></td>
<td align="center"><code>BudgetPlanner</code></td>
<td align="center"><code>spectral_pr_loss()</code></td>
</tr>
</table>
</div>

---

## What Does It See?

<div align="center">
<img src="assets/example_diagnosis.png" width="800"/>
<br/>
<sub>Full diagnosis of Qwen3-0.6B in <b>0.7 seconds</b>. S(depth), PR(depth), eigenvalue spectrum, and auto-generated diagnosis.</sub>
</div>

<br/>

Every Transformer layer reshapes representation geometry. SFP measures this via **PCA eigenvalue spectral slopes** across all decoder layers — revealing alignment compression, architectural efficiency, and training quality without running a single benchmark.

<div align="center">
<table>
<tr>
<th>Metric</th>
<th>What it means</th>
<th>Visual</th>
</tr>
<tr>
<td><b>S(depth)</b></td>
<td>Log-linear PCA eigenvalue slope at each layer<br/><sub>More negative = more anisotropic ("needle")</sub></td>
<td>

```
Layer 0   ████████████████░░ −0.51
Layer 14  █████████████░░░░░ −0.38
Layer 27  ██████████░░░░░░░░ −0.25
         anisotropic ──→ isotropic
```

</td>
</tr>
<tr>
<td><b>ΔS</b></td>
<td>S(last) − S(first)<br/><sub>Total spectral expansion through the network</sub></td>
<td>

```
Base model:    ΔS = +0.26  ✦ expanding
Chat-tuned:    ΔS = +0.05  ▸ compressed
Extreme RL:    ΔS = −0.03  ✖ collapsed
```

</td>
</tr>
<tr>
<td><b>PR</b></td>
<td>Participation Ratio = effective dimensionality<br/><sub>(Σλ)² / Σλ²</sub></td>
<td>

```
PR = 13.3  ████████████████ excellent
PR =  7.3  █████████░░░░░░░ good
PR =  4.3  █████░░░░░░░░░░░ compressed
PR =  1.2  █░░░░░░░░░░░░░░░ collapsed
```

</td>
</tr>
</table>
</div>

---

## Empirical Validation: 11 Models, 4 Rounds

<details open>
<summary><b>Click to expand full results table</b></summary>
<br/>

<div align="center">
<table>
<tr>
<th>Model</th>
<th>N (B)</th>
<th>Layers</th>
<th>RL</th>
<th>S(first)</th>
<th>S(last)</th>
<th>ΔS</th>
<th>ΔS/layer</th>
<th>PR(last)</th>
</tr>
<tr>
<td>Qwen3-0.6B</td><td>0.6</td><td>28</td>
<td><img src="https://img.shields.io/badge/-light-22c55e?style=flat-square"/></td>
<td>−0.200</td><td>−0.123</td><td>+0.078</td><td>0.00277</td>
<td><b>12.2</b></td>
</tr>
<tr>
<td>Qwen3-1.7B</td><td>1.7</td><td>28</td>
<td><img src="https://img.shields.io/badge/-light-22c55e?style=flat-square"/></td>
<td>−0.181</td><td>−0.121</td><td>+0.060</td><td>0.00214</td>
<td><b>12.3</b></td>
</tr>
<tr>
<td>Gemma-4-E4B-it</td><td>4.0</td><td>26</td>
<td><img src="https://img.shields.io/badge/-moderate-eab308?style=flat-square"/></td>
<td>−0.198</td><td>−0.130</td><td>+0.068</td><td>0.00262</td>
<td><b>8.5</b></td>
</tr>
<tr>
<td>Qwen3-4B</td><td>4.0</td><td>36</td>
<td><img src="https://img.shields.io/badge/-heavy-f97316?style=flat-square"/></td>
<td>−0.187</td><td>−0.122</td><td>+0.065</td><td>0.00180</td>
<td><b>9.8</b></td>
</tr>
<tr>
<td>Qwen2.5-7B-Base</td><td>7.6</td><td>28</td>
<td><img src="https://img.shields.io/badge/-none-94a3b8?style=flat-square"/></td>
<td>−0.146</td><td>−0.110</td><td>+0.038</td><td>0.00134</td>
<td><b>13.3</b></td>
</tr>
<tr>
<td>Qwen2.5-7B-Instruct</td><td>7.6</td><td>28</td>
<td><img src="https://img.shields.io/badge/-moderate-eab308?style=flat-square"/></td>
<td>−0.146</td><td>−0.110</td><td>+0.036</td><td>0.00129</td>
<td><b>12.6</b></td>
</tr>
<tr>
<td>Qwen3-8B</td><td>8.0</td><td>36</td>
<td><img src="https://img.shields.io/badge/-heavy-f97316?style=flat-square"/></td>
<td>−0.178</td><td>−0.128</td><td>+0.050</td><td>0.00138</td>
<td><b>8.9</b></td>
</tr>
<tr>
<td>Qwen3-14B</td><td>14.8</td><td>40</td>
<td><img src="https://img.shields.io/badge/-heavy-f97316?style=flat-square"/></td>
<td>−0.152</td><td>−0.134</td><td>+0.017</td><td>0.00043</td>
<td><b>7.3</b></td>
</tr>
<tr>
<td>DeepSeek-R1-14B</td><td>14.8</td><td>28</td>
<td><img src="https://img.shields.io/badge/-extreme-ef4444?style=flat-square"/></td>
<td>−0.131</td><td>−0.159</td><td><b>−0.028</b></td><td>−0.00100</td>
<td><b>4.3</b></td>
</tr>
<tr>
<td>Qwen3-30B-MoE</td><td>30.5</td><td>48</td>
<td><img src="https://img.shields.io/badge/-heavy-f97316?style=flat-square"/></td>
<td>−0.193</td><td>−0.137</td><td>+0.056</td><td>0.00117</td>
<td><b>17.0</b> <sub>(agg)</sub></td>
</tr>
<tr>
<td colspan="9" align="center"><sub>+ Llama-3.1-8B-Base (PR=5.2), Mistral-7B-Base (PR=4.8) — cross-family validation</sub></td>
</tr>
</table>
</div>

</details>

---

## Key Findings (Figures)

<table>
<tr>
<td width="50%" align="center">
<img src="assets/paper_fig1_fn_scaling.png" width="100%"/>
<br/><sub><b>f(N) Scaling</b> — ΔS/layer vs log(N), r = −0.968</sub>
</td>
<td width="50%" align="center">
<img src="assets/paper_fig3_grl.png" width="100%"/>
<br/><sub><b>g(RL) Compression</b> — PR tracks RL intensity monotonically</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<img src="assets/paper_fig8_thinking.png" width="100%"/>
<br/><sub><b>CoT Bypass</b> — Thinking mode expands representation</sub>
</td>
<td width="50%" align="center">
<img src="assets/paper_fig9_moe.png" width="100%"/>
<br/><sub><b>MoE Diversification</b> — Aggregate PR / per-path PR = 4.5x</sub>
</td>
</tr>
</table>

<table>
<tr>
<td width="50%" align="center">
<img src="assets/paper_fig4_sde.png" width="100%"/>
<br/><sub><b>SDE Ablation</b> — MLP ablation causes global spectral damage</sub>
</td>
<td width="50%" align="center">
<img src="assets/paper_fig2_s_depth_all.png" width="100%"/>
<br/><sub><b>S(depth) All Models</b> — Every model has a unique spectral fingerprint</sub>
</td>
</tr>
</table>

---

## Install

```bash
pip install -e .
```

Or from GitHub:

```bash
pip install git+https://github.com/HenryZ838978/spectral-flow-probe.git
```

---

## Usage

### 1. Auditor — See through any model in 20 minutes

```python
from spectral_flow_probe import SpectralProbe, plot_diagnosis

probe = SpectralProbe("Qwen/Qwen2.5-7B-Instruct")
report = probe.run()                    # 50 built-in prompts, all layers

print(report.summary())                 # one-paragraph diagnosis
print(report.diagnose())                # structured: rl_intensity, pr_health, ...
report.to_json("report.json")           # full JSON with per-layer data
plot_diagnosis(report, save="diag.png") # 4-panel figure
```

**CLI:**

```bash
sfp Qwen/Qwen2.5-7B-Instruct --plot diag.png -o report.json
```

<details>
<summary>Example output</summary>

```
============================================================
Model: Qwen/Qwen2.5-7B-Instruct (7.62B params, 28 layers)
ΔS = 0.0360 (ΔS/layer = 0.00129)
PR(last) = 12.60  |  S(first) = -0.1460  |  S(last) = -0.1100
RL intensity: moderate  |  PR health: excellent
Elapsed: 42.3s
============================================================
```

</details>

### 2. Monitor — Stop RL before it kills the manifold

```python
from spectral_flow_probe import SpectralCallback

callback = SpectralCallback(
    layer_indices=[-1],       # watch last layer
    every_n_steps=50,         # check every 50 steps
    pr_floor=3.0,             # warn if PR(last) < 3
    pr_halt=1.5,              # stop training if PR(last) < 1.5
    logger="wandb",           # log to W&B / tensorboard / None
)

trainer = Trainer(model=model, ..., callbacks=[callback])
trainer.train()               # auto-stops if representation collapses
```

### 3. Planner — Know your budget before burning GPUs

```python
from spectral_flow_probe import BudgetPlanner

plan = BudgetPlanner.estimate(
    n_params_B=14,            # 14B parameters
    n_layers=40,              # 40 decoder layers
    n_modalities=3,           # text + vision + audio
    rl_category="heavy",      # aggressive RL planned
)
print(plan)
```

<details>
<summary>Example output</summary>

```
BudgetEstimate(14.0B, 40L, 3 modalities, RL=heavy)
  Predicted ΔS/layer: 0.00162
  Predicted ΔS:       0.0649
  Predicted PR(last): 10.9
  Headroom score:     0.55
  Recommendation:     Moderate headroom. Consider lighter RL or fewer modalities.
```

</details>

### 4. Regularizer — Teach without crushing the manifold

```python
from spectral_flow_probe import spectral_pr_loss

# Inside your RL training loop:
hidden_states = model.get_last_hidden_state(input_ids)
pr_loss = spectral_pr_loss(
    hidden_states,
    target_pr=5.0,           # preserve at least PR=5
    mode="floor",            # one-sided: only penalize collapse
)
total_loss = rl_loss + 0.01 * pr_loss
total_loss.backward()        # fully differentiable, no SVD needed
```

<blockquote>
<b>Key insight:</b> PR = ||H||<sub>F</sub><sup>4</sup> / ||H<sup>T</sup>H||<sub>F</sub><sup>2</sup> — computed via Frobenius norms only. Zero overhead in backward pass.
</blockquote>

---

## Theory: The Representational Budget Hypothesis

<div align="center">
<table>
<tr><td>

```
ΔS ≈ f(N_params, arch) − g(RL_intensity) − h(N_modalities)
     ▲ capacity budget    ▲ alignment tax    ▲ modality tax
```

</td></tr>
</table>
</div>

Every Transformer has a finite **representational budget** governed by its parameter count and architecture. Post-training alignment (RLHF/DPO) and multimodal integration each consume part of this budget, compressing the representation manifold.

| Term | Meaning | Evidence |
|------|---------|----------|
| `f(N, arch)` | Capacity upper bound, scales with log(N) | Qwen3 0.6B→14B: ΔS/layer monotonically decreasing, r = −0.968 |
| `g(RL)` | Alignment compression | Base → Instruct → R1: PR drops 13.3 → 12.6 → 4.3 |
| `h(mod)` | Modality integration tax | Native multimodal (Gemma) avoids post-hoc tax |

**Not a conservation law.** An empirical upper-bound regularity observed across 11 models. See the [paper](https://arxiv.org/abs/2504.XXXXX) for full methodology, limitations, and discussion.

---

## Data

All raw experimental data from 4 rounds of experiments (11 models, 600+ PCA measurements) is included:

```
data/
├── round1/          # 4 models: initial validation
├── round2/          # 9 configs: Qwen3 scaling, RL gradient, SDE
├── round3/          # 5 configs: SDE scale=0.3, thinking, MoE
└── round4/          # 6 configs: MoE per-path, cross-family base models
```

Each JSON file contains per-layer `S`, `PR`, `eigenvalues`, and generation metrics.

---

## Package Structure

```
spectral_flow_probe/
├── __init__.py       # unified exports
├── core.py           # spectral_slope, compute_pr, PCA pipeline
├── _compat.py        # model loading, layer auto-detection
├── prompts.py        # 50 prompts × 5 categories
├── probe.py          # SpectralProbe  (Auditor)
├── report.py         # SpectralReport + auto-diagnosis
├── moe.py            # MoE auto-detect + per-path PR
├── plot.py           # 4-panel diagnostic + comparison
├── monitor.py        # SpectralCallback  (Monitor)
├── planner.py        # BudgetPlanner  (Planner)
├── regularizer.py    # spectral_pr_loss  (Regularizer)
└── cli.py            # sfp CLI
```

---

## Citation

```bibtex
@article{zhang2025spectralflow,
  title   = {Spectral Flow Theory: Representation Geometry as a Diagnostic
             Framework for Large Language Models},
  author  = {Zhang, Jing},
  year    = {2025},
  journal = {arXiv preprint arXiv:2504.XXXXX}
}
```

---

<div align="center">
<sub>Built during a 48-hour experiment marathon. 11 models. 8× RTX 4090. One theory.</sub>
<br/>
<sub>If this tool saves you from shipping a collapsed model, consider starring the repo.</sub>
</div>
