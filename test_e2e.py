"""End-to-end smoke test for SFP v2 — exercises all entry points."""
import torch

MODEL = "/cache/zhangjing/models/Qwen3-0.6B"

print("=" * 70)
print("  SFP v2 End-to-End Smoke Test")
print("=" * 70)

# ── 1. SpectralProbe.scan() — radar fingerprint ──
print("\n[1/5] SpectralProbe.scan() — 7-band radar")
from spectral_flow_probe import SpectralProbe

probe = SpectralProbe(MODEL)
fp = probe.scan()
print(fp.summary())
assert len(fp.bands) == 7
assert fp.mean_pr > 0
assert fp.bandwidth_ratio > 0
print("  [PASS]")

# ── 2. RotationAnalyzer.profile() — single-model spectrum ──
print("\n[2/5] RotationAnalyzer.profile() — single-model spectrum")
from spectral_flow_probe import RotationAnalyzer

ra = RotationAnalyzer()
profile = ra.profile(MODEL, gpu_id=0 if torch.cuda.is_available() else "cpu")
print(profile.summary())
assert "q_proj" in profile.per_component
print("  [PASS]")

# ── 3. BandwidthDiagnostic — baseline + data mix audit ──
print("\n[3/5] BandwidthDiagnostic — baseline + data mix")
from spectral_flow_probe import BandwidthDiagnostic, BAND_KEYS

diag = BandwidthDiagnostic()
baseline = diag.diagnose_baseline(fp)
print(f"  Weak bands: {[b['band_name'] for b in baseline['weak_bands']]}")
for r in baseline["recommendations"][:3]:
    print(f"    • {r}")

# Fake data mix — overweight code, no creative
data_mix = dict.fromkeys(BAND_KEYS, 0.05)
data_mix["band4_code"] = 0.50
data_mix["band2_instruction"] = 0.30
data_mix["band3_creative"] = 0.0
mix_report = diag.audit_data_mix(fp, data_mix)
print()
print(mix_report.summary())
print("  [PASS]")

# ── 4. SpectralCallback — instantiation only (no real training) ──
print("\n[4/5] SpectralCallback — instantiation")
from spectral_flow_probe import SpectralCallback

cb = SpectralCallback(every_n_steps=50, bands=["band2_instruction", "band4_code"])
assert cb.band_keys == ["band2_instruction", "band4_code"]
print("  Created with 2 bands, every 50 steps. [PASS]")

# ── 5. Regularizer — differentiable per-band loss ──
print("\n[5/5] spectral_pr_loss — differentiable per-band loss")
from spectral_flow_probe import spectral_pr_loss, compute_pr_differentiable

H = torch.randn(32, 256, requires_grad=True)
pr = compute_pr_differentiable(H)
print(f"  PR(random H) = {pr.item():.2f}")
loss = spectral_pr_loss(H, target_pr=5.0, mode="floor")
loss.backward()
print(f"  Loss = {loss.item():.4f}, grad norm = {H.grad.norm().item():.4f}")
assert H.grad is not None
print("  [PASS]")

# ── Plot test ──
print("\n[Bonus] Radar plot")
from spectral_flow_probe import plot_radar

plot_radar(fp, save="/cache/zhangjing/spectral-flow-probe/test_radar.png")
print("  Plot saved to test_radar.png")

print("\n" + "=" * 70)
print("  ALL ENTRY POINTS PASSED")
print("=" * 70)
