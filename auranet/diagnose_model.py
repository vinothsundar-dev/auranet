#!/usr/bin/env python3
"""Diagnose model checkpoint and test audio quality issues."""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path

print("=" * 60)
print("🔍 AuraNet Diagnostic — Checkpoint & Audio Analysis")
print("=" * 60)

# =============================================================================
# 1. Checkpoint Analysis
# =============================================================================

model_path = "/Users/vinoth-14902/Documents/Models/best_model_v3_1026.pt"
print(f"\n📦 Loading checkpoint: {model_path}")

ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

print(f"   Type: {type(ckpt)}")

if isinstance(ckpt, dict):
    print(f"   Keys: {list(ckpt.keys())}")

    if 'epoch' in ckpt:
        print(f"   Epoch: {ckpt['epoch']}")
    if 'best_loss' in ckpt:
        print(f"   Best loss: {ckpt['best_loss']:.6f}")
    if 'metrics' in ckpt:
        print(f"   Metrics: {ckpt['metrics']}")

    # Get state dict
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        print("   Using: model_state_dict")
    elif 'ema_state' in ckpt:
        state = ckpt['ema_state']
        print("   Using: ema_state")
    else:
        state = ckpt
        print("   Using: direct state dict")

    print(f"\n   State dict keys ({len(state)} total):")
    for i, (k, v) in enumerate(list(state.items())[:10]):
        print(f"      {k}: {v.shape}")
    if len(state) > 10:
        print(f"      ... and {len(state) - 10} more")

    # Check for NaN/Inf
    nan_count = 0
    inf_count = 0
    zero_count = 0
    for k, v in state.items():
        if torch.isnan(v).any():
            nan_count += 1
            print(f"      ⚠️ NaN in: {k}")
        if torch.isinf(v).any():
            inf_count += 1
            print(f"      ⚠️ Inf in: {k}")
        if (v == 0).all() and v.numel() > 1:
            zero_count += 1

    print(f"\n   Weight health:")
    print(f"      NaN tensors: {nan_count}")
    print(f"      Inf tensors: {inf_count}")
    print(f"      All-zero tensors: {zero_count}")

    if nan_count == 0 and inf_count == 0:
        print("      ✅ Weights appear healthy")
    else:
        print("      ❌ Weights may be corrupted!")

# =============================================================================
# 2. Audio File Analysis
# =============================================================================

print("\n" + "=" * 60)
print("🔊 Test Audio Analysis")
print("=" * 60)

noisy_path = "test_audio/noisy_sample.wav"
clean_path = "test_audio/clean_reference.wav"
enhanced_path = "outputs/enhanced.wav"

if Path(noisy_path).exists() and Path(clean_path).exists():
    noisy, sr1 = sf.read(noisy_path, dtype='float32')
    clean, sr2 = sf.read(clean_path, dtype='float32')

    print(f"\n   Noisy: {len(noisy)} samples @ {sr1}Hz ({len(noisy)/sr1:.2f}s)")
    print(f"      Peak: {np.abs(noisy).max():.4f}, RMS: {np.sqrt(np.mean(noisy**2)):.4f}")

    print(f"\n   Clean: {len(clean)} samples @ {sr2}Hz ({len(clean)/sr2:.2f}s)")
    print(f"      Peak: {np.abs(clean).max():.4f}, RMS: {np.sqrt(np.mean(clean**2)):.4f}")

    # Correlation
    min_len = min(len(noisy), len(clean))
    corr = np.corrcoef(noisy[:min_len], clean[:min_len])[0, 1]
    print(f"\n   Correlation (noisy ↔ clean): {corr:.4f}")

    if corr < 0.3:
        print("      ⚠️ LOW - Files may be MISMATCHED!")
    elif corr > 0.6:
        print("      ✅ Files appear matched")

    # Check enhanced output
    if Path(enhanced_path).exists():
        enhanced, sr3 = sf.read(enhanced_path, dtype='float32')
        print(f"\n   Enhanced: {len(enhanced)} samples @ {sr3}Hz")
        print(f"      Peak: {np.abs(enhanced).max():.4f}, RMS: {np.sqrt(np.mean(enhanced**2)):.4f}")

        min_len = min(len(enhanced), len(clean), len(noisy))

        corr_enh_clean = np.corrcoef(enhanced[:min_len], clean[:min_len])[0, 1]
        corr_enh_noisy = np.corrcoef(enhanced[:min_len], noisy[:min_len])[0, 1]
        corr_noisy_clean = np.corrcoef(noisy[:min_len], clean[:min_len])[0, 1]

        print(f"\n   Enhanced correlations:")
        print(f"      Enhanced ↔ Clean: {corr_enh_clean:.4f}")
        print(f"      Enhanced ↔ Noisy: {corr_enh_noisy:.4f}")
        print(f"      Noisy ↔ Clean:    {corr_noisy_clean:.4f}")

        if corr_enh_clean > corr_noisy_clean:
            print(f"\n      ✅ Model IMPROVED similarity to clean")
        elif abs(corr_enh_noisy - corr_noisy_clean) < 0.01:
            print(f"\n      ⚠️ Enhanced is nearly identical to noisy (model doing nothing?)")
        else:
            print(f"\n      ⚠️ Model may have DEGRADED quality")

# =============================================================================
# 3. Quick Model Test
# =============================================================================

print("\n" + "=" * 60)
print("🧪 Quick Model Forward Pass Test")
print("=" * 60)

from model_v3 import create_auranet_v3
from utils.stft import CausalSTFT
import yaml

# Load config
with open("config_v3.yaml") as f:
    config = yaml.safe_load(f)

# Create model
model = create_auranet_v3(config)

# Load weights
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)

model.eval()

# Create STFT
stft_cfg = config.get("stft", {})
stft = CausalSTFT(
    n_fft=stft_cfg.get("n_fft", 256),
    hop_length=stft_cfg.get("hop_size", 80),
    win_length=stft_cfg.get("window_size", 160),
)

# Test with synthetic input
print("\n   Testing with synthetic audio...")
x = torch.randn(1, 16000)  # 1 second white noise
x_stft = stft(x)

with torch.no_grad():
    y_stft, _, _ = model(x_stft)
    y = stft.inverse(y_stft)

print(f"   Input:  shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}]")
print(f"   Output: shape={y.shape}, range=[{y.min():.3f}, {y.max():.3f}]")

# Check if output is different from input
diff = (y.squeeze()[:x.shape[-1]] - x.squeeze()).abs().mean()
print(f"   Mean diff from input: {diff:.6f}")

if diff < 0.001:
    print("   ⚠️ Output is nearly IDENTICAL to input (model may be passthrough)")
elif diff > 2.0:
    print("   ⚠️ Output is VERY different (may be corrupted)")
else:
    print("   ✅ Output differs appropriately from input")

# Test with actual noisy file
print("\n   Testing with actual noisy audio...")
noisy_audio = torch.from_numpy(noisy).float().unsqueeze(0)
noisy_stft = stft(noisy_audio)

with torch.no_grad():
    enhanced_stft, _, _ = model(noisy_stft)
    enhanced_audio = stft.inverse(enhanced_stft, length=noisy_audio.shape[-1])

enhanced_np = enhanced_audio.squeeze().numpy()
print(f"   Enhanced: range=[{enhanced_np.min():.3f}, {enhanced_np.max():.3f}]")

# Compare with clean
corr_test = np.corrcoef(enhanced_np[:len(clean)], clean[:len(enhanced_np)])[0, 1]
print(f"   Correlation with clean: {corr_test:.4f}")

# Mask analysis
print("\n   Model mask analysis (is it learning?):")
mask = (enhanced_stft / (noisy_stft + 1e-8)).abs()
print(f"      Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
print(f"      Mask mean:  {mask.mean():.4f}")

if mask.mean() > 0.9 and mask.mean() < 1.1:
    print("      ⚠️ Mask is close to 1.0 (model may not be trained)")
else:
    print("      ✅ Mask shows variation (model is applying transformation)")

print("\n" + "=" * 60)
print("📋 DIAGNOSIS COMPLETE")
print("=" * 60)
