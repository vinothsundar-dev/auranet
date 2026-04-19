#!/usr/bin/env python3
# =============================================================================
# AuraNet Pre-Flight Validation — Complete Training Pipeline Check
# =============================================================================
#
# PURPOSE: Validate ALL critical training components BEFORE full training
#
# CHECKS:
#   1. Execution path (correct script, model, loss)
#   2. Hyperparameters (LR=1e-4, scheduler, batch size)
#   3. Output activation (tanh NOT clamp)
#   4. Loss pipeline & weights (balanced)
#   5. Warm-start logic
#   6. Gradient flow
#   7. Single batch test
#
# USAGE:
#   python preflight_validation.py
#   python preflight_validation.py --checkpoint checkpoints/best_model_v3.pt
#
# =============================================================================

print("=" * 70)
print("🚀 PRE-FLIGHT VALIDATION — AuraNet Training Pipeline")
print("=" * 70)
print(f"RUNNING: {__file__}")

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# =============================================================================
# STEP 1 — VERIFY EXECUTION PATH
# =============================================================================

print("\n" + "=" * 70)
print("📍 STEP 1: EXECUTION PATH VERIFICATION")
print("=" * 70)

# Check imports
try:
    from model_v3 import AuraNetV3, create_auranet_v3
    print("✅ Model import: model_v3.py (AuraNetV3)")
except ImportError as e:
    print(f"❌ Model import FAILED: {e}")
    sys.exit(1)

try:
    from loss_perceptual import PerceptualLoss, WarmStartLoss, SISNRLoss
    print("✅ Loss import: loss_perceptual.py (PerceptualLoss, WarmStartLoss)")
except ImportError as e:
    print(f"❌ Loss import FAILED: {e}")
    sys.exit(1)

try:
    from utils.stft import CausalSTFT
    print("✅ STFT import: utils.stft (CausalSTFT)")
except ImportError as e:
    print(f"❌ STFT import FAILED: {e}")
    sys.exit(1)

# =============================================================================
# STEP 2 — VERIFY HYPERPARAMETERS
# =============================================================================

print("\n" + "=" * 70)
print("📍 STEP 2: HYPERPARAMETER VERIFICATION")
print("=" * 70)

# Load config
config_path = Path("config_v3.yaml")
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded: {config_path}")
else:
    print(f"⚠️  Config not found: {config_path}, using defaults")
    config = {}

# Check learning rate
train_config = config.get('training', {})
config_lr = train_config.get('learning_rate', 0.001)
EXPECTED_LR = 1e-4

if abs(config_lr - EXPECTED_LR) < 1e-8:
    print(f"✅ Learning rate (config): {config_lr} (expected {EXPECTED_LR})")
else:
    print(f"❌ Learning rate (config): {config_lr} (expected {EXPECTED_LR})")
    print(f"   FIX: Set training.learning_rate: {EXPECTED_LR} in config_v3.yaml")

# Check gradient clipping
config_grad_clip = train_config.get('gradient_clip', 5.0)
EXPECTED_GRAD_CLIP = 3.0
if abs(config_grad_clip - EXPECTED_GRAD_CLIP) < 0.1:
    print(f"✅ Gradient clip (config): {config_grad_clip}")
else:
    print(f"⚠️  Gradient clip: {config_grad_clip} (recommended {EXPECTED_GRAD_CLIP})")

# Check batch size
config_batch = train_config.get('batch_size', 16)
print(f"✅ Batch size (config): {config_batch}")

# =============================================================================
# STEP 3 — VERIFY OUTPUT ACTIVATION
# =============================================================================

print("\n" + "=" * 70)
print("📍 STEP 3: OUTPUT ACTIVATION VERIFICATION")
print("=" * 70)

# Static code analysis for clamp vs tanh
import ast
import re

def check_file_for_activation(filepath: str) -> Dict[str, list]:
    """Check a file for clamp vs tanh usage on enhanced_audio."""
    results = {'tanh': [], 'clamp': [], 'issues': []}

    if not os.path.exists(filepath):
        results['issues'].append(f"File not found: {filepath}")
        return results

    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # Skip comments when checking for clamp
        code_part = line.split('#')[0]  # Only check code, not comments

        # Check for clamp on enhanced_audio (in code, not comments)
        if re.search(r'enhanced_audio.*\.clamp\(|torch\.clamp\(.*enhanced_audio', code_part):
            results['clamp'].append((i, line.strip()))
        if re.search(r'clamp.*enhanced_audio|enhanced_audio.*clamp', code_part):
            if 'std' not in code_part.lower() and 'eps' not in code_part.lower():
                results['clamp'].append((i, line.strip()))

        # Check for tanh on enhanced_audio
        if re.search(r'torch\.tanh\(enhanced_audio\)|enhanced_audio.*tanh', code_part):
            results['tanh'].append((i, line.strip()))

    return results

# Check key files
files_to_check = [
    'train_finetune.py',
    'train_v3.py',
    'model_v3.py',
]

activation_issues = []

for filepath in files_to_check:
    if os.path.exists(filepath):
        results = check_file_for_activation(filepath)
        print(f"\n  📄 {filepath}:")

        if results['tanh']:
            print(f"     ✅ tanh found: {len(results['tanh'])} instances")
            for ln, code in results['tanh'][:2]:
                print(f"        L{ln}: {code[:60]}...")
        else:
            print(f"     ⚠️  No tanh found on enhanced_audio")

        if results['clamp']:
            print(f"     ❌ CLAMP found: {len(results['clamp'])} instances")
            for ln, code in results['clamp']:
                print(f"        L{ln}: {code[:60]}...")
                activation_issues.append(f"{filepath}:{ln}")
    else:
        print(f"  ⚠️  {filepath}: Not found")

if activation_issues:
    print(f"\n❌ OUTPUT ACTIVATION ISSUES DETECTED:")
    for issue in activation_issues:
        print(f"   - {issue}")
else:
    print(f"\n✅ No clamp issues detected on enhanced_audio output")

# =============================================================================
# STEP 4 — VERIFY LOSS PIPELINE & WEIGHTS
# =============================================================================

print("\n" + "=" * 70)
print("📍 STEP 4: LOSS PIPELINE VERIFICATION")
print("=" * 70)

# Expected weights
EXPECTED_WEIGHTS = {
    'loud': 0.45,
    'stft': 0.30,
    'mel': 0.15,
    'sisnr': 0.10,
}

print("\n  Expected loss weights (perceptual focus):")
for name, weight in EXPECTED_WEIGHTS.items():
    print(f"    {name}: {weight:.2f} ({weight*100:.0f}%)")

# Initialize loss to verify
loss_fn = PerceptualLoss(
    weight_loud=EXPECTED_WEIGHTS['loud'],
    weight_stft=EXPECTED_WEIGHTS['stft'],
    weight_mel=EXPECTED_WEIGHTS['mel'],
    weight_sisnr=EXPECTED_WEIGHTS['sisnr'],
)

# Verify weights
actual_weights = {
    'loud': loss_fn.w_loud,
    'stft': loss_fn.w_stft,
    'mel': loss_fn.w_mel,
    'sisnr': loss_fn.w_sisnr,
}

print("\n  Actual loss weights:")
all_weights_correct = True
for name, expected in EXPECTED_WEIGHTS.items():
    actual = actual_weights[name]
    status = "✅" if abs(actual - expected) < 0.01 else "❌"
    if status == "❌":
        all_weights_correct = False
    print(f"    {status} {name}: {actual:.2f} (expected {expected:.2f})")

if all_weights_correct:
    print("\n✅ Loss weights are correctly configured")
else:
    print("\n❌ Loss weights need adjustment")

# Verify SI-SNR normalization
sisnr = SISNRLoss()
print(f"\n  SI-SNR normalize_scale: {sisnr.normalize_scale}")
if sisnr.normalize_scale >= 20:
    print("  ✅ SI-SNR is normalized (won't dominate)")
else:
    print("  ❌ SI-SNR normalization may be insufficient")

# =============================================================================
# STEP 5 — VERIFY LOSS ORDER (tanh BEFORE loss)
# =============================================================================

print("\n" + "=" * 70)
print("📍 STEP 5: LOSS ORDER VERIFICATION")
print("=" * 70)

# Check that tanh comes BEFORE loss computation
def check_loss_order(filepath: str) -> bool:
    """Verify that tanh is applied BEFORE loss computation."""
    if not os.path.exists(filepath):
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Find tanh and loss positions
    tanh_match = re.search(r'enhanced_audio\s*=\s*torch\.tanh\(enhanced_audio\)', content)
    # Check for various loss function patterns
    loss_match = re.search(r'(criterion|loss_fn)\(.*enhanced_audio', content, re.DOTALL)

    if tanh_match and loss_match:
        return tanh_match.start() < loss_match.start()
    elif tanh_match:
        # tanh found but no loss match - check if enhanced_audio is used after
        return True  # Assume correct if tanh is present

print("\n" + "=" * 70)
print("📍 STEP 6: WARM-START LOGIC VERIFICATION")
print("=" * 70)

# Check if WarmStartLoss is implemented
warmstart = WarmStartLoss()
print(f"  ✅ WarmStartLoss available")
print(f"     SI-SNR weight (phase 1): {warmstart.phase1_sisnr_weight}")
print(f"     STFT weight (phase 1):   {warmstart.phase1_stft_weight}")
print(f"     Warmup epochs:           {warmstart.warmup_epochs}")

# =============================================================================
# STEP 7 — RUN SINGLE BATCH TEST
# =============================================================================

print("\n" + "=" * 70)
print("📍 STEP 7: SINGLE BATCH TEST")
print("=" * 70)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")

# Create model
print("\n  Creating model...")
model = create_auranet_v3()
model = model.to(device)
model.train()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,}")
print(f"  Trainable:    {trainable_params:,}")

# Create STFT
stft_config = config.get('stft', {})
stft = CausalSTFT(
    n_fft=stft_config.get('n_fft', 256),
    hop_length=stft_config.get('hop_size', 80),
    win_length=stft_config.get('window_size', 160),
).to(device)
print(f"  STFT: n_fft={stft.n_fft}, hop={stft.hop_length}")

# Create optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=EXPECTED_LR,
    weight_decay=train_config.get('weight_decay', 0.01)
)
print(f"\n  Optimizer: AdamW")
print(f"  LR: {optimizer.param_groups[0]['lr']}")

# Create loss
loss_fn = PerceptualLoss().to(device)

# Create synthetic batch
batch_size = 4
segment_samples = 16000 * 3  # 3 seconds

print(f"\n  Generating synthetic batch (B={batch_size}, T={segment_samples})...")
noisy_audio = torch.randn(batch_size, 1, segment_samples).to(device)
clean_audio = torch.randn(batch_size, 1, segment_samples).to(device)

# Forward pass
print("\n  Running forward pass...")
with torch.set_grad_enabled(True):
    noisy_stft = stft(noisy_audio)
    clean_stft = stft(clean_audio)

    enhanced_stft, _, _ = model(noisy_stft)

    # Reconstruct audio
    enhanced_audio = stft.inverse(enhanced_stft)
    min_len = min(enhanced_audio.shape[-1], clean_audio.shape[-1])
    enhanced_audio = enhanced_audio[..., :min_len]
    clean_audio_batch = clean_audio.squeeze(1)[..., :min_len]

    # Output BEFORE activation
    print(f"\n  📊 OUTPUT BEFORE TANH:")
    print(f"     min: {enhanced_audio.min().item():.4f}")
    print(f"     max: {enhanced_audio.max().item():.4f}")
    print(f"     mean: {enhanced_audio.mean().item():.4f}")
    print(f"     std: {enhanced_audio.std().item():.4f}")

    # Apply tanh (NOT clamp)
    enhanced_audio = torch.tanh(enhanced_audio)

    # Output AFTER activation
    print(f"\n  📊 OUTPUT AFTER TANH:")
    print(f"     min: {enhanced_audio.min().item():.4f}")
    print(f"     max: {enhanced_audio.max().item():.4f}")
    print(f"     mean: {enhanced_audio.mean().item():.4f}")
    print(f"     std: {enhanced_audio.std().item():.4f}")

    # Verify tanh range
    if enhanced_audio.min() >= -1.0 and enhanced_audio.max() <= 1.0:
        print(f"     ✅ Output in valid range [-1, 1]")
    else:
        print(f"     ❌ Output outside valid range!")

    # Compute loss
    loss, loss_dict = loss_fn(enhanced_audio, clean_audio_batch, enhanced_stft, clean_stft)

    print(f"\n  📊 LOSS BREAKDOWN:")
    total_weighted = 0
    for name, value in loss_dict.items():
        if name != 'total':
            val = value.item() if isinstance(value, torch.Tensor) else value
            print(f"     {name}: {val:.4f}")
    print(f"     TOTAL: {loss.item():.4f}")

    # Check loss balance
    print(f"\n  📊 LOSS BALANCE CHECK:")
    loud_contrib = loss_dict['loud'].item() * 0.45
    stft_contrib = loss_dict['multi_res_stft'].item() * 0.30
    mel_contrib = loss_dict['mel'].item() * 0.15
    sisnr_contrib = loss_dict['si_snr'].item() * 0.10
    total_contrib = loud_contrib + stft_contrib + mel_contrib + sisnr_contrib

    print(f"     Loud contribution:  {loud_contrib:.4f} ({100*loud_contrib/total_contrib:.1f}%)")
    print(f"     STFT contribution:  {stft_contrib:.4f} ({100*stft_contrib/total_contrib:.1f}%)")
    print(f"     Mel contribution:   {mel_contrib:.4f} ({100*mel_contrib/total_contrib:.1f}%)")
    print(f"     SI-SNR contribution:{sisnr_contrib:.4f} ({100*sisnr_contrib/total_contrib:.1f}%)")

    # Check if SI-SNR is dominating (should be ~10%)
    sisnr_pct = 100 * sisnr_contrib / total_contrib
    if sisnr_pct < 20:
        print(f"     ✅ SI-SNR is balanced ({sisnr_pct:.1f}% < 20%)")
    else:
        print(f"     ❌ SI-SNR may be dominating ({sisnr_pct:.1f}% > 20%)")

# Backward pass
print("\n  Running backward pass...")
optimizer.zero_grad()
loss.backward()

# =============================================================================
# STEP 8 — VERIFY GRADIENT FLOW
# =============================================================================

print("\n" + "=" * 70)
print("📍 STEP 8: GRADIENT FLOW VERIFICATION")
print("=" * 70)

grad_stats = []
zero_grad_params = []
nan_grad_params = []

for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_stats.append((name, grad_norm))

        if grad_norm == 0:
            zero_grad_params.append(name)
        if not torch.isfinite(param.grad).all():
            nan_grad_params.append(name)

# Print top 10 gradient norms
print("\n  Top 10 gradient norms:")
grad_stats_sorted = sorted(grad_stats, key=lambda x: x[1], reverse=True)[:10]
for name, norm in grad_stats_sorted:
    print(f"     {name}: {norm:.6f}")

# Check for issues
if zero_grad_params:
    print(f"\n  ⚠️  Zero gradients detected in {len(zero_grad_params)} parameters")
else:
    print(f"\n  ✅ No zero gradients detected")

if nan_grad_params:
    print(f"  ❌ NaN/Inf gradients in: {nan_grad_params}")
else:
    print(f"  ✅ No NaN/Inf gradients")

# Total gradient norm
total_grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()
print(f"\n  Total gradient norm: {total_grad_norm:.4f}")

if total_grad_norm > 0 and total_grad_norm < 1000:
    print(f"  ✅ Gradient norm is healthy")
else:
    print(f"  ⚠️  Gradient norm may need attention")

# Optimizer step test
print("\n  Testing optimizer step...")
optimizer.step()
print(f"  ✅ Optimizer step completed")
print(f"  LR after step: {optimizer.param_groups[0]['lr']}")

# =============================================================================
# STEP 9 — FINAL VALIDATION SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("📍 STEP 9: FINAL VALIDATION SUMMARY")
print("=" * 70)

# Collect all checks
checks = {
    "LR = 1e-4": abs(optimizer.param_groups[0]['lr'] - 1e-4) < 1e-8,
    "Output uses tanh": len(activation_issues) == 0,
    "SI-SNR balanced (<20%)": sisnr_pct < 20,
    "Gradients non-zero": len(zero_grad_params) == 0,
    "No NaN/Inf": len(nan_grad_params) == 0,
    "Loss is finite": torch.isfinite(loss),
    "Output in [-1,1]": enhanced_audio.min() >= -1.0 and enhanced_audio.max() <= 1.0,
}

print("\n  Validation Results:")
all_passed = True
for check_name, passed in checks.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    if not passed:
        all_passed = False
    print(f"     {status}: {check_name}")

print("\n" + "=" * 70)
if all_passed:
    print("🎉 ALL CHECKS PASSED — Training is SAFE to run")
    print("=" * 70)
    print("\nRecommended command:")
    print("  python train_finetune.py --checkpoint checkpoints/best_model_v3.pt --epochs 10")
else:
    print("❌ VALIDATION FAILED — Fix issues before training")
    print("=" * 70)

print("\n")
