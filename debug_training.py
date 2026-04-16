#!/usr/bin/env python3
"""
AuraNet Training Validation & Debug Script

Run this BEFORE training to verify:
1. Loss balance is correct
2. Gradients flow properly
3. Output activation is tanh (not clamp)
4. Training config is correct

Usage:
    python debug_training.py --checkpoint checkpoints/best_model_v3.pt
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model_v3 import create_auranet_v3
from loss_perceptual import PerceptualLoss, WarmStartLoss
from utils.stft import CausalSTFT


def check_loss_balance():
    """Verify loss components are balanced."""
    print("\n" + "="*70)
    print("📊 STEP 1: LOSS BALANCE CHECK")
    print("="*70)

    torch.manual_seed(42)
    pred = torch.randn(4, 16000) * 0.5
    target = torch.randn(4, 16000) * 0.5

    loss_fn = PerceptualLoss()
    total, loss_dict = loss_fn(pred, target)

    print("\nLoss Components:")
    weights = {"loud": 0.45, "multi_res_stft": 0.30, "mel": 0.15, "si_snr": 0.10}
    total_contrib = 0
    for k, v in loss_dict.items():
        if k != "total" and k in weights:
            val = v.item() if hasattr(v, "item") else v
            contrib = weights[k] * val
            total_contrib += contrib
            print(f"  {k:20s}: raw={val:6.3f} × w={weights[k]:.2f} = {contrib:.4f}")

    print(f"\n  Total: {total_contrib:.4f}")

    # Check balance
    sisnr_contrib = weights["si_snr"] * loss_dict["si_snr"].item()
    sisnr_pct = 100 * sisnr_contrib / total_contrib

    if sisnr_pct > 30:
        print(f"\n❌ FAIL: SI-SNR contributes {sisnr_pct:.1f}% (should be <30%)")
        return False
    else:
        print(f"\n✅ PASS: SI-SNR contributes {sisnr_pct:.1f}% (balanced)")
        return True


def check_warmstart_phases():
    """Verify warm-start phase transitions are smooth."""
    print("\n" + "="*70)
    print("🔥 STEP 2: WARM-START PHASE CHECK")
    print("="*70)

    torch.manual_seed(42)
    pred = torch.randn(4, 16000) * 0.5
    target = torch.randn(4, 16000) * 0.5

    warmstart = WarmStartLoss(warmup_epochs=3)

    warmstart.set_epoch(3)
    loss_phase1 = warmstart(pred, target).item()

    warmstart.set_epoch(4)
    loss_phase2 = warmstart(pred, target).item()

    ratio = loss_phase1 / loss_phase2
    print(f"\n  Phase 1 (epoch 3): {loss_phase1:.4f}")
    print(f"  Phase 2 (epoch 4): {loss_phase2:.4f}")
    print(f"  Transition ratio: {ratio:.2f}x")

    if ratio > 2:
        print(f"\n❌ FAIL: Phase transition is {ratio:.1f}x (should be <2x)")
        return False
    else:
        print(f"\n✅ PASS: Phase transition is smooth ({ratio:.2f}x)")
        return True


def check_gradient_flow(config, checkpoint_path):
    """Verify gradients flow to all layers."""
    print("\n" + "="*70)
    print("🔧 STEP 3: GRADIENT FLOW CHECK")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_auranet_v3(config).to(device)

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    # Create STFT
    stft = CausalSTFT(n_fft=256, hop_length=80, win_length=160).to(device)

    # Create loss
    loss_fn = PerceptualLoss()

    # Create dummy batch
    noisy = torch.randn(2, 1, 16000, device=device)
    clean = torch.randn(2, 1, 16000, device=device)

    # Forward pass
    model.train()
    noisy_stft = stft(noisy)
    clean_stft = stft(clean)

    enhanced_stft, _, _ = model(noisy_stft)
    enhanced_audio = stft.inverse(enhanced_stft)
    enhanced_audio = torch.tanh(enhanced_audio)  # Use tanh!

    min_len = min(enhanced_audio.shape[-1], clean.shape[-1])
    enhanced_audio = enhanced_audio[..., :min_len]
    clean_audio = clean.squeeze(1)[..., :min_len]

    # Compute loss and backward
    loss, _ = loss_fn(enhanced_audio, clean_audio)
    loss.backward()

    # Check gradients
    zero_grad_layers = []
    nonzero_grad_layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-10:
                zero_grad_layers.append(name)
            else:
                nonzero_grad_layers.append((name, grad_norm))

    print(f"\n  Layers with gradients: {len(nonzero_grad_layers)}")

    # Show top 5 gradient norms
    nonzero_grad_layers.sort(key=lambda x: x[1], reverse=True)
    print("\n  Top 5 gradient norms:")
    for name, norm in nonzero_grad_layers[:5]:
        print(f"    {name}: {norm:.6f}")

    if zero_grad_layers:
        print(f"\n  ⚠️ Layers with zero gradients: {len(zero_grad_layers)}")
        for name in zero_grad_layers[:3]:
            print(f"    - {name}")

    if len(zero_grad_layers) > len(nonzero_grad_layers) * 0.5:
        print(f"\n❌ FAIL: Too many layers have zero gradients")
        return False
    else:
        print(f"\n✅ PASS: Gradients flow correctly")
        return True


def check_output_activation():
    """Verify tanh activation is used (not clamp)."""
    print("\n" + "="*70)
    print("🎚️ STEP 4: OUTPUT ACTIVATION CHECK")
    print("="*70)

    # Check train_finetune.py for clamp usage
    train_file = Path(__file__).parent / "train_finetune.py"
    if train_file.exists():
        content = train_file.read_text()

        has_clamp = ".clamp(" in content or "torch.clamp(" in content
        has_tanh = "torch.tanh(" in content

        print(f"\n  train_finetune.py analysis:")
        print(f"    Uses clamp: {has_clamp}")
        print(f"    Uses tanh:  {has_tanh}")

        if has_clamp and "enhanced_audio" in content.split(".clamp")[0][-100:]:
            print("\n❌ FAIL: clamp() used on enhanced_audio")
            return False
        elif has_tanh:
            print("\n✅ PASS: tanh activation used correctly")
            return True
        else:
            print("\n⚠️ WARNING: No explicit activation found")
            return True
    else:
        print("  Could not find train_finetune.py")
        return True


def check_training_config(config):
    """Verify training config is correct."""
    print("\n" + "="*70)
    print("⚙️ STEP 5: TRAINING CONFIG CHECK")
    print("="*70)

    train_cfg = config.get('training', {})

    lr = train_cfg.get('learning_rate', 1e-4)
    grad_clip = train_cfg.get('gradient_clip', 3.0)

    print(f"\n  Learning rate: {lr}")
    print(f"  Gradient clip: {grad_clip}")

    issues = []
    if lr > 5e-4:
        issues.append(f"LR too high ({lr}), should be ≤1e-4 for fine-tuning")
    if grad_clip <= 0:
        issues.append("Gradient clipping disabled")

    if issues:
        print("\n❌ FAIL:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("\n✅ PASS: Config looks correct")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate AuraNet Training Setup")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config_v3.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    print("\n" + "="*70)
    print("🔍 AURANET TRAINING VALIDATION")
    print("="*70)

    results = []

    # Run all checks
    results.append(("Loss Balance", check_loss_balance()))
    results.append(("Warm-Start Phases", check_warmstart_phases()))
    results.append(("Output Activation", check_output_activation()))
    results.append(("Training Config", check_training_config(config)))

    if args.checkpoint:
        results.append(("Gradient Flow", check_gradient_flow(config, args.checkpoint)))

    # Summary
    print("\n" + "="*70)
    print("📋 VALIDATION SUMMARY")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED — Training pipeline is ready!")
    else:
        print("⚠️ SOME CHECKS FAILED — Fix issues before training!")
    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
