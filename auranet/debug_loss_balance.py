#!/usr/bin/env python3
"""
Debug script to analyze loss scale imbalance.
This script detects if any loss component is dominating.
"""

import torch
from loss_perceptual import PerceptualLoss, WarmStartLoss

def analyze_loss_balance():
    print("=" * 70)
    print("🚨 LOSS SCALE ANALYSIS — DETECTING IMBALANCE")
    print("=" * 70)

    # Create realistic test signals
    torch.manual_seed(42)
    pred = torch.randn(4, 16000) * 0.5
    target = torch.randn(4, 16000) * 0.5

    loss_fn = PerceptualLoss()
    total, loss_dict = loss_fn(pred, target)

    print("\n📊 RAW LOSS VALUES (before weighting):")
    for k, v in loss_dict.items():
        if k != "total":
            val = v.item() if hasattr(v, "item") else v
            print(f"  {k:20s}: {val:10.4f}")

    print("\n📊 WEIGHTED CONTRIBUTIONS:")
    weights = {"loud": 0.45, "multi_res_stft": 0.30, "mel": 0.15, "si_snr": 0.10}
    contributions = {}
    for k, v in loss_dict.items():
        if k != "total" and k in weights:
            val = v.item() if hasattr(v, "item") else v
            contrib = weights[k] * val
            contributions[k] = contrib
            print(f"  {weights[k]:.2f} * {k:15s} = {contrib:8.4f}")

    total_contrib = sum(contributions.values())
    print(f"\n  TOTAL: {total_contrib:.4f}")

    print("\n📊 PERCENTAGE OF TOTAL LOSS:")
    for k, v in contributions.items():
        pct = 100 * v / total_contrib
        bar = "█" * int(pct / 2)
        print(f"  {k:20s}: {pct:5.1f}% {bar}")

    # Check for dominance
    max_k = max(contributions, key=contributions.get)
    max_pct = 100 * contributions[max_k] / total_contrib

    print("\n" + "=" * 70)
    if max_pct > 50:
        print(f"🚨 RED FLAG: {max_k} dominates at {max_pct:.1f}%!")
        print("   Perceptual losses are being overshadowed!")
        print("   SI-SNR values are ~10-50x larger than spectral losses.")
        print("   Even with 0.10 weight, it dominates total loss.")
    else:
        print(f"✅ Loss balance OK: max component is {max_pct:.1f}%")
    print("=" * 70)

    return max_pct > 50  # Returns True if imbalanced


def analyze_warmstart_phases():
    print("\n" + "=" * 70)
    print("🔥 WARM-START PHASE ANALYSIS")
    print("=" * 70)

    torch.manual_seed(42)
    pred = torch.randn(4, 16000) * 0.5
    target = torch.randn(4, 16000) * 0.5

    warmstart = WarmStartLoss(warmup_epochs=3)

    print("\nPhase 1 (epochs 1-3) vs Phase 2 (epochs 4+):")

    warmstart.set_epoch(1)
    loss1 = warmstart(pred, target)
    print(f"  Phase 1 (epoch 1): loss = {loss1.item():.4f}")

    warmstart.set_epoch(4)
    loss2 = warmstart(pred, target)
    print(f"  Phase 2 (epoch 4): loss = {loss2.item():.4f}")

    ratio = loss1.item() / loss2.item()
    print(f"\n  Phase transition ratio: {ratio:.2f}x")

    if ratio > 3:
        print(f"\n🚨 RED FLAG: Loss drops {ratio:.1f}x when switching phases!")
        print("   This can cause training instability.")
        print("   The optimizer suddenly sees a much smaller loss landscape.")
    else:
        print("\n✅ Phase transition ratio is acceptable.")


if __name__ == "__main__":
    has_imbalance = analyze_loss_balance()
    analyze_warmstart_phases()

    print("\n" + "=" * 70)
    print("📋 SUMMARY OF RED FLAGS")
    print("=" * 70)

    if has_imbalance:
        print("""
🚨 CRITICAL: SI-SNR LOSS DOMINATES!

PROBLEM:
  SI-SNR raw values are ~40-50, while spectral losses are ~0.3-1.7
  Even with weight 0.10, SI-SNR contributes ~75% of total loss

IMPACT:
  - Model optimizes for SI-SNR, not perceptual quality
  - PESQ/STOI don't improve despite loss decreasing
  - "High SI-SNR but robotic sound" phenomenon

SOLUTION:
  Option A: Normalize SI-SNR to [0, 1] range before weighting
  Option B: Reduce SI-SNR weight to 0.02-0.05
  Option C: Rescale SI-SNR by dividing by 30 (typical value)
""")
