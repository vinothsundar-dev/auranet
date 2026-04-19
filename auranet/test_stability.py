#!/usr/bin/env python3
"""Numerical stability validation for all loss components."""
import torch
from loss_v3 import (
    SISNRLoss, CompressedMSELoss, MultiResSTFTLoss,
    EnergyPreservationLoss, LogMelLoss, TemporalConsistencyLoss,
    AuraNetV3Loss, loudness_normalize,
)

torch.manual_seed(42)
B, N = 4, 32000
all_finite = True

pred = torch.randn(B, N) * 0.3
target = torch.randn(B, N) * 0.3
pred_zero = torch.zeros(B, N)
pred_tiny = torch.ones(B, N) * 1e-8
pred_large = torch.ones(B, N) * 10.0

pred_stft = torch.randn(B, 2, 100, 129) * 0.1
tgt_stft = torch.randn(B, 2, 100, 129) * 0.1


def check(name, val):
    global all_finite
    ok = torch.isfinite(val).item() if val.dim() == 0 else torch.isfinite(val).all().item()
    all_finite = all_finite and ok
    v = val.item() if val.dim() == 0 else "tensor"
    print(f"  {name:30s}  finite={ok}  val={v}")
    return ok


print("=== SI-SNR Loss ===")
l = SISNRLoss()
check("normal", l(pred, target))
check("zero pred", l(pred_zero, target))
check("tiny pred", l(pred_tiny, target))
check("large pred", l(pred_large, target))

print("\n=== Compressed MSE Loss ===")
l = CompressedMSELoss()
check("normal", l(pred_stft, tgt_stft))
check("zero pred", l(torch.zeros(B, 2, 100, 129), tgt_stft))

print("\n=== Multi-Res STFT Loss ===")
l = MultiResSTFTLoss()
check("normal", l(pred, target))
check("zero pred", l(pred_zero, target))
check("tiny pred", l(pred_tiny, target))

print("\n=== Energy Preservation Loss ===")
l = EnergyPreservationLoss()
v = l(pred, target)
check("normal", v)
assert v.item() <= 10.0, f"Energy loss not clamped: {v.item()}"
v2 = l(pred_zero, target)
check("zero pred", v2)
assert v2.item() <= 10.0, f"Energy loss not clamped: {v2.item()}"
check("large pred", l(pred_large, target))

print("\n=== LogMel Loss ===")
l = LogMelLoss()
check("normal", l(pred, target))
check("zero pred", l(pred_zero, target))
check("tiny pred", l(pred_tiny, target))

print("\n=== Temporal Consistency Loss ===")
l = TemporalConsistencyLoss()
check("normal", l(pred_stft, tgt_stft))
check("zero pred", l(torch.zeros(B, 2, 100, 129), tgt_stft))

print("\n=== Combined AuraNetV3Loss ===")
combined = AuraNetV3Loss()
total, d = combined(pred_stft, tgt_stft, pred, target)
check("normal total", total)
for k, v in d.items():
    check(f"  {k}", v)

total2, d2 = combined(torch.zeros(B, 2, 100, 129), tgt_stft, pred_zero, target)
check("zero-pred total", total2)
for k, v in d2.items():
    check(f"  {k}", v)

print("\n=== loudness_normalize ===")
check("normal", loudness_normalize(pred, target))
check("zero pred", loudness_normalize(pred_zero, target))

print()
if all_finite:
    print("=" * 50)
    print("ALL TESTS PASSED - every loss component is finite")
    print("=" * 50)
else:
    print("SOME TESTS FAILED")
