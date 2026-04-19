#!/usr/bin/env python3
"""Test model at different SNR levels with synthetic audio."""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

sys.path.insert(0, '.')
from model_v3 import create_auranet_v3
from utils.stft import CausalSTFT
from metrics import compute_pesq, compute_stoi
import yaml

print("=" * 60)
print("🎧 Model Evaluation at Different SNR Levels")
print("=" * 60)

# Load config
with open("config_v3.yaml") as f:
    config = yaml.safe_load(f)

# Create model
model = create_auranet_v3(config)
ckpt = torch.load('/Users/vinoth-14902/Documents/Models/best_model_v3_1026.pt',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Create STFT
stft_cfg = config.get("stft", {})
stft = CausalSTFT(
    n_fft=stft_cfg.get("n_fft", 256),
    hop_length=stft_cfg.get("hop_size", 80),
    win_length=stft_cfg.get("window_size", 160),
)

# Create synthetic clean speech-like signal
sr = 16000
duration = 3.0
n_samples = int(sr * duration)
t = np.linspace(0, duration, n_samples)

# Simulate speech with harmonics
freq = 200  # F0
clean = np.zeros(n_samples, dtype=np.float32)
for h in range(1, 6):
    clean += (0.7 ** h) * np.sin(2 * np.pi * freq * h * t).astype(np.float32)

# Add envelope
envelope = (np.sin(np.pi * t / duration) ** 2).astype(np.float32)
clean = clean * envelope * 0.5

print("\n📊 Testing at different SNR levels:")
print("-" * 60)
print(f"{'SNR':>5} | {'Input PESQ':>10} | {'Output PESQ':>11} | {'Input STOI':>10} | {'Output STOI':>11}")
print("-" * 60)

for snr_db in [0, 5, 10, 15, 20]:
    # Calculate noise level
    signal_power = np.mean(clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = (np.random.randn(n_samples) * np.sqrt(noise_power)).astype(np.float32)

    noisy = clean + noise
    peak = np.abs(noisy).max()
    if peak > 0:
        noisy = noisy / peak

    # Process through model
    noisy_tensor = torch.from_numpy(noisy).float().unsqueeze(0)
    noisy_stft = stft(noisy_tensor)

    with torch.no_grad():
        enhanced_stft, _, _ = model(noisy_stft)
        enhanced_audio = stft.inverse(enhanced_stft, length=n_samples)

    enhanced = enhanced_audio.squeeze().numpy()
    enhanced = enhanced[:len(clean)]

    # Compute metrics
    pesq_out = compute_pesq(enhanced, clean, sr)
    stoi_out = compute_stoi(enhanced, clean, sr)
    pesq_in = compute_pesq(noisy[:len(clean)], clean, sr)
    stoi_in = compute_stoi(noisy[:len(clean)], clean, sr)

    print(f"{snr_db:>3}dB | {pesq_in:>10.3f} | {pesq_out:>11.3f} | {stoi_in:>10.3f} | {stoi_out:>11.3f}")

print("-" * 60)

# Show training metrics from checkpoint
print("\n📋 Checkpoint training metrics (on real speech):")
print(f"   Epoch:     {ckpt.get('epoch', 'N/A')}")
print(f"   Val PESQ:  {ckpt.get('val_pesq', 0):.3f}")
print(f"   Val STOI:  {ckpt.get('val_stoi', 0):.3f}")
print(f"   Val SI-SNR: {ckpt.get('val_si_snr', 0):.2f} dB")

# Real-time performance test
print("\n⏱️  Real-time Performance Test:")
import time

# Batch processing speed
x = torch.randn(1, sr * 3)  # 3 seconds
x_stft = stft(x)

# Warmup
with torch.no_grad():
    for _ in range(3):
        model(x_stft)

# Benchmark
times = []
for _ in range(10):
    t0 = time.perf_counter()
    with torch.no_grad():
        model(x_stft)
    times.append(time.perf_counter() - t0)

avg_time = np.mean(times) * 1000
rtf = avg_time / 3000  # 3 seconds of audio
print(f"   Batch (3s): {avg_time:.1f}ms ({1/rtf:.1f}x real-time)")

# Frame-by-frame speed
hop = stft_cfg.get("hop_size", 80)
frame_duration_ms = 1000 * hop / sr

frame_times = []
hidden = None
for i in range(100):
    frame = torch.randn(1, 1, hop)
    frame_stft = stft(frame)
    t0 = time.perf_counter()
    with torch.no_grad():
        _, hidden, _ = model(frame_stft, hidden)
    frame_times.append((time.perf_counter() - t0) * 1000)

avg_frame = np.mean(frame_times)
print(f"   Frame ({hop} samples): {avg_frame:.3f}ms (frame={frame_duration_ms:.1f}ms)")

if avg_frame < frame_duration_ms:
    print(f"   ✅ REAL-TIME CAPABLE")
else:
    print(f"   ⚠️  Not real-time on this device")

print("\n" + "=" * 60)
