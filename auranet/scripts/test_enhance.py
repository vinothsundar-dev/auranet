#!/usr/bin/env python3
"""
AuraNet Audio Test — Load noisy audio, enhance it, compare, and plot.

Usage:
    python scripts/test_enhance.py path/to/noisy.wav
    python scripts/test_enhance.py path/to/noisy.wav --model checkpoints/best_model.pt
    python scripts/test_enhance.py path/to/noisy.wav --output enhanced.wav
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer import AuraNetInference


def load_audio_file(path, target_sr=16000):
    """Load and resample audio to mono 16kHz."""
    data, sr = sf.read(str(path), dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=-1)
    if sr != target_sr:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform.unsqueeze(0)).squeeze(0)
    return waveform, target_sr


def plot_comparison(noisy, enhanced, sr, save_path="outputs/comparison.png"):
    """Plot waveforms, spectrograms, and difference."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    noisy_np = noisy.numpy()
    enhanced_np = enhanced.numpy()
    diff_np = noisy_np - enhanced_np  # removed noise

    duration = len(noisy_np) / sr
    t = np.linspace(0, duration, len(noisy_np))

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle("AuraNet Enhancement — Before vs After", fontsize=16, fontweight="bold")

    # --- Row 1: Waveforms ---
    axes[0, 0].plot(t, noisy_np, color="#e74c3c", linewidth=0.3, alpha=0.8)
    axes[0, 0].set_title("Noisy Input (Waveform)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlim(0, duration)

    axes[0, 1].plot(t, enhanced_np, color="#2ecc71", linewidth=0.3, alpha=0.8)
    axes[0, 1].set_title("Enhanced Output (Waveform)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].set_xlim(0, duration)

    # --- Row 2: Spectrograms ---
    n_fft = 512
    hop = 160

    spec_noisy = np.abs(np.array([
        np.fft.rfft(noisy_np[i:i + n_fft] * np.hanning(n_fft))
        for i in range(0, len(noisy_np) - n_fft, hop)
    ])).T
    spec_enhanced = np.abs(np.array([
        np.fft.rfft(enhanced_np[i:i + n_fft] * np.hanning(n_fft))
        for i in range(0, len(enhanced_np) - n_fft, hop)
    ])).T

    vmin = -60
    vmax = 0

    axes[1, 0].imshow(
        20 * np.log10(spec_noisy + 1e-8), aspect="auto", origin="lower",
        cmap="magma", vmin=vmin, vmax=vmax,
        extent=[0, duration, 0, sr / 2 / 1000]
    )
    axes[1, 0].set_title("Noisy Spectrogram")
    axes[1, 0].set_ylabel("Frequency (kHz)")

    axes[1, 1].imshow(
        20 * np.log10(spec_enhanced + 1e-8), aspect="auto", origin="lower",
        cmap="magma", vmin=vmin, vmax=vmax,
        extent=[0, duration, 0, sr / 2 / 1000]
    )
    axes[1, 1].set_title("Enhanced Spectrogram")
    axes[1, 1].set_ylabel("Frequency (kHz)")

    # --- Row 3: Removed noise ---
    axes[2, 0].plot(t, diff_np, color="#f39c12", linewidth=0.3, alpha=0.8)
    axes[2, 0].set_title("Removed Noise (Waveform)")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].set_xlim(0, duration)

    spec_diff = np.abs(np.array([
        np.fft.rfft(diff_np[i:i + n_fft] * np.hanning(n_fft))
        for i in range(0, len(diff_np) - n_fft, hop)
    ])).T

    axes[2, 1].imshow(
        20 * np.log10(spec_diff + 1e-8), aspect="auto", origin="lower",
        cmap="magma", vmin=vmin, vmax=vmax,
        extent=[0, duration, 0, sr / 2 / 1000]
    )
    axes[2, 1].set_title("Removed Noise (Spectrogram)")
    axes[2, 1].set_ylabel("Frequency (kHz)")

    # --- Row 4: Stats ---
    axes[3, 0].axis("off")
    axes[3, 1].axis("off")

    rms_noisy = np.sqrt(np.mean(noisy_np ** 2))
    rms_enhanced = np.sqrt(np.mean(enhanced_np ** 2))
    rms_diff = np.sqrt(np.mean(diff_np ** 2))

    # SI-SDR (using enhanced as estimate, noisy as reference isn't standard,
    # but shows how much the signal changed)
    noise_energy = np.sum(diff_np ** 2)
    signal_energy = np.sum(enhanced_np ** 2)
    snr_improvement = 10 * np.log10(signal_energy / (noise_energy + 1e-8))

    stats_text = (
        f"Duration: {duration:.2f}s  |  Sample Rate: {sr} Hz\n\n"
        f"RMS Noisy:    {rms_noisy:.4f}\n"
        f"RMS Enhanced: {rms_enhanced:.4f}\n"
        f"RMS Removed:  {rms_diff:.4f}\n\n"
        f"Noise Removed: {rms_diff / (rms_noisy + 1e-8) * 100:.1f}% of input energy\n"
        f"Output SNR:    {snr_improvement:.1f} dB"
    )
    axes[3, 0].text(
        0.1, 0.5, stats_text, fontsize=13, fontfamily="monospace",
        verticalalignment="center", transform=axes[3, 0].transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8)
    )

    for ax in axes.flat:
        if ax.get_xlabel() == "":
            ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Plot saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test AuraNet enhancement on a noisy audio file")
    parser.add_argument("input", help="Path to noisy audio file (.wav)")
    parser.add_argument("--model", default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--output", default=None, help="Output enhanced .wav path")
    parser.add_argument("--plot", default="outputs/comparison.png", help="Plot output path")
    parser.add_argument("--device", default=None, help="Device (auto if omitted)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = REPO_ROOT / "outputs" / f"{input_path.stem}_enhanced.wav"

    # Load model
    print("🔧 Loading model...")
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    enhancer = AuraNetInference(model_path=str(model_path), device=device)

    # Load noisy audio
    print(f"🔊 Loading: {input_path}")
    noisy, sr = load_audio_file(input_path)
    print(f"   Duration: {len(noisy) / sr:.2f}s, Samples: {len(noisy)}")

    # Enhance
    print("🔄 Enhancing...")
    enhanced = enhancer.process_file(str(input_path), str(output_path))
    if enhanced.dim() > 1:
        enhanced = enhanced.squeeze()

    # Trim to same length
    min_len = min(len(noisy), len(enhanced))
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len].cpu()

    # Save noisy copy for comparison
    noisy_copy_path = REPO_ROOT / "outputs" / f"{input_path.stem}_noisy.wav"
    noisy_copy_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(noisy_copy_path), noisy.numpy(), sr)

    # Save removed noise
    removed_noise = noisy - enhanced
    noise_path = REPO_ROOT / "outputs" / f"{input_path.stem}_removed_noise.wav"
    sf.write(str(noise_path), removed_noise.numpy(), sr)

    # Plot
    plot_path = Path(args.plot)
    if not plot_path.is_absolute():
        plot_path = REPO_ROOT / plot_path
    plot_comparison(noisy, enhanced, sr, save_path=str(plot_path))

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ RESULTS")
    print(f"{'='*60}")
    print(f"   Noisy input:    {noisy_copy_path}")
    print(f"   Enhanced:       {output_path}")
    print(f"   Removed noise:  {noise_path}")
    print(f"   Plot:           {plot_path}")
    print(f"\n🎧 To listen:")
    print(f"   open {noisy_copy_path}")
    print(f"   open {output_path}")


if __name__ == "__main__":
    main()
