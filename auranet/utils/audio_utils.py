# =============================================================================
# Audio Utilities for AuraNet
# =============================================================================
#
# Handles audio I/O, normalization, and mixing for training data generation.
# =============================================================================

import torch
import torch.nn.functional as F
import soundfile as sf
from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np


def load_audio(
    path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if necessary.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True

    Returns:
        Tuple of (waveform [1, N] or [C, N], sample_rate)
    """
    data, sr = sf.read(str(path), dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T  # [samples, channels] -> [channels, samples]

    # Convert to mono if requested
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform, sample_rate


def save_audio(
    waveform: torch.Tensor,
    path: Union[str, Path],
    sample_rate: int = 16000,
) -> None:
    """
    Save waveform to audio file.

    Args:
        waveform: Audio tensor [1, N] or [N]
        path: Output path
        sample_rate: Sample rate
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Ensure CPU tensor
    if waveform.is_cuda:
        waveform = waveform.cpu()

    # Ensure float32 and proper range [-1, 1]
    waveform = waveform.float()
    waveform = torch.clamp(waveform, -1.0, 1.0)

    sf.write(str(path), waveform.squeeze(0).numpy(), sample_rate)


def normalize_audio(
    waveform: torch.Tensor,
    target_db: float = -23.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize audio to target loudness (RMS-based).

    Args:
        waveform: Audio tensor
        target_db: Target RMS level in dB
        eps: Small constant for numerical stability

    Returns:
        Normalized waveform
    """
    rms = torch.sqrt(torch.mean(waveform ** 2) + eps)
    current_db = 20 * torch.log10(rms + eps)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    return waveform * gain


def compute_rms(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute RMS (Root Mean Square) of waveform."""
    return torch.sqrt(torch.mean(waveform ** 2) + eps)


def compute_snr(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.

    Args:
        clean: Clean signal
        noisy: Noisy signal (clean + noise)
        eps: Small constant for stability

    Returns:
        SNR in dB
    """
    noise = noisy - clean
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)

    snr = 10 * torch.log10(signal_power / (noise_power + eps) + eps)
    return snr.item()


def mix_audio_with_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mix clean audio with noise at specified SNR.

    BIOMIMETIC DESIGN NOTE:
    - We preserve the clean signal exactly and scale the noise
    - This maintains the natural dynamics of the speech/music
    - Noise is scaled to achieve target SNR

    Args:
        clean: Clean audio [1, N] or [N]
        noise: Noise audio [1, M] (will be repeated/trimmed to match clean)
        snr_db: Target SNR in dB
        eps: Small constant for stability

    Returns:
        Tuple of (noisy_mix, scaled_noise)
    """
    # Ensure same shape
    if clean.dim() == 1:
        clean = clean.unsqueeze(0)
    if noise.dim() == 1:
        noise = noise.unsqueeze(0)

    clean_len = clean.shape[-1]
    noise_len = noise.shape[-1]

    # Handle noise length mismatch
    if noise_len < clean_len:
        # Repeat noise to match clean length
        repeats = (clean_len // noise_len) + 1
        noise = noise.repeat(1, repeats)

    # Trim or randomly crop noise to match clean length
    if noise.shape[-1] > clean_len:
        start = torch.randint(0, noise.shape[-1] - clean_len + 1, (1,)).item()
        noise = noise[:, start:start + clean_len]

    # Compute RMS values
    clean_rms = compute_rms(clean, eps)
    noise_rms = compute_rms(noise, eps)

    # Compute scaling factor for noise to achieve target SNR
    # SNR = 10 * log10(P_signal / P_noise)
    # P_noise_target = P_signal / (10 ** (SNR/10))
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_scale = target_noise_rms / (noise_rms + eps)

    # Scale noise and create mixture
    scaled_noise = noise * noise_scale
    noisy_mix = clean + scaled_noise

    return noisy_mix, scaled_noise


def random_crop(
    waveform: torch.Tensor,
    crop_length: int,
) -> torch.Tensor:
    """
    Randomly crop waveform to specified length.

    Args:
        waveform: Audio tensor [C, N]
        crop_length: Desired length in samples

    Returns:
        Cropped waveform [C, crop_length]
    """
    length = waveform.shape[-1]

    if length <= crop_length:
        # Pad if too short
        pad_amount = crop_length - length
        waveform = F.pad(waveform, (0, pad_amount), mode='constant', value=0)
        return waveform

    # Random starting point
    start = torch.randint(0, length - crop_length + 1, (1,)).item()
    return waveform[..., start:start + crop_length]


def apply_random_gain(
    waveform: torch.Tensor,
    min_gain_db: float = -6.0,
    max_gain_db: float = 6.0,
) -> torch.Tensor:
    """
    Apply random gain augmentation.

    Args:
        waveform: Audio tensor
        min_gain_db: Minimum gain in dB
        max_gain_db: Maximum gain in dB

    Returns:
        Waveform with random gain applied
    """
    gain_db = torch.empty(1).uniform_(min_gain_db, max_gain_db).item()
    gain = 10 ** (gain_db / 20)
    return waveform * gain


def compute_loudness_envelope(
    waveform: torch.Tensor,
    frame_size: int = 160,
    hop_size: int = 80,
) -> torch.Tensor:
    """
    Compute frame-wise loudness envelope (RMS).

    BIOMIMETIC NOTE:
    - This approximates the cochlear envelope detection
    - Used in loss function to preserve dynamics

    Args:
        waveform: Audio [B, N] or [N]
        frame_size: Frame size in samples
        hop_size: Hop size in samples

    Returns:
        Loudness envelope [B, T] or [T]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    batch_size = waveform.shape[0]
    length = waveform.shape[1]

    # Number of frames
    n_frames = (length - frame_size) // hop_size + 1

    # Extract frames
    frames = waveform.unfold(dimension=1, size=frame_size, step=hop_size)
    # frames shape: [B, T, frame_size]

    # Compute RMS per frame
    envelope = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)
    # envelope shape: [B, T]

    return envelope


def generate_synthetic_noise(
    length: int,
    noise_type: str = "white",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate synthetic noise for training.

    Args:
        length: Number of samples
        noise_type: Type of noise ('white', 'pink', 'brown')
        device: Target device

    Returns:
        Noise waveform [1, length]
    """
    if noise_type == "white":
        # White noise: uniform spectrum
        noise = torch.randn(1, length, device=device)

    elif noise_type == "pink":
        # Pink noise: 1/f spectrum (approximation using filtering)
        white = torch.randn(1, length, device=device)
        # Simple approximation: apply smoothing
        kernel_size = 7
        kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        noise = F.conv1d(white.unsqueeze(0), kernel, padding=kernel_size // 2)
        noise = noise.squeeze(0)

    elif noise_type == "brown":
        # Brown noise: 1/f^2 spectrum (cumulative sum of white)
        white = torch.randn(1, length, device=device)
        noise = torch.cumsum(white, dim=-1)
        noise = noise - noise.mean()  # Remove DC
        noise = noise / (noise.abs().max() + 1e-8)  # Normalize

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Normalize to prevent clipping
    noise = noise / (noise.abs().max() + 1e-8) * 0.9

    return noise


if __name__ == "__main__":
    print("Testing audio utilities...")

    # Test synthetic noise generation
    for noise_type in ["white", "pink", "brown"]:
        noise = generate_synthetic_noise(16000, noise_type)
        print(f"{noise_type} noise - shape: {noise.shape}, rms: {compute_rms(noise):.4f}")

    # Test mixing
    clean = torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 16000)).unsqueeze(0)
    noise = generate_synthetic_noise(16000, "white")

    for snr in [0, 10, 20]:
        noisy, _ = mix_audio_with_noise(clean, noise, snr)
        actual_snr = compute_snr(clean, noisy)
        print(f"Target SNR: {snr} dB, Actual SNR: {actual_snr:.2f} dB")

    # Test loudness envelope
    envelope = compute_loudness_envelope(clean, frame_size=160, hop_size=80)
    print(f"Envelope shape: {envelope.shape}")

    print("\n✅ Audio utilities test passed!")
