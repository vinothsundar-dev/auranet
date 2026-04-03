# =============================================================================
# AuraNet V3 Dataset — Dynamic On-the-Fly Mixing
# =============================================================================
#
# KEY CHANGES FROM V1:
# 1. Broader SNR range with bias toward moderate SNRs (~5-15 dB)
# 2. Speed perturbation augmentation (0.95x–1.05x)
# 3. Reverb simulation via simple FIR convolution
# 4. Noise type–aware mixing (stationary vs non-stationary)
# 5. RIR augmentation placeholder for real room impulse responses
# 6. Works with existing CausalSTFT and audio_utils
# =============================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import glob
import math
import numpy as np

from utils.stft import CausalSTFT
from utils.audio_utils import (
    load_audio,
    normalize_audio,
    mix_audio_with_noise,
    random_crop,
    apply_random_gain,
    generate_synthetic_noise,
)


# =============================================================================
# Production Augmentations: RIR, Codec simulation, Bandwidth limitation
# =============================================================================

def generate_synthetic_rir(length: int = 4000, sample_rate: int = 16000,
                           rt60: float = 0.0,
                           room_size: str = '') -> torch.Tensor:
    """
    Generate a synthetic room impulse response (RIR).

    Uses exponentially decaying noise shaped by a target RT60.
    Optionally selects RT60 based on room_size preset.

    Args:
        length: RIR length in samples (default 4000 = 250ms @ 16kHz)
        sample_rate: Audio sample rate
        rt60: Reverberation time in seconds (0 = random or room_size-based)
        room_size: 'small' (0.1-0.3s), 'medium' (0.3-0.6s), 'large' (0.5-1.0s),
                   'car' (0.05-0.15s), '' = random from all

    Returns:
        RIR tensor [1, length], normalized so max(abs) = 1
    """
    if rt60 <= 0:
        if room_size == 'small':
            rt60 = random.uniform(0.1, 0.3)
        elif room_size == 'medium':
            rt60 = random.uniform(0.3, 0.6)
        elif room_size == 'large':
            rt60 = random.uniform(0.5, 1.0)
        elif room_size == 'car':
            rt60 = random.uniform(0.05, 0.15)
        else:
            # Random room: weighted toward smaller rooms (more common)
            rt60 = random.choice([
                random.uniform(0.05, 0.15),  # car
                random.uniform(0.1, 0.3),    # small room
                random.uniform(0.1, 0.3),    # small room (double weight)
                random.uniform(0.3, 0.6),    # medium room
                random.uniform(0.5, 1.0),    # large room
            ])

    # Exponential decay envelope: amplitude drops to -60dB at t=RT60
    t = torch.arange(length, dtype=torch.float32) / sample_rate
    decay = torch.exp(-6.908 * t / max(rt60, 0.01))  # ln(1000) ≈ 6.908

    # Shape noise with decay
    rir = torch.randn(length) * decay

    # Direct path is always the strongest
    rir[0] = 1.0

    # Early reflections (2-10 discrete reflections in first 20ms)
    n_early = random.randint(2, 10)
    early_end = min(int(0.02 * sample_rate), length)
    for _ in range(n_early):
        pos = random.randint(1, max(1, early_end - 1))
        rir[pos] += random.uniform(0.2, 0.7) * random.choice([-1, 1])

    # Normalize
    rir = rir / (rir.abs().max() + 1e-8)
    return rir.unsqueeze(0)  # [1, length]


def apply_rir(audio: torch.Tensor, rir: Optional[torch.Tensor] = None,
              sample_rate: int = 16000) -> torch.Tensor:
    """
    Apply room impulse response to audio via convolution.

    Pipeline: clean → convolve(clean, rir)[:len(clean)] → energy-match

    Args:
        audio: [1, N] or [N] waveform
        rir: [1, L] impulse response. None = generate random synthetic RIR.
        sample_rate: Audio sample rate

    Returns:
        Reverberant audio, same shape as input, energy-matched and clipped to [-1, 1].
    """
    was_1d = audio.dim() == 1
    if was_1d:
        audio = audio.unsqueeze(0)

    if rir is None:
        rir = generate_synthetic_rir(sample_rate=sample_rate)

    # Convolve: audio [1, N] with rir [1, L]
    orig_energy = torch.sqrt(torch.mean(audio ** 2) + 1e-8)
    reverb = F.conv1d(
        audio.unsqueeze(0),           # [1, 1, N]
        rir.unsqueeze(0),             # [1, 1, L]
        padding=rir.shape[-1] - 1,
    ).squeeze(0)                      # [1, N + L - 1]

    # Trim to original length
    reverb = reverb[..., :audio.shape[-1]]

    # Energy-match to avoid loudness change
    reverb_energy = torch.sqrt(torch.mean(reverb ** 2) + 1e-8)
    reverb = reverb * (orig_energy / (reverb_energy + 1e-8))

    # Normalize to [-1, 1]
    peak = reverb.abs().max()
    if peak > 1.0:
        reverb = reverb / peak

    if was_1d:
        reverb = reverb.squeeze(0)
    return reverb


def apply_lowpass(audio: torch.Tensor, cutoff_hz: int = 0,
                  sample_rate: int = 16000) -> torch.Tensor:
    """
    Apply low-pass filter to simulate bandwidth limitation.

    Simulates telephone (3-4kHz), mobile (6kHz), or VoIP (8kHz) conditions.

    Args:
        audio: [1, N] or [N] waveform
        cutoff_hz: Cutoff frequency. 0 = random from [3000, 4000, 6000, 8000].
        sample_rate: Audio sample rate

    Returns:
        Filtered audio, same shape.
    """
    if cutoff_hz <= 0:
        cutoff_hz = random.choice([3000, 4000, 6000, 8000])
    normalized_cutoff = cutoff_hz / (sample_rate / 2.0)
    if normalized_cutoff >= 1.0:
        return audio

    was_1d = audio.dim() == 1
    if was_1d:
        audio = audio.unsqueeze(0)

    # Windowed-sinc FIR low-pass filter
    filter_len = 101  # longer filter = sharper cutoff
    half = filter_len // 2
    n = torch.arange(filter_len, dtype=torch.float32) - half
    h = torch.where(
        n == 0,
        torch.tensor(2.0 * normalized_cutoff),
        torch.sin(2.0 * math.pi * normalized_cutoff * n) / (math.pi * n + 1e-12),
    )
    window = torch.hamming_window(filter_len, dtype=torch.float32)
    h = h * window
    h = h / h.sum()

    h = h.to(audio.device)
    filtered = F.conv1d(
        audio.unsqueeze(0),
        h.view(1, 1, -1),
        padding=half,
    ).squeeze(0)

    filtered = filtered[..., :audio.shape[-1]]
    if was_1d:
        filtered = filtered.squeeze(0)
    return filtered


def apply_resample(audio: torch.Tensor, target_sr: int = 0,
                   sample_rate: int = 16000) -> torch.Tensor:
    """
    Simulate codec resampling: downsample then upsample back.

    16kHz → target_sr → 16kHz introduces aliasing and quality loss
    typical of phone codecs (AMR-NB: 8kHz, AMR-WB: 16kHz, SILK: 12kHz).

    Args:
        audio: [1, N] or [N] waveform at sample_rate
        target_sr: Intermediate sample rate. 0 = random from [8000, 12000].
        sample_rate: Original sample rate

    Returns:
        Degraded audio at original sample_rate, same shape.
    """
    if target_sr <= 0:
        target_sr = random.choice([8000, 12000])
    if target_sr >= sample_rate:
        return audio

    was_1d = audio.dim() == 1
    if was_1d:
        audio = audio.unsqueeze(0)

    orig_len = audio.shape[-1]
    # Downsample
    down_len = int(orig_len * target_sr / sample_rate)
    downsampled = F.interpolate(
        audio.unsqueeze(0), size=down_len, mode='linear', align_corners=False
    ).squeeze(0)
    # Upsample back
    upsampled = F.interpolate(
        downsampled.unsqueeze(0), size=orig_len, mode='linear', align_corners=False
    ).squeeze(0)

    if was_1d:
        upsampled = upsampled.squeeze(0)
    return upsampled


def apply_clipping(audio: torch.Tensor, clip_level: float = 0.0) -> torch.Tensor:
    """
    Simulate microphone clipping / digital saturation.

    Args:
        audio: Waveform tensor
        clip_level: Clipping threshold (0 = random between 0.3 and 0.8 of peak)

    Returns:
        Clipped audio.
    """
    if clip_level <= 0:
        peak = audio.abs().max().item()
        clip_level = peak * random.uniform(0.3, 0.8)
    return torch.clamp(audio, -clip_level, clip_level)


def apply_quantization(audio: torch.Tensor, bits: int = 0) -> torch.Tensor:
    """
    Simulate low-bitrate quantization noise.

    Reduces effective bit depth to simulate lossy codec artifacts.

    Args:
        audio: Waveform tensor (expected in [-1, 1] range)
        bits: Effective bits. 0 = random from [6, 7, 8] (64-256 levels).
              8 bits ≈ μ-law telephone, 6 bits ≈ aggressive compression.

    Returns:
        Quantized audio with stepped waveform artifacts.
    """
    if bits <= 0:
        bits = random.choice([6, 7, 8])
    levels = 2 ** bits
    return torch.round(audio * levels) / levels


def apply_codec_chain(audio: torch.Tensor, sample_rate: int = 16000,
                      lowpass_prob: float = 0.5,
                      resample_prob: float = 0.3,
                      clipping_prob: float = 0.1,
                      quantization_prob: float = 0.2) -> torch.Tensor:
    """
    Apply a random chain of codec degradations.

    Each effect is applied independently with its own probability.
    This simulates real-world audio that passes through multiple processing
    stages (mic → codec → network → decoder → speaker).

    Args:
        audio: Waveform tensor
        sample_rate: Audio sample rate
        lowpass_prob: Probability of bandwidth limitation
        resample_prob: Probability of downsample/upsample cycle
        clipping_prob: Probability of clipping distortion
        quantization_prob: Probability of bit-depth reduction

    Returns:
        Degraded audio, same shape. Normalized to [-1, 1].
    """
    if random.random() < lowpass_prob:
        audio = apply_lowpass(audio, sample_rate=sample_rate)

    if random.random() < resample_prob:
        audio = apply_resample(audio, sample_rate=sample_rate)

    if random.random() < clipping_prob:
        audio = apply_clipping(audio)

    if random.random() < quantization_prob:
        audio = apply_quantization(audio)

    # Normalize to [-1, 1]
    peak = audio.abs().max()
    if peak > 1.0:
        audio = audio / peak

    return audio


# Legacy alias for backward compatibility
def apply_bandwidth_limitation(audio: torch.Tensor,
                               sample_rate: int = 16000) -> torch.Tensor:
    """Backward-compatible wrapper. Use apply_lowpass() for new code."""
    return apply_lowpass(audio, sample_rate=sample_rate)


class AuraNetV3Dataset(Dataset):
    """
    Training dataset with improved dynamic mixing.

    Improvements over V1:
    - SNR distribution biased toward moderate values (better for speech)
    - On-the-fly speed perturbation (0.95x–1.05x)
    - Random frequency masking for robustness
    - Proper handling of short/long clips via wrap-around padding
    """

    def __init__(
        self,
        clean_dir: Optional[Union[str, Path]] = None,
        noise_dir: Optional[Union[str, Path]] = None,
        rir_dir: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        segment_length: float = 3.0,
        snr_range: Tuple[float, float] = (-5.0, 25.0),
        n_fft: int = 256,
        hop_length: int = 80,
        win_length: int = 160,
        augment: bool = True,
        synthetic_mode: bool = False,
        num_synthetic_samples: int = 1000,
        speed_perturb: bool = True,
        rir_prob: float = 0.3,
        codec_prob: float = 0.2,
        bandwidth_prob: float = 0.15,
        clipping_prob: float = 0.05,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.augment = augment
        self.synthetic_mode = synthetic_mode
        self.num_synthetic_samples = num_synthetic_samples
        self.speed_perturb = speed_perturb
        self.rir_prob = rir_prob
        self.codec_prob = codec_prob
        self.bandwidth_prob = bandwidth_prob
        self.clipping_prob = clipping_prob

        self.stft = CausalSTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.return_stft = True  # Set False to return audio-only (GPU STFT in training loop)

        # In-memory audio cache: path → tensor (eliminates disk I/O after first load)
        self._audio_cache = {}

        # Load real RIR files if directory provided
        self.rir_files = []
        if rir_dir is not None:
            rir_path = Path(rir_dir)
            if rir_path.exists():
                for ext in ("*.wav", "*.flac"):
                    self.rir_files.extend(
                        glob.glob(str(rir_path / "**" / ext), recursive=True)
                    )
                if self.rir_files:
                    print(f"   RIR augmentation: {len(self.rir_files)} real RIRs loaded")

        if synthetic_mode:
            self.clean_files = list(range(num_synthetic_samples))
            self.noise_files = list(range(num_synthetic_samples))
        else:
            self.clean_files = self._scan_dir(clean_dir) if clean_dir else []
            self.noise_files = self._scan_dir(noise_dir) if noise_dir else []

            if len(self.clean_files) == 0:
                print("WARNING: No clean files found — falling back to synthetic mode.")
                self.synthetic_mode = True
                self.clean_files = list(range(num_synthetic_samples))
                self.noise_files = list(range(num_synthetic_samples))

    def _scan_dir(self, directory):
        directory = Path(directory)
        if not directory.exists():
            return []
        files = []
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            files.extend(glob.glob(str(directory / "**" / ext), recursive=True))
        return sorted(files)

    def _load_cached(self, path: str) -> torch.Tensor:
        """Load audio with LRU-style cache, capped at max_cache_mb."""
        if path in self._audio_cache:
            return self._audio_cache[path].clone()
        waveform, _ = load_audio(path, self.sample_rate)
        # Only cache if under memory limit (default 4 GB)
        cache_bytes = sum(v.nelement() * 4 for v in self._audio_cache.values())
        max_bytes = getattr(self, '_max_cache_mb', 4096) * 1024 * 1024
        if cache_bytes < max_bytes:
            self._audio_cache[path] = waveform
        return waveform.clone()

    def preload_audio(self, max_gb: float = 4.0):
        """Pre-load audio into RAM up to max_gb limit. Safe for Kaggle (13GB RAM)."""
        import time
        self._max_cache_mb = int(max_gb * 1024)
        t0 = time.time()
        total = len(self.clean_files) + len(self.noise_files)
        loaded = 0
        cache_bytes = 0
        max_bytes = int(max_gb * 1024 * 1024 * 1024)

        # Preload noise first (smaller set, used every sample)
        for path in self.noise_files:
            if cache_bytes >= max_bytes:
                break
            if isinstance(path, str):
                try:
                    waveform, _ = load_audio(path, self.sample_rate)
                    self._audio_cache[path] = waveform
                    cache_bytes += waveform.nelement() * 4
                except Exception:
                    pass
            loaded += 1

        # Then preload clean files with remaining budget
        for path in self.clean_files:
            if cache_bytes >= max_bytes:
                break
            if isinstance(path, str):
                try:
                    waveform, _ = load_audio(path, self.sample_rate)
                    self._audio_cache[path] = waveform
                    cache_bytes += waveform.nelement() * 4
                except Exception:
                    pass
            loaded += 1
            if loaded % 2000 == 0:
                print(f"   Preloading... {len(self._audio_cache)}/{total} "
                      f"({cache_bytes / (1024**3):.1f}/{max_gb:.0f} GB)")

        elapsed = time.time() - t0
        cache_mb = cache_bytes / (1024**2)
        print(f"   ✅ Cached {len(self._audio_cache)}/{total} files "
              f"({cache_mb:.0f} MB, cap {max_gb:.0f} GB) in {elapsed:.1f}s")
        remaining = total - len(self._audio_cache)
        if remaining > 0:
            print(f"   ℹ️  {remaining} files will be loaded on-the-fly (OS page cache)")

    # ------------------------------------------------------------------
    # Synthetic generators (unchanged from V1, kept for completeness)
    # ------------------------------------------------------------------
    def _generate_synthetic_clean(self):
        length = self.segment_samples
        t = torch.linspace(0, length / self.sample_rate, length)
        audio = torch.zeros(1, length)
        n_harm = random.randint(3, 8)
        f0 = random.uniform(100, 400)
        for h in range(1, n_harm + 1):
            amp = 1.0 / h
            phase = random.uniform(0, 2 * math.pi)
            audio += amp * torch.sin(2 * math.pi * f0 * h * t + phase).unsqueeze(0)
        mod = 1.0 - 0.5 + 0.5 * torch.sin(2 * math.pi * random.uniform(2, 10) * t)
        audio = audio * mod.unsqueeze(0)
        for _ in range(random.randint(5, 20)):
            pos = random.randint(0, length - 100)
            tl = random.randint(10, 50)
            tr = torch.randn(tl) * random.uniform(0.5, 1.5) * torch.hann_window(tl)
            audio[0, pos:pos + tl] += tr
        audio = audio / (audio.abs().max() + 1e-8) * 0.8
        return audio

    def _generate_synthetic_noise(self):
        noise_type = random.choice(["white", "pink", "brown"])
        return generate_synthetic_noise(self.segment_samples, noise_type)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    def _random_speed(self, audio):
        """Speed perturbation via resampling (0.95x–1.05x)."""
        if not self.speed_perturb or random.random() > 0.3:
            return audio
        speed = random.uniform(0.95, 1.05)
        orig_len = audio.shape[-1]
        new_len = int(orig_len / speed)
        audio = F.interpolate(audio.unsqueeze(0), size=new_len, mode='linear',
                              align_corners=False).squeeze(0)
        # Restore original length
        if audio.shape[-1] > orig_len:
            audio = audio[..., :orig_len]
        elif audio.shape[-1] < orig_len:
            audio = F.pad(audio, (0, orig_len - audio.shape[-1]))
        return audio

    def _biased_snr(self):
        """
        Sample SNR with bias toward moderate values (5–15 dB).
        50% chance: moderate [5, 15]
        25% chance: low [-5, 5]
        25% chance: high [15, 25]
        """
        r = random.random()
        if r < 0.5:
            return random.uniform(5.0, 15.0)
        elif r < 0.75:
            return random.uniform(self.snr_range[0], 5.0)
        else:
            return random.uniform(15.0, self.snr_range[1])

    def _safe_crop(self, audio, target_len):
        """Crop or wrap-pad audio to target_len."""
        if audio.shape[-1] >= target_len:
            return random_crop(audio, target_len)
        # Wrap-pad short clips
        reps = (target_len // audio.shape[-1]) + 1
        audio = audio.repeat(1, reps) if audio.dim() == 2 else audio.repeat(reps)
        return audio[..., :target_len]

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if self.synthetic_mode:
            clean_audio = self._generate_synthetic_clean()
            noise_audio = self._generate_synthetic_noise()
        else:
            clean_path = self.clean_files[idx]
            clean_audio = self._load_cached(clean_path)
            noise_idx = random.randint(0, len(self.noise_files) - 1)
            noise_audio = self._load_cached(self.noise_files[noise_idx])

        # Crop to segment length (with wrap-padding for short clips)
        clean_audio = self._safe_crop(clean_audio, self.segment_samples)
        noise_audio = self._safe_crop(noise_audio, self.segment_samples)

        # ================================================================
        # Augmentation pipeline:
        #   clean → gain → speed → RIR → codec degradation → mix → clip
        # ================================================================
        if self.augment:
            # 1. Random gain
            clean_audio = apply_random_gain(clean_audio, -6.0, 6.0)

            # 2. Speed perturbation
            clean_audio = self._random_speed(clean_audio)

            # 3. RIR augmentation — apply reverb to clean speech
            if random.random() < self.rir_prob:
                if self.rir_files:
                    rir_path = random.choice(self.rir_files)
                    try:
                        rir_audio, _ = load_audio(rir_path, self.sample_rate)
                        if rir_audio.dim() == 1:
                            rir_audio = rir_audio.unsqueeze(0)
                        rir_audio = rir_audio[..., :8000]  # max 500ms
                        clean_audio = apply_rir(clean_audio, rir_audio, self.sample_rate)
                    except Exception:
                        clean_audio = apply_rir(clean_audio, None, self.sample_rate)
                else:
                    clean_audio = apply_rir(clean_audio, None, self.sample_rate)

            # 4. Codec degradation chain (lowpass + resample + quantization)
            if random.random() < self.codec_prob:
                clean_audio = apply_codec_chain(clean_audio, self.sample_rate)
                noise_audio = apply_codec_chain(noise_audio, self.sample_rate)

            # 4b. Standalone bandwidth limitation (separate from full codec chain)
            elif random.random() < self.bandwidth_prob:
                clean_audio = apply_lowpass(clean_audio, sample_rate=self.sample_rate)
                noise_audio = apply_lowpass(noise_audio, sample_rate=self.sample_rate)

        # 5. Biased SNR mixing
        snr = self._biased_snr()
        noisy_audio, _ = mix_audio_with_noise(clean_audio, noise_audio, snr)

        # 6. Clipping — applied AFTER mixing (simulates mic overload)
        if self.augment and random.random() < self.clipping_prob:
            noisy_audio = apply_clipping(noisy_audio)

        # Ensure shape [1, N]
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)
        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)

        if self.return_stft:
            # CPU STFT (legacy mode — works but slower)
            noisy_stft = self.stft(noisy_audio).squeeze(0)  # [2, T, F]
            clean_stft = self.stft(clean_audio).squeeze(0)
            return {
                "noisy_stft": noisy_stft,
                "clean_stft": clean_stft,
                "noisy_audio": noisy_audio.squeeze(0),
                "clean_audio": clean_audio.squeeze(0),
                "snr": torch.tensor(snr),
            }
        else:
            # Audio-only mode — STFT computed on GPU in training loop (faster)
            return {
                "noisy_audio": noisy_audio.squeeze(0),
                "clean_audio": clean_audio.squeeze(0),
                "snr": torch.tensor(snr),
            }


def create_v3_dataloader(clean_dir, noise_dir, batch_size=16, num_workers=2,
                          sample_rate=16000, segment_length=3.0, augment=True,
                          synthetic_mode=False, **kwargs):
    """Factory for V3 DataLoader."""
    ds = AuraNetV3Dataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        augment=augment,
        synthetic_mode=synthetic_mode,
        **kwargs,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
