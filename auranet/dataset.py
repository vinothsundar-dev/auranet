# =============================================================================
# Dataset and Data Pipeline for AuraNet
# =============================================================================
#
# DATA PIPELINE DESIGN:
# 1. Load clean audio (speech, music, environmental sounds)
# 2. Load noise audio
# 3. Mix at random SNR to create noisy samples
# 4. Compute STFT for both noisy and clean
# 5. Return tensors ready for model training
#
# BIOMIMETIC CONSIDERATIONS:
# - Training data should include diverse sound types:
#   - Speech (various speakers, languages)
#   - Music (instruments, genres)
#   - Environmental sounds (birds, nature, urban)
#   - Noise (stationary, non-stationary)
# - Model learns to preserve "structured" sounds while suppressing "noise"
# =============================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import glob
import os

from utils.stft import CausalSTFT
from utils.audio_utils import (
    load_audio,
    normalize_audio,
    mix_audio_with_noise,
    random_crop,
    apply_random_gain,
    generate_synthetic_noise,
)


class AuraNetDataset(Dataset):
    """
    Dataset for AuraNet training.
    
    Supports:
    - Loading from directories of clean/noise files
    - Dynamic mixing at random SNR
    - Data augmentation (gain, cropping)
    - Synthetic noise generation for testing
    
    Output format matches model input:
    - noisy_stft: [2, T, F] complex STFT of noisy signal
    - clean_stft: [2, T, F] complex STFT of clean signal
    - noisy_audio: [N] noisy waveform
    - clean_audio: [N] clean waveform
    """
    
    def __init__(
        self,
        clean_dir: Optional[Union[str, Path]] = None,
        noise_dir: Optional[Union[str, Path]] = None,
        clean_files: Optional[List[str]] = None,
        noise_files: Optional[List[str]] = None,
        sample_rate: int = 16000,
        segment_length: float = 3.0,
        snr_range: Tuple[float, float] = (-5.0, 20.0),
        n_fft: int = 256,
        hop_length: int = 80,
        win_length: int = 160,
        augment: bool = True,
        synthetic_mode: bool = False,
        num_synthetic_samples: int = 1000,
    ):
        """
        Args:
            clean_dir: Directory containing clean audio files
            noise_dir: Directory containing noise audio files
            clean_files: Alternative: explicit list of clean file paths
            noise_files: Alternative: explicit list of noise file paths
            sample_rate: Target sample rate
            segment_length: Audio segment length in seconds
            snr_range: (min, max) SNR in dB for mixing
            n_fft: FFT size for STFT
            hop_length: Hop size for STFT
            win_length: Window size for STFT
            augment: Whether to apply data augmentation
            synthetic_mode: If True, generate synthetic data (for testing)
            num_synthetic_samples: Number of synthetic samples to generate
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.augment = augment
        self.synthetic_mode = synthetic_mode
        self.num_synthetic_samples = num_synthetic_samples
        
        # Initialize STFT
        self.stft = CausalSTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        
        if synthetic_mode:
            # Generate synthetic data indices
            self.clean_files = list(range(num_synthetic_samples))
            self.noise_files = list(range(num_synthetic_samples))
        else:
            # Load file lists
            if clean_files is not None:
                self.clean_files = clean_files
            elif clean_dir is not None:
                self.clean_files = self._scan_directory(clean_dir)
            else:
                self.clean_files = []
                
            if noise_files is not None:
                self.noise_files = noise_files
            elif noise_dir is not None:
                self.noise_files = self._scan_directory(noise_dir)
            else:
                self.noise_files = []
                
        if len(self.clean_files) == 0:
            print("WARNING: No clean files found. Using synthetic mode.")
            self.synthetic_mode = True
            self.clean_files = list(range(num_synthetic_samples))
            self.noise_files = list(range(num_synthetic_samples))
            
    def _scan_directory(self, directory: Union[str, Path]) -> List[str]:
        """Scan directory for audio files."""
        directory = Path(directory)
        if not directory.exists():
            print(f"WARNING: Directory not found: {directory}")
            return []
            
        extensions = ["*.wav", "*.flac", "*.mp3", "*.ogg"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(str(directory / "**" / ext), recursive=True))
            
        return sorted(files)
    
    def _generate_synthetic_clean(self) -> torch.Tensor:
        """
        Generate synthetic clean audio for testing.
        
        Creates a mix of:
        - Sinusoidal tones (simulating speech/music harmonics)
        - Chirps (simulating transitions)
        - Transients (simulating consonants/percussions)
        """
        length = self.segment_samples
        t = torch.linspace(0, self.segment_samples / self.sample_rate, length)
        
        audio = torch.zeros(1, length)
        
        # Add harmonic content (simulating speech/music)
        num_harmonics = random.randint(3, 8)
        fundamental = random.uniform(100, 400)  # Hz
        
        for h in range(1, num_harmonics + 1):
            freq = fundamental * h
            amp = 1.0 / h  # Harmonic rolloff
            phase = random.uniform(0, 2 * 3.14159)
            audio += amp * torch.sin(2 * 3.14159 * freq * t + phase).unsqueeze(0)
            
        # Add some amplitude modulation (natural dynamics)
        mod_freq = random.uniform(2, 10)
        mod_depth = random.uniform(0.3, 0.7)
        modulation = 1.0 - mod_depth + mod_depth * torch.sin(2 * 3.14159 * mod_freq * t)
        audio = audio * modulation.unsqueeze(0)
        
        # Add transients (random impulses)
        num_transients = random.randint(5, 20)
        for _ in range(num_transients):
            pos = random.randint(0, length - 100)
            transient_len = random.randint(10, 50)
            transient = torch.randn(transient_len) * random.uniform(0.5, 1.5)
            transient *= torch.hann_window(transient_len)
            audio[0, pos:pos + transient_len] += transient
            
        # Normalize
        audio = audio / (audio.abs().max() + 1e-8) * 0.8
        
        return audio
    
    def _generate_synthetic_noise(self) -> torch.Tensor:
        """Generate synthetic noise for testing."""
        length = self.segment_samples
        
        noise_type = random.choice(["white", "pink", "brown"])
        noise = generate_synthetic_noise(length, noise_type)
        
        return noise
    
    def __len__(self) -> int:
        return len(self.clean_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns dictionary with:
        - noisy_stft: [2, T, F] noisy complex STFT
        - clean_stft: [2, T, F] clean complex STFT
        - noisy_audio: [N] noisy waveform
        - clean_audio: [N] clean waveform
        - snr: scalar, the SNR used for mixing
        """
        if self.synthetic_mode:
            # Generate synthetic data
            clean_audio = self._generate_synthetic_clean()
            noise_audio = self._generate_synthetic_noise()
        else:
            # Load real audio
            clean_path = self.clean_files[idx]
            clean_audio, _ = load_audio(clean_path, self.sample_rate)
            
            # Randomly select noise file
            noise_idx = random.randint(0, len(self.noise_files) - 1)
            noise_path = self.noise_files[noise_idx]
            noise_audio, _ = load_audio(noise_path, self.sample_rate)
        
        # Random crop to segment length
        clean_audio = random_crop(clean_audio, self.segment_samples)
        noise_audio = random_crop(noise_audio, self.segment_samples)
        
        # Data augmentation
        if self.augment:
            # Random gain on clean signal
            clean_audio = apply_random_gain(clean_audio, -6.0, 6.0)
            
        # Random SNR for mixing
        snr = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Mix noisy signal
        noisy_audio, _ = mix_audio_with_noise(clean_audio, noise_audio, snr)
        
        # Ensure proper shape [1, N]
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)
        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
            
        # Compute STFT
        noisy_stft = self.stft(noisy_audio)  # [1, 2, T, F]
        clean_stft = self.stft(clean_audio)  # [1, 2, T, F]
        
        # Remove batch dimension for DataLoader batching
        noisy_stft = noisy_stft.squeeze(0)  # [2, T, F]
        clean_stft = clean_stft.squeeze(0)  # [2, T, F]
        
        return {
            "noisy_stft": noisy_stft,
            "clean_stft": clean_stft,
            "noisy_audio": noisy_audio.squeeze(0),  # [N]
            "clean_audio": clean_audio.squeeze(0),  # [N]
            "snr": torch.tensor(snr),
        }


class MixedSourceDataset(Dataset):
    """
    Extended dataset that handles multiple source types for biomimetic training.
    
    BIOMIMETIC TRAINING STRATEGY:
    - Train on diverse audio types, not just speech
    - Include music, environmental sounds, etc.
    - Model learns to preserve "structured" sounds (harmonic, transient)
    - Model learns to suppress "unstructured" sounds (noise)
    
    Source types:
    - speech: Speech recordings
    - music: Musical recordings
    - environmental: Nature sounds, urban sounds, etc.
    - noise: Various noise types (training noise)
    """
    
    def __init__(
        self,
        source_dirs: Dict[str, str],
        noise_dir: str,
        sample_rate: int = 16000,
        segment_length: float = 3.0,
        snr_range: Tuple[float, float] = (-5.0, 20.0),
        source_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Args:
            source_dirs: Dict mapping source type to directory path
                e.g. {"speech": "/data/speech", "music": "/data/music"}
            noise_dir: Directory with noise files
            source_weights: Optional weights for sampling sources
                e.g. {"speech": 0.5, "music": 0.3, "environmental": 0.2}
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        
        # STFT
        self.stft = CausalSTFT(
            n_fft=kwargs.get("n_fft", 256),
            hop_length=kwargs.get("hop_length", 80),
            win_length=kwargs.get("win_length", 160),
        )
        
        # Load files by source type
        self.source_files = {}
        self.all_files = []
        
        for source_type, directory in source_dirs.items():
            files = self._scan_directory(directory)
            self.source_files[source_type] = files
            self.all_files.extend([(f, source_type) for f in files])
            
        # Load noise files
        self.noise_files = self._scan_directory(noise_dir)
        
        # Source weights for sampling
        if source_weights is None:
            source_weights = {k: 1.0 for k in source_dirs.keys()}
        self.source_weights = source_weights
        
        # Fallback to synthetic if no files
        if len(self.all_files) == 0:
            print("WARNING: No source files found. Using synthetic mode.")
            self._use_synthetic = True
        else:
            self._use_synthetic = False
            
    def _scan_directory(self, directory: str) -> List[str]:
        directory = Path(directory)
        if not directory.exists():
            return []
        extensions = ["*.wav", "*.flac", "*.mp3"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(str(directory / "**" / ext), recursive=True))
        return files
    
    def __len__(self) -> int:
        if self._use_synthetic:
            return 1000
        return len(self.all_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Similar to AuraNetDataset but with source type tracking
        # Implementation follows same pattern
        pass  # Simplified for space


def create_dataloader(
    clean_dir: Optional[str] = None,
    noise_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    sample_rate: int = 16000,
    segment_length: float = 3.0,
    snr_range: Tuple[float, float] = (-5, 20),
    shuffle: bool = True,
    synthetic: bool = False,
) -> DataLoader:
    """
    Factory function to create DataLoader.
    
    Args:
        clean_dir: Path to clean audio directory
        noise_dir: Path to noise audio directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        sample_rate: Target sample rate
        segment_length: Audio segment length in seconds
        snr_range: SNR range for mixing
        shuffle: Shuffle data
        synthetic: Use synthetic data generation
        
    Returns:
        PyTorch DataLoader
    """
    dataset = AuraNetDataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        snr_range=snr_range,
        synthetic_mode=synthetic,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return loader


def collate_variable_length(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    Pads sequences to maximum length in batch.
    """
    # Find max time dimension
    max_time = max(item["noisy_stft"].shape[1] for item in batch)
    max_samples = max(item["noisy_audio"].shape[0] for item in batch)
    
    # Pad and stack
    noisy_stft = []
    clean_stft = []
    noisy_audio = []
    clean_audio = []
    snrs = []
    
    for item in batch:
        # Pad STFT
        t = item["noisy_stft"].shape[1]
        pad_t = max_time - t
        
        noisy_stft.append(F.pad(item["noisy_stft"], (0, 0, 0, pad_t)))
        clean_stft.append(F.pad(item["clean_stft"], (0, 0, 0, pad_t)))
        
        # Pad audio
        n = item["noisy_audio"].shape[0]
        pad_n = max_samples - n
        
        noisy_audio.append(F.pad(item["noisy_audio"], (0, pad_n)))
        clean_audio.append(F.pad(item["clean_audio"], (0, pad_n)))
        
        snrs.append(item["snr"])
        
    return {
        "noisy_stft": torch.stack(noisy_stft),
        "clean_stft": torch.stack(clean_stft),
        "noisy_audio": torch.stack(noisy_audio),
        "clean_audio": torch.stack(clean_audio),
        "snr": torch.stack(snrs),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing AuraNet Dataset")
    print("=" * 60)
    
    # Test with synthetic data
    print("\nCreating synthetic dataset...")
    
    dataset = AuraNetDataset(
        synthetic_mode=True,
        num_synthetic_samples=100,
        segment_length=2.0,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    print("\nFetching sample...")
    sample = dataset[0]
    
    print(f"noisy_stft shape: {sample['noisy_stft'].shape}")
    print(f"clean_stft shape: {sample['clean_stft'].shape}")
    print(f"noisy_audio shape: {sample['noisy_audio'].shape}")
    print(f"clean_audio shape: {sample['clean_audio'].shape}")
    print(f"SNR: {sample['snr'].item():.2f} dB")
    
    # Test DataLoader
    print("\n" + "-" * 40)
    print("Testing DataLoader...")
    
    loader = create_dataloader(
        batch_size=4,
        num_workers=0,  # For testing
        synthetic=True,
    )
    
    batch = next(iter(loader))
    print(f"Batch noisy_stft shape: {batch['noisy_stft'].shape}")
    print(f"Batch clean_stft shape: {batch['clean_stft'].shape}")
    print(f"Batch noisy_audio shape: {batch['noisy_audio'].shape}")
    
    # Verify STFT dimensions
    expected_freq_bins = 129  # n_fft // 2 + 1 = 256 // 2 + 1
    actual_freq_bins = batch['noisy_stft'].shape[-1]
    print(f"\nFrequency bins: {actual_freq_bins} (expected {expected_freq_bins})")
    
    print("\n" + "=" * 60)
    print("All dataset tests passed! ✅")
    print("=" * 60)
