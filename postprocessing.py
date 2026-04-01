"""
AuraNet Post-processing for CI and Normal Hearing Users.

Provides adaptive output modes:
- Normal: Standard enhanced audio
- CI: Optimized for cochlear implant electrode mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

class AdaptivePostProcessor(nn.Module):
    """
    Post-processor that adapts output for different hearing profiles.

    Modes:
    - 'normal': Standard enhancement for normal hearing
    - 'ci': Optimized for cochlear implant users
    - 'hearing_aid': Mild amplification + enhancement
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1

        # CI electrode frequency bands (typical 22-channel CI)
        # Maps to ~200Hz - 8000Hz range
        self.ci_bands = self._create_ci_bands()

        # Frequency emphasis curves
        self.register_buffer('ci_emphasis', self._create_ci_emphasis())
        self.register_buffer('normal_emphasis', self._create_normal_emphasis())

    def _create_ci_bands(self):
        """Create frequency bands matching typical CI electrode mapping."""
        # Cochlear implant typical frequency allocation (22 electrodes)
        # Lower frequencies get more bandwidth (logarithmic spacing)
        ci_frequencies = [
            200, 313, 391, 469, 547, 625, 703, 828,
            953, 1078, 1203, 1375, 1563, 1813, 2063,
            2375, 2750, 3188, 3688, 4313, 5063, 5938, 7000
        ]
        return ci_frequencies

    def _create_ci_emphasis(self):
        """
        Create frequency emphasis curve for CI users.
        - Boost consonant frequencies (2-4 kHz) for speech clarity
        - Reduce low frequency rumble
        - Smooth transitions to prevent artifacts
        """
        freqs = torch.linspace(0, self.sample_rate / 2, self.n_freqs)
        emphasis = torch.ones(self.n_freqs)

        # Reduce sub-200Hz (not useful for CI)
        low_mask = freqs < 200
        emphasis[low_mask] *= 0.3

        # Boost 1.5-4kHz (consonant clarity - critical for speech understanding)
        consonant_mask = (freqs >= 1500) & (freqs <= 4000)
        emphasis[consonant_mask] *= 1.4

        # Slight boost 500-1500Hz (vowel fundamentals)
        vowel_mask = (freqs >= 500) & (freqs < 1500)
        emphasis[vowel_mask] *= 1.1

        # Roll off above 6kHz (CI electrode limit)
        high_mask = freqs > 6000
        emphasis[high_mask] *= 0.5

        # Smooth the curve
        emphasis = self._smooth_curve(emphasis, window=5)

        return emphasis.unsqueeze(0).unsqueeze(0)  # [1, 1, F]

    def _create_normal_emphasis(self):
        """
        Create frequency emphasis for normal hearing users.
        - Natural sound preservation
        - Mild presence boost for clarity in noise
        """
        freqs = torch.linspace(0, self.sample_rate / 2, self.n_freqs)
        emphasis = torch.ones(self.n_freqs)

        # Mild boost 2-5kHz for presence/clarity
        presence_mask = (freqs >= 2000) & (freqs <= 5000)
        emphasis[presence_mask] *= 1.15

        # Reduce sub-80Hz rumble
        rumble_mask = freqs < 80
        emphasis[rumble_mask] *= 0.5

        emphasis = self._smooth_curve(emphasis, window=3)

        return emphasis.unsqueeze(0).unsqueeze(0)  # [1, 1, F]

    def _smooth_curve(self, curve, window=5):
        """Apply smoothing to prevent harsh transitions."""
        kernel = torch.ones(window) / window
        padded = F.pad(curve.unsqueeze(0).unsqueeze(0), (window//2, window//2), mode='replicate')
        smoothed = F.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))
        return smoothed.squeeze()

    def forward(
        self,
        enhanced_stft: torch.Tensor,
        mode: Literal['normal', 'ci', 'hearing_aid'] = 'normal',
        ci_volume_boost: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply mode-specific post-processing.

        Args:
            enhanced_stft: Enhanced STFT [B, 2, T, F] (real/imag)
            mode: 'normal', 'ci', or 'hearing_aid'
            ci_volume_boost: Additional gain for CI users (1.0 = no change)

        Returns:
            Processed STFT [B, 2, T, F]
        """
        if mode == 'normal':
            return self._process_normal(enhanced_stft)
        elif mode == 'ci':
            return self._process_ci(enhanced_stft, ci_volume_boost)
        elif mode == 'hearing_aid':
            return self._process_hearing_aid(enhanced_stft)
        else:
            return enhanced_stft

    def _process_normal(self, stft: torch.Tensor) -> torch.Tensor:
        """Standard processing for normal hearing."""
        # Apply mild emphasis
        emphasis = self.normal_emphasis.to(stft.device)
        return stft * emphasis

    def _process_ci(self, stft: torch.Tensor, volume_boost: float = 1.0) -> torch.Tensor:
        """
        CI-optimized processing:
        - Frequency shaping for electrode mapping
        - Enhanced transients for consonant clarity
        - Compressed dynamics (CI has limited dynamic range)
        """
        device = stft.device
        emphasis = self.ci_emphasis.to(device)

        # Apply CI frequency emphasis
        processed = stft * emphasis

        # Compute magnitude and phase
        mag = torch.sqrt(processed[:, 0:1]**2 + processed[:, 1:2]**2 + 1e-8)
        phase = torch.atan2(processed[:, 1:2], processed[:, 0:1])

        # Compress dynamics (CI has ~20dB dynamic range vs 120dB normal)
        # Soft compression to fit CI's limited range
        mag_db = 20 * torch.log10(mag + 1e-8)

        # Compress: reduce dynamic range by 50%
        threshold_db = -40
        ratio = 2.0  # 2:1 compression above threshold

        above_thresh = mag_db > threshold_db
        compressed_db = torch.where(
            above_thresh,
            threshold_db + (mag_db - threshold_db) / ratio,
            mag_db
        )

        # Convert back to linear
        mag_compressed = torch.pow(10, compressed_db / 20)

        # Apply volume boost
        mag_compressed = mag_compressed * volume_boost

        # Reconstruct STFT
        real = mag_compressed * torch.cos(phase)
        imag = mag_compressed * torch.sin(phase)

        return torch.cat([real, imag], dim=1)

    def _process_hearing_aid(self, stft: torch.Tensor) -> torch.Tensor:
        """Processing for mild-moderate hearing loss (hearing aids)."""
        # Similar to normal but with:
        # - Stronger high-frequency boost (compensate for typical age-related loss)
        # - Mild compression

        freqs = torch.linspace(0, self.sample_rate / 2, self.n_freqs, device=stft.device)

        # High frequency boost (typical presbycusis pattern)
        emphasis = torch.ones(self.n_freqs, device=stft.device)

        high_mask = freqs > 1000
        if high_mask.any():
            # Progressive boost above 1kHz using tensor ops so CUDA inputs stay on device.
            boost = 1.0 + 0.3 * torch.log2(torch.clamp(freqs[high_mask] / 1000.0, min=1.0))
            emphasis[high_mask] = torch.clamp(boost, max=2.0)

        emphasis = emphasis.view(1, 1, 1, -1)
        return stft * emphasis


class DualModeEnhancer(nn.Module):
    """
    Wrapper that combines AuraNet with adaptive post-processing.

    Usage:
        enhancer = DualModeEnhancer(auranet_model)

        # For normal user
        enhanced = enhancer(noisy_audio, mode='normal')

        # For CI user
        enhanced = enhancer(noisy_audio, mode='ci')
    """

    def __init__(self, model, stft_module):
        super().__init__()
        self.model = model
        self.stft = stft_module
        self.postprocessor = AdaptivePostProcessor(
            sample_rate=16000,
            n_fft=stft_module.n_fft if hasattr(stft_module, 'n_fft') else 256
        )

    def _get_runtime_device(self) -> torch.device:
        """Infer the device from the STFT module first, then the model."""
        if hasattr(self.stft, 'window') and isinstance(self.stft.window, torch.Tensor):
            return self.stft.window.device

        for module in (self.model, self.stft, self.postprocessor):
            try:
                return next(module.parameters()).device
            except StopIteration:
                continue

        return torch.device('cpu')

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        mode: Literal['normal', 'ci', 'hearing_aid'] = 'normal',
        ci_volume_boost: float = 1.0,
    ) -> torch.Tensor:
        """
        Enhance audio with mode-specific processing.

        Args:
            audio: Input audio [B, N] or [N]
            mode: 'normal', 'ci', or 'hearing_aid'
            ci_volume_boost: Volume boost for CI mode (1.0-2.0 recommended)

        Returns:
            Enhanced audio tensor
        """
        input_device = audio.device
        runtime_device = self._get_runtime_device()

        # Handle dimensions
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Keep the STFT, model, and input on the same device.
        audio = audio.to(runtime_device)

        # STFT
        noisy_stft = self.stft(audio)

        # Model enhancement
        enhanced_stft, _, _ = self.model(noisy_stft)

        # Mode-specific post-processing
        processed_stft = self.postprocessor(enhanced_stft, mode, ci_volume_boost)

        # Inverse STFT
        enhanced_audio = self.stft.inverse(processed_stft)

        return enhanced_audio.squeeze().to(input_device)


def create_dual_mode_enhancer(model, stft):
    """Factory function to create dual-mode enhancer."""
    return DualModeEnhancer(model, stft)


# Quick test
if __name__ == "__main__":
    print("Testing AdaptivePostProcessor...")

    processor = AdaptivePostProcessor()

    # Fake STFT input [B, 2, T, F]
    fake_stft = torch.randn(1, 2, 100, 129)

    normal_out = processor(fake_stft, mode='normal')
    ci_out = processor(fake_stft, mode='ci')
    ha_out = processor(fake_stft, mode='hearing_aid')

    print(f"Input shape: {fake_stft.shape}")
    print(f"Normal output: {normal_out.shape}")
    print(f"CI output: {ci_out.shape}")
    print(f"Hearing aid output: {ha_out.shape}")
    print("✅ All modes working!")
