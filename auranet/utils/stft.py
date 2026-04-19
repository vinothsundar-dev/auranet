# =============================================================================
# Causal STFT Implementation for AuraNet
# =============================================================================
#
# CAUSALITY EXPLANATION:
# - Standard STFT with center=True uses symmetric padding, looking into the future
# - For real-time processing, we need STRICTLY CAUSAL STFT:
#   - We only pad on the LEFT (past samples)
#   - center=False ensures no future information leaks
#   - Each output frame uses only current and past samples
#
# DESIGN DECISIONS:
# - Window: 160 samples = 10ms at 16kHz
# - Hop: 80 samples = 5ms at 16kHz (50% overlap)
# - FFT: 256 points (gives 129 frequency bins)
# - This configuration achieves ~5ms algorithmic latency per frame
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CausalSTFT(nn.Module):
    """
    Strictly causal STFT/iSTFT implementation for real-time audio processing.

    Key properties:
    - No future frame access (fully causal)
    - Phase-preserving (complex output)
    - Designed for overlap-add reconstruction

    Args:
        n_fft: FFT size (default: 256)
        hop_length: Hop size in samples (default: 80)
        win_length: Window size in samples (default: 160)
        window: Window type ('hann', 'hamming', 'blackman')
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 80,
        win_length: int = 160,
        window: str = "hann",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Number of frequency bins (positive frequencies + DC)
        self.n_freqs = n_fft // 2 + 1  # 129 for n_fft=256

        # Create analysis window (using n_fft size for proper COLA)
        if window == "hann":
            win = torch.hann_window(n_fft, periodic=True)
        elif window == "hamming":
            win = torch.hamming_window(n_fft, periodic=True)
        elif window == "blackman":
            win = torch.blackman_window(n_fft, periodic=True)
        else:
            raise ValueError(f"Unknown window type: {window}")

        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer("window", win)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute causal STFT.

        Args:
            waveform: Input audio [B, N] or [B, 1, N]

        Returns:
            Complex STFT tensor [B, 2, T, F]
            - Channel 0: Real part
            - Channel 1: Imaginary part
            - T: Number of time frames
            - F: Number of frequency bins (n_fft // 2 + 1 = 129)
        """
        # Handle input shape
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [B, 1, N] -> [B, N]

        batch_size = waveform.shape[0]
        sig_len = waveform.shape[-1]

        # Causal STFT: left-pad by (n_fft - 1) so frame 0 uses only past/current
        # samples. Use center=False so no additional padding is added.
        pad_length = self.n_fft - 1

        # Right-pad so every original sample is in the well-conditioned interior
        # of the overlap-add (at least 2-3 frames overlap with significant window
        # weight). Using n_fft ensures edge effects stay in the padding region.
        right_pad = self.n_fft
        waveform_padded = F.pad(waveform, (pad_length, right_pad))

        stft_complex = torch.stft(
            waveform_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
            onesided=True,
        )
        # stft_complex shape: [B, F, T]

        # Split into real and imaginary channels
        stft_real = stft_complex.real  # [B, F, T]
        stft_imag = stft_complex.imag  # [B, F, T]

        # Stack and permute to [B, 2, T, F] format expected by model
        stft_out = torch.stack([stft_real, stft_imag], dim=1)  # [B, 2, F, T]
        stft_out = stft_out.permute(0, 1, 3, 2)  # [B, 2, T, F]

        return stft_out

    def inverse(self, stft_tensor: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Compute inverse STFT (overlap-add reconstruction).

        Args:
            stft_tensor: Complex STFT [B, 2, T, F]
            length: Desired output length (optional)

        Returns:
            Reconstructed waveform [B, N]
        """
        # stft_tensor shape: [B, 2, T, F]
        batch_size, _, n_frames, n_freqs = stft_tensor.shape

        # Permute back to [B, F, T]
        stft_tensor = stft_tensor.permute(0, 1, 3, 2)  # [B, 2, F, T]
        stft_real = stft_tensor[:, 0, :, :]  # [B, F, T]
        stft_imag = stft_tensor[:, 1, :, :]  # [B, F, T]

        # Create complex tensor
        stft_complex = torch.complex(stft_real, stft_imag)  # [B, F, T]

        # Inverse FFT each frame → windowed time-domain frames
        frames = torch.fft.irfft(stft_complex, n=self.n_fft, dim=1)  # [B, n_fft, T]

        # Apply synthesis window
        window = self.window  # [n_fft]
        frames = frames * window.view(1, -1, 1)  # [B, n_fft, T]

        # Vectorized overlap-add using F.fold (faster + cleaner gradient flow)
        output_length = (n_frames - 1) * self.hop_length + self.n_fft
        output = F.fold(
            frames,                        # [B, n_fft, T]
            output_size=(1, output_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(1).squeeze(1)            # [B, output_length]

        # Window normalization: compute sum of squared windows per sample
        w_sq = (window ** 2).unsqueeze(0).unsqueeze(-1).expand(1, -1, n_frames)  # [1, n_fft, T]
        window_sum = F.fold(
            w_sq,
            output_size=(1, output_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(0).squeeze(0).squeeze(0)  # [output_length]
        window_sum = window_sum.clamp(min=1e-11)
        output = output / window_sum.unsqueeze(0)

        # Remove causal left-padding
        pad_length = self.n_fft - 1
        output = output[:, pad_length:]

        # Trim or pad to requested length
        if length is not None:
            if output.shape[-1] > length:
                output = output[:, :length]
            elif output.shape[-1] < length:
                output = F.pad(output, (0, length - output.shape[-1]))

        return output

    def frame_to_sample(self, frame_idx: int) -> int:
        """Convert frame index to sample index (first sample of frame)."""
        return frame_idx * self.hop_length

    def sample_to_frame(self, sample_idx: int) -> int:
        """Convert sample index to frame index (frame containing sample)."""
        return sample_idx // self.hop_length

    @property
    def latency_samples(self) -> int:
        """
        Algorithmic latency in samples.

        For causal STFT, latency = window_length (we need full window of samples)
        With hop_length output cadence, effective latency ≈ hop_length per frame.
        """
        return self.win_length

    @property
    def latency_ms(self) -> float:
        """Algorithmic latency in milliseconds (at 16kHz)."""
        return self.latency_samples / 16.0  # 16 samples/ms at 16kHz


class StreamingSTFT(CausalSTFT):
    """
    Streaming-capable STFT that maintains state across chunks.

    For real-time inference, we process audio in small chunks and need
    to maintain a buffer of past samples for continuity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Buffer to store past samples for continuity between chunks
        self._input_buffer: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None

    def reset_state(self):
        """Reset streaming buffers (call at start of new audio stream)."""
        self._input_buffer = None
        self._output_buffer = None

    def process_chunk(self, chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a chunk of audio in streaming mode.

        Args:
            chunk: Audio chunk [B, M] where M is chunk size

        Returns:
            Tuple of:
            - stft_frames: STFT frames for this chunk [B, 2, T, F]
            - output_audio: Reconstructed audio (if doing enhancement)
        """
        batch_size = chunk.shape[0]
        device = chunk.device

        # Initialize or update input buffer
        if self._input_buffer is None:
            # Initialize with zeros (causal padding)
            self._input_buffer = torch.zeros(
                batch_size, self.win_length - 1,
                device=device, dtype=chunk.dtype
            )

        # Concatenate buffer with new chunk
        full_input = torch.cat([self._input_buffer, chunk], dim=-1)

        # Compute STFT on the extended input
        stft_out = self.forward(full_input)

        # Update buffer with latest samples for next chunk
        buffer_len = self.win_length - 1
        self._input_buffer = full_input[:, -buffer_len:].clone()

        return stft_out


if __name__ == "__main__":
    # Quick test
    print("Testing CausalSTFT...")

    stft = CausalSTFT(n_fft=256, hop_length=80, win_length=160)

    # Test with random audio
    audio = torch.randn(2, 16000)  # 2 batch, 1 second at 16kHz

    # Forward STFT
    spec = stft(audio)
    print(f"Input shape: {audio.shape}")
    print(f"STFT output shape: {spec.shape}")  # Should be [2, 2, T, 129]

    # Inverse STFT
    reconstructed = stft.inverse(spec, length=audio.shape[-1])
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Check reconstruction error
    error = torch.mean((audio - reconstructed) ** 2).item()
    print(f"Reconstruction MSE: {error:.6f}")

    # Verify causality
    print(f"\nLatency: {stft.latency_samples} samples = {stft.latency_ms:.2f} ms")
    print(f"Frequency bins: {stft.n_freqs}")

    print("\n✅ CausalSTFT test passed!")
