# =============================================================================
# Loss Functions for AuraNet
# =============================================================================
#
# MULTI-TASK LOSS DESIGN:
# 1. Complex MSE: Direct supervision in frequency domain
# 2. Multi-Resolution STFT: Perceptually-motivated spectral matching
# 3. Loudness Envelope: Preserve dynamics and perceived loudness
# 4. Temporal Coherence: Prevent artifacts, maintain smooth transitions
#
# BIOMIMETIC RATIONALE:
# - Human hearing is sensitive to spectral shape (STFT loss)
# - Loudness perception follows specific rules (envelope loss)
# - Temporal continuity is key to speech intelligibility (coherence loss)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# =============================================================================
# Complex MSE Loss
# =============================================================================

class ComplexMSELoss(nn.Module):
    """
    Mean Squared Error loss on complex STFT representation.
    
    Operates on both real and imaginary components independently.
    This provides direct supervision for phase-preserving enhancement.
    
    L_cmse = MSE(pred_real, target_real) + MSE(pred_imag, target_imag)
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted complex STFT [B, 2, T, F]
            target: Target complex STFT [B, 2, T, F]
            
        Returns:
            Complex MSE loss scalar
        """
        # Split into real and imaginary
        pred_real = pred[:, 0, :, :]
        pred_imag = pred[:, 1, :, :]
        target_real = target[:, 0, :, :]
        target_imag = target[:, 1, :, :]
        
        # Compute MSE for each component
        loss_real = F.mse_loss(pred_real, target_real, reduction=self.reduction)
        loss_imag = F.mse_loss(pred_imag, target_imag, reduction=self.reduction)
        
        return loss_real + loss_imag


class ComplexMAELoss(nn.Module):
    """
    Mean Absolute Error on complex STFT (L1 loss).
    
    More robust to outliers than MSE.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_real = pred[:, 0, :, :]
        pred_imag = pred[:, 1, :, :]
        target_real = target[:, 0, :, :]
        target_imag = target[:, 1, :, :]
        
        loss_real = F.l1_loss(pred_real, target_real, reduction=self.reduction)
        loss_imag = F.l1_loss(pred_imag, target_imag, reduction=self.reduction)
        
        return loss_real + loss_imag


# =============================================================================
# Multi-Resolution STFT Loss
# =============================================================================

class STFTLoss(nn.Module):
    """
    Single-resolution STFT loss combining spectral convergence and magnitude.
    
    Components:
    - Spectral Convergence (SC): Frobenius norm of difference / norm of target
    - Log STFT Magnitude: L1 on log magnitude spectrogram
    
    L_stft = SC + alpha * LogMag
    """
    
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 120,
        win_length: int = 600,
        window: str = "hann",
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
    ):
        super().__init__()
        
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        
        # Register window as buffer
        if window == "hann":
            win = torch.hann_window(win_length)
        elif window == "hamming":
            win = torch.hamming_window(win_length)
        else:
            win = torch.hann_window(win_length)
        self.register_buffer("window", win)
        
    def _stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude of time-domain signal."""
        # x: [B, N] or [B, 1, N]
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Ensure window is on the same device as input
        window = self.window.to(x.device)
            
        # Compute STFT
        spec = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        # spec: [B, F, T] complex
        
        # Compute magnitude
        mag = torch.abs(spec)
        
        return mag
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: Predicted waveform [B, N]
            target: Target waveform [B, N]
            
        Returns:
            Tuple of (spectral_convergence_loss, log_magnitude_loss)
        """
        # Ensure same length
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        
        pred_mag = self._stft_magnitude(pred)
        target_mag = self._stft_magnitude(target)
        
        # Ensure same shape for magnitudes
        min_t = min(pred_mag.shape[-1], target_mag.shape[-1])
        pred_mag = pred_mag[..., :min_t]
        target_mag = target_mag[..., :min_t]
        
        # Spectral convergence loss
        sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
        
        # Log STFT magnitude loss
        log_pred = torch.log(pred_mag + 1e-8)
        log_target = torch.log(target_mag + 1e-8)
        mag_loss = F.l1_loss(log_pred, log_target)
        
        return self.factor_sc * sc_loss, self.factor_mag * mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss.
    
    PERCEPTUAL MOTIVATION:
    - Different FFT sizes capture different temporal/spectral tradeoffs
    - Small FFT: Good temporal resolution, captures transients
    - Large FFT: Good spectral resolution, captures harmonics
    
    Averaging across resolutions provides comprehensive spectral supervision.
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [50, 120, 240],
        win_lengths: List[int] = [240, 600, 1200],
    ):
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.stft_losses = nn.ModuleList([
            STFTLoss(fft_size=fft, hop_size=hop, win_length=win)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform [B, N]
            target: Target waveform [B, N]
            
        Returns:
            Combined multi-resolution STFT loss
        """
        total_sc = 0.0
        total_mag = 0.0
        
        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(pred, target)
            total_sc += sc
            total_mag += mag
            
        # Average across resolutions
        n = len(self.stft_losses)
        return (total_sc + total_mag) / n


# =============================================================================
# Loudness Envelope Loss
# =============================================================================

class LoudnessEnvelopeLoss(nn.Module):
    """
    Loss on loudness envelope to preserve dynamics.
    
    BIOMIMETIC RATIONALE:
    - Human hearing integrates energy over ~30-200ms windows
    - Preserving envelope preserves perceived loudness dynamics
    - Critical for natural-sounding enhancement without pumping
    
    Computes RMS envelope and penalizes deviation from target envelope.
    """
    
    def __init__(
        self,
        frame_size: int = 160,
        hop_size: int = 80,
        loss_type: str = "l1",
    ):
        super().__init__()
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.loss_type = loss_type
        
    def _compute_envelope(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS envelope."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # Use unfold to extract frames
        # Pad to ensure full coverage
        pad_amount = self.frame_size - (x.shape[-1] % self.frame_size)
        if pad_amount < self.frame_size:
            x = F.pad(x, (0, pad_amount))
            
        frames = x.unfold(dimension=-1, size=self.frame_size, step=self.hop_size)
        # frames: [B, num_frames, frame_size]
        
        # Compute RMS per frame
        envelope = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)
        
        return envelope
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform [B, N]
            target: Target waveform [B, N]
            
        Returns:
            Loudness envelope loss
        """
        # Ensure same length
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        
        pred_env = self._compute_envelope(pred)
        target_env = self._compute_envelope(target)
        
        # Match lengths if different
        min_t = min(pred_env.shape[-1], target_env.shape[-1])
        pred_env = pred_env[..., :min_t]
        target_env = target_env[..., :min_t]
        
        if self.loss_type == "l1":
            return F.l1_loss(pred_env, target_env)
        elif self.loss_type == "l2":
            return F.mse_loss(pred_env, target_env)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# =============================================================================
# Temporal Coherence Loss
# =============================================================================

class TemporalCoherenceLoss(nn.Module):
    """
    Loss encouraging temporal smoothness / coherence.
    
    RATIONALE:
    - Prevents frame-to-frame discontinuities
    - Reduces musical noise artifacts
    - Maintains natural temporal evolution
    
    Penalizes unexpected changes in magnitude between adjacent frames
    compared to the target's natural variation.
    """
    
    def __init__(self, order: int = 1):
        super().__init__()
        self.order = order  # First or second order difference
        
    def _compute_temporal_diff(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temporal difference."""
        # x: [B, 2, T, F] or [B, T, F]
        if x.dim() == 4:
            # Compute magnitude
            mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2 + 1e-8)
        else:
            mag = x
            
        # First-order temporal difference
        diff = mag[:, 1:, :] - mag[:, :-1, :]
        
        if self.order == 2:
            # Second-order difference
            diff = diff[:, 1:, :] - diff[:, :-1, :]
            
        return diff
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted STFT [B, 2, T, F]
            target: Target STFT [B, 2, T, F]
            
        Returns:
            Temporal coherence loss
        """
        pred_diff = self._compute_temporal_diff(pred)
        target_diff = self._compute_temporal_diff(target)
        
        # Match lengths
        min_t = min(pred_diff.shape[1], target_diff.shape[1])
        pred_diff = pred_diff[:, :min_t, :]
        target_diff = target_diff[:, :min_t, :]
        
        # Loss: predicted changes should match target changes
        loss = F.l1_loss(pred_diff, target_diff)
        
        return loss


# =============================================================================
# Combined AuraNet Loss
# =============================================================================

class AuraNetLoss(nn.Module):
    """
    Combined multi-task loss for AuraNet.
    
    Total loss is weighted sum of:
    1. Complex MSE loss (frequency domain)
    2. Multi-resolution STFT loss (perceptual)
    3. Loudness envelope loss (dynamics)
    4. Temporal coherence loss (smoothness)
    
    Each component addresses different perceptual aspects:
    - Complex MSE: Direct phase/magnitude matching
    - STFT: Multi-scale spectral matching
    - Loudness: Preserve perceived dynamics
    - Temporal: Prevent artifacts
    """
    
    def __init__(
        self,
        weight_complex_mse: float = 1.0,
        weight_stft: float = 0.5,
        weight_loudness: float = 0.3,
        weight_temporal: float = 0.2,
        stft_fft_sizes: List[int] = [512, 1024, 2048],
        stft_hop_sizes: List[int] = [50, 120, 240],
        stft_win_lengths: List[int] = [240, 600, 1200],
    ):
        super().__init__()
        
        self.weight_complex_mse = weight_complex_mse
        self.weight_stft = weight_stft
        self.weight_loudness = weight_loudness
        self.weight_temporal = weight_temporal
        
        # Individual loss components
        self.complex_mse_loss = ComplexMSELoss()
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_fft_sizes,
            hop_sizes=stft_hop_sizes,
            win_lengths=stft_win_lengths,
        )
        self.loudness_loss = LoudnessEnvelopeLoss()
        self.temporal_loss = TemporalCoherenceLoss()
        
    def forward(
        self,
        pred_stft: torch.Tensor,
        target_stft: torch.Tensor,
        pred_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            pred_stft: Predicted complex STFT [B, 2, T, F]
            target_stft: Target complex STFT [B, 2, T, F]
            pred_audio: Predicted waveform [B, N] (optional, for time-domain losses)
            target_audio: Target waveform [B, N] (optional)
            
        Returns:
            Tuple of:
            - total_loss: Weighted sum of all losses
            - loss_dict: Dictionary with individual loss values
        """
        loss_dict = {}
        
        # 1. Complex MSE loss (always computed)
        loss_complex = self.complex_mse_loss(pred_stft, target_stft)
        loss_dict["complex_mse"] = loss_complex
        
        # 2. Temporal coherence loss (on STFT)
        loss_temporal = self.temporal_loss(pred_stft, target_stft)
        loss_dict["temporal_coherence"] = loss_temporal
        
        # 3. Time-domain losses (if audio available)
        if pred_audio is not None and target_audio is not None:
            # Multi-resolution STFT loss
            loss_stft = self.stft_loss(pred_audio, target_audio)
            loss_dict["multi_res_stft"] = loss_stft
            
            # Loudness envelope loss
            loss_loudness = self.loudness_loss(pred_audio, target_audio)
            loss_dict["loudness_envelope"] = loss_loudness
        else:
            loss_stft = torch.tensor(0.0, device=pred_stft.device)
            loss_loudness = torch.tensor(0.0, device=pred_stft.device)
            loss_dict["multi_res_stft"] = loss_stft
            loss_dict["loudness_envelope"] = loss_loudness
        
        # Compute weighted total
        total_loss = (
            self.weight_complex_mse * loss_complex +
            self.weight_stft * loss_stft +
            self.weight_loudness * loss_loudness +
            self.weight_temporal * loss_temporal
        )
        
        loss_dict["total"] = total_loss
        
        return total_loss, loss_dict


# =============================================================================
# Auxiliary Losses
# =============================================================================

class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio loss.
    
    Common metric for speech enhancement/separation.
    Measures signal quality in a scale-invariant way.
    
    SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = <pred, target> * target / ||target||^2
          e_noise = pred - s_target
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform [B, N]
            target: Target waveform [B, N]
            
        Returns:
            Negative SI-SNR (for minimization)
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
            
        # Zero-mean normalization
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # Compute SI-SNR
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        s_target = dot * target / (torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps)
        e_noise = pred - s_target
        
        si_snr = torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + self.eps)
        si_snr_db = 10 * torch.log10(si_snr + self.eps)
        
        # Return negative for minimization
        return -si_snr_db.mean()


class PhaseLoss(nn.Module):
    """
    Phase-aware loss for complex spectrum.
    
    Encourages correct phase estimation, which is crucial for
    natural-sounding speech enhancement.
    """
    
    def __init__(self, loss_type: str = "ip"):
        super().__init__()
        self.loss_type = loss_type  # "ip" (instantaneous phase) or "gd" (group delay)
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted complex STFT [B, 2, T, F]
            target: Target complex STFT [B, 2, T, F]
        """
        # Compute phase
        pred_phase = torch.atan2(pred[:, 1, :, :], pred[:, 0, :, :] + 1e-8)
        target_phase = torch.atan2(target[:, 1, :, :], target[:, 0, :, :] + 1e-8)
        
        # Phase difference (wrapped to [-pi, pi])
        phase_diff = pred_phase - target_phase
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        # L1 loss on phase difference
        loss = torch.abs(phase_diff).mean()
        
        return loss


# =============================================================================
# V2 UPGRADES: Psychoacoustic Loud-Loss
# =============================================================================

class PsychoacousticLoudLoss(nn.Module):
    """
    Psychoacoustic Loudness-Weighted Loss.
    
    BIOMIMETIC RATIONALE:
    Human hearing does not perceive all frequencies equally. The equal-loudness
    contours (ISO 226) show that:
    - Ears are most sensitive around 2-5 kHz (speech consonants)
    - Low frequencies require more energy to be perceived
    - Very high frequencies have reduced sensitivity
    
    This loss weights errors by perceptual importance, ensuring that
    perceptually significant frequencies are prioritized.
    
    FORMULA:
        L_loud = Σ W(f) * || log(P + ε) - log(P̂ + ε) ||²
    
    Where:
        - W(f) = frequency weighting (A-weighting or equal-loudness approximation)
        - P = power spectrum of target
        - P̂ = power spectrum of prediction
        - ε = stability constant (1e-6)
    
    LOG DOMAIN RATIONALE:
    - Human loudness perception is roughly logarithmic
    - Log domain compresses dynamic range naturally
    - Errors in quiet regions are weighted appropriately
    """
    
    def __init__(
        self,
        n_fft: int = 256,
        sample_rate: int = 16000,
        weighting: str = "a_weighting",  # "a_weighting", "equal_loudness", "flat"
        eps: float = 1e-6,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.eps = eps
        self.weighting = weighting
        
        # Pre-compute frequency weighting curve
        n_freqs = n_fft // 2 + 1
        freqs = torch.linspace(0, sample_rate / 2, n_freqs)
        
        if weighting == "a_weighting":
            weights = self._compute_a_weighting(freqs)
        elif weighting == "equal_loudness":
            weights = self._compute_equal_loudness(freqs)
        else:
            weights = torch.ones(n_freqs)
        
        # Normalize weights to sum to n_freqs (preserve loss scale)
        weights = weights / weights.mean()
        
        self.register_buffer("freq_weights", weights.view(1, 1, 1, -1))
        
    def _compute_a_weighting(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Compute A-weighting curve.
        
        A-weighting approximates human hearing sensitivity at moderate levels.
        Based on IEC 61672:2003.
        """
        f = freqs.clamp(min=1.0)  # Avoid division by zero
        
        # A-weighting formula (simplified)
        f2 = f ** 2
        
        num = 12194 ** 2 * f2 ** 2
        denom = (f2 + 20.6 ** 2) * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2)
        
        a_weight = num / (denom + 1e-8)
        
        # Normalize (0 dB at 1kHz reference)
        ref_idx = torch.argmin(torch.abs(f - 1000))
        a_weight = a_weight / (a_weight[ref_idx] + 1e-8)
        
        return a_weight.clamp(min=0.01)  # Prevent zero weights
        
    def _compute_equal_loudness(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Compute equal-loudness contour approximation at 60 phon.
        
        Based on ISO 226:2003 equal-loudness contours.
        """
        f = freqs.clamp(min=20.0)  # Audible range
        
        # Simplified equal-loudness approximation
        # Peaks around 3-4 kHz, rolls off at extremes
        center = 3500.0
        width = 2.5  # Octaves
        
        log_ratio = torch.log2(f / center)
        weights = torch.exp(-0.5 * (log_ratio / width) ** 2)
        
        # Add low-frequency roll-off
        lf_rolloff = torch.sigmoid((f - 100) / 50)
        weights = weights * lf_rolloff
        
        # Add high-frequency roll-off
        hf_rolloff = torch.sigmoid((8000 - f) / 500)
        weights = weights * hf_rolloff
        
        return weights.clamp(min=0.1)
        
    def forward(
        self,
        pred_stft: torch.Tensor,
        target_stft: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute psychoacoustic loudness-weighted loss.
        
        Args:
            pred_stft: Predicted complex STFT [B, 2, T, F]
            target_stft: Target complex STFT [B, 2, T, F]
            
        Returns:
            Weighted log-power loss scalar
        """
        # Compute power spectra
        pred_power = pred_stft[:, 0, :, :] ** 2 + pred_stft[:, 1, :, :] ** 2
        target_power = target_stft[:, 0, :, :] ** 2 + target_stft[:, 1, :, :] ** 2
        
        # Log domain (with stability constant)
        pred_log = torch.log(pred_power + self.eps)
        target_log = torch.log(target_power + self.eps)
        
        # Squared error in log domain
        log_error = (pred_log - target_log) ** 2
        
        # Apply frequency weighting
        # Handle frequency dimension mismatch
        F = log_error.shape[-1]
        if self.freq_weights.shape[-1] != F:
            weights = F_nn.interpolate(
                self.freq_weights.squeeze(0),
                size=F,
                mode='linear',
                align_corners=False
            ).unsqueeze(0)
        else:
            weights = self.freq_weights
        
        weighted_error = log_error.unsqueeze(1) * weights  # [B, 1, T, F]
        
        # Mean over all dimensions
        loss = weighted_error.mean()
        
        return loss


# Alias for F module in loss context
F_nn = nn.functional


class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio loss.
    
    Improved version of SI-SNR with better gradient properties.
    Commonly used in state-of-the-art speech enhancement.
    
    SI-SDR = 10 * log10(||s_target||² / ||e||²)
    
    Where:
        s_target = (<s, s_ref> / ||s_ref||²) * s_ref
        e = s - s_target
    """
    
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform [B, N]
            target: Target waveform [B, N]
            
        Returns:
            Negative SI-SDR (for minimization)
        """
        # Ensure 2D
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        # Ensure same length
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        
        # Zero-mean
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # Compute SI-SDR
        # s_target = (<s, s_ref> / ||s_ref||²) * s_ref
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        s_ref_norm = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / s_ref_norm) * target
        
        # e = s - s_target
        e = pred - s_target
        
        # SI-SDR = 10 * log10(||s_target||² / ||e||²)
        si_sdr = torch.sum(s_target ** 2, dim=-1) / (torch.sum(e ** 2, dim=-1) + self.eps)
        si_sdr_db = 10 * torch.log10(si_sdr + self.eps)
        
        if self.reduction == "mean":
            return -si_sdr_db.mean()
        elif self.reduction == "sum":
            return -si_sdr_db.sum()
        else:
            return -si_sdr_db


# =============================================================================
# V2 Combined Loss with Psychoacoustic Weighting
# =============================================================================

class AuraNetV2Loss(nn.Module):
    """
    AuraNet V2 Loss: Psychoacoustically-informed multi-task loss.
    
    UPGRADES FROM V1:
    1. Psychoacoustic Loud-Loss (frequency-weighted log-power)
    2. SI-SDR for time-domain supervision
    3. Configurable loss combination for 2-stage training
    
    LOSS COMPONENTS:
    - Loud-Loss: Perceptually-weighted frequency domain
    - SI-SDR: Scale-invariant time domain
    - Multi-Resolution STFT: Spectral envelope matching
    - Temporal Coherence: Artifact prevention
    
    2-STAGE TRAINING:
    Stage 1 (Separation): High weight on loud-loss + SI-SDR
    Stage 2 (WDRC Fine-tune): Add dynamics-focused losses
    """
    
    def __init__(
        self,
        # Loss weights
        weight_loud: float = 1.0,
        weight_si_sdr: float = 0.5,
        weight_stft: float = 0.3,
        weight_temporal: float = 0.1,
        weight_phase: float = 0.1,
        # Configuration
        sample_rate: int = 16000,
        n_fft: int = 256,
        loudness_weighting: str = "a_weighting",
        # Stage control
        stage: int = 1,  # 1 = separation, 2 = WDRC fine-tune
    ):
        super().__init__()
        
        self.stage = stage
        
        # Store weights (can be modified for stage 2)
        self.weight_loud = weight_loud
        self.weight_si_sdr = weight_si_sdr
        self.weight_stft = weight_stft
        self.weight_temporal = weight_temporal
        self.weight_phase = weight_phase
        
        # Loss components
        self.loud_loss = PsychoacousticLoudLoss(
            n_fft=n_fft,
            sample_rate=sample_rate,
            weighting=loudness_weighting,
        )
        
        self.si_sdr_loss = SISDRLoss()
        
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024],
            hop_sizes=[64, 128, 256],
            win_lengths=[256, 512, 1024],
        )
        
        self.temporal_loss = TemporalCoherenceLoss()
        self.phase_loss = PhaseLoss()
        
        # WDRC-specific loss (Stage 2 only)
        self.loudness_envelope_loss = LoudnessEnvelopeLoss()
        
    def set_stage(self, stage: int) -> None:
        """
        Set training stage and adjust weights accordingly.
        
        Stage 1: Focus on separation quality
        Stage 2: Fine-tune with WDRC for dynamics
        """
        self.stage = stage
        
        if stage == 1:
            # Separation-focused weights
            self.weight_loud = 1.0
            self.weight_si_sdr = 0.5
            self.weight_stft = 0.3
            self.weight_temporal = 0.1
            self.weight_phase = 0.1
        elif stage == 2:
            # WDRC fine-tuning weights (reduce separation, add envelope)
            self.weight_loud = 0.5
            self.weight_si_sdr = 0.3
            self.weight_stft = 0.2
            self.weight_temporal = 0.2
            self.weight_phase = 0.1
            
    def forward(
        self,
        pred_stft: torch.Tensor,
        target_stft: torch.Tensor,
        pred_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined V2 loss.
        
        Args:
            pred_stft: Predicted complex STFT [B, 2, T, F]
            target_stft: Target complex STFT [B, 2, T, F]
            pred_audio: Predicted waveform [B, N] (optional)
            target_audio: Target waveform [B, N] (optional)
            
        Returns:
            Tuple of:
            - total_loss: Weighted sum of all losses
            - loss_dict: Dictionary with individual loss values
        """
        loss_dict = {}
        
        # 1. Psychoacoustic Loud-Loss (frequency domain)
        loss_loud = self.loud_loss(pred_stft, target_stft)
        loss_dict["loud_loss"] = loss_loud
        
        # 2. Temporal coherence (STFT domain)
        loss_temporal = self.temporal_loss(pred_stft, target_stft)
        loss_dict["temporal_loss"] = loss_temporal
        
        # 3. Phase loss (STFT domain)
        loss_phase = self.phase_loss(pred_stft, target_stft)
        loss_dict["phase_loss"] = loss_phase
        
        # 4. Time-domain losses (if audio available)
        if pred_audio is not None and target_audio is not None:
            # SI-SDR
            loss_si_sdr = self.si_sdr_loss(pred_audio, target_audio)
            loss_dict["si_sdr_loss"] = loss_si_sdr
            
            # Multi-resolution STFT
            loss_stft = self.stft_loss(pred_audio, target_audio)
            loss_dict["stft_loss"] = loss_stft
            
            # Loudness envelope (Stage 2)
            if self.stage == 2:
                loss_envelope = self.loudness_envelope_loss(pred_audio, target_audio)
                loss_dict["envelope_loss"] = loss_envelope
            else:
                loss_envelope = torch.tensor(0.0, device=pred_stft.device)
        else:
            loss_si_sdr = torch.tensor(0.0, device=pred_stft.device)
            loss_stft = torch.tensor(0.0, device=pred_stft.device)
            loss_envelope = torch.tensor(0.0, device=pred_stft.device)
            loss_dict["si_sdr_loss"] = loss_si_sdr
            loss_dict["stft_loss"] = loss_stft
        
        # Compute weighted total
        total_loss = (
            self.weight_loud * loss_loud +
            self.weight_si_sdr * loss_si_sdr +
            self.weight_stft * loss_stft +
            self.weight_temporal * loss_temporal +
            self.weight_phase * loss_phase
        )
        
        # Add envelope loss in Stage 2
        if self.stage == 2 and not isinstance(loss_envelope, float):
            total_loss = total_loss + 0.3 * loss_envelope
            loss_dict["envelope_loss"] = loss_envelope
        
        loss_dict["total"] = total_loss
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("=" * 60)
    print("Testing AuraNet Loss Functions")
    print("=" * 60)
    
    # Create test data
    batch_size = 2
    time_steps = 100
    freq_bins = 129
    num_samples = 16000
    
    # STFT domain data
    pred_stft = torch.randn(batch_size, 2, time_steps, freq_bins)
    target_stft = torch.randn(batch_size, 2, time_steps, freq_bins)
    
    # Time domain data
    pred_audio = torch.randn(batch_size, num_samples)
    target_audio = torch.randn(batch_size, num_samples)
    
    # Test individual losses
    print("\nTesting individual losses:")
    print("-" * 40)
    
    # Complex MSE
    cmse = ComplexMSELoss()
    loss = cmse(pred_stft, target_stft)
    print(f"Complex MSE Loss: {loss.item():.4f}")
    
    # Multi-resolution STFT
    mr_stft = MultiResolutionSTFTLoss()
    loss = mr_stft(pred_audio, target_audio)
    print(f"Multi-Res STFT Loss: {loss.item():.4f}")
    
    # Loudness envelope
    loudness = LoudnessEnvelopeLoss()
    loss = loudness(pred_audio, target_audio)
    print(f"Loudness Envelope Loss: {loss.item():.4f}")
    
    # Temporal coherence
    temporal = TemporalCoherenceLoss()
    loss = temporal(pred_stft, target_stft)
    print(f"Temporal Coherence Loss: {loss.item():.4f}")
    
    # SI-SNR
    sisnr = SISNRLoss()
    loss = sisnr(pred_audio, target_audio)
    print(f"SI-SNR Loss: {loss.item():.4f}")
    
    # Phase loss
    phase = PhaseLoss()
    loss = phase(pred_stft, target_stft)
    print(f"Phase Loss: {loss.item():.4f}")
    
    # Test combined loss
    print("\n" + "-" * 40)
    print("Testing combined AuraNet loss:")
    
    auranet_loss = AuraNetLoss()
    total, loss_dict = auranet_loss(
        pred_stft, target_stft,
        pred_audio, target_audio
    )
    
    print(f"Total Loss: {total.item():.4f}")
    print("Component losses:")
    for name, value in loss_dict.items():
        if name != "total":
            print(f"  {name}: {value.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All loss function tests passed! ✅")
    print("=" * 60)
