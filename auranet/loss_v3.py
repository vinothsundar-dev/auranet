# =============================================================================
# AuraNet V3 Loss — Perceptual Multi-Objective Loss
# =============================================================================
#
# KEY CHANGES (Perceptual Quality Overhaul):
# 1. SI-SNR loss as PRIMARY driver (weight=1.0)
# 2. Compressed complex MSE for freq-domain detail (weight=0.3, reduced)
# 3. Multi-resolution STFT for spectral shape (weight=0.2, reduced)
# 4. NEW: Energy preservation loss — prevents over-suppression
# 5. NEW: Log-mel perceptual loss — aligns with human hearing
# 6. NEW: Loudness normalization helper for training & inference
#
# WHY THIS IMPROVES PERCEPTUAL QUALITY:
# - SI-SNR dominance → model optimizes what ears hear
# - Energy preservation → enhanced audio keeps speech loudness
# - Log-mel loss → emphasizes perceptually important bands
# - Reduced STFT weight → less numerical optimization, more quality
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio loss.

    Used in most state-of-the-art speech enhancement/separation.
    Directly correlates with perceptual quality.

    SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = (<s_hat, s> / ||s||^2) * s
          e_noise  = s_hat - s_target
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: [B, N] predicted waveform
            target: [B, N] clean waveform
        Returns:
            Negative SI-SNR (to minimize)
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # Zero-mean normalization
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        # s_target = (<s_hat, s> / ||s||^2) * s
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / s_target_energy) * target

        # e_noise = s_hat - s_target
        e_noise = pred - s_target

        signal_energy = torch.sum(s_target ** 2, dim=-1) + self.eps
        noise_energy = torch.sum(e_noise ** 2, dim=-1) + self.eps
        ratio = torch.clamp(signal_energy / (noise_energy + self.eps), min=self.eps, max=1e6)
        si_snr = 10 * torch.log10(ratio + self.eps)
        si_snr = torch.clamp(si_snr, min=-100.0, max=100.0)

        return -si_snr.mean()  # negative because we minimize


class CompressedMSELoss(nn.Module):
    """
    Power-law compressed MSE on complex STFT.

    Applies |X|^c * exp(j*angle(X)) compression before MSE.
    This emphasizes low-energy speech components (consonants, sibilants)
    that are critical for intelligibility but would be ignored by plain MSE.

    c=0.3 is standard in speech enhancement literature.
    """

    def __init__(self, compress_factor=0.3):
        super().__init__()
        self.c = compress_factor

    def _compress(self, stft):
        """Apply power-law compression to complex STFT."""
        real = stft[:, 0]  # [B, T, F]
        imag = stft[:, 1]
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-5)
        mag = torch.clamp(mag, min=1e-5)
        phase_cos = real / (mag + 1e-5)
        phase_sin = imag / (mag + 1e-5)

        mag_compressed = mag.pow(self.c)
        return mag_compressed * phase_cos, mag_compressed * phase_sin, mag_compressed

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2, T, F] predicted STFT
            target: [B, 2, T, F] clean STFT
        """
        pred_real_c, pred_imag_c, pred_mag_c = self._compress(pred)
        tgt_real_c, tgt_imag_c, tgt_mag_c = self._compress(target)

        # Complex compressed MSE
        loss_real = F.mse_loss(pred_real_c, tgt_real_c)
        loss_imag = F.mse_loss(pred_imag_c, tgt_imag_c)

        # Magnitude-only compressed MSE (extra supervision)
        loss_mag = F.mse_loss(pred_mag_c, tgt_mag_c)

        return loss_real + loss_imag + 0.5 * loss_mag


class MultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log magnitude)."""

    def __init__(self,
                 fft_sizes=(512, 1024, 2048),
                 hop_sizes=(50, 120, 240),
                 win_lengths=(240, 600, 1200)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        # Register windows as buffers
        for i, wl in enumerate(win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(wl))
        self.eps = 1e-5

    def _stft_mag(self, x, fft_size, hop_size, win_length, window):
        if x.dim() == 3:
            x = x.squeeze(1)
        window = window.to(x.device)
        spec = torch.stft(x, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window,
                          center=True, return_complex=True)
        return torch.clamp(torch.abs(spec), min=self.eps)

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        min_len = min(pred.shape[-1], target.shape[-1])
        pred, target = pred[..., :min_len], target[..., :min_len]

        total = 0.0
        for i, (fft_s, hop_s, win_l) in enumerate(
                zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f"window_{i}")
            pm = self._stft_mag(pred, fft_s, hop_s, win_l, window)
            tm = self._stft_mag(target, fft_s, hop_s, win_l, window)
            min_t = min(pm.shape[-1], tm.shape[-1])
            pm, tm = pm[..., :min_t], tm[..., :min_t]

            # Spectral convergence
            tm_norm = torch.norm(tm, p='fro')
            sc = torch.norm(tm - pm, p='fro') / (tm_norm + self.eps)
            sc = torch.clamp(sc, 0.0, 10.0)
            # Log magnitude
            log_pm = torch.log(torch.clamp(pm, min=1e-5) + 1e-5)
            log_tm = torch.log(torch.clamp(tm, min=1e-5) + 1e-5)
            log_pm = torch.clamp(log_pm, min=-20.0, max=20.0)
            log_tm = torch.clamp(log_tm, min=-20.0, max=20.0)
            lm = F.l1_loss(log_pm, log_tm)
            total += sc + lm

        return total / len(self.fft_sizes)


class EnergyPreservationLoss(nn.Module):
    """
    Penalizes energy mismatch between enhanced and clean audio.

    Prevents the model from over-suppressing speech (making output too quiet).
    Uses RMS ratio so it's scale-aware but not position-dependent.

    loss = |rms(enhanced) - rms(clean)| / (rms(clean) + eps)
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        pred_rms = torch.sqrt(torch.clamp(torch.mean(pred ** 2, dim=-1), min=0.0) + self.eps)
        target_rms = torch.sqrt(torch.clamp(torch.mean(target ** 2, dim=-1), min=0.0) + self.eps)
        pred_rms = torch.clamp(pred_rms, min=self.eps)
        target_rms = torch.clamp(target_rms, min=self.eps)

        # Relative energy difference
        loss = torch.abs(pred_rms - target_rms) / (target_rms + self.eps)
        return torch.clamp(loss.mean(), 0.0, 10.0)


class LogMelLoss(nn.Module):
    """
    Log-mel spectrogram loss — lightweight perceptual proxy.

    Mel scale approximates human auditory frequency resolution.
    Log compression models loudness perception.
    Much cheaper than a full perceptual model but captures the key property:
    humans care more about low/mid frequencies than high.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_mels=64, eps=1e-5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.register_buffer(
            'mel_fb',
            self._mel_filterbank(sample_rate, n_fft, n_mels)
        )
        self.register_buffer(
            'window',
            torch.hann_window(n_fft)
        )

    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels):
        """Create mel filterbank matrix [n_fft//2+1, n_mels]."""
        f_min, f_max = 0.0, sr / 2.0
        mel_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
        mel_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        bins = torch.floor((n_fft + 1) * hz_points / sr).long()
        n_freqs = n_fft // 2 + 1
        fb = torch.zeros(n_freqs, n_mels)

        for m in range(n_mels):
            f_left = bins[m]
            f_center = bins[m + 1]
            f_right = bins[m + 2]
            for k in range(f_left, f_center):
                if f_center > f_left:
                    fb[k, m] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right > f_center:
                    fb[k, m] = (f_right - k) / (f_right - f_center)
        return fb

    def _log_mel(self, x):
        """Compute log-mel spectrogram from waveform [B, N]."""
        spec = torch.stft(x, self.n_fft, self.hop_length,
                          window=self.window.to(x.device),
                          center=True, return_complex=True)
        mag = torch.clamp(torch.abs(spec), min=self.eps)  # [B, F, T]
        mel = torch.matmul(mag.transpose(1, 2), self.mel_fb.to(x.device))  # [B, T, n_mels]
        mel = torch.clamp(mel, min=1e-5)
        log_mel = torch.log(mel + 1e-5)
        log_mel = torch.clamp(log_mel, min=-20.0, max=20.0)
        return log_mel

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        pred_mel = self._log_mel(pred)
        target_mel = self._log_mel(target)
        return F.l1_loss(pred_mel, target_mel)


def loudness_normalize(enhanced, clean, min_scale=0.5, max_scale=2.0, eps=1e-5):
    """
    Match RMS of enhanced audio to clean audio.

    Differentiable — can be used in training loop.
    Clamped to [0.5, 2.0] to avoid instability.

    Args:
        enhanced: [B, N] enhanced waveform
        clean: [B, N] clean reference waveform
        min_scale: minimum gain factor
        max_scale: maximum gain factor

    Returns:
        Loudness-normalized enhanced audio [B, N]
    """
    enhanced_rms = torch.sqrt(torch.clamp(torch.mean(enhanced ** 2, dim=-1, keepdim=True), min=0.0) + eps)
    clean_rms = torch.sqrt(torch.clamp(torch.mean(clean ** 2, dim=-1, keepdim=True), min=0.0) + eps)
    enhanced_rms = torch.clamp(enhanced_rms, min=eps)
    clean_rms = torch.clamp(clean_rms, min=eps)
    scale = (clean_rms / (enhanced_rms + eps)).clamp(min_scale, max_scale)
    return enhanced * scale


class TemporalConsistencyLoss(nn.Module):
    """
    Penalizes abrupt frame-to-frame changes in the enhanced STFT.

    Computed as L1 of the first-order temporal difference between
    enhanced and clean STFT magnitude. Reduces musical noise / flutter
    artifacts without requiring architecture changes.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred_stft, target_stft):
        """
        Args:
            pred_stft: [B, 2, T, F]
            target_stft: [B, 2, T, F]
        """
        pred_mag = torch.sqrt(pred_stft[:, 0] ** 2 + pred_stft[:, 1] ** 2 + self.eps)
        tgt_mag = torch.sqrt(target_stft[:, 0] ** 2 + target_stft[:, 1] ** 2 + self.eps)
        pred_mag = torch.clamp(pred_mag, min=self.eps)
        tgt_mag = torch.clamp(tgt_mag, min=self.eps)

        # Temporal difference: how magnitude changes between adjacent frames
        pred_diff = pred_mag[:, 1:, :] - pred_mag[:, :-1, :]
        tgt_diff = tgt_mag[:, 1:, :] - tgt_mag[:, :-1, :]

        return F.l1_loss(pred_diff, tgt_diff)


class AuraNetV3Loss(nn.Module):
    """
    Combined loss for AuraNet V3 — Perceptual quality focused.

    Total = w1*SI-SNR + w2*CompressedMSE + w3*MultiResSTFT
          + w4*EnergyPreservation + w5*LogMel + w6*TemporalConsistency

    Weight rationale (v3.1 rebalance):
    - SI-SNR (1.0): Primary perceptual driver — correlates with MOS
    - Multi-res STFT (0.8): Increased — better spectral shape, less metallic
    - Compressed MSE (0.5): Increased — freq detail for consonants
    - Log-mel (0.3): Increased — perceptual frequency weighting
    - Energy preservation (0.1): Prevents over-suppression / softness
    - Temporal consistency (0.1): Reduces musical noise / flutter
    """

    def __init__(self,
                 weight_sisnr=1.0,
                 weight_compressed_mse=0.5,
                 weight_stft=0.8,
                 compress_factor=0.3,
                 weight_energy=0.1,
                 weight_logmel=0.3,
                 weight_temporal=0.1):
        super().__init__()
        self.w_sisnr = weight_sisnr
        self.w_cmse = weight_compressed_mse
        self.w_stft = weight_stft
        self.w_energy = weight_energy
        self.w_logmel = weight_logmel
        self.w_temporal = weight_temporal

        self.sisnr_loss = SISNRLoss()
        self.cmse_loss = CompressedMSELoss(compress_factor)
        self.stft_loss = MultiResSTFTLoss()
        self.energy_loss = EnergyPreservationLoss()
        self.logmel_loss = LogMelLoss()
        self.temporal_loss = TemporalConsistencyLoss()

    def forward(self, pred_stft, target_stft,
                pred_audio=None, target_audio=None):
        """
        Args:
            pred_stft: [B, 2, T, F]
            target_stft: [B, 2, T, F]
            pred_audio: [B, N] (optional)
            target_audio: [B, N] (optional)

        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}

        # Compressed complex MSE (always available)
        l_cmse = self.cmse_loss(pred_stft, target_stft)
        loss_dict["compressed_mse"] = l_cmse

        total = self.w_cmse * l_cmse

        # Time-domain losses
        if pred_audio is not None and target_audio is not None:
            l_sisnr = self.sisnr_loss(pred_audio, target_audio)
            loss_dict["si_snr"] = l_sisnr
            total = total + self.w_sisnr * l_sisnr

            l_stft = self.stft_loss(pred_audio, target_audio)
            loss_dict["multi_res_stft"] = l_stft
            total = total + self.w_stft * l_stft

            # Energy preservation — prevents over-suppression
            l_energy = self.energy_loss(pred_audio, target_audio)
            loss_dict["energy"] = l_energy
            total = total + self.w_energy * l_energy

            # Log-mel perceptual loss
            l_logmel = self.logmel_loss(pred_audio, target_audio)
            loss_dict["logmel"] = l_logmel
            total = total + self.w_logmel * l_logmel
        else:
            loss_dict["si_snr"] = torch.tensor(0.0, device=pred_stft.device)
            loss_dict["multi_res_stft"] = torch.tensor(0.0, device=pred_stft.device)
            loss_dict["energy"] = torch.tensor(0.0, device=pred_stft.device)
            loss_dict["logmel"] = torch.tensor(0.0, device=pred_stft.device)

        # Temporal consistency — reduces musical noise / frame jitter
        l_temporal = self.temporal_loss(pred_stft, target_stft)
        loss_dict["temporal"] = l_temporal
        total = total + self.w_temporal * l_temporal

        loss_dict["total"] = total
        return total, loss_dict
