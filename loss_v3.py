# =============================================================================
# AuraNet V3 Loss — Improved Multi-Objective Loss
# =============================================================================
#
# KEY CHANGES:
# 1. SI-SNR loss as primary (scale-invariant, perceptually motivated)
# 2. Complex compressed MSE (power-law compression before MSE)
# 3. Multi-resolution STFT unchanged (already good)
# 4. Removed loudness/temporal losses (SI-SNR covers these better)
# 5. Added magnitude-phase aware loss
#
# WHY THIS IMPROVES OVER V1:
# - SI-SNR directly optimizes speech quality metric
# - Compressed MSE emphasizes low-energy speech components
# - Combined freq + time domain supervision avoids local minima
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

    def __init__(self, eps=1e-8):
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

        si_snr = 10 * torch.log10(
            torch.sum(s_target ** 2, dim=-1) /
            (torch.sum(e_noise ** 2, dim=-1) + self.eps)
        )

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
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase_cos = real / (mag + 1e-8)
        phase_sin = imag / (mag + 1e-8)

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

    def _stft_mag(self, x, fft_size, hop_size, win_length, window):
        if x.dim() == 3:
            x = x.squeeze(1)
        window = window.to(x.device)
        spec = torch.stft(x, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window,
                          center=True, return_complex=True)
        return torch.abs(spec)

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
            sc = torch.norm(tm - pm, p='fro') / (torch.norm(tm, p='fro') + 1e-8)
            # Log magnitude
            lm = F.l1_loss(torch.log(pm + 1e-8), torch.log(tm + 1e-8))
            total += sc + lm

        return total / len(self.fft_sizes)


class AuraNetV3Loss(nn.Module):
    """
    Combined loss for AuraNet V3.

    Total = w1 * SI-SNR + w2 * CompressedMSE + w3 * MultiResSTFT

    Default weights tuned for speech enhancement:
    - SI-SNR (1.0): Primary perceptual quality driver
    - Compressed MSE (0.5): Frequency-domain detail, especially quiet parts
    - Multi-res STFT (0.3): Multi-scale spectral matching
    """

    def __init__(self,
                 weight_sisnr=1.0,
                 weight_compressed_mse=0.5,
                 weight_stft=0.3,
                 compress_factor=0.3):
        super().__init__()
        self.w_sisnr = weight_sisnr
        self.w_cmse = weight_compressed_mse
        self.w_stft = weight_stft

        self.sisnr_loss = SISNRLoss()
        self.cmse_loss = CompressedMSELoss(compress_factor)
        self.stft_loss = MultiResSTFTLoss()

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
        else:
            loss_dict["si_snr"] = torch.tensor(0.0, device=pred_stft.device)
            loss_dict["multi_res_stft"] = torch.tensor(0.0, device=pred_stft.device)

        loss_dict["total"] = total
        return total, loss_dict
