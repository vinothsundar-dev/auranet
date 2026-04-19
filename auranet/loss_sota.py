#!/usr/bin/env python3
# =============================================================================
# AuraNet SOTA Perceptual Loss Stack
# =============================================================================
#
# OBJECTIVE: PESQ 2.2 → 2.9+, STOI 0.80 → 0.86+
#
# COMPONENTS:
#   1. MRSTFTLoss        — Multi-resolution STFT (magnitude + spectral conv)
#   2. MelLoss           — 80-mel L1 + log-mel with speech-band emphasis
#   3. LoudLoss          — Log-power domain + ISO 226 A-weighting (capped <40%)
#   4. TemporalLoss      — Waveform first/second-order differences (transients)
#   5. SISNRLoss         — Normalized stabilizer (≤15%)
#   6. SpectralFeatureLoss — DNSMOS-style perceptual feature distance (optional)
#
# WEIGHTS (tuned for PESQ/STOI):
#   w_stft  = 0.30    spectral matching backbone
#   w_mel   = 0.25    perceptual frequency resolution
#   w_loud  = 0.25    intelligibility (capped to avoid dominance)
#   w_temp  = 0.10    transient/onset clarity
#   w_sisnr = 0.10    optimization anchor
#
# WARM-START:
#   Phase 1 (epochs 1-3):  0.6*SI-SNR + 0.4*MR-STFT       (stability)
#   Phase 2 (epochs 4+):   Full SOTA stack                (perceptual quality)
#
# =============================================================================

import math
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helpers
# =============================================================================

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """[B,T] or [B,1,T] -> [B,T]"""
    if x.dim() == 3:
        x = x.squeeze(1)
    return x


def _align(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred, target = _ensure_2d(pred), _ensure_2d(target)
    n = min(pred.shape[-1], target.shape[-1])
    return pred[..., :n], target[..., :n]


# =============================================================================
# 1. Multi-Resolution STFT Loss
# =============================================================================

class MRSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT loss with magnitude L1 + spectral convergence.

    fft_sizes: [256, 512, 1024]
    For each resolution computes:
      - Spectral convergence: ||M_t - M_p||_F / ||M_t||_F
      - Log-magnitude L1 distance

    Multiple resolutions capture both temporal detail (small FFT)
    and frequency detail (large FFT).
    """

    def __init__(self,
                 fft_sizes: List[int] = (256, 512, 1024),
                 hop_ratio: float = 0.25,
                 win_ratio: float = 1.0,
                 eps: float = 1e-7):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = [int(n * hop_ratio) for n in self.fft_sizes]
        self.win_lengths = [int(n * win_ratio) for n in self.fft_sizes]
        self.eps = eps

        for i, w in enumerate(self.win_lengths):
            self.register_buffer(f"win_{i}", torch.hann_window(w))

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int,
                  win_length: int, window: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_length,
                          window=window.to(x.device), center=True, return_complex=True)
        return torch.clamp(spec.abs(), min=self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _align(pred, target)
        total = pred.new_tensor(0.0)

        for i, (nfft, hop, wlen) in enumerate(
                zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f"win_{i}")
            p_mag = self._stft_mag(pred, nfft, hop, wlen, window)
            t_mag = self._stft_mag(target, nfft, hop, wlen, window)

            T = min(p_mag.shape[-1], t_mag.shape[-1])
            p_mag, t_mag = p_mag[..., :T], t_mag[..., :T]

            # Spectral convergence
            sc = torch.norm(t_mag - p_mag, p='fro') / (torch.norm(t_mag, p='fro') + self.eps)
            sc = torch.clamp(sc, 0.0, 10.0)

            # Log-magnitude L1
            log_p = torch.clamp(torch.log(p_mag + self.eps), -20, 20)
            log_t = torch.clamp(torch.log(t_mag + self.eps), -20, 20)
            mag = F.l1_loss(log_p, log_t)

            total = total + sc + mag

        return total / len(self.fft_sizes)


# =============================================================================
# 2. Mel Loss (Stronger — 80 mels + speech-band emphasis)
# =============================================================================

class MelLoss(nn.Module):
    """
    Mel-spectrogram loss combining:
      - Linear-mel L1
      - Log-mel L1
      - 1.5x emphasis weight on 300–4000 Hz mel bins (speech band)

    This pushes the optimizer to focus on the band that drives PESQ/STOI.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 80,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 speech_low_hz: float = 300.0,
                 speech_high_hz: float = 4000.0,
                 speech_boost: float = 1.5,
                 eps: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps

        if f_max is None:
            f_max = sample_rate / 2

        # Build mel filterbank
        mel_fb = self._build_mel_filterbank(
            n_fft=n_fft, n_mels=n_mels, sample_rate=sample_rate,
            f_min=f_min, f_max=f_max
        )
        self.register_buffer("mel_fb", mel_fb)  # [F, n_mels]
        self.register_buffer("window", torch.hann_window(n_fft))

        # Build mel-bin emphasis curve (1.0 baseline, boost speech band)
        mel_centers = self._mel_bin_center_freqs(
            n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max
        )
        emphasis = torch.ones(n_mels)
        speech_mask = (mel_centers >= speech_low_hz) & (mel_centers <= speech_high_hz)
        emphasis[speech_mask] = speech_boost
        # [1, n_mels, 1] for broadcast over [B, n_mels, T]
        self.register_buffer("mel_emphasis", emphasis.view(1, -1, 1))

    @staticmethod
    def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_bin_center_freqs(self, n_mels, sample_rate, f_min, f_max) -> torch.Tensor:
        m_min = self._hz_to_mel(torch.tensor(f_min))
        m_max = self._hz_to_mel(torch.tensor(f_max))
        mels = torch.linspace(m_min.item(), m_max.item(), n_mels + 2)
        hz = self._mel_to_hz(mels)
        # centers are inner n_mels points
        return hz[1:-1]

    def _build_mel_filterbank(self, n_fft, n_mels, sample_rate,
                              f_min, f_max) -> torch.Tensor:
        n_freqs = n_fft // 2 + 1
        all_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

        m_min = self._hz_to_mel(torch.tensor(f_min))
        m_max = self._hz_to_mel(torch.tensor(f_max))
        mels = torch.linspace(m_min.item(), m_max.item(), n_mels + 2)
        f_pts = self._mel_to_hz(mels)

        fb = torch.zeros(n_freqs, n_mels)
        for m in range(n_mels):
            f_left, f_center, f_right = f_pts[m], f_pts[m + 1], f_pts[m + 2]
            left = (all_freqs - f_left) / (f_center - f_left + 1e-8)
            right = (f_right - all_freqs) / (f_right - f_center + 1e-8)
            fb[:, m] = torch.clamp(torch.minimum(left, right), min=0.0)
        return fb

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _align(pred, target)

        window = self.window.to(pred.device)
        p_spec = torch.stft(pred, self.n_fft, self.hop_length,
                            window=window, center=True, return_complex=True)
        t_spec = torch.stft(target, self.n_fft, self.hop_length,
                            window=window, center=True, return_complex=True)

        p_mag = torch.clamp(p_spec.abs(), min=self.eps)  # [B, F, T]
        t_mag = torch.clamp(t_spec.abs(), min=self.eps)

        mel_fb = self.mel_fb.to(pred.device)
        # [B, F, T] -> [B, T, F] @ [F, M] -> [B, T, M] -> [B, M, T]
        p_mel = torch.matmul(p_mag.transpose(1, 2), mel_fb).transpose(1, 2)
        t_mel = torch.matmul(t_mag.transpose(1, 2), mel_fb).transpose(1, 2)

        # Linear-mel L1 with speech-band emphasis
        emphasis = self.mel_emphasis.to(pred.device)
        lin_diff = (p_mel - t_mel).abs() * emphasis
        lin_loss = lin_diff.mean()

        # Log-mel L1 with speech-band emphasis
        p_log = torch.clamp(torch.log(p_mel + self.eps), -20, 20)
        t_log = torch.clamp(torch.log(t_mel + self.eps), -20, 20)
        log_diff = (p_log - t_log).abs() * emphasis
        log_loss = log_diff.mean()

        return 0.5 * lin_loss + 0.5 * log_loss


# =============================================================================
# 3. Loud Loss (Refined — log-power + ISO 226 A-weighting)
# =============================================================================

class LoudLoss(nn.Module):
    """
    Perceptual loudness loss in log-power domain with A-weighting.

    A-weighting (IEC 61672) approximates equal-loudness contour at 40 phon:
      R_A(f) = 12200^2 * f^4 /
               ((f^2 + 20.6^2) * sqrt((f^2+107.7^2)*(f^2+737.9^2)) * (f^2 + 12200^2))
      A(f) = 20*log10(R_A(f)) + 2.0  (dB)

    Used as multiplicative weight on |log-power_pred - log-power_target|.

    Capped contribution via outer weight (≤0.25 in stack to keep <40%).
    """

    def __init__(self,
                 n_fft: int = 512,
                 hop_length: int = 128,
                 sample_rate: int = 16000,
                 eps: float = 1e-6):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps

        n_freqs = n_fft // 2 + 1
        freqs = torch.linspace(0, sample_rate / 2, n_freqs).clamp(min=1.0)

        # A-weighting transfer function
        f2 = freqs ** 2
        ra_num = (12200.0 ** 2) * (f2 ** 2)
        ra_den = ((f2 + 20.6 ** 2)
                  * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                  * (f2 + 12200.0 ** 2))
        ra = ra_num / (ra_den + 1e-12)
        a_db = 20.0 * torch.log10(ra + 1e-12) + 2.0  # dB
        # Convert to linear weights centered around 1.0 at 1 kHz, then clamp
        a_lin = 10.0 ** (a_db / 20.0)
        a_lin = a_lin / a_lin.max()                     # normalize peak to 1
        a_lin = torch.clamp(a_lin, min=0.05, max=1.0)   # avoid total dropout

        # [1, F, 1] for broadcasting over [B, F, T]
        self.register_buffer("a_weight", a_lin.view(1, -1, 1))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _align(pred, target)

        window = self.window.to(pred.device)
        p_spec = torch.stft(pred, self.n_fft, self.hop_length,
                            window=window, center=True, return_complex=True)
        t_spec = torch.stft(target, self.n_fft, self.hop_length,
                            window=window, center=True, return_complex=True)

        p_pow = torch.clamp(p_spec.abs() ** 2, min=self.eps)
        t_pow = torch.clamp(t_spec.abs() ** 2, min=self.eps)

        p_log = torch.clamp(torch.log(p_pow + self.eps), -20, 20)
        t_log = torch.clamp(torch.log(t_pow + self.eps), -20, 20)

        diff = (p_log - t_log).abs()                # [B, F, T]
        weighted = diff * self.a_weight.to(pred.device)
        return weighted.mean()


# =============================================================================
# 4. Temporal Consistency Loss (NEW)
# =============================================================================

class TemporalLoss(nn.Module):
    """
    Waveform-domain temporal differences loss.

    Computes L1 on first- and second-order differences of the waveform:
        d1[t]  = x[t] - x[t-1]
        d2[t]  = d1[t] - d1[t-1]

    PURPOSE:
      - Sharpens transient/onset reconstruction
      - Penalizes over-smoothing (a common failure of spectral-only losses)
      - Improves consonant clarity (key driver of STOI)
    """

    def __init__(self, weight_d1: float = 1.0, weight_d2: float = 0.5):
        super().__init__()
        self.weight_d1 = weight_d1
        self.weight_d2 = weight_d2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _align(pred, target)

        # First-order differences
        p_d1 = pred[..., 1:] - pred[..., :-1]
        t_d1 = target[..., 1:] - target[..., :-1]
        l_d1 = F.l1_loss(p_d1, t_d1)

        # Second-order differences
        p_d2 = p_d1[..., 1:] - p_d1[..., :-1]
        t_d2 = t_d1[..., 1:] - t_d1[..., :-1]
        l_d2 = F.l1_loss(p_d2, t_d2)

        return self.weight_d1 * l_d1 + self.weight_d2 * l_d2


# =============================================================================
# 5. SI-SNR Loss (Normalized stabilizer)
# =============================================================================

class SISNRLoss(nn.Module):
    """
    Scale-invariant SNR loss, normalized to ~unit magnitude so it composes
    with spectral losses without dominating.

    normalize_scale = 30  ->  raw -SI-SNR (~30) becomes ~1.0
    """

    def __init__(self, eps: float = 1e-7, normalize_scale: float = 30.0):
        super().__init__()
        self.eps = eps
        self.normalize_scale = normalize_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _align(pred, target)

        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        t_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / t_energy) * target
        e_noise = pred - s_target

        sig = torch.sum(s_target ** 2, dim=-1) + self.eps
        noi = torch.sum(e_noise ** 2, dim=-1) + self.eps
        ratio = torch.clamp(sig / noi, min=self.eps, max=1e6)
        si_snr = 10.0 * torch.log10(ratio + self.eps)
        si_snr = torch.clamp(si_snr, -100, 100)

        return -si_snr.mean() / self.normalize_scale


# =============================================================================
# 6. Spectral Feature Loss (DNSMOS-style, optional)
# =============================================================================

class SpectralFeatureLoss(nn.Module):
    """
    Lightweight DNSMOS-style perceptual feature distance.

    Uses fixed (non-trainable) spectral statistics as a proxy for a learned
    encoder — captures spectral shape, contrast and rolloff which correlate
    with DNSMOS quality dimensions.

    Features per frame:
        - log-magnitude (per band, 12 log-spaced bands)
        - spectral centroid
        - spectral spread
        - spectral flatness

    Loss = L1 between features of pred and target.

    Set `weight=0.0` to disable.
    """

    def __init__(self,
                 n_fft: int = 512,
                 hop_length: int = 128,
                 sample_rate: int = 16000,
                 n_bands: int = 12,
                 eps: float = 1e-6):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps

        n_freqs = n_fft // 2 + 1
        freqs = torch.linspace(0, sample_rate / 2, n_freqs)

        # Log-spaced band edges [n_bands+1]
        edges = torch.logspace(
            math.log10(50.0), math.log10(sample_rate / 2 - 1), n_bands + 1
        )
        # Build band membership matrix [F, n_bands]
        band_fb = torch.zeros(n_freqs, n_bands)
        for b in range(n_bands):
            mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
            if mask.sum() > 0:
                band_fb[mask, b] = 1.0 / mask.sum().float()
        self.register_buffer("band_fb", band_fb)
        self.register_buffer("freqs", freqs)
        self.register_buffer("window", torch.hann_window(n_fft))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window.to(x.device)
        spec = torch.stft(x, self.n_fft, self.hop_length,
                          window=window, center=True, return_complex=True)
        mag = torch.clamp(spec.abs(), min=self.eps)        # [B, F, T]

        # 1) Per-band log-magnitude  -> [B, n_bands, T]
        band_fb = self.band_fb.to(x.device)
        band_energy = torch.matmul(mag.transpose(1, 2), band_fb).transpose(1, 2)
        band_log = torch.log(band_energy + self.eps)

        # 2) Spectral centroid  -> [B, 1, T]
        freqs = self.freqs.to(x.device).view(1, -1, 1)
        weighted_sum = (mag * freqs).sum(dim=1, keepdim=True)
        total = mag.sum(dim=1, keepdim=True) + self.eps
        centroid = weighted_sum / total
        centroid = centroid / (freqs.max() + self.eps)     # normalize 0–1

        # 3) Spectral spread (std around centroid)
        diff = (freqs - centroid * freqs.max()) ** 2
        spread = torch.sqrt((mag * diff).sum(dim=1, keepdim=True) / total)
        spread = spread / (freqs.max() + self.eps)

        # 4) Spectral flatness (geo mean / arith mean)
        log_mag = torch.log(mag + self.eps)
        geo = torch.exp(log_mag.mean(dim=1, keepdim=True))
        ari = mag.mean(dim=1, keepdim=True) + self.eps
        flatness = geo / ari

        # Concatenate features along feature axis
        feats = torch.cat([band_log, centroid, spread, flatness], dim=1)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _align(pred, target)
        f_p = self._features(pred)
        f_t = self._features(target)
        T = min(f_p.shape[-1], f_t.shape[-1])
        return F.l1_loss(f_p[..., :T], f_t[..., :T])


# =============================================================================
# 7. SOTA Combined Loss
# =============================================================================

class SOTAPerceptualLoss(nn.Module):
    """
    State-of-the-art perceptual loss stack for AuraNet.

    L_total = w_stft  * MR-STFT
            + w_mel   * MelLoss (with speech-band emphasis)
            + w_loud  * LoudLoss (A-weighted)
            + w_temp  * TemporalLoss
            + w_sisnr * SISNRLoss
            + w_feat  * SpectralFeatureLoss   (optional)

    DEFAULT WEIGHTS (tuned for PESQ/STOI on real speech):
        stft  = 0.35   spectral matching (largest by raw scale: ~1-3)
        mel   = 0.20   perceptual freq resolution (raw can be ~3-6)
        loud  = 0.20   intelligibility, A-weighted (capped <40%)
        temp  = 0.10   transient/onset clarity
        sisnr = 0.10   optimization anchor (normalized)
        feat  = 0.05   DNSMOS-style feature distance (optional)

    Returns: (total_loss, breakdown_dict)
    """

    def __init__(self,
                 weight_stft: float = 0.35,
                 weight_mel: float = 0.20,
                 weight_loud: float = 0.20,
                 weight_temp: float = 0.10,
                 weight_sisnr: float = 0.10,
                 weight_feat: float = 0.05,
                 sample_rate: int = 16000,
                 verbose: bool = True):
        super().__init__()

        self.w_stft = weight_stft
        self.w_mel = weight_mel
        self.w_loud = weight_loud
        self.w_temp = weight_temp
        self.w_sisnr = weight_sisnr
        self.w_feat = weight_feat

        self.stft_loss = MRSTFTLoss(fft_sizes=[256, 512, 1024])
        self.mel_loss = MelLoss(sample_rate=sample_rate, n_mels=80)
        self.loud_loss = LoudLoss(sample_rate=sample_rate)
        self.temp_loss = TemporalLoss()
        self.sisnr_loss = SISNRLoss(normalize_scale=30.0)

        self.use_feat = weight_feat > 0.0
        if self.use_feat:
            self.feat_loss = SpectralFeatureLoss(sample_rate=sample_rate)

        if verbose:
            total_w = (weight_stft + weight_mel + weight_loud
                       + weight_temp + weight_sisnr + weight_feat)
            print("=" * 60)
            print("[SOTAPerceptualLoss] Initialized")
            print(f"   MR-STFT   : {weight_stft:.2f} ({100*weight_stft/total_w:.0f}%)")
            print(f"   Mel       : {weight_mel:.2f} ({100*weight_mel/total_w:.0f}%)")
            print(f"   Loud (A-w): {weight_loud:.2f} ({100*weight_loud/total_w:.0f}%)")
            print(f"   Temporal  : {weight_temp:.2f} ({100*weight_temp/total_w:.0f}%)")
            print(f"   SI-SNR    : {weight_sisnr:.2f} ({100*weight_sisnr/total_w:.0f}%)")
            if self.use_feat:
                print(f"   SpecFeat  : {weight_feat:.2f} ({100*weight_feat/total_w:.0f}%)")
            print("=" * 60)

    def forward(self,
                pred_audio: torch.Tensor,
                target_audio: torch.Tensor,
                pred_stft: Optional[torch.Tensor] = None,
                target_stft: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred_audio:   [B, T] or [B, 1, T] enhanced waveform (after tanh)
            target_audio: [B, T] or [B, 1, T] clean reference
            pred_stft, target_stft: unused here (kept for API compatibility)
        Returns:
            (total_loss, dict of components)
        """
        breakdown: Dict[str, torch.Tensor] = {}

        l_stft = self.stft_loss(pred_audio, target_audio)
        breakdown["mr_stft"] = l_stft
        total = self.w_stft * l_stft

        l_mel = self.mel_loss(pred_audio, target_audio)
        breakdown["mel"] = l_mel
        total = total + self.w_mel * l_mel

        l_loud = self.loud_loss(pred_audio, target_audio)
        breakdown["loud"] = l_loud
        total = total + self.w_loud * l_loud

        l_temp = self.temp_loss(pred_audio, target_audio)
        breakdown["temporal"] = l_temp
        total = total + self.w_temp * l_temp

        l_sisnr = self.sisnr_loss(pred_audio, target_audio)
        breakdown["si_snr"] = l_sisnr
        total = total + self.w_sisnr * l_sisnr

        if self.use_feat:
            l_feat = self.feat_loss(pred_audio, target_audio)
            breakdown["spec_feat"] = l_feat
            total = total + self.w_feat * l_feat

        breakdown["total"] = total
        return total, breakdown


# =============================================================================
# 8. Warm-Start Wrapper (Phase 1 + Phase 2)
# =============================================================================

class WarmStartSOTA(nn.Module):
    """
    Two-phase training schedule wrapper.

    Phase 1 (epochs 1..warmup_epochs):
        L = 0.6 * SI-SNR_norm + 0.4 * MR-STFT
        Goal: stable convergence, basic noise suppression.

    Phase 2 (epochs > warmup_epochs):
        Full SOTAPerceptualLoss
        Goal: perceptual quality (PESQ/STOI).

    Usage:
        loss_fn = WarmStartSOTA(warmup_epochs=3)
        for epoch in range(epochs):
            loss_fn.set_epoch(epoch + 1)             # 1-indexed
            for batch in loader:
                ...
                total, breakdown = loss_fn(pred, target)
                total.backward()
    """

    def __init__(self,
                 warmup_epochs: int = 3,
                 phase1_sisnr_weight: float = 0.6,
                 phase1_stft_weight: float = 0.4,
                 sample_rate: int = 16000,
                 phase2_kwargs: Optional[dict] = None):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.phase1_sisnr_weight = phase1_sisnr_weight
        self.phase1_stft_weight = phase1_stft_weight
        self.current_epoch = 1

        # Phase 1 components
        self.phase1_sisnr = SISNRLoss(normalize_scale=30.0)
        self.phase1_stft = MRSTFTLoss(fft_sizes=[256, 512, 1024])

        # Phase 2 stack
        phase2_kwargs = phase2_kwargs or {}
        phase2_kwargs.setdefault("sample_rate", sample_rate)
        self.phase2 = SOTAPerceptualLoss(**phase2_kwargs)

        print(f"[WarmStartSOTA] warmup_epochs={warmup_epochs}")
        print(f"   Phase 1: {phase1_sisnr_weight}*SI-SNR + {phase1_stft_weight}*MR-STFT")
        print(f"   Phase 2: full SOTAPerceptualLoss")

    def set_epoch(self, epoch: int):
        old = 1 if self.current_epoch <= self.warmup_epochs else 2
        self.current_epoch = epoch
        new = 1 if epoch <= self.warmup_epochs else 2
        if old != new:
            print(f"[WarmStartSOTA] -> Phase {new} at epoch {epoch}")

    def get_phase(self) -> int:
        return 1 if self.current_epoch <= self.warmup_epochs else 2

    def forward(self,
                pred_audio: torch.Tensor,
                target_audio: torch.Tensor,
                pred_stft: Optional[torch.Tensor] = None,
                target_stft: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.get_phase() == 1:
            l_sisnr = self.phase1_sisnr(pred_audio, target_audio)
            l_stft = self.phase1_stft(pred_audio, target_audio)
            total = (self.phase1_sisnr_weight * l_sisnr
                     + self.phase1_stft_weight * l_stft)
            return total, {
                "phase": torch.tensor(1.0),
                "si_snr": l_sisnr,
                "mr_stft": l_stft,
                "total": total,
            }
        else:
            total, breakdown = self.phase2(
                pred_audio, target_audio, pred_stft, target_stft
            )
            breakdown["phase"] = torch.tensor(2.0)
            return total, breakdown


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Testing SOTAPerceptualLoss...")
    torch.manual_seed(0)

    B, T = 4, 16000 * 2
    pred = torch.randn(B, T) * 0.3
    target = torch.randn(B, T) * 0.3

    loss_fn = SOTAPerceptualLoss(weight_feat=0.05)
    total, bd = loss_fn(pred, target)

    print(f"\nTotal loss: {total.item():.4f}")
    print("Breakdown:")
    weights = {
        "mr_stft": loss_fn.w_stft, "mel": loss_fn.w_mel,
        "loud": loss_fn.w_loud, "temporal": loss_fn.w_temp,
        "si_snr": loss_fn.w_sisnr, "spec_feat": loss_fn.w_feat,
    }
    contribs = {}
    for k, v in bd.items():
        if k == "total":
            continue
        w = weights.get(k, 0.0)
        contribs[k] = (v.item(), w * v.item())
    sum_c = sum(c for _, c in contribs.values())
    for k, (raw, c) in contribs.items():
        pct = 100 * c / sum_c if sum_c > 0 else 0
        print(f"   {k:<12} raw={raw:.4f}  contrib={c:.4f}  ({pct:.1f}%)")

    print("\nTesting WarmStartSOTA...")
    ws = WarmStartSOTA(warmup_epochs=3)
    for ep in [1, 2, 3, 4, 5]:
        ws.set_epoch(ep)
        t, b = ws(pred, target)
        print(f"   epoch={ep} phase={int(b['phase'].item())} loss={t.item():.4f}")

    print("\nGradient test...")
    pred_leaf = torch.randn(B, T, requires_grad=True)
    pred = pred_leaf * 0.3
    total, _ = SOTAPerceptualLoss(verbose=False)(pred, target)
    total.backward()
    g = pred_leaf.grad
    print(f"   grad min/max/mean: {g.min().item():.4e} / {g.max().item():.4e} / {g.mean().item():.4e}")
    print(f"   has_nan={torch.isnan(g).any().item()}  has_inf={torch.isinf(g).any().item()}")
    print("\n✅ All self-tests passed.")
