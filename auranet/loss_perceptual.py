# =============================================================================
# AuraNet Perceptual Loss Stack — State-of-the-Art Audio Quality
# =============================================================================
#
# OBJECTIVE: Transform AuraNet from "high SI-SNR" → "perceptually natural"
#
# Target Metrics:
#   PESQ: ≥2.8 (from ~2.3-2.5)
#   STOI: ≥0.86 (from ~0.81)
#   SI-SNR: maintain ≥15 dB
#
# CORRECTED LOSS FORMULA:
#   L_total = 0.45 * LoudLoss (frequency-weighted log-power)
#           + 0.30 * MultiResolutionSTFTLoss ([256, 512, 1024])
#           + 0.15 * MelSpectrogramLoss (80 mels)
#           + 0.10 * SI_SNR_Loss (stabilizer)
#
# WARM-START STRATEGY:
#   Phase 1 (epochs 1-3): 0.6*SI-SNR + 0.4*STFT (stability)
#   Phase 2 (epochs 4+):  Full perceptual loss (quality)
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


# =============================================================================
# 1. LoudLoss — Frequency-Weighted Log-Power Domain
# =============================================================================

class LoudLoss(nn.Module):
    """
    Improved Loudness Loss operating in log-power domain.

    KEY FEATURES:
    1. Log-power domain: log(|X|^2 + eps) for perceptual alignment
    2. Frequency weighting: boost 1–4 kHz band (critical for intelligibility)
    3. A-weighting approximation for human loudness perception

    The 1–4 kHz band contains most speech consonant energy and is where
    hearing is most sensitive (ISO 226 equal-loudness contours).
    """

    def __init__(self, n_fft=512, hop_length=128, sample_rate=16000,
                 boost_low_hz=1000, boost_high_hz=4000, boost_factor=2.0,
                 eps=1e-6):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.sample_rate = sample_rate

        # Create frequency weighting curve
        n_freqs = n_fft // 2 + 1
        freqs = torch.linspace(0, sample_rate / 2, n_freqs)

        # Base weight = 1.0
        weights = torch.ones(n_freqs)

        # Boost 1-4 kHz region (intelligibility-critical)
        boost_mask = (freqs >= boost_low_hz) & (freqs <= boost_high_hz)
        weights[boost_mask] = boost_factor

        # Smooth transitions with cosine ramps
        ramp_width_hz = 200
        ramp_samples = int(ramp_width_hz / (sample_rate / 2) * n_freqs)

        # Low transition (before boost region)
        low_bin = (freqs < boost_low_hz).sum().item()
        if low_bin > ramp_samples:
            ramp = torch.linspace(1, boost_factor, ramp_samples)
            ramp = 0.5 * (1 - torch.cos(math.pi * torch.linspace(0, 1, ramp_samples))) * (boost_factor - 1) + 1
            weights[low_bin - ramp_samples:low_bin] = ramp

        # High transition (after boost region)
        high_bin = (freqs <= boost_high_hz).sum().item()
        if high_bin + ramp_samples < n_freqs:
            ramp = 0.5 * (1 + torch.cos(math.pi * torch.linspace(0, 1, ramp_samples))) * (boost_factor - 1) + 1
            weights[high_bin:high_bin + ramp_samples] = ramp

        # A-weighting approximation (simplified)
        # Roll off below 500 Hz and above 6 kHz
        low_rolloff = freqs < 500
        weights[low_rolloff] *= torch.clamp(freqs[low_rolloff] / 500, min=0.3)

        high_rolloff = freqs > 6000
        weights[high_rolloff] *= torch.clamp(1 - (freqs[high_rolloff] - 6000) / 2000, min=0.3)

        self.register_buffer('freq_weights', weights.unsqueeze(0).unsqueeze(0))  # [1, 1, F]
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, N] or [B, 1, N] predicted waveform
            target: [B, N] or [B, 1, N] clean waveform
        Returns:
            Scalar loss value
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # Compute STFT
        window = self.window.to(pred.device)

        pred_spec = torch.stft(pred, self.n_fft, self.hop_length,
                               window=window, center=True, return_complex=True)
        target_spec = torch.stft(target, self.n_fft, self.hop_length,
                                 window=window, center=True, return_complex=True)

        # Log-power domain
        pred_power = torch.clamp(torch.abs(pred_spec) ** 2, min=self.eps)
        target_power = torch.clamp(torch.abs(target_spec) ** 2, min=self.eps)

        pred_log = torch.log(pred_power + self.eps)
        target_log = torch.log(target_power + self.eps)

        # Clamp for numerical stability
        pred_log = torch.clamp(pred_log, min=-20, max=20)
        target_log = torch.clamp(target_log, min=-20, max=20)

        # Apply frequency weighting: [B, F, T] -> weighted difference
        weights = self.freq_weights.to(pred.device)  # [1, 1, F]

        diff = (pred_log - target_log).abs()  # [B, F, T]
        diff = diff.transpose(1, 2)  # [B, T, F]
        weighted_diff = diff * weights  # [B, T, F]

        return weighted_diff.mean()


# =============================================================================
# 2. Multi-Resolution STFT Loss — [256, 512, 1024]
# =============================================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss using FFT sizes [256, 512, 1024].

    For each resolution:
    - Spectral convergence (Frobenius norm ratio)
    - Log magnitude L1 loss

    Multiple resolutions capture both fine temporal detail (small FFT)
    and spectral detail (large FFT).
    """

    def __init__(self, fft_sizes=(256, 512, 1024),
                 hop_ratio=0.25, win_ratio=1.0, eps=1e-5):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = [int(fft * hop_ratio) for fft in fft_sizes]
        self.win_lengths = [int(fft * win_ratio) for fft in fft_sizes]
        self.eps = eps

        # Register windows as buffers
        for i, win_len in enumerate(self.win_lengths):
            self.register_buffer(f'window_{i}', torch.hann_window(win_len))

    def _compute_mag(self, x: torch.Tensor, fft_size: int, hop_size: int,
                     win_length: int, window: torch.Tensor) -> torch.Tensor:
        """Compute magnitude spectrogram."""
        if x.dim() == 3:
            x = x.squeeze(1)

        spec = torch.stft(x, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window.to(x.device),
                          center=True, return_complex=True)
        return torch.clamp(torch.abs(spec), min=self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, N] or [B, 1, N] predicted waveform
            target: [B, N] or [B, 1, N] clean waveform
        Returns:
            Scalar loss value
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        total_loss = 0.0

        for i, (fft_s, hop_s, win_l) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):

            window = getattr(self, f'window_{i}')

            pred_mag = self._compute_mag(pred, fft_s, hop_s, win_l, window)
            target_mag = self._compute_mag(target, fft_s, hop_s, win_l, window)

            # Align time dimension
            min_t = min(pred_mag.shape[-1], target_mag.shape[-1])
            pred_mag = pred_mag[..., :min_t]
            target_mag = target_mag[..., :min_t]

            # Spectral convergence: ||target - pred|| / ||target||
            diff_norm = torch.norm(target_mag - pred_mag, p='fro')
            target_norm = torch.norm(target_mag, p='fro') + self.eps
            sc_loss = diff_norm / target_norm
            sc_loss = torch.clamp(sc_loss, 0, 10)

            # Log magnitude L1
            log_pred = torch.log(pred_mag + self.eps)
            log_target = torch.log(target_mag + self.eps)
            log_pred = torch.clamp(log_pred, -20, 20)
            log_target = torch.clamp(log_target, -20, 20)
            mag_loss = F.l1_loss(log_pred, log_target)

            total_loss = total_loss + sc_loss + mag_loss

        return total_loss / len(self.fft_sizes)


# =============================================================================
# 3. Mel Spectrogram Loss — 80 Mels (torchaudio-style)
# =============================================================================

class MelSpectrogramLoss(nn.Module):
    """
    Mel spectrogram L1 loss using 80 mel bands.

    Mel scale approximates human auditory frequency resolution.
    80 mels is standard for speech (matches ASR/TTS systems).
    """

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256,
                 n_mels=80, f_min=0.0, f_max=8000.0, eps=1e-5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps

        # Create mel filterbank
        mel_fb = self._create_mel_filterbank(
            sample_rate, n_fft, n_mels, f_min, f_max
        )
        self.register_buffer('mel_fb', mel_fb)
        self.register_buffer('window', torch.hann_window(n_fft))

    @staticmethod
    def _create_mel_filterbank(sr: int, n_fft: int, n_mels: int,
                               f_min: float, f_max: float) -> torch.Tensor:
        """Create mel filterbank matrix [n_fft//2+1, n_mels]."""
        # Mel-scale conversion
        def hz_to_mel(hz):
            return 2595 * math.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])

        # Convert to FFT bins
        bins = torch.floor((n_fft + 1) * hz_points / sr).long()

        n_freqs = n_fft // 2 + 1
        fb = torch.zeros(n_freqs, n_mels)

        for m in range(n_mels):
            f_left = bins[m].item()
            f_center = bins[m + 1].item()
            f_right = bins[m + 2].item()

            # Rising edge
            for k in range(f_left, f_center):
                if f_center > f_left:
                    fb[k, m] = (k - f_left) / (f_center - f_left)

            # Falling edge
            for k in range(f_center, f_right):
                if f_right > f_center:
                    fb[k, m] = (f_right - k) / (f_right - f_center)

        return fb

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, N] or [B, 1, N] predicted waveform
            target: [B, N] or [B, 1, N] clean waveform
        Returns:
            Scalar loss value
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        window = self.window.to(pred.device)
        mel_fb = self.mel_fb.to(pred.device)

        # Compute STFT magnitude
        pred_spec = torch.stft(pred, self.n_fft, self.hop_length,
                               window=window, center=True, return_complex=True)
        target_spec = torch.stft(target, self.n_fft, self.hop_length,
                                 window=window, center=True, return_complex=True)

        pred_mag = torch.clamp(torch.abs(pred_spec), min=self.eps)  # [B, F, T]
        target_mag = torch.clamp(torch.abs(target_spec), min=self.eps)

        # Apply mel filterbank: [B, F, T] @ [F, M] -> [B, M, T]
        pred_mel = torch.matmul(pred_mag.transpose(1, 2), mel_fb)  # [B, T, M]
        target_mel = torch.matmul(target_mag.transpose(1, 2), mel_fb)

        # Log compression
        pred_log_mel = torch.log(pred_mel + self.eps)
        target_log_mel = torch.log(target_mel + self.eps)

        pred_log_mel = torch.clamp(pred_log_mel, -20, 20)
        target_log_mel = torch.clamp(target_log_mel, -20, 20)

        return F.l1_loss(pred_log_mel, target_log_mel)


# =============================================================================
# 4. SI-SNR Loss — Scale-Invariant Signal-to-Noise Ratio
# =============================================================================

class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio loss with NORMALIZATION.

    CRITICAL FIX: Raw SI-SNR values are ~30-50, while spectral losses are ~0.5-2.
    Without normalization, SI-SNR dominates even with low weight.

    Solution: Divide by typical SI-SNR value (30) to normalize to ~1-2 range.
    This ensures balanced contribution with spectral losses.
    """

    def __init__(self, eps=1e-5, normalize_scale=30.0):
        super().__init__()
        self.eps = eps
        self.normalize_scale = normalize_scale  # Divide output by this value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, N] or [B, 1, N] predicted waveform
            target: [B, N] or [B, 1, N] clean waveform
        Returns:
            Normalized negative SI-SNR (to minimize)
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

        # s_target = (<pred, target> / ||target||^2) * target
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / s_target_energy) * target

        # e_noise = pred - s_target
        e_noise = pred - s_target

        # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        signal_energy = torch.sum(s_target ** 2, dim=-1) + self.eps
        noise_energy = torch.sum(e_noise ** 2, dim=-1) + self.eps

        ratio = torch.clamp(signal_energy / noise_energy, min=self.eps, max=1e6)
        si_snr = 10 * torch.log10(ratio + self.eps)
        si_snr = torch.clamp(si_snr, -100, 100)

        # NORMALIZE to match spectral loss scale (~1-2 range)
        # Raw negative SI-SNR is ~30-50, divide by 30 to get ~1-2
        normalized = -si_snr.mean() / self.normalize_scale

        return normalized


# =============================================================================
# 5. Harmonic Preservation Loss (Optional)
# =============================================================================

class HarmonicPreservationLoss(nn.Module):
    """
    Lightweight harmonic preservation loss using spectral contrast.

    Encourages preservation of harmonic structures (peaks vs. valleys)
    by matching the spectral contrast pattern between enhanced and clean.

    Spectral contrast = difference between peaks and valleys in each band.
    Harmonics show up as high contrast; noise shows low contrast.
    """

    def __init__(self, n_fft=1024, hop_length=256, n_bands=6, eps=1e-5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bands = n_bands
        self.eps = eps

        self.register_buffer('window', torch.hann_window(n_fft))

        # Create octave-based frequency bands
        n_freqs = n_fft // 2 + 1
        band_edges = torch.logspace(
            math.log10(1), math.log10(n_freqs - 1), n_bands + 1
        ).long()
        band_edges = torch.clamp(band_edges, 0, n_freqs - 1)
        self.register_buffer('band_edges', band_edges)

    def _spectral_contrast(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral contrast for each band.

        Args:
            mag: [B, F, T] magnitude spectrogram
        Returns:
            contrast: [B, n_bands, T] spectral contrast
        """
        B, F, T = mag.shape
        contrasts = []

        for i in range(self.n_bands):
            low = self.band_edges[i].item()
            high = self.band_edges[i + 1].item()

            if high <= low:
                high = low + 1

            band = mag[:, low:high, :]  # [B, band_size, T]

            # Peak: top 20% mean
            # Valley: bottom 20% mean
            k = max(1, band.shape[1] // 5)

            # Sort along frequency axis
            sorted_band, _ = torch.sort(band, dim=1)

            valley = sorted_band[:, :k, :].mean(dim=1)  # [B, T]
            peak = sorted_band[:, -k:, :].mean(dim=1)   # [B, T]

            # Contrast = peak - valley (in log domain for stability)
            contrast = torch.log(peak + self.eps) - torch.log(valley + self.eps)
            contrast = torch.clamp(contrast, -10, 10)
            contrasts.append(contrast)

        return torch.stack(contrasts, dim=1)  # [B, n_bands, T]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, N] or [B, 1, N] predicted waveform
            target: [B, N] or [B, 1, N] clean waveform
        Returns:
            Scalar loss value
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        window = self.window.to(pred.device)

        # Compute magnitude spectrograms
        pred_spec = torch.stft(pred, self.n_fft, self.hop_length,
                               window=window, center=True, return_complex=True)
        target_spec = torch.stft(target, self.n_fft, self.hop_length,
                                 window=window, center=True, return_complex=True)

        pred_mag = torch.clamp(torch.abs(pred_spec), min=self.eps)
        target_mag = torch.clamp(torch.abs(target_spec), min=self.eps)

        # Compute spectral contrast
        pred_contrast = self._spectral_contrast(pred_mag)
        target_contrast = self._spectral_contrast(target_mag)

        return F.l1_loss(pred_contrast, target_contrast)


# =============================================================================
# Combined Perceptual Loss
# =============================================================================

class PerceptualLoss(nn.Module):
    """
    Combined perceptual loss stack for state-of-the-art audio quality.

    L_total = w_loud * LoudLoss
            + w_stft * MultiResolutionSTFTLoss
            + w_mel * MelSpectrogramLoss
            + w_sisnr * SISNRLoss

    CORRECTED WEIGHTS (for PESQ ≥2.8, STOI ≥0.86):
    - LoudLoss: 0.45 (primary perceptual driver)
    - MultiResSTFT: 0.30 (spectral detail)
    - Mel: 0.15 (perceptual bands)
    - SI-SNR: 0.10 (stabilizer only)
    """

    def __init__(self,
                 weight_loud=0.45,
                 weight_stft=0.30,
                 weight_mel=0.15,
                 weight_sisnr=0.10,
                 weight_harmonic=0.0,
                 use_harmonic=False,
                 sample_rate=16000):
        super().__init__()

        self.w_loud = weight_loud
        self.w_stft = weight_stft
        self.w_mel = weight_mel
        self.w_sisnr = weight_sisnr
        self.w_harmonic = weight_harmonic
        self.use_harmonic = use_harmonic

        # Initialize loss components
        self.loud_loss = LoudLoss(
            n_fft=512, hop_length=128, sample_rate=sample_rate,
            boost_low_hz=1000, boost_high_hz=4000, boost_factor=2.0
        )

        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=(256, 512, 1024),
            hop_ratio=0.25, win_ratio=1.0
        )

        self.mel_loss = MelSpectrogramLoss(
            sample_rate=sample_rate, n_fft=1024, hop_length=256,
            n_mels=80, f_min=0.0, f_max=sample_rate // 2
        )

        self.sisnr_loss = SISNRLoss()

        if use_harmonic:
            self.harmonic_loss = HarmonicPreservationLoss(
                n_fft=1024, hop_length=256, n_bands=6
            )

        print(f"PerceptualLoss initialized (CORRECTED WEIGHTS):")
        print(f"  LoudLoss (1-4kHz boost):     {weight_loud:.2f}")
        print(f"  MultiResSTFT [256,512,1024]: {weight_stft:.2f}")
        print(f"  MelSpectrogram (80 mels):    {weight_mel:.2f}")
        print(f"  SI-SNR (stabilizer):         {weight_sisnr:.2f}")
        if use_harmonic:
            print(f"  HarmonicPreservation:        {weight_harmonic:.2f}")

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor,
                pred_stft: Optional[torch.Tensor] = None,
                target_stft: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined perceptual loss.

        Args:
            pred_audio: [B, N] or [B, 1, N] enhanced waveform
            target_audio: [B, N] or [B, 1, N] clean waveform
            pred_stft: (optional) [B, 2, T, F] for additional losses
            target_stft: (optional) [B, 2, T, F]

        Returns:
            total_loss: Scalar
            loss_dict: Dictionary with individual components
        """
        loss_dict = {}

        # 1. LoudLoss (frequency-weighted log-power)
        l_loud = self.loud_loss(pred_audio, target_audio)
        loss_dict['loud'] = l_loud
        total = self.w_loud * l_loud

        # 2. Multi-Resolution STFT Loss
        l_stft = self.stft_loss(pred_audio, target_audio)
        loss_dict['multi_res_stft'] = l_stft
        total = total + self.w_stft * l_stft

        # 3. Mel Spectrogram Loss
        l_mel = self.mel_loss(pred_audio, target_audio)
        loss_dict['mel'] = l_mel
        total = total + self.w_mel * l_mel

        # 4. SI-SNR Loss (stabilizer)
        l_sisnr = self.sisnr_loss(pred_audio, target_audio)
        loss_dict['si_snr'] = l_sisnr
        total = total + self.w_sisnr * l_sisnr

        # 5. Harmonic Preservation (optional)
        if self.use_harmonic:
            l_harmonic = self.harmonic_loss(pred_audio, target_audio)
            loss_dict['harmonic'] = l_harmonic
            total = total + self.w_harmonic * l_harmonic

        loss_dict['total'] = total
        return total, loss_dict

    def forward_with_breakdown(self, pred_audio: torch.Tensor,
                               target_audio: torch.Tensor) -> Dict:
        """
        Compute loss and return breakdown dictionary (for WarmStartLoss compatibility).

        Returns:
            Dictionary with 'total' and individual loss components
        """
        total, loss_dict = self.forward(pred_audio, target_audio)
        # Convert tensors to floats for the breakdown
        breakdown = {'total': total}
        for k, v in loss_dict.items():
            if k != 'total':
                breakdown[k] = v.item() if isinstance(v, torch.Tensor) else v
        return breakdown


# =============================================================================
# Loss Explanation Summary
# =============================================================================
"""
LOSS COMPONENT EXPLANATIONS:

1. LoudLoss (weight=0.5)
   - Operates in log-power domain: log(|X|² + eps)
   - Boosts 1–4 kHz region by 2x (intelligibility-critical band)
   - Applies A-weighting approximation for human perception
   - WHY: Human hearing is most sensitive in this range (ISO 226)

2. MultiResolutionSTFTLoss (weight=0.25)
   - FFT sizes: [256, 512, 1024]
   - Spectral convergence + log magnitude L1
   - Small FFT: temporal detail (transients, consonants)
   - Large FFT: spectral detail (harmonics, pitch)
   - WHY: Captures different time-frequency trade-offs

3. MelSpectrogramLoss (weight=0.15)
   - 80 mel bands (standard for speech systems)
   - Log compression for loudness perception
   - Mel scale matches human frequency resolution
   - WHY: Aligns with perceptual frequency importance

4. SISNRLoss (weight=0.10)
   - Scale-invariant SNR in time domain
   - Standard speech enhancement metric
   - Low weight to prevent over-optimization
   - WHY: Stabilizes training, maintains denoising

5. HarmonicPreservationLoss (weight=0.05, optional)
   - Spectral contrast matching (peaks vs valleys)
   - Preserves voiced speech harmonics
   - Reduces metallic/robotic artifacts
   - WHY: Harmonics are key to natural speech quality

RECOMMENDED HYPERPARAMETERS:
- Learning rate: 1e-4 (fine-tuning)
- Epochs: 5–10
- Scheduler: CosineAnnealingLR or ReduceLROnPlateau
- Gradient clipping: 3.0
- Batch size: 16–32

OUTPUT ACTIVATION:
Replace: output.clamp(-1, 1)
With:    output = torch.tanh(output)
Reason:  tanh is smooth and differentiable; clamp has zero gradients at edges
"""


# ==============================================================================
# WARM-START LOSS (Two-Phase Training Strategy)
# ==============================================================================

class WarmStartLoss(nn.Module):
    """
    Two-phase warm-start training loss.

    Phase 1 (epochs 1-3): Stability-focused
        L_total = 0.6 * SI_SNR + 0.4 * STFT_loss
        -> Establishes basic denoising before perceptual refinement

    Phase 2 (epochs 4+): Quality-focused
        L_total = Full PerceptualLoss (0.45*Loud + 0.30*STFT + 0.15*Mel + 0.10*SI-SNR)
        -> Fine-tunes for perceptual quality (PESQ/STOI)

    Usage:
        loss_fn = WarmStartLoss(warmup_epochs=3)
        for epoch in range(num_epochs):
            loss_fn.set_epoch(epoch + 1)  # 1-indexed
            for batch in dataloader:
                loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        warmup_epochs: int = 3,
        phase1_sisnr_weight: float = 0.6,
        phase1_stft_weight: float = 0.4,
        fft_sizes: list = [256, 512, 1024],
        sample_rate: int = 16000
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.phase1_sisnr_weight = phase1_sisnr_weight
        self.phase1_stft_weight = phase1_stft_weight
        self.current_epoch = 1

        # Phase 1 losses (stability)
        self.sisnr_loss = SISNRLoss()
        self.stft_loss = MultiResolutionSTFTLoss(fft_sizes=tuple(fft_sizes))

        # Phase 2 loss (perceptual quality)
        self.perceptual_loss = PerceptualLoss(
            use_harmonic=False,  # Disabled by default for stability
            sample_rate=sample_rate
        )

        print(f"[WarmStartLoss] Initialized:")
        print(f"  Phase 1 (epochs 1-{warmup_epochs}): {phase1_sisnr_weight}*SI-SNR + {phase1_stft_weight}*STFT")
        print(f"  Phase 2 (epochs {warmup_epochs+1}+): Full PerceptualLoss")

    def set_epoch(self, epoch: int):
        """Update current epoch (1-indexed)."""
        old_phase = 1 if self.current_epoch <= self.warmup_epochs else 2
        self.current_epoch = epoch
        new_phase = 1 if epoch <= self.warmup_epochs else 2

        if old_phase != new_phase:
            print(f"[WarmStartLoss] Switching from Phase {old_phase} to Phase {new_phase} at epoch {epoch}")

    def get_phase(self) -> int:
        """Return current training phase (1 or 2)."""
        return 1 if self.current_epoch <= self.warmup_epochs else 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on current training phase.

        Args:
            pred: Predicted signal [B, T] or [B, 1, T]
            target: Target signal [B, T] or [B, 1, T]

        Returns:
            Total loss scalar
        """
        # Squeeze channel dimension if present
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        phase = self.get_phase()

        if phase == 1:
            # Phase 1: Stability-focused (SI-SNR + STFT)
            sisnr = self.sisnr_loss(pred, target)
            stft = self.stft_loss(pred, target)
            total = self.phase1_sisnr_weight * sisnr + self.phase1_stft_weight * stft
            return total
        else:
            # Phase 2: Quality-focused (full perceptual loss)
            total, _ = self.perceptual_loss(pred, target)
            return total

    def forward_with_breakdown(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute loss with detailed breakdown for logging.

        Returns:
            Dictionary with 'total', 'phase', and individual loss components
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        phase = self.get_phase()

        if phase == 1:
            sisnr = self.sisnr_loss(pred, target)
            stft = self.stft_loss(pred, target)
            total = self.phase1_sisnr_weight * sisnr + self.phase1_stft_weight * stft
            return {
                'total': total,
                'phase': 1,
                'sisnr': sisnr.item(),
                'stft': stft.item(),
            }
        else:
            breakdown = self.perceptual_loss.forward_with_breakdown(pred, target)
            breakdown['phase'] = 2
            return breakdown


def create_warmstart_loss(warmup_epochs: int = 3, sample_rate: int = 16000) -> WarmStartLoss:
    """
    Factory function to create WarmStartLoss with recommended settings.

    Args:
        warmup_epochs: Number of epochs for Phase 1 (default: 3)
        sample_rate: Audio sample rate (default: 16000)

    Returns:
        WarmStartLoss instance
    """
    return WarmStartLoss(
        warmup_epochs=warmup_epochs,
        phase1_sisnr_weight=0.6,
        phase1_stft_weight=0.4,
        fft_sizes=[256, 512, 1024],
        sample_rate=sample_rate
    )
