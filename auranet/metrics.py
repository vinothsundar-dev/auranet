# =============================================================================
# AuraNet V3 — Industry-Standard Evaluation Metrics
# =============================================================================
# PESQ (Perceptual Evaluation of Speech Quality) — ITU-T P.862
# STOI (Short-Time Objective Intelligibility) — Taal et al. 2011
# SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
#
# These are the three standard metrics used by DNS Challenge, VoiceBank+DEMAND.
# =============================================================================

import torch
import numpy as np
from typing import Dict, Optional

# Lazy imports — these are optional dependencies
_pesq_fn = None
_stoi_fn = None


def _get_pesq():
    global _pesq_fn
    if _pesq_fn is None:
        try:
            from pesq import pesq
            _pesq_fn = pesq
        except ImportError:
            return None
    return _pesq_fn


def _get_stoi():
    global _stoi_fn
    if _stoi_fn is None:
        try:
            from pystoi import stoi
            _stoi_fn = stoi
        except ImportError:
            return None
    return _stoi_fn


def compute_pesq(pred: np.ndarray, target: np.ndarray,
                 sr: int = 16000, mode: str = 'wb') -> float:
    """
    Compute PESQ score (ITU-T P.862).

    Args:
        pred: Enhanced audio (1D numpy array)
        target: Clean reference audio (1D numpy array)
        sr: Sample rate (8000 for 'nb', 16000 for 'wb')
        mode: 'wb' (wideband, 16kHz) or 'nb' (narrowband, 8kHz)

    Returns:
        PESQ score: 1.0 (bad) to 4.5 (excellent)
        Returns -1.0 if pesq library not installed.
    """
    pesq_fn = _get_pesq()
    if pesq_fn is None:
        return -1.0

    min_len = min(len(pred), len(target))
    pred = pred[:min_len].astype(np.float64)
    target = target[:min_len].astype(np.float64)

    # PESQ requires at least 0.25s of audio
    if min_len < sr // 4:
        return -1.0

    try:
        return float(pesq_fn(sr, target, pred, mode))
    except Exception:
        return -1.0


def compute_stoi(pred: np.ndarray, target: np.ndarray,
                 sr: int = 16000, extended: bool = True) -> float:
    """
    Compute STOI (Short-Time Objective Intelligibility).

    Args:
        pred: Enhanced audio (1D numpy array)
        target: Clean reference audio (1D numpy array)
        sr: Sample rate
        extended: Use extended STOI (better correlation with intelligibility)

    Returns:
        STOI score: 0.0 (unintelligible) to 1.0 (perfect)
        Returns -1.0 if pystoi library not installed.
    """
    stoi_fn = _get_stoi()
    if stoi_fn is None:
        return -1.0

    min_len = min(len(pred), len(target))
    pred = pred[:min_len].astype(np.float64)
    target = target[:min_len].astype(np.float64)

    if min_len < sr // 4:
        return -1.0

    try:
        return float(stoi_fn(target, pred, sr, extended=extended))
    except Exception:
        return -1.0


def compute_si_snr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute SI-SNR in dB (numpy version).

    Args:
        pred: Enhanced audio (1D numpy)
        target: Clean reference (1D numpy)

    Returns:
        SI-SNR in dB (higher = better). Typical range: -5 to 25 dB.
    """
    min_len = min(len(pred), len(target))
    pred = pred[:min_len].astype(np.float64) - np.mean(pred[:min_len])
    target = target[:min_len].astype(np.float64) - np.mean(target[:min_len])

    dot = np.sum(pred * target)
    s_target = (dot / (np.sum(target ** 2) + eps)) * target
    e_noise = pred - s_target

    return float(10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + eps)))


def evaluate_batch(pred_audio: torch.Tensor, target_audio: torch.Tensor,
                   sr: int = 16000) -> Dict[str, float]:
    """
    Compute all metrics for a batch of audio.

    Args:
        pred_audio: [B, N] enhanced waveforms (torch tensor)
        target_audio: [B, N] clean waveforms (torch tensor)
        sr: Sample rate

    Returns:
        Dict with average PESQ, STOI, SI-SNR across batch.
    """
    if pred_audio.dim() == 3:
        pred_audio = pred_audio.squeeze(1)
    if target_audio.dim() == 3:
        target_audio = target_audio.squeeze(1)

    pred_np = pred_audio.detach().cpu().numpy()
    target_np = target_audio.detach().cpu().numpy()

    pesq_scores = []
    stoi_scores = []
    sisnr_scores = []

    for p, t in zip(pred_np, target_np):
        sisnr_scores.append(compute_si_snr(p, t))

        pesq_val = compute_pesq(p, t, sr)
        if pesq_val > 0:
            pesq_scores.append(pesq_val)

        stoi_val = compute_stoi(p, t, sr)
        if stoi_val > 0:
            stoi_scores.append(stoi_val)

    return {
        'pesq': float(np.mean(pesq_scores)) if pesq_scores else -1.0,
        'stoi': float(np.mean(stoi_scores)) if stoi_scores else -1.0,
        'si_snr': float(np.mean(sisnr_scores)),
    }


def evaluate_per_snr(pred_audio: torch.Tensor, target_audio: torch.Tensor,
                     snr_values: torch.Tensor, sr: int = 16000,
                     snr_bins: Optional[list] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by SNR bin.

    Args:
        pred_audio: [B, N] enhanced waveforms
        target_audio: [B, N] clean waveforms
        snr_values: [B] SNR value for each sample
        sr: Sample rate
        snr_bins: List of (low, high) tuples. Default: 5dB bins from -5 to 25.

    Returns:
        Dict mapping bin label to metrics dict.
    """
    if snr_bins is None:
        snr_bins = [(-5, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]

    if pred_audio.dim() == 3:
        pred_audio = pred_audio.squeeze(1)
    if target_audio.dim() == 3:
        target_audio = target_audio.squeeze(1)

    pred_np = pred_audio.detach().cpu().numpy()
    target_np = target_audio.detach().cpu().numpy()
    snr_np = snr_values.detach().cpu().numpy()

    results = {}
    for low, high in snr_bins:
        mask = (snr_np >= low) & (snr_np < high)
        if not mask.any():
            continue
        label = f"{low:+d}to{high:+d}dB"
        p_bin = pred_np[mask]
        t_bin = target_np[mask]

        pesq_s, stoi_s, sisnr_s = [], [], []
        for p, t in zip(p_bin, t_bin):
            sisnr_s.append(compute_si_snr(p, t))
            pv = compute_pesq(p, t, sr)
            if pv > 0:
                pesq_s.append(pv)
            sv = compute_stoi(p, t, sr)
            if sv > 0:
                stoi_s.append(sv)

        results[label] = {
            'pesq': float(np.mean(pesq_s)) if pesq_s else -1.0,
            'stoi': float(np.mean(stoi_s)) if stoi_s else -1.0,
            'si_snr': float(np.mean(sisnr_s)),
            'count': int(mask.sum()),
        }

    return results


def check_metrics_available() -> Dict[str, bool]:
    """Check which metric libraries are installed."""
    return {
        'pesq': _get_pesq() is not None,
        'stoi': _get_stoi() is not None,
    }
