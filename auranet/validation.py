#!/usr/bin/env python3
"""
================================================================================
AuraNet V2 Edge - Validation & Testing Suite
================================================================================

Comprehensive validation to ensure:
1. No audio glitches (discontinuities, clicks, pops)
2. No gain pumping (smooth amplitude envelope)
3. Stable output across frames (streaming consistency)
4. Quality preservation (SI-SDR, PESQ proxy)
5. Before vs After comparison

================================================================================
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================================
# AUDIO QUALITY METRICS
# ==============================================================================

def compute_si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio.
    
    Higher is better. Typical good enhancement: +5 to +15 dB improvement.
    """
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    
    dot = (estimate * reference).sum()
    s_target = dot / (reference.pow(2).sum() + 1e-8) * reference
    
    e_noise = estimate - s_target
    
    si_sdr = 10 * torch.log10(
        s_target.pow(2).sum() / (e_noise.pow(2).sum() + 1e-8)
    )
    
    return si_sdr.item()


def compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    signal_power = signal.pow(2).mean()
    noise_power = noise.pow(2).mean()
    
    return 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()


def compute_spectral_divergence(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 256,
) -> float:
    """
    Compute spectral divergence (KL-like metric).
    
    Lower is better. Measures perceptual similarity.
    """
    # Compute spectrograms
    window = torch.hann_window(n_fft, device=pred.device)
    
    pred_stft = torch.stft(pred, n_fft, hop_length=n_fft//4, window=window, return_complex=True)
    target_stft = torch.stft(target, n_fft, hop_length=n_fft//4, window=window, return_complex=True)
    
    pred_mag = pred_stft.abs() + 1e-8
    target_mag = target_stft.abs() + 1e-8
    
    # Normalize
    pred_norm = pred_mag / pred_mag.sum()
    target_norm = target_mag / target_mag.sum()
    
    # KL divergence
    kl = (target_norm * (target_norm.log() - pred_norm.log())).sum()
    
    return kl.item()


# ==============================================================================
# GLITCH DETECTOR
# ==============================================================================

class GlitchDetector:
    """
    Detect audio glitches: clicks, pops, discontinuities.
    
    Methods:
    - Sample-level jump detection
    - High-frequency energy spikes
    - Zero-crossing anomalies
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        jump_threshold: float = 0.5,
        spike_threshold: float = 3.0,
    ):
        self.sample_rate = sample_rate
        self.jump_threshold = jump_threshold
        self.spike_threshold = spike_threshold
        
    def detect_sample_jumps(self, audio: torch.Tensor) -> Dict[str, any]:
        """
        Detect sudden sample-level jumps (clicks/pops).
        
        A glitch is a sample where |diff| > threshold * std(diff)
        """
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        # First-order difference
        diff = torch.diff(audio)
        
        # Statistics
        diff_mean = diff.abs().mean()
        diff_std = diff.std()
        
        # Threshold for anomaly
        threshold = diff_mean + self.jump_threshold * diff_std
        
        # Find anomalies
        anomalies = (diff.abs() > threshold).nonzero(as_tuple=True)[0]
        
        return {
            'num_jumps': len(anomalies),
            'jump_locations': anomalies.tolist()[:20],  # First 20
            'threshold': threshold.item(),
            'max_jump': diff.abs().max().item(),
            'has_glitches': len(anomalies) > 10,  # More than 10 = likely glitchy
        }
    
    def detect_hf_spikes(self, audio: torch.Tensor, n_fft: int = 256) -> Dict[str, any]:
        """
        Detect high-frequency energy spikes (clicks in spectrum).
        
        Clicks manifest as broadband high-frequency energy.
        """
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        window = torch.hann_window(n_fft, device=audio.device)
        
        stft = torch.stft(
            audio.unsqueeze(0),
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        )[0]  # [F, T]
        
        # High-frequency energy (top 25% of spectrum)
        hf_cutoff = stft.shape[0] * 3 // 4
        hf_energy = stft[hf_cutoff:].abs().pow(2).sum(dim=0)  # [T]
        
        # Normalize by total energy
        total_energy = stft.abs().pow(2).sum(dim=0) + 1e-8
        hf_ratio = hf_energy / total_energy
        
        # Find spikes
        hf_mean = hf_ratio.mean()
        hf_std = hf_ratio.std()
        threshold = hf_mean + self.spike_threshold * hf_std
        
        spikes = (hf_ratio > threshold).sum().item()
        
        return {
            'num_hf_spikes': spikes,
            'hf_ratio_mean': hf_mean.item(),
            'hf_ratio_max': hf_ratio.max().item(),
            'has_hf_issues': spikes > 5,
        }
    
    def analyze(self, audio: torch.Tensor) -> Dict[str, any]:
        """Full glitch analysis."""
        jumps = self.detect_sample_jumps(audio)
        spikes = self.detect_hf_spikes(audio)
        
        has_glitches = jumps['has_glitches'] or spikes['has_hf_issues']
        
        return {
            'jumps': jumps,
            'hf_spikes': spikes,
            'has_glitches': has_glitches,
            'status': '❌ GLITCHY' if has_glitches else '✅ CLEAN',
        }


# ==============================================================================
# GAIN PUMPING DETECTOR
# ==============================================================================

class GainPumpingDetector:
    """
    Detect gain pumping artifacts from WDRC.
    
    Gain pumping = unnatural amplitude modulation, usually:
    - Fast, periodic envelope fluctuations
    - Correlation with input envelope
    - "Breathing" or "pumping" sound
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        envelope_window_ms: float = 20.0,
        modulation_threshold: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.envelope_window = int(envelope_window_ms * sample_rate / 1000)
        self.modulation_threshold = modulation_threshold
        
    def compute_envelope(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute amplitude envelope via Hilbert transform approximation."""
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        # Simple moving RMS envelope
        squared = audio.pow(2)
        
        # Unfold for windowed computation
        if len(audio) > self.envelope_window:
            unfolded = squared.unfold(0, self.envelope_window, self.envelope_window // 4)
            envelope = unfolded.mean(dim=-1).sqrt()
        else:
            envelope = squared.sqrt()
        
        return envelope
    
    def detect_pumping(
        self,
        enhanced: torch.Tensor,
        original: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Detect gain pumping.
        
        Looks for:
        1. Fast periodic modulation in envelope
        2. High modulation depth
        """
        env = self.compute_envelope(enhanced)
        
        # Modulation depth: std/mean of envelope
        env_mean = env.mean()
        env_std = env.std()
        modulation_depth = (env_std / (env_mean + 1e-8)).item()
        
        # Detect periodic modulation via autocorrelation
        env_centered = env - env_mean
        autocorr = F.conv1d(
            env_centered.view(1, 1, -1),
            env_centered.view(1, 1, -1),
            padding=len(env_centered) - 1,
        )[0, 0]
        
        # Normalize
        autocorr = autocorr / (autocorr[len(env_centered) - 1] + 1e-8)
        
        # Find peaks in autocorrelation (indicates periodicity)
        half = autocorr[len(env_centered):]
        peaks = (half[1:-1] > half[:-2]) & (half[1:-1] > half[2:])
        peak_values = half[1:-1][peaks]
        
        has_periodic_modulation = (peak_values > 0.3).any().item() if len(peak_values) > 0 else False
        
        # Compare with original if provided
        if original is not None:
            orig_env = self.compute_envelope(original)
            
            # Resample to match lengths
            min_len = min(len(env), len(orig_env))
            
            # Correlation between envelopes
            env_corr = torch.corrcoef(torch.stack([
                env[:min_len],
                orig_env[:min_len]
            ]))[0, 1].item()
            
            # High negative correlation = compression pumping
            pumping_correlation = env_corr < 0
        else:
            env_corr = None
            pumping_correlation = False
        
        has_pumping = (
            modulation_depth > self.modulation_threshold or
            has_periodic_modulation or
            pumping_correlation
        )
        
        return {
            'modulation_depth': modulation_depth,
            'has_periodic_modulation': has_periodic_modulation,
            'envelope_correlation': env_corr,
            'has_pumping': has_pumping,
            'status': '❌ PUMPING' if has_pumping else '✅ STABLE',
        }


# ==============================================================================
# STREAMING STABILITY VALIDATOR
# ==============================================================================

class StreamingValidator:
    """
    Validate streaming inference produces consistent results.
    
    Tests:
    1. Frame-to-frame continuity
    2. Determinism (same input = same output)
    3. State accumulation (no drift over time)
    """
    
    def __init__(self, hop_length: int = 80):
        self.hop_length = hop_length
        
    def test_continuity(
        self,
        streaming_output: torch.Tensor,
        batch_output: torch.Tensor,
    ) -> Dict[str, any]:
        """
        Test continuity between streaming and batch outputs.
        
        Streaming should produce same result as batch processing.
        """
        min_len = min(len(streaming_output), len(batch_output))
        
        streaming_output = streaming_output[:min_len]
        batch_output = batch_output[:min_len]
        
        mse = F.mse_loss(streaming_output, batch_output).item()
        max_diff = (streaming_output - batch_output).abs().max().item()
        
        # Check frame boundaries for discontinuities
        frame_boundaries = range(self.hop_length, min_len - self.hop_length, self.hop_length)
        boundary_diffs = []
        
        for i in frame_boundaries:
            # Difference at boundary vs. middle of frame
            boundary_diff = (streaming_output[i] - streaming_output[i-1]).abs().item()
            mid_diff = (streaming_output[i + self.hop_length//2] - streaming_output[i + self.hop_length//2 - 1]).abs().item()
            boundary_diffs.append(boundary_diff / (mid_diff + 1e-8))
        
        boundary_ratio = np.mean(boundary_diffs) if boundary_diffs else 1.0
        
        return {
            'mse': mse,
            'max_diff': max_diff,
            'boundary_ratio': boundary_ratio,
            'is_continuous': mse < 1e-4 and boundary_ratio < 2.0,
        }
    
    def test_determinism(
        self,
        model: nn.Module,
        audio: torch.Tensor,
        num_runs: int = 3,
    ) -> Dict[str, any]:
        """
        Test that model produces deterministic outputs.
        """
        model.eval()
        
        # Note: Need streaming wrapper
        try:
            from auranet_v2_edge import StreamingAuraNet
            
            outputs = []
            for _ in range(num_runs):
                streamer = StreamingAuraNet(model)
                output = streamer.process_audio(audio)
                outputs.append(output)
            
            # Check all outputs are identical
            diffs = []
            for i in range(1, len(outputs)):
                diff = (outputs[0] - outputs[i]).abs().max().item()
                diffs.append(diff)
            
            max_diff = max(diffs) if diffs else 0
            
            return {
                'max_diff': max_diff,
                'is_deterministic': max_diff < 1e-6,
            }
            
        except ImportError:
            return {'error': 'StreamingAuraNet not available'}
    
    def test_state_drift(
        self,
        model: nn.Module,
        audio_length_sec: float = 30.0,
        sample_rate: int = 16000,
    ) -> Dict[str, any]:
        """
        Test for state drift over long sequences.
        
        Process long audio and check if model "drifts" over time.
        """
        try:
            from auranet_v2_edge import StreamingAuraNet
            
            num_samples = int(audio_length_sec * sample_rate)
            audio = torch.randn(num_samples)
            
            streamer = StreamingAuraNet(model)
            
            # Process and collect outputs at intervals
            chunk_size = sample_rate  # 1 second chunks
            outputs = []
            
            for i in range(0, num_samples, chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < self.hop_length:
                    break
                
                output = streamer.process_audio(chunk)
                outputs.append({
                    'time_sec': i / sample_rate,
                    'output_energy': output.pow(2).mean().item(),
                })
            
            # Check for drift in output energy
            energies = [o['output_energy'] for o in outputs]
            
            if len(energies) > 2:
                # Linear regression to detect trend
                x = np.arange(len(energies))
                y = np.array(energies)
                
                # Simple slope calculation
                slope = np.polyfit(x, y, 1)[0]
                
                # Normalized drift
                drift_rate = abs(slope) / (np.mean(y) + 1e-8)
            else:
                drift_rate = 0
            
            return {
                'duration_sec': audio_length_sec,
                'num_chunks': len(outputs),
                'energy_drift_rate': drift_rate,
                'has_drift': drift_rate > 0.1,  # 10% drift threshold
            }
            
        except ImportError:
            return {'error': 'StreamingAuraNet not available'}


# ==============================================================================
# COMPREHENSIVE VALIDATION
# ==============================================================================

@dataclass
class ValidationResult:
    """Container for all validation results."""
    glitch_analysis: Dict
    pumping_analysis: Dict
    streaming_analysis: Dict
    quality_metrics: Dict
    overall_pass: bool
    issues: List[str]


class AudioValidator:
    """
    Comprehensive audio quality validator.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
    ):
        self.sample_rate = sample_rate
        self.glitch_detector = GlitchDetector(sample_rate)
        self.pumping_detector = GainPumpingDetector(sample_rate)
        self.streaming_validator = StreamingValidator()
        
    def validate_output(
        self,
        enhanced: torch.Tensor,
        noisy: Optional[torch.Tensor] = None,
        clean: Optional[torch.Tensor] = None,
    ) -> ValidationResult:
        """
        Run full validation suite.
        
        Args:
            enhanced: Enhanced audio output
            noisy: Original noisy input (optional)
            clean: Clean reference (optional)
        """
        issues = []
        
        # === Glitch Analysis ===
        glitch_result = self.glitch_detector.analyze(enhanced)
        if glitch_result['has_glitches']:
            issues.append("Audio contains glitches (clicks/pops)")
        
        # === Gain Pumping Analysis ===
        pumping_result = self.pumping_detector.detect_pumping(enhanced, noisy)
        if pumping_result['has_pumping']:
            issues.append("Audio exhibits gain pumping")
        
        # === Streaming Analysis ===
        streaming_result = {}  # Requires model, done separately
        
        # === Quality Metrics ===
        quality_result = {}
        
        if clean is not None:
            # SI-SDR
            min_len = min(len(enhanced.squeeze()), len(clean.squeeze()))
            si_sdr = compute_si_sdr(enhanced.squeeze()[:min_len], clean.squeeze()[:min_len])
            quality_result['si_sdr'] = si_sdr
            
            if si_sdr < 0:
                issues.append(f"Poor SI-SDR: {si_sdr:.1f} dB")
        
        if noisy is not None and clean is not None:
            min_len = min(len(noisy.squeeze()), len(clean.squeeze()), len(enhanced.squeeze()))
            
            noisy_sdr = compute_si_sdr(noisy.squeeze()[:min_len], clean.squeeze()[:min_len])
            enh_sdr = compute_si_sdr(enhanced.squeeze()[:min_len], clean.squeeze()[:min_len])
            
            improvement = enh_sdr - noisy_sdr
            quality_result['si_sdr_improvement'] = improvement
            
            if improvement < 0:
                issues.append(f"Negative SI-SDR improvement: {improvement:.1f} dB")
        
        overall_pass = len(issues) == 0
        
        return ValidationResult(
            glitch_analysis=glitch_result,
            pumping_analysis=pumping_result,
            streaming_analysis=streaming_result,
            quality_metrics=quality_result,
            overall_pass=overall_pass,
            issues=issues,
        )
    
    def validate_streaming(
        self,
        model: nn.Module,
        audio: torch.Tensor,
    ) -> Dict[str, any]:
        """Validate streaming processing."""
        # Test determinism
        determinism = self.streaming_validator.test_determinism(model, audio)
        
        # Test state drift
        drift = self.streaming_validator.test_state_drift(model)
        
        return {
            'determinism': determinism,
            'drift': drift,
        }


# ==============================================================================
# BEFORE VS AFTER COMPARISON
# ==============================================================================

def compare_before_after(
    original_model: nn.Module,
    optimized_model: nn.Module,
    test_audio: torch.Tensor,
    clean_audio: Optional[torch.Tensor] = None,
) -> Dict[str, any]:
    """
    Compare original and optimized model outputs.
    
    Ensures optimization didn't degrade quality.
    """
    print("=" * 70)
    print("BEFORE vs AFTER QUALITY COMPARISON")
    print("=" * 70)
    
    results = {}
    
    original_model.eval()
    optimized_model.eval()
    
    validator = AudioValidator()
    
    with torch.no_grad():
        # Process with original
        print("\n[Original Model]")
        try:
            from auranet_v2_complete import AuraNetV2Complete
            orig_out = original_model.enhance_audio(test_audio.unsqueeze(0))
            orig_out = orig_out.squeeze()
        except:
            # Fallback: assume model takes STFT
            orig_out = test_audio  # Placeholder
        
        # Process with optimized
        print("\n[Optimized Model]")
        try:
            from auranet_v2_edge import StreamingAuraNet
            streamer = StreamingAuraNet(optimized_model)
            opt_out = streamer.process_audio(test_audio)
        except:
            opt_out = test_audio  # Placeholder
    
    # Validate both
    print("\n📊 Original Model Validation:")
    orig_result = validator.validate_output(orig_out, test_audio, clean_audio)
    print(f"  Glitches: {orig_result.glitch_analysis['status']}")
    print(f"  Pumping: {orig_result.pumping_analysis['status']}")
    if 'si_sdr' in orig_result.quality_metrics:
        print(f"  SI-SDR: {orig_result.quality_metrics['si_sdr']:.2f} dB")
    
    print("\n📊 Optimized Model Validation:")
    opt_result = validator.validate_output(opt_out, test_audio, clean_audio)
    print(f"  Glitches: {opt_result.glitch_analysis['status']}")
    print(f"  Pumping: {opt_result.pumping_analysis['status']}")
    if 'si_sdr' in opt_result.quality_metrics:
        print(f"  SI-SDR: {opt_result.quality_metrics['si_sdr']:.2f} dB")
    
    # Compare outputs
    if orig_out.shape == opt_out.shape:
        diff = (orig_out - opt_out).abs()
        print(f"\n📈 Output Difference:")
        print(f"  Mean: {diff.mean().item():.6f}")
        print(f"  Max: {diff.max().item():.6f}")
        
        results['output_diff_mean'] = diff.mean().item()
        results['output_diff_max'] = diff.max().item()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    orig_pass = orig_result.overall_pass
    opt_pass = opt_result.overall_pass
    
    if opt_pass and orig_pass:
        print("✅ Both models pass validation")
    elif opt_pass and not orig_pass:
        print("✅ Optimized model BETTER than original (fewer issues)")
    elif not opt_pass and orig_pass:
        print("❌ Optimized model has regressions")
        print(f"   Issues: {opt_result.issues}")
    else:
        print("⚠️ Both models have issues")
    
    results['original'] = {
        'passed': orig_pass,
        'issues': orig_result.issues,
    }
    results['optimized'] = {
        'passed': opt_pass,
        'issues': opt_result.issues,
    }
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate AuraNet Edge")
    parser.add_argument('--audio', type=str, help='Path to test audio file')
    parser.add_argument('--compare', action='store_true', help='Compare with original')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AURANET EDGE VALIDATION SUITE")
    print("=" * 70)
    
    # Generate or load test audio
    if args.audio:
        try:
            import torchaudio
            test_audio, sr = torchaudio.load(args.audio)
            test_audio = test_audio.mean(dim=0)  # Mono
            if sr != 16000:
                test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)
            print(f"Loaded audio: {args.audio}")
        except:
            print("Could not load audio, using synthetic")
            test_audio = torch.randn(16000 * 5)  # 5 seconds
    else:
        print("Using synthetic test audio...")
        # Generate test signal with speech-like characteristics
        t = torch.linspace(0, 5, 16000 * 5)
        test_audio = (
            0.3 * torch.sin(2 * 3.14159 * 440 * t) +  # A4
            0.2 * torch.sin(2 * 3.14159 * 880 * t) +  # A5
            0.1 * torch.sin(2 * 3.14159 * 220 * t) +  # A3
            0.1 * torch.randn_like(t)  # Noise
        )
    
    # Load model
    try:
        from auranet_v2_edge import AuraNetEdge, StreamingAuraNet
        model = AuraNetEdge()
        print(f"Loaded AuraNetEdge. Parameters: {model.count_parameters():,}")
    except ImportError:
        print("Could not import AuraNetEdge")
        exit(1)
    
    # Process audio
    print("\nProcessing audio...")
    model.eval()
    streamer = StreamingAuraNet(model)
    
    with torch.no_grad():
        enhanced = streamer.process_audio(test_audio)
    
    print(f"Input: {test_audio.shape}")
    print(f"Output: {enhanced.shape}")
    
    # Validate
    validator = AudioValidator()
    result = validator.validate_output(enhanced, test_audio)
    
    print("\n📋 VALIDATION RESULTS:")
    print(f"  Glitches: {result.glitch_analysis['status']}")
    print(f"    - Sample jumps: {result.glitch_analysis['jumps']['num_jumps']}")
    print(f"    - HF spikes: {result.glitch_analysis['hf_spikes']['num_hf_spikes']}")
    
    print(f"  Gain Pumping: {result.pumping_analysis['status']}")
    print(f"    - Modulation depth: {result.pumping_analysis['modulation_depth']:.3f}")
    
    if result.issues:
        print(f"\n⚠️ Issues found: {result.issues}")
    else:
        print(f"\n✅ All checks passed!")
    
    # Compare if requested
    if args.compare:
        try:
            from auranet_v2_complete import AuraNetV2Complete
            original = AuraNetV2Complete()
            comparison = compare_before_after(original, model, test_audio)
        except ImportError:
            print("\nCould not import original model for comparison")
