#!/usr/bin/env python3
# =============================================================================
# AuraNet V3 — Complete Inference & Quality Evaluation
# =============================================================================
#
# PURPOSE: Validate trained model produces natural, clean, perceptually
#          improved audio in real-time
#
# TESTS:
#   1. Load model checkpoint
#   2. Process noisy audio (file-based)
#   3. Evaluate PESQ/STOI/SI-SNR
#   4. Measure real-time performance
#   5. Streaming simulation (frame-by-frame)
#   6. Quality analysis
#
# USAGE:
#   python evaluate_model.py --model /path/to/model.pt
#   python evaluate_model.py --model /path/to/model.pt --noisy test.wav --clean ref.wav
#
# =============================================================================

print("=" * 70)
print("🎧 AuraNet V3 — Inference & Quality Evaluation")
print("=" * 70)

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import yaml

try:
    import soundfile as sf
except ImportError:
    print("❌ soundfile not installed. Run: pip install soundfile")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_v3 import AuraNetV3, create_auranet_v3
from utils.stft import CausalSTFT
from metrics import compute_pesq, compute_stoi


# =============================================================================
# Evaluation Engine
# =============================================================================

class AuraNetEvaluator:
    """Complete evaluation pipeline for AuraNet V3."""

    def __init__(self, model_path: str, device: str = None):
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"\n📍 Device: {self.device}")

        # Load config
        self.config = self._find_config(model_path)
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 16000)

        # STFT config (MUST match training)
        stft_cfg = self.config.get("stft", {})
        self.n_fft = stft_cfg.get("n_fft", 256)
        self.hop_length = stft_cfg.get("hop_size", 80)
        self.win_length = stft_cfg.get("window_size", 160)

        print(f"\n📐 STFT Config (matching training):")
        print(f"   n_fft:      {self.n_fft}")
        print(f"   hop_length: {self.hop_length} ({1000*self.hop_length/self.sample_rate:.1f}ms)")
        print(f"   win_length: {self.win_length} ({1000*self.win_length/self.sample_rate:.1f}ms)")

        # Create STFT module
        self.stft = CausalSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        ).to(self.device)

        # Load model
        self.model = self._load_model(model_path)

        # Streaming state
        self._hidden = None
        self._input_buffer = None

    def _find_config(self, model_path: str) -> dict:
        """Find config file near model or in project root."""
        candidates = [
            Path(model_path).parent / "config_v3.yaml",
            Path(model_path).parent.parent / "config_v3.yaml",
            Path(__file__).parent / "config_v3.yaml",
            Path(__file__).parent / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    print(f"   Config: {p}")
                    return yaml.safe_load(f) or {}
        print("   ⚠️  No config found, using defaults")
        return {}

    def _load_model(self, model_path: str) -> AuraNetV3:
        """Load model checkpoint."""
        print(f"\n📦 Loading model: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create model
        model = create_auranet_v3(self.config).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                epoch = checkpoint.get("epoch", "?")
                print(f"   Loaded state_dict from epoch {epoch}")
            elif "ema_state" in checkpoint:
                # Try EMA weights first (usually better)
                model.load_state_dict(checkpoint["ema_state"])
                print(f"   Loaded EMA state_dict")
            else:
                model.load_state_dict(checkpoint)
                print(f"   Loaded direct state_dict")
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Model stats
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = total_params * 4 / 1024 / 1024
        print(f"   Parameters: {total_params:,} ({size_mb:.2f} MB)")

        return model

    # =========================================================================
    # File-Based Inference
    # =========================================================================

    @torch.no_grad()
    def process_file(self, input_path: str, output_path: str = None) -> Tuple[np.ndarray, float]:
        """
        Process complete audio file.

        Args:
            input_path: Path to noisy WAV
            output_path: Optional output path

        Returns:
            (enhanced_audio, processing_time_ms)
        """
        print(f"\n🔊 Processing: {input_path}")

        # Load audio
        audio, sr = sf.read(input_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Mono

        # Resample if needed
        if sr != self.sample_rate:
            import torchaudio
            audio = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            audio = audio.squeeze(0).numpy()
            print(f"   Resampled: {sr} → {self.sample_rate} Hz")

        duration = len(audio) / self.sample_rate
        print(f"   Duration: {duration:.2f}s ({len(audio):,} samples)")

        # Normalize input
        input_peak = np.abs(audio).max()
        if input_peak > 0:
            audio = audio / input_peak

        # To tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        orig_len = audio_tensor.shape[-1]

        # STFT
        noisy_stft = self.stft(audio_tensor)  # [1, 2, T, F]

        # Forward pass
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        t0 = time.perf_counter()

        enhanced_stft, _, _ = self.model(noisy_stft)

        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Apply tanh activation (NOT clamp - preserves gradients)
        # Note: For inference this is optional since model output is bounded by mask

        # iSTFT
        enhanced = self.stft.inverse(enhanced_stft, length=orig_len)
        enhanced = enhanced.squeeze().cpu().numpy()

        # Restore original scale
        enhanced = enhanced * input_peak

        # Prevent clipping
        peak = np.abs(enhanced).max()
        if peak > 0.99:
            enhanced = enhanced / peak * 0.95

        # Stats
        rtf = elapsed_ms / (duration * 1000)
        print(f"   ⏱️  Processing: {elapsed_ms:.1f}ms ({1/rtf:.1f}x real-time)")

        # Save
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, enhanced, self.sample_rate)
            print(f"   💾 Saved: {output_path}")

        return enhanced, elapsed_ms

    # =========================================================================
    # Streaming Inference (Frame-by-Frame)
    # =========================================================================

    def reset_stream(self):
        """Reset streaming state."""
        self._hidden = None
        self._input_buffer = torch.zeros(1, self.n_fft - self.hop_length, device=self.device)

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process single audio chunk (streaming mode).

        Args:
            chunk: [hop_length] audio samples

        Returns:
            [hop_length] enhanced samples
        """
        chunk = torch.from_numpy(chunk).float().to(self.device)
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)

        # Build frame from buffer
        frame = torch.cat([self._input_buffer, chunk], dim=-1)
        self._input_buffer = frame[:, self.hop_length:]

        # STFT single frame
        noisy_stft = self.stft(frame)

        # Forward with persistent hidden state
        enhanced_stft, self._hidden, _ = self.model(noisy_stft, self._hidden)

        # iSTFT
        enhanced_frame = self.stft.inverse(enhanced_stft).squeeze()

        # Return hop_length samples
        if len(enhanced_frame) >= self.hop_length:
            output = enhanced_frame[:self.hop_length]
        else:
            output = F.pad(enhanced_frame, (0, self.hop_length - len(enhanced_frame)))

        return output.cpu().numpy()

    @torch.no_grad()
    def process_streaming(self, input_path: str, output_path: str = None) -> Tuple[np.ndarray, Dict]:
        """
        Simulate streaming inference (frame-by-frame).

        Returns:
            (enhanced_audio, timing_stats)
        """
        print(f"\n🌊 Streaming simulation: {input_path}")

        # Load audio
        audio, sr = sf.read(input_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            import torchaudio
            audio = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            audio = audio.squeeze(0).numpy()

        duration = len(audio) / self.sample_rate

        # Normalize
        input_peak = np.abs(audio).max()
        if input_peak > 0:
            audio = audio / input_peak

        # Process frame-by-frame
        self.reset_stream()

        frame_times = []
        output_chunks = []
        n_frames = len(audio) // self.hop_length

        for i in range(n_frames):
            start_idx = i * self.hop_length
            chunk = audio[start_idx:start_idx + self.hop_length]

            if len(chunk) < self.hop_length:
                chunk = np.pad(chunk, (0, self.hop_length - len(chunk)))

            t0 = time.perf_counter()
            enhanced_chunk = self.process_chunk(chunk)
            frame_times.append((time.perf_counter() - t0) * 1000)

            output_chunks.append(enhanced_chunk)

        enhanced = np.concatenate(output_chunks)
        enhanced = enhanced * input_peak

        # Prevent clipping
        peak = np.abs(enhanced).max()
        if peak > 0.99:
            enhanced = enhanced / peak * 0.95

        # Timing stats
        frame_times = np.array(frame_times)
        frame_duration_ms = 1000 * self.hop_length / self.sample_rate

        stats = {
            "n_frames": n_frames,
            "frame_duration_ms": frame_duration_ms,
            "mean_latency_ms": frame_times.mean(),
            "max_latency_ms": frame_times.max(),
            "p95_latency_ms": np.percentile(frame_times, 95),
            "p99_latency_ms": np.percentile(frame_times, 99),
            "real_time_factor": frame_times.mean() / frame_duration_ms,
        }

        print(f"   Frames: {n_frames}")
        print(f"   Frame duration: {frame_duration_ms:.2f}ms (hop={self.hop_length})")
        print(f"   Mean latency:   {stats['mean_latency_ms']:.3f}ms")
        print(f"   Max latency:    {stats['max_latency_ms']:.3f}ms")
        print(f"   P95 latency:    {stats['p95_latency_ms']:.3f}ms")
        print(f"   RTF:            {stats['real_time_factor']:.4f}")

        if stats['mean_latency_ms'] < frame_duration_ms:
            print(f"   ✅ REAL-TIME CAPABLE (latency < frame duration)")
        else:
            print(f"   ⚠️  NOT real-time (latency > frame duration)")

        # Save
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, enhanced, self.sample_rate)
            print(f"   💾 Saved: {output_path}")

        return enhanced, stats

    # =========================================================================
    # Quality Evaluation
    # =========================================================================

    def evaluate(self, enhanced: np.ndarray, clean: np.ndarray, noisy: np.ndarray = None) -> Dict:
        """
        Compute quality metrics.

        Args:
            enhanced: Enhanced audio
            clean: Clean reference
            noisy: Optional noisy input (for improvement calculation)

        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Computing quality metrics...")

        # Align lengths
        min_len = min(len(enhanced), len(clean))
        enhanced = enhanced[:min_len]
        clean = clean[:min_len]

        metrics = {}

        # PESQ
        pesq_score = compute_pesq(enhanced, clean, self.sample_rate, mode='wb')
        if pesq_score > 0:
            metrics['PESQ'] = pesq_score
            print(f"   PESQ:  {pesq_score:.3f} (range: 1.0-4.5)")
        else:
            print(f"   PESQ:  ❌ Not available (install pesq library)")

        # STOI
        stoi_score = compute_stoi(enhanced, clean, self.sample_rate, extended=True)
        if stoi_score > 0:
            metrics['STOI'] = stoi_score
            print(f"   STOI:  {stoi_score:.3f} (range: 0.0-1.0)")
        else:
            print(f"   STOI:  ❌ Not available (install pystoi library)")

        # SI-SNR
        sisnr = self._compute_sisnr(enhanced, clean)
        metrics['SI-SNR'] = sisnr
        print(f"   SI-SNR: {sisnr:.2f} dB")

        # If noisy input provided, compute improvements
        if noisy is not None:
            noisy = noisy[:min_len]

            noisy_pesq = compute_pesq(noisy, clean, self.sample_rate, mode='wb')
            noisy_stoi = compute_stoi(noisy, clean, self.sample_rate, extended=True)
            noisy_sisnr = self._compute_sisnr(noisy, clean)

            print(f"\n   📈 Improvement over noisy input:")
            if noisy_pesq > 0 and metrics.get('PESQ', -1) > 0:
                delta_pesq = metrics['PESQ'] - noisy_pesq
                print(f"      ΔPESQ:  {delta_pesq:+.3f} ({noisy_pesq:.3f} → {metrics['PESQ']:.3f})")
                metrics['ΔPESQ'] = delta_pesq

            if noisy_stoi > 0 and metrics.get('STOI', -1) > 0:
                delta_stoi = metrics['STOI'] - noisy_stoi
                print(f"      ΔSTOI:  {delta_stoi:+.3f} ({noisy_stoi:.3f} → {metrics['STOI']:.3f})")
                metrics['ΔSTOI'] = delta_stoi

            delta_sisnr = sisnr - noisy_sisnr
            print(f"      ΔSI-SNR: {delta_sisnr:+.2f} dB ({noisy_sisnr:.2f} → {sisnr:.2f})")
            metrics['ΔSI-SNR'] = delta_sisnr

        return metrics

    def _compute_sisnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Scale-Invariant SNR."""
        pred = pred - pred.mean()
        target = target - target.mean()

        # s_target = <pred, target> / ||target||^2 * target
        dot = np.sum(pred * target)
        s_target_energy = np.sum(target ** 2) + 1e-8
        s_target = (dot / s_target_energy) * target

        # e_noise = pred - s_target
        e_noise = pred - s_target

        # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        signal_energy = np.sum(s_target ** 2) + 1e-8
        noise_energy = np.sum(e_noise ** 2) + 1e-8

        sisnr = 10 * np.log10(signal_energy / noise_energy)
        return float(sisnr)

    # =========================================================================
    # Audio Analysis
    # =========================================================================

    def analyze_audio(self, enhanced: np.ndarray, clean: np.ndarray = None) -> Dict:
        """
        Analyze audio characteristics.

        Returns signal statistics for quality assessment.
        """
        print(f"\n🔬 Audio Analysis:")

        analysis = {}

        # Basic stats
        analysis['peak'] = float(np.abs(enhanced).max())
        analysis['rms'] = float(np.sqrt(np.mean(enhanced ** 2)))
        analysis['dynamic_range_db'] = float(20 * np.log10(analysis['peak'] / (analysis['rms'] + 1e-8)))

        print(f"   Peak amplitude:  {analysis['peak']:.4f}")
        print(f"   RMS:             {analysis['rms']:.4f}")
        print(f"   Dynamic range:   {analysis['dynamic_range_db']:.1f} dB")

        # Check for clipping
        n_clipped = np.sum(np.abs(enhanced) > 0.99)
        analysis['clipped_samples'] = int(n_clipped)
        if n_clipped > 0:
            print(f"   ⚠️  Clipped samples: {n_clipped}")
        else:
            print(f"   ✅ No clipping detected")

        # Spectral analysis
        if len(enhanced) >= 1024:
            spectrum = np.abs(np.fft.rfft(enhanced[:8192]))
            freqs = np.fft.rfftfreq(8192, 1/self.sample_rate)

            # Find spectral centroid
            centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)
            analysis['spectral_centroid_hz'] = float(centroid)
            print(f"   Spectral centroid: {centroid:.0f} Hz")

        # Compare with clean if available
        if clean is not None:
            clean = clean[:len(enhanced)]

            # Correlation
            corr = np.corrcoef(enhanced, clean)[0, 1]
            analysis['correlation'] = float(corr)
            print(f"   Correlation:     {corr:.4f}")

            # MSE
            mse = np.mean((enhanced - clean) ** 2)
            analysis['mse'] = float(mse)
            print(f"   MSE:             {mse:.6f}")

        return analysis


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AuraNet V3 Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--noisy", type=str, default=None, help="Path to noisy audio")
    parser.add_argument("--clean", type=str, default=None, help="Path to clean reference")
    parser.add_argument("--output", type=str, default="outputs/enhanced.wav", help="Output path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--streaming", action="store_true", help="Test streaming mode")
    parser.add_argument("--checkpoint-info", action="store_true", help="Show checkpoint training metrics")
    args = parser.parse_args()

    # Show checkpoint info if requested
    if args.checkpoint_info:
        import torch
        ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
        print("\n📋 CHECKPOINT TRAINING METRICS")
        print("=" * 50)
        print(f"   Epoch:     {ckpt.get('epoch', 'N/A')}")
        print(f"   Val Loss:  {ckpt.get('val_loss', 'N/A')}")
        print(f"   Val PESQ:  {ckpt.get('val_pesq', 'N/A')}")
        print(f"   Val STOI:  {ckpt.get('val_stoi', 'N/A')}")
        print(f"   Val SI-SNR: {ckpt.get('val_si_snr', 'N/A')} dB")
        print("=" * 50)
        if args.noisy is None:
            return

    # Initialize evaluator
    evaluator = AuraNetEvaluator(args.model, device=args.device)

    # Use default test files if not specified
    test_dir = Path(__file__).parent / "test_audio"
    noisy_path = args.noisy or str(test_dir / "noisy_sample.wav")
    clean_path = args.clean or str(test_dir / "clean_reference.wav")

    if not os.path.exists(noisy_path):
        print(f"\n❌ Noisy audio not found: {noisy_path}")
        print("   Provide --noisy argument or add test_audio/noisy_sample.wav")
        return

    # Process file
    enhanced, process_time = evaluator.process_file(noisy_path, args.output)

    # Streaming test
    if args.streaming:
        stream_output = args.output.replace(".wav", "_streaming.wav")
        enhanced_stream, stream_stats = evaluator.process_streaming(noisy_path, stream_output)

    # Evaluate if clean reference available
    if os.path.exists(clean_path):
        clean, _ = sf.read(clean_path, dtype='float32')
        if clean.ndim > 1:
            clean = clean.mean(axis=1)

        # Load noisy for comparison
        noisy, _ = sf.read(noisy_path, dtype='float32')
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)

        metrics = evaluator.evaluate(enhanced, clean, noisy)
        analysis = evaluator.analyze_audio(enhanced, clean)
    else:
        print(f"\n⚠️  Clean reference not found: {clean_path}")
        print("   Skipping PESQ/STOI evaluation")
        analysis = evaluator.analyze_audio(enhanced)

    # Summary
    print("\n" + "=" * 70)
    print("📋 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"   Model:   {args.model}")
    print(f"   Input:   {noisy_path}")
    print(f"   Output:  {args.output}")

    if 'metrics' in dir() and metrics:
        print(f"\n   Quality Metrics:")
        if 'PESQ' in metrics:
            status = "✅" if metrics['PESQ'] >= 2.8 else "⚠️"
            print(f"      {status} PESQ:   {metrics['PESQ']:.3f} (target ≥2.8)")
        if 'STOI' in metrics:
            status = "✅" if metrics['STOI'] >= 0.86 else "⚠️"
            print(f"      {status} STOI:   {metrics['STOI']:.3f} (target ≥0.86)")
        if 'SI-SNR' in metrics:
            status = "✅" if metrics['SI-SNR'] >= 15 else "⚠️"
            print(f"      {status} SI-SNR: {metrics['SI-SNR']:.2f} dB (target ≥15)")

    if args.streaming and 'stream_stats' in dir():
        print(f"\n   Streaming Performance:")
        status = "✅" if stream_stats['mean_latency_ms'] < 10 else "⚠️"
        print(f"      {status} Mean latency: {stream_stats['mean_latency_ms']:.3f}ms (target <10ms)")
        print(f"      RTF: {stream_stats['real_time_factor']:.4f}")

    print("\n" + "=" * 70)
    print("🎧 LISTEN to the output and check for:")
    print("   • Clarity — is speech clear and understandable?")
    print("   • Naturalness — does it sound like natural speech?")
    print("   • Artifacts — any musical noise, robotic sounds?")
    print("   • Harshness — any unpleasant high frequencies?")
    print("=" * 70)


if __name__ == "__main__":
    main()
