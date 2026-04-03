#!/usr/bin/env python3
# =============================================================================
# AuraNet V3 Inference — File & Streaming
# =============================================================================
#
# Modes:
# 1. File-based: Load entire file, process, save
# 2. Streaming: Frame-by-frame with persistent GRU state
#
# The streaming mode is critical for real-time deployment.
# GRU hidden state is maintained across frames for continuity.
# =============================================================================

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
import soundfile as sf
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_v3 import AuraNetV3, create_auranet_v3
from utils.stft import CausalSTFT


class AuraNetV3Inference:
    """Inference engine for AuraNet V3 (file-based and streaming)."""

    def __init__(self, model_path, config=None, device=None):
        # Device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load config
        if config is None:
            config = self._find_config(model_path)
        self.config = config
        self.sample_rate = config.get("audio", {}).get("sample_rate", 16000)

        # STFT
        stft_cfg = config.get("stft", {})
        self.stft = CausalSTFT(
            n_fft=stft_cfg.get("n_fft", 256),
            hop_length=stft_cfg.get("hop_size", 80),
            win_length=stft_cfg.get("window_size", 160),
        ).to(self.device)
        self.hop_length = stft_cfg.get("hop_size", 80)
        self.win_length = stft_cfg.get("window_size", 160)
        self.n_fft = stft_cfg.get("n_fft", 256)

        # Model
        self.model = create_auranet_v3(config).to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        print(f"Loaded V3 model from {model_path} on {self.device}")

        # Streaming state
        self._hidden = None

    def _find_config(self, model_path):
        """Find config.yaml near the model or in project root."""
        candidates = [
            Path(model_path).parent.parent / "config_v3.yaml",
            Path(model_path).parent.parent / "config.yaml",
            Path(__file__).parent / "config_v3.yaml",
            Path(__file__).parent / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    return yaml.safe_load(f) or {}
        return {}

    # ------------------------------------------------------------------
    # File-based inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def process_file(self, input_path, output_path=None):
        """
        Process a complete audio file.

        Args:
            input_path: Path to noisy .wav
            output_path: Optional save path

        Returns:
            enhanced audio tensor [N]
        """
        # Load
        audio_data, sr = sf.read(str(input_path), dtype='float32')
        audio = torch.from_numpy(audio_data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != self.sample_rate:
            import torchaudio
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio.unsqueeze(0)).squeeze(0)

        audio = audio.unsqueeze(0).to(self.device)  # [1, N]
        orig_len = audio.shape[-1]

        # STFT
        noisy_stft = self.stft(audio)  # [1, 2, T, F]

        # Forward
        t0 = time.time()
        enhanced_stft, _, _ = self.model(noisy_stft)
        elapsed = time.time() - t0

        # iSTFT
        enhanced = self.stft.inverse(enhanced_stft, length=orig_len)
        enhanced = enhanced.squeeze().cpu()

        # Normalize to prevent clipping
        peak = enhanced.abs().max()
        if peak > 0.99:
            enhanced = enhanced / peak * 0.95

        print(f"  Processed {orig_len/self.sample_rate:.2f}s in {elapsed*1000:.1f}ms "
              f"({orig_len/self.sample_rate/elapsed:.1f}x real-time)")

        # Save
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(str(output_path), enhanced.numpy(), self.sample_rate)
            print(f"  Saved: {output_path}")

        return enhanced

    # ------------------------------------------------------------------
    # Streaming inference (frame-by-frame)
    # ------------------------------------------------------------------
    def reset_stream(self):
        """Reset streaming state for a new audio stream."""
        self._hidden = None
        self._input_buffer = torch.zeros(1, self.n_fft - self.hop_length,
                                         device=self.device)
        self._output_buffer = torch.zeros(1, self.win_length,
                                          device=self.device)

    @torch.no_grad()
    def process_chunk(self, chunk):
        """
        Process a single audio chunk (frame-by-frame streaming).

        Args:
            chunk: [hop_length] audio samples (numpy or tensor)

        Returns:
            [hop_length] enhanced audio samples (numpy)
        """
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk).float()
        chunk = chunk.to(self.device)
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)  # [1, hop_length]

        # Concatenate with buffer to form a full frame
        frame = torch.cat([self._input_buffer, chunk], dim=-1)  # [1, win_length]

        # Update buffer (keep last n_fft-hop samples)
        self._input_buffer = frame[:, self.hop_length:]

        # STFT of single frame
        noisy_stft = self.stft(frame)  # [1, 2, 1, F]

        # Model forward with persistent hidden state
        enhanced_stft, self._hidden, _ = self.model(noisy_stft, self._hidden)

        # iSTFT
        enhanced_frame = self.stft.inverse(enhanced_stft)
        enhanced_frame = enhanced_frame.squeeze()

        # Return hop_length samples
        output = enhanced_frame[:self.hop_length] if len(enhanced_frame) >= self.hop_length \
            else F.pad(enhanced_frame, (0, self.hop_length - len(enhanced_frame)))

        return output.cpu().numpy()

    @torch.no_grad()
    def process_streaming(self, input_path, output_path=None):
        """
        Simulate streaming inference on a file (for benchmarking).

        Processes chunk-by-chunk and measures latency.
        """
        audio_data, sr = sf.read(str(input_path), dtype='float32')
        if sr != self.sample_rate:
            import torchaudio
            audio_t = torch.from_numpy(audio_data).float()
            audio_t = torchaudio.transforms.Resample(sr, self.sample_rate)(audio_t.unsqueeze(0)).squeeze(0)
            audio_data = audio_t.numpy()

        self.reset_stream()
        output_chunks = []
        latencies = []
        n_chunks = len(audio_data) // self.hop_length

        for i in range(n_chunks):
            chunk = audio_data[i * self.hop_length:(i + 1) * self.hop_length]
            t0 = time.time()
            enhanced_chunk = self.process_chunk(chunk)
            latencies.append(time.time() - t0)
            output_chunks.append(enhanced_chunk)

        enhanced = np.concatenate(output_chunks)

        # Stats
        avg_latency = np.mean(latencies) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000
        chunk_ms = self.hop_length / self.sample_rate * 1000
        print(f"\n  Streaming stats:")
        print(f"    Chunks: {n_chunks}")
        print(f"    Chunk size: {self.hop_length} samples ({chunk_ms:.1f}ms)")
        print(f"    Avg latency: {avg_latency:.2f}ms")
        print(f"    P99 latency: {p99_latency:.2f}ms")
        print(f"    Real-time: {'✅ YES' if avg_latency < chunk_ms else '❌ NO'}")

        if output_path:
            sf.write(str(output_path), enhanced, self.sample_rate)
            print(f"    Saved: {output_path}")

        return enhanced, {"avg_ms": avg_latency, "p99_ms": p99_latency}

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------
    @torch.no_grad()
    def benchmark(self, num_frames=100):
        """Benchmark single-frame inference latency."""
        self.reset_stream()
        chunk = torch.randn(self.hop_length).numpy()

        # Warmup
        for _ in range(10):
            self.process_chunk(chunk)

        latencies = []
        for _ in range(num_frames):
            t0 = time.time()
            self.process_chunk(chunk)
            latencies.append(time.time() - t0)

        avg = np.mean(latencies) * 1000
        p50 = np.median(latencies) * 1000
        p99 = np.percentile(latencies, 99) * 1000
        chunk_ms = self.hop_length / self.sample_rate * 1000

        print(f"\n  Latency Benchmark ({num_frames} frames):")
        print(f"    Chunk: {self.hop_length} samples ({chunk_ms:.1f}ms)")
        print(f"    Avg:   {avg:.2f}ms")
        print(f"    P50:   {p50:.2f}ms")
        print(f"    P99:   {p99:.2f}ms")
        print(f"    Budget: {chunk_ms:.1f}ms → {'✅ OK' if avg < chunk_ms else '❌ OVER'}")
        return {"avg_ms": avg, "p50_ms": p50, "p99_ms": p99, "budget_ms": chunk_ms}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AuraNet V3 Inference")
    parser.add_argument("input", help="Input .wav file")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--model", default="checkpoints/best_model_v3.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming (frame-by-frame) mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark")
    args = parser.parse_args()

    output = args.output or f"outputs/{Path(args.input).stem}_v3_enhanced.wav"
    engine = AuraNetV3Inference(args.model, device=args.device)

    if args.benchmark:
        engine.benchmark()
    elif args.streaming:
        engine.process_streaming(args.input, output)
    else:
        engine.process_file(args.input, output)


if __name__ == "__main__":
    main()
