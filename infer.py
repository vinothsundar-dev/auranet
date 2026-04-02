# =============================================================================
# Inference Script for AuraNet
# =============================================================================
#
# INFERENCE MODES:
# 1. File-based: Process complete audio files
# 2. Streaming: Real-time chunk-by-chunk processing
# 3. Batch: Process multiple files
#
# REAL-TIME CONSIDERATIONS:
# - Stateful GRU: Hidden state maintained across chunks
# - Overlap-add: Smooth reconstruction between chunks
# - Minimal latency: ~5ms algorithmic latency per frame
# =============================================================================

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from model import AuraNet, create_auranet, apply_wdrc
from utils.stft import CausalSTFT, StreamingSTFT
from utils.audio_utils import load_audio, save_audio, normalize_audio


class AuraNetInference:
    """
    AuraNet inference wrapper for both file and streaming modes.

    Handles:
    - Model loading and device management
    - Stateful streaming with GRU hidden state
    - Overlap-add reconstruction
    - Neural-WDRC application
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None,
        device: str = "auto",
        apply_wdrc_gain: bool = True,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            config: Optional configuration dictionary
            device: Device to run on ("auto", "cuda", "cpu", "mps")
            apply_wdrc_gain: Whether to apply Neural-WDRC gain
        """
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model
        if model_path and os.path.exists(model_path):
            self.model, self.config = self._load_model(model_path)
        else:
            print("No model path provided. Using randomly initialized model.")
            self.model = create_auranet(config)
            self.config = config or {}

        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize STFT
        stft_config = self.config.get("stft", {})
        self.stft = CausalSTFT(
            n_fft=stft_config.get("n_fft", 256),
            hop_length=stft_config.get("hop_size", 80),
            win_length=stft_config.get("window_size", 160),
        ).to(self.device)

        self.apply_wdrc_gain = apply_wdrc_gain
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 16000)

        # Streaming state
        self._gru_hidden = None
        self._streaming_stft = None

    def _load_model(self, path: str) -> Tuple[AuraNet, Dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict) and "config" in checkpoint:
            config = checkpoint["config"]
        else:
            # Fallback: load config.yaml from project root
            config = {}
            config_candidates = [
                Path(path).parent.parent / "config.yaml",  # checkpoints/../config.yaml
                Path.cwd() / "config.yaml",
                Path(__file__).parent / "config.yaml",
            ]
            for cfg_path in config_candidates:
                if cfg_path.exists():
                    import yaml
                    with open(cfg_path) as f:
                        config = yaml.safe_load(f) or {}
                    print(f"Loaded config from: {cfg_path}")
                    break

        model = create_auranet(config)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print(f"Loaded model from {path}")
        return model, config

    @torch.no_grad()
    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        normalize_output: bool = True,
    ) -> torch.Tensor:
        """
        Process a complete audio file.

        Args:
            input_path: Path to input audio file
            output_path: Optional path to save enhanced audio
            normalize_output: Normalize output to prevent clipping

        Returns:
            Enhanced audio tensor [N]
        """
        print(f"\nProcessing: {input_path}")

        # Load audio
        audio, sr = load_audio(input_path, self.sample_rate)
        audio = audio.to(self.device)

        print(f"  Input: {audio.shape[-1]} samples ({audio.shape[-1]/sr:.2f}s)")

        # Time the inference
        start_time = time.time()

        # Ensure batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # [1, 1, N]

        # Compute STFT
        noisy_stft = self.stft(audio.squeeze(1))  # [B, 2, T, F]

        # Model forward pass
        enhanced_stft, wdrc_params, _ = self.model(noisy_stft)

        # Inverse STFT
        enhanced_audio = self.stft.inverse(enhanced_stft, length=audio.shape[-1])

        # Apply Neural-WDRC if enabled
        if self.apply_wdrc_gain:
            enhanced_audio = apply_wdrc(
                enhanced_audio,
                wdrc_params,
                hop_length=self.stft.hop_length,
            )

        inference_time = time.time() - start_time

        # Remove batch dimension
        enhanced_audio = enhanced_audio.squeeze(0)

        # Normalize to prevent clipping
        if normalize_output:
            max_val = enhanced_audio.abs().max()
            if max_val > 0.99:
                enhanced_audio = enhanced_audio / max_val * 0.95

        print(f"  Inference time: {inference_time*1000:.1f}ms")
        print(f"  Output: {enhanced_audio.shape[-1]} samples")

        # Save if path provided
        if output_path:
            save_audio(enhanced_audio.cpu(), output_path, self.sample_rate)
            print(f"  Saved to: {output_path}")

        return enhanced_audio.cpu()

    def reset_streaming_state(self):
        """Reset streaming state for new audio stream."""
        self._gru_hidden = None
        if self._streaming_stft is not None:
            self._streaming_stft.reset_state()

    @torch.no_grad()
    def process_chunk(
        self,
        chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process a single audio chunk for real-time streaming.

        STREAMING ARCHITECTURE:
        1. Maintain STFT buffer for continuous processing
        2. Maintain GRU hidden state across chunks
        3. Output enhanced audio with same latency

        Args:
            chunk: Audio chunk [N] or [1, N]

        Returns:
            Tuple of (enhanced_chunk, wdrc_params)
        """
        # Ensure proper shape
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)

        chunk = chunk.to(self.device)

        # Initialize streaming STFT if needed
        if self._streaming_stft is None:
            self._streaming_stft = StreamingSTFT(
                n_fft=self.stft.n_fft,
                hop_length=self.stft.hop_length,
                win_length=self.stft.win_length,
            ).to(self.device)

        # Process chunk through streaming STFT
        stft_frames, _ = self._streaming_stft.process_chunk(chunk)

        # Model forward with maintained hidden state
        enhanced_stft, wdrc_params, self._gru_hidden = self.model(
            stft_frames,
            hidden=self._gru_hidden,
        )

        # Inverse STFT
        enhanced_chunk = self.stft.inverse(enhanced_stft, length=chunk.shape[-1])

        # Apply WDRC
        if self.apply_wdrc_gain:
            enhanced_chunk = apply_wdrc(
                enhanced_chunk,
                wdrc_params,
                hop_length=self.stft.hop_length,
            )

        return enhanced_chunk.squeeze(0).cpu(), wdrc_params

    @torch.no_grad()
    def process_streaming(
        self,
        input_path: str,
        output_path: str,
        chunk_size_ms: int = 10,
    ):
        """
        Simulate streaming processing on a file.

        Processes audio in chunks to demonstrate real-time capability.

        Args:
            input_path: Input audio file
            output_path: Output audio file
            chunk_size_ms: Chunk size in milliseconds
        """
        print(f"\nStreaming processing: {input_path}")
        print(f"  Chunk size: {chunk_size_ms}ms")

        # Load audio
        audio, sr = load_audio(input_path, self.sample_rate)
        audio = audio.squeeze()  # [N]

        chunk_samples = int(chunk_size_ms * self.sample_rate / 1000)
        num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

        print(f"  Total chunks: {num_chunks}")

        # Reset streaming state
        self.reset_streaming_state()

        # Process chunks
        output_chunks = []
        processing_times = []

        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(audio))
            chunk = audio[start_idx:end_idx]

            # Pad if necessary
            if len(chunk) < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - len(chunk)))

            # Time the processing
            start_time = time.time()
            enhanced_chunk, _ = self.process_chunk(chunk)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            output_chunks.append(enhanced_chunk)

        # Concatenate output
        output_audio = torch.cat(output_chunks)[:len(audio)]

        # Statistics
        avg_time = sum(processing_times) / len(processing_times) * 1000
        chunk_duration = chunk_size_ms

        print(f"\n  Average processing time: {avg_time:.2f}ms per {chunk_duration}ms chunk")

        if avg_time < chunk_duration:
            print(f"  ✅ Real-time capable ({avg_time/chunk_duration*100:.1f}% of real-time)")
        else:
            print(f"  ⚠️ Slower than real-time ({avg_time/chunk_duration*100:.1f}% of real-time)")

        # Save output
        save_audio(output_audio, output_path, self.sample_rate)
        print(f"  Saved to: {output_path}")

    @torch.no_grad()
    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = [".wav", ".flac", ".mp3"],
    ):
        """
        Process all audio files in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: Audio file extensions to process
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        files = []
        for ext in extensions:
            files.extend(input_dir.glob(f"**/*{ext}"))

        print(f"\nBatch processing {len(files)} files...")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")

        for i, input_path in enumerate(files):
            # Create output path
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\n[{i+1}/{len(files)}]", end=" ")

            try:
                self.process_file(str(input_path), str(output_path))
            except Exception as e:
                print(f"  Error: {e}")

        print(f"\n✅ Batch processing complete!")

    def measure_latency(self) -> Dict[str, float]:
        """
        Measure inference latency.

        Returns dictionary with latency measurements.
        """
        print("\nMeasuring inference latency...")

        # Create test input
        test_duration_ms = 100
        test_samples = int(test_duration_ms * self.sample_rate / 1000)
        test_input = torch.randn(1, test_samples).to(self.device)

        # Warm up
        for _ in range(5):
            stft = self.stft(test_input)
            enhanced, _, _ = self.model(stft)
            _ = self.stft.inverse(enhanced)

        # Measure
        num_runs = 100
        times = []

        for _ in range(num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            stft = self.stft(test_input)
            enhanced, _, _ = self.model(stft)
            _ = self.stft.inverse(enhanced)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time = sum(times) / len(times) * 1000
        min_time = min(times) * 1000
        max_time = max(times) * 1000

        # Calculate effective latency
        # Algorithmic latency = STFT window size
        stft_latency_ms = self.stft.win_length / self.sample_rate * 1000
        total_latency = stft_latency_ms + avg_time

        results = {
            "stft_latency_ms": stft_latency_ms,
            "inference_avg_ms": avg_time,
            "inference_min_ms": min_time,
            "inference_max_ms": max_time,
            "total_latency_ms": total_latency,
            "input_duration_ms": test_duration_ms,
        }

        print(f"  STFT latency:     {stft_latency_ms:.2f}ms")
        print(f"  Inference (avg):  {avg_time:.2f}ms")
        print(f"  Inference (min):  {min_time:.2f}ms")
        print(f"  Inference (max):  {max_time:.2f}ms")
        print(f"  Total latency:    {total_latency:.2f}ms")
        print(f"  Target: ≤10ms, Status: {'✅ PASS' if total_latency <= 10 else '⚠️ CHECK'}")

        return results


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="AuraNet Inference")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default=None,
                        help="Input audio file or directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output audio file or directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu, mps)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Chunk size in ms for streaming")
    parser.add_argument("--no-wdrc", action="store_true",
                        help="Disable Neural-WDRC")
    parser.add_argument("--measure-latency", action="store_true",
                        help="Measure inference latency")

    args = parser.parse_args()

    # Initialize inference engine
    engine = AuraNetInference(
        model_path=args.model,
        device=args.device,
        apply_wdrc_gain=not args.no_wdrc,
    )

    # Measure latency if requested
    if args.measure_latency:
        engine.measure_latency()
        if args.input is None:
            return

    # Need input for processing
    if args.input is None:
        print("Error: --input is required for processing")
        sys.exit(1)

    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        if input_path.is_file():
            args.output = str(input_path.with_suffix(".enhanced.wav"))
        else:
            args.output = str(input_path) + "_enhanced"

    # Process based on input type
    input_path = Path(args.input)

    if input_path.is_file():
        if args.streaming:
            engine.process_streaming(args.input, args.output, args.chunk_size)
        else:
            engine.process_file(args.input, args.output)
    elif input_path.is_dir():
        engine.batch_process(args.input, args.output)
    else:
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
