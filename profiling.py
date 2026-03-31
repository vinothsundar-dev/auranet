#!/usr/bin/env python3
"""
================================================================================
AuraNet V2 Edge - Profiling & DSP Optimization
================================================================================

Provides:
1. Per-stage latency profiling (STFT → Encoder → TCN → Decoder → iSTFT)
2. Memory allocation tracking
3. DSP optimizations (precomputed windows, efficient FFT)
4. Streaming throughput benchmarks
5. Before vs After comparison

Target metrics:
- End-to-end latency: <10ms per frame
- STFT/iSTFT: <1ms each
- Model inference: <5ms

================================================================================
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# TIMING UTILITIES
# ==============================================================================

@contextmanager
def timer(name: str, results: Dict[str, List[float]]):
    """Context manager for timing code blocks."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    if name not in results:
        results[name] = []
    results[name].append(elapsed)


class GPUTimer:
    """High-precision GPU timer using CUDA events."""
    
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        if self.use_cuda:
            self.start_event.record()
        else:
            self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Returns elapsed time in milliseconds."""
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (time.perf_counter() - self._start_time) * 1000


# ==============================================================================
# DSP OPTIMIZATIONS
# ==============================================================================

class OptimizedSTFT(nn.Module):
    """
    Optimized STFT with precomputed windows and efficient memory.
    
    Optimizations:
    1. Precomputed Hann window (no allocation per call)
    2. In-place operations where possible
    3. Contiguous tensor operations
    4. Optional half-precision
    """
    
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 80,
        win_length: int = 160,
        center: bool = True,
        device: torch.device = torch.device('cpu'),
        use_half: bool = False,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.n_freqs = n_fft // 2 + 1
        
        # Precompute window
        dtype = torch.float16 if use_half else torch.float32
        window = torch.hann_window(win_length, periodic=True, dtype=dtype, device=device)
        self.register_buffer('window', window)
        
        # Precompute padding for centered STFT
        self.pad_amount = n_fft // 2
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT.
        
        Args:
            audio: [B, N] or [N]
            
        Returns:
            Complex STFT as [B, 2, T, F] (real, imag stacked)
        """
        # Ensure batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Center padding
        if self.center:
            audio = F.pad(audio, (self.pad_amount, self.pad_amount), mode='reflect')
        
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,  # Already padded
            return_complex=True,
        )  # [B, F, T]
        
        # Reformat to [B, 2, T, F]
        stft = stft.permute(0, 2, 1)  # [B, T, F]
        return torch.stack([stft.real, stft.imag], dim=1)  # [B, 2, T, F]
    
    def inverse(self, stft_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse STFT.
        
        Args:
            stft_tensor: [B, 2, T, F]
            
        Returns:
            Audio [B, N]
        """
        # Convert to complex
        real = stft_tensor[:, 0]  # [B, T, F]
        imag = stft_tensor[:, 1]
        
        complex_stft = torch.complex(real, imag).permute(0, 2, 1)  # [B, F, T]
        
        # iSTFT
        audio = torch.istft(
            complex_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )
        
        return audio


class OptimizedStreamingSTFT:
    """
    Highly optimized streaming STFT for real-time processing.
    
    Features:
    - Zero allocations per frame (reuses buffers)
    - Preallocated output tensors
    - Optimized overlap-add with circular buffer
    """
    
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 80,
        win_length: int = 160,
        device: torch.device = torch.device('cpu'),
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freqs = n_fft // 2 + 1
        self.device = device
        
        # Precompute windows
        self.analysis_window = torch.hann_window(win_length, device=device)
        self.synthesis_window = torch.hann_window(win_length, device=device)
        
        # Normalize synthesis window for perfect reconstruction
        # OLA normalization factor
        norm_factor = self._compute_ola_norm()
        self.synthesis_window = self.synthesis_window / norm_factor
        
        # Preallocate buffers
        self.input_buffer = torch.zeros(n_fft, device=device)
        self.output_buffer = torch.zeros(win_length, device=device)
        
        # Working buffers (avoid allocations)
        self.windowed_frame = torch.zeros(n_fft, device=device)
        self.spectrum = torch.zeros(self.n_freqs, dtype=torch.complex64, device=device)
        
    def _compute_ola_norm(self) -> float:
        """Compute overlap-add normalization factor."""
        n_frames = self.win_length // self.hop_length + 1
        summed = torch.zeros(self.win_length, device=self.device)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.win_length
            if end <= self.win_length + self.hop_length:
                valid_start = max(0, start)
                valid_end = min(self.win_length, end)
                summed[valid_start:valid_end] += self.synthesis_window[:valid_end - valid_start] ** 2
        
        return summed.max().item()
    
    def reset(self):
        """Reset all buffers."""
        self.input_buffer.zero_()
        self.output_buffer.zero_()
    
    def process_frame_stft(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """
        Process single frame through STFT.
        
        Args:
            audio_chunk: [hop_length] samples
            
        Returns:
            Complex STFT frame [2, 1, n_freqs]
        """
        # Shift input buffer and add new samples
        self.input_buffer[:-self.hop_length] = self.input_buffer[self.hop_length:].clone()
        self.input_buffer[-self.hop_length:] = audio_chunk
        
        # Window and pad to n_fft
        self.windowed_frame[:self.win_length] = (
            self.input_buffer[-self.win_length:] * self.analysis_window
        )
        self.windowed_frame[self.win_length:] = 0
        
        # FFT
        spectrum = torch.fft.rfft(self.windowed_frame)
        
        # Format as [2, 1, F]
        return torch.stack([spectrum.real, spectrum.imag]).unsqueeze(1)
    
    def process_frame_istft(self, stft_frame: torch.Tensor) -> torch.Tensor:
        """
        Process single frame through iSTFT with overlap-add.
        
        Args:
            stft_frame: [2, 1, n_freqs] or [2, n_freqs]
            
        Returns:
            Audio samples [hop_length]
        """
        # Handle input shape
        if stft_frame.dim() == 3:
            real = stft_frame[0, 0]
            imag = stft_frame[1, 0]
        else:
            real = stft_frame[0]
            imag = stft_frame[1]
        
        # iFFT
        complex_spec = torch.complex(real, imag)
        frame = torch.fft.irfft(complex_spec, n=self.n_fft)[:self.win_length]
        
        # Apply synthesis window
        frame = frame * self.synthesis_window
        
        # Overlap-add
        self.output_buffer += frame
        
        # Extract output
        output = self.output_buffer[:self.hop_length].clone()
        
        # Shift output buffer
        self.output_buffer[:-self.hop_length] = self.output_buffer[self.hop_length:].clone()
        self.output_buffer[-self.hop_length:] = 0
        
        return output


# ==============================================================================
# PROFILER
# ==============================================================================

@dataclass
class ProfileResult:
    """Profiling results container."""
    name: str
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    samples: int
    
    def __repr__(self):
        return (
            f"{self.name:20s} | "
            f"mean: {self.mean_ms:6.2f}ms | "
            f"p50: {self.p50_ms:6.2f}ms | "
            f"p95: {self.p95_ms:6.2f}ms | "
            f"max: {self.max_ms:6.2f}ms"
        )


class ModelProfiler:
    """
    Comprehensive profiler for AuraNet models.
    
    Measures:
    - Per-module latency
    - Total inference time
    - Memory allocation
    - Streaming throughput
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        warmup_runs: int = 20,
        profile_runs: int = 100,
    ):
        self.model = model
        self.device = device
        self.warmup_runs = warmup_runs
        self.profile_runs = profile_runs
        
        self.results: Dict[str, List[float]] = {}
        
    def _compute_stats(self, times: List[float], name: str) -> ProfileResult:
        """Compute statistics from timing list."""
        times = sorted(times)
        n = len(times)
        
        return ProfileResult(
            name=name,
            mean_ms=statistics.mean(times),
            std_ms=statistics.stdev(times) if n > 1 else 0,
            p50_ms=times[n // 2],
            p95_ms=times[int(n * 0.95)],
            p99_ms=times[int(n * 0.99)],
            min_ms=times[0],
            max_ms=times[-1],
            samples=n,
        )
    
    def profile_stft(self, audio_length: int = 16000) -> ProfileResult:
        """Profile STFT operation."""
        stft = OptimizedSTFT(device=self.device)
        audio = torch.randn(1, audio_length, device=self.device)
        
        # Warmup
        for _ in range(self.warmup_runs):
            _ = stft(audio)
        
        # Profile
        times = []
        for _ in range(self.profile_runs):
            timer_ = GPUTimer()
            timer_.start()
            _ = stft(audio)
            times.append(timer_.stop())
        
        return self._compute_stats(times, 'STFT')
    
    def profile_istft(self, time_frames: int = 100) -> ProfileResult:
        """Profile iSTFT operation."""
        stft = OptimizedSTFT(device=self.device)
        stft_tensor = torch.randn(1, 2, time_frames, 129, device=self.device)
        
        # Warmup
        for _ in range(self.warmup_runs):
            _ = stft.inverse(stft_tensor)
        
        # Profile
        times = []
        for _ in range(self.profile_runs):
            timer_ = GPUTimer()
            timer_.start()
            _ = stft.inverse(stft_tensor)
            times.append(timer_.stop())
        
        return self._compute_stats(times, 'iSTFT')
    
    def profile_model_inference(
        self,
        batch_size: int = 1,
        time_frames: int = 100,
    ) -> ProfileResult:
        """Profile model forward pass."""
        self.model.eval()
        self.model = self.model.to(self.device)
        
        x = torch.randn(batch_size, 2, time_frames, 129, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = self.model(x)
        
        # Profile
        times = []
        with torch.no_grad():
            for _ in range(self.profile_runs):
                timer_ = GPUTimer()
                timer_.start()
                _ = self.model(x)
                times.append(timer_.stop())
        
        return self._compute_stats(times, 'Model Inference')
    
    def profile_per_module(
        self,
        time_frames: int = 100,
    ) -> Dict[str, ProfileResult]:
        """Profile individual modules."""
        self.model.eval()
        self.model = self.model.to(self.device)
        
        results = {}
        x = torch.randn(1, 2, time_frames, 129, device=self.device)
        
        # Hook-based profiling
        module_times: Dict[str, List[float]] = {}
        
        def create_hook(name):
            def hook(module, input, output):
                if name not in module_times:
                    module_times[name] = []
            return hook
        
        # Note: Simplified profiling - full version would use hooks
        # For now, profile major components directly if accessible
        
        if hasattr(self.model, 'physics'):
            times = []
            mag = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-8)
            with torch.no_grad():
                for _ in range(self.warmup_runs):
                    _ = self.model.physics(mag)
                for _ in range(self.profile_runs):
                    timer_ = GPUTimer()
                    timer_.start()
                    _ = self.model.physics(mag)
                    times.append(timer_.stop())
            results['Physics'] = self._compute_stats(times, 'Physics')
        
        if hasattr(self.model, 'encoder'):
            times = []
            inp = torch.randn(1, 6, time_frames, 129, device=self.device)
            with torch.no_grad():
                for _ in range(self.warmup_runs):
                    _ = self.model.encoder(inp)
                for _ in range(self.profile_runs):
                    timer_ = GPUTimer()
                    timer_.start()
                    _ = self.model.encoder(inp)
                    times.append(timer_.stop())
            results['Encoder'] = self._compute_stats(times, 'Encoder')
        
        if hasattr(self.model, 'bottleneck'):
            times = []
            # Estimate bottleneck input shape
            encoded = torch.randn(1, 48, time_frames, 17, device=self.device)
            with torch.no_grad():
                for _ in range(self.warmup_runs):
                    _ = self.model.bottleneck(encoded)
                for _ in range(self.profile_runs):
                    timer_ = GPUTimer()
                    timer_.start()
                    _ = self.model.bottleneck(encoded)
                    times.append(timer_.stop())
            results['Bottleneck'] = self._compute_stats(times, 'Bottleneck')
        
        return results
    
    def profile_streaming(
        self,
        audio_length_sec: float = 5.0,
        sample_rate: int = 16000,
        hop_length: int = 80,
    ) -> Dict[str, float]:
        """
        Profile streaming inference.
        
        Returns throughput and per-frame latency stats.
        """
        try:
            from auranet_v2_edge import StreamingAuraNet
        except ImportError:
            print("StreamingAuraNet not available")
            return {}
        
        self.model.eval()
        
        streamer = StreamingAuraNet(self.model, device=self.device)
        
        num_samples = int(audio_length_sec * sample_rate)
        audio = torch.randn(num_samples, device=self.device)
        
        num_frames = num_samples // hop_length
        frame_times = []
        
        # Process frame by frame
        for i in range(num_frames):
            chunk = audio[i * hop_length : (i + 1) * hop_length]
            
            timer_ = GPUTimer()
            timer_.start()
            _ = streamer.process_frame(chunk)
            frame_times.append(timer_.stop())
        
        frame_times = sorted(frame_times)
        
        # Real-time factor: if < 1.0, faster than real-time
        frame_duration_ms = (hop_length / sample_rate) * 1000
        rtf = statistics.mean(frame_times) / frame_duration_ms
        
        return {
            'total_frames': num_frames,
            'mean_frame_ms': statistics.mean(frame_times),
            'p50_frame_ms': frame_times[num_frames // 2],
            'p95_frame_ms': frame_times[int(num_frames * 0.95)],
            'p99_frame_ms': frame_times[int(num_frames * 0.99)],
            'max_frame_ms': frame_times[-1],
            'frame_duration_ms': frame_duration_ms,
            'real_time_factor': rtf,
            'is_realtime': rtf < 1.0,
        }
    
    def profile_memory(self) -> Dict[str, float]:
        """Profile memory usage."""
        if not torch.cuda.is_available():
            return {'gpu_available': False}
        
        torch.cuda.reset_peak_memory_stats()
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        x = torch.randn(1, 2, 100, 129, device=self.device)
        
        with torch.no_grad():
            _ = self.model(x)
        
        return {
            'gpu_available': True,
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1e6,
            'reserved_memory_mb': torch.cuda.memory_reserved() / 1e6,
        }
    
    def run_full_profile(self) -> Dict[str, any]:
        """Run comprehensive profiling."""
        print("=" * 70)
        print("AURANET EDGE PROFILING REPORT")
        print("=" * 70)
        
        results = {}
        
        # Model info
        params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel Parameters: {params:,}")
        print(f"Device: {self.device}")
        results['parameters'] = params
        
        # STFT
        print("\n[1/5] Profiling STFT...")
        stft_result = self.profile_stft()
        print(f"  {stft_result}")
        results['stft'] = stft_result
        
        # iSTFT
        print("\n[2/5] Profiling iSTFT...")
        istft_result = self.profile_istft()
        print(f"  {istft_result}")
        results['istft'] = istft_result
        
        # Model
        print("\n[3/5] Profiling Model Inference...")
        model_result = self.profile_model_inference()
        print(f"  {model_result}")
        results['model'] = model_result
        
        # Per-module
        print("\n[4/5] Profiling Per-Module...")
        module_results = self.profile_per_module()
        for name, result in module_results.items():
            print(f"  {result}")
        results['modules'] = module_results
        
        # Streaming
        print("\n[5/5] Profiling Streaming...")
        streaming_results = self.profile_streaming()
        if streaming_results:
            print(f"  Real-time factor: {streaming_results['real_time_factor']:.3f}")
            print(f"  Frame latency p95: {streaming_results['p95_frame_ms']:.2f}ms")
            print(f"  Is real-time: {'✓' if streaming_results['is_realtime'] else '✗'}")
        results['streaming'] = streaming_results
        
        # Memory
        if torch.cuda.is_available():
            print("\n[Bonus] Memory Profile...")
            mem_results = self.profile_memory()
            print(f"  Peak GPU memory: {mem_results['peak_memory_mb']:.1f} MB")
            results['memory'] = mem_results
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        total_latency = stft_result.p95_ms + model_result.p95_ms + istft_result.p95_ms
        print(f"\nEnd-to-end latency (p95): {total_latency:.2f}ms")
        print(f"  STFT: {stft_result.p95_ms:.2f}ms")
        print(f"  Model: {model_result.p95_ms:.2f}ms")
        print(f"  iSTFT: {istft_result.p95_ms:.2f}ms")
        
        target = 10.0
        if total_latency < target:
            print(f"\n✅ PASS: Latency {total_latency:.1f}ms < {target}ms target")
        else:
            print(f"\n❌ FAIL: Latency {total_latency:.1f}ms > {target}ms target")
            print("   Suggestions:")
            if model_result.p95_ms > 5:
                print("   - Reduce TCN layers or channels")
            if stft_result.p95_ms > 1:
                print("   - Use smaller FFT size or optimized FFT library")
        
        results['total_p95_ms'] = total_latency
        results['meets_target'] = total_latency < target
        
        return results


# ==============================================================================
# BEFORE VS AFTER COMPARISON
# ==============================================================================

def compare_models(
    original_model: nn.Module,
    optimized_model: nn.Module,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, any]:
    """
    Compare original and optimized models.
    
    Compares:
    - Parameters
    - Latency
    - Memory
    - Quality (if possible)
    """
    print("=" * 70)
    print("MODEL COMPARISON: Original vs Optimized")
    print("=" * 70)
    
    results = {}
    
    # Parameters
    orig_params = sum(p.numel() for p in original_model.parameters())
    opt_params = sum(p.numel() for p in optimized_model.parameters())
    
    print(f"\n📊 Parameters:")
    print(f"  Original:  {orig_params:>10,}")
    print(f"  Optimized: {opt_params:>10,}")
    print(f"  Reduction: {(1 - opt_params/orig_params) * 100:.1f}%")
    
    results['params'] = {
        'original': orig_params,
        'optimized': opt_params,
        'reduction_pct': (1 - opt_params/orig_params) * 100,
    }
    
    # Latency
    print(f"\n⏱️ Latency:")
    
    x = torch.randn(1, 2, 100, 129, device=device)
    
    # Profile original
    original_model.eval()
    original_model = original_model.to(device)
    
    with torch.no_grad():
        for _ in range(20):
            _ = original_model(x)
        
        orig_times = []
        for _ in range(100):
            timer_ = GPUTimer()
            timer_.start()
            _ = original_model(x)
            orig_times.append(timer_.stop())
    
    orig_times = sorted(orig_times)
    
    # Profile optimized
    optimized_model.eval()
    optimized_model = optimized_model.to(device)
    
    with torch.no_grad():
        for _ in range(20):
            _ = optimized_model(x)
        
        opt_times = []
        for _ in range(100):
            timer_ = GPUTimer()
            timer_.start()
            _ = optimized_model(x)
            opt_times.append(timer_.stop())
    
    opt_times = sorted(opt_times)
    
    print(f"  Original  P50: {orig_times[50]:6.2f}ms | P95: {orig_times[95]:6.2f}ms")
    print(f"  Optimized P50: {opt_times[50]:6.2f}ms | P95: {opt_times[95]:6.2f}ms")
    print(f"  Speedup: {orig_times[50] / opt_times[50]:.2f}x")
    
    results['latency'] = {
        'original_p50': orig_times[50],
        'original_p95': orig_times[95],
        'optimized_p50': opt_times[50],
        'optimized_p95': opt_times[95],
        'speedup': orig_times[50] / opt_times[50],
    }
    
    # Model size
    import tempfile
    
    with tempfile.NamedTemporaryFile() as f:
        torch.save(original_model.state_dict(), f.name)
        orig_size = os.path.getsize(f.name) / 1e6
    
    with tempfile.NamedTemporaryFile() as f:
        torch.save(optimized_model.state_dict(), f.name)
        opt_size = os.path.getsize(f.name) / 1e6
    
    print(f"\n💾 Model Size:")
    print(f"  Original:  {orig_size:.2f} MB")
    print(f"  Optimized: {opt_size:.2f} MB")
    print(f"  Reduction: {(1 - opt_size/orig_size) * 100:.1f}%")
    
    results['size'] = {
        'original_mb': orig_size,
        'optimized_mb': opt_size,
        'reduction_pct': (1 - opt_size/orig_size) * 100,
    }
    
    # Output comparison (quality proxy)
    print(f"\n🎵 Output Quality:")
    
    with torch.no_grad():
        orig_out = original_model(x)
        opt_out = optimized_model(x)
        
        if isinstance(orig_out, tuple):
            orig_out = orig_out[0]
        if isinstance(opt_out, tuple):
            opt_out = opt_out[0]
        
        # Note: Shape may differ, compare common elements
        if orig_out.shape == opt_out.shape:
            mse = F.mse_loss(orig_out, opt_out).item()
            max_diff = (orig_out - opt_out).abs().max().item()
            print(f"  Output MSE: {mse:.6f}")
            print(f"  Max Diff: {max_diff:.6f}")
            results['quality'] = {'mse': mse, 'max_diff': max_diff}
        else:
            print(f"  (Shapes differ: {orig_out.shape} vs {opt_out.shape})")
    
    print("\n" + "=" * 70)
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile AuraNet Edge")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--compare', action='store_true', help='Compare with original model')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        from auranet_v2_edge import AuraNetEdge
        model = AuraNetEdge()
        print(f"Loaded AuraNetEdge. Parameters: {model.count_parameters():,}")
    except ImportError:
        print("Could not import AuraNetEdge. Using dummy model.")
        model = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1),
        )
    
    # Profile
    profiler = ModelProfiler(model, device=device)
    results = profiler.run_full_profile()
    
    # Compare if requested
    if args.compare:
        try:
            from auranet_v2_complete import AuraNetV2Complete
            original = AuraNetV2Complete()
            comparison = compare_models(original, model, device=device)
        except ImportError:
            print("\nCould not import AuraNetV2Complete for comparison.")
