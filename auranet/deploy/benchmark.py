#!/usr/bin/env python3
"""
AuraNet Mobile Deployment Benchmark
====================================

Comprehensive benchmarking for all model variants and formats.

Reports:
- Model size (disk, memory)
- Inference latency (CPU, GPU)
- Real-time factor
- Memory usage
- Expected mobile performance

Usage:
    python benchmark.py --all
    python benchmark.py --pytorch
    python benchmark.py --onnx exports/model.onnx
"""

import os
import sys
import argparse
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Benchmark result for a single model/format."""
    name: str
    format: str  # pytorch, onnx, coreml, tflite
    
    # Size metrics
    file_size_mb: float
    memory_mb: float
    
    # Performance metrics
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Derived metrics
    frames_per_second: float
    real_time_factor: float  # <1 is real-time capable
    
    # Model info
    parameters: int
    quantization: Optional[str]
    device: str
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# PyTorch Benchmarking
# =============================================================================

def benchmark_pytorch(
    model: torch.nn.Module,
    name: str,
    input_shape: tuple = (1, 2, 100, 129),
    num_warmup: int = 20,
    num_runs: int = 100,
    device: str = 'cpu',
) -> BenchmarkResult:
    """
    Benchmark PyTorch model performance.
    """
    model = model.to(device)
    model.eval()
    
    # Get model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_memory_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Get parameter count
    num_params = sum(p.numel() for p in model.parameters())
    
    # Create input
    x = torch.randn(*input_shape, device=device)
    hidden = None
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x, hidden)
    
    # Synchronize if GPU
    if device != 'cpu':
        torch.cuda.synchronize() if torch.cuda.is_available() else torch.mps.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x, hidden)
            if device != 'cpu':
                torch.cuda.synchronize() if torch.cuda.is_available() else torch.mps.synchronize()
            latencies.append(time.perf_counter() - start)
    
    latencies_ms = np.array(latencies) * 1000
    
    # Calculate audio duration (100 frames @ 80 hop = 8000 samples = 0.5s @ 16kHz)
    audio_duration_s = input_shape[2] * 80 / 16000
    
    return BenchmarkResult(
        name=name,
        format='pytorch',
        file_size_mb=total_memory_mb,
        memory_mb=total_memory_mb,
        avg_latency_ms=float(np.mean(latencies_ms)),
        std_latency_ms=float(np.std(latencies_ms)),
        min_latency_ms=float(np.min(latencies_ms)),
        max_latency_ms=float(np.max(latencies_ms)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        p99_latency_ms=float(np.percentile(latencies_ms, 99)),
        frames_per_second=1000 / np.mean(latencies_ms) * input_shape[2],
        real_time_factor=float(np.mean(latencies_ms) / 1000 / audio_duration_s),
        parameters=num_params,
        quantization=None,
        device=device,
    )


# =============================================================================
# ONNX Benchmarking
# =============================================================================

def benchmark_onnx(
    onnx_path: str,
    name: str,
    input_shape: tuple = (1, 2, 100, 129),
    num_warmup: int = 20,
    num_runs: int = 100,
) -> BenchmarkResult:
    """
    Benchmark ONNX model with ONNX Runtime.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed")
        return None
    
    # Get file size
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    # Create session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, sess_options)
    
    # Get inputs
    inputs = {}
    for inp in session.get_inputs():
        if 'hidden' in inp.name.lower():
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            inputs[inp.name] = np.zeros(shape, dtype=np.float32)
        else:
            inputs[inp.name] = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(num_warmup):
        session.run(None, inputs)
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        latencies.append(time.perf_counter() - start)
    
    latencies_ms = np.array(latencies) * 1000
    audio_duration_s = input_shape[2] * 80 / 16000
    
    # Estimate memory (file size is a rough proxy)
    memory_mb = file_size_mb * 1.5  # Account for runtime overhead
    
    return BenchmarkResult(
        name=name,
        format='onnx',
        file_size_mb=file_size_mb,
        memory_mb=memory_mb,
        avg_latency_ms=float(np.mean(latencies_ms)),
        std_latency_ms=float(np.std(latencies_ms)),
        min_latency_ms=float(np.min(latencies_ms)),
        max_latency_ms=float(np.max(latencies_ms)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        p99_latency_ms=float(np.percentile(latencies_ms, 99)),
        frames_per_second=1000 / np.mean(latencies_ms) * input_shape[2],
        real_time_factor=float(np.mean(latencies_ms) / 1000 / audio_duration_s),
        parameters=0,  # Not easily extractable from ONNX
        quantization=None,
        device='cpu',
    )


# =============================================================================  
# Mobile Performance Estimation
# =============================================================================

def estimate_mobile_performance(result: BenchmarkResult) -> Dict[str, Any]:
    """
    Estimate mobile device performance based on desktop benchmarks.
    
    Uses typical CPU/GPU performance ratios:
    - Desktop CPU → Mobile CPU: ~3-5x slower
    - Desktop CPU → Mobile GPU: ~1.5-2x slower (with delegates)
    - Desktop CPU → Neural Engine: ~0.8-1.2x (often faster)
    """
    
    mobile_estimates = {
        'ios_cpu_a15': {
            'factor': 2.5,
            'latency_ms': result.avg_latency_ms * 2.5,
            'real_time': result.real_time_factor * 2.5 < 1.0,
        },
        'ios_neural_engine_a15': {
            'factor': 0.8,
            'latency_ms': result.avg_latency_ms * 0.8,
            'real_time': result.real_time_factor * 0.8 < 1.0,
        },
        'android_cpu_snapdragon_8': {
            'factor': 3.0,
            'latency_ms': result.avg_latency_ms * 3.0,
            'real_time': result.real_time_factor * 3.0 < 1.0,
        },
        'android_gpu_adreno': {
            'factor': 1.5,
            'latency_ms': result.avg_latency_ms * 1.5,
            'real_time': result.real_time_factor * 1.5 < 1.0,
        },
        'android_nnapi': {
            'factor': 1.2,
            'latency_ms': result.avg_latency_ms * 1.2,
            'real_time': result.real_time_factor * 1.2 < 1.0,
        },
    }
    
    return mobile_estimates


# =============================================================================
# Comparison Report
# =============================================================================

def print_benchmark_report(results: List[BenchmarkResult]):
    """
    Print formatted benchmark comparison report.
    """
    print("\n" + "=" * 90)
    print("📊 AURANET MOBILE DEPLOYMENT BENCHMARK")
    print("=" * 90)
    
    # Summary table
    print(f"\n{'Model':<25} {'Format':<10} {'Size (MB)':<12} {'Latency (ms)':<15} {'RTF':<8} {'RT?'}")
    print("-" * 90)
    
    for r in results:
        rt_status = "✅" if r.real_time_factor < 1.0 else "❌"
        print(f"{r.name:<25} {r.format:<10} {r.file_size_mb:>8.2f}    "
              f"{r.avg_latency_ms:>6.2f} ± {r.std_latency_ms:>5.2f}  {r.real_time_factor:>6.3f}  {rt_status}")
    
    # Detailed metrics
    print(f"\n{'Model':<25} {'Min (ms)':<12} {'Max (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("-" * 75)
    
    for r in results:
        print(f"{r.name:<25} {r.min_latency_ms:>8.2f}    {r.max_latency_ms:>8.2f}    "
              f"{r.p95_latency_ms:>8.2f}    {r.p99_latency_ms:>8.2f}")
    
    # Mobile estimates
    print("\n" + "=" * 90)
    print("📱 ESTIMATED MOBILE PERFORMANCE")
    print("=" * 90)
    
    best_result = min(results, key=lambda r: r.avg_latency_ms)
    estimates = estimate_mobile_performance(best_result)
    
    print(f"\nBased on best model: {best_result.name}")
    print(f"Desktop latency: {best_result.avg_latency_ms:.2f} ms")
    print(f"\n{'Platform':<30} {'Est. Latency (ms)':<20} {'Real-time?'}")
    print("-" * 60)
    
    for platform, data in estimates.items():
        rt_status = "✅" if data['real_time'] else "❌"
        print(f"{platform:<30} {data['latency_ms']:>12.2f}        {rt_status}")
    
    # Recommendations
    print("\n" + "=" * 90)
    print("💡 RECOMMENDATIONS")
    print("=" * 90)
    
    # Find best for each criterion
    smallest = min(results, key=lambda r: r.file_size_mb)
    fastest = min(results, key=lambda r: r.avg_latency_ms)
    
    print(f"""
    📦 Smallest model:  {smallest.name} ({smallest.file_size_mb:.2f} MB)
    ⚡ Fastest model:   {fastest.name} ({fastest.avg_latency_ms:.2f} ms)
    
    iOS Deployment:
    - Use Core ML with Neural Engine for best performance
    - Apply FP16 quantization (2x smaller, minimal quality loss)
    - For older devices, use INT8 weights-only quantization
    
    Android Deployment:
    - Use TFLite with GPU delegate for modern devices
    - Apply dynamic quantization for good balance
    - For low-end devices, use full INT8 quantization
    
    Streaming Setup:
    - Process 100 frames (~500ms audio) per call
    - Maintain hidden state between calls
    - Target <10ms latency for comfortable real-time
    """)
    
    print("=" * 90)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark AuraNet models')
    parser.add_argument('--all', action='store_true', help='Benchmark all models')
    parser.add_argument('--pytorch', action='store_true', help='Benchmark PyTorch models')
    parser.add_argument('--onnx', type=str, help='Benchmark specific ONNX model')
    parser.add_argument('--frames', type=int, default=100, help='Number of time frames')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    args = parser.parse_args()
    
    results: List[BenchmarkResult] = []
    input_shape = (1, 2, args.frames, 129)
    
    print("=" * 70)
    print("AuraNet Benchmark Suite")
    print("=" * 70)
    print(f"Input shape: {input_shape}")
    print(f"Audio duration: {args.frames * 80 / 16000 * 1000:.0f} ms")
    print(f"Benchmark runs: {args.runs}")
    
    # Benchmark PyTorch models
    if args.all or args.pytorch:
        print(f"\n🔥 Benchmarking PyTorch Models...")
        
        from model import AuraNet
        from model_optimized import AuraNetLite
        from model_optimized_v2 import AuraNetLiteV2
        
        models = [
            ("AuraNet Original", AuraNet()),
            ("AuraNet Lite V1", AuraNetLite()),
            ("AuraNet Lite V2 (GRU)", AuraNetLiteV2(use_tcn=False)),
            ("AuraNet Lite V2 (TCN)", AuraNetLiteV2(use_tcn=True)),
        ]
        
        for name, model in models:
            print(f"  Benchmarking: {name}...")
            result = benchmark_pytorch(
                model, name, input_shape,
                num_runs=args.runs,
            )
            results.append(result)
            gc.collect()
    
    # Benchmark ONNX model
    if args.onnx:
        print(f"\n📦 Benchmarking ONNX Model: {args.onnx}")
        name = Path(args.onnx).stem
        result = benchmark_onnx(
            args.onnx, name, input_shape,
            num_runs=args.runs,
        )
        if result:
            results.append(result)
    
    # Check for exported ONNX models
    if args.all:
        export_dir = Path(__file__).parent / "exports"
        if export_dir.exists():
            for onnx_file in export_dir.glob("*.onnx"):
                print(f"  Benchmarking: {onnx_file.name}...")
                result = benchmark_onnx(
                    str(onnx_file), onnx_file.stem, input_shape,
                    num_runs=args.runs,
                )
                if result:
                    results.append(result)
    
    # Print report
    if results:
        print_benchmark_report(results)
        
        # Save to JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo models benchmarked. Use --all or --pytorch to run benchmarks.")
    
    return results


if __name__ == "__main__":
    main()
