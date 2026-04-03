# =============================================================================
# Export Utilities for AuraNet
# =============================================================================
#
# EXPORT FORMATS:
# 1. TorchScript: For deployment in PyTorch environments
# 2. ONNX: For cross-platform deployment (mobile, NPU, etc.)
# 3. INT8 Quantized: For edge deployment with reduced memory/compute
#
# DEPLOYMENT CONSIDERATIONS:
# - Edge devices have limited memory (INT8 reduces by 4x)
# - NPUs often prefer ONNX format
# - TorchScript preserves full model fidelity
# =============================================================================

import os
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.quantization as quant

from model import AuraNet, create_auranet


class AuraNetExporter:
    """
    Export AuraNet to various deployment formats.
    
    Supports:
    - TorchScript (trace and script modes)
    - ONNX with dynamic axes
    - INT8 quantization (static and dynamic)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            config: Optional configuration dictionary
        """
        # Load or create model
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            
            if "config" in checkpoint:
                self.config = checkpoint["config"]
            else:
                self.config = config or {}
                
            self.model = create_auranet(self.config)
            
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"Loaded model from {model_path}")
        else:
            print("Creating new model (no checkpoint provided)")
            self.config = config or {}
            self.model = create_auranet(self.config)
            
        self.model.eval()
        
    def export_torchscript(
        self,
        output_path: str,
        method: str = "trace",
        example_input_shape: Tuple[int, ...] = (1, 2, 100, 129),
    ) -> str:
        """
        Export model to TorchScript format.
        
        TorchScript enables:
        - Running without Python interpreter
        - Optimization passes
        - Mobile deployment via PyTorch Mobile
        
        Args:
            output_path: Output file path (.pt)
            method: "trace" or "script"
            example_input_shape: Shape of example input for tracing
            
        Returns:
            Path to exported model
        """
        print(f"\nExporting to TorchScript ({method} mode)...")
        
        # Create example input
        example_input = torch.randn(*example_input_shape)
        
        if method == "trace":
            # Trace-based export (captures specific execution path)
            # Good for models without control flow
            with torch.no_grad():
                # Need to wrap to handle multiple outputs
                class TracingWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        
                    def forward(self, x):
                        enhanced, _, _ = self.model(x)
                        return enhanced
                        
                wrapper = TracingWrapper(self.model)
                scripted = torch.jit.trace(wrapper, example_input)
                
        elif method == "script":
            # Script-based export (analyzes source code)
            # Better for models with control flow
            scripted = torch.jit.script(self.model)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
            
        # Optimize for inference
        scripted = torch.jit.optimize_for_inference(scripted)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(output_path))
        
        # Verify
        loaded = torch.jit.load(str(output_path))
        with torch.no_grad():
            if method == "trace":
                original_out = self.model(example_input)[0]
            else:
                original_out = self.model(example_input)[0]
            loaded_out = loaded(example_input)
            
            if hasattr(loaded_out, "__iter__") and not isinstance(loaded_out, torch.Tensor):
                loaded_out = loaded_out[0]
                
            max_diff = torch.max(torch.abs(original_out - loaded_out)).item()
            print(f"  Max output difference: {max_diff:.6e}")
            
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved to: {output_path}")
        print(f"  File size: {file_size:.2f} MB")
        
        return str(output_path)
    
    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 14,
        example_input_shape: Tuple[int, ...] = (1, 2, 100, 129),
        dynamic_axes: bool = True,
    ) -> str:
        """
        Export model to ONNX format.
        
        ONNX enables:
        - Cross-platform deployment
        - Hardware acceleration (TensorRT, CoreML, etc.)
        - NPU compatibility
        
        Args:
            output_path: Output file path (.onnx)
            opset_version: ONNX opset version
            example_input_shape: Shape of example input
            dynamic_axes: Enable dynamic batch/time dimensions
            
        Returns:
            Path to exported model
        """
        print(f"\nExporting to ONNX (opset {opset_version})...")
        
        # Create example input
        example_input = torch.randn(*example_input_shape)
        
        # Create wrapper for clean export
        class ONNXWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                enhanced, wdrc, _ = self.model(x)
                # Return just enhanced STFT and gain
                # (most important outputs for deployment)
                return enhanced, wdrc["gain"]
                
        wrapper = ONNXWrapper(self.model)
        wrapper.eval()
        
        # Configure dynamic axes
        if dynamic_axes:
            dyn_axes = {
                "input": {0: "batch", 2: "time"},
                "enhanced": {0: "batch", 2: "time"},
                "gain": {0: "batch", 1: "time"},
            }
        else:
            dyn_axes = None
            
        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example_input,
                str(output_path),
                opset_version=opset_version,
                input_names=["input"],
                output_names=["enhanced", "gain"],
                dynamic_axes=dyn_axes,
                do_constant_folding=True,
                export_params=True,
            )
            
        # Verify with ONNX runtime if available
        try:
            import onnxruntime as ort
            
            # Create session
            sess = ort.InferenceSession(
                str(output_path),
                providers=["CPUExecutionProvider"],
            )
            
            # Run inference
            input_np = example_input.numpy()
            outputs = sess.run(None, {"input": input_np})
            
            # Compare
            with torch.no_grad():
                enhanced_py, wdrc_py, _ = self.model(example_input)
                gain_py = wdrc_py["gain"]
                
            max_diff_enhanced = abs(outputs[0] - enhanced_py.numpy()).max()
            max_diff_gain = abs(outputs[1] - gain_py.numpy()).max()
            
            print(f"  ONNX verification:")
            print(f"    Enhanced max diff: {max_diff_enhanced:.6e}")
            print(f"    Gain max diff: {max_diff_gain:.6e}")
            
        except ImportError:
            print("  Note: Install onnxruntime for verification")
            
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved to: {output_path}")
        print(f"  File size: {file_size:.2f} MB")
        
        return str(output_path)
    
    def quantize_dynamic(
        self,
        output_path: str,
    ) -> str:
        """
        Apply dynamic quantization (INT8 weights, FP32 activations).
        
        Dynamic quantization:
        - Quantizes weights to INT8 at save time
        - Quantizes activations dynamically at runtime
        - No calibration data required
        - ~2-4x speedup on CPU
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to quantized model
        """
        print("\nApplying dynamic quantization...")
        
        # Apply dynamic quantization to Linear and GRU layers
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.GRU},
            dtype=torch.qint8,
        )
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model.state_dict(), output_path)
        
        # Compare sizes
        original_size = sum(
            p.numel() * p.element_size() 
            for p in self.model.parameters()
        ) / (1024 * 1024)
        
        quantized_size = output_path.stat().st_size / (1024 * 1024)
        
        print(f"  Original model size: {original_size:.2f} MB")
        print(f"  Quantized model size: {quantized_size:.2f} MB")
        print(f"  Compression ratio: {original_size/quantized_size:.2f}x")
        print(f"  Saved to: {output_path}")
        
        return str(output_path)
    
    def quantize_static(
        self,
        output_path: str,
        calibration_data: Optional[torch.Tensor] = None,
        num_calibration_batches: int = 100,
    ) -> str:
        """
        Apply static quantization (INT8 weights and activations).
        
        Static quantization:
        - Quantizes both weights and activations to INT8
        - Requires calibration data to determine ranges
        - Maximum speedup and size reduction
        - Best for edge deployment
        
        Args:
            output_path: Output file path
            calibration_data: Optional calibration tensor [N, 2, T, F]
            num_calibration_batches: Number of calibration batches if generating
            
        Returns:
            Path to quantized model
        """
        print("\nApplying static quantization...")
        
        # Set quantization config
        self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(self.model, inplace=False)
        
        # Calibration
        print("  Running calibration...")
        
        if calibration_data is None:
            # Generate synthetic calibration data
            calibration_data = torch.randn(num_calibration_batches, 2, 100, 129)
            
        model_prepared.eval()
        with torch.no_grad():
            for i in range(min(num_calibration_batches, len(calibration_data))):
                batch = calibration_data[i:i+1]
                _ = model_prepared(batch)
                
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_quantized.state_dict(), output_path)
        
        print(f"  Saved to: {output_path}")
        
        return str(output_path)
    
    def export_all(
        self,
        output_dir: str,
        name_prefix: str = "auranet",
    ) -> Dict[str, str]:
        """
        Export to all supported formats.
        
        Args:
            output_dir: Output directory
            name_prefix: Prefix for output files
            
        Returns:
            Dictionary mapping format to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        # TorchScript
        try:
            exports["torchscript"] = self.export_torchscript(
                output_dir / f"{name_prefix}_traced.pt"
            )
        except Exception as e:
            print(f"TorchScript export failed: {e}")
            
        # ONNX
        try:
            exports["onnx"] = self.export_onnx(
                output_dir / f"{name_prefix}.onnx"
            )
        except Exception as e:
            print(f"ONNX export failed: {e}")
            
        # Dynamic quantization
        try:
            exports["quantized_dynamic"] = self.quantize_dynamic(
                output_dir / f"{name_prefix}_int8_dynamic.pt"
            )
        except Exception as e:
            print(f"Dynamic quantization failed: {e}")
            
        print(f"\n✅ Exports complete!")
        print("Exported files:")
        for fmt, path in exports.items():
            print(f"  {fmt}: {path}")
            
        return exports


def benchmark_model(
    model: AuraNet,
    input_shape: Tuple[int, ...] = (1, 2, 100, 129),
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
        device: Device to run on
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    example_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(example_input)
            
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            start = time.perf_counter()
            _ = model(example_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            times.append(time.perf_counter() - start)
            
    times_ms = [t * 1000 for t in times]
    
    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": (sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
    }


def main():
    """Main entry point for export utilities."""
    parser = argparse.ArgumentParser(description="Export AuraNet model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="exported_models",
                        help="Output directory")
    parser.add_argument("--format", type=str, default="all",
                        choices=["all", "torchscript", "onnx", "quantized"],
                        help="Export format")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for benchmark")
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = AuraNetExporter(model_path=args.model)
    
    # Run benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 60)
        print("Running benchmark...")
        print("=" * 60)
        
        stats = benchmark_model(exporter.model, device=args.device)
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Min:  {stats['min_ms']:.2f} ms")
        print(f"  Max:  {stats['max_ms']:.2f} ms")
        print(f"  Std:  {stats['std_ms']:.2f} ms")
        
    # Export
    output_dir = Path(args.output_dir)
    
    if args.format == "all":
        exporter.export_all(output_dir)
    elif args.format == "torchscript":
        exporter.export_torchscript(output_dir / "auranet.pt")
    elif args.format == "onnx":
        exporter.export_onnx(output_dir / "auranet.onnx")
    elif args.format == "quantized":
        exporter.quantize_dynamic(output_dir / "auranet_int8.pt")


if __name__ == "__main__":
    main()
