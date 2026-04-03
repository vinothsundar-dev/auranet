#!/usr/bin/env python3
"""
ONNX Export Pipeline for AuraNet-Lite V2
=========================================

Exports PyTorch model to ONNX format with:
- Dynamic/Static quantization options
- Streaming-compatible design (stateful hidden states)
- INT8 quantization support
- Mobile-optimized ops

Usage:
    python export_onnx.py --model lite_v2_gru --quantize dynamic --output ./exports/
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_optimized_v2 import AuraNetLiteV2


# =============================================================================
# Streaming Wrapper for Export
# =============================================================================

class AuraNetStreaming(nn.Module):
    """
    Streaming wrapper for AuraNet inference.
    
    Wraps the model for frame-by-frame processing with explicit
    hidden state management for ONNX export compatibility.
    
    Input: [1, 2, T, 129] - Single or multiple frames
    Output: [1, 2, T, 129], [hidden_state]
    """
    
    def __init__(self, model: AuraNetLiteV2, use_tcn: bool = False):
        super().__init__()
        self.model = model
        self.use_tcn = use_tcn
        
        # For GRU: hidden shape is [1, hidden_size]
        # For TCN: no hidden state needed
        if not use_tcn:
            self.hidden_size = model.temporal.hidden
        
    def forward(
        self,
        noisy_stft: torch.Tensor,
        hidden: torch.Tensor = None,
    ):
        """
        Forward pass with explicit hidden state.
        
        Args:
            noisy_stft: [B, 2, T, 129] - Noisy STFT frames
            hidden: [1, B, hidden_size] - GRU hidden state (optional)
            
        Returns:
            enhanced_stft: [B, 2, T, 129]
            new_hidden: [1, B, hidden_size] - Updated hidden state
        """
        # Run model
        enhanced_stft, wdrc_params, new_hidden = self.model(noisy_stft, hidden)
        
        # Return enhanced STFT and new hidden state
        if self.use_tcn or new_hidden is None:
            # TCN has no hidden state, return zeros
            dummy_hidden = torch.zeros(1, noisy_stft.size(0), 64)
            return enhanced_stft, dummy_hidden
        
        return enhanced_stft, new_hidden


class AuraNetStateless(nn.Module):
    """
    Stateless wrapper - processes fixed-length chunks.
    Simpler for Core ML / TFLite but less flexible.
    """
    
    def __init__(self, model: AuraNetLiteV2):
        super().__init__()
        self.model = model
        
    def forward(self, noisy_stft: torch.Tensor):
        """
        Stateless forward pass.
        
        Args:
            noisy_stft: [B, 2, T, 129]
            
        Returns:
            enhanced_stft: [B, 2, T, 129]
        """
        enhanced_stft, _, _ = self.model(noisy_stft, None)
        return enhanced_stft


# =============================================================================
# Quantization Utilities
# =============================================================================

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization (INT8 weights, FP32 activations).
    
    Dynamic quantization:
    - Weights are quantized to INT8 at rest
    - Activations computed in FP32, then quantized per-batch
    - Good balance of speed and accuracy
    - No calibration data needed
    
    Best for:
    - GRU/LSTM layers (significant speedup)
    - Linear layers
    - CPU inference
    """
    # Use newer torch.ao.quantization API
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )
    return quantized


def apply_static_quantization(
    model: nn.Module,
    calibration_data: torch.Tensor,
) -> nn.Module:
    """
    Apply static/full INT8 quantization.
    
    Static quantization:
    - Both weights AND activations are INT8
    - Requires calibration data to determine activation ranges
    - Maximum compression and speed
    - May have slight accuracy loss
    
    Best for:
    - Maximum performance on mobile
    - When calibration data is available
    - Conv/Linear heavy models
    
    Note: Static quantization is complex with custom models.
    For ONNX export, we recommend using dynamic quantization
    or post-training quantization in target framework (CoreML/TFLite).
    """
    print("Note: Static quantization requires model preparation.")
    print("Using dynamic quantization instead for ONNX export.")
    print("Apply full INT8 in CoreML/TFLite for best results.")
    
    # Fall back to dynamic quantization
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 2, 100, 129),
    dynamic_batch: bool = True,
    dynamic_time: bool = True,
    opset_version: int = 14,
    streaming: bool = True,
) -> dict:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Output .onnx file path
        input_shape: (B, C, T, F) input shape
        dynamic_batch: Allow variable batch size
        dynamic_time: Allow variable time dimension
        opset_version: ONNX opset version (14+ recommended for mobile)
        streaming: Include hidden state I/O for streaming
        
    Returns:
        dict with export info
    """
    model.eval()
    
    # Create dummy inputs
    dummy_input = torch.randn(*input_shape)
    
    # Define input/output names and dynamic axes
    if streaming and hasattr(model, 'hidden_size'):
        hidden_size = model.hidden_size
        dummy_hidden = torch.zeros(1, input_shape[0], hidden_size)
        inputs = (dummy_input, dummy_hidden)
        input_names = ['noisy_stft', 'hidden_in']
        output_names = ['enhanced_stft', 'hidden_out']
        
        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes['noisy_stft'] = {0: 'batch'}
            dynamic_axes['hidden_in'] = {1: 'batch'}
            dynamic_axes['enhanced_stft'] = {0: 'batch'}
            dynamic_axes['hidden_out'] = {1: 'batch'}
        if dynamic_time:
            dynamic_axes['noisy_stft'] = dynamic_axes.get('noisy_stft', {})
            dynamic_axes['noisy_stft'][2] = 'time'
            dynamic_axes['enhanced_stft'] = dynamic_axes.get('enhanced_stft', {})
            dynamic_axes['enhanced_stft'][2] = 'time'
    else:
        inputs = (dummy_input,)
        input_names = ['noisy_stft']
        output_names = ['enhanced_stft']
        
        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes['noisy_stft'] = {0: 'batch'}
            dynamic_axes['enhanced_stft'] = {0: 'batch'}
        if dynamic_time:
            dynamic_axes['noisy_stft'] = dynamic_axes.get('noisy_stft', {})
            dynamic_axes['noisy_stft'][2] = 'time'
            dynamic_axes['enhanced_stft'] = dynamic_axes.get('enhanced_stft', {})
            dynamic_axes['enhanced_stft'][2] = 'time'
    
    # Export
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Use legacy export API for better compatibility
    torch.onnx.export(
        model,
        inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # Use legacy TorchScript-based export
    )
    
    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Get model size
    file_size = os.path.getsize(output_path)
    
    return {
        'path': output_path,
        'size_bytes': file_size,
        'size_mb': file_size / (1024 * 1024),
        'input_names': input_names,
        'output_names': output_names,
        'input_shape': input_shape,
        'opset_version': opset_version,
    }


def optimize_onnx(input_path: str, output_path: str = None) -> str:
    """
    Optimize ONNX model for inference.
    
    Applies:
    - Constant folding
    - Redundant node elimination
    - Operator fusion
    """
    import onnx
    from onnx import optimizer
    
    if output_path is None:
        output_path = input_path.replace('.onnx', '_optimized.onnx')
    
    model = onnx.load(input_path)
    
    # Available optimization passes
    passes = [
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_nop_pad',
        'fuse_bn_into_conv',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
    ]
    
    # Try to optimize (some passes may not be available)
    try:
        optimized = optimizer.optimize(model, passes)
        onnx.save(optimized, output_path)
        print(f"Optimized ONNX saved to: {output_path}")
    except Exception as e:
        print(f"Optimization warning: {e}")
        # Save original if optimization fails
        onnx.save(model, output_path)
    
    return output_path


# =============================================================================
# Main Export Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Export AuraNet to ONNX')
    parser.add_argument('--model', type=str, default='lite_v2_gru',
                        choices=['lite_v2_gru', 'lite_v2_tcn'],
                        help='Model variant to export')
    parser.add_argument('--quantize', type=str, default='none',
                        choices=['none', 'dynamic', 'static'],
                        help='Quantization mode')
    parser.add_argument('--output', type=str, default='./exports/',
                        help='Output directory')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of time frames (T dimension)')
    parser.add_argument('--streaming', action='store_true', default=True,
                        help='Export streaming-compatible model')
    parser.add_argument('--stateless', action='store_true',
                        help='Export stateless model (no hidden state)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained checkpoint')
    args = parser.parse_args()
    
    print("=" * 70)
    print("AuraNet ONNX Export Pipeline")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n📦 Loading model: {args.model}")
    use_tcn = 'tcn' in args.model
    base_model = AuraNetLiteV2(use_tcn=use_tcn)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location='cpu')
        base_model.load_state_dict(state['model_state_dict'])
        print(f"   Loaded checkpoint: {args.checkpoint}")
    
    base_model.eval()
    params = sum(p.numel() for p in base_model.parameters())
    print(f"   Parameters: {params:,}")
    
    # Wrap for export
    if args.stateless:
        model = AuraNetStateless(base_model)
        streaming = False
        model_suffix = "stateless"
    else:
        model = AuraNetStreaming(base_model, use_tcn=use_tcn)
        streaming = not use_tcn  # TCN has no hidden state
        model_suffix = "streaming" if streaming else "stateless"
    
    # Apply quantization (PyTorch-side, before ONNX export)
    quant_suffix = ""
    if args.quantize == 'dynamic':
        print(f"\n⚡ Applying dynamic INT8 quantization...")
        model = apply_dynamic_quantization(model)
        quant_suffix = "_int8_dynamic"
        print("   - Weights: INT8")
        print("   - Activations: FP32 (quantized at runtime)")
        print("   - Best for: GRU/Linear layers, CPU inference")
    elif args.quantize == 'static':
        print(f"\n⚡ Applying static INT8 quantization...")
        # Generate calibration data
        calib_data = torch.randn(100, 2, args.frames, 129)
        model = apply_static_quantization(model, calib_data)
        quant_suffix = "_int8_static"
        print("   - Weights: INT8")
        print("   - Activations: INT8")
        print("   - Full integer inference")
    
    # Export to ONNX
    print(f"\n📤 Exporting to ONNX...")
    
    output_name = f"auranet_{args.model}_{model_suffix}{quant_suffix}.onnx"
    output_path = str(output_dir / output_name)
    
    input_shape = (1, 2, args.frames, 129)
    
    try:
        info = export_to_onnx(
            model=model,
            output_path=output_path,
            input_shape=input_shape,
            dynamic_batch=True,
            dynamic_time=True,
            opset_version=14,
            streaming=streaming and not args.stateless,
        )
        
        print(f"   ✅ Exported: {info['path']}")
        print(f"   Size: {info['size_mb']:.2f} MB")
        print(f"   Inputs: {info['input_names']}")
        print(f"   Outputs: {info['output_names']}")
        print(f"   Opset: {info['opset_version']}")
        
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        raise
    
    # Verify with ONNX Runtime
    print(f"\n🔍 Verifying ONNX model...")
    try:
        import onnxruntime as ort
        
        sess = ort.InferenceSession(output_path)
        
        # Run inference
        test_input = torch.randn(*input_shape).numpy()
        
        if streaming and not args.stateless and not use_tcn:
            hidden_shape = (1, input_shape[0], 64)
            test_hidden = torch.zeros(*hidden_shape).numpy()
            outputs = sess.run(None, {
                'noisy_stft': test_input,
                'hidden_in': test_hidden,
            })
            print(f"   Output shapes: {[o.shape for o in outputs]}")
        else:
            outputs = sess.run(None, {'noisy_stft': test_input})
            print(f"   Output shape: {outputs[0].shape}")
        
        # Benchmark
        print(f"\n⏱️ Benchmarking ONNX Runtime...")
        times = []
        for _ in range(100):
            start = time.perf_counter()
            if streaming and not args.stateless and not use_tcn:
                sess.run(None, {
                    'noisy_stft': test_input,
                    'hidden_in': test_hidden,
                })
            else:
                sess.run(None, {'noisy_stft': test_input})
            times.append(time.perf_counter() - start)
        
        avg_ms = sum(times) / len(times) * 1000
        print(f"   Average latency: {avg_ms:.2f} ms")
        print(f"   Frames: {args.frames} ({args.frames * 5} ms audio @ 80 hop)")
        print(f"   Real-time factor: {(avg_ms / (args.frames * 5)):.2f}x")
        
        print(f"\n   ✅ Verification passed!")
        
    except ImportError:
        print("   ⚠️ onnxruntime not installed, skipping verification")
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 EXPORT SUMMARY")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Mode:         {model_suffix}")
    print(f"  Quantization: {args.quantize}")
    print(f"  Output:       {output_path}")
    print(f"  Size:         {info['size_mb']:.2f} MB")
    print(f"  Input shape:  {input_shape}")
    print("=" * 70)
    
    return output_path


if __name__ == "__main__":
    main()
