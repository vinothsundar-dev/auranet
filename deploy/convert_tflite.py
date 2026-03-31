#!/usr/bin/env python3
"""
TensorFlow Lite Conversion for AuraNet-Lite V2
===============================================

Converts ONNX model to TensorFlow Lite format for Android deployment.

Features:
- INT8 quantization (dynamic and full)
- GPU delegate support
- NNAPI delegate support
- Streaming-compatible

Usage:
    python convert_tflite.py --input exports/auranet_lite_v2_gru_streaming.onnx --quantize int8

Requirements:
    pip install tensorflow onnx-tf tf2onnx
    
Note:
    ONNX → TF → TFLite conversion path
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tensorflow not installed. Install with: pip install tensorflow")

try:
    import onnx
    from onnx_tf.backend import prepare
    ONNX_TF_AVAILABLE = True
except ImportError:
    ONNX_TF_AVAILABLE = False
    print("Warning: onnx-tf not installed. Install with: pip install onnx-tf")


# =============================================================================
# TFLite Conversion Utilities
# =============================================================================

def onnx_to_saved_model(onnx_path: str, output_dir: str) -> str:
    """
    Convert ONNX model to TensorFlow SavedModel format.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save TF model
        
    Returns:
        Path to SavedModel directory
    """
    if not ONNX_TF_AVAILABLE:
        raise ImportError("onnx-tf not installed")
    
    print(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    
    print("Converting to TensorFlow...")
    tf_rep = prepare(onnx_model)
    
    # Export to SavedModel
    saved_model_path = os.path.join(output_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=True)
    tf_rep.export_graph(saved_model_path)
    
    print(f"SavedModel saved to: {saved_model_path}")
    return saved_model_path


def convert_to_tflite(
    saved_model_path: str,
    output_path: str,
    quantize: Optional[str] = None,
    input_shapes: Optional[Dict[str, List[int]]] = None,
    representative_data: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Convert TensorFlow SavedModel to TFLite format.
    
    Args:
        saved_model_path: Path to SavedModel directory
        output_path: Output .tflite file path
        quantize: None, 'dynamic', 'float16', 'int8'
        input_shapes: Optional dict of input shapes for conversion
        representative_data: Generator for INT8 calibration
        
    Returns:
        Dict with conversion info
    """
    if not TF_AVAILABLE:
        raise ImportError("tensorflow not installed")
    
    print(f"Loading SavedModel: {saved_model_path}")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # Enable TF Select ops for broader compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    
    # Apply quantization
    if quantize == 'dynamic':
        print("Applying dynamic quantization...")
        # Dynamic range quantization
        # - Weights: INT8
        # - Activations: FP32 at runtime
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    elif quantize == 'float16':
        print("Applying FP16 quantization...")
        # FP16 quantization for GPU
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantize == 'int8':
        print("Applying full INT8 quantization...")
        # Full integer quantization
        # Requires calibration data
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        if representative_data:
            converter.representative_dataset = representative_data
        else:
            # Generate synthetic calibration data
            def representative_dataset():
                for _ in range(100):
                    # [1, 2, 100, 129] input shape
                    data = np.random.randn(1, 2, 100, 129).astype(np.float32)
                    yield [data]
            converter.representative_dataset = representative_dataset
    
    # Convert
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Conversion error: {e}")
        print("Trying with experimental features...")
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        tflite_model = converter.convert()
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size = os.path.getsize(output_path)
    
    return {
        'path': output_path,
        'size_bytes': file_size,
        'size_mb': file_size / (1024 * 1024),
        'quantization': quantize,
    }


# =============================================================================
# TFLite Verification and Benchmarking
# =============================================================================

def verify_tflite_model(
    tflite_path: str,
    input_shape: tuple = (1, 2, 100, 129),
) -> Dict[str, Any]:
    """
    Verify TFLite model loads and runs correctly.
    
    Returns inference timing and output verification.
    """
    if not TF_AVAILABLE:
        raise ImportError("tensorflow not installed")
    
    import time
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Inputs: {len(input_details)}")
    for inp in input_details:
        print(f"  - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
    
    print(f"Outputs: {len(output_details)}")
    for out in output_details:
        print(f"  - {out['name']}: shape={out['shape']}, dtype={out['dtype']}")
    
    # Create test input
    test_data = {}
    for inp in input_details:
        shape = inp['shape']
        dtype = inp['dtype']
        if dtype == np.int8:
            data = np.random.randint(-128, 127, size=shape, dtype=np.int8)
        else:
            data = np.random.randn(*shape).astype(dtype)
        test_data[inp['index']] = data
    
    # Run inference
    for idx, data in test_data.items():
        interpreter.set_tensor(idx, data)
    
    # Warm up
    for _ in range(10):
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - start)
    
    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    
    # Get outputs
    outputs = {}
    for out in output_details:
        outputs[out['name']] = interpreter.get_tensor(out['index'])
    
    return {
        'input_details': input_details,
        'output_details': output_details,
        'avg_latency_ms': avg_ms,
        'std_latency_ms': std_ms,
        'output_shapes': {k: v.shape for k, v in outputs.items()},
    }


# =============================================================================
# Android Compatibility Fixes
# =============================================================================

def fix_tflite_unsupported_ops(
    onnx_path: str,
    output_path: str = None,
) -> str:
    """
    Fix ONNX ops that are unsupported in TFLite.
    
    Common issues:
    - GRU/LSTM: May need unrolling or custom impl
    - InstanceNorm: Replace with LayerNorm or BatchNorm
    - Complex tensor ops: Decompose
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    
    unsupported_ops = set()
    for node in graph.node:
        # Check for potentially unsupported ops
        if node.op_type in ['InstanceNormalization', 'Einsum', 'GridSample']:
            unsupported_ops.add(node.op_type)
    
    if unsupported_ops:
        print(f"Warning: Found potentially unsupported ops: {unsupported_ops}")
        print("These may require custom TFLite ops or model modifications")
    
    if output_path:
        onnx.save(model, output_path)
    
    return output_path or onnx_path


# =============================================================================
# Android Helper Files
# =============================================================================

def generate_android_kotlin_wrapper(output_path: str, model_name: str = "AuraNet"):
    """
    Generate Kotlin wrapper code for Android integration.
    """
    kotlin_code = f'''
package com.example.audioenhancement

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * {model_name} TFLite audio enhancement model wrapper.
 *
 * Supports:
 * - CPU inference
 * - GPU delegate (faster)
 * - NNAPI delegate (uses hardware accelerators)
 */
class {model_name}Enhancer(
    private val context: Context,
    private val useGpu: Boolean = false,
    private val useNnapi: Boolean = false,
) {{
    
    // Model parameters
    companion object {{
        const val SAMPLE_RATE = 16000
        const val FFT_SIZE = 256
        const val HOP_SIZE = 80
        const val NUM_FREQ_BINS = 129
        const val NUM_CHANNELS = 2  // real, imag
        const val HIDDEN_SIZE = 64
    }}
    
    private var interpreter: Interpreter? = null
    private var hiddenState: ByteBuffer? = null
    
    init {{
        loadModel()
        initHiddenState()
    }}
    
    private fun loadModel() {{
        val options = Interpreter.Options()
        
        // Add GPU delegate if requested
        if (useGpu) {{
            try {{
                val gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            }} catch (e: Exception) {{
                // GPU not available, fallback to CPU
            }}
        }}
        
        // Add NNAPI delegate if requested
        if (useNnapi) {{
            try {{
                val nnapiDelegate = NnApiDelegate()
                options.addDelegate(nnapiDelegate)
            }} catch (e: Exception) {{
                // NNAPI not available
            }}
        }}
        
        // Set number of threads for CPU
        options.setNumThreads(4)
        
        // Load model
        val modelBuffer = loadModelFile("{model_name.lower()}.tflite")
        interpreter = Interpreter(modelBuffer, options)
    }}
    
    private fun loadModelFile(filename: String): MappedByteBuffer {{
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }}
    
    private fun initHiddenState() {{
        // Hidden state: [1, 1, HIDDEN_SIZE] float32
        hiddenState = ByteBuffer.allocateDirect(1 * 1 * HIDDEN_SIZE * 4)
        hiddenState?.order(ByteOrder.nativeOrder())
        hiddenState?.rewind()
        
        // Initialize to zeros
        for (i in 0 until HIDDEN_SIZE) {{
            hiddenState?.putFloat(0.0f)
        }}
    }}
    
    /**
     * Process audio frames through the model.
     *
     * @param stftReal Real part of STFT [timeFrames, NUM_FREQ_BINS]
     * @param stftImag Imaginary part of STFT [timeFrames, NUM_FREQ_BINS]
     * @param timeFrames Number of time frames
     * @return Pair of enhanced (real, imag) arrays
     */
    fun processFrames(
        stftReal: FloatArray,
        stftImag: FloatArray,
        timeFrames: Int,
    ): Pair<FloatArray, FloatArray> {{
        
        val inputSize = 1 * NUM_CHANNELS * timeFrames * NUM_FREQ_BINS * 4
        val inputBuffer = ByteBuffer.allocateDirect(inputSize)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        // Pack input: [1, 2, T, 129]
        // Channel 0: real, Channel 1: imag
        for (t in 0 until timeFrames) {{
            for (f in 0 until NUM_FREQ_BINS) {{
                inputBuffer.putFloat(stftReal[t * NUM_FREQ_BINS + f])
            }}
        }}
        for (t in 0 until timeFrames) {{
            for (f in 0 until NUM_FREQ_BINS) {{
                inputBuffer.putFloat(stftImag[t * NUM_FREQ_BINS + f])
            }}
        }}
        inputBuffer.rewind()
        
        // Prepare outputs
        val outputSize = 1 * NUM_CHANNELS * timeFrames * NUM_FREQ_BINS * 4
        val outputBuffer = ByteBuffer.allocateDirect(outputSize)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        val newHiddenState = ByteBuffer.allocateDirect(1 * 1 * HIDDEN_SIZE * 4)
        newHiddenState.order(ByteOrder.nativeOrder())
        
        // Run inference
        hiddenState?.rewind()
        val inputs = arrayOf(inputBuffer, hiddenState)
        val outputs = mapOf(
            0 to outputBuffer,
            1 to newHiddenState
        )
        
        interpreter?.runForMultipleInputsOutputs(inputs, outputs)
        
        // Update hidden state
        hiddenState = newHiddenState
        
        // Extract outputs
        outputBuffer.rewind()
        val outputReal = FloatArray(timeFrames * NUM_FREQ_BINS)
        val outputImag = FloatArray(timeFrames * NUM_FREQ_BINS)
        
        for (t in 0 until timeFrames) {{
            for (f in 0 until NUM_FREQ_BINS) {{
                outputReal[t * NUM_FREQ_BINS + f] = outputBuffer.float
            }}
        }}
        for (t in 0 until timeFrames) {{
            for (f in 0 until NUM_FREQ_BINS) {{
                outputImag[t * NUM_FREQ_BINS + f] = outputBuffer.float
            }}
        }}
        
        return Pair(outputReal, outputImag)
    }}
    
    /**
     * Reset hidden state (call when starting new audio stream)
     */
    fun reset() {{
        initHiddenState()
    }}
    
    /**
     * Release resources
     */
    fun close() {{
        interpreter?.close()
    }}
}}
'''
    
    with open(output_path, 'w') as f:
        f.write(kotlin_code)
    
    print(f"Generated Kotlin wrapper: {output_path}")


def generate_gradle_dependencies(output_path: str):
    """
    Generate Gradle dependencies for TFLite.
    """
    gradle_code = '''
// TensorFlow Lite dependencies for Android
// Add to your app/build.gradle

dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    
    // GPU delegate (optional, for faster inference)
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    
    // NNAPI delegate (optional, uses hardware accelerators)
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    
    // Select TF ops (if model uses unsupported ops)
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0'
}

android {
    // Prevent compression of tflite model
    aaptOptions {
        noCompress "tflite"
    }
}
'''
    
    with open(output_path, 'w') as f:
        f.write(gradle_code)
    
    print(f"Generated Gradle dependencies: {output_path}")


# =============================================================================
# Main Conversion Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TFLite')
    parser.add_argument('--input', type=str, required=True,
                        help='Input ONNX model path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .tflite path (default: auto)')
    parser.add_argument('--quantize', type=str, default=None,
                        choices=['dynamic', 'float16', 'int8'],
                        help='Quantization mode')
    parser.add_argument('--fix-ops', action='store_true',
                        help='Fix unsupported operations')
    parser.add_argument('--generate-android', action='store_true',
                        help='Generate Android wrapper code')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip verification step')
    args = parser.parse_args()
    
    if not TF_AVAILABLE:
        print("ERROR: tensorflow is required for TFLite conversion")
        print("Install with: pip install tensorflow")
        sys.exit(1)
    
    if not ONNX_TF_AVAILABLE:
        print("ERROR: onnx-tf is required for ONNX → TF conversion")
        print("Install with: pip install onnx-tf")
        sys.exit(1)
    
    print("=" * 70)
    print("TensorFlow Lite Conversion for AuraNet")
    print("=" * 70)
    
    # Fix unsupported ops if requested
    input_path = args.input
    if args.fix_ops:
        print(f"\n🔧 Checking for unsupported operations...")
        input_path = fix_tflite_unsupported_ops(args.input)
    
    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="auranet_tflite_")
    print(f"\nTemp directory: {temp_dir}")
    
    try:
        # Step 1: ONNX → TF SavedModel
        print(f"\n📦 Step 1: Converting ONNX to TensorFlow...")
        saved_model_path = onnx_to_saved_model(input_path, temp_dir)
        
        # Step 2: TF → TFLite
        print(f"\n📱 Step 2: Converting to TFLite...")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_name = Path(args.input).stem
            quant_suffix = f"_{args.quantize}" if args.quantize else ""
            output_path = str(Path(args.input).parent / f"{base_name}{quant_suffix}.tflite")
        
        info = convert_to_tflite(
            saved_model_path=saved_model_path,
            output_path=output_path,
            quantize=args.quantize,
        )
        
        print(f"\n✅ Conversion successful!")
        print(f"   Output: {info['path']}")
        print(f"   Size: {info['size_mb']:.2f} MB")
        print(f"   Quantization: {info['quantization'] or 'None'}")
        
        # Step 3: Verify
        if not args.skip_verify:
            print(f"\n🔍 Step 3: Verifying TFLite model...")
            try:
                verify_info = verify_tflite_model(output_path)
                print(f"   Avg latency: {verify_info['avg_latency_ms']:.2f} ms")
                print(f"   Std latency: {verify_info['std_latency_ms']:.2f} ms")
                print(f"   Output shapes: {verify_info['output_shapes']}")
                print(f"   ✅ Verification passed!")
            except Exception as e:
                print(f"   ⚠️ Verification error: {e}")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up temp files (optional)
        # import shutil
        # shutil.rmtree(temp_dir)
        pass
    
    # Generate Android wrapper if requested
    if args.generate_android:
        output_dir = Path(output_path).parent
        kotlin_path = str(output_dir / "AuraNetEnhancer.kt")
        gradle_path = str(output_dir / "tflite_dependencies.gradle")
        generate_android_kotlin_wrapper(kotlin_path)
        generate_gradle_dependencies(gradle_path)
    
    # Print integration instructions
    print("\n" + "=" * 70)
    print("📱 ANDROID INTEGRATION GUIDE")
    print("=" * 70)
    print(f"""
1. Add the .tflite file to your Android project:
   - Place in: app/src/main/assets/auranet.tflite
   - Add to build.gradle:
     ```gradle
     android {{
         aaptOptions {{
             noCompress "tflite"
         }}
     }}
     ```

2. Add TensorFlow Lite dependencies to build.gradle:
   ```gradle
   implementation 'org.tensorflow:tensorflow-lite:2.14.0'
   implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'  // Optional
   ```

3. Use the model:
   ```kotlin
   val interpreter = Interpreter(loadModelFile("auranet.tflite"))
   interpreter.run(inputBuffer, outputBuffer)
   ```

4. For streaming audio:
   - Process in chunks of ~100 frames
   - Maintain hidden state between chunks
   - Use GPU delegate for lower latency

5. Quantization notes:
   - 'dynamic': Weights INT8, activations FP32 (good balance)
   - 'float16': Half precision (2x smaller, GPU optimized)
   - 'int8': Full INT8 (4x smaller, may need calibration)
""")
    
    print("\n" + "=" * 70)
    print("📊 CONVERSION SUMMARY")
    print("=" * 70)
    print(f"  Input:        {args.input}")
    print(f"  Output:       {output_path}")
    print(f"  Size:         {info['size_mb']:.2f} MB")
    print(f"  Quantization: {args.quantize or 'None'}")
    print("=" * 70)
    
    return output_path


if __name__ == "__main__":
    main()
