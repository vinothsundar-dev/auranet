#!/usr/bin/env python3
# =============================================================================
# AuraNet V3 Export — ONNX + TFLite + INT8 Quantization
# =============================================================================

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from model_v3 import AuraNetV3, create_auranet_v3


def export_onnx(model, config, output_path="deploy/exports/auranet_v3.onnx",
                opset=17, streaming=False):
    """
    Export model to ONNX.

    Two modes:
    - Stateless (streaming=False): Full sequence input
    - Streaming (streaming=True): Single-frame with hidden state I/O
    """
    model.eval()

    stft_cfg = config.get("stft", {})
    n_fft = stft_cfg.get("n_fft", 256)
    freq_bins = n_fft // 2 + 1

    if streaming:
        # Single-frame ONNX export with hidden state
        gru_cfg = config.get("model", {}).get("bottleneck", {})
        h = gru_cfg.get("hidden_size", 128)
        n_layers = gru_cfg.get("num_layers", 2)

        dummy_stft = torch.randn(1, 2, 1, freq_bins)
        dummy_hidden = torch.zeros(n_layers, 1, h)

        # Wrapper that exposes hidden state as explicit I/O
        class StreamingWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, noisy_stft, hidden_in):
                enhanced, hidden_out, _ = self.model(noisy_stft, hidden_in)
                return enhanced, hidden_out

        wrapper = StreamingWrapper(model)
        wrapper.eval()

        output_path = str(output_path).replace(".onnx", "_streaming.onnx")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            wrapper,
            (dummy_stft, dummy_hidden),
            output_path,
            opset_version=opset,
            input_names=["noisy_stft", "hidden_in"],
            output_names=["enhanced_stft", "hidden_out"],
            dynamic_axes={
                "noisy_stft": {0: "batch"},
                "hidden_in": {1: "batch"},
                "enhanced_stft": {0: "batch"},
                "hidden_out": {1: "batch"},
            },
        )
        print(f"✅ ONNX streaming exported: {output_path}")

    else:
        # Full sequence
        T = 100
        dummy_stft = torch.randn(1, 2, T, freq_bins)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_stft,
            output_path,
            opset_version=opset,
            input_names=["noisy_stft"],
            output_names=["enhanced_stft"],
            dynamic_axes={
                "noisy_stft": {0: "batch", 2: "time"},
                "enhanced_stft": {0: "batch", 2: "time"},
            },
        )
        print(f"✅ ONNX stateless exported: {output_path}")

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"   Size: {size_mb:.1f} MB")
    return output_path


def quantize_int8(model_path, output_path=None):
    """Post-training dynamic INT8 quantization."""
    model = torch.load(model_path, map_location="cpu", weights_only=True)

    if output_path is None:
        output_path = str(model_path).replace(".pt", "_int8.pt")

    # Dynamic quantization targets: Linear and GRU layers
    quantized = torch.quantization.quantize_dynamic(
        model if isinstance(model, nn.Module) else None,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )

    if quantized is not None:
        torch.save(quantized.state_dict(), output_path)
        print(f"✅ INT8 quantized model: {output_path}")
    else:
        print("⚠️  Quantization requires a model instance, not a state_dict.")
        print("   Load the model first, then quantize.")

    return output_path


def export_torchscript(model, output_path="deploy/exports/auranet_v3_traced.pt"):
    """Export as TorchScript (traced)."""
    model.eval()
    model = model.to("cpu")

    stft_bins = 129
    dummy = torch.randn(1, 2, 100, stft_bins)

    traced = torch.jit.trace(model, dummy)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(output_path)

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"✅ TorchScript exported: {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export AuraNet V3")
    parser.add_argument("--model", default="checkpoints/best_model_v3.pt")
    parser.add_argument("--config", default="config_v3.yaml")
    parser.add_argument("--format", choices=["onnx", "torchscript", "all"], default="all")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    # Config
    cfg_path = Path(args.config)
    config = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = yaml.safe_load(f) or {}

    # Load model
    model = create_auranet_v3(config)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    out_dir = Path("deploy/exports")

    if args.format in ("onnx", "all"):
        export_onnx(model, config, str(out_dir / "auranet_v3.onnx"))
        if args.streaming:
            export_onnx(model, config, str(out_dir / "auranet_v3.onnx"), streaming=True)

    if args.format in ("torchscript", "all"):
        export_torchscript(model, str(out_dir / "auranet_v3_traced.pt"))

    if args.quantize:
        print("\n--- INT8 Quantization ---")
        quantize_int8(args.model)


if __name__ == "__main__":
    main()
