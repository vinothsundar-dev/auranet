#!/usr/bin/env python3
"""Enhance a noisy audio file using trained AuraNet model."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import torch
import torchaudio
from infer import AuraNetInference


def main():
    parser = argparse.ArgumentParser(description="Enhance noisy audio with AuraNet")
    parser.add_argument("input", help="Path to noisy audio file (.wav)")
    parser.add_argument("--model", default="checkpoints/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--output", default=None, help="Output path (default: input_enhanced.wav)")
    parser.add_argument("--device", default=None, help="Device (auto-detected if omitted)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"

    # Load model
    enhancer = AuraNetInference(model_path=str(model_path), device=args.device)

    # Enhance
    print(f"🔄 Processing: {input_path}")
    enhanced = enhancer.process_file(str(input_path), str(output_path))

    print(f"✅ Saved: {output_path}")
    print(f"   Duration: {enhanced.shape[-1] / 16000:.1f}s")


if __name__ == "__main__":
    main()
