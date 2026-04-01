import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer import AuraNetInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained AuraNet model locally.")
    parser.add_argument(
        "--model",
        default="checkpoints/best_model.pt",
        help="Path to a trained model file or checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for loading the model.",
    )
    parser.add_argument(
        "--show-arch",
        action="store_true",
        help="Print the full model architecture after loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    enhancer = AuraNetInference(model_path=str(model_path), device=args.device)

    parameter_count = sum(parameter.numel() for parameter in enhancer.model.parameters())

    print("\nModel loaded successfully")
    print(f"Path: {model_path}")
    print(f"Device: {enhancer.device}")
    print(f"Sample rate: {enhancer.sample_rate}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Model class: {enhancer.model.__class__.__name__}")

    if args.show_arch:
        print("\nArchitecture:\n")
        print(enhancer.model)


if __name__ == "__main__":
    main()
