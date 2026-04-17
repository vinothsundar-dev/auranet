# =============================================================================
# Training Script for AuraNet
# =============================================================================
#
# TRAINING PIPELINE:
# 1. Load configuration
# 2. Initialize model, optimizer, scheduler
# 3. Setup data loaders
# 4. Training loop with:
#    - Mixed precision (AMP) for efficiency
#    - Gradient clipping for stability
#    - Multi-task loss computation
#    - Checkpointing
# 5. Optional: Quantization-aware training (QAT)
#
# OPTIMIZATION STRATEGIES:
# - AdamW optimizer with weight decay
# - Cosine annealing learning rate
# - Gradient clipping to prevent exploding gradients
# - Mixed precision for faster training on GPU
# =============================================================================

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Any
import yaml
import time
from datetime import datetime

# =============================================================================
# Performance Configuration - MUST be before torch imports
# =============================================================================
# Disable CUDA_LAUNCH_BLOCKING for performance (don't sync after every kernel)
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# Disable cuDNN benchmark to avoid warmup stalls
# benchmark=True can cause long initial delays while cuDNN searches for optimal algorithms
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

# Disable torch._dynamo completely to avoid hidden compilation overhead
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()
except (ImportError, AttributeError):
    pass  # Older PyTorch version without dynamo

from model import AuraNet, create_auranet
from loss import AuraNetLoss
from dataset import AuraNetDataset, create_dataloader
from utils.stft import CausalSTFT


class Trainer:
    """
    AuraNet Trainer class.

    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    - QAT preparation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[AuraNet] = None,
        device: str = "auto",
    ):
        """
        Args:
            config: Configuration dictionary
            model: Optional pre-initialized model
            device: Device to train on ("auto", "cuda", "cpu", "mps")
        """
        self.config = config

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
        if model is not None:
            self.model = model
        else:
            self.model = create_auranet(config)

        self.model = self.model.to(self.device)

        # Initialize STFT for reconstruction
        stft_config = config.get("stft", {})
        self.stft = CausalSTFT(
            n_fft=stft_config.get("n_fft", 256),
            hop_length=stft_config.get("hop_size", 80),
            win_length=stft_config.get("window_size", 160),
        ).to(self.device)

        # Initialize loss function
        loss_config = config.get("loss", {})
        self.criterion = AuraNetLoss(
            weight_complex_mse=loss_config.get("complex_mse", 1.0),
            weight_stft=loss_config.get("multi_resolution_stft", 0.5),
            weight_loudness=loss_config.get("loudness_envelope", 0.3),
            weight_temporal=loss_config.get("temporal_coherence", 0.2),
        )

        # Initialize optimizer
        train_config = config.get("training", {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config.get("learning_rate", 0.0001),  # FIXED: default 1e-4
            weight_decay=train_config.get("weight_decay", 0.01),
        )

        # Learning rate scheduler
        self.num_epochs = train_config.get("num_epochs", 100)
        scheduler_type = train_config.get("lr_scheduler", "cosine")

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6,
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        else:
            self.scheduler = None

        # Mixed precision scaler
        self.use_amp = train_config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.grad_clip = train_config.get("gradient_clip", 5.0)

        # Checkpointing
        self.checkpoint_dir = Path(train_config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = train_config.get("save_every_n_epochs", 5)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # QAT settings
        qat_config = train_config.get("qat", {})
        self.use_qat = qat_config.get("enabled", False)
        self.qat_start_epoch = qat_config.get("start_epoch", 50)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns dictionary of average losses for the epoch.
        """
        self.model.train()

        epoch_losses = {
            "total": 0.0,
            "complex_mse": 0.0,
            "multi_res_stft": 0.0,
            "loudness_envelope": 0.0,
            "temporal_coherence": 0.0,
        }
        num_batches = 0

        # Use tqdm with mininterval=0 to ensure updates from first batch
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {self.current_epoch + 1}",
            mininterval=0,  # Update immediately from first batch
            leave=True
        )

        for batch_idx, batch in pbar:
            # Move to device
            noisy_stft = batch["noisy_stft"].to(self.device)
            clean_stft = batch["clean_stft"].to(self.device)
            noisy_audio = batch["noisy_audio"].to(self.device)
            clean_audio = batch["clean_audio"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    enhanced_stft, wdrc_params, _ = self.model(noisy_stft)

                    # Reconstruct audio for time-domain losses
                    enhanced_audio = self.stft.inverse(enhanced_stft)

                    # Compute loss
                    loss, loss_dict = self.criterion(
                        enhanced_stft, clean_stft,
                        enhanced_audio, clean_audio,
                    )

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                enhanced_stft, wdrc_params, _ = self.model(noisy_stft)

                # Reconstruct audio
                enhanced_audio = self.stft.inverse(enhanced_stft)

                # Compute loss
                loss, loss_dict = self.criterion(
                    enhanced_stft, clean_stft,
                    enhanced_audio, clean_audio,
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()

            num_batches += 1
            self.global_step += 1

            # Update tqdm progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Returns dictionary of average validation losses.
        """
        self.model.eval()

        val_losses = {
            "total": 0.0,
            "complex_mse": 0.0,
            "multi_res_stft": 0.0,
            "loudness_envelope": 0.0,
            "temporal_coherence": 0.0,
        }
        num_batches = 0

        for batch in val_loader:
            noisy_stft = batch["noisy_stft"].to(self.device)
            clean_stft = batch["clean_stft"].to(self.device)
            noisy_audio = batch["noisy_audio"].to(self.device)
            clean_audio = batch["clean_audio"].to(self.device)

            # Forward pass
            enhanced_stft, wdrc_params, _ = self.model(noisy_stft)
            enhanced_audio = self.stft.inverse(enhanced_stft)

            # Compute loss
            loss, loss_dict = self.criterion(
                enhanced_stft, clean_stft,
                enhanced_audio, clean_audio,
            )

            for key in val_losses:
                if key in loss_dict:
                    val_losses[key] += loss_dict[key].item()

            num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        return val_losses

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def prepare_qat(self):
        """
        Prepare model for Quantization-Aware Training.

        QAT simulates INT8 quantization during training, allowing the model
        to adapt to quantization effects and maintain accuracy after deployment.
        """
        print("Preparing model for Quantization-Aware Training (QAT)...")

        # Configure QAT
        self.model.train()

        # Fuse modules for better quantization
        # (Conv + BatchNorm + ReLU -> single fused module)
        torch.quantization.fuse_modules(
            self.model,
            [["encoder.blocks.0.conv.pointwise",
              "encoder.blocks.0.norm",
              "encoder.blocks.0.activation"]],
            inplace=True
        )

        # Prepare for QAT
        qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        self.model.qconfig = qconfig

        torch.quantization.prepare_qat(self.model, inplace=True)

        print("QAT preparation complete.")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        print("\n" + "=" * 60)
        print("Starting AuraNet Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"QAT enabled: {self.use_qat}")
        print("=" * 60 + "\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)

            # Enable QAT at specified epoch
            if self.use_qat and epoch == self.qat_start_epoch:
                self.prepare_qat()

            # Training
            train_losses = self.train_epoch(train_loader)

            # Validation
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                is_best = val_losses["total"] < self.best_val_loss

                if is_best:
                    self.best_val_loss = val_losses["total"]
            else:
                val_losses = None
                is_best = train_losses["total"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = train_losses["total"]

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric = val_losses["total"] if val_losses else train_losses["total"]
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            # Logging
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            if val_losses:
                print(f"  Val Loss:   {val_losses['total']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.1f}s")

            if is_best:
                print("  *** New best model! ***")

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", is_best)
            elif is_best:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", is_best=True)

        # Final checkpoint
        self.save_checkpoint("final_model.pt")

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train AuraNet model")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--clean-dir", type=str, default=None,
                        help="Directory with clean audio files")
    parser.add_argument("--noise-dir", type=str, default=None,
                        help="Directory with noise audio files")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu, mps)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {}

    # Override config with command line args
    if args.epochs:
        config.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    # Get data config
    data_config = config.get("data", {})
    train_config = config.get("training", {})

    # Determine data source
    clean_dir = args.clean_dir or data_config.get("train_clean_dir")
    noise_dir = args.noise_dir or data_config.get("train_noise_dir")
    use_synthetic = args.synthetic or (clean_dir is None and noise_dir is None)

    if use_synthetic:
        print("\nUsing synthetic data for training...")

    # Create data loaders
    train_loader = create_dataloader(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        batch_size=train_config.get("batch_size", 16),
        num_workers=data_config.get("num_workers", 4),
        segment_length=data_config.get("segment_length", 3.0),
        snr_range=tuple(data_config.get("snr_range", [-5, 20])),
        synthetic=use_synthetic,
    )

    # Validation loader (optional)
    val_clean_dir = data_config.get("val_clean_dir")
    val_noise_dir = data_config.get("val_noise_dir")

    if val_clean_dir and val_noise_dir:
        val_loader = create_dataloader(
            clean_dir=val_clean_dir,
            noise_dir=val_noise_dir,
            batch_size=train_config.get("batch_size", 16),
            num_workers=data_config.get("num_workers", 4),
            shuffle=False,
        )
    else:
        # Use subset of training data for validation in synthetic mode
        val_loader = create_dataloader(
            batch_size=train_config.get("batch_size", 16),
            num_workers=0,
            synthetic=True,
        )

    # Initialize trainer
    trainer = Trainer(config, device=args.device)

    # Start training
    trainer.train(
        train_loader,
        val_loader,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
