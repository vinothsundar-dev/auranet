#!/usr/bin/env python3
# =============================================================================
# AuraNet V3 Training Script
# =============================================================================
#
# Improvements over train.py:
# 1. ReduceLROnPlateau scheduler (responds to actual plateau)
# 2. Warmup phase (first 5 epochs linear ramp-up)
# 3. Train/validation split (90/10)
# 4. Early stopping (patience=15)
# 5. Per-epoch detailed logging with loss breakdown
# 6. Exponential moving average (EMA) of model weights
# 7. Proper validation with SI-SNR metric
#
# Usage:
#   python train_v3.py --config config_v3.yaml
#   python train_v3.py --clean-dir datasets/speech --noise-dir datasets/noise
#   python train_v3.py --synthetic --epochs 50
# =============================================================================

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional
from copy import deepcopy

# =============================================================================
# Performance Configuration - MUST be before torch imports
# =============================================================================
# Disable CUDA_LAUNCH_BLOCKING for performance (don't sync after every kernel)
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

from model_v3 import AuraNetV3, create_auranet_v3
from loss_v3 import AuraNetV3Loss
from dataset_v3 import AuraNetV3Dataset, create_v3_dataloader
from utils.stft import CausalSTFT


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                       for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model):
        """Copy EMA weights into model (for evaluation)."""
        self.backup = {name: p.clone() for name, p in model.named_parameters()
                       if p.requires_grad and name in self.shadow}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


class TrainerV3:
    """AuraNet V3 Trainer with improved training loop."""

    def __init__(self, config, device="auto"):
        self.config = config

        # Device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"Device: {self.device}")

        # Model
        self.model = create_auranet_v3(config).to(self.device)

        # STFT
        stft_cfg = config.get("stft", {})
        self.stft = CausalSTFT(
            n_fft=stft_cfg.get("n_fft", 256),
            hop_length=stft_cfg.get("hop_size", 80),
            win_length=stft_cfg.get("window_size", 160),
        ).to(self.device)

        # Loss
        loss_cfg = config.get("loss", {})
        self.criterion = AuraNetV3Loss(
            weight_sisnr=loss_cfg.get("si_snr", 1.0),
            weight_compressed_mse=loss_cfg.get("compressed_mse", 0.5),
            weight_stft=loss_cfg.get("multi_res_stft", 0.3),
            compress_factor=loss_cfg.get("compress_factor", 0.3),
        )

        # Optimizer
        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=train_cfg.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )

        # Scheduler — ReduceLROnPlateau
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=train_cfg.get("lr_patience", 5),
            min_lr=1e-6,
        )

        # Training config
        self.num_epochs = train_cfg.get("num_epochs", 100)
        self.grad_clip = train_cfg.get("gradient_clip", 3.0)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.patience = train_cfg.get("early_stop_patience", 15)

        # AMP (CUDA only) — disabled by default for stability
        self.use_amp = train_cfg.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # EMA
        self.ema = EMA(self.model, decay=0.999)

        # Checkpointing
        self.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.no_improve_count = 0

    def _warmup_lr(self, epoch):
        """Linear warmup for first N epochs."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr * warmup_factor

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        loss_sums = {}
        n = 0

        # Use tqdm with mininterval=0 to ensure updates from first batch
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {self.current_epoch + 1}",
            mininterval=0,  # Update immediately from first batch
            leave=True
        )

        for batch_idx, batch in pbar:
            noisy_stft = batch["noisy_stft"].to(self.device)
            clean_stft = batch["clean_stft"].to(self.device)
            clean_audio = batch["clean_audio"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    enhanced_stft, _, _ = self.model(noisy_stft)
                    enhanced_audio = self.stft.inverse(enhanced_stft)
                    # Numerical safety: clamp output
                    enhanced_audio = torch.nan_to_num(enhanced_audio, nan=0.0, posinf=1.0, neginf=-1.0)
                    enhanced_audio = torch.clamp(enhanced_audio, -1.0, 1.0)
                    enhanced_audio = enhanced_audio + 1e-6
                    loss, ld = self.criterion(enhanced_stft, clean_stft,
                                              enhanced_audio, clean_audio)

                # NaN/Inf guard
                if not torch.isfinite(loss):
                    print(f"  [WARN] Skipping batch {batch_idx} — non-finite loss")
                    self.optimizer.zero_grad()
                    continue

                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                enhanced_stft, _, _ = self.model(noisy_stft)
                enhanced_audio = self.stft.inverse(enhanced_stft)
                # Numerical safety: clamp output
                enhanced_audio = torch.nan_to_num(enhanced_audio, nan=0.0, posinf=1.0, neginf=-1.0)
                enhanced_audio = torch.clamp(enhanced_audio, -1.0, 1.0)
                enhanced_audio = enhanced_audio + 1e-6
                loss, ld = self.criterion(enhanced_stft, clean_stft,
                                          enhanced_audio, clean_audio)

                # NaN/Inf guard
                if not torch.isfinite(loss):
                    print(f"  [WARN] Skipping batch {batch_idx} — non-finite loss")
                    self.optimizer.zero_grad()
                    continue

                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.ema.update(self.model)
            total_loss += loss.item()
            for k, v in ld.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
            n += 1

            # Update tqdm progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg = {k: v / n for k, v in loss_sums.items()}
        avg["total"] = total_loss / n
        return avg

    @torch.no_grad()
    def validate(self, loader):
        # Use EMA weights for validation
        self.ema.apply(self.model)
        self.model.eval()

        total_loss = 0.0
        loss_sums = {}
        n = 0

        for batch in loader:
            noisy_stft = batch["noisy_stft"].to(self.device)
            clean_stft = batch["clean_stft"].to(self.device)
            clean_audio = batch["clean_audio"].to(self.device)

            enhanced_stft, _, _ = self.model(noisy_stft)
            enhanced_audio = self.stft.inverse(enhanced_stft)
            loss, ld = self.criterion(enhanced_stft, clean_stft,
                                      enhanced_audio, clean_audio)

            total_loss += loss.item()
            for k, v in ld.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
            n += 1

        self.ema.restore(self.model)
        avg = {k: v / max(n, 1) for k, v in loss_sums.items()}
        avg["total"] = total_loss / max(n, 1)
        return avg

    def save_checkpoint(self, tag, is_best=False):
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_shadow": self.ema.shadow,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(ckpt, path)
        if is_best:
            # Save EMA weights as best_model (these are typically better)
            self.ema.apply(self.model)
            torch.save(self.model.state_dict(), self.ckpt_dir / "best_model_v3.pt")
            self.ema.restore(self.model)
            print(f"  💾 Best model saved (val_loss={self.best_val_loss:.4f})")

    def resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_shadow" in ckpt:
            self.ema.shadow = ckpt["ema_shadow"]
        self.current_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {self.current_epoch}")

    def train(self, train_loader, val_loader=None):
        print(f"\n{'='*60}")
        print(f"AuraNet V3 Training")
        print(f"{'='*60}")
        print(f"  Epochs:    {self.num_epochs}")
        print(f"  LR:        {self.lr}")
        print(f"  Warmup:    {self.warmup_epochs} epochs")
        print(f"  Patience:  {self.patience}")
        print(f"  AMP:       {self.use_amp}")
        print(f"  Device:    {self.device}")
        print(f"  Batches:   {len(train_loader)}")
        if val_loader:
            print(f"  Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            # Warmup LR
            self._warmup_lr(epoch)

            # Train
            train_losses = self.train_epoch(train_loader)
            elapsed = time.time() - t0

            # Validate
            val_losses = None
            if val_loader:
                val_losses = self.validate(val_loader)

            # LR scheduling (on validation loss if available, else train)
            metric = val_losses["total"] if val_losses else train_losses["total"]
            if epoch >= self.warmup_epochs:
                self.scheduler.step(metric)

            # Logging
            lr_now = self.optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch+1}/{self.num_epochs} ({elapsed:.1f}s) | LR={lr_now:.2e}")
            print(f"  Train: total={train_losses['total']:.4f} "
                  f"si_snr={train_losses.get('si_snr', 0):.4f} "
                  f"cmse={train_losses.get('compressed_mse', 0):.4f} "
                  f"stft={train_losses.get('multi_res_stft', 0):.4f}")
            if val_losses:
                print(f"  Val:   total={val_losses['total']:.4f} "
                      f"si_snr={val_losses.get('si_snr', 0):.4f} "
                      f"cmse={val_losses.get('compressed_mse', 0):.4f} "
                      f"stft={val_losses.get('multi_res_stft', 0):.4f}")

            # Checkpointing
            is_best = metric < self.best_val_loss
            if is_best:
                self.best_val_loss = metric
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            self.save_checkpoint("latest_v3", is_best=is_best)
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_v3_epoch_{epoch+1}")

            # Early stopping
            if self.no_improve_count >= self.patience:
                print(f"\n⏹  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {self.patience} epochs)")
                break

        print(f"\n✅ Training complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"   Best model: {self.ckpt_dir / 'best_model_v3.pt'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train AuraNet V3")
    parser.add_argument("--config", type=str, default="config_v3.yaml")
    parser.add_argument("--clean-dir", type=str, default=None)
    parser.add_argument("--noise-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        print(f"Config loaded: {config_path}")
    else:
        config = {}
        print("No config file found, using defaults")

    # CLI overrides
    if args.epochs:
        config.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr

    # Data paths
    train_cfg = config.get("training", {})
    clean_dir = args.clean_dir or train_cfg.get("clean_dir", "datasets/speech")
    noise_dir = args.noise_dir or train_cfg.get("noise_dir", "datasets/noise")
    batch_size = train_cfg.get("batch_size", 16)
    synthetic = args.synthetic or not (Path(clean_dir).exists() and Path(noise_dir).exists())

    if synthetic:
        print("Using synthetic data")

    # Create dataset
    full_dataset = AuraNetV3Dataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        sample_rate=config.get("audio", {}).get("sample_rate", 16000),
        segment_length=train_cfg.get("segment_length", 3.0),
        augment=True,
        synthetic_mode=synthetic,
        num_synthetic_samples=train_cfg.get("num_synthetic_samples", 2000),
    )

    # Train/val split (90/10)
    total = len(full_dataset)
    val_size = max(1, int(total * 0.1))
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=1, pin_memory=True)

    print(f"Dataset: {train_size} train / {val_size} val samples")

    # Trainer
    trainer = TrainerV3(config, device=args.device)

    if args.resume:
        trainer.resume(args.resume)

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
