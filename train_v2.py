#!/usr/bin/env python3
"""
AuraNet V2 Training Script with 2-Stage Training.

This is a NEW training script that works alongside the existing train.py,
specifically designed for the V2 model with deep filtering and loud-loss.

TRAINING STRATEGY:
==================

Stage 1: Separation Training
- Focus: Deep filtering quality, noise reduction
- Loss: Loud-Loss + SI-SDR + Multi-Res STFT
- Duration: ~80% of total epochs
- WDRC: Frozen or low learning rate

Stage 2: WDRC Fine-tuning
- Focus: Dynamic range, loudness consistency
- Loss: Add envelope loss, reduce separation weights
- Duration: ~20% of total epochs
- WDRC: Unfrozen with full learning rate

USAGE:
======
# Full 2-stage training
python train_v2.py --data-dir datasets --epochs 100

# Stage 1 only
python train_v2.py --data-dir datasets --stage 1 --epochs 80

# Stage 2 from checkpoint
python train_v2.py --stage 2 --checkpoint checkpoints/stage1_best.pt

# Use V1 model (backward compatibility)
python train_v2.py --model v1
"""

import os
import sys
import argparse
import yaml
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import AuraNet, AuraNetV2, create_auranet, create_auranet_v2
from loss import AuraNetLoss, AuraNetV2Loss
from dataset import AuraNetDataset
from utils.stft import CausalSTFT


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "model": {
        "version": "v2",  # "v1" or "v2"
        "encoder_channels": [16, 32, 64, 128],
        "gru_hidden": 256,
        "gru_layers": 1,
        "decoder_channels": [64, 32, 16],
        "filter_order": 2,
        "use_physics_conditioning": True,
        "use_deep_filtering": True,
    },
    "training": {
        "batch_size": 16,
        "epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "grad_clip": 5.0,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        # 2-Stage settings
        "stage1_epochs_ratio": 0.8,  # 80% for separation
        "stage2_wdrc_lr_mult": 1.0,  # WDRC learning rate multiplier in stage 2
    },
    "data": {
        "sample_rate": 16000,
        "segment_length": 3.0,
        "snr_range": [-5, 20],
    },
    "loss": {
        "weight_loud": 1.0,
        "weight_si_sdr": 0.5,
        "weight_stft": 0.3,
        "weight_temporal": 0.1,
        "weight_phase": 0.1,
        "loudness_weighting": "a_weighting",
    },
    "checkpoint": {
        "save_dir": "checkpoints",
        "save_every": 5,
    },
}


# =============================================================================
# Learning Rate Scheduler with Warmup
# =============================================================================

class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch: int) -> float:
        """Update learning rate based on epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = max(self.base_lrs[i] * scale, self.min_lr)
        
        return self.optimizer.param_groups[0]['lr']


# =============================================================================
# V2 Trainer with 2-Stage Training
# =============================================================================

class TrainerV2:
    """
    AuraNet V2 Trainer with 2-Stage Training Support.
    
    FEATURES:
    - Model creation (V1 or V2)
    - Psychoacoustic loss computation
    - 2-stage training logic
    - Mixed precision training
    - Checkpointing and resume
    
    2-STAGE TRAINING:
    Stage 1 (80% of epochs): Focus on separation with loud-loss + SI-SDR
    Stage 2 (20% of epochs): Fine-tune WDRC with envelope loss
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_dir: str,
        device: torch.device,
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.device = device
        
        # Create model
        self.model = self._create_model()
        self.model.to(device)
        
        # Create STFT module for audio reconstruction
        self.stft = CausalSTFT(
            n_fft=256,
            hop_length=80,
            win_length=160,
        ).to(device)
        
        # Create loss function
        self.criterion = self._create_criterion()
        
        # Create optimizer with parameter groups (separate WDRC)
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config["training"]["warmup_epochs"],
            total_epochs=config["training"]["epochs"],
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = torch.cuda.is_available()
        
        # Training state
        self.current_epoch = 0
        self.current_stage = 1
        self.best_loss = float('inf')
        
        # Create data loaders
        self.train_loader, self.val_loader = self._create_data_loaders(data_dir)
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
            
    def _create_model(self) -> nn.Module:
        """Create model based on config version."""
        version = self.config["model"].get("version", "v2")
        
        if version == "v2":
            model = AuraNetV2(
                encoder_channels=tuple(self.config["model"]["encoder_channels"]),
                gru_hidden=self.config["model"]["gru_hidden"],
                gru_layers=self.config["model"]["gru_layers"],
                decoder_channels=tuple(self.config["model"]["decoder_channels"]),
                filter_order=self.config["model"].get("filter_order", 2),
                use_physics_conditioning=self.config["model"].get("use_physics_conditioning", True),
                use_deep_filtering=self.config["model"].get("use_deep_filtering", True),
            )
            print(f"✅ Created AuraNet V2 with {model.count_parameters():,} parameters")
            print(f"   - Deep Filtering: {self.config['model'].get('use_deep_filtering', True)}")
            print(f"   - Physics Conditioning: {self.config['model'].get('use_physics_conditioning', True)}")
            print(f"   - Filter Order: K={self.config['model'].get('filter_order', 2)}")
        else:
            model = create_auranet(self.config)
            print(f"✅ Created AuraNet V1 with {model.count_parameters():,} parameters")
            
        return model
        
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on config."""
        version = self.config["model"].get("version", "v2")
        
        if version == "v2":
            criterion = AuraNetV2Loss(
                weight_loud=self.config["loss"]["weight_loud"],
                weight_si_sdr=self.config["loss"]["weight_si_sdr"],
                weight_stft=self.config["loss"]["weight_stft"],
                weight_temporal=self.config["loss"]["weight_temporal"],
                weight_phase=self.config["loss"]["weight_phase"],
                sample_rate=self.config["data"]["sample_rate"],
                loudness_weighting=self.config["loss"]["loudness_weighting"],
            )
            print(f"✅ Using V2 Loss (Psychoacoustic Loud-Loss + SI-SDR)")
        else:
            criterion = AuraNetLoss()
            print(f"✅ Using V1 Loss (Complex MSE + Multi-Res STFT)")
            
        return criterion
        
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer with separate parameter groups.
        
        WDRC parameters are grouped separately for stage-2 fine-tuning.
        """
        wdrc_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if "wdrc" in name.lower():
                wdrc_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {"params": other_params, "lr": self.config["training"]["learning_rate"]},
            {"params": wdrc_params, "lr": self.config["training"]["learning_rate"], "name": "wdrc"},
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config["training"]["weight_decay"],
        )
        
        print(f"✅ Optimizer: AdamW (LR={self.config['training']['learning_rate']})")
        
        return optimizer
        
    def _create_data_loaders(
        self,
        data_dir: str,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        
        speech_dir = os.path.join(data_dir, "speech")
        noise_dir = os.path.join(data_dir, "noise")
        
        # Check if directories exist
        has_data = os.path.exists(speech_dir) and os.path.exists(noise_dir)
        
        if has_data:
            # Real data mode
            train_dataset = AuraNetDataset(
                clean_dir=speech_dir,
                noise_dir=noise_dir,
                sample_rate=self.config["data"]["sample_rate"],
                segment_length=self.config["data"]["segment_length"],
                snr_range=tuple(self.config["data"]["snr_range"]),
                augment=True,
            )
            
            val_dataset = AuraNetDataset(
                clean_dir=speech_dir,
                noise_dir=noise_dir,
                sample_rate=self.config["data"]["sample_rate"],
                segment_length=self.config["data"]["segment_length"],
                snr_range=tuple(self.config["data"]["snr_range"]),
                augment=False,
            )
        else:
            train_dataset = None
            val_dataset = None
        
        # Fallback to synthetic if no real data
        if train_dataset is None or len(train_dataset) == 0:
            print("⚠️ No real data found, using synthetic mode")
            train_dataset = AuraNetDataset(
                synthetic_mode=True,
                num_synthetic_samples=1000,
            )
            val_dataset = AuraNetDataset(
                synthetic_mode=True,
                num_synthetic_samples=100,
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )
        
        print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        
        return train_loader, val_loader
        
    def set_stage(self, stage: int) -> None:
        """
        Set training stage and adjust learning rates.
        
        Stage 1: Separation training
        - Full model training
        - WDRC with lower learning rate (0.1x)
        
        Stage 2: WDRC fine-tuning
        - Reduce main model LR (0.5x)
        - Full WDRC learning rate
        - Add envelope loss
        """
        self.current_stage = stage
        
        # Update loss weights for stage
        if hasattr(self.criterion, 'set_stage'):
            self.criterion.set_stage(stage)
        
        if stage == 1:
            print("\n" + "="*60)
            print("🔧 STAGE 1: Separation Training")
            print("   Focus: Deep filtering quality, noise reduction")
            print("   Loss: Loud-Loss + SI-SDR + Multi-Res STFT")
            print("="*60)
            
            # WDRC with reduced learning rate
            for group in self.optimizer.param_groups:
                if group.get("name") == "wdrc":
                    group["lr"] = self.config["training"]["learning_rate"] * 0.1
                    
        elif stage == 2:
            print("\n" + "="*60)
            print("🔧 STAGE 2: WDRC Fine-tuning")
            print("   Focus: Dynamic range, loudness consistency")
            print("   Loss: + Envelope loss, adjusted weights")
            print("="*60)
            
            # WDRC with full learning rate
            wdrc_mult = self.config["training"].get("stage2_wdrc_lr_mult", 1.0)
            for group in self.optimizer.param_groups:
                if group.get("name") == "wdrc":
                    group["lr"] = self.config["training"]["learning_rate"] * wdrc_mult
                else:
                    # Reduce other params slightly
                    group["lr"] = self.config["training"]["learning_rate"] * 0.5
                    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1} [Stage {self.current_stage}]"
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            noisy_stft = batch['noisy_stft'].to(self.device)
            clean_stft = batch['clean_stft'].to(self.device)
            clean_audio = batch.get('clean_audio')
            
            if clean_audio is not None:
                clean_audio = clean_audio.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                enhanced_stft, wdrc_params, _ = self.model(noisy_stft)
                
                # Reconstruct audio for time-domain loss
                if clean_audio is not None:
                    enhanced_audio = self.stft.inverse(enhanced_stft)
                    
                    # Match lengths
                    min_len = min(enhanced_audio.shape[-1], clean_audio.shape[-1])
                    enhanced_audio = enhanced_audio[..., :min_len]
                    clean_audio_batch = clean_audio[..., :min_len]
                else:
                    enhanced_audio = None
                    clean_audio_batch = None
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    enhanced_stft,
                    clean_stft,
                    enhanced_audio,
                    clean_audio_batch,
                )
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["grad_clip"]
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item() if torch.is_tensor(value) else value
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average losses
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        
        for key in loss_components:
            loss_components[key] /= n_batches
        
        return {"loss": avg_loss, **loss_components}
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {}
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            noisy_stft = batch['noisy_stft'].to(self.device)
            clean_stft = batch['clean_stft'].to(self.device)
            clean_audio = batch.get('clean_audio')
            
            if clean_audio is not None:
                clean_audio = clean_audio.to(self.device)
            
            enhanced_stft, _, _ = self.model(noisy_stft)
            
            if clean_audio is not None:
                enhanced_audio = self.stft.inverse(enhanced_stft)
                min_len = min(enhanced_audio.shape[-1], clean_audio.shape[-1])
                enhanced_audio = enhanced_audio[..., :min_len]
                clean_audio_batch = clean_audio[..., :min_len]
            else:
                enhanced_audio = None
                clean_audio_batch = None
            
            loss, loss_dict = self.criterion(
                enhanced_stft,
                clean_stft,
                enhanced_audio,
                clean_audio_batch,
            )
            
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item() if torch.is_tensor(value) else value
        
        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches
        
        for key in loss_components:
            loss_components[key] /= n_batches
        
        return {"loss": avg_loss, **loss_components}
        
    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save training checkpoint."""
        save_dir = Path(self.config["checkpoint"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "stage": self.current_stage,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }
        
        torch.save(checkpoint, save_dir / filename)
        
        if is_best:
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"💾 Saved best model (loss: {self.best_loss:.4f})")
            
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.current_stage = checkpoint.get("stage", 1)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        
        print(f"✅ Resumed from checkpoint (epoch {self.current_epoch}, stage {self.current_stage})")
        
    def train(self, start_stage: int = 1) -> None:
        """
        Full training loop with 2-stage training.
        
        Args:
            start_stage: Which stage to start from (1 or 2)
        """
        total_epochs = self.config["training"]["epochs"]
        stage1_epochs = int(total_epochs * self.config["training"]["stage1_epochs_ratio"])
        
        print(f"\n{'='*60}")
        print(f"🚀 AuraNet V2 Training")
        print(f"{'='*60}")
        print(f"   Total epochs: {total_epochs}")
        print(f"   Stage 1 (Separation): epochs 1-{stage1_epochs}")
        print(f"   Stage 2 (WDRC Fine-tune): epochs {stage1_epochs+1}-{total_epochs}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config['training']['batch_size']}")
        print(f"{'='*60}\n")
        
        # Set initial stage
        if start_stage == 2:
            self.current_stage = 2
            self.current_epoch = stage1_epochs
        
        self.set_stage(self.current_stage)
        
        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch
            
            # Check for stage transition
            if epoch == stage1_epochs and self.current_stage == 1:
                self.set_stage(2)
                # Save stage 1 checkpoint
                self.save_checkpoint("stage1_final.pt")
            
            # Update learning rate
            current_lr = self.scheduler.step(epoch)
            print(f"\n📈 Epoch {epoch+1}/{total_epochs} | LR: {current_lr:.2e} | Stage: {self.current_stage}")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            
            # Log individual losses
            for key, value in train_metrics.items():
                if key != 'loss' and key != 'total':
                    print(f"      {key}: {value:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['loss']
            
            # Save checkpoint
            if (epoch + 1) % self.config["checkpoint"]["save_every"] == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pt", is_best)
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        print(f"\n✅ Training complete! Best loss: {self.best_loss:.4f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AuraNet V2 with 2-Stage Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 2-stage training
  python train_v2.py --data-dir datasets --epochs 100
  
  # Stage 1 only
  python train_v2.py --data-dir datasets --stage 1 --epochs 80
  
  # Stage 2 from checkpoint  
  python train_v2.py --stage 2 --checkpoint checkpoints/stage1_final.pt
  
  # Use V1 model (backward compatibility)
  python train_v2.py --model v1
        """
    )
    
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--data-dir", type=str, default="datasets",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--model", type=str, choices=["v1", "v2"], default="v2",
        help="Model version (default: v2)"
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], default=1,
        help="Training stage to start from"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda, mps, cpu)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        for key in DEFAULT_CONFIG:
            if key not in config:
                config[key] = DEFAULT_CONFIG[key]
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Override from command line
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.model:
        config["model"]["version"] = args.model
    
    # Select device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU - training will be slow!")
    
    # Create trainer
    trainer = TrainerV2(
        config=config,
        data_dir=args.data_dir,
        device=device,
        resume_from=args.checkpoint,
    )
    
    # Train
    trainer.train(start_stage=args.stage)


if __name__ == "__main__":
    main()
