#!/usr/bin/env python3
# =============================================================================
# AuraNet Fine-Tuning for Perceptual Quality
# =============================================================================
#
# PURPOSE: Transform AuraNet from "high SI-SNR" → "perceptually natural"
#
# TARGET METRICS:
#   PESQ: 2.8–3.1 (from ~2.5)
#   STOI: 0.86–0.90 (from ~0.83)
#   SI-SNR: maintain ≥16 dB
#
# STRATEGY:
#   Stage 1 (done): Train with SI-SNR + base losses
#   Stage 2 (this): Fine-tune with perceptual loss stack
#
# USAGE:
#   python train_finetune.py --checkpoint checkpoints/best_model_v3.pt
#   python train_finetune.py --checkpoint checkpoints/best_model_v3.pt --epochs 10
#   python train_finetune.py --checkpoint checkpoints/best_model_v3.pt --freeze-encoder
#
# =============================================================================

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from copy import deepcopy

# Performance configuration (before torch imports)
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# Disable cuDNN benchmark for stability
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

# Disable torch._dynamo
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()
except (ImportError, AttributeError):
    pass

from model_v3 import AuraNetV3, create_auranet_v3
from loss_perceptual import PerceptualLoss
from dataset_v3 import AuraNetV3Dataset
from utils.stft import CausalSTFT
from metrics import compute_pesq, compute_stoi


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Copy EMA weights into model (for evaluation)."""
        self.backup = {
            name: p.clone()
            for name, p in model.named_parameters()
            if p.requires_grad and name in self.shadow
        }
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


# =============================================================================
# STFT Helper with tanh output
# =============================================================================

def stft_to_waveform_tanh(stft_module: CausalSTFT, enhanced_stft: torch.Tensor,
                          length: int) -> torch.Tensor:
    """
    Convert STFT to waveform with tanh activation (not hard clamp).

    tanh advantages over clamp:
    - Smooth and differentiable everywhere
    - Gradients don't vanish at boundaries
    - Natural saturation behavior like analog equipment
    """
    enhanced_audio = stft_module.inverse(enhanced_stft)

    # Trim/pad to target length
    if enhanced_audio.shape[-1] > length:
        enhanced_audio = enhanced_audio[..., :length]
    elif enhanced_audio.shape[-1] < length:
        pad = length - enhanced_audio.shape[-1]
        enhanced_audio = F.pad(enhanced_audio, (0, pad))

    # Apply tanh for smooth saturation (instead of hard clamp)
    enhanced_audio = torch.tanh(enhanced_audio)

    return enhanced_audio


# =============================================================================
# Fine-Tuning Trainer
# =============================================================================

class PerceptualFineTuner:
    """
    Fine-tuning trainer for perceptual quality improvement.

    KEY FEATURES:
    - Load pretrained AuraNet V3
    - Perceptual loss stack (LoudLoss + MultiResSTFT + Mel + SI-SNR)
    - Optional encoder freezing
    - Validation with PESQ/STOI/SI-SNR
    - Audio sample saving
    - CosineAnnealingLR scheduler
    """

    def __init__(self, config: Dict, checkpoint_path: str, device: str = 'auto',
                 freeze_encoder: bool = False):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.freeze_encoder = freeze_encoder

        # Device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Device: {self.device}")

        # Load model
        self.model = self._load_pretrained_model()

        # STFT for audio reconstruction
        stft_cfg = config.get('stft', {})
        self.stft = CausalSTFT(
            n_fft=stft_cfg.get('n_fft', 256),
            hop_length=stft_cfg.get('hop_size', 80),
            win_length=stft_cfg.get('window_size', 160),
        ).to(self.device)

        # Perceptual loss
        train_cfg = config.get('training', {})
        self.criterion = PerceptualLoss(
            weight_loud=train_cfg.get('weight_loud', 0.50),
            weight_stft=train_cfg.get('weight_stft', 0.25),
            weight_mel=train_cfg.get('weight_mel', 0.15),
            weight_sisnr=train_cfg.get('weight_sisnr', 0.10),
            weight_harmonic=train_cfg.get('weight_harmonic', 0.05),
            use_harmonic=train_cfg.get('use_harmonic', True),
            sample_rate=config.get('audio', {}).get('sample_rate', 16000),
        )

        # Optimizer — lower LR for fine-tuning
        self.lr = train_cfg.get('learning_rate', 1e-4)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=train_cfg.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
        )

        # Scheduler
        self.num_epochs = train_cfg.get('num_epochs', 10)
        scheduler_type = train_cfg.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6,
            )
        else:  # plateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=2, min_lr=1e-6,
            )

        self.scheduler_type = scheduler_type

        # Gradient clipping
        self.grad_clip = train_cfg.get('gradient_clip', 3.0)

        # EMA
        self.ema = EMA(self.model, decay=0.999)

        # Checkpointing
        self.ckpt_dir = Path(train_cfg.get('checkpoint_dir', 'checkpoints'))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Audio samples directory
        self.samples_dir = Path('audio_samples')
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.current_epoch = 0
        self.best_pesq = 0.0
        self.best_stoi = 0.0

    def _load_pretrained_model(self) -> nn.Module:
        """Load pretrained AuraNet V3 checkpoint."""
        print(f"Loading pretrained model from: {self.checkpoint_path}")

        # Create model architecture
        model = create_auranet_v3(self.config)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is the state dict directly
            model.load_state_dict(checkpoint)

        model = model.to(self.device)

        # Optionally freeze encoder
        if self.freeze_encoder:
            print("Freezing encoder layers...")
            for name, param in model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        return model

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        loss_sums = {}
        n = 0

        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {self.current_epoch + 1}",
            mininterval=0,
            leave=True
        )

        for batch_idx, batch in pbar:
            noisy_audio = batch['noisy_audio'].to(self.device)
            clean_audio = batch['clean_audio'].to(self.device)

            # Handle dimensions
            if noisy_audio.dim() == 2:
                noisy_audio = noisy_audio.unsqueeze(1)
            if clean_audio.dim() == 2:
                clean_audio = clean_audio.unsqueeze(1)

            self.optimizer.zero_grad()

            # Forward pass
            noisy_stft = self.stft(noisy_audio)
            clean_stft = self.stft(clean_audio)

            enhanced_stft, _, _ = self.model(noisy_stft)

            # Reconstruct audio with tanh (not clamp)
            enhanced_audio = self.stft.inverse(enhanced_stft)
            min_len = min(enhanced_audio.shape[-1], clean_audio.shape[-1])
            enhanced_audio = enhanced_audio[..., :min_len]
            clean_audio_batch = clean_audio.squeeze(1)[..., :min_len]

            # Apply tanh for smooth saturation (replaces hard clamp)
            enhanced_audio = torch.tanh(enhanced_audio)

            # Compute perceptual loss
            loss, loss_dict = self.criterion(
                enhanced_audio, clean_audio_batch,
                enhanced_stft, clean_stft
            )

            # NaN guard
            if not torch.isfinite(loss):
                print(f"  [WARN] Non-finite loss at batch {batch_idx}, skipping")
                self.optimizer.zero_grad()
                continue

            # Backward
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.ema.update(self.model)

            # Accumulate losses
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
            n += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average
        avg = {k: v / max(n, 1) for k, v in loss_sums.items()}
        avg['total'] = total_loss / max(n, 1)
        return avg

    @torch.no_grad()
    def validate(self, loader: DataLoader, save_samples: bool = False) -> Dict[str, float]:
        """Validate with perceptual metrics."""
        # Use EMA weights
        self.ema.apply(self.model)
        self.model.eval()

        total_loss = 0.0
        loss_sums = {}
        pesq_scores = []
        stoi_scores = []
        sisnr_scores = []
        n = 0

        sample_rate = self.config.get('audio', {}).get('sample_rate', 16000)

        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
            noisy_audio = batch['noisy_audio'].to(self.device)
            clean_audio = batch['clean_audio'].to(self.device)

            if noisy_audio.dim() == 2:
                noisy_audio = noisy_audio.unsqueeze(1)
            if clean_audio.dim() == 2:
                clean_audio = clean_audio.unsqueeze(1)

            noisy_stft = self.stft(noisy_audio)
            clean_stft = self.stft(clean_audio)

            enhanced_stft, _, _ = self.model(noisy_stft)
            enhanced_audio = self.stft.inverse(enhanced_stft)

            min_len = min(enhanced_audio.shape[-1], clean_audio.shape[-1])
            enhanced_audio = enhanced_audio[..., :min_len]
            clean_audio_batch = clean_audio.squeeze(1)[..., :min_len]

            # Apply tanh
            enhanced_audio = torch.tanh(enhanced_audio)

            # Loss
            loss, loss_dict = self.criterion(
                enhanced_audio, clean_audio_batch,
                enhanced_stft, clean_stft
            )

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
            n += 1

            # Compute perceptual metrics for first few samples
            if batch_idx < 10:
                for i in range(min(2, enhanced_audio.shape[0])):
                    pred_np = enhanced_audio[i].cpu().numpy()
                    target_np = clean_audio_batch[i].cpu().numpy()

                    pesq_score = compute_pesq(pred_np, target_np, sample_rate, 'wb')
                    stoi_score = compute_stoi(pred_np, target_np, sample_rate)

                    if pesq_score > 0:
                        pesq_scores.append(pesq_score)
                    if stoi_score > 0:
                        stoi_scores.append(stoi_score)

                    # SI-SNR
                    pred_t = torch.from_numpy(pred_np)
                    target_t = torch.from_numpy(target_np)
                    pred_t = pred_t - pred_t.mean()
                    target_t = target_t - target_t.mean()
                    dot = (pred_t * target_t).sum()
                    s_target = (dot / (target_t.pow(2).sum() + 1e-8)) * target_t
                    e_noise = pred_t - s_target
                    sisnr = 10 * torch.log10(
                        s_target.pow(2).sum() / (e_noise.pow(2).sum() + 1e-8) + 1e-8
                    )
                    sisnr_scores.append(sisnr.item())

            # Save audio samples
            if save_samples and batch_idx == 0:
                self._save_audio_samples(
                    noisy_audio.squeeze(1)[0].cpu(),
                    enhanced_audio[0].cpu(),
                    clean_audio_batch[0].cpu(),
                    sample_rate
                )

        self.ema.restore(self.model)

        # Average
        avg = {k: v / max(n, 1) for k, v in loss_sums.items()}
        avg['total'] = total_loss / max(n, 1)

        # Perceptual metrics
        avg['pesq'] = np.mean(pesq_scores) if pesq_scores else -1.0
        avg['stoi'] = np.mean(stoi_scores) if stoi_scores else -1.0
        avg['si_snr_metric'] = np.mean(sisnr_scores) if sisnr_scores else 0.0

        return avg

    def _save_audio_samples(self, noisy: torch.Tensor, enhanced: torch.Tensor,
                            clean: torch.Tensor, sample_rate: int):
        """Save audio samples for listening test."""
        try:
            import soundfile as sf
        except ImportError:
            print("soundfile not installed, skipping audio sample saving")
            return

        epoch_dir = self.samples_dir / f"epoch_{self.current_epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        sf.write(epoch_dir / "noisy.wav", noisy.numpy(), sample_rate)
        sf.write(epoch_dir / "enhanced.wav", enhanced.numpy(), sample_rate)
        sf.write(epoch_dir / "clean.wav", clean.numpy(), sample_rate)
        print(f"  Audio samples saved to {epoch_dir}")

    def save_checkpoint(self, tag: str, is_best: bool = False):
        """Save checkpoint."""
        ckpt = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_shadow': self.ema.shadow,
            'best_pesq': self.best_pesq,
            'best_stoi': self.best_stoi,
            'config': self.config,
        }

        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(ckpt, path)

        if is_best:
            # Save EMA weights as best model
            self.ema.apply(self.model)
            torch.save(self.model.state_dict(), self.ckpt_dir / "best_model_perceptual.pt")
            self.ema.restore(self.model)
            print(f"  💾 Best perceptual model saved (PESQ={self.best_pesq:.3f}, STOI={self.best_stoi:.3f})")

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Full fine-tuning loop."""
        print(f"\n{'='*60}")
        print("AuraNet Perceptual Fine-Tuning")
        print(f"{'='*60}")
        print(f"  Pretrained:    {self.checkpoint_path}")
        print(f"  Epochs:        {self.num_epochs}")
        print(f"  Learning Rate: {self.lr}")
        print(f"  Scheduler:     {self.scheduler_type}")
        print(f"  Freeze Encoder:{self.freeze_encoder}")
        print(f"  Device:        {self.device}")
        print(f"  Batches/epoch: {len(train_loader)}")
        if val_loader:
            print(f"  Val batches:   {len(val_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            # Train
            train_losses = self.train_epoch(train_loader)
            elapsed = time.time() - t0

            # Validate
            val_losses = None
            if val_loader:
                val_losses = self.validate(val_loader, save_samples=(epoch % 2 == 0))

            # LR scheduling
            if self.scheduler_type == 'cosine':
                self.scheduler.step()
            elif val_losses:
                self.scheduler.step(val_losses['total'])

            # Logging
            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.num_epochs} ({elapsed:.1f}s) | LR={lr_now:.2e}")
            print(f"  Train: total={train_losses['total']:.4f} "
                  f"loud={train_losses.get('loud', 0):.4f} "
                  f"stft={train_losses.get('multi_res_stft', 0):.4f} "
                  f"mel={train_losses.get('mel', 0):.4f} "
                  f"sisnr={train_losses.get('si_snr', 0):.4f}")

            if val_losses:
                print(f"  Val:   total={val_losses['total']:.4f}")
                print(f"  Metrics: PESQ={val_losses['pesq']:.3f} "
                      f"STOI={val_losses['stoi']:.3f} "
                      f"SI-SNR={val_losses['si_snr_metric']:.2f} dB")

            # Check for best model (based on PESQ)
            is_best = False
            if val_losses and val_losses['pesq'] > 0:
                if val_losses['pesq'] > self.best_pesq:
                    self.best_pesq = val_losses['pesq']
                    self.best_stoi = val_losses['stoi']
                    is_best = True

            # Save checkpoints
            self.save_checkpoint("latest_perceptual", is_best=is_best)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"finetune_epoch_{epoch+1}")

        print(f"\n✅ Fine-tuning complete!")
        print(f"   Best PESQ: {self.best_pesq:.3f}")
        print(f"   Best STOI: {self.best_stoi:.3f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AuraNet Perceptual Fine-Tuning")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--config", type=str, default="config_v3.yaml",
                        help="Path to config file")
    parser.add_argument("--clean-dir", type=str, default=None,
                        help="Directory with clean audio")
    parser.add_argument("--noise-dir", type=str, default=None,
                        help="Directory with noise audio")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder layers")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu, mps)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
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
    config.setdefault('training', {})
    config['training']['num_epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['training']['batch_size'] = args.batch_size

    # Data paths
    clean_dir = args.clean_dir or config.get('training', {}).get('clean_dir', 'datasets/speech')
    noise_dir = args.noise_dir or config.get('training', {}).get('noise_dir', 'datasets/noise')
    batch_size = args.batch_size

    synthetic = args.synthetic or not (Path(clean_dir).exists() and Path(noise_dir).exists())

    if synthetic:
        print("Using synthetic data for fine-tuning")

    # Create dataset
    full_dataset = AuraNetV3Dataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        sample_rate=config.get('audio', {}).get('sample_rate', 16000),
        segment_length=config.get('training', {}).get('segment_length', 3.0),
        augment=True,
        synthetic_mode=synthetic,
        num_synthetic_samples=config.get('training', {}).get('num_synthetic_samples', 2000),
    )

    # Train/val split (90/10)
    total = len(full_dataset)
    val_size = max(1, int(total * 0.1))
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )

    print(f"Dataset: {train_size} train / {val_size} val samples")

    # Create trainer
    trainer = PerceptualFineTuner(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        freeze_encoder=args.freeze_encoder
    )

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
