#!/usr/bin/env python3
"""
================================================================================
AuraNet V2 Training with Stability & Profiling
================================================================================

Production-ready training script with:
- Gradient clipping (prevents explosion)
- NaN detection (stops early if unstable)
- Loss component logging
- Per-epoch latency profiling
- Checkpoint with validation

Usage:
    python train_v2_stable.py --config configs/default.yaml

================================================================================
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class TrainConfig:
    """Training configuration with safe defaults."""
    
    # Model
    sample_rate: int = 16000
    n_fft: int = 256
    hop_length: int = 80
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
    # Stability
    gradient_clip_norm: float = 5.0  # Max gradient norm
    nan_threshold: int = 3  # Stop after N NaN batches
    
    # Loss weights (Stage 1: Separation)
    loud_loss_weight: float = 1.0
    sisdr_loss_weight: float = 0.3
    phase_loss_weight: float = 0.1
    wdrc_loss_weight: float = 0.0  # Disabled in Stage 1
    
    # Scheduler
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5
    
    # Profiling
    profile_latency: bool = True
    target_latency_ms: float = 10.0
    
    # Stage
    stage: int = 1  # 1 = Separation, 2 = WDRC fine-tune


# ==============================================================================
# NaN DETECTOR
# ==============================================================================

class NaNDetector:
    """Detects and handles NaN values during training."""
    
    def __init__(self, threshold: int = 3):
        self.nan_count = 0
        self.threshold = threshold
        
    def check(self, loss: torch.Tensor, gradients: Optional[List[torch.Tensor]] = None) -> bool:
        """
        Check for NaN values.
        
        Returns:
            True if training should stop.
        """
        has_nan = False
        
        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[NaN] Loss is NaN/Inf: {loss.item()}")
            has_nan = True
        
        # Check gradients
        if gradients:
            for i, grad in enumerate(gradients):
                if grad is not None and (torch.isnan(grad).any() or torch.isinf(grad).any()):
                    logger.warning(f"[NaN] Gradient {i} contains NaN/Inf")
                    has_nan = True
                    break
        
        if has_nan:
            self.nan_count += 1
            if self.nan_count >= self.threshold:
                logger.error(f"[NaN] Threshold reached ({self.nan_count}). Stopping.")
                return True
        else:
            # Reset on good batch
            self.nan_count = 0
        
        return False


# ==============================================================================
# LOSS TRACKER
# ==============================================================================

class LossTracker:
    """Tracks individual loss components for debugging."""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        
    def update(self, losses: Dict[str, float]):
        """Update with new loss values."""
        for name, value in losses.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
    
    def get_stats(self, last_n: int = 100) -> Dict[str, Dict[str, float]]:
        """Get statistics for each loss component."""
        stats = {}
        for name, values in self.history.items():
            recent = values[-last_n:]
            stats[name] = {
                'mean': sum(recent) / len(recent),
                'min': min(recent),
                'max': max(recent),
            }
        return stats
    
    def log_summary(self, epoch: int):
        """Log summary for epoch."""
        stats = self.get_stats()
        logger.info(f"Epoch {epoch} Loss Components:")
        for name, s in stats.items():
            logger.info(f"  {name}: mean={s['mean']:.4f} min={s['min']:.4f} max={s['max']:.4f}")


# ==============================================================================
# LATENCY PROFILER
# ==============================================================================

class LatencyProfiler:
    """Profiles model latency during training."""
    
    def __init__(self, model: nn.Module, device: torch.device, config: TrainConfig):
        self.model = model
        self.device = device
        self.target_ms = config.target_latency_ms
        
    def profile(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Profile model latency.
        
        Returns dict with latency stats.
        """
        self.model.eval()
        
        # Create test input: 100 frames ~= 0.5s of audio
        x = torch.randn(1, 2, 100, 129, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(x)
        
        # Measure
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(x)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            times.append((time.perf_counter() - start) * 1000)
        
        times = sorted(times)
        
        self.model.train()
        
        return {
            'mean_ms': sum(times) / len(times),
            'p50_ms': times[len(times) // 2],
            'p95_ms': times[int(len(times) * 0.95)],
            'p99_ms': times[int(len(times) * 0.99)],
            'meets_target': times[int(len(times) * 0.95)] < self.target_ms,
        }


# ==============================================================================
# TRAINING STEP
# ==============================================================================

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: TrainConfig,
    nan_detector: NaNDetector,
) -> Tuple[float, Dict[str, float], bool]:
    """
    Single training step with stability checks.
    
    Returns:
        total_loss, loss_components, should_stop
    """
    model.train()
    optimizer.zero_grad()
    
    # Unpack batch
    noisy = batch['noisy']  # [B, N]
    clean = batch['clean']  # [B, N]
    
    # Forward pass
    try:
        output, gru_out, _ = model(noisy)
    except RuntimeError as e:
        logger.error(f"Forward pass failed: {e}")
        return float('nan'), {}, True
    
    # Compute loss
    try:
        loss_dict = loss_fn(output, clean, gru_out)
        total_loss = loss_dict['total']
    except RuntimeError as e:
        logger.error(f"Loss computation failed: {e}")
        return float('nan'), {}, True
    
    # Backward pass
    try:
        total_loss.backward()
    except RuntimeError as e:
        logger.error(f"Backward pass failed: {e}")
        return float('nan'), {}, True
    
    # Check for NaN
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    should_stop = nan_detector.check(total_loss, gradients)
    
    if should_stop:
        return float('nan'), {}, True
    
    # ==== GRADIENT CLIPPING ====
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=config.gradient_clip_norm
    )
    
    # Log if gradients were clipped significantly
    if grad_norm > config.gradient_clip_norm * 0.9:
        logger.warning(f"[Grad] Large gradient norm: {grad_norm:.2f}")
    
    # Optimizer step
    optimizer.step()
    
    # Convert loss dict to floats
    loss_components = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    return total_loss.item(), loss_components, False


# ==============================================================================
# VALIDATION STEP
# ==============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run validation and compute metrics.
    """
    model.eval()
    
    total_loss = 0
    si_sdr_total = 0
    num_batches = 0
    
    for batch in val_loader:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        
        output, gru_out, _ = model(noisy)
        
        loss_dict = loss_fn(output, clean, gru_out)
        total_loss += loss_dict['total'].item()
        
        # Compute SI-SDR improvement
        si_sdr_noisy = compute_si_sdr(noisy, clean)
        si_sdr_enhanced = compute_si_sdr(output, clean)
        si_sdr_total += (si_sdr_enhanced - si_sdr_noisy).mean().item()
        
        num_batches += 1
    
    return {
        'val_loss': total_loss / num_batches,
        'si_sdr_improvement': si_sdr_total / num_batches,
    }


def compute_si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compute SI-SDR."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    
    dot = (estimate * reference).sum(dim=-1, keepdim=True)
    s_target = dot / (reference.pow(2).sum(dim=-1, keepdim=True) + 1e-8) * reference
    
    e_noise = estimate - s_target
    
    si_sdr = 10 * torch.log10(
        s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + 1e-8)
    )
    
    return si_sdr


# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
    metrics: Dict[str, float],
):
    """Save training checkpoint."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'metrics': metrics,
    }
    
    path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint: {path}")
    
    # Also save as 'latest'
    latest_path = os.path.join(config.checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> int:
    """Load checkpoint and return starting epoch."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return checkpoint['epoch'] + 1


# ==============================================================================
# SYNTHETIC DATASET (for testing)
# ==============================================================================

class SyntheticDataset(Dataset):
    """Synthetic dataset for testing training loop."""
    
    def __init__(self, num_samples: int = 1000, sample_length: int = 16000):
        self.num_samples = num_samples
        self.sample_length = sample_length
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate clean speech (sinusoid mixture)
        t = torch.linspace(0, 1, self.sample_length)
        clean = (
            0.5 * torch.sin(2 * 3.14159 * 440 * t) +
            0.3 * torch.sin(2 * 3.14159 * 880 * t) +
            0.2 * torch.sin(2 * 3.14159 * 1320 * t)
        )
        
        # Add noise
        snr_db = torch.rand(1) * 20 - 5  # -5 to 15 dB
        noise_level = 10 ** (-snr_db / 20) * clean.std()
        noise = torch.randn_like(clean) * noise_level
        noisy = clean + noise
        
        return {
            'clean': clean,
            'noisy': noisy,
        }


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(config: TrainConfig):
    """Main training loop."""
    
    logger.info("=" * 60)
    logger.info("AuraNet V2 Training with Stability Enhancements")
    logger.info("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Model
    try:
        from auranet_v2_complete import AuraNetV2Complete
        model = AuraNetV2Complete().to(device)
    except ImportError:
        logger.error("Could not import AuraNetV2Complete. Run from auranet directory.")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params:,}")
    
    # Loss
    try:
        from auranet_v2_complete import AuraNetV2Loss
        loss_fn = AuraNetV2Loss(
            loud_weight=config.loud_loss_weight,
            sisdr_weight=config.sisdr_loss_weight,
            phase_weight=config.phase_loss_weight,
            wdrc_weight=config.wdrc_loss_weight,
        ).to(device)
    except ImportError:
        logger.warning("Using basic MSE loss (AuraNetV2Loss not found)")
        loss_fn = lambda pred, target, gru: {'total': F.mse_loss(pred, target)}
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.epochs // 4,
        T_mult=2,
        eta_min=config.min_lr,
    )
    
    # Data
    logger.info("Creating synthetic dataset for testing...")
    train_dataset = SyntheticDataset(num_samples=1000)
    val_dataset = SyntheticDataset(num_samples=100)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Utilities
    nan_detector = NaNDetector(threshold=config.nan_threshold)
    loss_tracker = LossTracker()
    latency_profiler = LatencyProfiler(model, device, config) if config.profile_latency else None
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{config.epochs} {'='*20}")
        
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Train step
            loss, loss_components, should_stop = train_step(
                model, batch, optimizer, loss_fn, config, nan_detector
            )
            
            if should_stop:
                logger.error("Training stopped due to instability.")
                return
            
            epoch_losses.append(loss)
            loss_tracker.update(loss_components)
            
            if batch_idx % 20 == 0:
                logger.info(f"  Batch {batch_idx}: loss={loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Log epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
        loss_tracker.log_summary(epoch)
        
        # Validation
        val_metrics = validate(model, val_loader, loss_fn, device)
        logger.info(f"Validation: loss={val_metrics['val_loss']:.4f}, SI-SDR imp={val_metrics['si_sdr_improvement']:.2f} dB")
        
        # Latency profiling
        if latency_profiler and epoch % 5 == 0:
            latency = latency_profiler.profile()
            status = "✓" if latency['meets_target'] else "✗"
            logger.info(f"Latency: p95={latency['p95_ms']:.2f}ms (target: {config.target_latency_ms}ms) {status}")
        
        # Checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, config, val_metrics)
        
        # Best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved: val_loss={best_val_loss:.4f}")
    
    logger.info("\nTraining complete!")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AuraNet V2")
    parser.add_argument("--stage", type=int, default=1, help="Training stage (1=separation, 2=WDRC)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        stage=args.stage,
    )
    
    # Adjust for Stage 2
    if args.stage == 2:
        config.loud_loss_weight = 0.5
        config.wdrc_loss_weight = 0.5
        logger.info("Stage 2: WDRC fine-tuning enabled")
    
    train(config)
