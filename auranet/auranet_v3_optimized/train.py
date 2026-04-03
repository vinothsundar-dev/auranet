# =============================================================================
# AuraNet V3 Optimized — Training Script
# =============================================================================
# Loss:  0.6 * MSE  +  0.4 * L1  (on complex STFT domain)
# Optimizer: AdamW + ReduceLROnPlateau
# Features: gradient clipping, dynamic noise mixing, validation loop
# =============================================================================

import os
import glob
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import soundfile as sf

from model import AuraNetV3, create_model, apply_wdrc, N_FFT, HOP_LENGTH


# ── Dataset ──────────────────────────────────────────────────────────────────

class SpeechEnhancementDataset(Dataset):
    """
    Loads clean + noise audio and mixes on-the-fly at a random SNR.

    Directory layout expected:
        clean_dir/  *.wav
        noise_dir/  *.wav

    If paired_dir is given instead (noisy/ + clean/ subdirs), it loads
    pre-mixed pairs directly.
    """

    def __init__(self,
                 clean_dir: str,
                 noise_dir: str,
                 sample_rate: int = 16000,
                 segment_length: float = 3.0,
                 snr_range: tuple = (-5, 20)):
        self.sr = sample_rate
        self.seg_samples = int(segment_length * sample_rate)
        self.snr_low, self.snr_high = snr_range

        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "**", "*.wav"), recursive=True)
                                  + glob.glob(os.path.join(clean_dir, "**", "*.flac"), recursive=True))
        self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
                                  + glob.glob(os.path.join(noise_dir, "**", "*.flac"), recursive=True))

        if not self.clean_files:
            raise FileNotFoundError(f"No audio files found in {clean_dir}")
        if not self.noise_files:
            raise FileNotFoundError(f"No audio files found in {noise_dir}")

        print(f"Dataset: {len(self.clean_files)} clean, {len(self.noise_files)} noise files")

    def __len__(self):
        return len(self.clean_files)

    def _load_random_segment(self, path: str) -> np.ndarray:
        info = sf.info(path)
        total = info.frames
        if total <= self.seg_samples:
            audio, _ = sf.read(path, dtype="float32")
            # Pad if shorter
            if len(audio) < self.seg_samples:
                audio = np.pad(audio, (0, self.seg_samples - len(audio)))
            return audio[:self.seg_samples]
        start = random.randint(0, total - self.seg_samples)
        audio, _ = sf.read(path, start=start, stop=start + self.seg_samples, dtype="float32")
        return audio

    def __getitem__(self, idx):
        clean = self._load_random_segment(self.clean_files[idx])
        noise = self._load_random_segment(random.choice(self.noise_files))

        # Mix at random SNR
        snr_db = random.uniform(self.snr_low, self.snr_high)
        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-8)
        noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-8)
        target_noise_rms = clean_rms / (10 ** (snr_db / 20))
        noise = noise * (target_noise_rms / noise_rms)

        noisy = clean + noise

        # Normalize to prevent clipping
        peak = max(np.abs(noisy).max(), 1e-8)
        noisy = noisy / peak
        clean = clean / peak

        return (torch.from_numpy(noisy).float(),
                torch.from_numpy(clean).float())


# ── Loss ─────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """Loss = 0.6 * MSE + 0.4 * L1  (time-domain)."""

    def __init__(self, mse_weight: float = 0.6, l1_weight: float = 0.4):
        super().__init__()
        self.mse_w = mse_weight
        self.l1_w = l1_weight

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        # Align lengths
        min_len = min(enhanced.shape[-1], clean.shape[-1])
        enhanced = enhanced[..., :min_len]
        clean = clean[..., :min_len]

        mse = F.mse_loss(enhanced, clean)
        l1 = F.l1_loss(enhanced, clean)
        return self.mse_w * mse + self.l1_w * l1


# ── Trainer ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = create_model()
    model = model.to(device)

    # Datasets
    train_ds = SpeechEnhancementDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        segment_length=args.segment_length,
        snr_range=(args.snr_low, args.snr_high),
    )
    val_ds = SpeechEnhancementDataset(
        clean_dir=args.val_clean_dir or args.clean_dir,
        noise_dir=args.val_noise_dir or args.noise_dir,
        segment_length=args.segment_length,
        snr_range=(args.snr_low, args.snr_high),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    # Loss, optimizer, scheduler
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # ── Training loop ──
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []

        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            enhanced, _, _ = model(noisy)

            loss = criterion(enhanced, clean)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_losses.append(loss.item())

            if (batch_idx + 1) % args.log_interval == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}")

        avg_train = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                enhanced, _, _ = model(noisy)
                val_losses.append(criterion(enhanced, clean).item())

        avg_val = np.mean(val_losses)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={avg_train:.4f} | val={avg_val:.4f} | lr={lr_now:.6f}")

        # Scheduler step
        scheduler.step(avg_val)

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
            }, ckpt_dir / "best_model.pt")
            print(f"  ⭐ Saved best model (val={avg_val:.4f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
            }, ckpt_dir / f"checkpoint_epoch_{epoch}.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AuraNet V3 Optimized")
    # Data
    p.add_argument("--clean-dir", required=True, help="Directory of clean speech .wav files")
    p.add_argument("--noise-dir", required=True, help="Directory of noise .wav files")
    p.add_argument("--val-clean-dir", default=None, help="Validation clean dir (default: same as train)")
    p.add_argument("--val-noise-dir", default=None, help="Validation noise dir (default: same as train)")
    p.add_argument("--segment-length", type=float, default=3.0, help="Segment length in seconds")
    p.add_argument("--snr-low", type=float, default=-5, help="Min SNR (dB) for mixing")
    p.add_argument("--snr-high", type=float, default=20, help="Max SNR (dB) for mixing")
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
