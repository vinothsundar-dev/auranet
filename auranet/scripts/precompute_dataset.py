#!/usr/bin/env python3
"""
precompute_dataset.py — AuraNet V3 Offline Pair Generator

Generates noisy-clean audio pairs using the full V3 augmentation pipeline,
saves each as a .pt file. CachedAuraNetDataset then loads these at >10×
the speed of live FLAC decoding + augmentation, eliminating the CPU bottleneck.

Each .pt file contains:
    {
        "noisy_audio": FloatTensor[N],   # 1-D waveform, 2s @ 16kHz = 32000 samples
        "clean_audio": FloatTensor[N],
    }

Usage:
    python scripts/precompute_dataset.py \\
        --clean_dir /kaggle/working/LibriSpeech/train-clean-100 \\
        --noise_dir  /path/to/noise \\
        --output_dir /kaggle/working/auranet_pt_cache \\
        --n_pairs 20000 \\
        --workers 4

Resumable: already-generated pair_XXXXXX.pt files are skipped.
"""

import argparse
import glob
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

# ── Make repo root importable ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_v3 import (
    apply_rir,
    apply_codec_chain,
    apply_lowpass,
    apply_clipping,
    generate_synthetic_rir,
)
from utils.audio_utils import (
    load_audio,
    mix_audio_with_noise,
    random_crop,
    apply_random_gain,
)

# ── Per-worker shared state ───────────────────────────────────────────────────
# Populated once in the worker process by _init(); not passed per-task.
_wstate: dict = {}


def _init(clean_paths, noise_paths, seg_samples, sample_rate, out_dir, snr_low, snr_high):
    """Worker initializer — runs once per worker process."""
    _wstate.update(
        clean=clean_paths,
        noise=noise_paths,
        seg=seg_samples,
        sr=sample_rate,
        out=out_dir,
        snr_low=snr_low,
        snr_high=snr_high,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_crop(audio: torch.Tensor, n: int) -> torch.Tensor:
    """Crop randomly, or wrap-pad if shorter than n."""
    if audio.shape[-1] >= n:
        return random_crop(audio, n)
    reps = (n // audio.shape[-1]) + 1
    audio = audio.repeat(1, reps) if audio.dim() == 2 else audio.repeat(reps)
    return audio[..., :n]


# ── Worker task ───────────────────────────────────────────────────────────────
def _generate_pair(idx: int) -> int:
    """
    Generate one noisy-clean pair and save it.

    Returns idx on success, -1 on error.
    Skips if output file already exists (allows resuming interrupted runs).
    """
    out_path = os.path.join(_wstate["out"], f"pair_{idx:06d}.pt")
    if os.path.exists(out_path):
        return idx  # Resumable — skip already generated

    # Per-pair deterministic RNG for clean/noise selection
    rng = random.Random(idx)

    try:
        clean, _ = load_audio(rng.choice(_wstate["clean"]), _wstate["sr"])
        noise, _ = load_audio(rng.choice(_wstate["noise"]), _wstate["sr"])

        clean = _safe_crop(clean, _wstate["seg"])
        noise = _safe_crop(noise, _wstate["seg"])

        # ── Augmentation pipeline (mirrors AuraNetV3Dataset) ──────────────
        clean = apply_random_gain(clean, -6.0, 6.0)

        if rng.random() < 0.3:
            rir = generate_synthetic_rir(length=1600, sample_rate=_wstate["sr"])
            clean = apply_rir(clean, rir, _wstate["sr"])

        if rng.random() < 0.2:
            clean = apply_codec_chain(clean, _wstate["sr"])
            noise = apply_codec_chain(noise, _wstate["sr"])
        elif rng.random() < 0.15:
            clean = apply_lowpass(clean, sample_rate=_wstate["sr"])
            noise = apply_lowpass(noise, sample_rate=_wstate["sr"])

        snr = rng.uniform(_wstate["snr_low"], _wstate["snr_high"])
        noisy, _ = mix_audio_with_noise(clean, noise, snr)

        if rng.random() < 0.05:
            noisy = apply_clipping(noisy)

        # Flatten to [N] for compact storage
        if clean.dim() == 2:
            clean = clean.squeeze(0)
        if noisy.dim() == 2:
            noisy = noisy.squeeze(0)

        torch.save({"noisy_audio": noisy, "clean_audio": clean}, out_path)
        return idx

    except Exception:
        return -1


# ── Main Entry Point ──────────────────────────────────────────────────────────
def precompute(
    clean_dir: str,
    noise_dir: str,
    output_dir: str,
    n_pairs: int = 20000,
    workers: int = 4,
    segment_seconds: float = 2.0,
    sample_rate: int = 16000,
    snr_range: tuple = (-5.0, 25.0),
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Scan source directories
    clean_paths = []
    for ext in ("**/*.flac", "**/*.wav", "**/*.mp3"):
        clean_paths.extend(glob.glob(os.path.join(clean_dir, ext), recursive=True))

    noise_paths = []
    for ext in ("**/*.flac", "**/*.wav"):
        noise_paths.extend(glob.glob(os.path.join(noise_dir, ext), recursive=True))

    if not clean_paths:
        raise FileNotFoundError(f"No audio files found in: {clean_dir}")
    if not noise_paths:
        raise FileNotFoundError(f"No audio files found in: {noise_dir}")

    seg_samples = int(segment_seconds * sample_rate)
    existing = len(glob.glob(os.path.join(output_dir, "pair_*.pt")))
    pair_bytes = seg_samples * 4 * 2  # 2 tensors × float32
    est_gb = (n_pairs * pair_bytes) / (1024 ** 3)

    print(f"📦 AuraNet V3 — Offline Pair Precomputation")
    print(f"   Clean:    {len(clean_paths):,} files  ({clean_dir})")
    print(f"   Noise:    {len(noise_paths):,} files  ({noise_dir})")
    print(f"   Output:   {output_dir}")
    print(f"   Pairs:    {n_pairs:,}  (est. {est_gb:.1f} GB on disk)")
    print(f"   Segment:  {segment_seconds}s → {seg_samples} samples")
    print(f"   SNR:      [{snr_range[0]}, {snr_range[1]}] dB")
    print(f"   Workers:  {workers}")
    if existing:
        print(f"   Resuming: {existing:,} pairs already exist — skipping")
    print()

    try:
        from tqdm.auto import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    t0 = time.time()
    ok = 0
    errors = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init,
        initargs=(
            clean_paths, noise_paths,
            seg_samples, sample_rate, output_dir,
            snr_range[0], snr_range[1],
        ),
    ) as pool:
        futures = {pool.submit(_generate_pair, i): i for i in range(n_pairs)}

        if use_tqdm:
            pbar = tqdm(total=n_pairs, unit="pair", desc="Generating pairs")

        for fut in as_completed(futures):
            result = fut.result()
            if result >= 0:
                ok += 1
            else:
                errors += 1
            if use_tqdm:
                pbar.update(1)
                if ok % 500 == 0 and ok > 0:
                    elapsed = time.time() - t0
                    pbar.set_postfix({"speed": f"{ok/elapsed:.1f} pairs/s"})

        if use_tqdm:
            pbar.close()

    elapsed = time.time() - t0
    final_count = len(glob.glob(os.path.join(output_dir, "pair_*.pt")))
    size_gb = sum(
        os.path.getsize(f)
        for f in glob.glob(os.path.join(output_dir, "pair_*.pt"))
    ) / (1024 ** 3)

    print(f"\n✅ Done: {final_count:,} pairs in {elapsed/60:.1f} min")
    print(f"   Disk usage:  {size_gb:.2f} GB")
    print(f"   Throughput:  {ok / elapsed:.1f} pairs/sec")
    if errors:
        print(f"   ⚠️  Errors:   {errors} pairs failed (skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute AuraNet V3 dataset pairs offline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clean_dir",  required=True,  help="Clean speech directory")
    parser.add_argument("--noise_dir",  required=True,  help="Noise directory")
    parser.add_argument("--output_dir", required=True,  help="Output .pt cache directory")
    parser.add_argument("--n_pairs",    type=int,   default=20000, help="Number of pairs to generate")
    parser.add_argument("--workers",    type=int,   default=4,     help="Parallel worker processes")
    parser.add_argument("--segment_seconds", type=float, default=2.0,  help="Segment length in seconds")
    parser.add_argument("--sample_rate",     type=int,   default=16000, help="Audio sample rate")
    parser.add_argument("--snr_low",  type=float, default=-5.0, help="Minimum SNR dB")
    parser.add_argument("--snr_high", type=float, default=25.0, help="Maximum SNR dB")
    args = parser.parse_args()

    precompute(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        n_pairs=args.n_pairs,
        workers=args.workers,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        snr_range=(args.snr_low, args.snr_high),
    )
