"""
Robust DEMAND Noise Dataset Downloader & Preprocessor
======================================================
Downloads the DEMAND (Diverse Environments Multichannel Acoustic Noise Database)
from Zenodo, converts to 16kHz mono, and chunks into 10-second segments.

Falls back to MUSAN noise if DEMAND download fails entirely.

Usage:
    python scripts/download_demand.py --output /path/to/noise/dir

For Kaggle/Colab, import and call download_and_prepare_demand() directly.
"""

import os
import sys
import time
import glob
import shutil
import zipfile
import hashlib
import argparse
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
from tqdm.auto import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
MIN_CHUNK_SECONDS = 3

# 15 environments available at 16kHz on Zenodo (SCAFE only has 48k)
DEMAND_ENVIRONMENTS = [
    "DKITCHEN", "DLIVING", "DWASHING",
    "NFIELD", "NPARK", "NRIVER",
    "OHALLWAY", "OMEETING", "OOFFICE",
    "PCAFETER", "PRESTO", "PSTATION",
    "SPSQUARE", "STRAFFIC",
    "TBUS", "TCAR", "TMETRO",
]

# MD5 checksums for integrity verification (from Zenodo)
DEMAND_MD5 = {
    "DKITCHEN_16k.zip": "7ffbf52d7f4699f96927846103dc8788",
    "DLIVING_16k.zip": "46741384d9e434a0bd8b3ec1830b6052",
    "DWASHING_16k.zip": "7e5ee9437ce9409c5f9a779b6212a240",
    "NFIELD_16k.zip": "a740046c6f4e174e16f5d568aaec5024",
    "NPARK_16k.zip": "80f1385a34d7f1705758926b57f138ce",
    "NRIVER_16k.zip": "54264db61d3fe073fb81f2e40e0d19b5",
    "OHALLWAY_16k.zip": "fe918bbb0e63e73d09ba7f4843ef33f1",
    "OMEETING_16k.zip": "62f7cfe7fe6d30b7d8a215fe37c2dfd2",
    "OOFFICE_16k.zip": "7b61cc2d182d5a654cb9c3101ddd4041",
    "PCAFETER_16k.zip": "99927d148128254141a9417d051510bb",
    "PRESTO_16k.zip": "b98d2e6854eeebb397f29a8ad7457092",
    "PSTATION_16k.zip": "d7448009f6c2aeb6ba570375df1750a3",
    "SPSQUARE_16k.zip": "205d0e7b8fe74504a2f8d252fc414b9e",
    "STRAFFIC_16k.zip": "2efa87262f272bbf9ba578088e81939c",
    "TBUS_16k.zip": "706b11b0d8504f9f3b3f3211e91b3863",
    "TCAR_16k.zip": "4d930012796bd298932245a26189f973",
    "TMETRO_16k.zip": "95daf4df678e13b120e14211e6d89571",
}

# URL templates (tried in order)
ZENODO_URLS = [
    "https://zenodo.org/records/1227121/files/{filename}?download=1",
    "https://zenodo.org/record/1227121/files/{filename}?download=1",
    "https://zenodo.org/api/records/1227121/files/{filename}/content",
]

# MUSAN fallback
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ═══════════════════════════════════════════════════════════════════════════════
# Download utilities
# ═══════════════════════════════════════════════════════════════════════════════

def md5_file(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, output_path: str, max_retries: int = MAX_RETRIES) -> bool:
    """Download a file with retry logic. Returns True on success."""
    for attempt in range(1, max_retries + 1):
        try:
            # Try wget first (most reliable on Kaggle/Colab)
            result = subprocess.run(
                ["wget", "-q", "--tries=2", "--timeout=60", "-O", output_path, url],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0 and os.path.exists(output_path):
                size = os.path.getsize(output_path)
                if size > 1000:  # Not an error page
                    return True
                else:
                    os.remove(output_path)

            # Fallback to curl
            result = subprocess.run(
                ["curl", "-sL", "--retry", "2", "--connect-timeout", "30",
                 "-o", output_path, url],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0 and os.path.exists(output_path):
                size = os.path.getsize(output_path)
                if size > 1000:
                    return True
                else:
                    os.remove(output_path)

        except (subprocess.TimeoutExpired, Exception) as e:
            if attempt < max_retries:
                print(f"      Retry {attempt}/{max_retries} ({e})")
                time.sleep(RETRY_DELAY * attempt)
            continue

        if attempt < max_retries:
            print(f"      Retry {attempt}/{max_retries}")
            time.sleep(RETRY_DELAY * attempt)

    return False


def download_demand_env(env: str, demand_dir: str) -> bool:
    """Download and extract a single DEMAND environment. Returns True on success."""
    filename = f"{env}_16k.zip"
    zip_path = os.path.join(demand_dir, filename)
    env_dir = os.path.join(demand_dir, env)

    # Already extracted
    if os.path.exists(env_dir) and glob.glob(os.path.join(env_dir, "*.wav")):
        return True

    # Try each URL template
    for url_template in ZENODO_URLS:
        url = url_template.format(filename=filename)
        if download_file(url, zip_path):
            # Verify MD5 if we have checksum
            expected_md5 = DEMAND_MD5.get(filename)
            if expected_md5:
                actual_md5 = md5_file(zip_path)
                if actual_md5 != expected_md5:
                    print(f"      ⚠️  MD5 mismatch for {filename} (got {actual_md5[:8]}...)")
                    os.remove(zip_path)
                    continue

            # Extract
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(demand_dir)
                os.remove(zip_path)

                if os.path.exists(env_dir) and glob.glob(os.path.join(env_dir, "*.wav")):
                    return True
            except zipfile.BadZipFile:
                print(f"      ⚠️  Corrupt zip: {filename}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                continue

    # Clean up partial download
    if os.path.exists(zip_path):
        os.remove(zip_path)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Audio preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def process_wav_to_chunks(wav_path: str, output_dir: str, prefix: str,
                          counter: int) -> Tuple[int, float]:
    """
    Load WAV, convert to 16kHz mono, chunk into 10s segments.
    Returns (num_chunks_written, total_duration_seconds).
    """
    try:
        data, sr = sf.read(wav_path, dtype='float32')
    except Exception:
        return 0, 0.0

    # Convert to mono
    if len(data.shape) > 1:
        data = data[:, 0]  # Use channel 1

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample as scipy_resample
        num_samples = int(len(data) * SAMPLE_RATE / sr)
        data = scipy_resample(data, num_samples).astype(np.float32)

    # Normalize to [-1, 1]
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak

    # Chunk into CHUNK_SECONDS segments
    chunk_len = CHUNK_SECONDS * SAMPLE_RATE
    min_len = MIN_CHUNK_SECONDS * SAMPLE_RATE
    chunks_written = 0
    total_duration = 0.0

    for start in range(0, len(data), chunk_len):
        chunk = data[start:start + chunk_len]
        if len(chunk) < min_len:
            continue  # Skip very short clips

        # Pad last chunk if needed (only if > min_len)
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))

        out_name = f"{prefix}_{counter + chunks_written:04d}.wav"
        sf.write(os.path.join(output_dir, out_name), chunk, SAMPLE_RATE)
        chunks_written += 1
        total_duration += len(chunk) / SAMPLE_RATE

    return chunks_written, total_duration


# ═══════════════════════════════════════════════════════════════════════════════
# MUSAN fallback
# ═══════════════════════════════════════════════════════════════════════════════

def download_musan_fallback(noise_dir: str) -> Tuple[int, float]:
    """Download MUSAN noise as fallback. Returns (num_files, total_hours)."""
    print("\n🔄 DEMAND failed — falling back to MUSAN noise dataset...")
    musan_dir = os.path.join(os.path.dirname(noise_dir), "musan_raw")
    musan_tar = musan_dir + ".tar.gz"
    os.makedirs(musan_dir, exist_ok=True)

    # Download MUSAN (~11 GB, but we only need noise subset ~1 GB)
    print("   ⬇️  Downloading MUSAN from OpenSLR (noise subset)...")
    if not download_file(MUSAN_URL, musan_tar):
        print("   ❌ MUSAN download also failed")
        return 0, 0.0

    # Extract only noise directory
    print("   📦 Extracting noise files...")
    try:
        subprocess.run(
            ["tar", "xzf", musan_tar, "-C", musan_dir, "--wildcards", "musan/noise/*"],
            capture_output=True, text=True, timeout=600,
        )
    except Exception:
        subprocess.run(
            ["tar", "xzf", musan_tar, "-C", musan_dir],
            capture_output=True, text=True, timeout=600,
        )

    if os.path.exists(musan_tar):
        os.remove(musan_tar)

    # Process MUSAN noise files
    musan_noise = os.path.join(musan_dir, "musan", "noise")
    if not os.path.exists(musan_noise):
        print("   ❌ MUSAN extraction failed")
        return 0, 0.0

    total_chunks = 0
    total_duration = 0.0
    wav_files = glob.glob(os.path.join(musan_noise, "**", "*.wav"), recursive=True)
    for wav_file in wav_files:
        chunks, dur = process_wav_to_chunks(
            wav_file, noise_dir, "musan", total_chunks
        )
        total_chunks += chunks
        total_duration += dur

    # Clean up raw download
    shutil.rmtree(musan_dir, ignore_errors=True)

    hours = total_duration / 3600
    print(f"   ✅ MUSAN fallback: {total_chunks} chunks ({hours:.1f} hours)")
    return total_chunks, hours


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def download_and_prepare_demand(
    noise_dir: str,
    demand_dir: Optional[str] = None,
    environments: Optional[list] = None,
    skip_existing: bool = True,
    fallback_musan: bool = True,
) -> dict:
    """
    Download DEMAND dataset, preprocess to 16kHz mono, chunk into 10s segments.

    Args:
        noise_dir: Output directory for processed noise WAVs
        demand_dir: Where to store raw DEMAND downloads (default: sibling of noise_dir)
        environments: List of environments to download (default: all 17)
        skip_existing: Skip if noise_dir already has DEMAND files
        fallback_musan: Fall back to MUSAN if DEMAND fails

    Returns:
        dict with 'environments', 'total_files', 'total_hours', 'source'
    """
    os.makedirs(noise_dir, exist_ok=True)
    if demand_dir is None:
        demand_dir = os.path.join(os.path.dirname(noise_dir), "DEMAND")
    os.makedirs(demand_dir, exist_ok=True)

    if environments is None:
        environments = DEMAND_ENVIRONMENTS

    # Check if already processed
    existing_demand = glob.glob(os.path.join(noise_dir, "demand_*.wav"))
    if skip_existing and len(existing_demand) > 50:
        hours = len(existing_demand) * CHUNK_SECONDS / 3600
        print(f"✅ DEMAND noise already present: {len(existing_demand)} chunks (~{hours:.1f}h)")
        return {
            'environments': '(cached)',
            'total_files': len(existing_demand),
            'total_hours': hours,
            'source': 'demand_cached',
        }

    print(f"⬇️  Downloading DEMAND noise dataset ({len(environments)} environments)...")
    print(f"   Source: Zenodo (doi:10.5281/zenodo.1227121)")
    print(f"   Target: {noise_dir}")

    # ── Download environments (single tqdm bar) ──
    downloaded_envs = []
    failed_envs = []
    t0 = time.time()

    pbar = tqdm(environments, desc="Downloading DEMAND", unit="env")
    for env in pbar:
        if download_demand_env(env, demand_dir):
            downloaded_envs.append(env)
        else:
            failed_envs.append(env)
        pbar.set_postfix(ok=len(downloaded_envs), fail=len(failed_envs), current=env)
    pbar.close()

    if failed_envs:
        print(f"   ⚠️  Failed: {', '.join(failed_envs)}")

    # ── Process downloaded environments ──
    total_chunks = 0
    total_duration = 0.0
    env_stats = {}

    if downloaded_envs:
        pbar = tqdm(downloaded_envs, desc="Processing DEMAND", unit="env")
        for env in pbar:
            env_dir = os.path.join(demand_dir, env)
            wav_files = sorted(glob.glob(os.path.join(env_dir, "*ch01*.wav")))
            if not wav_files:
                wav_files = sorted(glob.glob(os.path.join(env_dir, "*.wav")))[:1]

            env_chunks = 0
            env_duration = 0.0
            for wav_file in wav_files:
                chunks, dur = process_wav_to_chunks(
                    wav_file, noise_dir, f"demand", total_chunks
                )
                total_chunks += chunks
                env_chunks += chunks
                total_duration += dur
                env_duration += dur

            env_stats[env] = {'chunks': env_chunks, 'duration_s': env_duration}
            pbar.set_postfix(chunks=total_chunks, duration=f"{total_duration/60:.0f}m")
        pbar.close()

    total_hours = total_duration / 3600

    # ── Fallback to MUSAN if DEMAND mostly failed ──
    if len(downloaded_envs) < 5 and fallback_musan:
        musan_files, musan_hours = download_musan_fallback(noise_dir)
        total_chunks += musan_files
        total_hours += musan_hours
        source = 'musan_fallback' if len(downloaded_envs) == 0 else 'demand+musan'
    else:
        source = 'demand'

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"📊 DEMAND Noise Dataset Summary")
    print(f"{'='*55}")
    print(f"  Source:          {source}")
    print(f"  Environments:    {len(downloaded_envs)}/{len(environments)}")
    if env_stats:
        for env, stats in env_stats.items():
            print(f"    └─ {env:12s} {stats['chunks']:3d} chunks ({stats['duration_s']/60:.1f} min)")
    print(f"  Total files:     {total_chunks:,}")
    print(f"  Total duration:  {total_hours:.1f} hours")
    print(f"  Format:          {SAMPLE_RATE} Hz, mono, {CHUNK_SECONDS}s chunks")
    print(f"  Output:          {noise_dir}")
    print(f"{'='*55}")

    return {
        'environments': downloaded_envs,
        'total_files': total_chunks,
        'total_hours': total_hours,
        'source': source,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & prepare DEMAND noise dataset")
    parser.add_argument("--output", type=str, default="data/noise",
                        help="Output directory for processed noise WAVs")
    parser.add_argument("--demand-dir", type=str, default=None,
                        help="Directory for raw DEMAND downloads")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable MUSAN fallback")
    args = parser.parse_args()

    download_and_prepare_demand(
        noise_dir=args.output,
        demand_dir=args.demand_dir,
        fallback_musan=not args.no_fallback,
    )
