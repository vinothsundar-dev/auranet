"""
Download a 5GB shuffled subset of DNS Challenge dataset.
Run this in Google Colab or locally.
"""

import os
import random
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# Configuration
SUBSET_SIZE_GB = 5
SAMPLE_RATE = 16000
TARGET_HOURS = 100  # ~5GB at 16kHz mono

# DNS Challenge URLs (from Microsoft's DNS Challenge)
DNS_URLS = {
    "clean_speech": [
        # LibriSpeech clean (smaller chunks)
        "https://www.openslr.org/resources/12/train-clean-100.tar.gz",  # 6.3GB - we'll sample from this
    ],
    "noise": [
        # Audioset noise samples (curated subset)
        "https://zenodo.org/record/3678171/files/demand.zip",  # 1.6GB
    ]
}

# Alternative: Use torchaudio datasets (easier)
def download_with_torchaudio(data_dir, num_hours=50):
    """Download using torchaudio - recommended approach."""
    import torchaudio
    import torch
    import soundfile as sf
    
    print("📥 Downloading LibriSpeech subset for clean speech...")
    
    clean_dir = Path(data_dir) / "speech"
    noise_dir = Path(data_dir) / "noise"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    
    # Download LibriSpeech train-clean-100 (6.3GB, ~100 hours)
    # We'll sample ~50 hours from it
    try:
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(data_dir),
            url="train-clean-100",
            download=True
        )
        
        # Shuffle and save subset
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        total_duration = 0
        target_duration = num_hours * 3600  # Convert to seconds
        
        print(f"📊 Selecting ~{num_hours} hours of speech...")
        for i, idx in enumerate(tqdm(indices)):
            if total_duration >= target_duration:
                break
                
            waveform, sample_rate, *_ = dataset[idx]
            duration = waveform.shape[1] / sample_rate
            total_duration += duration
            
            # Resample to 16kHz if needed
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
            
            # Save as WAV
            output_path = clean_dir / f"speech_{i:05d}.wav"
            torchaudio.save(str(output_path), waveform, SAMPLE_RATE)
        
        print(f"✅ Saved {total_duration/3600:.1f} hours of clean speech")
        
    except Exception as e:
        print(f"⚠️ LibriSpeech download failed: {e}")
        print("   Falling back to synthetic generation...")
        generate_synthetic_speech(clean_dir, num_hours=10)
    
    # Download DEMAND noise dataset
    download_demand_noise(noise_dir)
    
    return clean_dir, noise_dir


def download_demand_noise(noise_dir):
    """Download DEMAND noise dataset (~1.6GB)."""
    print("\n📥 Downloading DEMAND noise dataset...")
    
    url = "https://zenodo.org/record/1227121/files/demand.zip"
    zip_path = Path(noise_dir).parent / "demand.zip"
    
    try:
        # Download
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='MB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract
        print("📦 Extracting noise files...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(noise_dir)
        
        # Cleanup
        zip_path.unlink()
        print("✅ DEMAND noise dataset ready")
        
    except Exception as e:
        print(f"⚠️ DEMAND download failed: {e}")
        print("   Generating synthetic noise...")
        generate_synthetic_noise(noise_dir)


def generate_synthetic_speech(output_dir, num_hours=10):
    """Generate synthetic speech-like signals for testing."""
    import numpy as np
    import soundfile as sf
    
    print(f"🔧 Generating {num_hours} hours of synthetic speech signals...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    duration_per_file = 10  # seconds
    num_files = int(num_hours * 3600 / duration_per_file)
    
    for i in tqdm(range(num_files)):
        t = np.linspace(0, duration_per_file, SAMPLE_RATE * duration_per_file)
        
        # Create speech-like signal with varying pitch
        base_freq = np.random.uniform(80, 250)  # Fundamental frequency
        signal = np.zeros_like(t)
        
        # Add harmonics with varying amplitudes
        for h in range(1, 8):
            amp = 1.0 / (h ** 1.5) * np.random.uniform(0.5, 1.5)
            freq = base_freq * h * (1 + 0.02 * np.sin(2 * np.pi * 3 * t))  # Vibrato
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        # Add amplitude modulation (syllable-like)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) * np.sin(2 * np.pi * 0.3 * t)
        signal *= envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.7
        
        sf.write(output_dir / f"synth_speech_{i:05d}.wav", signal.astype(np.float32), SAMPLE_RATE)
    
    print(f"✅ Generated {num_files} synthetic speech files")


def generate_synthetic_noise(output_dir, num_files=50):
    """Generate various noise types."""
    import numpy as np
    import soundfile as sf
    
    print(f"🔧 Generating {num_files} synthetic noise files...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    duration = 60  # 1 minute each
    
    noise_types = ['white', 'pink', 'brown', 'babble', 'traffic']
    
    for i in tqdm(range(num_files)):
        noise_type = random.choice(noise_types)
        samples = SAMPLE_RATE * duration
        
        if noise_type == 'white':
            noise = np.random.randn(samples)
        elif noise_type == 'pink':
            # Pink noise (1/f)
            white = np.random.randn(samples)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(samples)
            freqs[0] = 1
            fft = fft / np.sqrt(freqs)
            noise = np.fft.irfft(fft, samples)
        elif noise_type == 'brown':
            # Brown noise (1/f^2)
            noise = np.cumsum(np.random.randn(samples))
            noise = noise - np.mean(noise)
        elif noise_type == 'babble':
            # Simulate babble with overlapping tones
            noise = np.zeros(samples)
            for _ in range(20):
                freq = np.random.uniform(100, 400)
                phase = np.random.uniform(0, 2 * np.pi)
                t = np.linspace(0, duration, samples)
                noise += np.sin(2 * np.pi * freq * t + phase) * np.random.uniform(0.1, 0.3)
            noise += np.random.randn(samples) * 0.3
        else:  # traffic
            noise = np.random.randn(samples)
            # Low-pass filter simulation
            from scipy.ndimage import uniform_filter1d
            noise = uniform_filter1d(noise, size=100)
            noise += np.random.randn(samples) * 0.1
        
        # Normalize
        noise = noise / np.max(np.abs(noise)) * 0.8
        
        sf.write(output_dir / f"{noise_type}_{i:03d}.wav", noise.astype(np.float32), SAMPLE_RATE)
    
    print(f"✅ Generated {num_files} noise files")


def download_dns_subset(data_dir, target_gb=5):
    """Main function to download ~5GB DNS-like dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Target: ~{target_gb}GB dataset")
    print(f"📂 Output: {data_dir}\n")
    
    # Calculate hours based on target size
    # 16kHz mono WAV: ~115MB per hour
    target_hours = (target_gb * 1024) / 115
    
    clean_dir, noise_dir = download_with_torchaudio(
        data_dir, 
        num_hours=int(target_hours * 0.8)  # 80% speech, 20% noise
    )
    
    print(f"\n✅ Dataset ready!")
    print(f"   📂 Speech: {clean_dir}")
    print(f"   📂 Noise: {noise_dir}")
    
    # Count files
    speech_count = len(list(clean_dir.glob("*.wav")))
    noise_count = len(list(noise_dir.glob("**/*.wav")))
    print(f"   📊 {speech_count} speech files, {noise_count} noise files")
    
    return clean_dir, noise_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./datasets", help="Output directory")
    parser.add_argument("--size", type=int, default=5, help="Target size in GB")
    args = parser.parse_args()
    
    download_dns_subset(args.output, target_gb=args.size)
