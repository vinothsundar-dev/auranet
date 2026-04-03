# =============================================================================
# AuraNet V3 Optimized — Streaming Inference
# =============================================================================
# Processes 10 ms frames one at a time with persistent GRU hidden state.
# Suitable for real-time headset / cochlear-implant deployment.
# =============================================================================

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf

from model import AuraNetV3, create_model, apply_wdrc, N_FFT, HOP_LENGTH


class StreamingInference:
    """
    Frame-by-frame streaming processor.

    Each call to `process_frame()` accepts exactly `hop_length` new samples
    (10 ms @ 16 kHz = 160 samples) and returns `hop_length` enhanced samples.

    State maintained between calls:
        - GRU hidden state
        - STFT input buffer (overlapping window)
        - Overlap-add output buffer

    Latency: 1 STFT frame = hop_length / sample_rate = 10 ms algorithmic.
    """

    def __init__(self,
                 model: AuraNetV3,
                 device: str = "cpu",
                 apply_wdrc_gain: bool = True):
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        self.n_fft = model.n_fft
        self.hop = model.hop_length
        self.apply_wdrc_gain = apply_wdrc_gain

        # STFT analysis window
        self.window = torch.hann_window(self.n_fft, device=self.device)

        # Persistent state
        self.hidden: torch.Tensor | None = None        # GRU state
        self.input_buffer = torch.zeros(self.n_fft, device=self.device)  # ring buffer
        self.output_buffer = torch.zeros(self.n_fft, device=self.device) # OLA accumulator

    def reset(self):
        """Clear all internal state (call when starting a new stream)."""
        self.hidden = None
        self.input_buffer.zero_()
        self.output_buffer.zero_()

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame of `hop_length` samples.

        Args:
            frame: np.float32 array of shape (hop_length,)

        Returns:
            np.float32 array of shape (hop_length,) — enhanced audio
        """
        # Shift input buffer left by hop, append new frame
        self.input_buffer = torch.roll(self.input_buffer, -self.hop)
        self.input_buffer[-self.hop:] = torch.from_numpy(frame).to(self.device)

        # Windowed STFT of current buffer
        windowed = self.input_buffer * self.window
        spec = torch.fft.rfft(windowed)  # [F]
        stft_frame = torch.stack([spec.real, spec.imag], dim=0)  # [2, F]
        stft_frame = stft_frame.unsqueeze(0).unsqueeze(2)  # [1, 2, 1, F]

        # Run encoder → GRU → decoder
        encoded, skips = self.model.encoder(stft_frame)
        bn_out, gru_feat, self.hidden = self.model.bottleneck(encoded, self.hidden)
        raw_mask = self.model.decoder(bn_out, skips)

        if raw_mask.shape[2:] != stft_frame.shape[2:]:
            raw_mask = F.interpolate(raw_mask, size=stft_frame.shape[2:],
                                     mode='bilinear', align_corners=False)

        mask = torch.tanh(raw_mask)
        mr = mask[0, 0, 0]  # [F]
        mi = mask[0, 1, 0]  # [F]
        xr = stft_frame[0, 0, 0]  # [F]
        xi = stft_frame[0, 1, 0]  # [F]

        enh_real = mr * xr - mi * xi
        enh_imag = mr * xi + mi * xr
        enh_spec = torch.complex(enh_real, enh_imag)

        # iFFT → time-domain frame
        enh_time = torch.fft.irfft(enh_spec, n=self.n_fft)  # [n_fft]
        enh_time = enh_time * self.window

        # Overlap-add
        self.output_buffer += enh_time
        out_frame = self.output_buffer[:self.hop].clone()

        # Shift output buffer
        self.output_buffer = torch.roll(self.output_buffer, -self.hop)
        self.output_buffer[-self.hop:] = 0.0

        # Optional WDRC
        if self.apply_wdrc_gain:
            wdrc_params = self.model.wdrc(gru_feat)  # gru_feat: [1, 1, H]
            gain = wdrc_params["gain"][0, 0].item()
            out_frame = out_frame * gain

        return out_frame.cpu().numpy()


# ── File-based inference ─────────────────────────────────────────────────────

def process_file(model: AuraNetV3,
                 input_path: str,
                 output_path: str,
                 device: str = "cpu",
                 use_streaming: bool = True) -> float:
    """
    Enhance an audio file.

    Args:
        model: trained AuraNetV3
        input_path: path to noisy .wav
        output_path: path to save enhanced .wav
        device: cpu / cuda
        use_streaming: if True uses frame-by-frame; else full-file forward

    Returns:
        processing time in seconds
    """
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    assert sr == 16000, f"Expected 16 kHz, got {sr}"

    t0 = time.time()

    if use_streaming:
        # Frame-by-frame streaming
        streamer = StreamingInference(model, device=device, apply_wdrc_gain=True)
        hop = streamer.hop
        # Pad to full frames
        pad_len = (hop - len(audio) % hop) % hop
        audio_padded = np.pad(audio, (0, pad_len))

        output_frames = []
        for i in range(0, len(audio_padded), hop):
            frame = audio_padded[i:i + hop]
            out = streamer.process_frame(frame)
            output_frames.append(out)

        enhanced = np.concatenate(output_frames)[:len(audio)]
    else:
        # Full-file (non-streaming) mode
        model.eval()
        model.to(device)
        with torch.no_grad():
            noisy_t = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            enhanced_t, wdrc_p, _ = model(noisy_t)
            enhanced_t = apply_wdrc(enhanced_t, wdrc_p, hop_length=HOP_LENGTH)
            enhanced = enhanced_t.squeeze(0).cpu().numpy()

    elapsed = time.time() - t0

    # Normalize output to prevent clipping
    peak = np.abs(enhanced).max()
    if peak > 0.99:
        enhanced = enhanced * 0.99 / peak

    sf.write(output_path, enhanced, sr)
    return elapsed


# ── Batch processing ─────────────────────────────────────────────────────────

def process_directory(model: AuraNetV3,
                      input_dir: str,
                      output_dir: str,
                      device: str = "cpu",
                      use_streaming: bool = True):
    """Process all .wav files in a directory."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(in_dir.glob("**/*.wav")) + sorted(in_dir.glob("**/*.flac"))
    print(f"Processing {len(wav_files)} files...")

    total_time = 0.0
    total_audio = 0.0

    for f in wav_files:
        rel = f.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        elapsed = process_file(model, str(f), str(out_path),
                               device=device, use_streaming=use_streaming)

        info = sf.info(str(f))
        dur = info.duration
        total_time += elapsed
        total_audio += dur
        rtf = elapsed / dur if dur > 0 else 0
        print(f"  {rel.name}: {dur:.1f}s → {elapsed:.2f}s  (RTF={rtf:.3f})")

    if total_audio > 0:
        print(f"\nTotal: {total_audio:.1f}s audio in {total_time:.1f}s "
              f"(RTF={total_time/total_audio:.3f})")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AuraNet V3 Inference")
    p.add_argument("--input", required=True, help="Input .wav file or directory")
    p.add_argument("--output", required=True, help="Output .wav file or directory")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--streaming", action="store_true", default=True,
                   help="Use frame-by-frame streaming (default)")
    p.add_argument("--no-streaming", dest="streaming", action="store_false",
                   help="Use full-file forward pass")
    return p.parse_args()


def main():
    args = parse_args()

    # Load model
    model = create_model()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    inp = Path(args.input)
    if inp.is_dir():
        process_directory(model, args.input, args.output,
                          device=args.device, use_streaming=args.streaming)
    else:
        elapsed = process_file(model, args.input, args.output,
                               device=args.device, use_streaming=args.streaming)
        info = sf.info(args.input)
        rtf = elapsed / info.duration if info.duration > 0 else 0
        print(f"Done in {elapsed:.2f}s (RTF={rtf:.3f})")


if __name__ == "__main__":
    main()
