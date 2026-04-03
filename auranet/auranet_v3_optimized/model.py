# =============================================================================
# AuraNet V3 Optimized — Model Definition
# =============================================================================
# Lightweight causal CRN for real-time speech enhancement.
# Optimized for <3 MB, <10 ms latency, CPU/ARM deployment.
#
# Architecture:
#   STFT [B,2,T,161] → Encoder (DSConv2d ×4) → GRU(2-layer, h=128)
#   → Decoder (U-Net skips) → complex mask [B,2,T,161] → iSTFT
#   + WDRC sidechain for hearing-device dynamic range control
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


# ── STFT parameters ─────────────────────────────────────────────────────────
N_FFT = 320          # 20 ms window at 16 kHz
HOP_LENGTH = 160     # 10 ms hop → <10 ms latency
FREQ_BINS = N_FFT // 2 + 1  # 161


# ── Building blocks ─────────────────────────────────────────────────────────

class CausalConv2d(nn.Module):
    """2-D convolution with causal (left-only) padding in time."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1),
                 groups=1, bias=True):
        super().__init__()
        kt, kf = kernel_size
        self.pad_time = kt - 1  # causal: pad only left
        self.pad_freq = (kf - 1) // 2  # symmetric
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=(0, self.pad_freq), groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_time > 0:
            x = F.pad(x, (0, 0, self.pad_time, 0))
        return self.conv(x)


class CausalTransposeConv2d(nn.Module):
    """Transposed 2-D conv for frequency upsampling (time preserved)."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 2),
                 groups=1, bias=True):
        super().__init__()
        kt, kf = kernel_size
        self.time_crop = kt - 1
        out_pad = (0, stride[1] - 1) if stride[1] > 1 else (0, 0)
        self.conv_t = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=(0, kf // 2), output_padding=out_pad,
            groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_t(x)
        if self.time_crop > 0 and out.shape[2] > x.shape[2]:
            out = out[:, :, :x.shape[2], :]
        return out


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable 2-D convolution (causal in time)."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.dw = CausalConv2d(in_ch, in_ch, kernel_size, stride=stride,
                               groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DepthwiseSeparableTransposeConv2d(nn.Module):
    """Depthwise-separable transposed 2-D convolution."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 2)):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.dw_t = CausalTransposeConv2d(out_ch, out_ch, kernel_size,
                                           stride=stride, groups=out_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_t(self.pw(x))


# ── Encoder ──────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 2)):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(in_ch, out_ch, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """
    4-block encoder: 2 → 16 → 32 → 64 → 80
    Frequency: 161 → 81 → 41 → 21 → 11
    """
    CHANNELS = (16, 32, 64, 80)

    def __init__(self, in_channels: int = 2,
                 channels: Tuple[int, ...] = CHANNELS):
        super().__init__()
        blocks = []
        ch = in_channels
        for out_ch in channels:
            blocks.append(EncoderBlock(ch, out_ch))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        skips = []
        for blk in self.blocks:
            x = blk(x)
            skips.append(x)
        return x, skips


# ── Temporal Bottleneck ──────────────────────────────────────────────────────

class TemporalBottleneck(nn.Module):
    """
    2-layer unidirectional GRU.
    Maintains hidden state across frames for streaming.
    """

    def __init__(self, input_channels: int = 80, input_freq_bins: int = 11,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_channels = input_channels
        self.input_freq_bins = input_freq_bins
        self.hidden_size = hidden_size
        self.input_size = input_channels * input_freq_bins

        self.norm_in = nn.LayerNorm(self.input_size)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.projection = nn.Linear(hidden_size, self.input_size)
        self.norm_out = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, T, Fq = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B, T, -1)
        residual = x_flat

        x_flat = self.norm_in(x_flat)
        gru_out, hidden_out = self.gru(x_flat, hidden)
        proj = self.norm_out(self.projection(gru_out)) + residual

        decoded = proj.reshape(B, T, C, Fq).permute(0, 2, 1, 3)
        return decoded, gru_out, hidden_out


# ── Decoder ──────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Skip connections via 1×1 projection + addition (not concat)."""

    def __init__(self, in_ch, out_ch, skip_ch, kernel_size=(3, 3), stride=(1, 2)):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_ch, in_ch, 1, bias=False)
        self.conv = DepthwiseSeparableTransposeConv2d(in_ch, out_ch, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = x + self.skip_proj(skip)
        return self.act(self.norm(self.conv(x)))


class Decoder(nn.Module):
    """
    U-Net decoder: 80 → 64 → 32 → 16 channels.
    Final conv outputs 2 channels (complex mask: real + imag).
    """
    CHANNELS = (80, 64, 32, 16)

    def __init__(self, in_channels: int = 80,
                 channels: Tuple[int, ...] = CHANNELS,
                 encoder_channels: Tuple[int, ...] = (16, 32, 64, 80)):
        super().__init__()
        skip_channels = list(reversed(encoder_channels))  # [96,64,32,16]

        blocks = []
        ch = in_channels
        for i, out_ch in enumerate(channels):
            blocks.append(DecoderBlock(ch, out_ch, skip_channels[i]))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # Final 1×1 conv to 2-channel mask
        self.mask_conv = nn.Conv2d(ch, 2, 1, bias=True)

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        skips_r = skips[::-1]
        for i, blk in enumerate(self.blocks):
            x = blk(x, skips_r[i])
        return self.mask_conv(x)


# ── WDRC Sidechain ───────────────────────────────────────────────────────────

class NeuralWDRC(nn.Module):
    """
    Wide Dynamic Range Compression for hearing devices.
    MLP: 128 → 64 → 32 → 4 (attack, release, ratio, gain).
    """

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 4),
        )

    def forward(self, gru_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            gru_out: [B, T, hidden_size]
        Returns:
            dict with attack, release, ratio, gain — each [B, T]
        """
        params = self.mlp(gru_out)
        return {
            "attack":  torch.sigmoid(params[..., 0]),
            "release": torch.sigmoid(params[..., 1]),
            "ratio":   torch.clamp(F.softplus(params[..., 2]) + 1.0, 1.0, 20.0),
            "gain":    torch.sigmoid(params[..., 3]) * 2.0,
        }


def apply_wdrc(audio: torch.Tensor,
               wdrc_params: Dict[str, torch.Tensor],
               hop_length: int = HOP_LENGTH,
               threshold: float = 0.3,
               eps: float = 1e-8) -> torch.Tensor:
    """Apply frame-level WDRC to time-domain audio [B, N]."""
    B, N = audio.shape

    ratio = F.interpolate(wdrc_params["ratio"].unsqueeze(1), size=N,
                          mode='linear', align_corners=False).squeeze(1)
    gain = F.interpolate(wdrc_params["gain"].unsqueeze(1), size=N,
                         mode='linear', align_corners=False).squeeze(1)

    env = torch.abs(audio)
    compressed = torch.where(
        env > threshold,
        threshold + (env - threshold) / ratio,
        env,
    )
    return audio * (compressed / (env + eps)) * gain


# ── AuraNet V3 Optimized ────────────────────────────────────────────────────

class AuraNetV3(nn.Module):
    """
    Production-ready causal CRN.

    Input:  raw waveform [B, 1, samples] at 16 kHz
    Output: enhanced waveform [B, 1, samples]
    """

    def __init__(self,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH,
                 encoder_channels: Tuple[int, ...] = (16, 32, 64, 80),
                 gru_hidden: int = 128,
                 gru_layers: int = 2,
                 decoder_channels: Tuple[int, ...] = (80, 64, 32, 16)):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        freq_bins = n_fft // 2 + 1  # 161

        # STFT window (registered as buffer — not a parameter)
        self.register_buffer("window", torch.hann_window(n_fft))

        # Encoder
        self.encoder = Encoder(in_channels=2, channels=encoder_channels)

        # Compute encoded frequency dimension
        f = freq_bins
        for _ in encoder_channels:
            f = (f + 1) // 2
        enc_freq_bins = f

        # GRU bottleneck
        self.bottleneck = TemporalBottleneck(
            input_channels=encoder_channels[-1],
            input_freq_bins=enc_freq_bins,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=encoder_channels[-1],
            channels=decoder_channels,
            encoder_channels=encoder_channels,
        )

        # WDRC sidechain
        self.wdrc = NeuralWDRC(input_dim=gru_hidden)

    # ── helpers ──

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Waveform [B, N] → real+imag [B, 2, T, F]."""
        spec = torch.stft(x, self.n_fft, self.hop_length,
                          window=self.window, return_complex=True)  # [B, F, T]
        return torch.stack([spec.real, spec.imag], dim=1).transpose(2, 3)  # [B,2,T,F]

    def istft(self, stft_out: torch.Tensor, length: int) -> torch.Tensor:
        """real+imag [B, 2, T, F] → waveform [B, N]."""
        stft_out = stft_out.transpose(2, 3)  # [B,2,F,T]
        spec = torch.complex(stft_out[:, 0], stft_out[:, 1])  # [B, F, T]
        return torch.istft(spec, self.n_fft, self.hop_length,
                           window=self.window, length=length)

    # ── forward ──

    def forward(self, noisy: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            noisy: [B, 1, N] or [B, N] raw noisy waveform at 16 kHz
            hidden: optional GRU hidden state for streaming

        Returns:
            enhanced: [B, N] enhanced waveform
            wdrc_params: dict of WDRC parameters
            hidden_out: updated GRU hidden state
        """
        if noisy.dim() == 3:
            noisy = noisy.squeeze(1)

        orig_len = noisy.shape[-1]
        noisy_stft = self.stft(noisy)  # [B, 2, T, F]

        # Encode
        encoded, skips = self.encoder(noisy_stft)

        # Temporal modeling
        bottleneck_out, gru_features, hidden_out = self.bottleneck(encoded, hidden)

        # Decode → raw mask [B, 2, T, F]
        raw_mask = self.decoder(bottleneck_out, skips)

        # Ensure mask matches STFT shape
        if raw_mask.shape[2:] != noisy_stft.shape[2:]:
            raw_mask = F.interpolate(raw_mask, size=noisy_stft.shape[2:],
                                     mode='bilinear', align_corners=False)

        # Bounded mask via tanh
        mask = torch.tanh(raw_mask)

        # Complex masking: mask_real * noisy_real − mask_imag * noisy_imag, etc.
        mr, mi = mask[:, 0:1], mask[:, 1:2]
        xr, xi = noisy_stft[:, 0:1], noisy_stft[:, 1:2]
        enhanced_stft = torch.cat([mr * xr - mi * xi,
                                   mr * xi + mi * xr], dim=1)

        # Reconstruct waveform
        enhanced = self.istft(enhanced_stft, length=orig_len)

        # WDRC parameters
        wdrc_params = self.wdrc(gru_features)

        return enhanced, wdrc_params, hidden_out

    # ── ONNX / deployment helpers ──

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6


# ── Factory ──────────────────────────────────────────────────────────────────

def create_model(config: Optional[dict] = None) -> AuraNetV3:
    """Create AuraNetV3 with optional config override."""
    kwargs = {}
    if config:
        m = config.get("model", {})
        if "encoder" in m and "channels" in m["encoder"]:
            kwargs["encoder_channels"] = tuple(m["encoder"]["channels"])
        if "decoder" in m and "channels" in m["decoder"]:
            kwargs["decoder_channels"] = tuple(m["decoder"]["channels"])
        bn = m.get("bottleneck", {})
        if "hidden_size" in bn:
            kwargs["gru_hidden"] = bn["hidden_size"]
        if "num_layers" in bn:
            kwargs["gru_layers"] = bn["num_layers"]
        s = config.get("stft", {})
        if "n_fft" in s:
            kwargs["n_fft"] = s["n_fft"]
        if "hop_length" in s:
            kwargs["hop_length"] = s["hop_length"]

    model = AuraNetV3(**kwargs)
    n = model.count_parameters()
    mb = model.model_size_mb()
    print(f"AuraNet V3 Optimized: {n:,} params ({mb:.2f} MB)")
    return model


# ── ONNX export ──────────────────────────────────────────────────────────────

def export_onnx(model: AuraNetV3, path: str = "auranet_v3.onnx",
                opset: int = 17) -> None:
    """Export model to ONNX for deployment."""
    model.eval()
    # Use STFT-domain forward for ONNX (avoids torch.stft op issues)
    # We export the core: stft_in → stft_out
    dummy_stft = torch.randn(1, 2, 50, model.n_fft // 2 + 1)

    class _STFTCore(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.encoder = parent.encoder
            self.bottleneck = parent.bottleneck
            self.decoder = parent.decoder
            self.n_fft = parent.n_fft

        def forward(self, noisy_stft):
            encoded, skips = self.encoder(noisy_stft)
            bn_out, gru_feat, _ = self.bottleneck(encoded, None)
            raw_mask = self.decoder(bn_out, skips)
            if raw_mask.shape[2:] != noisy_stft.shape[2:]:
                raw_mask = F.interpolate(raw_mask, size=noisy_stft.shape[2:],
                                         mode='bilinear', align_corners=False)
            mask = torch.tanh(raw_mask)
            mr, mi = mask[:, 0:1], mask[:, 1:2]
            xr, xi = noisy_stft[:, 0:1], noisy_stft[:, 1:2]
            return torch.cat([mr * xr - mi * xi, mr * xi + mi * xr], dim=1)

    core = _STFTCore(model)
    core.eval()

    torch.onnx.export(
        core, dummy_stft, path,
        opset_version=opset,
        input_names=["noisy_stft"],
        output_names=["enhanced_stft"],
        dynamic_axes={
            "noisy_stft": {0: "batch", 2: "time"},
            "enhanced_stft": {0: "batch", 2: "time"},
        },
        dynamo=False,
    )
    print(f"ONNX exported → {path}")


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = create_model()

    # Waveform input test
    x = torch.randn(1, 1, 16000)  # 1 second @ 16 kHz
    enhanced, wdrc_p, h = model(x)
    print(f"Input:    {x.shape}")
    print(f"Output:   {enhanced.shape}")
    print(f"Hidden:   {h.shape}")
    print(f"WDRC keys: {list(wdrc_p.keys())}")

    # Size check
    mb = model.model_size_mb()
    print(f"\nSize: {mb:.2f} MB  {'✅ <3 MB' if mb < 3 else '❌ >3 MB'}")
    print(f"Params: {model.count_parameters():,}")
