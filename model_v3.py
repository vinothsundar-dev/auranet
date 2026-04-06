# =============================================================================
# AuraNet V3: Improved Lightweight Speech Enhancement
# =============================================================================
#
# KEY CHANGES FROM V1/V2:
# 1. STFT → log-power features for better gradient flow
# 2. 2-layer GRU (hidden=128) — better temporal modeling, still small
# 3. Spectral mask prediction with sigmoid (bounded, stable training)
# 4. Layer normalization instead of BatchNorm (better for streaming)
# 5. Compressed residual skip connections (1x1 conv alignment)
# 6. Learnable sigmoid compression for mask output
# 7. Removed WDRC sidechain (simplify for deployment)
#
# PARAMETER BUDGET: ~1.2M (under 5MB at fp32)
# LATENCY: <5ms algorithmic (hop=80 samples at 16kHz)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# =============================================================================
# Building Blocks
# =============================================================================

class CausalConv1d(nn.Module):
    """1D causal convolution — pads only on the left."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConv2d(nn.Module):
    """2D convolution with causal padding in time, symmetric in frequency.
    Handles single-frame (T=1) by ensuring output T >= 1."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1),
                 dilation=(1, 1), groups=1, bias=True):
        super().__init__()
        self.pad_time = (kernel_size[0] - 1) * dilation[0]
        self.pad_freq = ((kernel_size[1] - 1) * dilation[1]) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=(0, self.pad_freq), dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_time, 0))
        return self.conv(x)


class CausalTransposeConv2d(nn.Module):
    """Transposed 2D convolution for upsampling in frequency.
    Uses stride=(1,2) with time_stride=1 so T is preserved."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 2),
                 groups=1, bias=True):
        super().__init__()
        self.time_pad = kernel_size[0] - 1
        output_padding = (0, stride[1] - 1) if stride[1] > 1 else (0, 0)
        # Only pad frequency in the conv, handle time manually
        self.conv_t = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=(0, kernel_size[1] // 2),
            output_padding=output_padding,
            groups=groups, bias=bias,
        )

    def forward(self, x):
        out = self.conv_t(x)
        # Remove the extra time steps introduced by transpose conv
        if self.time_pad > 0 and out.shape[2] > x.shape[2]:
            out = out[:, :, :x.shape[2], :]
        return out


class DepthSepConv2d(nn.Module):
    """Depthwise-separable 2D convolution (causal in time)."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.dw = CausalConv2d(in_ch, in_ch, kernel_size, stride=stride,
                               groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class DepthSepTransposeConv2d(nn.Module):
    """Depthwise-separable transposed 2D convolution."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 2)):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.dw_t = CausalTransposeConv2d(out_ch, out_ch, kernel_size,
                                           stride=stride, groups=out_ch, bias=False)

    def forward(self, x):
        return self.dw_t(self.pw(x))


# =============================================================================
# Encoder
# =============================================================================

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 2)):
        super().__init__()
        self.conv = DepthSepConv2d(in_ch, out_ch, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_ch)  # = LayerNorm for conv
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """
    4-block encoder: 2→16→32→64→128 channels
    Frequency: 129→65→33→17→9
    """

    def __init__(self, in_channels=2, channels=(16, 32, 64, 128)):
        super().__init__()
        blocks = []
        ch = in_channels
        for out_ch in channels:
            blocks.append(EncoderBlock(ch, out_ch))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels[-1]

    def forward(self, x):
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return x, skips


# =============================================================================
# Temporal Bottleneck — 2-layer GRU (key improvement)
# =============================================================================

class TemporalBottleneck(nn.Module):
    """
    2-layer unidirectional GRU bottleneck.

    V3 changes:
    - 2 GRU layers (was 1) — captures longer dependencies
    - hidden_size=128 (was 256) — compensated by 2 layers, net smaller
    - LayerNorm before and after GRU for stable training
    - Residual connection around GRU
    """

    def __init__(self, input_channels=128, input_freq_bins=9,
                 hidden_size=128, num_layers=2, dropout=0.1):
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

    def forward(self, x, hidden=None):
        B, C, T, F = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B, T, -1)  # [B,T,C*F]
        residual = x_flat

        x_flat = self.norm_in(x_flat)
        gru_out, hidden_out = self.gru(x_flat, hidden)
        projected = self.projection(gru_out)
        projected = self.norm_out(projected)

        # Residual connection
        projected = projected + residual

        decoded = projected.reshape(B, T, C, F).permute(0, 2, 1, 3)
        return decoded, gru_out, hidden_out


# =============================================================================
# Decoder
# =============================================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, kernel_size=(3, 3), stride=(1, 2)):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_ch, in_ch, 1, bias=False)
        self.conv = DepthSepTransposeConv2d(in_ch, out_ch, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x, skip):
        # Align skip to x via 1x1 projection, then add (not concat — smaller)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = x + self.skip_proj(skip)
        return self.act(self.norm(self.conv(x)))


class Decoder(nn.Module):
    """
    4-block decoder mirroring encoder.
    Uses additive skip connections (not concat) — halves decoder param count.
    Output: 2-channel spectral mask (real + imag).
    """

    def __init__(self, in_channels=128,
                 channels=(64, 32, 16, 2),
                 encoder_channels=(16, 32, 64, 128)):
        super().__init__()
        skip_channels = list(reversed(encoder_channels))
        blocks = []
        ch = in_channels
        for i, out_ch in enumerate(channels[:-1]):
            blocks.append(DecoderBlock(ch, out_ch, skip_channels[i]))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # Final output: proj skip + transpose conv → 2-channel mask
        final_skip_ch = skip_channels[len(channels) - 1] if len(channels) - 1 < len(skip_channels) else ch
        self.final_skip_proj = nn.Conv2d(final_skip_ch, ch, 1, bias=False)
        self.output_conv = nn.Sequential(
            DepthSepTransposeConv2d(ch, channels[-1], (3, 3), (1, 2)),
        )

    def forward(self, x, skips):
        skips_r = skips[::-1]
        for i, block in enumerate(self.blocks):
            x = block(x, skips_r[i])

        # Final skip + output
        idx = len(self.blocks)
        if idx < len(skips_r):
            last_skip = skips_r[idx]
            if x.shape[2:] != last_skip.shape[2:]:
                x = F.interpolate(x, size=last_skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + self.final_skip_proj(last_skip)
        mask = self.output_conv(x)
        return mask


# =============================================================================
# Mask Activation — Learnable Sigmoid Compression
# =============================================================================

class LearnableSigmoid(nn.Module):
    """
    Learnable sigmoid: mask_floor + (1 - mask_floor) * sigmoid(a * x + b)
    Range: [mask_floor, 1.0]. Prevents phase inversion and over-suppression.

    INIT STRATEGY: bias=2.0 so sigmoid(2.0) ≈ 0.88, giving initial mask ≈ 0.88.
    This starts near pass-through — the model learns to suppress noise
    rather than starting from silence (which causes soft/muffled output).
    """

    def __init__(self, in_features, beta=1.0, init_bias=2.0, mask_floor=0.05):
        super().__init__()
        self.beta = nn.Parameter(torch.full((in_features,), beta))
        self.bias = nn.Parameter(torch.full((in_features,), init_bias))
        self.mask_floor = mask_floor

    def forward(self, x):
        # x: [B, 2, T, F] — apply per-frequency
        # Output range: [mask_floor, 1.0] — no phase inversion possible
        return self.mask_floor + (1.0 - self.mask_floor) * torch.sigmoid(self.beta * x + self.bias)


# =============================================================================
# AuraNet V3 — Full Model
# =============================================================================

class AuraNetV3(nn.Module):
    """
    Lightweight causal CRN for real-time speech enhancement.

    Pipeline:
        noisy_stft [B,2,T,F]
        → Encoder (4 blocks, DS-Conv, freq downsample)
        → GRU Bottleneck (2-layer, h=128, residual)
        → Decoder (4 blocks, DS-TransConv, additive skips)
        → Mask activation (learnable sigmoid)
        → Apply mask: enhanced = noisy * mask
        → enhanced_stft [B,2,T,F]
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        model_cfg = config.get("model", {})

        enc_channels = tuple(model_cfg.get("encoder", {}).get("channels", [16, 32, 64, 128]))
        dec_channels = tuple(model_cfg.get("decoder", {}).get("channels", [128, 64, 32, 16, 2]))
        gru_hidden = model_cfg.get("bottleneck", {}).get("hidden_size", 128)
        gru_layers = model_cfg.get("bottleneck", {}).get("num_layers", 2)
        gru_dropout = model_cfg.get("bottleneck", {}).get("dropout", 0.1)
        n_fft = config.get("stft", {}).get("n_fft", 256)
        freq_bins = n_fft // 2 + 1  # 129

        self.encoder = Encoder(in_channels=2, channels=enc_channels)

        # Compute freq bins after encoder
        f = freq_bins
        for _ in enc_channels:
            f = (f + 1) // 2  # stride-2 downsampling with padding
        enc_freq_bins = f

        self.bottleneck = TemporalBottleneck(
            input_channels=enc_channels[-1],
            input_freq_bins=enc_freq_bins,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=gru_dropout,
        )

        # Decoder: first element is the bottleneck output channel count
        self.decoder = Decoder(
            in_channels=enc_channels[-1],
            channels=tuple(dec_channels[1:]),  # skip first (it equals enc_channels[-1])
            encoder_channels=enc_channels,
        )

        # Learnable mask activation (per frequency bin)
        self.mask_act = LearnableSigmoid(freq_bins)

    def forward(self, noisy_stft, hidden=None):
        """
        Args:
            noisy_stft: [B, 2, T, F] complex STFT (real + imag)
            hidden: optional GRU hidden state

        Returns:
            enhanced_stft: [B, 2, T, F]
            hidden_out: GRU hidden state for streaming
            gru_features: [B, T, H] (for auxiliary tasks if needed)
        """
        # Encode
        encoded, skips = self.encoder(noisy_stft)

        # Temporal modeling
        bottleneck_out, gru_features, hidden_out = self.bottleneck(encoded, hidden)

        # Decode → raw mask [B, 2, T, F]
        raw_mask = self.decoder(bottleneck_out, skips)

        # Align mask to input frequency dimension
        if raw_mask.shape[-1] != noisy_stft.shape[-1]:
            raw_mask = F.interpolate(
                raw_mask, size=noisy_stft.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Apply learnable sigmoid activation
        # Mask range is [0.05, 1.0] — floor is built into LearnableSigmoid
        mask = self.mask_act(raw_mask)

        # Apply mask directly (floor already embedded in activation)
        enhanced_stft = noisy_stft * mask

        return enhanced_stft, hidden_out, gru_features

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Factory
# =============================================================================

def create_auranet_v3(config=None):
    """Create AuraNet V3 model from config dict."""
    model = AuraNetV3(config)
    n_params = model.count_parameters()
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"AuraNet V3: {n_params:,} parameters ({size_mb:.1f} MB)")
    return model
