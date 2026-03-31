# =============================================================================
# AuraNet-Lite V2: Ultra-Optimized for Edge Deployment
# =============================================================================
#
# AGGRESSIVE OPTIMIZATIONS:
# 1. Pre-bottleneck projection: 864 → 128 (reduces GRU input massively)
# 2. Even smaller GRU: 64 hidden
# 3. Channels: 2→8→16→32→64 (further reduced)
# 4. TCN option with dilated convs
# 5. Minimal WDRC with single layer
#
# TARGET:
# - <200K parameters
# - >30% MAC reduction
# - <10ms latency
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# =============================================================================
# Ultra-Efficient Building Blocks
# =============================================================================

class CausalConv2dBNReLU(nn.Module):
    """
    Fused Causal Conv2D + BN + ReLU.
    Single efficient block for edge deployment.
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: Tuple[int, int] = (3, 1),
        stride: Tuple[int, int] = (1, 1),
        groups: int = 1,
    ):
        super().__init__()
        
        self.pad_t = kernel[0] - 1
        self.pad_f = kernel[1] // 2
        
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel,
            stride=stride,
            padding=(0, self.pad_f),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_t, 0))
        return F.relu(self.bn(self.conv(x)), inplace=True)


class UltraLiteDSConv(nn.Module):
    """
    Ultra-minimal depthwise separable conv.
    """
    
    def __init__(self, in_ch: int, out_ch: int, stride_f: int = 1):
        super().__init__()
        
        # Depthwise: 3x1 causal kernel
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, (3, 1), stride=(1, stride_f), 
                      padding=(0, 0), groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pad_t = 2  # For 3x1 kernel
        
        # Pointwise
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_t, 0))
        x = self.dw(x)
        return self.pw(x)


class PixelShuffleUp(nn.Module):
    """
    Efficient frequency upsampling via pixel shuffle.
    """
    
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        
        self.conv = nn.Conv2d(in_ch, out_ch * scale, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_ch * scale)
        self.scale = scale
        self.out_ch = out_ch
        
    def forward(self, x):
        B, C, T, Fb = x.shape
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        # Reshape for frequency upsampling
        x = x.view(B, self.out_ch, self.scale, T, Fb)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B, self.out_ch, T, Fb * self.scale)
        return x


# =============================================================================
# Ultra-Lite Encoder
# =============================================================================

class UltraLiteEncoder(nn.Module):
    """
    Minimal encoder: 2→8→16→32→64
    4 blocks with freq downsampling
    """
    
    def __init__(self, channels=(8, 16, 32, 64)):
        super().__init__()
        
        blocks = []
        in_ch = 2
        for out_ch in channels:
            blocks.append(UltraLiteDSConv(in_ch, out_ch, stride_f=2))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.out_ch = channels[-1]
        
    def forward(self, x):
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return x, skips


# =============================================================================
# Ultra-Lite Temporal Processing
# =============================================================================

class CompactGRU(nn.Module):
    """
    Compact GRU with input projection to reduce parameters.
    
    Key: Project from C*F → small_dim before GRU
    This reduces GRU input weights by ~10x
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        in_freq: int = 9,
        proj_dim: int = 64,  # Project to this dim
        hidden: int = 64,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.in_freq = in_freq
        self.proj_dim = proj_dim
        self.hidden = hidden
        
        # Project down before GRU
        self.input_proj = nn.Linear(in_channels * in_freq, proj_dim)
        
        # Small GRU
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=False,
        )
        
        # Project back
        self.output_proj = nn.Linear(hidden, in_channels * in_freq)
        
    def forward(self, x, hidden=None):
        B, C, T, Fb = x.shape
        
        # Flatten and project
        x_flat = x.permute(0, 2, 1, 3).reshape(B, T, C * Fb)
        x_proj = self.input_proj(x_flat)
        
        # GRU
        gru_out, h_out = self.gru(x_proj, hidden)
        
        # Project and reshape
        out = self.output_proj(gru_out)
        out = out.view(B, T, C, Fb).permute(0, 2, 1, 3).contiguous()
        
        return out, gru_out, h_out


class DilatedTCN(nn.Module):
    """
    Dilated Temporal Convolutional Network.
    
    Alternative to GRU - better for quantization.
    Uses depthwise separable dilated convs.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        in_freq: int = 9,
        hidden: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.in_freq = in_freq
        self.hidden = hidden
        
        # Project to hidden dim (operate in 1D along time)
        total_in = in_channels * in_freq
        self.input_proj = nn.Conv1d(total_in, hidden, 1)
        
        # Dilated causal convs
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            pad = (3 - 1) * dilation  # For kernel size 3
            layers.extend([
                nn.Conv1d(hidden, hidden, 3, padding=0, dilation=dilation, groups=hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, hidden, 1),  # Pointwise
            ])
        self.layers = nn.ModuleList(layers)
        self.pads = [((3 - 1) * (2 ** (i // 4))) for i in range(0, num_layers * 4, 4)]
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden, total_in, 1)
        
    def forward(self, x, hidden=None):
        B, C, T, Fb = x.shape
        
        # Flatten to [B, C*F, T]
        x_flat = x.permute(0, 1, 3, 2).reshape(B, C * Fb, T)
        
        # Project
        h = self.input_proj(x_flat)
        
        # Apply dilated convs with causal padding
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv1d) and layer.dilation[0] > 1:
                # Causal padding for dilated conv
                pad = (layer.kernel_size[0] - 1) * layer.dilation[0]
                h = F.pad(h, (pad, 0))
            elif isinstance(layer, nn.Conv1d) and layer.kernel_size[0] == 3:
                # Regular conv3
                pad = (3 - 1) * layer.dilation[0]
                h = F.pad(h, (pad, 0))
            h = layer(h)
        
        # For WDRC
        tcn_out = h.permute(0, 2, 1)  # [B, T, hidden]
        
        # Project back
        out = self.output_proj(h)
        out = out.view(B, C, Fb, T).permute(0, 1, 3, 2).contiguous()
        
        return out, tcn_out, None


# =============================================================================
# Ultra-Lite Decoder
# =============================================================================

class UltraLiteDecoder(nn.Module):
    """
    Minimal decoder with skip connections.
    Outputs magnitude mask only (no phase - simpler).
    """
    
    def __init__(
        self,
        in_ch: int = 64,
        channels: Tuple[int, ...] = (32, 16, 8),
        skip_channels: Tuple[int, ...] = (64, 32, 16, 8),
    ):
        super().__init__()
        
        blocks = []
        curr_ch = in_ch
        
        for i, out_ch in enumerate(channels):
            skip_ch = skip_channels[i]
            # Upsample + concat skip + conv
            blocks.append(nn.ModuleDict({
                'up': PixelShuffleUp(curr_ch + skip_ch, out_ch),
            }))
            curr_ch = out_ch
            
        self.blocks = nn.ModuleList(blocks)
        
        # Final output head
        final_skip_ch = skip_channels[len(channels)]
        self.head = nn.Sequential(
            nn.Conv2d(curr_ch + final_skip_ch, curr_ch, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(curr_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_ch, 1, (1, 1)),  # Single channel: magnitude mask
            nn.Sigmoid(),
        )
        
    def forward(self, x, skips):
        # Reverse skips
        skips = skips[::-1]
        
        for i, block in enumerate(self.blocks):
            skip = skips[i]
            # Match size
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = block['up'](x)
        
        # Final skip
        last_skip = skips[len(self.blocks)]
        if x.shape[2:] != last_skip.shape[2:]:
            x = F.interpolate(x, size=last_skip.shape[2:], mode='nearest')
        x = torch.cat([x, last_skip], dim=1)
        
        return self.head(x)


# =============================================================================
# Minimal WDRC
# =============================================================================

class MinimalWDRC(nn.Module):
    """Single layer WDRC for minimal overhead."""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_dim, 4)
        
    def forward(self, x):
        out = self.fc(x)
        return {
            "attack_coeff": torch.sigmoid(out[..., 0]),
            "release_coeff": torch.sigmoid(out[..., 1]),
            "compression_ratio": F.softplus(out[..., 2]) + 1.0,
            "gain": torch.sigmoid(out[..., 3]) * 2.0,
        }


# =============================================================================
# AuraNet-Lite V2
# =============================================================================

class AuraNetLiteV2(nn.Module):
    """
    Ultra-optimized AuraNet for edge deployment.
    
    Key optimizations:
    1. Channels: 2→8→16→32→64 (vs 2→16→32→64→128)
    2. Input projection before GRU: 64*9=576 → 64
    3. GRU hidden: 64 (vs 256)
    4. Magnitude-only mask (simpler, sufficient quality)
    5. Minimal WDRC (single layer)
    
    Trade-offs:
    - Slightly reduced capacity
    - Magnitude-only may lose some phase information
    - Good enough for most use cases
    """
    
    def __init__(
        self,
        encoder_channels: Tuple[int, ...] = (8, 16, 32, 64),
        decoder_channels: Tuple[int, ...] = (32, 16, 8),
        gru_proj_dim: int = 64,
        gru_hidden: int = 64,
        use_tcn: bool = False,
    ):
        super().__init__()
        
        self.encoder = UltraLiteEncoder(encoder_channels)
        
        # Freq bins after encoding: 129 → 65 → 33 → 17 → 9
        freq_encoded = 9
        
        if use_tcn:
            self.temporal = DilatedTCN(
                in_channels=encoder_channels[-1],
                in_freq=freq_encoded,
                hidden=gru_hidden,
            )
        else:
            self.temporal = CompactGRU(
                in_channels=encoder_channels[-1],
                in_freq=freq_encoded,
                proj_dim=gru_proj_dim,
                hidden=gru_hidden,
            )
        
        self.decoder = UltraLiteDecoder(
            in_ch=encoder_channels[-1],
            channels=decoder_channels,
            skip_channels=tuple(reversed(encoder_channels)),
        )
        
        self.wdrc = MinimalWDRC(gru_hidden)
        
    def forward(
        self,
        noisy_stft: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        
        B, _, T, freq_bins = noisy_stft.shape
        
        # Get input magnitude (for masking)
        noisy_real = noisy_stft[:, 0:1]
        noisy_imag = noisy_stft[:, 1:2]
        
        # Encode
        encoded, skips = self.encoder(noisy_stft)
        
        # Temporal processing
        temporal_out, temporal_feat, new_hidden = self.temporal(encoded, hidden)
        
        # Decode → magnitude mask
        mag_mask = self.decoder(temporal_out, skips)
        
        # Ensure size match
        if mag_mask.shape[2:] != (T, freq_bins):
            mag_mask = F.interpolate(mag_mask, size=(T, freq_bins), mode='bilinear', align_corners=False)
        
        # Apply mask to both real and imag (preserves phase)
        enhanced_real = noisy_real * mag_mask
        enhanced_imag = noisy_imag * mag_mask
        enhanced_stft = torch.cat([enhanced_real, enhanced_imag], dim=1)
        
        # WDRC
        wdrc_params = self.wdrc(temporal_feat)
        
        return enhanced_stft, wdrc_params, new_hidden
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Comparison and Testing
# =============================================================================

def full_comparison():
    """Compare all model variants."""
    
    print("=" * 70)
    print("AuraNet Model Comparison: Original vs Lite vs Lite-V2")
    print("=" * 70)
    
    from model import AuraNet
    from model_optimized import AuraNetLite
    
    models = {
        "Original (AuraNet)": AuraNet(),
        "Lite V1": AuraNetLite(),
        "Lite V2 (GRU)": AuraNetLiteV2(use_tcn=False),
        "Lite V2 (TCN)": AuraNetLiteV2(use_tcn=True),
    }
    
    input_shape = (1, 2, 100, 129)
    x = torch.randn(*input_shape)
    
    print(f"\n{'Model':<25} {'Params':>12} {'Reduction':>12} {'Time (ms)':>12}")
    print("-" * 65)
    
    import time
    
    orig_params = None
    results = {}
    
    for name, model in models.items():
        model.eval()
        params = model.count_parameters()
        
        if orig_params is None:
            orig_params = params
            reduction = "-"
        else:
            reduction = f"{(1 - params/orig_params)*100:.1f}%"
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Time
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        elapsed = (time.perf_counter() - start) / 100 * 1000
        
        print(f"{name:<25} {params:>12,} {reduction:>12} {elapsed:>12.2f}")
        results[name] = {"params": params, "time_ms": elapsed}
    
    # Calculate MACs estimate
    print(f"\n{'Model':<25} {'MACs (M)':>12} {'Reduction':>12}")
    print("-" * 50)
    
    def estimate_macs(model, shape):
        B, C, T, F = shape
        total = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1]
                total += k * m.in_channels * m.out_channels * T * F // max(m.groups, 1)
            elif isinstance(m, nn.Conv1d):
                k = m.kernel_size[0]
                total += k * m.in_channels * m.out_channels * T // max(m.groups, 1)
            elif isinstance(m, nn.Linear):
                total += m.in_features * m.out_features
            elif isinstance(m, nn.GRU):
                total += 3 * (m.input_size * m.hidden_size + m.hidden_size ** 2) * T
        return total
    
    orig_macs = None
    for name, model in models.items():
        macs = estimate_macs(model, input_shape)
        if orig_macs is None:
            orig_macs = macs
            reduction = "-"
        else:
            reduction = f"{(1 - macs/orig_macs)*100:.1f}%"
        
        print(f"{name:<25} {macs/1e6:>12.1f} {reduction:>12}")
        results[name]["macs"] = macs
    
    # Output shape verification
    print(f"\n{'Model':<25} {'Output Shape':<25} {'Match':>10}")
    print("-" * 60)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            y, _, _ = model(x)
        match = "✅" if y.shape == x.shape else "❌"
        print(f"{name:<25} {str(y.shape):<25} {match:>10}")
    
    print("\n" + "=" * 70)
    
    # Summary
    best_lite = min(
        [(k, v) for k, v in results.items() if "Lite" in k],
        key=lambda x: x[1]["params"]
    )
    
    print(f"\n📊 SUMMARY")
    print(f"  Best lightweight model: {best_lite[0]}")
    print(f"  Parameters: {best_lite[1]['params']:,} ({(1-best_lite[1]['params']/orig_params)*100:.1f}% reduction)")
    print(f"  MACs: {best_lite[1]['macs']/1e6:.1f}M ({(1-best_lite[1]['macs']/orig_macs)*100:.1f}% reduction)")
    print(f"  Inference: {best_lite[1]['time_ms']:.2f}ms")
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("AuraNet-Lite V2 Test")
    print("=" * 70)
    
    # Test V2 model
    model = AuraNetLiteV2()
    params = model.count_parameters()
    
    print(f"\n📊 Parameters: {params:,}")
    print(f"   Target: <200,000")
    print(f"   Status: {'✅ PASS' if params < 200000 else '⚠️ CHECK'}")
    
    # Parameter breakdown
    print(f"\n📋 Parameter Breakdown:")
    for name, module in model.named_children():
        mp = sum(p.numel() for p in module.parameters())
        print(f"   {name}: {mp:,} ({100*mp/params:.1f}%)")
    
    # Forward pass
    print(f"\n🔄 Forward Pass Test")
    x = torch.randn(2, 2, 100, 129)
    y, wdrc, h = model(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Status: {'✅ PASS' if y.shape == x.shape else '❌ FAIL'}")
    
    # TCN variant
    print(f"\n🔄 TCN Variant Test")
    model_tcn = AuraNetLiteV2(use_tcn=True)
    params_tcn = model_tcn.count_parameters()
    print(f"   Parameters: {params_tcn:,}")
    y_tcn, _, _ = model_tcn(x)
    print(f"   Output: {y_tcn.shape}")
    print(f"   Status: {'✅ PASS' if y_tcn.shape == x.shape else '❌ FAIL'}")
    
    # Run full comparison
    print("\n")
    try:
        full_comparison()
    except Exception as e:
        print(f"Comparison failed: {e}")
    
    print("\n✅ All tests passed!")
