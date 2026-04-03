# ============================================================
# AURANET V2 QUICK DIAGNOSTIC - Run in Colab
# ============================================================
# Copy this cell to your Colab notebook and run it
# It will profile your model and identify issues

import torch
import torch.nn as nn
import time

def run_auranet_diagnostic():
    """
    Quick diagnostic for AuraNet V2.
    Call this after defining your model.
    """
    print("=" * 60)
    print("AURANET V2 QUICK DIAGNOSTIC")
    print("=" * 60)
    
    # Try to import model
    try:
        from auranet_v2_complete import AuraNetV2Complete
        model = AuraNetV2Complete()
    except:
        print("⚠️ Could not import AuraNetV2Complete")
        print("   Make sure auranet_v2_complete.py is in the current directory")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    # === 1. PARAMETER COUNT ===
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    budget = 1_500_000
    
    print(f"\n📊 PARAMETERS")
    print(f"   Total: {total:,}")
    print(f"   Trainable: {trainable:,}")
    print(f"   Budget: {budget:,}")
    print(f"   Status: {'✅ PASS' if total < budget else '❌ FAIL - reduce model size'}")
    
    # === 2. MEMORY ===
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"\n💾 MEMORY")
    print(f"   Model size: {mem_mb:.2f} MB")
    
    # === 3. LATENCY ===
    print(f"\n⏱️ LATENCY")
    x = torch.randn(1, 2, 100, 129, device=device)  # ~0.5s of audio
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    times.sort()
    p50 = times[50]
    p95 = times[95]
    mean_t = sum(times) / len(times)
    target = 10.0
    
    print(f"   Mean: {mean_t:.2f} ms")
    print(f"   P50: {p50:.2f} ms")
    print(f"   P95: {p95:.2f} ms")
    print(f"   Target: <{target} ms")
    
    if p95 < target:
        print(f"   Status: ✅ PASS")
    elif p95 < 15:
        print(f"   Status: ⚠️ CLOSE - consider optimization")
    else:
        print(f"   Status: ❌ FAIL - too slow")
        print(f"   Suggestions:")
        print(f"     - Reduce GRU hidden size (256→128)")
        print(f"     - Use depthwise separable convs")
        print(f"     - Fuse Conv+BN layers")
    
    # === 4. CAUSALITY CHECK ===
    print(f"\n🔒 CAUSALITY CHECK")
    with torch.no_grad():
        # Test: modify future frame, check past frames unchanged
        x1 = torch.randn(1, 2, 20, 129, device=device)
        x2 = x1.clone()
        x2[:, :, -1, :] = torch.randn(1, 2, 129, device=device)  # Modify last frame
        
        y1, _, _ = model(x1)
        y2, _, _ = model(x2)
        
        # Past frames should be identical
        diff = (y1[:, :, :-1, :] - y2[:, :, :-1, :]).abs().max().item()
    
    if diff < 1e-5:
        print(f"   Status: ✅ CAUSAL (diff={diff:.2e})")
    else:
        print(f"   Status: ❌ NOT CAUSAL (diff={diff:.2e})")
        print(f"   Check: bidirectional GRU? symmetric padding?")
    
    # === 5. NaN CHECK ===
    print(f"\n🔍 NUMERICAL STABILITY")
    with torch.no_grad():
        output, _, _ = model(x)
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
    
    if not has_nan and not has_inf:
        print(f"   Status: ✅ STABLE (no NaN/Inf)")
    else:
        print(f"   Status: ❌ UNSTABLE")
        if has_nan:
            print(f"     NaN detected in output")
        if has_inf:
            print(f"     Inf detected in output")
        print(f"   Fix: Add clamping, use safe_log, check loss")
    
    # === 6. MODULE BREAKDOWN ===
    print(f"\n📦 MODULE BREAKDOWN")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        pct = 100 * params / total
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"   {name:25s} {params:>8,} ({pct:5.1f}%) {bar}")
    
    # === 7. GPU MEMORY (if CUDA) ===
    if device.type == 'cuda':
        print(f"\n🎮 GPU MEMORY")
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"   Allocated: {allocated:.1f} MB")
        print(f"   Reserved: {reserved:.1f} MB")
    
    # === SUMMARY ===
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    issues = []
    if total >= budget:
        issues.append("Too many parameters")
    if p95 >= target:
        issues.append("Latency too high")
    if diff >= 1e-5:
        issues.append("Not causal")
    if has_nan or has_inf:
        issues.append("Numerical instability")
    
    if not issues:
        print("✅ All checks passed! Model is production-ready.")
    else:
        print(f"⚠️ Issues found: {', '.join(issues)}")
        print("   See detailed output above for fixes.")
    
    return {
        'params': total,
        'latency_p95_ms': p95,
        'is_causal': diff < 1e-5,
        'is_stable': not has_nan and not has_inf,
        'passed_all': len(issues) == 0,
    }


# Run diagnostic
if __name__ == "__main__":
    results = run_auranet_diagnostic()
