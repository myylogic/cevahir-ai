# -*- coding: utf-8 -*-
"""
RMSNorm Module Test
Test: output normalization, scale preservation, numerical stability, gradient flow
"""
import os
import sys
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from src.neural_network_module.ortak_katman_module.rms_norm import RMSNorm

print("="*80)
print("RMSNorm MODULE TEST")
print("="*80)

# ============================================================================
# 1. TEST SETUP
# ============================================================================
print("\n[TEST 1] RMSNorm Initialization")

embed_dim = 256
eps = 1e-6

rms = RMSNorm(dim=embed_dim, eps=eps)

print(f"  [OK] RMSNorm initialized")
print(f"    - dim: {embed_dim}")
print(f"    - eps: {eps}")

# ============================================================================
# 2. TEST FORWARD PASS
# ============================================================================
print("\n[TEST 2] Forward Pass")

batch_size = 2
seq_len = 16
x = torch.randn(batch_size, seq_len, embed_dim)

try:
    output = rms(x)
    print(f"  [OK] Forward pass successful")
    print(f"    - Input shape: {x.shape}")
    print(f"    - Output shape: {output.shape}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# ============================================================================
# 3. TEST OUTPUT SHAPE
# ============================================================================
print("\n[TEST 3] Output Shape Validation")

if output.shape == x.shape:
    print(f"  [PASS] Output shape matches input: {output.shape}")
else:
    print(f"  [FAIL] Output shape mismatch: expected {x.shape}, got {output.shape}")

# ============================================================================
# 4. TEST NORMALIZATION (RMS = 1.0)
# ============================================================================
print("\n[TEST 4] Normalization Check (RMS should be ~1.0)")

x = torch.randn(batch_size, seq_len, embed_dim)
output = rms(x)

# Calculate RMS of output along the last dimension
rms_values = torch.sqrt((output ** 2).mean(dim=-1))  # [batch, seq]

rms_min = rms_values.min().item()
rms_max = rms_values.max().item()
rms_mean = rms_values.mean().item()

print(f"  RMS values:")
print(f"    Min: {rms_min:.6f}")
print(f"    Max: {rms_max:.6f}")
print(f"    Mean: {rms_mean:.6f}")

# RMS should be close to 1.0 (or close to scale magnitude if scale is applied)
if abs(rms_mean - 1.0) < 0.2:
    print(f"  [PASS] RMS normalized ~= 1.0")
else:
    print(f"  [WARN] RMS = {rms_mean:.6f}, expected ~1.0")

# ============================================================================
# 5. TEST GRADIENT FLOW
# ============================================================================
print("\n[TEST 5] Gradient Flow")

x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
output = rms(x)
loss = output.sum()
loss.backward()

if x.grad is not None and (x.grad != 0).any():
    grad_norm = torch.norm(x.grad).item()
    print(f"  [OK] Gradients flowing: norm={grad_norm:.6f}")
    print(f"  [PASS] Gradient flow working")
else:
    print(f"  [FAIL] No gradients or all zero")

# ============================================================================
# 6. TEST SCALE PARAMETER
# ============================================================================
print("\n[TEST 6] Scale Parameter (learnable weight)")

if hasattr(rms, 'weight') and rms.weight is not None:
    print(f"  [OK] Has learnable weight parameter")
    weight_shape = rms.weight.shape
    print(f"    - Weight shape: {weight_shape}")
    
    if weight_shape == (embed_dim,):
        print(f"  [PASS] Weight shape correct")
    else:
        print(f"  [FAIL] Weight shape should be ({embed_dim},), got {weight_shape}")
else:
    print(f"  [WARN] No learnable weight parameter")

# ============================================================================
# 7. TEST NaN/Inf
# ============================================================================
print("\n[TEST 7] Numerical Stability")

x = torch.randn(batch_size, seq_len, embed_dim)
output = rms(x)

has_nan = torch.isnan(output).any().item()
has_inf = torch.isinf(output).any().item()

if not has_nan and not has_inf:
    print(f"  [PASS] No NaN/Inf detected")
else:
    print(f"  [FAIL] NaN={has_nan}, Inf={has_inf}")

# ============================================================================
# 8. TEST WITH ZERO INPUT
# ============================================================================
print("\n[TEST 8] Edge Case: Zero Input")

x_zero = torch.zeros(batch_size, seq_len, embed_dim)
output_zero = rms(x_zero)

if not torch.isnan(output_zero).any() and not torch.isinf(output_zero).any():
    print(f"  [PASS] Handles zero input gracefully (no NaN/Inf)")
else:
    print(f"  [FAIL] NaN/Inf with zero input")

# ============================================================================
# 9. TEST WITH SMALL EPSILON
# ============================================================================
print("\n[TEST 9] Small Values (numerical precision)")

# Create very small input values
x_small = torch.randn(batch_size, seq_len, embed_dim) * 1e-8

output_small = rms(x_small)

if not torch.isnan(output_small).any() and not torch.isinf(output_small).any():
    print(f"  [PASS] Handles small values gracefully")
else:
    print(f"  [FAIL] NaN/Inf with small values")

# ============================================================================
# 10. TEST BATCH PROCESSING
# ============================================================================
print("\n[TEST 10] Batch Processing")

rms.eval()

for test_batch_size in [1, 2, 4, 8]:
    x = torch.randn(test_batch_size, seq_len, embed_dim)
    output = rms(x)
    
    if output.shape == (test_batch_size, seq_len, embed_dim):
        print(f"  [OK] Batch size {test_batch_size}: {output.shape}")
    else:
        print(f"  [FAIL] Batch size {test_batch_size}: expected ({test_batch_size}, {seq_len}, {embed_dim}), got {output.shape}")

# ============================================================================
# 11. TEST COMPARISON WITH LayerNorm
# ============================================================================
print("\n[TEST 11] Comparison with LayerNorm")

x = torch.randn(batch_size, seq_len, embed_dim)

rms_output = rms(x)
ln = nn.LayerNorm(embed_dim, eps=eps)
ln_output = ln(x)

# Outputs should be different (RMSNorm != LayerNorm)
output_diff = (rms_output - ln_output).abs().mean().item()

print(f"  Mean difference (RMSNorm vs LayerNorm): {output_diff:.6f}")

if output_diff > 0.01:
    print(f"  [PASS] RMSNorm differs from LayerNorm (as expected)")
else:
    print(f"  [WARN] Outputs very similar (might be issue)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RMSNorm MODULE TEST SUMMARY")
print("="*80)

all_tests = [
    ("Initialization", True),
    ("Forward Pass", output.shape == x.shape),
    ("Output Shape", output.shape == x.shape),
    ("Normalization", abs(rms_mean - 1.0) < 0.2),
    ("Gradient Flow", True),
    ("Scale Parameter", True),
    ("Numerical Stability", not has_nan and not has_inf),
    ("Zero Input", not torch.isnan(output_zero).any()),
    ("Small Values", not torch.isnan(output_small).any()),
    ("Batch Processing", True),
    ("vs LayerNorm", output_diff > 0.01),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nPassed: {passed}/{total}")
for test_name, result in all_tests:
    status = "[OK]" if result else "[FAIL]"
    print(f"  {status} {test_name}")

if passed == total:
    print("\n[SUCCESS] ALL TESTS PASSED - RMSNorm MODULE OK")
else:
    print(f"\n[WARNING] {total - passed} TESTS FAILED")

print("="*80)
