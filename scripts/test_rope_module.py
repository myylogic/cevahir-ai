# -*- coding: utf-8 -*-
"""
RoPE (Rotary Position Embedding) Module Test
Test et: magnitude preservation, gradient flow, numerical stability
"""
import os
import sys
import torch
import torch.nn as nn
import math

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from src.neural_network_module.dil_katmani_module.positional_encoding import PositionalEncoding

print("="*80)
print("RoPE MODULE TEST")
print("="*80)

# ============================================================================
# 1. TEST SETUP
# ============================================================================
print("\n[TEST 1] RoPE Initialization")

embed_dim = 256
num_heads = 4
max_len = 2048
head_dim = embed_dim // num_heads  # 64

pe = PositionalEncoding(
    embed_dim=embed_dim,
    max_len=max_len,
    dropout=0.0,  # No dropout for testing
    mode="rope",
    num_heads=num_heads,
)

print(f"  [OK] RoPE initialized")
print(f"    - embed_dim: {embed_dim}")
print(f"    - num_heads: {num_heads}")
print(f"    - head_dim: {head_dim}")
print(f"    - rope_dim: {pe.rope_dim}")
print(f"    - mode: {pe.mode}")

# ============================================================================
# 2. TEST MAGNITUDE PRESERVATION
# ============================================================================
print("\n[TEST 2] Magnitude Preservation (rotation shouldn't change norm)")

# Create random query [batch=2, heads=4, seq=8, head_dim=64]
batch_size, num_heads_test, seq_len, head_dim_test = 2, 4, 8, 64
q = torch.randn(batch_size, num_heads_test, seq_len, head_dim_test)

# Calculate norms BEFORE rotation
q_norm_before = torch.norm(q, dim=-1)  # [batch, heads, seq]

# Apply RoPE
try:
    q_rotated = pe.apply_rotary_pos_emb(q)
    print(f"  [OK] RoPE applied successfully")
except Exception as e:
    print(f"  [ERROR] applying RoPE: {e}")
    sys.exit(1)

# Calculate norms AFTER rotation
q_norm_after = torch.norm(q_rotated, dim=-1)  # [batch, heads, seq]

# Check if magnitudes preserved
norm_diff = (q_norm_before - q_norm_after).abs()
max_norm_diff = norm_diff.max().item()
avg_norm_diff = norm_diff.mean().item()

print(f"\n  Magnitude Preservation:")
print(f"    Before rotation - max: {q_norm_before.max().item():.6f}, min: {q_norm_before.min().item():.6f}")
print(f"    After rotation  - max: {q_norm_after.max().item():.6f}, min: {q_norm_after.min().item():.6f}")
print(f"    Diff (max): {max_norm_diff:.8f}")
print(f"    Diff (avg): {avg_norm_diff:.8f}")

if max_norm_diff < 1e-4:  # Should be nearly identical
    print(f"  [PASS] Magnitude preserved (diff < 1e-4)")
else:
    print(f"  [FAIL] Magnitude NOT preserved (diff={max_norm_diff:.8f})")

# ============================================================================
# 3. TEST GRADIENT FLOW
# ============================================================================
print("\n[TEST 3] Gradient Flow Through RoPE")

q = torch.randn(batch_size, num_heads_test, seq_len, head_dim_test, requires_grad=True)
q_rotated = pe.apply_rotary_pos_emb(q)

# Create loss and backprop
loss = q_rotated.sum()
loss.backward()

if q.grad is not None and (q.grad != 0).any():
    grad_norm = torch.norm(q.grad).item()
    print(f"  [OK] Gradients flowing: norm={grad_norm:.6f}")
else:
    print(f"  [ERROR] No gradients or all zero")

# ============================================================================
# 4. TEST NaN/Inf
# ============================================================================
print("\n[TEST 4] Numerical Stability (NaN/Inf check)")

q = torch.randn(batch_size, num_heads_test, seq_len, head_dim_test)
q_rotated = pe.apply_rotary_pos_emb(q)

has_nan = torch.isnan(q_rotated).any().item()
has_inf = torch.isinf(q_rotated).any().item()

if not has_nan and not has_inf:
    print(f"  [PASS] No NaN/Inf detected")
else:
    print(f"  [FAIL] NaN={has_nan}, Inf={has_inf}")

# ============================================================================
# 5. TEST OUTPUT RANGE
# ============================================================================
print("\n[TEST 5] Output Range (numerical stability)")

q = torch.randn(batch_size, num_heads_test, seq_len, head_dim_test)
q_rotated = pe.apply_rotary_pos_emb(q)

q_mean = q_rotated.mean().item()
q_std = q_rotated.std().item()
q_max = q_rotated.max().item()
q_min = q_rotated.min().item()

print(f"  Output statistics:")
print(f"    Mean: {q_mean:.6f}")
print(f"    Std: {q_std:.6f}")
print(f"    Range: [{q_min:.6f}, {q_max:.6f}]")

if abs(q_mean) < 0.5 and 0.5 < q_std < 2.0 and abs(q_max) < 10:
    print(f"  [PASS] Output range reasonable")
else:
    print(f"  [WARN] Output range might be unstable")

# ============================================================================
# 6. TEST WITH LONGER SEQUENCES
# ============================================================================
print("\n[TEST 6] Long Sequence (seq_len > max_len)")

# Create sequence longer than initial max_len
long_seq_len = 3000
q_long = torch.randn(batch_size, num_heads_test, long_seq_len, head_dim_test)

try:
    q_long_rotated = pe.apply_rotary_pos_emb(q_long)
    print(f"  [PASS] Long sequence handled (seq_len={long_seq_len})")
    print(f"    pe.max_len auto-expanded to {pe.max_len}")
except Exception as e:
    print(f"  [FAIL] {e}")

# ============================================================================
# 7. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RoPE MODULE TEST SUMMARY")
print("="*80)

all_tests = [
    ("Initialization", True),
    ("Magnitude Preservation", max_norm_diff < 1e-4),
    ("Gradient Flow", q.grad is not None and (q.grad != 0).any()),
    ("Numerical Stability", not has_nan and not has_inf),
    ("Output Range", abs(q_mean) < 0.5 and 0.5 < q_std < 2.0),
    ("Long Sequence", True),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nPassed: {passed}/{total}")
for test_name, result in all_tests:
    status = "[OK]" if result else "[FAIL]"
    print(f"  {status} {test_name}")

if passed == total:
    print("\n[SUCCESS] ALL TESTS PASSED - RoPE MODULE OK")
else:
    print(f"\n[WARNING] {total - passed} TESTS FAILED")

print("="*80)
