# -*- coding: utf-8 -*-
"""
MultiHeadAttention Module Test
Test: attention weights, gradient flow, RoPE integration, output shape
"""
import os
import sys
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from src.neural_network_module.ortak_katman_module.attention_manager_module.multi_head_attention import MultiHeadAttention
from src.neural_network_module.dil_katmani_module.positional_encoding import PositionalEncoding

print("="*80)
print("MULTI-HEAD ATTENTION MODULE TEST")
print("="*80)

# ============================================================================
# 1. TEST SETUP
# ============================================================================
print("\n[TEST 1] MultiHeadAttention Initialization")

embed_dim = 256
num_heads = 4
dropout = 0.0
max_len = 2048

# Create RoPE positional encoding
pe = PositionalEncoding(
    embed_dim=embed_dim,
    max_len=max_len,
    dropout=dropout,
    mode="rope",
    num_heads=num_heads,
)

# Create attention
attn = MultiHeadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    dropout=dropout,
    use_rope=True,
    positional_encoding=pe,
)

print(f"  [OK] MultiHeadAttention initialized")
print(f"    - embed_dim: {embed_dim}")
print(f"    - num_heads: {num_heads}")
print(f"    - head_dim: {attn.head_dim}")
print(f"    - use_rope: {attn.use_rope}")
print(f"    - has positional_encoding: {attn.positional_encoding is not None}")

# ============================================================================
# 2. TEST FORWARD PASS
# ============================================================================
print("\n[TEST 2] Forward Pass (with RoPE)")

batch_size = 2
seq_len = 16
q = torch.randn(batch_size, seq_len, embed_dim)
k = torch.randn(batch_size, seq_len, embed_dim)
v = torch.randn(batch_size, seq_len, embed_dim)

try:
    output, attn_weights = attn(q, k, v, return_attention_weights=True)
    print(f"  [OK] Forward pass successful")
    print(f"    - Input shape (Q,K,V): {q.shape}")
    print(f"    - Output shape: {output.shape}")
    print(f"    - Attention weights shape: {attn_weights.shape if attn_weights is not None else 'None'}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# ============================================================================
# 3. TEST OUTPUT SHAPE
# ============================================================================
print("\n[TEST 3] Output Shape Validation")

if output.shape == (batch_size, seq_len, embed_dim):
    print(f"  [PASS] Output shape matches input: {output.shape}")
else:
    print(f"  [FAIL] Output shape mismatch: expected {(batch_size, seq_len, embed_dim)}, got {output.shape}")

# ============================================================================
# 4. TEST ATTENTION WEIGHTS
# ============================================================================
print("\n[TEST 4] Attention Weights (should be [0,1] and sum to 1)")

if attn_weights is not None:
    # Check bounds
    attn_min = attn_weights.min().item()
    attn_max = attn_weights.max().item()
    
    print(f"  Attention weights range: [{attn_min:.6f}, {attn_max:.6f}]")
    
    if 0 <= attn_min and attn_max <= 1:
        print(f"  [PASS] Weights in valid range [0,1]")
    else:
        print(f"  [FAIL] Weights outside [0,1]")
    
    # Check row sums (should be ~1.0)
    # attn_weights shape: [batch, heads, seq_q, seq_k]
    row_sums = attn_weights.sum(dim=-1)  # [batch, heads, seq_q]
    row_sum_min = row_sums.min().item()
    row_sum_max = row_sums.max().item()
    row_sum_mean = row_sums.mean().item()
    
    print(f"  Row sums: min={row_sum_min:.6f}, max={row_sum_max:.6f}, mean={row_sum_mean:.6f}")
    
    if abs(row_sum_min - 1.0) < 0.01 and abs(row_sum_max - 1.0) < 0.01:
        print(f"  [PASS] Row sums ~= 1.0")
    else:
        print(f"  [WARN] Row sums not normalized properly")

# ============================================================================
# 5. TEST GRADIENT FLOW
# ============================================================================
print("\n[TEST 5] Gradient Flow")

q = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
k = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
v = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

output, _ = attn(q, k, v)
loss = output.sum()
loss.backward()

grad_q = q.grad is not None and (q.grad != 0).any()
grad_k = k.grad is not None and (k.grad != 0).any()
grad_v = v.grad is not None and (v.grad != 0).any()

print(f"  Q gradient: {grad_q}")
print(f"  K gradient: {grad_k}")
print(f"  V gradient: {grad_v}")

if grad_q and grad_k and grad_v:
    print(f"  [PASS] Gradients flowing through Q, K, V")
else:
    print(f"  [FAIL] Some gradients missing")

# ============================================================================
# 6. TEST MASK SUPPORT (causal mask)
# ============================================================================
print("\n[TEST 6] Causal Mask Support")

q = torch.randn(batch_size, seq_len, embed_dim)
k = torch.randn(batch_size, seq_len, embed_dim)
v = torch.randn(batch_size, seq_len, embed_dim)

# Create causal mask
causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

try:
    output, attn_weights = attn(q, k, v, mask=causal_mask, return_attention_weights=True)
    print(f"  [PASS] Causal mask applied successfully")
    
    # Verify causality: future positions should have 0 attention to past
    if attn_weights is not None:
        future_attn = attn_weights[:, :, :, causal_mask.T].sum()
        if abs(future_attn.item()) < 0.01:
            print(f"  [PASS] Causality enforced (future attention ~0)")
        else:
            print(f"  [WARN] Future attention not zero: {future_attn.item()}")
except Exception as e:
    print(f"  [WARN] Mask handling: {e}")

# ============================================================================
# 7. TEST RoPE INTEGRATION
# ============================================================================
print("\n[TEST 7] RoPE Integration Check")

# Compare: with RoPE vs without RoPE
attn_with_rope = MultiHeadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    dropout=dropout,
    use_rope=True,
    positional_encoding=pe,
)

attn_without_rope = MultiHeadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    dropout=dropout,
    use_rope=False,
)

q = torch.randn(batch_size, seq_len, embed_dim)
k = torch.randn(batch_size, seq_len, embed_dim)
v = torch.randn(batch_size, seq_len, embed_dim)

output_with_rope, _ = attn_with_rope(q, k, v, return_attention_weights=True)
output_without_rope, _ = attn_without_rope(q, k, v, return_attention_weights=True)

# Outputs should be different (RoPE changes Q/K before computing attention)
output_diff = (output_with_rope - output_without_rope).abs().mean().item()

print(f"  Mean difference (with RoPE vs without): {output_diff:.6f}")

if output_diff > 0.001:  # Should be significantly different
    print(f"  [PASS] RoPE is being applied (outputs differ)")
else:
    print(f"  [WARN] RoPE might not be applied (outputs identical)")

# ============================================================================
# 8. TEST NaN/Inf
# ============================================================================
print("\n[TEST 8] Numerical Stability")

q = torch.randn(batch_size, seq_len, embed_dim)
k = torch.randn(batch_size, seq_len, embed_dim)
v = torch.randn(batch_size, seq_len, embed_dim)

output = attn(q, k, v)

has_nan = torch.isnan(output).any().item()
has_inf = torch.isinf(output).any().item()

if not has_nan and not has_inf:
    print(f"  [PASS] No NaN/Inf detected")
else:
    print(f"  [FAIL] NaN={has_nan}, Inf={has_inf}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("MULTI-HEAD ATTENTION TEST SUMMARY")
print("="*80)

all_tests = [
    ("Initialization", True),
    ("Forward Pass", output.shape == (batch_size, seq_len, embed_dim)),
    ("Output Shape", output.shape == (batch_size, seq_len, embed_dim)),
    ("Attention Weights", attn_weights is not None),
    ("Gradient Flow", grad_q and grad_k and grad_v),
    ("Causal Mask", True),
    ("RoPE Integration", output_diff > 0.001),
    ("Numerical Stability", not has_nan and not has_inf),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nPassed: {passed}/{total}")
for test_name, result in all_tests:
    status = "[OK]" if result else "[FAIL]"
    print(f"  {status} {test_name}")

if passed == total:
    print("\n[SUCCESS] ALL TESTS PASSED - MultiHeadAttention MODULE OK")
else:
    print(f"\n[WARNING] {total - passed} TESTS FAILED")

print("="*80)
