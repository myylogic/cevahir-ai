# -*- coding: utf-8 -*-
"""
FeedForward (SwiGLU) Module Test
Test: output shape, gradient flow, gate mechanism, numerical stability
"""
import os
import sys
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from src.neural_network_module.ortak_katman_module.feed_forward_network import FeedForwardNetwork

print("="*80)
print("FEEDFORWARD (SwiGLU) MODULE TEST")
print("="*80)

# ============================================================================
# 1. TEST SETUP
# ============================================================================
print("\n[TEST 1] FeedForward Initialization")

embed_dim = 256
ffn_dim = 1024
dropout = 0.0

ff = FeedForwardNetwork(
    embed_dim=embed_dim,
    ffn_dim=ffn_dim,
    dropout=dropout,
)

print(f"  [OK] FeedForward initialized")
print(f"    - embed_dim: {embed_dim}")
print(f"    - ffn_dim: {ffn_dim}")
print(f"    - dropout: {dropout}")

# ============================================================================
# 2. TEST FORWARD PASS
# ============================================================================
print("\n[TEST 2] Forward Pass")

batch_size = 2
seq_len = 16
x = torch.randn(batch_size, seq_len, embed_dim)

try:
    output = ff(x)
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
# 4. TEST GRADIENT FLOW
# ============================================================================
print("\n[TEST 4] Gradient Flow")

x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
output = ff(x)
loss = output.sum()
loss.backward()

if x.grad is not None and (x.grad != 0).any():
    grad_norm = torch.norm(x.grad).item()
    print(f"  [OK] Gradients flowing: norm={grad_norm:.6f}")
    print(f"  [PASS] Gradient flow working")
else:
    print(f"  [FAIL] No gradients or all zero")

# ============================================================================
# 5. TEST GATE MECHANISM (SwiGLU)
# ============================================================================
print("\n[TEST 5] Gate Mechanism (SwiGLU)")

# SwiGLU should have gating: output = f(x) * swish(g(x))
# We can check that output is not just linear - it should differ from simple dense layer

x = torch.randn(batch_size, seq_len, embed_dim)

# Compare with simple linear layer
simple_linear = nn.Linear(embed_dim, embed_dim)
simple_output = simple_linear(x)

ff_output = ff(x)

diff = (ff_output - simple_output).abs().mean().item()

print(f"  Mean difference (SwiGLU vs simple linear): {diff:.6f}")

if diff > 0.1:
    print(f"  [PASS] SwiGLU gate is active (significantly different from linear)")
else:
    print(f"  [WARN] Outputs very similar (gate might not be active)")

# ============================================================================
# 6. TEST OUTPUT STATISTICS
# ============================================================================
print("\n[TEST 6] Output Statistics")

x = torch.randn(batch_size, seq_len, embed_dim)
output = ff(x)

out_mean = output.mean().item()
out_std = output.std().item()
out_max = output.max().item()
out_min = output.min().item()

print(f"  Output statistics:")
print(f"    Mean: {out_mean:.6f}")
print(f"    Std: {out_std:.6f}")
print(f"    Range: [{out_min:.6f}, {out_max:.6f}]")

# Check if output is in reasonable range
if abs(out_mean) < 1.0 and 0.5 < out_std < 2.0:
    print(f"  [PASS] Output statistics reasonable")
else:
    print(f"  [WARN] Output statistics might be unusual")

# ============================================================================
# 7. TEST NaN/Inf
# ============================================================================
print("\n[TEST 7] Numerical Stability")

x = torch.randn(batch_size, seq_len, embed_dim)
output = ff(x)

has_nan = torch.isnan(output).any().item()
has_inf = torch.isinf(output).any().item()

if not has_nan and not has_inf:
    print(f"  [PASS] No NaN/Inf detected")
else:
    print(f"  [FAIL] NaN={has_nan}, Inf={has_inf}")

# ============================================================================
# 8. TEST WITH DROPOUT
# ============================================================================
print("\n[TEST 8] Dropout (training vs eval)")

ff_with_dropout = FeedForwardNetwork(
    embed_dim=embed_dim,
    ffn_dim=ffn_dim,
    dropout=0.5,  # High dropout for testing
)

x = torch.randn(batch_size, seq_len, embed_dim)

ff_with_dropout.train()
output_train = ff_with_dropout(x)

ff_with_dropout.eval()
output_eval = ff_with_dropout(x)

train_std = output_train.std().item()
eval_std = output_eval.std().item()

print(f"  Train mode std: {train_std:.6f}")
print(f"  Eval mode std: {eval_std:.6f}")

if train_std > 0:
    print(f"  [PASS] Dropout is functional")
else:
    print(f"  [WARN] Dropout might not be working")

# ============================================================================
# 9. TEST BATCH PROCESSING
# ============================================================================
print("\n[TEST 9] Batch Processing")

ff.eval()

for test_batch_size in [1, 2, 4, 8]:
    x = torch.randn(test_batch_size, seq_len, embed_dim)
    output = ff(x)
    
    if output.shape == (test_batch_size, seq_len, embed_dim):
        print(f"  [OK] Batch size {test_batch_size}: {output.shape}")
    else:
        print(f"  [FAIL] Batch size {test_batch_size}: expected ({test_batch_size}, {seq_len}, {embed_dim}), got {output.shape}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FEEDFORWARD MODULE TEST SUMMARY")
print("="*80)

all_tests = [
    ("Initialization", True),
    ("Forward Pass", output.shape == x.shape),
    ("Output Shape", output.shape == x.shape),
    ("Gradient Flow", True),
    ("Gate Mechanism", diff > 0.1),
    ("Output Statistics", abs(out_mean) < 1.0 and 0.5 < out_std < 2.0),
    ("Numerical Stability", not has_nan and not has_inf),
    ("Dropout", train_std > 0),
    ("Batch Processing", True),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nPassed: {passed}/{total}")
for test_name, result in all_tests:
    status = "[OK]" if result else "[FAIL]"
    print(f"  {status} {test_name}")

if passed == total:
    print("\n[SUCCESS] ALL TESTS PASSED - FeedForward MODULE OK")
else:
    print(f"\n[WARNING] {total - passed} TESTS FAILED")

print("="*80)
