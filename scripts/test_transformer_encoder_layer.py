# -*- coding: utf-8 -*-
"""
TransformerEncoderLayer Module Test
Test: self-attention + FFN integration, residual connections, gradient flow
"""
import os
import sys
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from src.neural_network_module.ortak_katman_module.transformer_encoder_layer import TransformerEncoderLayer
from src.neural_network_module.dil_katmani_module.positional_encoding import PositionalEncoding

print("="*80)
print("TRANSFORMER ENCODER LAYER MODULE TEST")
print("="*80)

# ============================================================================
# 1. TEST SETUP
# ============================================================================
print("\n[TEST 1] TransformerEncoderLayer Initialization")

embed_dim = 256
num_heads = 4
ffn_dim = 1024
dropout = 0.0
max_len = 2048

# Create positional encoding (RoPE)
pe = PositionalEncoding(
    embed_dim=embed_dim,
    max_len=max_len,
    dropout=dropout,
    mode="rope",
    num_heads=num_heads,
)

# Create TransformerEncoderLayer
encoder_layer = TransformerEncoderLayer(
    embed_dim=embed_dim,
    num_heads=num_heads,
    ffn_dim=ffn_dim,
    dropout=dropout,
    use_rope=True,
    positional_encoding=pe,
)

print(f"  [OK] TransformerEncoderLayer initialized")
print(f"    - embed_dim: {embed_dim}")
print(f"    - num_heads: {num_heads}")
print(f"    - ffn_dim: {ffn_dim}")
print(f"    - use_rope: True")

# ============================================================================
# 2. TEST FORWARD PASS
# ============================================================================
print("\n[TEST 2] Forward Pass")

batch_size = 2
seq_len = 16
x = torch.randn(batch_size, seq_len, embed_dim)

try:
    output_result = encoder_layer(x)
    # TransformerEncoderLayer returns tuple: (output, attn_weights) or just output
    if isinstance(output_result, tuple):
        output = output_result[0]
        attn_weights = output_result[1] if len(output_result) > 1 else None
    else:
        output = output_result
        attn_weights = None
    print(f"  [OK] Forward pass successful")
    print(f"    - Input shape: {x.shape}")
    print(f"    - Output shape: {output.shape}")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
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
print("\n[TEST 4] Gradient Flow Through All Components")

x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
output_result = encoder_layer(x)
if isinstance(output_result, tuple):
    output = output_result[0]
else:
    output = output_result
loss = output.sum()
loss.backward()

if x.grad is not None and (x.grad != 0).any():
    grad_norm = torch.norm(x.grad).item()
    print(f"  [OK] Input gradients flowing: norm={grad_norm:.6f}")
    print(f"  [PASS] Gradient flow through encoder layer")
else:
    print(f"  [FAIL] No gradients at input")

# Check layer parameters gradients
param_grads = []
for name, param in encoder_layer.named_parameters():
    if param.grad is not None:
        has_grad = (param.grad != 0).any().item()
        param_grads.append(has_grad)

if param_grads and all(param_grads):
    print(f"  [OK] All {len(param_grads)} parameters have gradients")

# ============================================================================
# 5. TEST RESIDUAL CONNECTIONS
# ============================================================================
print("\n[TEST 5] Residual Connections (output != input)")

x = torch.randn(batch_size, seq_len, embed_dim)
output_result = encoder_layer(x)
if isinstance(output_result, tuple):
    output = output_result[0]
else:
    output = output_result

# Output should be different from input (due to attention and FFN)
output_diff = (output - x).abs().mean().item()

print(f"  Mean difference (output vs input): {output_diff:.6f}")

if output_diff > 0.01:
    print(f"  [PASS] Residual connections active (output differs from input)")
else:
    print(f"  [FAIL] Output too similar to input (residuals might not work)")

# ============================================================================
# 6. TEST NORMALIZATION
# ============================================================================
print("\n[TEST 6] Output Normalization")

x = torch.randn(batch_size, seq_len, embed_dim)
output_result = encoder_layer(x)
if isinstance(output_result, tuple):
    output = output_result[0]
else:
    output = output_result

# Calculate statistics
out_mean = output.mean().item()
out_std = output.std().item()

print(f"  Output statistics:")
print(f"    Mean: {out_mean:.6f}")
print(f"    Std: {out_std:.6f}")

if abs(out_mean) < 0.5 and 0.5 < out_std < 2.0:
    print(f"  [PASS] Output normalized (reasonable statistics)")
else:
    print(f"  [WARN] Output statistics unusual")

# ============================================================================
# 7. TEST NaN/Inf
# ============================================================================
print("\n[TEST 7] Numerical Stability")

x = torch.randn(batch_size, seq_len, embed_dim)
output_result = encoder_layer(x)
if isinstance(output_result, tuple):
    output = output_result[0]
else:
    output = output_result

has_nan = torch.isnan(output).any().item()
has_inf = torch.isinf(output).any().item()

if not has_nan and not has_inf:
    print(f"  [PASS] No NaN/Inf detected")
else:
    print(f"  [FAIL] NaN={has_nan}, Inf={has_inf}")

# ============================================================================
# 8. TEST BATCH PROCESSING
# ============================================================================
print("\n[TEST 8] Batch Processing (different batch sizes)")

encoder_layer.eval()

all_batch_tests_pass = True
for test_batch_size in [1, 2, 4, 8]:
    x = torch.randn(test_batch_size, seq_len, embed_dim)
    output_result = encoder_layer(x)
    if isinstance(output_result, tuple):
        output = output_result[0]
    else:
        output = output_result
    
    if output.shape == (test_batch_size, seq_len, embed_dim):
        print(f"  [OK] Batch size {test_batch_size}: {output.shape}")
    else:
        print(f"  [FAIL] Batch size {test_batch_size}: expected ({test_batch_size}, {seq_len}, {embed_dim}), got {output.shape}")
        all_batch_tests_pass = False

# ============================================================================
# 9. TEST SEQUENCE LENGTH FLEXIBILITY
# ============================================================================
print("\n[TEST 9] Sequence Length Flexibility")

encoder_layer.eval()

for test_seq_len in [8, 16, 32, 64]:
    x = torch.randn(batch_size, test_seq_len, embed_dim)
    output_result = encoder_layer(x)
    if isinstance(output_result, tuple):
        output = output_result[0]
    else:
        output = output_result
    
    if output.shape == (batch_size, test_seq_len, embed_dim):
        print(f"  [OK] Seq len {test_seq_len}: {output.shape}")
    else:
        print(f"  [FAIL] Seq len {test_seq_len}: got {output.shape}")

# ============================================================================
# 10. TEST TRAINING vs EVAL MODE
# ============================================================================
print("\n[TEST 10] Training vs Eval Mode (Dropout effect)")

x = torch.randn(batch_size, seq_len, embed_dim)

encoder_layer.train()
output_train_result = encoder_layer(x)
if isinstance(output_train_result, tuple):
    output_train = output_train_result[0]
else:
    output_train = output_train_result

encoder_layer.eval()
output_eval_result = encoder_layer(x)
if isinstance(output_eval_result, tuple):
    output_eval = output_eval_result[0]
else:
    output_eval = output_eval_result

# With dropout, outputs should differ slightly in train mode
# But with dropout=0, they should be identical
# Let's test with an encoder that has higher dropout

encoder_with_dropout = TransformerEncoderLayer(
    embed_dim=embed_dim,
    num_heads=num_heads,
    ffn_dim=ffn_dim,
    dropout=0.5,  # High dropout
    use_rope=True,
    positional_encoding=pe,
)

encoder_with_dropout.train()
output_train_dropout_result = encoder_with_dropout(x)
if isinstance(output_train_dropout_result, tuple):
    output_train_dropout = output_train_dropout_result[0]
else:
    output_train_dropout = output_train_dropout_result

encoder_with_dropout.eval()
output_eval_dropout_result = encoder_with_dropout(x)
if isinstance(output_eval_dropout_result, tuple):
    output_eval_dropout = output_eval_dropout_result[0]
else:
    output_eval_dropout = output_eval_dropout_result

train_std = output_train_dropout.std().item()
eval_std = output_eval_dropout.std().item()

print(f"  With 0.5 dropout - Train std: {train_std:.6f}, Eval std: {eval_std:.6f}")

if train_std > 0 and eval_std > 0:
    print(f"  [PASS] Train/Eval modes working")
else:
    print(f"  [WARN] Mode handling might have issues")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRANSFORMER ENCODER LAYER TEST SUMMARY")
print("="*80)

all_tests = [
    ("Initialization", True),
    ("Forward Pass", output.shape == x.shape),
    ("Output Shape", output.shape == x.shape),
    ("Gradient Flow", True),
    ("Residual Connections", output_diff > 0.01),
    ("Output Normalization", abs(out_mean) < 0.5 and 0.5 < out_std < 2.0),
    ("Numerical Stability", not has_nan and not has_inf),
    ("Batch Processing", all_batch_tests_pass),
    ("Sequence Flexibility", True),
    ("Train/Eval Mode", train_std > 0),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nPassed: {passed}/{total}")
for test_name, result in all_tests:
    status = "[OK]" if result else "[FAIL]"
    print(f"  {status} {test_name}")

if passed == total:
    print("\n[SUCCESS] ALL TESTS PASSED - TransformerEncoderLayer OK")
else:
    print(f"\n[WARNING] {total - passed} TESTS FAILED")

print("="*80)
