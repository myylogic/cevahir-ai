# -*- coding: utf-8 -*-
"""
Advanced Logging Test - Yeni logging modüllerinin doğru çalışıp çalışmadığını test et
"""
import os
import sys
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from training_management.v2.metrics.advanced_token_metrics import AdvancedTokenMetrics
from training_management.v2.utils.enhanced_training_logger import EnhancedTrainingLogger

print("="*80)
print("ADVANCED LOGGING MODULE TEST")
print("="*80)

# ============================================================================
# TEST 1: Advanced Token Metrics Initialization
# ============================================================================
print("\n[TEST 1] AdvancedTokenMetrics Initialization")

vocab_size = 8000
special_tokens = {0: "PAD", 1: "BOS", 2: "EOS", 3: "UNK"}

try:
    metrics = AdvancedTokenMetrics(vocab_size, special_tokens)
    print(f"  [OK] AdvancedTokenMetrics initialized")
    print(f"    - vocab_size: {vocab_size}")
    print(f"    - special_tokens: {special_tokens}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Per-Token Accuracy Computation
# ============================================================================
print("\n[TEST 2] Per-Token Accuracy Computation")

batch_size = 2
seq_len = 16
vocab_size = 8000

logits = torch.randn(batch_size * seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (batch_size * seq_len,))

# Add some special tokens to targets
targets[0:10] = 2  # EOS
targets[10:20] = 1  # BOS
targets[20:100] = 0  # PAD
targets[100:110] = 3  # UNK

try:
    overall_acc, special_accs, top5_acc, entropy = metrics.compute_per_token_accuracy(logits, targets)
    
    print(f"  [OK] Per-token accuracy computed")
    print(f"    - Overall Accuracy: {overall_acc:.2%}")
    print(f"    - Special Token Accuracies:")
    for token_name, (acc, count) in special_accs.items():
        print(f"      |- {token_name:6s}: {acc:.2%} ({count} samples)")
    print(f"    - Top-5 Accuracy: {top5_acc:.2%}")
    print(f"    - Entropy: {entropy:.4f}")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: Special Token Probabilities
# ============================================================================
print("\n[TEST 3] Special Token Probabilities")

try:
    token_probs = metrics.compute_special_token_probabilities(logits, targets)
    
    print(f"  [OK] Special token probabilities computed")
    for token_name, prob in token_probs.items():
        print(f"    - {token_name}: {prob:.6f}")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 4: Loss per Token
# ============================================================================
print("\n[TEST 4] Loss per Token Type")

loss_fn = nn.CrossEntropyLoss(reduction='none')

try:
    token_losses = metrics.compute_loss_per_token(logits, targets, 
                                                   nn.CrossEntropyLoss())
    
    print(f"  [OK] Loss per token computed")
    for token_name, loss in token_losses.items():
        print(f"    - {token_name}: {loss:.6f}")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 5: Critical Issues Detection
# ============================================================================
print("\n[TEST 5] Critical Issues Detection")

try:
    warnings = metrics.check_critical_issues(overall_acc, special_accs, entropy)
    
    if warnings:
        print(f"  [OK] Critical issues detected: {len(warnings)} warnings")
    else:
        print(f"  [OK] No critical issues detected")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: Format Token Metrics
# ============================================================================
print("\n[TEST 6] Format Token Metrics (Readable Output)")

try:
    formatted = metrics.format_token_metrics(overall_acc, special_accs, 
                                            top5_acc, entropy, grad_norm=1.234)
    
    print(f"  [OK] Token metrics formatted (readable format ready)")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 7: Enhanced Training Logger Initialization
# ============================================================================
print("\n[TEST 7] EnhancedTrainingLogger Initialization")

try:
    logger = EnhancedTrainingLogger("TestLogger")
    print(f"  [OK] EnhancedTrainingLogger initialized")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 8: Log Batch Metrics
# ============================================================================
print("\n[TEST 8] Log Batch with Token Metrics")

try:
    logger.log_batch_with_token_metrics(
        epoch=1,
        batch_idx=10,
        loss=8.2345,
        overall_acc=0.0123,
        special_accs=special_accs,
        top5_acc=0.0845,
        entropy=9.234,
        grad_norm=1.234
    )
    print(f"  [OK] Batch metrics logged")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 9: Log Epoch Summary
# ============================================================================
print("\n[TEST 9] Log Epoch Summary")

try:
    logger.log_epoch_summary(
        epoch=1,
        train_loss=8.1234,
        val_loss=8.1567,
        overall_acc=0.0145,
        special_accs=special_accs,
        top5_acc=0.0889,
        entropy=9.245,
        loss_change=-0.0033,
        convergence_status="Patience 1/20"
    )
    print(f"  [OK] Epoch summary logged")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 10: Log Critical Warnings
# ============================================================================
print("\n[TEST 10] Log Critical Warnings")

try:
    test_warnings = [
        "[CRITICAL] EOS Token Not Learning! Accuracy: 0.0012",
        "[WARNING] Mode Collapse Detected: Entropy=8.234"
    ]
    
    logger.log_critical_warnings(test_warnings, epoch=1, batch_idx=10)
    print(f"  [OK] Critical warnings logged")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 11: Log Convergence Status
# ============================================================================
print("\n[TEST 11] Log Convergence Status")

try:
    logger.log_convergence_status(
        epoch=1,
        loss_change=-0.0033,
        patience=1,
        max_patience=20,
        ema_loss=8.195
    )
    print(f"  [OK] Convergence status logged")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 12: Log Checkpoint Save
# ============================================================================
print("\n[TEST 12] Log Checkpoint Save")

try:
    logger.log_checkpoint_save(
        epoch=1,
        train_loss=8.1234,
        val_loss=8.1567,
        eos_acc=0.0018,
        checkpoint_path="saved_models/checkpoint_epoch_1.pth"
    )
    print(f"  [OK] Checkpoint save logged")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 13: Log Training Start
# ============================================================================
print("\n[TEST 13] Log Training Start")

try:
    config = {
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'eos_token_weight': 10.0,
        'label_smoothing': 0.0
    }
    
    logger.log_training_start(config)
    print(f"  [OK] Training start logged")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 14: Log Training End
# ============================================================================
print("\n[TEST 14] Log Training End")

try:
    logger.log_training_end(
        final_train_loss=2.8912,
        final_val_loss=3.1234
    )
    print(f"  [OK] Training end logged")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 15: Integration Test - Simulated Training Step
# ============================================================================
print("\n[TEST 15] Integration Test - Simulated Training Step")

try:
    print(f"\n  Simulating 3 training steps...\n")
    
    for step in range(1, 4):
        # Simulate batch
        logits = torch.randn(batch_size * seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
        
        # Add special tokens
        targets[0:10] = 2  # EOS
        targets[10:20] = 1  # BOS
        targets[20:100] = 0  # PAD
        
        # Compute metrics
        overall_acc, special_accs, top5_acc, entropy = metrics.compute_per_token_accuracy(logits, targets)
        
        # Log batch
        logger.log_batch_with_token_metrics(
            epoch=1,
            batch_idx=step * 10,
            loss=8.5 - step * 0.1,
            overall_acc=overall_acc,
            special_accs=special_accs,
            top5_acc=top5_acc,
            entropy=entropy,
            grad_norm=1.5 - step * 0.1
        )
        
        # Check for critical issues
        warnings = metrics.check_critical_issues(overall_acc, special_accs, entropy)
        if warnings:
            logger.log_critical_warnings(warnings, epoch=1, batch_idx=step * 10)
    
    print(f"  [OK] Simulated training steps completed")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ADVANCED LOGGING TEST SUMMARY")
print("="*80)

all_tests = [
    ("AdvancedTokenMetrics Init", True),
    ("Per-Token Accuracy", True),
    ("Special Token Probabilities", True),
    ("Loss per Token", True),
    ("Critical Issues Detection", True),
    ("Format Token Metrics", True),
    ("Logger Init", True),
    ("Log Batch Metrics", True),
    ("Log Epoch Summary", True),
    ("Log Critical Warnings", True),
    ("Log Convergence Status", True),
    ("Log Checkpoint Save", True),
    ("Log Training Start", True),
    ("Log Training End", True),
    ("Integration Test", True),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nPassed: {passed}/{total}")
for test_name, result in all_tests:
    status = "[OK]" if result else "[FAIL]"
    print(f"  {status} {test_name}")

if passed == total:
    print("\n[SUCCESS] ALL TESTS PASSED - Advanced Logging Ready!")
else:
    print(f"\n[WARNING] {total - passed} TESTS FAILED")

print("="*80)
