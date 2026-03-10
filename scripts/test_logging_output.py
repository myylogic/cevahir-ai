# -*- coding: utf-8 -*-
"""
Quick test to verify that logging output is actually generated.
Loglama çıkışını test eder.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training_management.v2.metrics.advanced_token_metrics import AdvancedTokenMetrics
from training_management.v2.utils.enhanced_training_logger import EnhancedTrainingLogger

# Setup logging - show everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s: %(message)s'
)

def test_logging_output():
    """Test that logging actually produces output."""
    print("\n" + "=" * 80)
    print("TESTING LOGGING OUTPUT - EOS/BOS/PAD Metrikleri")
    print("=" * 80 + "\n")
    
    special_tokens = {0: "PAD", 1: "BOS", 2: "EOS", 3: "UNK"}
    metrics = AdvancedTokenMetrics(vocab_size=8000, special_tokens_dict=special_tokens)
    logger = EnhancedTrainingLogger("TestLogging")
    
    # Create dummy logits and targets
    batch_size, seq_len, vocab_size = 4, 16, 8000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, 8000, (batch_size, seq_len))
    
    # Compute metrics
    overall_acc, special_accs, top5_acc, entropy = metrics.compute_per_token_accuracy(
        logits=logits,
        targets=targets
    )
    
    print("\n[COMPUTED METRICS]")
    print(f"  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Top-5 Accuracy: {top5_acc:.4f}")
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Special Token Accuracies: {special_accs}")
    
    # Test batch logging
    print("\n[LOGGING BATCH WITH TOKEN METRICS]")
    logger.log_batch_with_token_metrics(
        epoch=1,
        batch_idx=10,
        loss=0.5432,
        overall_acc=overall_acc,
        special_accs=special_accs,
        top5_acc=top5_acc,
        entropy=entropy,
        grad_norm=0.1234
    )
    
    # Test epoch summary
    print("\n[LOGGING EPOCH SUMMARY]")
    logger.log_epoch_summary(
        epoch=1,
        train_loss=0.5234,
        val_loss=0.5432,
        overall_acc=overall_acc,
        special_accs=special_accs,
        top5_acc=top5_acc,
        entropy=entropy,
        loss_change=-0.0234,
        convergence_status="Improving"
    )
    
    # Test critical warnings
    print("\n[LOGGING CRITICAL WARNINGS]")
    issues = metrics.check_critical_issues(overall_acc, special_accs, entropy)
    if issues:
        logger.log_critical_warnings(
            warnings=issues,
            epoch=1,
            batch_idx=10
        )
    
    print("\n" + "=" * 80)
    print("LOGGING TEST COMPLETED")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_logging_output()
