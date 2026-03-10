# -*- coding: utf-8 -*-
"""
Test script to verify that AdvancedTokenMetrics and EnhancedTrainingLogger
are properly integrated into TrainingLoop.

Eğitim loop'undaki token metrikleri entegrasyonunu test eder.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training_management.v2.core.training_loop import TrainingLoop
from training_management.v2.metrics.advanced_token_metrics import AdvancedTokenMetrics
from training_management.v2.utils.enhanced_training_logger import EnhancedTrainingLogger
from training_management.v2.core.loss_computation import LossComputation
from training_management.v2.core.gradient_manager import GradientManager
from training_management.v2.core.batch_processor import BatchProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_training_loop_initialization():
    """Test that TrainingLoop initializes with metrics and logger."""
    logger.info("[TEST 1] TrainingLoop initialization with metrics/logger")
    
    # Mock components
    device = torch.device("cpu")
    config = {
        "grad_accum_steps": 1,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "unk_token_id": 3,
        "vocab_size": 8000,
        "use_amp": False,
        "use_progress_bar": False,
        "batch_size": 32,
        "seq_len": 128,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 20,
    }
    
    # Create minimal model (just a wrapper)
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 8000)
        
        def forward(self, x):
            batch_size, seq_len, _ = x.shape
            return self.linear(x)  # Project to vocab size
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create loss computation, gradient manager, batch processor
    # First create criterion
    criterion = torch.nn.CrossEntropyLoss(
        reduction='none',
        label_smoothing=0.0
    )
    
    loss_comp = LossComputation(
        criterion=criterion,
        logger=logger
    )
    grad_mgr = GradientManager(max_grad_norm=1.0)
    batch_proc = BatchProcessor()
    
    # Create TrainingLoop
    try:
        training_loop = TrainingLoop(
            model=model,
            optimizer=optimizer,
            loss_computation=loss_comp,
            gradient_manager=grad_mgr,
            batch_processor=batch_proc,
            device=device,
            config=config
        )
        
        # Check initialization
        assert hasattr(training_loop, 'advanced_metrics'), "Missing advanced_metrics"
        assert hasattr(training_loop, 'enhanced_logger'), "Missing enhanced_logger"
        assert isinstance(training_loop.advanced_metrics, AdvancedTokenMetrics), \
            "advanced_metrics is not AdvancedTokenMetrics"
        assert isinstance(training_loop.enhanced_logger, EnhancedTrainingLogger), \
            "enhanced_logger is not EnhancedTrainingLogger"
        assert hasattr(training_loop, 'last_batch_metrics'), "Missing last_batch_metrics"
        
        logger.info("[OK] TrainingLoop initialized correctly with metrics and logger")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] TrainingLoop initialization failed: {e}", exc_info=True)
        return False

def test_advanced_metrics_initialization():
    """Test that AdvancedTokenMetrics is initialized correctly."""
    logger.info("[TEST 2] AdvancedTokenMetrics initialization")
    
    try:
        special_tokens = {0: "PAD", 1: "BOS", 2: "EOS", 3: "UNK"}
        metrics = AdvancedTokenMetrics(vocab_size=8000, special_tokens_dict=special_tokens)
        
        assert metrics.vocab_size == 8000, "Vocab size mismatch"
        assert metrics.special_tokens == special_tokens, "Special tokens mismatch"
        
        logger.info("[OK] AdvancedTokenMetrics initialized correctly")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] AdvancedTokenMetrics initialization failed: {e}", exc_info=True)
        return False

def test_enhanced_logger_initialization():
    """Test that EnhancedTrainingLogger is initialized correctly."""
    logger.info("[TEST 3] EnhancedTrainingLogger initialization")
    
    try:
        logger_instance = EnhancedTrainingLogger("TestLogger")
        
        assert hasattr(logger_instance, 'log_batch_with_token_metrics'), \
            "Missing log_batch_with_token_metrics method"
        assert hasattr(logger_instance, 'log_epoch_summary'), \
            "Missing log_epoch_summary method"
        assert hasattr(logger_instance, 'log_critical_warnings'), \
            "Missing log_critical_warnings method"
        
        logger.info("[OK] EnhancedTrainingLogger initialized correctly")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] EnhancedTrainingLogger initialization failed: {e}", exc_info=True)
        return False

def test_metrics_computation():
    """Test that advanced metrics can be computed."""
    logger.info("[TEST 4] Advanced metrics computation")
    
    try:
        special_tokens = {0: "PAD", 1: "BOS", 2: "EOS", 3: "UNK"}
        metrics = AdvancedTokenMetrics(vocab_size=8000, special_tokens_dict=special_tokens)
        
        # Create dummy logits and targets
        batch_size, seq_len, vocab_size = 4, 16, 8000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, 8000, (batch_size, seq_len))
        
        # Compute metrics
        overall_acc, special_accs, top5_acc, entropy = metrics.compute_per_token_accuracy(
            logits=logits,
            targets=targets
        )
        
        assert 0.0 <= overall_acc <= 1.0, f"Invalid overall_acc: {overall_acc}"
        assert 0.0 <= top5_acc <= 1.0, f"Invalid top5_acc: {top5_acc}"
        assert entropy >= 0.0, f"Invalid entropy: {entropy}"
        assert isinstance(special_accs, dict), f"special_accs is not dict: {type(special_accs)}"
        
        logger.info(f"[OK] Metrics computed: overall_acc={overall_acc:.4f}, "
                   f"top5_acc={top5_acc:.4f}, entropy={entropy:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Metrics computation failed: {e}", exc_info=True)
        return False

def test_critical_issues_detection():
    """Test that critical issues are detected."""
    logger.info("[TEST 5] Critical issues detection")
    
    try:
        special_tokens = {0: "PAD", 1: "BOS", 2: "EOS", 3: "UNK"}
        metrics = AdvancedTokenMetrics(vocab_size=8000, special_tokens_dict=special_tokens)
        
        # Simulate low accuracy and high entropy (should trigger warnings)
        overall_acc = 0.05  # Very low
        special_accs = {
            "PAD": (0.01, 100),  # 1% accuracy for PAD
            "EOS": (0.0, 100),   # 0% accuracy for EOS
        }
        entropy = 9.5  # Very high (near max for uniform 8000)
        
        issues = metrics.check_critical_issues(overall_acc, special_accs, entropy)
        
        logger.info(f"[OK] Detected {len(issues)} critical issues: {issues}")
        assert len(issues) > 0, "No critical issues detected when should have"
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Critical issues detection failed: {e}", exc_info=True)
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("TESTING ADVANCED LOGGING INTEGRATION")
    logger.info("=" * 80)
    
    tests = [
        ("AdvancedTokenMetrics Initialization", test_advanced_metrics_initialization),
        ("EnhancedTrainingLogger Initialization", test_enhanced_logger_initialization),
        ("Metrics Computation", test_metrics_computation),
        ("Critical Issues Detection", test_critical_issues_detection),
        ("TrainingLoop Initialization", test_training_loop_initialization),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}...")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        logger.info(f"{status} {test_name}")
    
    logger.info("=" * 80)
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    logger.info("=" * 80)
    
    return all(results.values())

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
