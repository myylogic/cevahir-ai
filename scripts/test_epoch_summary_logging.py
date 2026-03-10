# -*- coding: utf-8 -*-
"""
Test script to verify epoch summary logging works with None values.
"""

import sys
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training_management.v2.utils.enhanced_training_logger import EnhancedTrainingLogger

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_epoch_summary_with_none_values():
    """Test that epoch summary handles None values gracefully."""
    logger.info("[TEST] Epoch Summary with None values")
    
    try:
        logger_instance = EnhancedTrainingLogger("TestEpochSummary")
        
        # Test case 1: Training epoch summary (val_loss=None)
        logger.info("\n[CASE 1] Training Epoch Summary (train_loss set, val_loss=None)")
        logger_instance.log_epoch_summary(
            epoch=1,
            train_loss=9.5234,
            val_loss=None,
            overall_acc=0.1389,
            special_accs={"PAD": (0.0003, 50171), "UNK": (0.7396, 96)},
            top5_acc=0.6555,
            entropy=10.9922,
            loss_change=-0.0234,
            convergence_status="Improving"
        )
        
        # Test case 2: Validation epoch summary (both set)
        logger.info("\n[CASE 2] Validation Epoch Summary (both train and val_loss set)")
        logger_instance.log_epoch_summary(
            epoch=1,
            train_loss=9.5234,
            val_loss=9.4127,
            overall_acc=0.2154,
            special_accs={"PAD": (0.0315, 48000), "UNK": (0.9896, 96)},
            top5_acc=0.8816,
            entropy=10.9922,
            loss_change=None,
            convergence_status="Validation"
        )
        
        # Test case 3: Edge case - both None (shouldn't happen but handle it)
        logger.info("\n[CASE 3] Edge case (both None - shouldn't happen)")
        logger_instance.log_epoch_summary(
            epoch=1,
            train_loss=None,
            val_loss=None,
            overall_acc=0.15,
            special_accs={},
            top5_acc=0.65,
            entropy=10.99,
            loss_change=None,
            convergence_status="Unknown"
        )
        
        logger.info("\n[OK] All epoch summary tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Epoch summary test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_epoch_summary_with_none_values()
    sys.exit(0 if success else 1)
