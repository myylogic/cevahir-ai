"""
Training Management V2
======================

Modular, SOLID-compliant training management system.

V2 Features:
- Modular architecture (15+ modules)
- SOLID principles compliance
- Comprehensive test coverage
- Industry-standard design
- EOS weight bug fix (from V1)
"""

# Core modules
from .core.training_manager import TrainingManager
from .core.training_loop import TrainingLoop
from .core.loss_computation import LossComputation
from .core.gradient_manager import GradientManager
from .core.batch_processor import BatchProcessor

# Metrics modules
from .metrics.metrics_tracker import MetricsTracker
from .metrics.metrics_calculator import MetricsCalculator
from .metrics.advanced_metrics import AdvancedMetrics

# Utils (from V1, reused in V2)
from .utils import TrainingLogger, TrainingScheduler, CheckpointManager, TrainingVisualizer

__version__ = "2.0.0"

__all__ = [
    "TrainingManager",
    "TrainingLoop",
    "LossComputation",
    "GradientManager",
    "BatchProcessor",
    "MetricsTracker",
    "MetricsCalculator",
    "AdvancedMetrics",
    "TrainingLogger",
    "TrainingScheduler",
    "CheckpointManager",
    "TrainingVisualizer",
]
