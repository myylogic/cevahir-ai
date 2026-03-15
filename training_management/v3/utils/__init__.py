"""
Cevahir V3 Training Management - Utils Paketi
==============================================
V3 yardımcı araçlarını dışa aktarır:
  - EMA                  : Exponential Moving Average ağırlık yönetimi
  - TrainingScheduler    : LR scheduler factory
  - CheckpointManager    : Atomik checkpoint kayıt/yükleme
  - MetricsTracker       : Epoch/batch metrik geçmişi
  - TrainingLogger       : Renkli ve JSONL-tabanlı training logger
"""

# Mevcut araçlar
try:
    from .ema import EMA
    _HAS_EMA = True
except ImportError:
    EMA = None  # type: ignore[assignment,misc]
    _HAS_EMA = False

try:
    from .training_scheduler import TrainingScheduler, SchedulerConfig, SchedulerType
    _HAS_SCHEDULER = True
except ImportError:
    TrainingScheduler = None  # type: ignore[assignment,misc]
    SchedulerConfig   = None  # type: ignore[assignment,misc]
    SchedulerType     = None  # type: ignore[assignment,misc]
    _HAS_SCHEDULER = False

# V3 yeni araçlar
from .checkpoint_manager import CheckpointManager, CheckpointData, CheckpointVerifier
from .metrics_tracker import MetricsTracker
from .training_logger import TrainingLogger

__all__ = [
    # Mevcut
    "EMA",
    "TrainingScheduler",
    "SchedulerConfig",
    "SchedulerType",
    # Yeni V3
    "CheckpointManager",
    "CheckpointData",
    "CheckpointVerifier",
    "MetricsTracker",
    "TrainingLogger",
]
