# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ — Training Management V3
================================================================================
Modül : training_management.v3
Görev : Cevahir V3 eğitim yönetim sisteminin üst-düzey paketi.

Yenilikler (V2 → V3):
  Kayıp Fonksiyonu:
    - Label Smoothing (Szegedy et al. 2016)
    - Entropy Regularization — entropy collapse önleme (Pereyra et al. 2017)
    - Focal Loss — imbalanced sınıflar için (Lin et al. 2017)
    - Auxiliary MoE/MoD loss ağırlığı

  Exposure Bias:
    - Scheduled Sampling (Bengio et al. 2015)

  Optimizasyon:
    - SAM Optimizer (Foret et al. 2021)
    - Lookahead Optimizer (Zhang et al. 2019)
    - Layer-wise LR Decay — LLRD
    - Cosine Annealing with Warm Restarts (Loshchilov & Hutter 2016)

  Ağırlık Ortalama:
    - EMA Weights (Yazici et al. 2019)
    - SWA — Stochastic Weight Averaging (Izmailov et al. 2018)

  Veri:
    - Curriculum Learning (Bengio et al. 2009)

  İzleme:
    - Inference Quality Probe (her N epoch'ta üretim kalitesi ölçümü)
    - Gradient Health Monitor
    - Adaptive Gradient Clipping — AGC (Brock et al. 2021)

  Güvenlik:
    - NaN Kurtarma (nan_tolerance adım sonrası checkpoint'e geri dön)
    - Loss Spike Detection (N-sigma pencere ile anlık sıçramaları yakala)

Kullanım::

    from training_management.v3 import TrainingManager, CompositeLossManager

    manager = TrainingManager(config=train_config, ...)
    manager.train()

Paket yapısı::

    training_management/v3/
    ├── __init__.py                  ← Bu dosya
    ├── core/
    │   ├── training_manager.py      ← Ana eğitim orkestratörü (Facade)
    │   ├── training_loop.py         ← Epoch döngüsü + Scheduled Sampling
    │   ├── loss_manager.py          ← CompositeLossManager (CE + Entropy + Focal + Aux)
    │   ├── gradient_manager.py      ← Gradient kırpma, AGC, gürültü
    │   └── batch_processor.py       ← Batch ön işleme, curriculum filtreleme
    ├── utils/
    │   ├── checkpoint_manager.py    ← Atomik kayıt, slot yönetimi, rotasyon
    │   ├── metrics_tracker.py       ← Epoch/batch metrik geçmişi
    │   ├── training_logger.py       ← Renkli terminal + JSONL logging
    │   ├── ema.py                   ← EMA weights
    │   └── training_scheduler.py   ← Scheduler factory (Cosine, ReduceOnPlateau, vb.)
    ├── optimizers/
    │   ├── sam.py                   ← SAM Optimizer
    │   └── lookahead.py             ← Lookahead Optimizer
    └── curriculum/
        └── curriculum_sampler.py   ← Curriculum Learning veri sıralayıcısı

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sürüm bilgisi
# ---------------------------------------------------------------------------

__version__: str = "3.0.0"
__author__:  str = "Muhammed Yasin Yılmaz"

# ---------------------------------------------------------------------------
# Core bileşenler — lazy import ile
# ---------------------------------------------------------------------------
# Her import bloğu kendi hata mesajını üretir; bir modül eksikse
# sadece o modül import edilemez, gerisi çalışmaya devam eder.

# --- Core ---
try:
    from training_management.v3.core.training_manager import (
        TrainingManager,
        TrainingManagerConfig,
    )
    _HAS_TRAINING_MANAGER = True
except ImportError as _e:
    logger.warning("[V3] TrainingManager import edilemedi: %s", _e)
    TrainingManager = None  # type: ignore[assignment,misc]
    TrainingManagerConfig = None  # type: ignore[assignment,misc]
    _HAS_TRAINING_MANAGER = False

try:
    from training_management.v3.core.loss_manager import (
        CompositeLossManager,
        LossConfig,
        LossOutput,
    )
    _HAS_LOSS_MANAGER = True
except ImportError as _e:
    logger.warning("[V3] CompositeLossManager import edilemedi: %s", _e)
    CompositeLossManager = None  # type: ignore[assignment,misc]
    LossConfig = None            # type: ignore[assignment,misc]
    LossOutput = None            # type: ignore[assignment,misc]
    _HAS_LOSS_MANAGER = False

try:
    from training_management.v3.core.training_loop import (
        TrainingLoop,
        TrainingLoopConfig,
        EpochMetrics,
        ScheduledSamplingMixin,
    )
    _HAS_TRAINING_LOOP = True
except ImportError as _e:
    logger.warning("[V3] TrainingLoop import edilemedi: %s", _e)
    TrainingLoop = None              # type: ignore[assignment,misc]
    TrainingLoopConfig = None        # type: ignore[assignment,misc]
    EpochMetrics = None              # type: ignore[assignment,misc]
    ScheduledSamplingMixin = None    # type: ignore[assignment,misc]
    _HAS_TRAINING_LOOP = False

try:
    from training_management.v3.core.gradient_manager import GradientManager
    _HAS_GRADIENT_MANAGER = True
except ImportError as _e:
    logger.warning("[V3] GradientManager import edilemedi: %s", _e)
    GradientManager = None  # type: ignore[assignment,misc]
    _HAS_GRADIENT_MANAGER = False

try:
    from training_management.v3.core.batch_processor import BatchProcessor
    _HAS_BATCH_PROCESSOR = True
except ImportError as _e:
    logger.warning("[V3] BatchProcessor import edilemedi: %s", _e)
    BatchProcessor = None  # type: ignore[assignment,misc]
    _HAS_BATCH_PROCESSOR = False

# --- Utils ---
try:
    from training_management.v3.utils.checkpoint_manager import (
        CheckpointManager,
        CheckpointData,
    )
    _HAS_CHECKPOINT_MANAGER = True
except ImportError as _e:
    logger.warning("[V3] CheckpointManager import edilemedi: %s", _e)
    CheckpointManager = None    # type: ignore[assignment,misc]
    CheckpointData = None       # type: ignore[assignment,misc]
    _HAS_CHECKPOINT_MANAGER = False

try:
    from training_management.v3.safety.checkpoint_verifier import CheckpointVerifier
    _HAS_CHECKPOINT_VERIFIER = True
except ImportError as _e:
    logger.warning("[V3] CheckpointVerifier import edilemedi: %s", _e)
    CheckpointVerifier = None   # type: ignore[assignment,misc]
    _HAS_CHECKPOINT_VERIFIER = False

try:
    from training_management.v3.utils.metrics_tracker import MetricsTracker
    _HAS_METRICS_TRACKER = True
except ImportError as _e:
    logger.warning("[V3] MetricsTracker import edilemedi: %s", _e)
    MetricsTracker = None  # type: ignore[assignment,misc]
    _HAS_METRICS_TRACKER = False

try:
    from training_management.v3.utils.training_logger import TrainingLogger
    _HAS_TRAINING_LOGGER = True
except ImportError as _e:
    logger.warning("[V3] TrainingLogger import edilemedi: %s", _e)
    TrainingLogger = None  # type: ignore[assignment,misc]
    _HAS_TRAINING_LOGGER = False

try:
    from training_management.v3.utils.ema import EMA
    _HAS_EMA = True
except ImportError as _e:
    logger.debug("[V3] EMA import edilemedi: %s", _e)
    EMA = None  # type: ignore[assignment,misc]
    _HAS_EMA = False

try:
    from training_management.v3.utils.training_scheduler import (
        TrainingScheduler,
        SchedulerConfig,
        SchedulerType,
    )
    _HAS_SCHEDULER = True
except ImportError as _e:
    logger.debug("[V3] TrainingScheduler import edilemedi: %s", _e)
    TrainingScheduler = None  # type: ignore[assignment,misc]
    SchedulerConfig = None    # type: ignore[assignment,misc]
    SchedulerType = None      # type: ignore[assignment,misc]
    _HAS_SCHEDULER = False

# --- Optimizers ---
try:
    from training_management.v3.optimizers.sam import SAM
    _HAS_SAM = True
except ImportError as _e:
    logger.debug("[V3] SAM import edilemedi: %s", _e)
    SAM = None  # type: ignore[assignment,misc]
    _HAS_SAM = False

try:
    from training_management.v3.optimizers.lookahead import Lookahead
    _HAS_LOOKAHEAD = True
except ImportError as _e:
    logger.debug("[V3] Lookahead import edilemedi: %s", _e)
    Lookahead = None  # type: ignore[assignment,misc]
    _HAS_LOOKAHEAD = False

# --- Curriculum ---
try:
    from training_management.v3.curriculum.curriculum_manager import CurriculumManager
    # Backward-compat alias
    CurriculumSampler = CurriculumManager
    _HAS_CURRICULUM = True
except ImportError as _e:
    logger.debug("[V3] CurriculumManager import edilemedi: %s", _e)
    CurriculumManager = None   # type: ignore[assignment,misc]
    CurriculumSampler = None   # type: ignore[assignment,misc]
    _HAS_CURRICULUM = False

# ---------------------------------------------------------------------------
# __all__ — dışa aktarılacak isimler
# ---------------------------------------------------------------------------

__all__: List[str] = [
    # Sürüm
    "__version__",
    "__author__",
    # Core
    "TrainingManager",
    "TrainingManagerConfig",
    "CompositeLossManager",
    "LossConfig",
    "LossOutput",
    "TrainingLoop",
    "TrainingLoopConfig",
    "EpochMetrics",
    "ScheduledSamplingMixin",
    "GradientManager",
    "BatchProcessor",
    # Utils
    "CheckpointManager",
    "CheckpointData",
    "MetricsTracker",
    "TrainingLogger",
    "EMA",
    "TrainingScheduler",
    "SchedulerConfig",
    "SchedulerType",
    # Optimizers
    "SAM",
    "Lookahead",
    # Safety
    "CheckpointVerifier",
    # Curriculum
    "CurriculumManager",
    "CurriculumSampler",   # alias
]

# ---------------------------------------------------------------------------
# Kullanılabilirlik özeti (geliştirici bilgisi)
# ---------------------------------------------------------------------------

def get_availability() -> dict:
    """
    V3 bileşenlerinin import durumunu sözlük olarak döner.

    Kullanım::

        from training_management.v3 import get_availability
        avail = get_availability()
        print(avail)
    """
    return {
        "TrainingManager":    _HAS_TRAINING_MANAGER,
        "CompositeLossManager": _HAS_LOSS_MANAGER,
        "TrainingLoop":       _HAS_TRAINING_LOOP,
        "GradientManager":    _HAS_GRADIENT_MANAGER,
        "BatchProcessor":     _HAS_BATCH_PROCESSOR,
        "CheckpointManager":    _HAS_CHECKPOINT_MANAGER,
        "CheckpointVerifier":   _HAS_CHECKPOINT_VERIFIER,
        "MetricsTracker":       _HAS_METRICS_TRACKER,
        "TrainingLogger":       _HAS_TRAINING_LOGGER,
        "EMA":                  _HAS_EMA,
        "TrainingScheduler":    _HAS_SCHEDULER,
        "SAM":                  _HAS_SAM,
        "Lookahead":            _HAS_LOOKAHEAD,
        "CurriculumManager":    _HAS_CURRICULUM,
    }


def _log_availability() -> None:
    """Eksik bileşenleri INFO düzeyinde loglar."""
    avail = get_availability()
    missing = [k for k, v in avail.items() if not v]
    available = [k for k, v in avail.items() if v]

    logger.info(
        "[training_management.v3] v%s yüklendi — %d/%d bileşen hazır",
        __version__,
        len(available),
        len(avail),
    )
    if missing:
        logger.debug("[training_management.v3] Eksik bileşenler: %s", missing)


# Paket import edildiğinde availability'yi logla
_log_availability()
