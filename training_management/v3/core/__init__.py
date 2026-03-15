"""
Cevahir V3 Eğitim Yönetim Sistemi - Core Modülü
================================================
Bu modül, Cevahir Türkçe dil modelinin V3 eğitim altyapısının
çekirdek bileşenlerini dışa aktarır.

Bileşenler:
    - TrainingManager  : Ana eğitim orkestratörü (Facade pattern)
    - TrainingLoop     : Epoch bazlı eğitim döngüsü
    - CompositeLossManager : Bileşik kayıp fonksiyonu yöneticisi
    - GradientManager  : Gradyan hesaplama ve kırpma yöneticisi
    - BatchProcessor   : Batch verisi ön işleme
"""

from training_management.v3.core.loss_manager import (
    CompositeLossManager,
    LossConfig,
    LossOutput,
)
from training_management.v3.core.batch_processor import BatchProcessor
from training_management.v3.core.gradient_manager import GradientManager
from training_management.v3.core.training_loop import (
    TrainingLoop,
    TrainingLoopConfig,
    EpochMetrics,
    ScheduledSamplingMixin,
)
from training_management.v3.core.training_manager import (
    TrainingManager,
    TrainingManagerConfig,
)

__all__ = [
    "TrainingManager",
    "TrainingManagerConfig",
    "TrainingLoop",
    "TrainingLoopConfig",
    "EpochMetrics",
    "ScheduledSamplingMixin",
    "CompositeLossManager",
    "LossConfig",
    "LossOutput",
    "GradientManager",
    "BatchProcessor",
]
