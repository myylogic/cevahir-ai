# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: __init__.py
Modül: model_management
Görev: model_management paketinin genel API'si. Tüm public sınıf, fonksiyon
       ve exception'ları tek noktadan dışa aktarır.

       Katmanlar (içeri → dışarı):
       ─────────────────────────────────────────────────────────────────────
       exceptions.py     → hata hiyerarşisi (CevahirModelError ve alt sınıflar)
       config_schema.py  → typed config dataclass'ları (CevahirConfig, vb.)
       profiler.py       → model profil araçları (ModelProfiler, ParamStats, …)
       health_monitor.py → sağlık izleme (ModelHealthMonitor, HealthReport, …)
       model_initializer → model/optimizer/scheduler oluşturma
       model_saver.py    → checkpoint kaydetme (SHA-256, akıllı budama)
       model_loader.py   → checkpoint yükleme (versiyon kontrolü, integrity)
       model_manager.py  → üst düzey API (ModelManager)
       ─────────────────────────────────────────────────────────────────────

KULLANIM:
    from model_management import ModelManager, CevahirConfig, CevahirModelError
    from model_management import ModelProfiler, ModelHealthMonitor
    from model_management import OOMRecoveryError, CheckpointNotFoundError

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

# ── Hata Hiyerarşisi ──────────────────────────────────────────────────────────
from .exceptions import (
    # Kök
    CevahirModelError,

    # Başlatma / derleme
    ModelNotInitializedError,
    ModelBuildError,
    QuantizationError,

    # Checkpoint
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointCorruptError,
    CheckpointVersionError,

    # Forward / Inference
    ForwardError,
    OOMRecoveryError,

    # Device
    DeviceError,
    DeviceMismatchError,

    # Şekil / Vocab
    ShapeError,
    VocabSizeMismatchError,

    # Dağıtık eğitim
    DistributedSetupError,

    # Sağlık testi
    HealthCheckError,
)

# ── Typed Config Şeması ───────────────────────────────────────────────────────
from .config_schema import (
    ModelArchConfig,
    TrainingConfig,
    CheckpointConfig,
    DistributedConfig,
    QuantConfig,
    CevahirConfig,
)

# ── Model Profil Araçları ─────────────────────────────────────────────────────
from .profiler import (
    ModelProfiler,
    ParamStats,
    MemorySnapshot,
    FlopEstimate,
    TimingResult,
)

# ── Model Sağlık İzleme ───────────────────────────────────────────────────────
from .health_monitor import (
    ModelHealthMonitor,
    HealthReport,
    GradientHealth,
    WeightHealth,
    AttentionHealth,
)

# ── Çekirdek Bileşenler ───────────────────────────────────────────────────────
from .model_manager import ModelManager
from .model_initializer import ModelInitializer
from .model_saver import ModelSaver
from .model_loader import ModelLoader

# ── Genel API Listesi ─────────────────────────────────────────────────────────
__all__ = [
    # Ana API
    "ModelManager",
    "ModelInitializer",
    "ModelSaver",
    "ModelLoader",

    # Config
    "CevahirConfig",
    "ModelArchConfig",
    "TrainingConfig",
    "CheckpointConfig",
    "DistributedConfig",
    "QuantConfig",

    # Profil
    "ModelProfiler",
    "ParamStats",
    "MemorySnapshot",
    "FlopEstimate",
    "TimingResult",

    # Sağlık
    "ModelHealthMonitor",
    "HealthReport",
    "GradientHealth",
    "WeightHealth",
    "AttentionHealth",

    # Hatalar — kök
    "CevahirModelError",

    # Hatalar — başlatma
    "ModelNotInitializedError",
    "ModelBuildError",
    "QuantizationError",

    # Hatalar — checkpoint
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointCorruptError",
    "CheckpointVersionError",

    # Hatalar — forward
    "ForwardError",
    "OOMRecoveryError",

    # Hatalar — device
    "DeviceError",
    "DeviceMismatchError",

    # Hatalar — şekil
    "ShapeError",
    "VocabSizeMismatchError",

    # Hatalar — dağıtık
    "DistributedSetupError",

    # Hatalar — sağlık
    "HealthCheckError",
]

# ── Paket Versiyonu ───────────────────────────────────────────────────────────
__version__ = "4.1.0"
__author__ = "Muhammed Yasin Yılmaz"
