"""
================================================================================
CEVAHIR-AI PROJESİ — Training System V3
================================================================================

training_system/v3 — Gelişmiş Eğitim Sistemi

Kritik değişiklikler (V2 → V3):
    1. CACHE ZORUNLU: Cache olmadan eğitim başlamaz (CacheNotFoundError)
    2. GPU BATCHING: BucketBatchSampler + DynamicPaddingCollator
    3. CONFIG V3: 55+ parametre (entropy, focal, EMA, SAM, curriculum, ...)
    4. SOURCE-ID SPLIT: Data leakage önleme
    5. CHECKSUM: Cache bütünlük doğrulama

Zorunlu eğitim akışı:
    1. python tokenizer_management/train_bpe.py    [BPE eğitimi]
    2. python training_system/prepare_cache.py     [Cache hazırlama]
    3. python training_system/train.py             [Model eğitimi]

================================================================================
"""

from .core.training_service_v3 import TrainingServiceV3
from .core.config_manager_v3 import ConfigManagerV3
from .data.cache_v3 import DataCacheV3, CacheNotFoundError, CacheIntegrityError
from .data.dataset_v3 import CevahirDataset
from .data.sampler_v3 import BucketBatchSampler
from .data.collator_v3 import DynamicPaddingCollator
from .data.dataloader_v3 import create_dataloaders_v3, create_dataloader_v3

__all__ = [
    # Servis
    "TrainingServiceV3",
    "ConfigManagerV3",

    # Cache
    "DataCacheV3",
    "CacheNotFoundError",
    "CacheIntegrityError",

    # Veri
    "CevahirDataset",
    "BucketBatchSampler",
    "DynamicPaddingCollator",
    "create_dataloaders_v3",
    "create_dataloader_v3",
]


def get_version() -> str:
    return "3.0.0"


def check_availability() -> dict:
    """V3 bileşen durumunu kontrol et."""
    status = {}

    try:
        from .core.training_service_v3 import TrainingServiceV3
        status["TrainingServiceV3"] = True
    except Exception as e:
        status["TrainingServiceV3"] = False

    try:
        from .data.cache_v3 import DataCacheV3
        status["DataCacheV3"] = True
    except Exception as e:
        status["DataCacheV3"] = False

    try:
        from .data.sampler_v3 import BucketBatchSampler
        status["BucketBatchSampler"] = True
    except Exception as e:
        status["BucketBatchSampler"] = False

    try:
        from .data.collator_v3 import DynamicPaddingCollator
        status["DynamicPaddingCollator"] = True
    except Exception as e:
        status["DynamicPaddingCollator"] = False

    try:
        from .data.dataloader_v3 import create_dataloaders_v3
        status["DataLoaderV3"] = True
    except Exception as e:
        status["DataLoaderV3"] = False

    try:
        from .core.config_manager_v3 import ConfigManagerV3
        status["ConfigManagerV3"] = True
    except Exception as e:
        status["ConfigManagerV3"] = False

    ok_count = sum(1 for v in status.values() if v)
    total = len(status)
    print(f"\n[Training System V3] Bileşen durumu: {ok_count}/{total}")
    for name, ok in status.items():
        icon = "[OK]" if ok else "[HATA]"
        print(f"  {icon} {name}")

    return status
