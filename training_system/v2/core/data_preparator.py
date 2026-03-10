# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: data_preparator.py
Modül: training_system/v2/core

⚠️ DEPRECATED: Bu dosya artık kullanılmıyor!

prepare_from_cache metodu training_service.py içine taşındı.
Helper metodlar (_apply_autoregressive_formatting, _split_train_val, vb.)
artık kullanılmıyor ve iş akışını bozuyordu.

YENİ KULLANIM:
- prepare_from_cache metodu training_service.py içinde
- Formatlama prepare_cache.py'de yapılıyor
- Helper metodlar kaldırıldı

Bu dosya geriye dönük uyumluluk için bırakıldı ama kullanılmamalı.

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import warnings

warnings.warn(
    "data_preparator.py DEPRECATED: prepare_from_cache metodu training_service.py içine taşındı. "
    "Bu dosyayı kullanmayın!",
    DeprecationWarning,
    stacklevel=2
)

# Geriye dönük uyumluluk için boş class (import hatalarını önlemek için)
class DataPreparator:
    """
    ⚠️ DEPRECATED: Bu class artık kullanılmıyor!
    
    prepare_from_cache metodu training_service.py içine taşındı.
    """
    def __init__(self, logger=None):
        warnings.warn(
            "DataPreparator DEPRECATED: prepare_from_cache metodu training_service.py içine taşındı.",
            DeprecationWarning,
            stacklevel=2
        )
        self.logger = logger
    
    def prepare_from_cache(self, *args, **kwargs):
        raise RuntimeError(
            "DataPreparator.prepare_from_cache DEPRECATED: "
            "Bu metod training_service.py içine taşındı. TrainingService.prepare_from_cache kullanın."
        )
