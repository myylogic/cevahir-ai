# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: bpe_validator.py
Modül: training_system/v2/core
Görev: BPE Validator - BPE Dosya Validasyonu. BPE vocab ve merges dosyalarının
       varlığını kontrol etme. Sabit vocab stratejisi: Vocab sadece train_bpe.py
       ile oluşturulmalı. BPE dosya validasyonu ve kontrolü.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (BPE dosya validasyonu)
- Design Patterns: Validator Pattern (dosya validasyonu)
- Endüstri Standartları: BPE file validation

KULLANIM:
- BPE dosyalarının varlığını kontrol etmek için
- Vocab ve merges dosyalarını doğrulamak için
- Eğitim öncesi validasyon için

BAĞIMLILIKLAR:
- os: Dosya sistemi işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import os
from typing import Optional, Any


class BPEValidator:
    """BPE dosyalarının varlığını kontrol eden validator"""
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Args:
            logger: Logger instance (opsiyonel)
        """
        self.logger = logger
    
    def validate_files(
        self,
        vocab_path: str,
        merges_path: str
    ) -> None:
        """
        BPE dosyalarının var olduğunu kontrol et.
        Yoksa RuntimeError fırlatır.
        
        Args:
            vocab_path: Vocab dosyası yolu
            merges_path: Merges dosyası yolu
            
        Raises:
            RuntimeError: Dosyalar bulunamazsa
        """
        def file_exists_and_nonempty(path: str) -> bool:
            """Dosyanın var ve boş olmadığını kontrol et"""
            return os.path.exists(path) and os.path.getsize(path) > 0
        
        vocab_exists = file_exists_and_nonempty(vocab_path)
        merges_exists = file_exists_and_nonempty(merges_path)
        
        if not vocab_exists or not merges_exists:
            missing_files = []
            if not vocab_exists:
                missing_files.append(f"Vocab: {vocab_path}")
            if not merges_exists:
                missing_files.append(f"Merges: {merges_path}")
            
            error_msg = (
                f"BPE dosyaları bulunamadı!\n"
                f"{chr(10).join(missing_files)}\n"
                f"\n⚠️ SABİT VOCAB STRATEJİSİ:\n"
                f"Vocab ve merges sadece 'train_bpe.py' ile oluşturulmalı.\n"
                f"Lütfen önce 'python tokenizer_management/train_bpe.py' çalıştırın.\n"
                f"TrainingService'te rebuild yapılmaz (sabit vocab stratejisi)."
            )
            
            if self.logger:
                self.logger.error(error_msg)
            
            raise RuntimeError(error_msg)
        
        if self.logger:
            self.logger.info(
                f"[OK] BPE dosyaları doğrulandı: vocab={vocab_path}, merges={merges_path}"
            )

