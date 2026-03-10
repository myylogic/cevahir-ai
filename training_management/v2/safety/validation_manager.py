# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: validation_manager.py
Modül: training_management/v2/safety
Görev: Validation Manager - Input/output validation utility. Model input/output
       validasyonu, shape ve type validasyonu işlemlerini yapar. Eğitim sırasında
       veri bütünlüğünü sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (validasyon)
- Design Patterns: Manager Pattern (validasyon yönetimi)
- Endüstri Standartları: Input/output validation best practices

KULLANIM:
- Model input validasyonu için
- Model output validasyonu için
- Shape ve type validasyonu için

BAĞIMLILIKLAR:
- torch: Tensor işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Optional, Any
import torch


class ValidationManager:
    """
    Validation utility.
    
    Responsibilities:
    - Validate model inputs
    - Validate model outputs
    - Validate shapes and types
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize ValidationManager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
    
    def validate_logits_shape(
        self,
        logits: torch.Tensor,
        expected_shape: tuple,
        vocab_size: int
    ) -> bool:
        """
        Validate logits shape.
        
        Args:
            logits: Logits tensor
            expected_shape: Expected shape (B, T)
            vocab_size: Vocabulary size
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Raises:
            ValueError: If shape is invalid
        """
        if logits.dim() != 3:
            raise ValueError(
                f"Logits must be 3D [B, T, V], got {logits.dim()}D"
            )
        
        B, T, V = logits.shape
        
        if B != expected_shape[0]:
            raise ValueError(
                f"Batch size mismatch: expected {expected_shape[0]}, got {B}"
            )
        
        if T != expected_shape[1]:
            raise ValueError(
                f"Sequence length mismatch: expected {expected_shape[1]}, got {T}"
            )
        
        if V != vocab_size:
            raise ValueError(
                f"Vocab size mismatch: expected {vocab_size}, got {V}"
            )
        
        return True

