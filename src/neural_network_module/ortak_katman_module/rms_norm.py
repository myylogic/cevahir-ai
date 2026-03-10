# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: rms_norm.py
Modül: src/neural_network_module/ortak_katman_module
Görev: RMSNorm (Root Mean Square Layer Normalization) - LayerNorm'un daha hızlı
       ve daha stabil bir alternatifi. GPT-3+, LLaMA, PaLM standardı. Mean
       hesaplaması yok, variance hesaplaması yok, daha hızlı ve stabil. Referans:
       Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (RMS normalization işlemleri),
                     Open/Closed (genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Normalization Pattern (RMS normalization)
- Endüstri Standartları: GPT-3+, LLaMA, PaLM RMSNorm standardı

KULLANIM:
- RMS normalization için
- Layer normalization alternatifi için
- Hızlı ve stabil normalizasyon için

BAĞIMLILIKLAR:
- torch.nn: Module base class

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import torch
import torch.nn as nn
import logging


class RMSNorm(nn.Module):
    """
    [OK] V4: RMSNorm (Root Mean Square Layer Normalization)
    Endüstri standardı: GPT-3+, LLaMA, PaLM
    
    Formula:
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g
    
    Args:
        dim: Normalize edilecek dimension
        eps: Numerical stability için epsilon (default: 1e-6)
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        log_level: int = logging.INFO,
    ):
        super().__init__()
        
        # [OK] ENDÜSTRİ STANDARDI: Parametre validasyonu
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim ({dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError(f"eps ({eps}) pozitif bir sayı olmalıdır.")
        
        self.dim = dim
        self.eps = float(eps)
        
        # Scale parameter (learnable)
        self.scale = nn.Parameter(torch.ones(dim))
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"[V4] RMSNorm initialized: dim={dim}, eps={eps}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [..., dim] - Input tensor (son dimension normalize edilir)
        
        Returns:
            x: [..., dim] - Normalized tensor
        
        Formula:
            RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale
        """
        # [OK] V4: RMSNorm hesaplama (GPT-3+, LLaMA standardı)
        # mean(x^2) hesapla (son dimension üzerinde)
        x_norm = torch.mean(x ** 2, dim=-1, keepdim=True)  # [..., 1]
        
        # sqrt(mean(x^2) + eps)
        x_norm = torch.rsqrt(x_norm + self.eps)  # [..., 1] - rsqrt = 1/sqrt (daha hızlı)
        
        # x / sqrt(mean(x^2) + eps) * scale
        x = x * x_norm * self.scale  # [..., dim]
        
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

