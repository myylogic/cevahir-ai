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
import torch.nn.functional as F
import logging


class RMSNorm(nn.Module):
    """
    [V5] RMSNorm (Root Mean Square Layer Normalization)
    Endüstri standardı: GPT-3+, LLaMA 2, Mistral, Gemma

    Formula:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale

    Numerik Stabilite (FP16/BF16):
        Norm hesaplaması daima float32'de yapılır; overflow/NaN önlenir.
        Sonuç, girişin orijinal dtype'ına geri cast edilir.
        Referans: LLaMA 2 / Mistral implementasyonu.

    Performans (PyTorch 2.4+):
        torch.nn.functional.rms_norm fused kernel varsa kullanılır.
        Yoksa manuel float32-upcast implementasyona fallback yapılır.

    Args:
        dim: Normalize edilecek son boyut.
        eps: Sayısal stabilite için epsilon (default: 1e-6).
    """

    # PyTorch 2.4+ fused kernel kontrolü — sınıf yüklendiğinde bir kez yapılır
    _USE_FUSED: bool = hasattr(F, "rms_norm")
    
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
        
        self.logger.info(
            f"[V5] RMSNorm initialized: dim={dim}, eps={eps}, "
            f"fused_kernel={self._USE_FUSED}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [..., dim] — Giriş tensörü (son boyut normalize edilir).

        Returns:
            [..., dim] — Normalize edilmiş tensör, girişle aynı dtype.

        Akış:
            1. PyTorch 2.4+: F.rms_norm fused kernel (en hızlı yol).
            2. Diğer: float32 upcast → rsqrt → orijinal dtype'a cast.
               FP16/BF16'da x*x overflow riskini ortadan kaldırır.
        """
        # --- Path 1: PyTorch 2.4+ fused kernel ---
        # scale, input dtype'ına cast edilir (FP16/BF16 uyumluluğu için)
        if self._USE_FUSED:
            return F.rms_norm(x, (self.dim,), self.scale.to(x.dtype), self.eps)

        # --- Path 2: Float32 upcast (FP16/BF16 overflow koruması) ---
        # LLaMA 2 / Mistral standardı:
        # Norm hesaplaması float32'de yapılır → NaN/inf riski ortadan kalkar.
        # Sonuç girişin orijinal dtype'ına geri döndürülür.
        orig_dtype = x.dtype
        x_fp32 = x.float()                                          # FP16/BF16 → FP32

        # mean(x^2) — x*x, x**2'den hafif daha hızlı (pow dispatch yok)
        rms = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)    # [..., 1]
        x_normed = x_fp32 * torch.rsqrt(rms + self.eps)            # [..., dim]

        # Scale FP32'ye cast edilir, sonuç orijinal dtype'a döndürülür
        return (x_normed * self.scale.float()).to(orig_dtype)
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

