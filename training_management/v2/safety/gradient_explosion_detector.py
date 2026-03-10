# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: gradient_explosion_detector.py
Modül: training_management/v2/safety
Görev: Gradient Explosion Detector - Gradient explosion detection utility.
       Gradient explosion tespiti, gradient istatistikleri raporlama ve
       remediation önerileri sunma işlemlerini yapar. Eğitim sırasında gradient
       sorunlarını erken tespit eder.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (gradient explosion detection)
- Design Patterns: Detector Pattern (sorun tespiti)
- Endüstri Standartları: Gradient stability detection best practices

KULLANIM:
- Gradient explosion tespiti için
- Gradient istatistikleri raporlama için
- Remediation önerileri sunma için

BAĞIMLILIKLAR:
- torch: Gradient işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Optional, Any, Dict
import torch


class GradientExplosionDetector:
    """
    Gradient explosion detection utility.
    
    Responsibilities:
    - Detect gradient explosion
    - Report gradient statistics
    - Suggest remediation
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(
        self,
        threshold: float = 10.0,
        logger: Optional[Any] = None
    ):
        """
        Initialize GradientExplosionDetector.
        
        Args:
            threshold: Gradient norm threshold for explosion detection
            logger: Optional logger instance
        """
        self.threshold = threshold
        self.logger = logger
    
    def detect(
        self,
        model: torch.nn.Module,
        max_grad_norm: float
    ) -> Dict[str, Any]:
        """
        Detect gradient explosion.
        
        Args:
            model: Model with gradients
            max_grad_norm: Maximum expected gradient norm
            
        Returns:
            Dictionary with detection results:
            {
                "has_explosion": bool,
                "max_grad_value": float,
                "total_norm": float,
                "recommendation": str
            }
        """
        max_grad_value = 0.0
        total_norm_sq = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                max_val = grad.abs().max().item()
                max_grad_value = max(max_grad_value, max_val)
                total_norm_sq += grad.norm(2).item() ** 2
        
        total_norm = total_norm_sq ** 0.5
        has_explosion = total_norm > max_grad_norm * self.threshold
        
        recommendation = ""
        if has_explosion:
            recommendation = (
                f"Gradient explosion detected! "
                f"Norm={total_norm:.2f} > threshold={max_grad_norm * self.threshold:.2f}. "
                f"Consider: 1) Lower learning rate, 2) Gradient clipping, 3) Check data"
            )
            if self.logger:
                self.logger.log_warning(recommendation)
        
        return {
            "has_explosion": has_explosion,
            "max_grad_value": max_grad_value,
            "total_norm": total_norm,
            "recommendation": recommendation
        }

