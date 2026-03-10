# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: metrics_tracker.py
Modül: training_management/v2/metrics
Görev: Metrics Tracker - Metrik takibi ve history yönetimi. Training/validation
       metriklerini takip eder, history tutar ve metrikleri export eder.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (metrik takibi)
- Design Patterns: Tracker Pattern (metrik takibi)
- Endüstri Standartları: Metrics tracking best practices

KULLANIM:
- Training/validation metriklerini takip etmek için
- Metrik history tutmak için
- Metrikleri export etmek için

BAĞIMLILIKLAR:
- Modül içi bağımlılıklar

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Dict, List, Optional, Any


class MetricsTracker:
    """
    Metrics tracking utility.
    
    Responsibilities:
    - Track training/validation metrics
    - Maintain history
    - Export metrics
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize MetricsTracker.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
        }
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
    ) -> None:
        """
        Update metrics history.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            accuracy: Accuracy
        """
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["accuracy"].append(accuracy)
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get metrics history (deep copy).
        
        Returns:
            Dictionary with metrics history (deep copy)
        """
        # Deep copy to prevent external modifications
        return {
            key: value.copy() for key, value in self.history.items()
        }

