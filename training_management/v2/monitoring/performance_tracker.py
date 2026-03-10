# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: performance_tracker.py
Modül: training_management/v2/monitoring
Görev: Performance Tracker - Performans metrik takibi. Batch processing time,
       samples/sec, tokens/sec takibi ve performans history yönetimi işlemlerini
       yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (performans takibi)
- Design Patterns: Tracker Pattern (performans takibi)
- Endüstri Standartları: Performance tracking best practices

KULLANIM:
- Batch processing time takibi için
- Samples/sec, tokens/sec takibi için
- Performans history yönetimi için

BAĞIMLILIKLAR:
- time: Zaman ölçümü

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Optional, Any, Dict, List
import time


class PerformanceTracker:
    """
    Performance metrics tracker.
    
    Responsibilities:
    - Track batch processing time
    - Track samples/sec, tokens/sec
    - Maintain performance history
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(
        self,
        enabled: bool = True,
        logger: Optional[Any] = None
    ):
        """
        Initialize PerformanceTracker.
        
        Args:
            enabled: Whether performance tracking is enabled
            logger: Optional logger instance
        """
        self.enabled = enabled
        self.logger = logger
        self.batch_times: List[float] = []
        self.start_time: Optional[float] = None
    
    def start_batch(self) -> None:
        """Mark batch start time."""
        if self.enabled:
            self.start_time = time.time()
    
    def end_batch(self, batch_size: int, seq_len: int) -> Optional[Dict[str, float]]:
        """
        Mark batch end time and calculate metrics.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Dictionary with performance metrics or None if disabled
        """
        if not self.enabled or self.start_time is None:
            return None
        
        elapsed = time.time() - self.start_time
        self.batch_times.append(elapsed)
        
        stats = {
            "batch_time_sec": elapsed,
            "samples_per_sec": batch_size / elapsed if elapsed > 0 else 0.0,
            "tokens_per_sec": (batch_size * seq_len) / elapsed if elapsed > 0 else 0.0,
        }
        
        return stats

