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
        self.clock = time.perf_counter  # monotonic; tests may substitute a deterministic clock
        self.batch_times: List[float] = []
        self.batch_samples: List[int] = []
        self.batch_tokens: List[int] = []
        self.start_time: Optional[float] = None
    
    def start_batch(self) -> None:
        """Mark batch start time."""
        if self.enabled:
            self.start_time = self.clock()
    
    def end_batch(self, batch_size: int, seq_len: int, tokens: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        Mark batch end time and calculate metrics.
        
        Each start_batch() is closed by exactly one end_batch(): a second end_batch()
        without a new start_batch() records nothing and returns None.

        Args:
            batch_size: Batch size
            seq_len: Sequence length (padded)
            tokens: Actual (non-pad) target tokens in the batch; batch_size * seq_len if None
            
        Returns:
            Dictionary with performance metrics or None if disabled
        """
        if not self.enabled or self.start_time is None:
            return None
        
        elapsed = self.clock() - self.start_time
        self.start_time = None
        n_tokens = int(tokens) if tokens is not None else int(batch_size) * int(seq_len)
        self.batch_times.append(elapsed)
        self.batch_samples.append(int(batch_size))
        self.batch_tokens.append(n_tokens)
        
        stats = {
            "batch_time_sec": elapsed,
            "samples_per_sec": batch_size / elapsed if elapsed > 0 else 0.0,
            "tokens_per_sec": n_tokens / elapsed if elapsed > 0 else 0.0,
            "tokens": n_tokens,
        }
        
        return stats

    def summary(self, start: int = 0, end: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        Aggregate the batches recorded in [start:end] (e.g. one epoch's training batches).

        Throughput is total samples / tokens over total measured time, so batches of
        different lengths are weighted by what they actually contained.

        Returns:
            Dictionary of totals and rates, or None if no batch was recorded in the range
        """
        times = self.batch_times[start:end]
        if not times:
            return None
        total_time = sum(times)
        samples = sum(self.batch_samples[start:end])
        tokens = sum(self.batch_tokens[start:end])
        return {
            "batches": len(times),
            "time_sec": total_time,
            "avg_batch_time_sec": total_time / len(times),
            "samples": samples,
            "tokens": tokens,
            "samples_per_sec": samples / total_time if total_time > 0 else 0.0,
            "tokens_per_sec": tokens / total_time if total_time > 0 else 0.0,
        }

