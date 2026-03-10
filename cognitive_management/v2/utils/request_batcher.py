# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: request_batcher.py
Modül: cognitive_management/v2/utils
Görev: Request Batcher - Request batching for efficient batch processing.
       Phase 6: Performance Optimization & Caching Enhancement. BatchRequest,
       RequestBatcher sınıflarını içerir. Request batching, batch processing,
       batch optimization ve efficient batch execution işlemlerini yapar.
       Akademik referans: Batching strategies for LLM inference, Batch processing
       optimization.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (request batching)
- Design Patterns: Batcher Pattern (request batching)
- Endüstri Standartları: Request batching best practices

KULLANIM:
- Request batching için
- Batch processing için
- Batch optimization için

BAĞIMLILIKLAR:
- threading: Thread-safe işlemler
- typing: Generic type hints

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import time
from collections import deque

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchRequest:
    """
    Individual request in a batch.
    
    Attributes:
        request_id: Unique request identifier
        data: Request data
        callback: Callback function to call with result
        timestamp: When request was added
    """
    request_id: str
    data: T
    callback: Callable[[R], None]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RequestBatcher(Generic[T, R]):
    """
    Request batcher for batch processing.
    
    Batches multiple requests and processes them together for efficiency.
    
    Features:
    - Configurable batch size
    - Time-based batching (wait time)
    - Automatic batch processing
    - Thread-safe
    """
    
    def __init__(
        self,
        batch_processor: Callable[[List[T]], List[R]],
        max_batch_size: int = 10,
        max_wait_time: float = 0.1,  # 100ms
        enable_auto_flush: bool = True,
    ):
        """
        Initialize request batcher.
        
        Args:
            batch_processor: Function to process batch of requests (takes List[T], returns List[R])
            max_batch_size: Maximum batch size before auto-flush
            max_wait_time: Maximum wait time before auto-flush (seconds)
            enable_auto_flush: Enable automatic batch flushing
        """
        self.batch_processor = batch_processor
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.enable_auto_flush = enable_auto_flush
        
        # Batch storage
        self._current_batch: List[BatchRequest] = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_flush = False
        
        if self.enable_auto_flush:
            self._start_flush_thread()
        
        # Statistics
        self._total_batches = 0
        self._total_requests = 0
    
    def add_request(
        self,
        request_id: str,
        data: T,
        callback: Callable[[R], None]
    ) -> None:
        """
        Add request to batch.
        
        Args:
            request_id: Unique request identifier
            data: Request data
            callback: Callback function to call with result
        """
        with self._condition:
            request = BatchRequest(
                request_id=request_id,
                data=data,
                callback=callback,
            )
            
            self._current_batch.append(request)
            self._total_requests += 1
            
            # Check if should flush
            if len(self._current_batch) >= self.max_batch_size:
                self._flush_batch_internal()
            
            # Notify flush thread
            self._condition.notify()
    
    def flush(self) -> None:
        """Manually flush current batch."""
        with self._condition:
            self._flush_batch_internal()
    
    def _flush_batch_internal(self) -> None:
        """Internal batch flush (must be called with lock held)."""
        if not self._current_batch:
            return
        
        # Get current batch
        batch_to_process = self._current_batch.copy()
        self._current_batch.clear()
        
        # Release lock before processing (batch_processor might take time)
        self._lock.release()
        try:
            # Process batch
            batch_data = [req.data for req in batch_to_process]
            batch_results = self.batch_processor(batch_data)
            
            # Map results back to requests
            for i, request in enumerate(batch_to_process):
                if i < len(batch_results):
                    result = batch_results[i]
                    try:
                        request.callback(result)
                    except Exception as e:
                        # Callback error shouldn't fail batch
                        import logging
                        logging.error(f"Request batcher callback error: {e}")
            
            self._total_batches += 1
            
        finally:
            self._lock.acquire()
    
    def _start_flush_thread(self) -> None:
        """Start background thread for time-based flushing."""
        def flush_worker():
            while not self._stop_flush:
                with self._condition:
                    # Wait for batch or timeout
                    self._condition.wait(timeout=self.max_wait_time)
                    
                    # Check if should flush
                    if self._current_batch:
                        # Check if oldest request is older than max_wait_time
                        oldest_timestamp = min(req.timestamp for req in self._current_batch)
                        elapsed = time.time() - oldest_timestamp
                        
                        if elapsed >= self.max_wait_time:
                            self._flush_batch_internal()
        
        self._flush_thread = threading.Thread(target=flush_worker, daemon=True)
        self._stop_flush = False
        self._flush_thread.start()
    
    def stop(self) -> None:
        """Stop batcher and flush remaining requests."""
        with self._lock:
            self._stop_flush = True
            # Flush remaining batch
            if self._current_batch:
                self._flush_batch_internal()
        
        # Wait for flush thread
        if self._flush_thread:
            self._flush_thread.join(timeout=1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        with self._lock:
            return {
                "current_batch_size": len(self._current_batch),
                "total_batches": self._total_batches,
                "total_requests": self._total_requests,
                "avg_batch_size": (
                    self._total_requests / self._total_batches
                    if self._total_batches > 0 else 0.0
                ),
                "max_batch_size": self.max_batch_size,
                "max_wait_time": self.max_wait_time,
            }


__all__ = [
    "RequestBatcher",
    "BatchRequest",
]

