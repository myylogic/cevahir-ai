# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: backend_adapter.py
Modül: cognitive_management/v2/adapters
Görev: Backend Adapter - V1 ModelAPI'yi V2 FullModelBackend'e adapte eder.
       Adapter Pattern kullanarak backward compatibility sağlar. V1 ve V2
       arasında köprü görevi görür.

MİMARİ:
- SOLID Prensipleri: Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Adapter Pattern (backend adaptation)
- Endüstri Standartları: Backward compatibility best practices

KULLANIM:
- V1 ModelAPI'yi V2'ye adapte etmek için
- Backward compatibility için
- Interface adaptation için

BAĞIMLILIKLAR:
- BackendProtocols: Backend interface'leri
- CognitiveTypes: Tip tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from cognitive_management.research.runtime import BudgetExhausted, request_model_api
from typing import Optional, Protocol
import math

# V1'den import
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.cognitive_types import DecodingConfig
from cognitive_management.v2.interfaces.backend_protocols import (
    FullModelBackend,
    TextGenerationBackend,
    ScoringBackend,
    EntropyBackend,
)


class ModelAPIAdapter(FullModelBackend):
    """
    V1 ModelAPI'yi V2 FullModelBackend'e adapte eder.
    Adapter Pattern kullanır.
    
    Phase 8: Enhanced with ConnectionPool for scalability.
    """
    
    def __init__(self, v1_model_api, cfg=None, connection_pool=None):
        """
        Initialize adapter.
        
        Args:
            v1_model_api: V1 ModelAPI instance
            cfg: Optional config for connection pooling
            connection_pool: Optional ConnectionPool instance
        """
        self._v1_api = v1_model_api
        self._cfg = cfg
        
        # Phase 8: ConnectionPool integration (optional)
        self._connection_pool = connection_pool
        if connection_pool is None and cfg and getattr(cfg.runtime, 'enable_connection_pool', False):
            try:
                from ..utils.connection_pool import ConnectionPool
                # Create connection pool factory
                def create_connection():
                    # For ModelAPI, we'll pool the API calls themselves
                    # This is a simple implementation - in production, you'd pool actual connections
                    return self._v1_api
                
                self._connection_pool = ConnectionPool(
                    factory=create_connection,
                    max_size=getattr(cfg.runtime, 'connection_pool_size', 10),
                    min_size=2,
                )
            except BudgetExhausted:
                raise
            except Exception as e:
                import logging
                logging.warning(f"ConnectionPool initialization başarısız: {e}")
                self._connection_pool = None
    
    # TextGenerationBackend
    def generate(
        self,
        prompt: str,
        decoding_config: DecodingConfig,
        **kwargs
    ) -> str:
        """Generate text"""
        # Phase 8: Use connection pool if available
        if self._connection_pool:
            try:
                conn = self._connection_pool.acquire()
                try:
                    return request_model_api(conn).generate(prompt, decoding_config, **kwargs)
                finally:
                    self._connection_pool.release(conn)
            except BudgetExhausted:
                raise
            except Exception:
                # Fallback to direct call if pool fails
                pass
        
        # Direct call (no pooling or pool disabled)
        return request_model_api(self._v1_api).generate(prompt, decoding_config, **kwargs)
    
    # ScoringBackend
    def score(
        self,
        prompt: str,
        candidate: str,
        **kwargs
    ) -> float:
        """Score candidate"""
        # Phase 8: Use connection pool if available
        if self._connection_pool:
            try:
                conn = self._connection_pool.acquire()
                try:
                    return request_model_api(conn).score(prompt, candidate, **kwargs)
                finally:
                    self._connection_pool.release(conn)
            except BudgetExhausted:
                raise
            except Exception:
                # Fallback to direct call if pool fails
                pass
        return request_model_api(self._v1_api).score(prompt, candidate, **kwargs)
    
    # EntropyBackend
    def estimate_entropy(
        self,
        text: str,
        **kwargs
    ) -> Optional[float]:
        """Normalized next-token entropy in [0, 1], or None when unavailable.

        None means there is no measurement: the backend has no probe, the probe
        failed, or ``entropy_details`` reported ``available=False`` (the bare
        ``entropy_estimate`` then returns a neutral 0.5 placeholder, which must
        not be routed on). Inside a research request errors still propagate so
        the budget ledger records the failed probe.
        """
        # Phase 8: Use connection pool if available
        if self._connection_pool:
            try:
                conn = self._connection_pool.acquire()
                try:
                    return self._measure_entropy(conn, text, **kwargs)
                finally:
                    self._connection_pool.release(conn)
            except BudgetExhausted:
                raise
            except Exception:
                # Fallback to direct call if pool fails
                pass
        try:
            return self._measure_entropy(self._v1_api, text, **kwargs)
        except BudgetExhausted:
            raise
        except Exception:
            from cognitive_management.research.runtime import current_runtime
            if current_runtime() is not None:
                raise
            return None

    @staticmethod
    def _measure_entropy(raw, text: str, **kwargs) -> Optional[float]:
        from cognitive_management.research.runtime import current_runtime
        if current_runtime() is not None:
            # The budgeted proxy accounts the probe and rejects unavailable ones.
            value = request_model_api(raw).entropy_estimate(text, **kwargs)
        else:
            details = getattr(raw, "entropy_details", None)
            if callable(details):
                result = details(text, **kwargs)
                if not result.get("available", False):
                    return None
                value = result.get("value")
            else:
                value = raw.entropy_estimate(text, **kwargs)
        if value is None or isinstance(value, bool):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    
    # Optional: Multimodal support
    def process_audio(self, audio_data: bytes) -> str:
        """Process audio (if supported)"""
        # Phase 8: Use connection pool if available
        if self._connection_pool:
            try:
                conn = self._connection_pool.acquire()
                try:
                    if hasattr(conn, 'process_audio'):
                        return conn.process_audio(audio_data)
                finally:
                    self._connection_pool.release(conn)
            except BudgetExhausted:
                raise
            except Exception:
                # Fallback to direct call if pool fails
                pass
        if hasattr(self._v1_api, 'process_audio'):
            return self._v1_api.process_audio(audio_data)
        raise NotImplementedError("Audio processing not supported")
    
    def process_image(self, image_data: bytes) -> str:
        """Process image (if supported)"""
        # Phase 8: Use connection pool if available
        if self._connection_pool:
            try:
                conn = self._connection_pool.acquire()
                try:
                    if hasattr(conn, 'process_image'):
                        return conn.process_image(image_data)
                finally:
                    self._connection_pool.release(conn)
            except BudgetExhausted:
                raise
            except Exception:
                # Fallback to direct call if pool fails
                pass
        if hasattr(self._v1_api, 'process_image'):
            return self._v1_api.process_image(image_data)
        raise NotImplementedError("Image processing not supported")
    
    def process_multimodal(
        self,
        text: str = None,
        audio: bytes = None,
        image: bytes = None
    ) -> str:
        """Process multimodal input (if supported)"""
        # Phase 8: Use connection pool if available
        if self._connection_pool:
            try:
                conn = self._connection_pool.acquire()
                try:
                    if hasattr(conn, 'process_multimodal'):
                        return conn.process_multimodal(text=text, audio=audio, image=image)
                finally:
                    self._connection_pool.release(conn)
            except BudgetExhausted:
                raise
            except Exception:
                # Fallback to direct call if pool fails
                pass
        if hasattr(self._v1_api, 'process_multimodal'):
            return self._v1_api.process_multimodal(text=text, audio=audio, image=image)
        raise NotImplementedError("Multimodal processing not supported")


__all__ = ["ModelAPIAdapter"]

