# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: backend_protocols.py
Modül: cognitive_management/v2/interfaces
Görev: Backend Protocol Definitions - Model backend için interface tanımları.
       SOLID: Interface Segregation Principle (ISP) uygulanmış. FullModelBackend,
       TextGenerationBackend, ScoringBackend, EntropyBackend ve KVCache interface
       tanımlarını içerir.

MİMARİ:
- SOLID Prensipleri: Interface Segregation Principle (ISP)
- Design Patterns: Protocol Pattern (backend protocols)
- Endüstri Standartları: Interface design best practices

KULLANIM:
- Backend interface tanımları için
- Model backend protocols için
- Interface segregation için

BAĞIMLILIKLAR:
- typing.Protocol: Protocol tanımları
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
from typing import Protocol, Iterator, Optional, Tuple
from dataclasses import dataclass

# V1'den import (backward compatibility)
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.cognitive_types import DecodingConfig


# KV Cache için type hint
@dataclass
class KVCache:
    """KV Cache data structure"""
    keys: Optional[any] = None
    values: Optional[any] = None
    position: Optional[int] = None


# =============================================================================
# Segregated Interfaces (ISP - Interface Segregation Principle)
# =============================================================================

class TextGenerationBackend(Protocol):
    """
    Text generation interface.
    Sadece text generation için gerekli metodlar.
    """
    def generate(
        self,
        prompt: str,
        decoding_config: DecodingConfig
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            decoding_config: Decoding configuration
            
        Returns:
            Generated text
        """
        ...


class StreamingBackend(Protocol):
    """
    Streaming generation interface.
    Token-by-token generation için.
    """
    def generate_streaming(
        self,
        prompt: str,
        decoding_config: DecodingConfig
    ) -> Iterator[str]:
        """
        Generate text token by token (streaming).
        
        Args:
            prompt: Input text prompt
            decoding_config: Decoding configuration
            
        Yields:
            Text tokens one by one
        """
        ...


class CachedBackend(Protocol):
    """
    KV cache support interface.
    Efficient inference için KV cache kullanımı.
    """
    def generate_with_cache(
        self,
        prompt: str,
        cache: Optional[KVCache],
        decoding_config: DecodingConfig
    ) -> Tuple[str, KVCache]:
        """
        Generate with KV cache for efficient inference.
        
        Args:
            prompt: Input text prompt
            cache: Previous KV cache (None for first call)
            decoding_config: Decoding configuration
            
        Returns:
            Tuple of (generated_text, new_cache)
        """
        ...


class ScoringBackend(Protocol):
    """
    Scoring interface.
    Candidate text scoring için.
    """
    def score(
        self,
        prompt: str,
        candidate: str
    ) -> float:
        """
        Score candidate text.
        
        Args:
            prompt: Input prompt
            candidate: Candidate text to score
            
        Returns:
            Score (higher is better)
        """
        ...


class EntropyBackend(Protocol):
    """
    Entropy estimation interface.
    Text uncertainty estimation için.
    """
    def estimate_entropy(
        self,
        text: str
    ) -> float:
        """
        Estimate text entropy (uncertainty).
        
        Args:
            text: Input text
            
        Returns:
            Entropy value (0.0 - 3.0 typically)
        """
        ...


class MultimodalBackend(Protocol):
    """
    Multimodal processing interface.
    Audio, image, video processing için.
    """
    def process_audio(
        self,
        audio_data: bytes
    ) -> str:
        """
        Process audio input.
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            Processed text
        """
        ...
    
    def process_image(
        self,
        image_data: bytes
    ) -> str:
        """
        Process image input.
        
        Args:
            image_data: Image bytes
            
        Returns:
            Processed text
        """
        ...
    
    def process_multimodal(
        self,
        text: Optional[str] = None,
        audio: Optional[bytes] = None,
        image: Optional[bytes] = None
    ) -> str:
        """
        Process multimodal input.
        
        Args:
            text: Optional text input
            audio: Optional audio bytes
            image: Optional image bytes
            
        Returns:
            Processed text
        """
        ...


# =============================================================================
# Composite Interface (Full-Featured Backend)
# =============================================================================

class FullModelBackend(
    TextGenerationBackend,
    StreamingBackend,
    CachedBackend,
    ScoringBackend,
    EntropyBackend,
    MultimodalBackend,
    Protocol
):
    """
    Full-featured model backend.
    Tüm özellikleri destekleyen backend interface'i.
    
    Not: Bu interface tüm metodları içerir, ancak implementasyon
    sadece desteklenen metodları implement edebilir.
    """
    pass


__all__ = [
    "KVCache",
    "TextGenerationBackend",
    "StreamingBackend",
    "CachedBackend",
    "ScoringBackend",
    "EntropyBackend",
    "MultimodalBackend",
    "FullModelBackend",
]

