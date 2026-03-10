# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: embedding_adapter.py
Modül: cognitive_management/v2/components
Görev: V2 Embedding Adapter - Embedding model adapter for vector memory system.
       Phase 7.1: Vector Memory Enhancement. Supports multiple embedding providers:
       sentence-transformers (default), OpenAI embeddings API, custom embeddings.
       Akademik referans: Reimers & Gurevych (2019), Cer et al. (2018).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (embedding adaptation),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Adapter Pattern (embedding adapter)
- Endüstri Standartları: Embedding adapter best practices

KULLANIM:
- Embedding generation için
- Multiple provider desteği için
- Vector memory için

BAĞIMLILIKLAR:
- sentence-transformers: Embedding modelleri
- OpenAI API: Embedding API (opsiyonel)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Protocol, List, Optional, Union
from abc import ABC, abstractmethod

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError, MemoryError


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]: ...
    @property
    def dimension(self) -> int: ...


class BaseEmbeddingAdapter(ABC):
    """
    Base class for embedding adapters.
    Provides common interface for different embedding providers.
    """
    
    def __init__(self, cfg: CognitiveManagerConfig):
        """
        Initialize embedding adapter.
        
        Args:
            cfg: Cognitive manager configuration
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        self.cfg = cfg
        self._model: Optional[EmbeddingModel] = None
        self._dimension: Optional[int] = None
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors (list of floats)
        """
        pass
    
    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text into an embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (list of floats)
        """
        if not text or not text.strip():
            raise ValidationError("text boş olamaz.")
        
        embeddings = self.encode([text])
        if not embeddings:
            raise MemoryError("Embedding oluşturulamadı.")
        return embeddings[0]


class SentenceTransformersAdapter(BaseEmbeddingAdapter):
    """
    Sentence Transformers embedding adapter.
    
    Default model: all-MiniLM-L6-v2 (384 dimensions)
    Fast, efficient, multilingual support.
    """
    
    def __init__(self, cfg: CognitiveManagerConfig):
        """
        Initialize Sentence Transformers adapter.
        
        Args:
            cfg: Cognitive manager configuration
        """
        super().__init__(cfg)
        self._load_model()
    
    def _load_model(self) -> None:
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers paketi yüklü değil. "
                "Yüklemek için: pip install sentence-transformers"
            )
        
        model_name = self.cfg.memory.embedding_model or "all-MiniLM-L6-v2"
        
        try:
            self._model = SentenceTransformer(model_name)
            # Get dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            raise MemoryError(f"Embedding model yüklenemedi: {e}") from e
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            raise MemoryError("Model henüz yüklenmedi.")
        return self._dimension
    
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Encode text(s) using Sentence Transformers.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        if self._model is None:
            raise MemoryError("Embedding model yüklenmemiş.")
        
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        # Filter empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            # Encode texts
            embeddings = self._model.encode(
                non_empty_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False,
            )
            
            # Convert numpy array to list of lists
            if embeddings.ndim == 1:
                return [embeddings.tolist()]
            else:
                return embeddings.tolist()
        
        except Exception as e:
            raise MemoryError(f"Embedding oluşturulamadı: {e}") from e


class OpenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    OpenAI embeddings API adapter.
    
    Model: text-embedding-ada-002 (1536 dimensions)
    Requires OpenAI API key.
    """
    
    def __init__(self, cfg: CognitiveManagerConfig):
        """
        Initialize OpenAI embeddings adapter.
        
        Args:
            cfg: Cognitive manager configuration
        """
        super().__init__(cfg)
        
        if not self.cfg.memory.openai_api_key:
            raise ValidationError("OpenAI API key gerekli (memory.openai_api_key).")
        
        self._api_key = self.cfg.memory.openai_api_key
        self._model_name = "text-embedding-ada-002"
        self._dimension = 1536  # text-embedding-ada-002 dimension
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Encode text(s) using OpenAI embeddings API.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai paketi yüklü değil. "
                "Yüklemek için: pip install openai"
            )
        
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        # Filter empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            # Call OpenAI API
            response = openai.Embedding.create(
                model=self._model_name,
                input=non_empty_texts,
                api_key=self._api_key,
            )
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in response["data"]]
            return embeddings
        
        except Exception as e:
            raise MemoryError(f"OpenAI embedding oluşturulamadı: {e}") from e


def create_embedding_adapter(cfg: CognitiveManagerConfig) -> Optional[BaseEmbeddingAdapter]:
    """
    Factory function to create embedding adapter based on configuration.
    
    Args:
        cfg: Cognitive manager configuration
        
    Returns:
        Embedding adapter instance or None if disabled
    """
    if not cfg.memory.enable_vector_memory:
        return None
    
    provider = cfg.memory.embedding_provider.lower()
    
    if provider == "none":
        return None
    elif provider == "sentence-transformers":
        return SentenceTransformersAdapter(cfg)
    elif provider == "openai":
        return OpenAIEmbeddingAdapter(cfg)
    else:
        raise ValueError(f"Geçersiz embedding provider: {provider}")


__all__ = [
    "BaseEmbeddingAdapter",
    "SentenceTransformersAdapter",
    "OpenAIEmbeddingAdapter",
    "create_embedding_adapter",
    "EmbeddingModel",
]

