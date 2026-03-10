# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: base.py
Modül: cognitive_management/v2/components/vector_store
Görev: Vector Store Base Interface - Base interface and abstract implementation
       for vector stores. Phase 7.1: Vector Memory Enhancement. VectorStoreResult,
       VectorStore ve BaseVectorStore interface tanımlarını içerir. Vector storage
       ve retrieval işlemleri için temel interface sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (vector store interface),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Interface Pattern (vector store interface)
- Endüstri Standartları: Vector store best practices

KULLANIM:
- Vector store interface tanımları için
- Base vector store implementation için
- Vector store result için

BAĞIMLILIKLAR:
- abc: Abstract base classes
- dataclasses: Dataclass tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class VectorStoreResult:
    """
    Result from vector store search.
    
    Attributes:
        content: Retrieved text content
        metadata: Additional metadata (role, timestamp, etc.)
        score: Similarity score (0.0-1.0, higher is better)
        id: Unique identifier for the stored item
    """
    content: str
    metadata: Dict[str, Any]
    score: float
    id: Optional[str] = None


class VectorStore(Protocol):
    """
    Protocol for vector store implementations.
    
    Vector stores provide semantic search capabilities for memory retrieval.
    """
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension required by this vector store."""
        ...
    
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add texts with embeddings to vector store.
        
        Args:
            texts: List of texts to store
            embeddings: List of embedding vectors (one per text)
            metadata: List of metadata dictionaries (one per text)
            ids: Optional list of unique IDs (if None, auto-generated)
        """
        ...
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorStoreResult]:
        """
        Search for similar texts using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0.0-1.0)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of VectorStoreResult ordered by similarity (highest first)
        """
        ...
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete items by IDs.
        
        Args:
            ids: List of IDs to delete
        """
        ...
    
    def clear(self) -> None:
        """Clear all items from vector store."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with stats (count, dimension, etc.)
        """
        ...


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    Provides common functionality and enforces interface.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize base vector store.
        
        Args:
            dimension: Embedding dimension
        """
        self._dimension = dimension
        self._count = 0
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def _validate_embedding(self, embedding: List[float]) -> None:
        """
        Validate embedding dimension.
        
        Args:
            embedding: Embedding vector to validate
            
        Raises:
            ValueError: If dimension mismatch
        """
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {len(embedding)}"
            )
    
    def _validate_embeddings(self, embeddings: List[List[float]]) -> None:
        """
        Validate list of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Raises:
            ValueError: If dimension mismatch
        """
        for i, emb in enumerate(embeddings):
            try:
                self._validate_embedding(emb)
            except ValueError as e:
                raise ValueError(f"Embedding {i}: {e}") from e
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize similarity scores to 0.0-1.0 range.
        
        Args:
            scores: Raw similarity scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "count": self._count,
            "dimension": self._dimension,
        }
    
    @abstractmethod
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add items to vector store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorStoreResult]:
        """Search for similar items."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete items by IDs."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all items."""
        pass


__all__ = [
    "VectorStore",
    "VectorStoreResult",
    "BaseVectorStore",
]

