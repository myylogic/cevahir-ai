# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: memory_vector_store.py
Modül: cognitive_management/v2/components/vector_store
Görev: Memory Vector Store - In-memory vector store implementation using cosine
       similarity. Phase 7.1: Vector Memory Enhancement. Simple, fast, no
       external dependencies. Good for development and testing. In-memory vector
       storage with fast cosine similarity search.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (memory vector store),
                     Dependency Inversion (BaseVectorStore interface'e bağımlı)
- Design Patterns: Store Pattern (memory vector store)
- Endüstri Standartları: In-memory vector store best practices

KULLANIM:
- In-memory vector storage için
- Development ve testing için
- Fast cosine similarity search için

BAĞIMLILIKLAR:
- BaseVectorStore: Base vector store interface
- math: Cosine similarity hesaplama

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import math
from collections import defaultdict

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError
from .base import BaseVectorStore, VectorStoreResult


class MemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store using cosine similarity.
    
    Stores vectors in memory with fast cosine similarity search.
    No external dependencies, perfect for development and testing.
    """
    
    def __init__(self, cfg: CognitiveManagerConfig, dimension: Optional[int] = None):
        """
        Initialize memory vector store.
        
        Args:
            cfg: Cognitive manager configuration
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
        """
        # Use provided dimension or default
        embedding_dimension = dimension or 384  # Default: all-MiniLM-L6-v2 dimension
        super().__init__(dimension=embedding_dimension)
        
        self.cfg = cfg
        # Storage: list of (text, embedding, metadata, id)
        self._items: List[tuple[str, List[float], Dict[str, Any], str]] = []
        # ID to index mapping for fast deletion
        self._id_to_index: Dict[str, int] = {}
        # Next auto-generated ID
        self._next_id = 0
    
    def _get_next_id(self) -> str:
        """Generate next unique ID."""
        id_str = f"mem_{self._next_id}"
        self._next_id += 1
        return id_str
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0-1.0)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension.")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        # Cosine similarity is [-1, 1], normalize to [0, 1]
        return (similarity + 1.0) / 2.0
    
    def _filter_metadata(
        self,
        metadata: Dict[str, Any],
        filter_metadata: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches filter.
        
        Args:
            metadata: Item metadata
            filter_metadata: Filter criteria
            
        Returns:
            True if matches, False otherwise
        """
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add items to memory vector store.
        
        Args:
            texts: List of texts
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: Optional list of IDs
        """
        # Validate inputs
        if not texts:
            return
        
        if len(texts) != len(embeddings) or len(texts) != len(metadata):
            raise ValidationError(
                "texts, embeddings, ve metadata aynı uzunlukta olmalı."
            )
        
        # Set dimension from first embedding if not set
        if self._count == 0 and embeddings:
            self._dimension = len(embeddings[0])
        
        # Validate embeddings
        self._validate_embeddings(embeddings)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [self._get_next_id() for _ in texts]
        elif len(ids) != len(texts):
            raise ValidationError("ids listesi texts ile aynı uzunlukta olmalı.")
        
        # Add items
        for text, embedding, meta, item_id in zip(texts, embeddings, metadata, ids):
            # Check if ID already exists
            if item_id in self._id_to_index:
                # Update existing item
                idx = self._id_to_index[item_id]
                self._items[idx] = (text, embedding, meta, item_id)
            else:
                # Add new item
                idx = len(self._items)
                self._items.append((text, embedding, meta, item_id))
                self._id_to_index[item_id] = idx
                self._count += 1
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorStoreResult]:
        """
        Search for similar items using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            score_threshold: Minimum similarity score
            filter_metadata: Optional metadata filters
            
        Returns:
            List of VectorStoreResult ordered by similarity
        """
        self._validate_embedding(query_embedding)
        
        if not self._items:
            return []
        
        # Calculate similarities
        results = []
        for text, embedding, metadata, item_id in self._items:
            # Apply metadata filter
            if filter_metadata and not self._filter_metadata(metadata, filter_metadata):
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            # Apply score threshold
            if score_threshold is not None and similarity < score_threshold:
                continue
            
            results.append(VectorStoreResult(
                content=text,
                metadata=metadata,
                score=similarity,
                id=item_id,
            ))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k
        return results[:top_k]
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete items by IDs.
        
        Args:
            ids: List of IDs to delete
        """
        # Build set for fast lookup
        ids_to_delete = set(ids)
        
        # Remove items (in reverse order to maintain indices)
        new_items = []
        new_id_to_index = {}
        deleted_count = 0
        
        for idx, (text, embedding, metadata, item_id) in enumerate(self._items):
            if item_id not in ids_to_delete:
                new_idx = len(new_items)
                new_items.append((text, embedding, metadata, item_id))
                new_id_to_index[item_id] = new_idx
            else:
                deleted_count += 1
        
        self._items = new_items
        self._id_to_index = new_id_to_index
        self._count -= deleted_count
    
    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()
        self._id_to_index.clear()
        self._count = 0
        self._next_id = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = super().get_stats()
        stats.update({
            "provider": "memory",
            "items_count": len(self._items),
        })
        return stats


__all__ = ["MemoryVectorStore"]

