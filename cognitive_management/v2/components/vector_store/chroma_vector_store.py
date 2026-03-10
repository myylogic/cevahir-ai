# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: chroma_vector_store.py
Modül: cognitive_management/v2/components/vector_store
Görev: Chroma Vector Store - ChromaDB integration for vector storage and retrieval.
       Phase 7.1: Vector Memory Enhancement. Chroma is an open-source vector
       database perfect for production use. Supports persistent storage and efficient
       similarity search. Production-ready vector store implementation.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (Chroma vector store),
                     Dependency Inversion (BaseVectorStore interface'e bağımlı)
- Design Patterns: Store Pattern (Chroma vector store)
- Endüstri Standartları: Vector database best practices

KULLANIM:
- ChromaDB vector storage için
- Persistent vector storage için
- Efficient similarity search için

BAĞIMLILIKLAR:
- BaseVectorStore: Base vector store interface
- chromadb: ChromaDB library

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
import os
from pathlib import Path

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError, MemoryError
from .base import BaseVectorStore, VectorStoreResult


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store implementation.
    
    Provides persistent vector storage with efficient similarity search.
    """
    
    def __init__(self, cfg: CognitiveManagerConfig, dimension: Optional[int] = None):
        """
        Initialize Chroma vector store.
        
        Args:
            cfg: Cognitive manager configuration
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
        """
        # Use provided dimension or default
        embedding_dimension = dimension or 384  # Default: all-MiniLM-L6-v2
        super().__init__(dimension=embedding_dimension)
        
        self.cfg = cfg
        
        # Initialize Chroma client
        self._chroma_client = None
        self._collection = None
        self._collection_name = cfg.memory.vector_store_collection_name or "cognitive_memory"
        
        self._initialize_chroma()
    
    def _initialize_chroma(self) -> None:
        """Initialize Chroma client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb paketi yüklü değil. "
                "Yüklemek için: pip install chromadb"
            )
        
        try:
            # Determine persistence path
            if self.cfg.memory.vector_store_path:
                persist_directory = str(Path(self.cfg.memory.vector_store_path).expanduser().resolve())
                # Create directory if it doesn't exist
                os.makedirs(persist_directory, exist_ok=True)
            else:
                # In-memory mode (no persistence)
                persist_directory = None
            
            # Create Chroma client
            if persist_directory:
                self._chroma_client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                self._chroma_client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False)
                )
            
            # Get or create collection
            try:
                self._collection = self._chroma_client.get_collection(
                    name=self._collection_name
                )
            except Exception:
                # Collection doesn't exist, create it
                self._collection = self._chroma_client.create_collection(
                    name=self._collection_name,
                    metadata={"description": "Cognitive Management memory storage"}
                )
            
            # Update dimension if collection already has data
            # Chroma doesn't store dimension explicitly, so we infer from config
            # Default to 384 (all-MiniLM-L6-v2) but can be overridden
            collection_count = self._collection.count()
            if collection_count > 0:
                # Try to get dimension from existing embeddings
                # For now, use default - actual dimension will be validated on add
                pass
            
            self._count = collection_count
            
        except Exception as e:
            raise MemoryError(f"Chroma initialization başarısız: {e}") from e
    
    def _set_dimension_from_embedding(self, embedding: List[float]) -> None:
        """
        Set dimension from first embedding.
        
        Args:
            embedding: First embedding vector
        """
        if self._count == 0:
            self._dimension = len(embedding)
    
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add items to Chroma vector store.
        
        Args:
            texts: List of texts
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: Optional list of IDs
        """
        if not texts:
            return
        
        # Validate inputs
        if len(texts) != len(embeddings) or len(texts) != len(metadata):
            raise ValidationError(
                "texts, embeddings, ve metadata aynı uzunlukta olmalı."
            )
        
        # Set dimension from first embedding
        if embeddings:
            self._set_dimension_from_embedding(embeddings[0])
        
        # Validate embeddings
        self._validate_embeddings(embeddings)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"chroma_{i}_{self._count + i}" for i in range(len(texts))]
        elif len(ids) != len(texts):
            raise ValidationError("ids listesi texts ile aynı uzunlukta olmalı.")
        
        # Prepare data for Chroma
        # Chroma expects: ids, embeddings, documents, metadatas
        documents = texts
        embeddings_list = embeddings
        metadatas = []
        
        # Convert metadata to Chroma format (all values must be str, int, float, or bool)
        for meta in metadata:
            chroma_meta = {}
            for key, value in meta.items():
                if isinstance(value, (str, int, float, bool)):
                    chroma_meta[key] = value
                else:
                    # Convert non-serializable types to string
                    chroma_meta[key] = str(value)
            metadatas.append(chroma_meta)
        
        try:
            # Add to Chroma collection
            # If IDs already exist, Chroma will update them
            self._collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )
            
            # Update count
            self._count = self._collection.count()
            
        except Exception as e:
            raise MemoryError(f"Chroma'ya ekleme başarısız: {e}") from e
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorStoreResult]:
        """
        Search for similar items in Chroma.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            score_threshold: Minimum similarity score (Chroma uses distance, so we convert)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of VectorStoreResult ordered by similarity
        """
        self._validate_embedding(query_embedding)
        
        if self._count == 0:
            return []
        
        try:
            # Prepare where clause for metadata filtering
            where_clause = None
            if filter_metadata:
                # Chroma uses simple equality filters
                # For complex filters, we'll filter results after retrieval
                where_clause = filter_metadata
            
            # Search in Chroma
            # Chroma returns results sorted by distance (lower is better)
            # We'll convert distance to similarity score
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Get more results to filter by threshold
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
            
            # Extract results
            vector_store_results = []
            
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    doc_id = results["ids"][0][i]
                    doc_content = results["documents"][0][i]
                    doc_metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    # Convert distance to similarity score
                    # Chroma uses L2 distance, we normalize to 0-1 similarity
                    # For cosine similarity embeddings, we can use: similarity = 1 - distance
                    # But L2 distance needs different normalization
                    # Assuming normalized embeddings, we use: similarity = 1 / (1 + distance)
                    similarity = 1.0 / (1.0 + distance)
                    
                    # Apply score threshold
                    if score_threshold is not None and similarity < score_threshold:
                        continue
                    
                    # Apply additional metadata filtering if needed
                    if filter_metadata and where_clause is None:
                        if not all(
                            doc_metadata.get(key) == value
                            for key, value in filter_metadata.items()
                        ):
                            continue
                    
                    vector_store_results.append(VectorStoreResult(
                        content=doc_content,
                        metadata=doc_metadata,
                        score=similarity,
                        id=doc_id,
                    ))
            
            # Sort by score (descending) - Chroma already sorts by distance
            # but we want to ensure similarity-based sorting
            vector_store_results.sort(key=lambda x: x.score, reverse=True)
            
            # Return top_k
            return vector_store_results[:top_k]
            
        except Exception as e:
            raise MemoryError(f"Chroma arama başarısız: {e}") from e
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete items by IDs from Chroma.
        
        Args:
            ids: List of IDs to delete
        """
        if not ids:
            return
        
        try:
            self._collection.delete(ids=ids)
            self._count = self._collection.count()
        except Exception as e:
            raise MemoryError(f"Chroma'dan silme başarısız: {e}") from e
    
    def clear(self) -> None:
        """Clear all items from Chroma collection."""
        try:
            # Delete collection and recreate
            self._chroma_client.delete_collection(name=self._collection_name)
            self._collection = self._chroma_client.create_collection(
                name=self._collection_name,
                metadata={"description": "Cognitive Management memory storage"}
            )
            self._count = 0
        except Exception as e:
            raise MemoryError(f"Chroma temizleme başarısız: {e}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Chroma vector store statistics."""
        stats = super().get_stats()
        stats.update({
            "provider": "chroma",
            "collection_name": self._collection_name,
            "persistent": self.cfg.memory.vector_store_path is not None,
        })
        return stats


__all__ = ["ChromaVectorStore"]

