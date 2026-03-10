# -*- coding: utf-8 -*-
"""
Vector Store Components
=======================
Vector database integrations for semantic search and RAG.

Phase 7.1: Vector Memory Enhancement
"""

from typing import Optional
from cognitive_management.config import CognitiveManagerConfig

# Import base interface and implementations
from .base import VectorStore, VectorStoreResult
from .memory_vector_store import MemoryVectorStore
from .chroma_vector_store import ChromaVectorStore

__all__ = [
    "VectorStore",
    "VectorStoreResult",
    "MemoryVectorStore",
    "ChromaVectorStore",
    "create_vector_store",
]


def create_vector_store(cfg: CognitiveManagerConfig, dimension: Optional[int] = None) -> VectorStore:
    """
    Factory function to create vector store based on configuration.
    
    Args:
        cfg: Cognitive manager configuration
        dimension: Optional embedding dimension (if None, uses default)
        
    Returns:
        Vector store instance
    """
    provider = cfg.memory.vector_store_provider.lower()
    
    if provider == "memory":
        return MemoryVectorStore(cfg, dimension=dimension)
    elif provider == "chroma":
        return ChromaVectorStore(cfg, dimension=dimension)
    elif provider == "pinecone":
        # TODO: Phase 7.1 - Pinecone implementation
        raise NotImplementedError("Pinecone vector store henüz implement edilmedi.")
    elif provider == "weaviate":
        # TODO: Phase 7.1 - Weaviate implementation
        raise NotImplementedError("Weaviate vector store henüz implement edilmedi.")
    elif provider == "qdrant":
        # TODO: Phase 7.1 - Qdrant implementation
        raise NotImplementedError("Qdrant vector store henüz implement edilmedi.")
    elif provider == "milvus":
        # TODO: Phase 7.1 - Milvus implementation
        raise NotImplementedError("Milvus vector store henüz implement edilmedi.")
    else:
        raise ValueError(f"Geçersiz vector store provider: {provider}")

