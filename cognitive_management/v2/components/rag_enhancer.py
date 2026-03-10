# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: rag_enhancer.py
Modül: cognitive_management/v2/components
Görev: RAG (Retrieval-Augmented Generation) Enhancer - Enhances context with
       retrieved information for better generation. Phase 7.1: Vector Memory
       Enhancement. Retrieval, context enhancement ve generation işlemlerini
       yapar. Akademik referans: Lewis et al. (2020), Guu et al. (2020).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (RAG enhancement),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Enhancer Pattern (RAG enhancement)
- Endüstri Standartları: RAG best practices

KULLANIM:
- Context enhancement için
- Retrieval için
- Generation için

BAĞIMLILIKLAR:
- MemoryService: Memory service interface
- Config: Yapılandırma

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

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError
from cognitive_management.v2.interfaces.component_protocols import MemoryService as IMemoryService


class RAGEnhancer:
    """
    Retrieval-Augmented Generation enhancer.
    
    Enhances user queries with retrieved context from memory using vector search.
    This helps the model provide more accurate and contextually relevant responses.
    """
    
    def __init__(
        self,
        memory_service: IMemoryService,
        cfg: CognitiveManagerConfig
    ):
        """
        Initialize RAG enhancer.
        
        Args:
            memory_service: Memory service for context retrieval
            cfg: Cognitive manager configuration
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        
        self.memory_service = memory_service
        self.cfg = cfg
        self.enabled = cfg.memory.enable_rag
    
    def enhance_context(
        self,
        user_message: str,
        existing_context: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Enhance context with retrieved information from memory.
        
        Retrieves relevant context from memory using semantic search and
        combines it with existing context to provide richer information
        for model generation.
        
        Args:
            user_message: Current user message/query
            existing_context: Optional existing context (from history, etc.)
            top_k: Number of retrieved documents (defaults to config value)
            
        Returns:
            Enhanced context string
        """
        if not self.enabled:
            # RAG disabled, return existing context or user message
            if existing_context:
                return existing_context
            return user_message
        
        if not user_message or not user_message.strip():
            return existing_context or ""
        
        # Get top_k from config if not provided
        top_k = top_k or self.cfg.memory.rag_top_k
        
        # Retrieve relevant context from memory
        retrieved_items = self.memory_service.retrieve_context(
            query=user_message,
            top_k=top_k
        )
        
        # If no retrieved items, return existing context
        if not retrieved_items:
            return existing_context or user_message
        
        # Format retrieved context
        retrieved_text = self._format_retrieved_context(retrieved_items)
        
        # Combine with existing context
        enhanced_context = self._combine_context(
            user_message=user_message,
            existing_context=existing_context,
            retrieved_context=retrieved_text
        )
        
        return enhanced_context
    
    def _format_retrieved_context(
        self,
        retrieved_items: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved items into context string.
        
        Args:
            retrieved_items: List of retrieved context items
            
        Returns:
            Formatted context string
        """
        if not retrieved_items:
            return ""
        
        formatted_parts = []
        for i, item in enumerate(retrieved_items, 1):
            content = item.get("content", "").strip()
            role = item.get("role", "assistant")
            score = item.get("score", 0.0)
            
            if not content:
                continue
            
            # Format: [Reference X] (role): content
            formatted_parts.append(
                f"[Reference {i}] ({role}): {content}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _combine_context(
        self,
        user_message: str,
        existing_context: Optional[str],
        retrieved_context: str
    ) -> str:
        """
        Combine user message, existing context, and retrieved context.
        
        Args:
            user_message: Current user message
            existing_context: Optional existing context
            retrieved_context: Retrieved context from memory
            
        Returns:
            Combined enhanced context
        """
        parts = []
        
        # Add existing context if present
        if existing_context and existing_context.strip():
            parts.append(existing_context)
        
        # Add retrieved context
        if retrieved_context and retrieved_context.strip():
            parts.append(
                "[RETRIEVED CONTEXT FROM MEMORY]\n"
                "The following information from previous conversations may be relevant:\n"
                f"{retrieved_context}\n"
                "[END RETRIEVED CONTEXT]"
            )
        
        # Add user message
        parts.append(f"[CURRENT USER MESSAGE]\n{user_message}")
        
        # Combine with separator
        enhanced_context = "\n\n".join(parts)
        
        return enhanced_context
    
    def enhance_context_minimal(
        self,
        user_message: str,
        existing_context: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Minimal context enhancement (inline format).
        
        Similar to enhance_context but uses a more compact format
        suitable for shorter context windows.
        
        Args:
            user_message: Current user message
            existing_context: Optional existing context
            top_k: Number of retrieved documents
            
        Returns:
            Enhanced context (minimal format)
        """
        if not self.enabled:
            return existing_context or user_message
        
        if not user_message or not user_message.strip():
            return existing_context or ""
        
        top_k = top_k or self.cfg.memory.rag_top_k
        
        # Retrieve relevant context
        retrieved_items = self.memory_service.retrieve_context(
            query=user_message,
            top_k=top_k
        )
        
        if not retrieved_items:
            return existing_context or user_message
        
        # Format as inline references
        references = []
        for item in retrieved_items:
            content = item.get("content", "").strip()
            if content:
                # Truncate if too long
                if len(content) > 150:
                    content = content[:150] + "..."
                references.append(content)
        
        if not references:
            return existing_context or user_message
        
        # Create minimal format
        references_text = " | ".join(references)
        
        # Combine
        if existing_context:
            return f"{existing_context}\n\n[Relevant context: {references_text}]\n\n{user_message}"
        else:
            return f"[Relevant context: {references_text}]\n\n{user_message}"


__all__ = ["RAGEnhancer"]

