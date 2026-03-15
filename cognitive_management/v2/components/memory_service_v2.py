# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: memory_service_v2.py
Modül: cognitive_management/v2/components
Görev: V2 Memory Service - Bağımsız implementasyon. V1'e bağımlı değil. Bellek
       yönetimi, context building, vector store entegrasyonu, embedding adapter
       kullanımı ve memory retrieval işlemlerini yapar. Vector memory enhancement
       ile genişletilmiş.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (memory yönetimi),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Service Pattern (memory service)
- Endüstri Standartları: Memory management best practices

KULLANIM:
- Bellek yönetimi için
- Context building için
- Vector store entegrasyonu için
- Memory retrieval için

BAĞIMLILIKLAR:
- VectorStore: Vector store işlemleri
- EmbeddingAdapter: Embedding işlemleri
- ContextPruning: Context pruning işlemleri
- Component Protocols: MemoryService interface

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import MemoryError, ValidationError
from cognitive_management.v2.interfaces.component_protocols import MemoryService as IMemoryService
from cognitive_management.v2.utils.context_pruning import (
    build_context as _build_context,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM_SUMMARY,
)

# Phase 7.1: Vector Memory Enhancement
from cognitive_management.v2.components.embedding_adapter import (
    create_embedding_adapter,
    BaseEmbeddingAdapter,
)
from cognitive_management.v2.components.vector_store import (
    create_vector_store,
    VectorStore,
    VectorStoreResult,
)


class MemoryServiceV2(IMemoryService):
    """
    V2 Memory Service - Bağımsız implementasyon.
    """
    
    def __init__(self, cfg: CognitiveManagerConfig):
        """
        Initialize V2 Memory Service.
        
        Phase 7.1: Enhanced with vector memory and embeddings support.
        
        Args:
            cfg: Cognitive manager configuration
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        self.cfg = cfg
        # Basit episodik not alanı (kalıcı kısa notlar)
        self._notes: List[str] = []
        # Episodic memory: Conversation turns with metadata
        self._episodic_memory: List[Dict[str, Any]] = []
        # Working memory: Recent context (last N turns)
        self._working_memory_size = 10  # Keep last 10 turns
        
        # Phase 7.1: Vector Memory Enhancement
        self._embedding_adapter: Optional[BaseEmbeddingAdapter] = None
        self._vector_store: Optional[VectorStore] = None
        self._vector_memory_enabled = False
        
        # Initialize vector memory if enabled
        if self.cfg.memory.enable_vector_memory:
            try:
                # Step 1: Create embedding adapter first to get dimension
                self._embedding_adapter = create_embedding_adapter(cfg)
                if self._embedding_adapter:
                    # Step 2: Get embedding dimension
                    embedding_dimension = self._embedding_adapter.dimension
                    
                    # Step 3: Create vector store with correct dimension
                    self._vector_store = create_vector_store(cfg, dimension=embedding_dimension)
                    
                    # Step 4: Verify dimensions match
                    if self._vector_store.dimension == embedding_dimension:
                        self._vector_memory_enabled = True
                    else:
                        # Dimension mismatch - log warning and disable
                        import logging
                        logging.warning(
                            f"Vector store dimension ({self._vector_store.dimension}) "
                            f"does not match embedding dimension ({embedding_dimension}). "
                            f"Vector memory disabled."
                        )
                        self._vector_memory_enabled = False
                else:
                    # Embedding adapter not created (disabled in config)
                    self._vector_memory_enabled = False
            except Exception as e:
                # Log warning but don't fail initialization
                # Fallback to keyword-based search
                import logging
                logging.warning(f"Vector memory initialization başarısız, keyword search kullanılacak: {e}")
                self._vector_memory_enabled = False
    
    def add_turn(
        self,
        history: List[Dict[str, Any]],
        role: str,
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Add conversation turn to history and episodic memory.
        
        Phase 3: Enhanced with episodic memory storage.
        
        Args:
            history: Current conversation history
            role: Role ("user" or "assistant")
            content: Message content
            
        Returns:
            Updated history
        """
        if not isinstance(history, list):
            raise MemoryError("history list olmalı.")
        r = (role or "").strip().lower()
        if r not in {ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM_SUMMARY}:
            r = ROLE_USER if r == "" else r  # bilinmeyene izin ver, etiketle
        c = (content or "").strip()
        if not c:
            # boş içerik ekleme
            return history
        
        # Add to history
        turn = {"role": r, "content": c}
        history.append(turn)
        
        # Add to episodic memory (for retrieval)
        memory_item = {
            "role": r,
            "content": c,
            "timestamp": len(self._episodic_memory),  # Simple index-based timestamp
        }
        self._episodic_memory.append(memory_item)
        
        # Phase 7.1: Add to vector store if enabled
        if self._vector_memory_enabled and self._embedding_adapter and self._vector_store:
            try:
                # Generate embedding
                embedding = self._embedding_adapter.encode_single(c)
                
                # Prepare metadata
                metadata = {
                    "role": r,
                    "timestamp": memory_item["timestamp"],
                }
                
                # Generate ID
                item_id = f"turn_{memory_item['timestamp']}"
                
                # Add to vector store
                self._vector_store.add(
                    texts=[c],
                    embeddings=[embedding],
                    metadata=[metadata],
                    ids=[item_id],
                )
            except Exception as e:
                # Log warning but continue (fallback to keyword search)
                import logging
                logging.warning(f"Vector store'a ekleme başarısız: {e}")
        
        # Maintain working memory size
        if len(self._episodic_memory) > self._working_memory_size * 2:
            # Keep only recent items (working memory optimization)
            # Note: Vector store items are kept (they provide long-term memory)
            self._episodic_memory = self._episodic_memory[-self._working_memory_size:]
        
        return history
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from memory using semantic search.
        
        Phase 7.1: Enhanced retrieval with:
        - Vector similarity search (if vector memory enabled)
        - Hybrid search (vector + keyword)
        - Keyword-based search (fallback)
        - Episodic memory search
        
        Args:
            query: Query text
            top_k: Number of relevant items to retrieve
            
        Returns:
            List of relevant context items with relevance scores
        """
        if not query or not query.strip():
            return []
        
        # Phase 7.1: Try vector search first if enabled
        vector_results = []
        keyword_results = []
        
        if self._vector_memory_enabled and self._embedding_adapter and self._vector_store:
            try:
                # Generate query embedding
                query_embedding = self._embedding_adapter.encode_single(query)
                
                # Search in vector store
                vector_search_results = self._vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    score_threshold=self.cfg.memory.rag_score_threshold,
                )
                
                # Convert VectorStoreResult to Dict format
                for result in vector_search_results:
                    vector_results.append({
                        "role": result.metadata.get("role", "assistant"),
                        "content": result.content,
                        "score": result.score,
                        "source": "vector_search",
                    })
            except Exception as e:
                # Log warning but continue with fallback
                import logging
                logging.warning(f"Vector search başarısız, keyword search kullanılacak: {e}")
        
        # Hybrid search: combine vector and keyword results
        alpha = self.cfg.memory.hybrid_search_alpha
        
        if vector_results and alpha > 0.0:
            # Use vector results (weighted by alpha)
            results = vector_results
            
            # Add keyword results if alpha < 1.0 (hybrid mode)
            if alpha < 1.0:
                keyword_results = self._keyword_search(query, top_k)
                # Combine results with weighted scores
                combined_results = self._combine_search_results(
                    vector_results, keyword_results, alpha
                )
                results = combined_results
        else:
            # Fallback to keyword search
            keyword_results = self._keyword_search(query, top_k)
            results = keyword_results
        
        # If still not enough results, use old semantic search as fallback
        if len(results) < top_k:
            semantic_results = self._semantic_search(query, top_k - len(results))
            results.extend(semantic_results)
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:top_k]
    
    def _combine_search_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        alpha: float
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search results with weighted scores.
        
        Args:
            vector_results: Vector search results
            keyword_results: Keyword search results
            alpha: Weight for vector results (0.0 = keyword only, 1.0 = vector only)
            
        Returns:
            Combined and deduplicated results
        """
        # Create content to result mapping
        content_to_result: Dict[str, Dict[str, Any]] = {}
        
        # Add vector results with alpha weight
        for result in vector_results:
            content = result["content"]
            content_to_result[content] = {
                **result,
                "score": result["score"] * alpha,
                "sources": ["vector_search"],
            }
        
        # Add keyword results with (1-alpha) weight
        for result in keyword_results:
            content = result["content"]
            keyword_score = result["score"] * (1.0 - alpha)
            
            if content in content_to_result:
                # Combine scores if same content found in both
                existing = content_to_result[content]
                existing["score"] = existing["score"] + keyword_score
                existing["sources"].append("keyword_search")
                existing["source"] = "hybrid_search"
            else:
                content_to_result[content] = {
                    **result,
                    "score": keyword_score,
                    "sources": ["keyword_search"],
                    "source": "keyword_search",
                }
        
        # Convert back to list and sort
        combined = list(content_to_result.values())
        combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        return combined
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Semantic search using vector similarity.
        
        Phase 3: Basic implementation with keyword-based similarity.
        Can be enhanced with actual embeddings (sentence-transformers, etc.)
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of relevant items with scores
        """
        if not hasattr(self, '_episodic_memory') or not self._episodic_memory:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for item in self._episodic_memory:
            content = item.get("content", "")
            if not content:
                continue
            
            content_lower = content.lower()
            content_words = set(content_lower.split())
            
            # Simple semantic similarity: word overlap + context similarity
            overlap = len(query_words & content_words)
            if overlap == 0:
                continue
            
            # Calculate similarity score
            # Jaccard similarity + keyword density
            union = len(query_words | content_words)
            jaccard = overlap / union if union > 0 else 0.0
            
            # Keyword density bonus
            keyword_density = sum(1 for word in query_words if word in content_lower) / len(query_words) if query_words else 0.0
            
            # Combined score
            score = (jaccard * 0.6) + (keyword_density * 0.4)
            
            if score > 0.1:  # Minimum threshold
                results.append({
                    "role": item.get("role", "assistant"),
                    "content": content,
                    "score": score,
                    "source": "episodic_memory"
                })
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Keyword-based search as fallback.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of relevant items
        """
        if not hasattr(self, '_episodic_memory') or not self._episodic_memory:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for item in self._episodic_memory:
            content = item.get("content", "")
            if not content:
                continue
            
            content_lower = content.lower()
            
            # Count keyword matches
            matches = sum(1 for word in query_words if word in content_lower)
            if matches == 0:
                continue
            
            # Score based on match count and position
            score = matches / len(query_words) if query_words else 0.0
            
            # Position bonus (earlier mentions are more relevant)
            first_match_pos = min(
                (content_lower.find(word) for word in query_words if word in content_lower),
                default=len(content_lower)
            )
            position_bonus = 1.0 - (first_match_pos / max(len(content_lower), 1))
            score = (score * 0.7) + (position_bonus * 0.3)
            
            if score > 0.2:  # Minimum threshold
                results.append({
                    "role": item.get("role", "assistant"),
                    "content": content,
                    "score": score,
                    "source": "keyword_search"
                })
        
        return results
    
    def summarize(
        self,
        history: List[Dict[str, Any]],
        max_length: int = 200
    ) -> str:
        """
        Summarize conversation history.
        
        Args:
            history: Conversation history
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        # Basit extractive summary
        summary_parts = []
        for item in history:
            content = item.get('content', '')
            if content:
                # İlk cümleyi al
                first_sentence = content.split('.')[0] if '.' in content else content[:100]
                summary_parts.append(first_sentence)
        
        summary = ' | '.join(summary_parts)
        if len(summary) > max_length:
            summary = summary[:max_length].rstrip() + ' …'
        
        return summary
    
    def build_context(
        self,
        *,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build context string from history.
        
        Args:
            user_message: Current user message
            history: Conversation history
            system_prompt: Optional system prompt
            
        Returns:
            Context string
        """
        try:
            return _build_context(self.cfg, user_message=user_message, history=history, system_prompt=system_prompt)
        except Exception as e:
            raise MemoryError("Bağlam oluşturma başarısız.") from e
    
    def summarize_if_needed(
        self,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Summarize history if needed (V1 compatibility).
        
        Args:
            history: Conversation history
            
        Returns:
            Summarized or original history
        """
        if not isinstance(history, list):
            raise MemoryError("history list olmalı.")
        
        # Eğitimsiz model için özetlemeyi devre dışı bırak
        if not self.cfg.memory.enable_session_summary:
            return history

        # Sadece son 3 turu koru - basit kırpma
        if len(history) > 6:
            return history[-3:]
        
        return history
    
    def inject_session_summary_if_needed(
        self,
        history: List[Dict[str, Any]]
    ) -> None:
        """
        Inject session summary into history if needed.
        
        Checks if we've reached the session_summary_every threshold and
        injects a system_summary role entry if needed.
        
        Args:
            history: Conversation history (modified in place)
        """
        if not isinstance(history, list):
            raise MemoryError("history list olmalı.")
        
        if not self.cfg.memory.enable_session_summary:
            return
        
        # Count user-assistant pairs (turns)
        # Each pair is one conversation turn
        user_turns = sum(1 for item in history if item.get("role") == ROLE_USER)
        
        # Check if we need to inject summary
        # session_summary_every=6 means inject after 6 turns
        if user_turns > 0 and user_turns % self.cfg.memory.session_summary_every == 0:
            # Check if summary already exists for this turn count
            # (avoid duplicate summaries)
            existing_summaries = [
                i for i, item in enumerate(history)
                if item.get("role") == ROLE_SYSTEM_SUMMARY
            ]
            
            # Only inject if we don't have a summary for this turn count
            if not existing_summaries or len(existing_summaries) < (user_turns // self.cfg.memory.session_summary_every):
                # Generate simple summary from recent history
                # Get last session_summary_every turns
                recent_turns = history[-self.cfg.memory.session_summary_every * 2:]  # *2 because user+assistant pairs
                summary_content = self._generate_session_summary(recent_turns)
                
                # Insert summary before the most recent turns
                # Find the position to insert (before the last session_summary_every turns)
                insert_pos = len(history) - (self.cfg.memory.session_summary_every * 2)
                insert_pos = max(0, insert_pos)
                
                summary_turn = {
                    "role": ROLE_SYSTEM_SUMMARY,
                    "content": summary_content,
                }
                history.insert(insert_pos, summary_turn)

                # Memory consolidation: persist summary to vector store for cross-session retrieval
                self._persist_summary_to_vector_store(
                    summary_text=summary_content,
                    turn_index=user_turns,
                )
    
    def _generate_session_summary(self, turns: List[Dict[str, Any]]) -> str:
        """
        Generate a meaningful session summary from conversation turns.

        Extracts actual content snippets (user questions and assistant answers)
        to create a semantically rich summary that can be retrieved later.

        Args:
            turns: List of conversation turns

        Returns:
            Summary text with actual content
        """
        if not turns:
            return "Önceki konuşma özeti yok."

        user_messages = [t.get("content", "") for t in turns if t.get("role") == ROLE_USER]
        assistant_messages = [t.get("content", "") for t in turns if t.get("role") == ROLE_ASSISTANT]

        summary_parts = ["[KONUŞMA ÖZETİ]"]

        # Include actual content from up to 3 most recent Q&A pairs
        pairs = list(zip(user_messages, assistant_messages))
        for i, (q, a) in enumerate(pairs[-3:], 1):
            q_snip = q[:150].rstrip() + ("..." if len(q) > 150 else "")
            a_snip = a[:200].rstrip() + ("..." if len(a) > 200 else "")
            summary_parts.append(f"S{i}: {q_snip}")
            summary_parts.append(f"C{i}: {a_snip}")

        return "\n".join(summary_parts)

    def _persist_summary_to_vector_store(self, summary_text: str, turn_index: int) -> None:
        """
        Persist session summary to vector store for cross-session retrieval.

        Session summaries are the most semantically rich items in episodic memory -
        they should always be available for retrieval in future sessions.

        Args:
            summary_text: Session summary text
            turn_index: Turn index for unique ID generation
        """
        if not self._vector_memory_enabled or not self._embedding_adapter or not self._vector_store:
            return

        try:
            embedding = self._embedding_adapter.encode_single(summary_text)
            item_id = f"summary_{turn_index}"
            self._vector_store.add(
                texts=[summary_text],
                embeddings=[embedding],
                metadata=[{"role": ROLE_SYSTEM_SUMMARY, "timestamp": turn_index}],
                ids=[item_id],
            )
        except Exception as e:
            import logging
            logging.debug(f"Session summary vector store'a yazılamadı: {e}")
    
    def prune(
        self,
        history: List[Dict[str, Any]],
        *,
        user_message: str
    ) -> List[Dict[str, Any]]:
        """
        Prune history based on importance (V1 compatibility).
        
        Args:
            history: Conversation history
            user_message: Current user message
            
        Returns:
            Pruned history
        """
        # Bu aşamada doğrudan history'i döndürüyoruz; asıl kırpma build_context içinde uygulanıyor.
        return list(history)
    
    def add_note(self, text: str) -> None:
        """
        Kısa kalıcı not ekler (oturumlar arası taşımak istediğin bilgiler için).
        """
        t = (text or "").strip()
        if not t:
            return
        if t not in self._notes:
            self._notes.append(t)

    def notes(self) -> List[str]:
        """Episodik notların kopyasını döndürür."""
        return list(self._notes)

    def clear_notes(self) -> None:
        """Episodik notları temizler."""
        self._notes.clear()


__all__ = ["MemoryServiceV2"]
