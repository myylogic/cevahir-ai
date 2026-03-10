# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: context_builder.py
Modül: chatting_management/components
Görev: Context Builder - Builds conversation context for Cevahir inference.
       SOLID Principle: Single Responsibility (context building only). Conversation
       context building, semantic search, memory integration ve context pruning
       işlemlerini yapar. Database operations'ı direkt handle etmez (storage layer
       kullanır).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (context building only)
- Design Patterns: Builder Pattern (context building)
- Endüstri Standartları: Context building best practices

KULLANIM:
- Conversation context building için
- Semantic search için
- Memory integration için
- Context pruning için

BAĞIMLILIKLAR:
- ConversationStorage: Conversation storage
- MemoryStorage: Memory storage
- Config: Yapılandırma

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
from typing import List, Dict, Any, Optional
from cognitive_management.cognitive_types import CognitiveState
from chatting_management.storage.conversation_storage import ConversationStorage
from chatting_management.storage.memory_storage import MemoryStorage
from chatting_management.config import ChattingConfig
from chatting_management.exceptions import ContextBuildingError

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds conversation context for Cevahir inference.
    
    SOLID Principle: Single Responsibility
    - Only responsible for context building
    - Does not handle database operations directly (uses storage layer)
    """
    
    def __init__(
        self,
        config: ChattingConfig,
        conversation_storage: ConversationStorage,
        memory_storage: Optional[MemoryStorage] = None
    ):
        """
        Initialize context builder.
        
        Args:
            config: Chatting configuration
            conversation_storage: Conversation storage instance
            memory_storage: Memory storage instance (optional)
        """
        self.config = config
        self.conversation_storage = conversation_storage
        self.memory_storage = memory_storage
    
    def build_context(
        self,
        user_id: str,
        session_id: str,
        current_message: str,
        max_tokens: Optional[int] = None
    ) -> CognitiveState:
        """
        Build cognitive state with full context.
        
        Args:
            user_id: User ID
            session_id: Session ID
            current_message: Current user message
            max_tokens: Maximum tokens for context (optional)
            
        Returns:
            CognitiveState with history and context
            
        Raises:
            ContextBuildingError: If context building fails
        """
        try:
            max_tokens = max_tokens or self.config.max_context_tokens
            
            # 1. Get recent conversation history
            history = self._get_recent_history(session_id, max_tokens)
            
            # 2. Add current message to history
            history.append({
                "role": "user",
                "content": current_message
            })
            
            # 3. Get user memory context (if enabled)
            if self.config.enable_user_memory and self.memory_storage:
                memory_context = self._get_memory_context(user_id, current_message)
                if memory_context:
                    # Add memory as system context
                    history.insert(0, {
                        "role": "system_summary",
                        "content": f"User context: {memory_context}"
                    })
            
            # 4. Build CognitiveState
            state = CognitiveState(
                history=history,
                step=len(history) // 2  # Approximate step count
            )
            
            logger.debug(f"Context built: {len(history)} messages, ~{self._estimate_tokens(history)} tokens")
            return state
            
        except Exception as e:
            logger.error(f"Error building context: {e}", exc_info=True)
            raise ContextBuildingError(f"Failed to build context: {e}") from e
    
    def _get_recent_history(
        self,
        session_id: str,
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation history within token limit.
        
        Args:
            session_id: Session ID
            max_tokens: Maximum tokens
            
        Returns:
            List of message dictionaries
        """
        try:
            # Get recent messages
            messages = self.conversation_storage.get_recent_messages(
                session_id=session_id,
                num_messages=self.config.max_recent_messages
            )
            
            # Convert to history format
            history = []
            total_tokens = 0
            
            # Process messages in reverse (oldest first)
            for message in messages:
                message_dict = {
                    "role": message.role,
                    "content": message.content
                }
                
                # Estimate tokens
                tokens = self._estimate_tokens([message_dict])
                
                if total_tokens + tokens > max_tokens:
                    break
                
                history.append(message_dict)
                total_tokens += tokens
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting recent history: {e}", exc_info=True)
            raise
    
    def _get_memory_context(
        self,
        user_id: str,
        query: str
    ) -> Optional[str]:
        """
        Get relevant user memory context.
        
        Args:
            user_id: User ID
            query: Current message (for semantic search)
            
        Returns:
            Memory context string or None
        """
        if not self.memory_storage:
            return None
        
        try:
            # Get high-priority memories
            memories = self.memory_storage.get_user_memories(
                user_id=user_id,
                priority="high",
                limit=self.config.memory_retrieval_top_k
            )
            
            if not memories:
                return None
            
            # Format memory context
            memory_texts = [f"- {m.content}" for m in memories[:self.config.memory_retrieval_top_k]]
            return "\n".join(memory_texts)
            
        except Exception as e:
            logger.warning(f"Error getting memory context: {e}", exc_info=True)
            return None
    
    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Estimate token count for messages.
        
        Simple estimation: ~4 characters per token (Turkish optimized).
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars / 4)  # Rough estimate: 4 chars per token

