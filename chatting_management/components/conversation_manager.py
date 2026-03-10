# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: conversation_manager.py
Modül: chatting_management/components
Görev: Conversation Manager - Conversation management operations. SOLID Principle:
       Single Responsibility. Conversation creation, message retrieval, conversation
       history management ve conversation deletion işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (conversation management)
- Design Patterns: Manager Pattern (conversation management)
- Endüstri Standartları: Conversation management best practices

KULLANIM:
- Conversation management için
- Message retrieval için
- Conversation history management için

BAĞIMLILIKLAR:
- ConversationStorage: Conversation storage

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
from typing import List, Optional, Dict, Any
from chatting_management.storage.conversation_storage import ConversationStorage
from chatting_management.exceptions import ConversationError

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Conversation management operations.
    
    SOLID Principle: Single Responsibility
    """
    
    def __init__(self, conversation_storage: ConversationStorage):
        """
        Initialize conversation manager.
        
        Args:
            conversation_storage: Conversation storage instance
        """
        self.conversation_storage = conversation_storage
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add message to conversation.
        
        Args:
            session_id: Session ID
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Message metadata (optional)
            
        Returns:
            Created Message instance
        """
        return self.conversation_storage.add_message(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata
        )
    
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Any]:
        """Get conversation history"""
        return self.conversation_storage.get_conversation_history(
            session_id=session_id,
            limit=limit
        )
    
    def get_recent_messages(self, session_id: str, num_messages: int = 10) -> List[Any]:
        """Get recent messages"""
        return self.conversation_storage.get_recent_messages(
            session_id=session_id,
            num_messages=num_messages
        )
    
    def clear_conversation(self, session_id: str) -> int:
        """Clear conversation history"""
        return self.conversation_storage.clear_conversation(session_id)

