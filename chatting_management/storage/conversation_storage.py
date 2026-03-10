# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: conversation_storage.py
Modül: chatting_management/storage
Görev: Conversation Storage - Conversation history persistence using database module.
       SOLID Principle: Dependency Inversion. Conversation storage, message persistence,
       message retrieval, conversation history management ve database operations
       işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Dependency Inversion (database interface'e bağımlı)
- Design Patterns: Storage Pattern (conversation storage)
- Endüstri Standartları: Data persistence best practices

KULLANIM:
- Conversation storage için
- Message persistence için
- Conversation history management için

BAĞIMLILIKLAR:
- database.UnitOfWork: Database unit of work
- database.models: Database models

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
from database import UnitOfWork
from database.models import Message
from database.utils.helpers import generate_uuid
from chatting_management.exceptions import ConversationError

logger = logging.getLogger(__name__)


class ConversationStorage:
    """
    Conversation storage using database module.
    
    SOLID Principle: Dependency Inversion
    """
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
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
        try:
            message_id = generate_uuid()
            message = Message(
                message_id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                metadata=metadata or {}
            )
            
            with UnitOfWork() as uow:
                created_message = uow.messages.create(message)
                logger.debug(f"Message added: {message_id} ({role}) to session {session_id}")
                return created_message
        except Exception as e:
            logger.error(f"Error adding message: {e}", exc_info=True)
            raise ConversationError(f"Failed to add message: {e}") from e
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Message]:
        """
        Get conversation history.
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages
            offset: Number of messages to skip
            
        Returns:
            List of messages (chronological order)
        """
        try:
            with UnitOfWork() as uow:
                return uow.messages.get_by_session_id(
                    session_id=session_id,
                    limit=limit,
                    offset=offset
                )
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}", exc_info=True)
            raise ConversationError(f"Failed to get conversation history: {e}") from e
    
    def get_recent_messages(
        self,
        session_id: str,
        num_messages: int = 10
    ) -> List[Message]:
        """
        Get recent messages (most recent first).
        
        Args:
            session_id: Session ID
            num_messages: Number of recent messages
            
        Returns:
            List of recent messages (most recent first)
        """
        try:
            with UnitOfWork() as uow:
                messages = uow.messages.get_recent_messages(
                    session_id=session_id,
                    num_messages=num_messages
                )
                # Reverse to get chronological order (oldest first)
                return list(reversed(messages))
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}", exc_info=True)
            raise ConversationError(f"Failed to get recent messages: {e}") from e
    
    def get_messages_by_role(
        self,
        session_id: str,
        role: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages by role.
        
        Args:
            session_id: Session ID
            role: Message role ('user' or 'assistant')
            limit: Maximum number of messages
            
        Returns:
            List of messages
        """
        try:
            with UnitOfWork() as uow:
                return uow.messages.get_by_role(
                    session_id=session_id,
                    role=role,
                    limit=limit
                )
        except Exception as e:
            logger.error(f"Error getting messages by role: {e}", exc_info=True)
            raise ConversationError(f"Failed to get messages by role: {e}") from e
    
    def clear_conversation(self, session_id: str) -> int:
        """
        Clear conversation history (delete all messages).
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of messages deleted
        """
        try:
            with UnitOfWork() as uow:
                messages = uow.messages.get_by_session_id(session_id=session_id)
                count = len(messages)
                
                for message in messages:
                    uow.messages.delete(message.message_id)
                
                logger.info(f"Conversation cleared: {count} messages deleted from session {session_id}")
                return count
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}", exc_info=True)
            raise ConversationError(f"Failed to clear conversation: {e}") from e

