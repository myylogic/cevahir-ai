# -*- coding: utf-8 -*-
"""
Message Repository Implementation
==================================

SOLID Principle: Dependency Inversion
- Implements IMessageRepository interface
- Extends BaseRepository
- Message-specific queries
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import Message
from database.repositories.base_repository import BaseRepository
from database.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class MessageRepository(BaseRepository[Message]):
    """
    Message repository with message-specific queries.
    
    SOLID Principle: Dependency Inversion
    - Implements repository interface
    """
    
    def __init__(self, session: Session):
        """Initialize message repository"""
        super().__init__(session, Message)
    
    def get_by_session_id(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages by session ID.
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages
            offset: Number of messages to skip
            
        Returns:
            List of messages (chronological order)
        """
        try:
            query = self.session.query(Message).filter_by(session_id=session_id).order_by(
                Message.created_at
            )
            
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting messages by session ID: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get messages for session: {session_id}") from e
    
    def get_recent_messages(
        self,
        session_id: str,
        num_messages: int = 10
    ) -> List[Message]:
        """
        Get recent messages for session (most recent first).
        
        Args:
            session_id: Session ID
            num_messages: Number of recent messages to retrieve
            
        Returns:
            List of recent messages (most recent first)
        """
        try:
            return self.session.query(Message).filter_by(
                session_id=session_id
            ).order_by(desc(Message.created_at)).limit(num_messages).all()
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get recent messages for session: {session_id}") from e
    
    def get_by_role(
        self,
        session_id: str,
        role: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages by role (user or assistant).
        
        Args:
            session_id: Session ID
            role: Message role ('user' or 'assistant')
            limit: Maximum number of messages
            
        Returns:
            List of messages
        """
        try:
            query = self.session.query(Message).filter_by(
                session_id=session_id,
                role=role
            ).order_by(Message.created_at)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting messages by role: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get messages by role for session: {session_id}") from e

