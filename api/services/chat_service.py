# -*- coding: utf-8 -*-
"""
Chat Service
============

Service layer for chat operations using ChattingManager.
"""

import logging
from typing import Dict, Any, Optional, List
from chatting_management import ChattingManager
from chatting_management.exceptions import (
    SessionNotFoundError,
    SessionAccessDeniedError,
    InvalidMessageError,
    CevahirIntegrationError,
)

logger = logging.getLogger(__name__)


class ChatService:
    """
    Chat service wrapping ChattingManager.
    
    Provides high-level chat operations for API layer.
    """
    
    def __init__(self, chatting_manager: ChattingManager):
        """
        Initialize chat service.
        
        Args:
            chatting_manager: ChattingManager instance
        """
        self.chatting_manager = chatting_manager
        logger.info("ChatService initialized")
    
    def send_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send message and get response.
        
        Args:
            user_id: User ID
            session_id: Session ID
            message: User message
            **kwargs: Additional parameters
        
        Returns:
            Response dictionary with message data
        
        Raises:
            SessionNotFoundError: If session not found
            SessionAccessDeniedError: If user doesn't own session
            InvalidMessageError: If message is invalid
            CevahirIntegrationError: If model processing fails
        """
        try:
            result = self.chatting_manager.send_message(
                user_id=user_id,
                session_id=session_id,
                message=message,
                **kwargs
            )
            
            logger.info(f"Message sent: user={user_id}, session={session_id}")
            return result
            
        except (SessionNotFoundError, SessionAccessDeniedError, InvalidMessageError, CevahirIntegrationError):
            raise
        except Exception as e:
            logger.error(f"Error in send_message: {e}", exc_info=True)
            raise CevahirIntegrationError(f"Failed to send message: {e}") from e
    
    def get_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            user_id: User ID
            session_id: Session ID
            limit: Maximum number of messages
        
        Returns:
            List of message dictionaries
        
        Raises:
            SessionNotFoundError: If session not found
            SessionAccessDeniedError: If user doesn't own session
        """
        try:
            messages = self.chatting_manager.get_conversation_history(
                session_id=session_id,
                user_id=user_id,
                limit=limit
            )
            
            logger.info(f"History retrieved: user={user_id}, session={session_id}, count={len(messages)}")
            return messages
            
        except (SessionNotFoundError, SessionAccessDeniedError):
            raise
        except Exception as e:
            logger.error(f"Error in get_conversation_history: {e}", exc_info=True)
            raise SessionNotFoundError(f"Failed to get conversation history: {e}") from e

