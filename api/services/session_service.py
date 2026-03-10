# -*- coding: utf-8 -*-
"""
Session Service
===============

Service layer for session operations using ChattingManager.
"""

import logging
from typing import Dict, Any, Optional, List
from chatting_management import ChattingManager
from chatting_management.exceptions import UserNotFoundError

logger = logging.getLogger(__name__)


class SessionService:
    """
    Session service wrapping ChattingManager.
    
    Provides high-level session operations for API layer.
    """
    
    def __init__(self, chatting_manager: ChattingManager):
        """
        Initialize session service.
        
        Args:
            chatting_manager: ChattingManager instance
        """
        self.chatting_manager = chatting_manager
        logger.info("SessionService initialized")
    
    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create new session.
        
        Args:
            user_id: User ID
            title: Session title (optional)
            **kwargs: Additional session metadata
        
        Returns:
            Session dictionary
        
        Raises:
            UserNotFoundError: If user not found
        """
        try:
            session = self.chatting_manager.create_session(
                user_id=user_id,
                title=title,
                **kwargs
            )
            
            logger.info(f"Session created: user={user_id}, session={session.get('session_id')}")
            return session
            
        except UserNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error in create_session: {e}", exc_info=True)
            raise UserNotFoundError(f"Failed to create session: {e}") from e
    
    def list_sessions(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List user sessions.
        
        Args:
            user_id: User ID
            limit: Maximum number of sessions
        
        Returns:
            List of session dictionaries
        """
        try:
            sessions = self.chatting_manager.list_sessions(
                user_id=user_id,
                limit=limit
            )
            
            logger.info(f"Sessions listed: user={user_id}, count={len(sessions)}")
            return sessions
            
        except Exception as e:
            logger.error(f"Error in list_sessions: {e}", exc_info=True)
            raise

