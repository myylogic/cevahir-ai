# -*- coding: utf-8 -*-
"""
Session Repository Implementation
==================================

SOLID Principle: Dependency Inversion
- Implements ISessionRepository interface
- Extends BaseRepository
- Session-specific queries
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import Session as SessionModel
from database.repositories.base_repository import BaseRepository
from database.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class SessionRepository(BaseRepository[SessionModel]):
    """
    Session repository with session-specific queries.
    
    SOLID Principle: Dependency Inversion
    - Implements repository interface
    """
    
    def __init__(self, session: Session):
        """Initialize session repository"""
        super().__init__(session, SessionModel)
    
    def get_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[SessionModel]:
        """
        Get sessions by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of sessions
            offset: Number of sessions to skip
            
        Returns:
            List of sessions
        """
        try:
            query = self.session.query(SessionModel).filter_by(user_id=user_id).order_by(
                desc(SessionModel.last_activity)
            )
            
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting sessions by user ID: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get sessions for user: {user_id}") from e
    
    def get_active_sessions(
        self,
        user_id: str,
        inactive_hours: int = 24
    ) -> List[SessionModel]:
        """
        Get active sessions for user (recently active).
        
        Args:
            user_id: User ID
            inactive_hours: Consider sessions active if activity within this many hours
            
        Returns:
            List of active sessions
        """
        try:
            threshold = datetime.utcnow() - timedelta(hours=inactive_hours)
            return self.session.query(SessionModel).filter(
                SessionModel.user_id == user_id,
                SessionModel.last_activity >= threshold
            ).order_by(desc(SessionModel.last_activity)).all()
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get active sessions for user: {user_id}") from e
    
    def update_last_activity(self, session_id: str) -> bool:
        """
        Update last activity timestamp for session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if updated, False if not found
        """
        try:
            session = self.get_by_id(session_id)
            if session is None:
                return False
            
            session.last_activity = datetime.utcnow()
            self.session.flush()
            return True
        except Exception as e:
            logger.error(f"Error updating last activity: {e}", exc_info=True)
            raise DatabaseError(f"Failed to update last activity for session: {session_id}") from e

