# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: session_storage.py
Modül: chatting_management/storage
Görev: Session Storage - Session data persistence using database module. SOLID
       Principle: Dependency Inversion. Session storage, session persistence,
       session retrieval, session update ve session deletion işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Dependency Inversion (database interface'e bağımlı)
- Design Patterns: Storage Pattern (session storage)
- Endüstri Standartları: Data persistence best practices

KULLANIM:
- Session storage için
- Session persistence için
- Session retrieval için

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
from typing import Optional, List, Dict, Any
from datetime import datetime
from database import UnitOfWork
from database.models import Session as SessionModel
from database.utils.helpers import generate_uuid
from chatting_management.exceptions import SessionNotFoundError

logger = logging.getLogger(__name__)


class SessionStorage:
    """
    Session storage using database module.
    
    SOLID Principle: Dependency Inversion
    """
    
    def get_session(self, session_id: str) -> Optional[SessionModel]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session instance or None if not found
        """
        try:
            with UnitOfWork() as uow:
                return uow.sessions.get_by_id(session_id)
        except Exception as e:
            logger.error(f"Error getting session: {e}", exc_info=True)
            raise
    
    def get_sessions_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[SessionModel]:
        """
        Get sessions by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of sessions
            
        Returns:
            List of sessions
        """
        try:
            with UnitOfWork() as uow:
                return uow.sessions.get_by_user_id(user_id=user_id, limit=limit)
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}", exc_info=True)
            raise
    
    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionModel:
        """
        Create new session.
        
        Args:
            user_id: User ID
            title: Session title (optional)
            metadata: Session metadata (optional)
            
        Returns:
            Created Session instance
        """
        try:
            session_id = generate_uuid()
            session = SessionModel(
                session_id=session_id,
                user_id=user_id,
                title=title or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                metadata=metadata or {}
            )
            
            with UnitOfWork() as uow:
                created_session = uow.sessions.create(session)
                logger.info(f"Session created: {session_id} for user {user_id}")
                return created_session
        except Exception as e:
            logger.error(f"Error creating session: {e}", exc_info=True)
            raise
    
    def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> SessionModel:
        """
        Update session metadata.
        
        Args:
            session_id: Session ID
            metadata: New metadata dictionary
            
        Returns:
            Updated Session instance
            
        Raises:
            SessionNotFoundError: If session not found
        """
        try:
            with UnitOfWork() as uow:
                session = uow.sessions.get_by_id(session_id)
                if session is None:
                    raise SessionNotFoundError(f"Session not found: {session_id}")
                
                # Merge metadata
                current_metadata = session.metadata or {}
                current_metadata.update(metadata)
                session.metadata = current_metadata
                
                updated_session = uow.sessions.update(session)
                logger.debug(f"Session metadata updated: {session_id}")
                return updated_session
        except SessionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating session metadata: {e}", exc_info=True)
            raise
    
    def update_last_activity(self, session_id: str) -> bool:
        """
        Update session last activity timestamp.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if updated, False if not found
        """
        try:
            with UnitOfWork() as uow:
                return uow.sessions.update_last_activity(session_id)
        except Exception as e:
            logger.error(f"Error updating last activity: {e}", exc_info=True)
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            with UnitOfWork() as uow:
                deleted = uow.sessions.delete(session_id)
                if deleted:
                    logger.info(f"Session deleted: {session_id}")
                return deleted
        except Exception as e:
            logger.error(f"Error deleting session: {e}", exc_info=True)
            raise

