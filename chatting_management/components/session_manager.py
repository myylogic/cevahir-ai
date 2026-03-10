# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: session_manager.py
Modül: chatting_management/components
Görev: Session Manager - Session management operations. SOLID Principle: Single
       Responsibility. Session creation, session retrieval, session update, session
       deletion ve session access control işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (session management)
- Design Patterns: Manager Pattern (session management)
- Endüstri Standartları: Session management best practices

KULLANIM:
- Session management için
- Session creation için
- Session access control için

BAĞIMLILIKLAR:
- SessionStorage: Session storage

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
from chatting_management.storage.session_storage import SessionStorage
from chatting_management.exceptions import SessionNotFoundError, SessionAccessDeniedError

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Session management operations.
    
    SOLID Principle: Single Responsibility
    """
    
    def __init__(self, session_storage: SessionStorage):
        """
        Initialize session manager.
        
        Args:
            session_storage: Session storage instance
        """
        self.session_storage = session_storage
    
    def get_session(self, session_id: str, user_id: Optional[str] = None):
        """
        Get session by ID with optional user validation.
        
        Args:
            session_id: Session ID
            user_id: User ID for validation (optional)
            
        Returns:
            Session instance
            
        Raises:
            SessionNotFoundError: If session not found
            SessionAccessDeniedError: If user doesn't own session
        """
        session = self.session_storage.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        if user_id and session.user_id != user_id:
            raise SessionAccessDeniedError(f"User {user_id} does not have access to session {session_id}")
        
        return session
    
    def create_session(self, user_id: str, title: Optional[str] = None, **kwargs):
        """Create new session"""
        return self.session_storage.create_session(user_id=user_id, title=title, **kwargs)
    
    def list_sessions(self, user_id: str, limit: Optional[int] = None) -> List[Any]:
        """List user sessions"""
        return self.session_storage.get_sessions_by_user_id(user_id=user_id, limit=limit)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        return self.session_storage.delete_session(session_id)
    
    def update_last_activity(self, session_id: str) -> bool:
        """Update session last activity"""
        return self.session_storage.update_last_activity(session_id)

