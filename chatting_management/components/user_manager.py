# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: user_manager.py
Modül: chatting_management/components
Görev: User Manager - User management operations. SOLID Principle: Single
       Responsibility (user operations only). User creation, user retrieval,
       user update, user deletion ve user profile management işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (user management)
- Design Patterns: Manager Pattern (user management)
- Endüstri Standartları: User management best practices

KULLANIM:
- User management için
- User creation için
- User profile management için

BAĞIMLILIKLAR:
- UserStorage: User storage

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
from typing import Optional, Dict, Any
from chatting_management.storage.user_storage import UserStorage
from chatting_management.exceptions import UserNotFoundError

logger = logging.getLogger(__name__)


class UserManager:
    """
    User management operations.
    
    SOLID Principle: Single Responsibility
    """
    
    def __init__(self, user_storage: UserStorage):
        """
        Initialize user manager.
        
        Args:
            user_storage: User storage instance
        """
        self.user_storage = user_storage
    
    def get_user(self, user_id: str):
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User instance
            
        Raises:
            UserNotFoundError: If user not found
        """
        user = self.user_storage.get_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User not found: {user_id}")
        return user
    
    def get_user_by_email(self, email: str) -> Optional[Any]:
        """Get user by email"""
        return self.user_storage.get_user_by_email(email)
    
    def create_user(
        self,
        email: str,
        google_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        """Create new user"""
        return self.user_storage.create_user(
            email=email,
            google_id=google_id,
            name=name,
            **kwargs
        )
    
    def update_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        return self.user_storage.update_user_preferences(user_id, preferences)

