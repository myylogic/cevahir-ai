# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: user_storage.py
Modül: chatting_management/storage
Görev: User Storage - User data persistence using database module. SOLID Principle:
       Dependency Inversion (depends on database interfaces). User storage, user
       persistence, user retrieval, user update ve user deletion işlemlerini yapar.
       Concrete database implementation'a bağımlı değil, interface'e bağımlı.

MİMARİ:
- SOLID Prensipleri: Dependency Inversion (database interface'e bağımlı)
- Design Patterns: Storage Pattern (user storage)
- Endüstri Standartları: Data persistence best practices

KULLANIM:
- User storage için
- User persistence için
- User retrieval için

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
from typing import Optional, Dict, Any
from database import UnitOfWork
from database.models import User
from database.utils.helpers import generate_uuid
from chatting_management.exceptions import UserNotFoundError

logger = logging.getLogger(__name__)


class UserStorage:
    """
    User storage using database module.
    
    SOLID Principle: Dependency Inversion
    - Depends on database.UnitOfWork (interface)
    - Not dependent on concrete database implementation
    """
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User instance or None if not found
        """
        try:
            with UnitOfWork() as uow:
                return uow.users.get_by_id(user_id)
        except Exception as e:
            logger.error(f"Error getting user: {e}", exc_info=True)
            raise
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.
        
        Args:
            email: User email
            
        Returns:
            User instance or None if not found
        """
        try:
            with UnitOfWork() as uow:
                return uow.users.get_by_email(email)
        except Exception as e:
            logger.error(f"Error getting user by email: {e}", exc_info=True)
            raise
    
    def get_user_by_google_id(self, google_id: str) -> Optional[User]:
        """
        Get user by Google OAuth ID.
        
        Args:
            google_id: Google OAuth ID
            
        Returns:
            User instance or None if not found
        """
        try:
            with UnitOfWork() as uow:
                return uow.users.get_by_google_id(google_id)
        except Exception as e:
            logger.error(f"Error getting user by Google ID: {e}", exc_info=True)
            raise
    
    def create_user(
        self,
        email: str,
        google_id: Optional[str] = None,
        name: Optional[str] = None,
        password_hash: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> User:
        """
        Create new user.
        
        Args:
            email: User email
            google_id: Google OAuth ID (optional)
            name: User name (optional)
            password_hash: Password hash (optional)
            preferences: User preferences (optional)
            
        Returns:
            Created User instance
        """
        try:
            user_id = generate_uuid()
            user = User(
                user_id=user_id,
                email=email,
                google_id=google_id,
                name=name,
                password_hash=password_hash,
                preferences=preferences or {}
            )
            
            with UnitOfWork() as uow:
                created_user = uow.users.create(user)
                logger.info(f"User created: {user_id} ({email})")
                return created_user
        except Exception as e:
            logger.error(f"Error creating user: {e}", exc_info=True)
            raise
    
    def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> User:
        """
        Update user preferences.
        
        Args:
            user_id: User ID
            preferences: New preferences dictionary
            
        Returns:
            Updated User instance
            
        Raises:
            UserNotFoundError: If user not found
        """
        try:
            with UnitOfWork() as uow:
                user = uow.users.get_by_id(user_id)
                if user is None:
                    raise UserNotFoundError(f"User not found: {user_id}")
                
                # Merge preferences
                current_prefs = user.preferences or {}
                current_prefs.update(preferences)
                user.preferences = current_prefs
                
                updated_user = uow.users.update(user)
                logger.info(f"User preferences updated: {user_id}")
                return updated_user
        except UserNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}", exc_info=True)
            raise

