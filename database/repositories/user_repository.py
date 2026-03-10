# -*- coding: utf-8 -*-
"""
User Repository Implementation
================================

SOLID Principle: Dependency Inversion
- Implements IUserRepository interface
- Extends BaseRepository
- Type-safe user operations
"""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session

from database.models import User
from database.repositories.base_repository import BaseRepository
from database.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository[User]):
    """
    User repository with user-specific queries.
    
    SOLID Principle: Dependency Inversion
    - Implements repository interface
    - High-level modules depend on interface
    """
    
    def __init__(self, session: Session):
        """Initialize user repository"""
        super().__init__(session, User)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: Email address
            
        Returns:
            User instance or None if not found
        """
        try:
            return self.session.query(User).filter_by(email=email).first()
        except Exception as e:
            logger.error(f"Error getting user by email: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get user by email: {email}") from e
    
    def get_by_google_id(self, google_id: str) -> Optional[User]:
        """
        Get user by Google OAuth ID.
        
        Args:
            google_id: Google OAuth ID
            
        Returns:
            User instance or None if not found
        """
        try:
            return self.session.query(User).filter_by(google_id=google_id).first()
        except Exception as e:
            logger.error(f"Error getting user by Google ID: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get user by Google ID: {google_id}") from e
    
    def get_by_user_id(self, user_id: str) -> Optional[User]:
        """
        Get user by user ID (alias for get_by_id).
        
        Args:
            user_id: User ID
            
        Returns:
            User instance or None if not found
        """
        return self.get_by_id(user_id)

