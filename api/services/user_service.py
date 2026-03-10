# -*- coding: utf-8 -*-
"""
User Service
============

Service layer for user operations using Database module.
"""

import logging
from typing import Dict, Any, Optional
from database import UnitOfWork
from database.models import User
from chatting_management.exceptions import UserNotFoundError

logger = logging.getLogger(__name__)


class UserService:
    """
    User service using Database module.
    
    Provides high-level user operations for API layer.
    """
    
    def get_user(self, user_id: str) -> Dict[str, Any]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
        
        Returns:
            User dictionary
        
        Raises:
            UserNotFoundError: If user not found
        """
        try:
            with UnitOfWork() as uow:
                user = uow.users.get_by_id(user_id)
                if user is None:
                    raise UserNotFoundError(f"User not found: {user_id}")
                
                return {
                    "user_id": user.user_id,
                    "email": user.email,
                    "name": user.name,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "metadata": user.metadata or {}
                }
                
        except UserNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error in get_user: {e}", exc_info=True)
            raise UserNotFoundError(f"Failed to get user: {e}") from e
    
    def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user preferences.
        
        Args:
            user_id: User ID
            preferences: Preferences dictionary
        
        Returns:
            Updated user dictionary
        
        Raises:
            UserNotFoundError: If user not found
        """
        try:
            with UnitOfWork() as uow:
                user = uow.users.get_by_id(user_id)
                if user is None:
                    raise UserNotFoundError(f"User not found: {user_id}")
                
                # Update preferences
                current_metadata = user.metadata or {}
                current_metadata.update(preferences)
                
                user.metadata = current_metadata
                uow.commit()
                
                logger.info(f"User preferences updated: user={user_id}")
                return {
                    "user_id": user.user_id,
                    "preferences": user.metadata
                }
                
        except UserNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error in update_user_preferences: {e}", exc_info=True)
            raise UserNotFoundError(f"Failed to update preferences: {e}") from e

