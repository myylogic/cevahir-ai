# -*- coding: utf-8 -*-
"""
User Memory Repository Implementation
======================================

SOLID Principle: Dependency Inversion
- Implements IUserMemoryRepository interface
- Extends BaseRepository
- Memory-specific queries
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import UserMemory
from database.repositories.base_repository import BaseRepository
from database.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class UserMemoryRepository(BaseRepository[UserMemory]):
    """
    User memory repository with memory-specific queries.
    
    SOLID Principle: Dependency Inversion
    - Implements repository interface
    """
    
    def __init__(self, session: Session):
        """Initialize user memory repository"""
        super().__init__(session, UserMemory)
    
    def get_by_user_id(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        priority: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[UserMemory]:
        """
        Get memories by user ID with optional filters.
        
        Args:
            user_id: User ID
            memory_type: Filter by memory type (optional)
            priority: Filter by priority (optional)
            limit: Maximum number of memories
            
        Returns:
            List of user memories
        """
        try:
            query = self.session.query(UserMemory).filter_by(user_id=user_id)
            
            if memory_type:
                query = query.filter_by(memory_type=memory_type)
            if priority:
                query = query.filter_by(priority=priority)
            
            # Order by priority (high first) and creation date
            query = query.order_by(
                desc(UserMemory.priority == 'high'),
                desc(UserMemory.created_at)
            )
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting user memories: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get memories for user: {user_id}") from e
    
    def get_by_priority(
        self,
        user_id: str,
        priority: str
    ) -> List[UserMemory]:
        """
        Get memories by priority.
        
        Args:
            user_id: User ID
            priority: Priority level ('high', 'medium', 'low')
            
        Returns:
            List of memories with specified priority
        """
        return self.get_by_user_id(user_id, priority=priority)
    
    def get_by_type(
        self,
        user_id: str,
        memory_type: str
    ) -> List[UserMemory]:
        """
        Get memories by type.
        
        Args:
            user_id: User ID
            memory_type: Memory type ('fact', 'preference', 'pattern', etc.)
            
        Returns:
            List of memories of specified type
        """
        return self.get_by_user_id(user_id, memory_type=memory_type)

