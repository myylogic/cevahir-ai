# -*- coding: utf-8 -*-
"""
Repository Interface
====================

SOLID Principle: Dependency Inversion
- Define repository interface (Protocol)
- Concrete repositories implement this interface
- High-level modules depend on interface, not implementation
"""

from typing import Protocol, TypeVar, Generic, Optional, List, Dict, Any
from abc import ABC, abstractmethod

T = TypeVar('T')


class IRepository(Protocol, Generic[T]):
    """
    Repository interface for data access operations.
    
    SOLID Principle: Dependency Inversion
    - High-level modules depend on this interface
    - Low-level modules (implementations) implement this interface
    
    Endüstri Standardı: Repository Pattern
    """
    
    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity instance or None if not found
        """
        ...
    
    @abstractmethod
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[T]:
        """
        Get all entities.
        
        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            
        Returns:
            List of entities
        """
        ...
    
    @abstractmethod
    def create(self, entity: T) -> T:
        """
        Create new entity.
        
        Args:
            entity: Entity instance to create
            
        Returns:
            Created entity instance
        """
        ...
    
    @abstractmethod
    def update(self, entity: T) -> T:
        """
        Update existing entity.
        
        Args:
            entity: Entity instance to update
            
        Returns:
            Updated entity instance
        """
        ...
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    @abstractmethod
    def count(self) -> int:
        """
        Get total count of entities.
        
        Returns:
            Total count
        """
        ...


class IUserRepository(Protocol):
    """User repository interface"""
    
    def get_by_email(self, email: str) -> Optional[Any]:
        """Get user by email"""
        ...
    
    def get_by_google_id(self, google_id: str) -> Optional[Any]:
        """Get user by Google ID"""
        ...


class ISessionRepository(Protocol):
    """Session repository interface"""
    
    def get_by_user_id(self, user_id: str, limit: Optional[int] = None) -> List[Any]:
        """Get sessions by user ID"""
        ...
    
    def get_active_sessions(self, user_id: str) -> List[Any]:
        """Get active sessions for user"""
        ...


class IMessageRepository(Protocol):
    """Message repository interface"""
    
    def get_by_session_id(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Any]:
        """Get messages by session ID"""
        ...
    
    def get_recent_messages(self, session_id: str, num_messages: int = 10) -> List[Any]:
        """Get recent messages for session"""
        ...


class IUserMemoryRepository(Protocol):
    """User memory repository interface"""
    
    def get_by_user_id(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        priority: Optional[str] = None
    ) -> List[Any]:
        """Get memories by user ID with optional filters"""
        ...
    
    def get_by_priority(self, user_id: str, priority: str) -> List[Any]:
        """Get memories by priority"""
        ...

