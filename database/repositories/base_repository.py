# -*- coding: utf-8 -*-
"""
Base Repository Implementation
===============================

SOLID Principle: Dependency Inversion
- Base repository with common CRUD operations
- Concrete repositories inherit from this
- Type-safe generic implementation

Endüstri Standardı: Repository Pattern
"""

import logging
from typing import TypeVar, Generic, Optional, List, Type
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database.exceptions import DatabaseError, DatabaseNotFoundError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(Generic[T]):
    """
    Base repository with common CRUD operations.
    
    SOLID Principle: Dependency Inversion
    - Concrete repositories inherit from this
    - Type-safe generic implementation
    
    Endüstri Standardı: Repository Pattern
    """
    
    def __init__(self, session: Session, model_class: Type[T]):
        """
        Initialize repository.
        
        Args:
            session: SQLAlchemy session
            model_class: SQLAlchemy model class
        """
        self.session = session
        self.model_class = model_class
    
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity instance or None if not found
        """
        try:
            # Get primary key column name dynamically
            pk_column = self._get_primary_key_column()
            return self.session.query(self.model_class).filter(
                pk_column == entity_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model_class.__name__} by ID: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get {self.model_class.__name__} by ID") from e
    
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[T]:
        """
        Get all entities.
        
        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            
        Returns:
            List of entities
        """
        try:
            query = self.session.query(self.model_class)
            
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.model_class.__name__}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get all {self.model_class.__name__}") from e
    
    def create(self, entity: T) -> T:
        """
        Create new entity.
        
        Args:
            entity: Entity instance to create
            
        Returns:
            Created entity instance
        """
        try:
            self.session.add(entity)
            self.session.flush()  # Get ID before commit
            logger.debug(f"Created {self.model_class.__name__}: {entity}")
            return entity
        except SQLAlchemyError as e:
            logger.error(f"Error creating {self.model_class.__name__}: {e}", exc_info=True)
            self.session.rollback()
            raise DatabaseError(f"Failed to create {self.model_class.__name__}") from e
    
    def update(self, entity: T) -> T:
        """
        Update existing entity.
        
        Args:
            entity: Entity instance to update
            
        Returns:
            Updated entity instance
        """
        try:
            self.session.merge(entity)
            self.session.flush()
            logger.debug(f"Updated {self.model_class.__name__}: {entity}")
            return entity
        except SQLAlchemyError as e:
            logger.error(f"Error updating {self.model_class.__name__}: {e}", exc_info=True)
            self.session.rollback()
            raise DatabaseError(f"Failed to update {self.model_class.__name__}") from e
    
    def delete(self, entity_id: str) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            entity = self.get_by_id(entity_id)
            if entity is None:
                return False
            
            self.session.delete(entity)
            self.session.flush()
            logger.debug(f"Deleted {self.model_class.__name__}: {entity_id}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error deleting {self.model_class.__name__}: {e}", exc_info=True)
            self.session.rollback()
            raise DatabaseError(f"Failed to delete {self.model_class.__name__}") from e
    
    def count(self) -> int:
        """
        Get total count of entities.
        
        Returns:
            Total count
        """
        try:
            return self.session.query(self.model_class).count()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model_class.__name__}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to count {self.model_class.__name__}") from e
    
    def _get_primary_key_column(self):
        """
        Get primary key column from model.
        
        Returns:
            Primary key column
        """
        # Get primary key column from model
        pk_columns = [col for col in self.model_class.__table__.columns if col.primary_key]
        if not pk_columns:
            raise DatabaseError(f"No primary key found for {self.model_class.__name__}")
        return pk_columns[0]

