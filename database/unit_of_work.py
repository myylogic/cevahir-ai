# -*- coding: utf-8 -*-
"""
Unit of Work Implementation
============================

SOLID Principle: Dependency Inversion
- Transaction management abstraction
- Multiple repository operations in single transaction
- Automatic rollback on error

Endüstri Standardı: Unit of Work Pattern
"""

import logging
from typing import Optional
from contextlib import contextmanager
from sqlalchemy.orm import Session

from database.connection import db
from database.repositories.user_repository import UserRepository
from database.repositories.session_repository import SessionRepository
from database.repositories.message_repository import MessageRepository
from database.repositories.user_memory_repository import UserMemoryRepository

logger = logging.getLogger(__name__)


class UnitOfWork:
    """
    Unit of Work for transaction management.
    
    SOLID Principle: Dependency Inversion
    - Transaction management abstraction
    - Multiple repositories in single transaction
    
    Endüstri Standardı: Unit of Work Pattern
    
    Usage:
        with UnitOfWork() as uow:
            user = uow.users.get_by_id("...")
            session = uow.sessions.create(...)
            uow.commit()  # Automatic on context exit (success)
    """
    
    def __init__(self):
        """Initialize Unit of Work"""
        self._session: Optional[Session] = None
        self._committed = False
    
    def __enter__(self):
        """Context manager entry - start transaction"""
        # Get session from connection manager (without auto-commit)
        self._session = db.get_session_factory()()
        self._committed = False
        
        # Initialize repositories with session
        self.users = UserRepository(self._session)
        self.sessions = SessionRepository(self._session)
        self.messages = MessageRepository(self._session)
        self.user_memories = UserMemoryRepository(self._session)
        
        logger.debug("UnitOfWork transaction started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - commit or rollback"""
        try:
            if exc_type is None and not self._committed:
                # No exception and not explicitly committed - auto commit
                self.commit()
            elif exc_type is not None:
                # Exception occurred - rollback
                self.rollback()
        finally:
            if self._session:
                self._session.close()
                db.get_session_factory().remove()
            logger.debug("UnitOfWork transaction ended")
    
    def commit(self) -> None:
        """
        Commit transaction.
        
        Note: In Unit of Work pattern, commit is usually called automatically
        on context exit. This method allows explicit commit if needed.
        
        Endüstri Standardı: Explicit transaction control
        """
        if self._session is None:
            raise RuntimeError("UnitOfWork not in transaction context")
        
        if self._committed:
            logger.warning("Transaction already committed")
            return
        
        try:
            self._session.commit()
            self._committed = True
            logger.debug("UnitOfWork transaction committed")
        except Exception as e:
            logger.error(f"Error committing transaction: {e}", exc_info=True)
            self._session.rollback()
            raise
    
    def rollback(self) -> None:
        """
        Rollback transaction.
        
        Endüstri Standardı: Explicit transaction control
        """
        if self._session is None:
            return
        
        try:
            self._session.rollback()
            logger.debug("UnitOfWork transaction rolled back")
        except Exception as e:
            logger.error(f"Error rolling back transaction: {e}", exc_info=True)
            raise

