# -*- coding: utf-8 -*-
"""
Database Connection Management
================================

Endüstri Standartları:
- Singleton Pattern: Shared connection pool
- Connection Pooling: Efficient resource management
- Context Manager: Automatic session cleanup
- Error Handling: Robust exception handling
- Logging: Comprehensive logging

SOLID Principles:
- Single Responsibility: Connection management only
- Open/Closed: Extensible via configuration
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine, event, text
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import SQLAlchemyError

from .config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Database connection manager with connection pooling.
    
    Singleton Pattern: Single instance for shared connection pool.
    Thread-safe: Uses scoped_session for thread-local sessions.
    
    Endüstri Standardı:
    - Connection pooling
    - Automatic connection management
    - Transaction support
    - Error handling
    """
    
    _instance: Optional['DatabaseConnection'] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[scoped_session] = None
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize connection manager (only once)"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = DatabaseConfig()
        self._engine = None
        self._session_factory = None
        self._initialized = True
        
        logger.info(f"DatabaseConnection initialized (type: {self.config.db_type})")
    
    def initialize(self) -> None:
        """
        Initialize database connection and connection pool.
        
        Endüstri Standardı:
        - Connection pooling (QueuePool)
        - Connection validation
        - Error handling
        """
        if self._engine is not None:
            logger.debug("Database connection already initialized")
            return
        
        try:
            connection_string = self.config.get_connection_string()
            
            logger.info(f"Initializing database connection: {self.config.db_type}")
            
            # SQLite doesn't need connection pooling (file-based database)
            # Use NullPool for SQLite, QueuePool for PostgreSQL/MySQL
            if self.config.db_type == "sqlite":
                # SQLite: Use NullPool (no pooling needed for file-based DB)
                # For in-memory SQLite, use StaticPool for thread safety
                if ":memory:" in connection_string or "mode=memory" in connection_string:
                    pool_class = StaticPool
                    pool_kwargs = {"connect_args": {"check_same_thread": False}}
                else:
                    pool_class = NullPool
                    pool_kwargs = {}
                
                self._engine = create_engine(
                    connection_string,
                    poolclass=pool_class,
                    echo=self.config.echo,
                    echo_pool=self.config.echo_pool,
                    **pool_kwargs
                )
            else:
                # PostgreSQL/MySQL: Use QueuePool with connection pooling
                self._engine = create_engine(
                    connection_string,
                    poolclass=QueuePool,
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    pool_timeout=self.config.pool_timeout,
                    pool_recycle=self.config.pool_recycle,
                    echo=self.config.echo,
                    echo_pool=self.config.echo_pool,
                )
            
            # Add connection pool event listeners
            self._setup_connection_events()
            
            # Create session factory (thread-safe)
            self._session_factory = scoped_session(
                sessionmaker(
                    bind=self._engine,
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False,  # Prevent lazy loading issues
                )
            )
            
            logger.info(
                f"Database connection initialized successfully. "
                f"Pool size: {self.config.pool_size}, "
                f"Max overflow: {self.config.max_overflow}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}", exc_info=True)
            raise
    
    def _setup_connection_events(self) -> None:
        """Setup SQLAlchemy event listeners for connection management"""
        
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            """SQLite specific settings"""
            if self.config.db_type == "sqlite":
                dbapi_conn.execute("PRAGMA foreign_keys=ON")
        
        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Connection checkout event"""
            logger.debug("Database connection checked out from pool")
        
        @event.listens_for(self._engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Connection checkin event"""
            logger.debug("Database connection returned to pool")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session (context manager).
        
        Endüstri Standardı:
        - Automatic transaction management
        - Automatic rollback on error
        - Automatic session cleanup
        
        Usage:
            with db.get_session() as session:
                user = session.query(User).filter_by(user_id="...").first()
                # Automatic commit on success
                # Automatic rollback on exception
        
        Yields:
            Session: SQLAlchemy session
            
        Raises:
            SQLAlchemyError: Database operation errors
        """
        if self._session_factory is None:
            self.initialize()
        
        session = self._session_factory()
        
        try:
            yield session
            session.commit()
            logger.debug("Database transaction committed")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database transaction rolled back: {e}", exc_info=True)
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error in database transaction: {e}", exc_info=True)
            raise
        finally:
            session.close()
            logger.debug("Database session closed")
    
    def get_engine(self) -> Engine:
        """
        Get database engine.
        
        Returns:
            SQLAlchemy engine
            
        Raises:
            RuntimeError: If engine is not initialized
        """
        if self._engine is None:
            self.initialize()
        return self._engine
    
    def get_session_factory(self) -> scoped_session:
        """
        Get session factory (for advanced use cases).
        
        Returns:
            Scoped session factory
            
        Raises:
            RuntimeError: If session factory is not initialized
        """
        if self._session_factory is None:
            self.initialize()
        return self._session_factory
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}", exc_info=True)
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status.
        
        Returns:
            Dictionary with pool statistics
        """
        if self._engine is None:
            return {"status": "not_initialized"}
        
        pool = self._engine.pool
        
        return {
            "status": "initialized",
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
        }
    
    def close(self) -> None:
        """
        Close all database connections.
        
        Endüstri Standardı:
        - Graceful shutdown
        - Connection pool cleanup
        """
        if self._session_factory:
            self._session_factory.remove()
            logger.debug("Session factory removed")
        
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connections closed")
    
    def __repr__(self) -> str:
        """String representation"""
        status = "initialized" if self._engine else "not_initialized"
        return f"DatabaseConnection(type={self.config.db_type}, status={status})"


# Global database connection instance (Singleton)
db = DatabaseConnection()

