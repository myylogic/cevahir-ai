# -*- coding: utf-8 -*-
"""
Database Connection Tests
==========================

Test Edilen Dosya: database/connection.py
Test Edilen Sınıf: DatabaseConnection

Endüstri Standartları: pytest, comprehensive coverage
"""

import pytest
from database.connection import DatabaseConnection
from database.exceptions import DatabaseConnectionError


class TestDatabaseConnection:
    """DatabaseConnection test suite"""
    
    def test_singleton_pattern(self):
        """Test singleton pattern implementation"""
        db1 = DatabaseConnection()
        db2 = DatabaseConnection()
        
        assert db1 is db2  # Same instance
    
    def test_initialize_sqlite(self, sqlite_config, temp_db_path):
        """Test SQLite connection initialization"""
        # Reset singleton
        DatabaseConnection._instance = None
        
        db_conn = DatabaseConnection()
        db_conn.config = sqlite_config
        db_conn.initialize()
        
        assert db_conn._engine is not None
        assert db_conn._session_factory is not None
    
    def test_get_session(self, test_db):
        """Test getting database session"""
        with test_db.get_session() as session:
            assert session is not None
            # Session should be usable
            result = session.execute("SELECT 1").scalar()
            assert result == 1
    
    def test_session_context_manager(self, test_db):
        """Test session context manager (auto-commit/rollback)"""
        from database.models import User
        from database.utils.helpers import generate_uuid
        
        # Test successful commit
        with test_db.get_session() as session:
            user = User(
                user_id=generate_uuid(),
                email="test@example.com"
            )
            session.add(user)
            # Auto-commit on context exit
        
        # Verify user was saved
        with test_db.get_session() as session:
            saved_user = session.query(User).filter_by(email="test@example.com").first()
            assert saved_user is not None
            assert saved_user.email == "test@example.com"
    
    def test_session_rollback_on_error(self, test_db):
        """Test session rollback on error"""
        from database.models import User
        from database.utils.helpers import generate_uuid
        
        # Create user first
        user_id = generate_uuid()
        with test_db.get_session() as session:
            user = User(user_id=user_id, email="test@example.com")
            session.add(user)
        
        # Try to create duplicate (should fail)
        with pytest.raises(Exception):  # Integrity error
            with test_db.get_session() as session:
                duplicate_user = User(user_id=generate_uuid(), email="test@example.com")  # Same email
                session.add(duplicate_user)
                # Should rollback on error
        
        # Original user should still exist
        with test_db.get_session() as session:
            user = session.query(User).filter_by(user_id=user_id).first()
            assert user is not None
    
    def test_get_engine(self, test_db):
        """Test getting database engine"""
        engine = test_db.get_engine()
        assert engine is not None
    
    def test_get_session_factory(self, test_db):
        """Test getting session factory"""
        factory = test_db.get_session_factory()
        assert factory is not None
    
    def test_test_connection(self, test_db):
        """Test connection testing"""
        assert test_db.test_connection() is True
    
    def test_get_pool_status(self, test_db):
        """Test getting connection pool status"""
        status = test_db.get_pool_status()
        assert "status" in status
    
    def test_close_connection(self, sqlite_config, temp_db_path):
        """Test closing database connection"""
        # Reset singleton
        DatabaseConnection._instance = None
        
        db_conn = DatabaseConnection()
        db_conn.config = sqlite_config
        db_conn.initialize()
        
        assert db_conn._engine is not None
        
        db_conn.close()
        
        # Engine should be disposed
        assert db_conn._engine is None
    
    def test_connection_repr(self, test_db):
        """Test connection string representation"""
        repr_str = repr(test_db)
        assert "DatabaseConnection" in repr_str
        assert "sqlite" in repr_str.lower() or "postgresql" in repr_str.lower()

