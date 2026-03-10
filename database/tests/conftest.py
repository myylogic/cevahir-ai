# -*- coding: utf-8 -*-
"""
Pytest Configuration and Fixtures for Database Module
======================================================

Endüstri Standartları:
- pytest fixtures kullanımı
- Test database isolation
- SQLite for testing (fast, no external dependencies)
- Fixture scope management
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Optional
from unittest.mock import Mock, MagicMock, patch

# Import database module
import sys
BASE_DIR = Path(__file__).parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database import db, DatabaseConfig, Base
from database.models import User, Session, Message, UserMemory, ConversationSummary
from database.connection import DatabaseConnection
from database.utils.helpers import generate_uuid


# =============================================================================
# Test Database Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def temp_db_path() -> Generator[Path, None, None]:
    """
    Temporary database file path for SQLite testing.
    
    Test Edilen Dosya: config.py (DatabaseConfig.sqlite_path)
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_cevahir.db"
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def sqlite_config(temp_db_path: Path) -> DatabaseConfig:
    """
    SQLite configuration for testing.
    
    Test Edilen Dosya: config.py (DatabaseConfig)
    """
    # Set environment variable for SQLite
    original_db_type = os.environ.get("DB_TYPE")
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["DB_SQLITE_PATH"] = str(temp_db_path)
    
    config = DatabaseConfig()
    config.db_type = "sqlite"
    config.sqlite_path = temp_db_path
    
    yield config
    
    # Restore original environment
    if original_db_type:
        os.environ["DB_TYPE"] = original_db_type
    elif "DB_TYPE" in os.environ:
        del os.environ["DB_TYPE"]
    if "DB_SQLITE_PATH" in os.environ:
        del os.environ["DB_SQLITE_PATH"]


@pytest.fixture(scope="function")
def test_db(sqlite_config: DatabaseConfig) -> Generator[DatabaseConnection, None, None]:
    """
    Test database connection (SQLite in-memory).
    
    Test Edilen Dosya: connection.py (DatabaseConnection)
    
    Creates a fresh database connection for each test.
    """
    # Reset singleton instance
    DatabaseConnection._instance = None
    DatabaseConnection._engine = None
    DatabaseConnection._session_factory = None
    
    # Create new instance with test config
    test_db = DatabaseConnection()
    test_db.config = sqlite_config
    
    # Initialize with SQLite
    test_db.initialize()
    
    # Create all tables
    Base.metadata.create_all(test_db.get_engine())
    
    yield test_db
    
    # Cleanup
    test_db.close()
    DatabaseConnection._instance = None
    DatabaseConnection._engine = None
    DatabaseConnection._session_factory = None


@pytest.fixture(scope="function")
def db_session(test_db: DatabaseConnection):
    """
    Database session fixture.
    
    Test Edilen Dosya: connection.py (get_session)
    """
    with test_db.get_session() as session:
        yield session
        # Session automatically commits/rolls back in context manager


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def sample_user() -> User:
    """
    Sample user for testing.
    
    Test Edilen Dosya: models.py (User)
    """
    return User(
        user_id=generate_uuid(),
        email="test@example.com",
        name="Test User",
        preferences={"theme": "dark"}
    )


@pytest.fixture
def sample_session(sample_user: User) -> Session:
    """
    Sample session for testing.
    
    Test Edilen Dosya: models.py (Session)
    """
    return Session(
        session_id=generate_uuid(),
        user_id=sample_user.user_id,
        title="Test Session"
    )


@pytest.fixture
def sample_message(sample_session: Session) -> Message:
    """
    Sample message for testing.
    
    Test Edilen Dosya: models.py (Message)
    """
    return Message(
        message_id=generate_uuid(),
        session_id=sample_session.session_id,
        role="user",
        content="Test message content"
    )


@pytest.fixture
def sample_user_memory(sample_user: User) -> UserMemory:
    """
    Sample user memory for testing.
    
    Test Edilen Dosya: models.py (UserMemory)
    """
    return UserMemory(
        memory_id=generate_uuid(),
        user_id=sample_user.user_id,
        memory_type="fact",
        content="User likes Python",
        priority="high"
    )


# =============================================================================
# Repository Fixtures
# =============================================================================

@pytest.fixture
def user_repository(db_session):
    """
    User repository fixture.
    
    Test Edilen Dosya: repositories/user_repository.py
    """
    from database.repositories.user_repository import UserRepository
    return UserRepository(db_session)


@pytest.fixture
def session_repository(db_session):
    """
    Session repository fixture.
    
    Test Edilen Dosya: repositories/session_repository.py
    """
    from database.repositories.session_repository import SessionRepository
    return SessionRepository(db_session)


@pytest.fixture
def message_repository(db_session):
    """
    Message repository fixture.
    
    Test Edilen Dosya: repositories/message_repository.py
    """
    from database.repositories.message_repository import MessageRepository
    return MessageRepository(db_session)


@pytest.fixture
def user_memory_repository(db_session):
    """
    User memory repository fixture.
    
    Test Edilen Dosya: repositories/user_memory_repository.py
    """
    from database.repositories.user_memory_repository import UserMemoryRepository
    return UserMemoryRepository(db_session)


# =============================================================================
# Unit of Work Fixtures
# =============================================================================

@pytest.fixture
def unit_of_work(test_db: DatabaseConnection):
    """
    Unit of Work fixture.
    
    Test Edilen Dosya: unit_of_work.py (UnitOfWork)
    """
    from database.unit_of_work import UnitOfWork
    
    # Patch db connection to use test_db
    with patch('database.unit_of_work.db', test_db):
        with UnitOfWork() as uow:
            yield uow


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_user(user_id: Optional[str] = None, email: str = "test@example.com") -> User:
    """Create test user."""
    return User(
        user_id=user_id or generate_uuid(),
        email=email,
        name="Test User"
    )


def create_test_session(user_id: str, session_id: Optional[str] = None, title: str = "Test Session") -> Session:
    """Create test session."""
    return Session(
        session_id=session_id or generate_uuid(),
        user_id=user_id,
        title=title
    )


def create_test_message(session_id: str, role: str = "user", content: str = "Test message") -> Message:
    """Create test message."""
    return Message(
        message_id=generate_uuid(),
        session_id=session_id,
        role=role,
        content=content
    )

