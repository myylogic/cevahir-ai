# -*- coding: utf-8 -*-
"""
Repository Implementations
===========================

SOLID Principle: Dependency Inversion
- Concrete repository implementations
- Implement repository interfaces
- High-level modules use interfaces, not implementations
"""

from database.repositories.base_repository import BaseRepository
from database.repositories.user_repository import UserRepository
from database.repositories.session_repository import SessionRepository
from database.repositories.message_repository import MessageRepository
from database.repositories.user_memory_repository import UserMemoryRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "SessionRepository",
    "MessageRepository",
    "UserMemoryRepository",
]

