# -*- coding: utf-8 -*-
"""
Database Module - PostgreSQL Database Management
================================================

Endüstri Standartları:
- SOLID Principles (Repository Pattern, Dependency Inversion)
- Clean Architecture (Layered design)
- Enterprise Features (Connection pooling, transaction management)
- Type Safety (Type hints, validation)

Mimari:
- Repository Pattern: Data access abstraction
- Unit of Work Pattern: Transaction management
- Factory Pattern: Connection factory
- Singleton Pattern: Connection pool

PostgreSQL Features:
- JSONB support (user preferences, metadata)
- Vector extension ready (pgvector - future)
- ACID transactions
- Scalable architecture
"""

from database.config import DatabaseConfig
from database.connection import DatabaseConnection, db
from database.models import (
    Base,
    User,
    Session,
    Message,
    UserMemory,
    ConversationSummary,
)
from database.unit_of_work import UnitOfWork

__all__ = [
    "DatabaseConfig",
    "DatabaseConnection",
    "db",
    "Base",
    "User",
    "Session",
    "Message",
    "UserMemory",
    "ConversationSummary",
    "UnitOfWork",
]

__version__ = "1.0.0"

