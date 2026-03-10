# -*- coding: utf-8 -*-
"""
Database Interfaces (Protocols)
================================

SOLID Principles - Dependency Inversion:
- Repository interfaces define contracts
- Implementations depend on interfaces, not concrete classes
- Easy to swap implementations (testing, different databases)
"""

from database.interfaces.repository import IRepository
from database.interfaces.unit_of_work import IUnitOfWork

__all__ = [
    "IRepository",
    "IUnitOfWork",
]

