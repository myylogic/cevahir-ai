# -*- coding: utf-8 -*-
"""
Unit of Work Interface
======================

SOLID Principle: Dependency Inversion
- Transaction management abstraction
- Multiple repository operations in single transaction
"""

from typing import Protocol, Optional
from abc import abstractmethod


class IUnitOfWork(Protocol):
    """
    Unit of Work interface for transaction management.
    
    SOLID Principle: Dependency Inversion
    - High-level modules depend on this interface
    - Transaction management abstraction
    
    Endüstri Standardı: Unit of Work Pattern
    """
    
    @abstractmethod
    def __enter__(self):
        """Context manager entry"""
        ...
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        ...
    
    @abstractmethod
    def commit(self) -> None:
        """Commit transaction"""
        ...
    
    @abstractmethod
    def rollback(self) -> None:
        """Rollback transaction"""
        ...

