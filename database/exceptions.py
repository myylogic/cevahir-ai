# -*- coding: utf-8 -*-
"""
Database Exceptions
===================

Custom exceptions for database operations.
Endüstri Standardı: Specific, actionable error messages.
"""


class DatabaseError(Exception):
    """Base exception for database operations"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection errors"""
    pass


class DatabaseConfigurationError(DatabaseError):
    """Database configuration errors"""
    pass


class DatabaseMigrationError(DatabaseError):
    """Database migration errors"""
    pass


class DatabaseQueryError(DatabaseError):
    """Database query errors"""
    pass


class DatabaseIntegrityError(DatabaseError):
    """Database integrity constraint violations"""
    pass


class DatabaseNotFoundError(DatabaseError):
    """Resource not found in database"""
    pass


class DatabaseValidationError(DatabaseError):
    """Data validation errors"""
    pass

