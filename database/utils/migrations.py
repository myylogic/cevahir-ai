# -*- coding: utf-8 -*-
"""
Database Migration Utilities
=============================

Migration utilities for database schema management.
Endüstri Standardı: Version-controlled schema changes
"""

import os
import logging
from pathlib import Path
from typing import Optional
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database.connection import db
from database.exceptions import DatabaseMigrationError

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Migration manager for database schema updates.
    
    Endüstri Standardı: Version-controlled migrations
    """
    
    def __init__(self, migrations_dir: Optional[Path] = None):
        """
        Initialize migration manager.
        
        Args:
            migrations_dir: Directory containing migration SQL files
        """
        if migrations_dir is None:
            migrations_dir = Path(__file__).parent.parent / "migrations"
        
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
    
    def create_schema(self, schema_file: Optional[Path] = None) -> None:
        """
        Create database schema from SQL file.
        
        Args:
            schema_file: Path to SQL schema file (default: schemas/schema_postgresql.sql)
            
        Raises:
            DatabaseMigrationError: If schema creation fails
        """
        if schema_file is None:
            schema_file = Path(__file__).parent.parent / "schemas" / "schema_postgresql.sql"
        
        schema_file = Path(schema_file)
        
        if not schema_file.exists():
            raise DatabaseMigrationError(f"Schema file not found: {schema_file}")
        
        try:
            logger.info(f"Creating database schema from: {schema_file}")
            
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            with db.get_session() as session:
                # Execute schema SQL
                session.execute(text(schema_sql))
                session.commit()
            
            logger.info("Database schema created successfully")
        
        except SQLAlchemyError as e:
            logger.error(f"Error creating schema: {e}", exc_info=True)
            raise DatabaseMigrationError(f"Failed to create schema: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error creating schema: {e}", exc_info=True)
            raise DatabaseMigrationError(f"Failed to create schema: {e}") from e
    
    def drop_all_tables(self) -> None:
        """
        Drop all tables (⚠️ DESTRUCTIVE - use with caution).
        
        Raises:
            DatabaseMigrationError: If drop fails
        """
        try:
            logger.warning("Dropping all database tables...")
            
            from database.models import Base
            
            Base.metadata.drop_all(db.get_engine())
            
            logger.warning("All database tables dropped")
        
        except Exception as e:
            logger.error(f"Error dropping tables: {e}", exc_info=True)
            raise DatabaseMigrationError(f"Failed to drop tables: {e}") from e
    
    def create_all_tables(self) -> None:
        """
        Create all tables from models (SQLAlchemy metadata).
        
        Raises:
            DatabaseMigrationError: If table creation fails
        """
        try:
            logger.info("Creating all database tables from models...")
            
            from database.models import Base
            
            Base.metadata.create_all(db.get_engine())
            
            logger.info("All database tables created successfully")
        
        except Exception as e:
            logger.error(f"Error creating tables: {e}", exc_info=True)
            raise DatabaseMigrationError(f"Failed to create tables: {e}") from e

