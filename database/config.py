# -*- coding: utf-8 -*-
"""
Database Configuration Management
==================================

PostgreSQL configuration with environment variable support.
Endüstri Standardı: 12-Factor App (config via environment)
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DatabaseConfig:
    """
    Database configuration for PostgreSQL.
    
    Endüstri Standardı:
    - Environment variable based configuration
    - Type-safe configuration
    - Validation on initialization
    """
    
    # Database Type (PostgreSQL for production)
    db_type: str = field(default_factory=lambda: os.getenv("DB_TYPE", "postgresql"))
    
    # PostgreSQL Configuration
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "cevahir"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "cevahir"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    
    # Connection Pool Configuration
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20")))
    pool_timeout: int = field(default_factory=lambda: int(os.getenv("DB_POOL_TIMEOUT", "30")))
    pool_recycle: int = field(default_factory=lambda: int(os.getenv("DB_POOL_RECYCLE", "3600")))  # 1 hour
    
    # SQLite Configuration (Development only)
    sqlite_path: Optional[Path] = field(default_factory=lambda: Path("data/cevahir.db") if os.getenv("DB_TYPE", "").lower() == "sqlite" else Path("data/cevahir.db"))
    
    # Connection String Options
    echo: bool = field(default_factory=lambda: os.getenv("DB_ECHO", "False").lower() == "true")
    echo_pool: bool = field(default_factory=lambda: os.getenv("DB_ECHO_POOL", "False").lower() == "true")
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate database configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.db_type not in {"postgresql", "sqlite", "mysql"}:
            raise ValueError(f"Invalid db_type: {self.db_type}. Must be 'postgresql', 'sqlite', or 'mysql'")
        
        if self.db_type == "postgresql":
            if not self.postgres_password:
                raise ValueError("POSTGRES_PASSWORD must be set for PostgreSQL")
            if not self.postgres_db:
                raise ValueError("POSTGRES_DB must be set for PostgreSQL")
        
        elif self.db_type == "sqlite":
            # SQLite doesn't need password, validate path instead
            if self.sqlite_path is None:
                self.sqlite_path = Path("data/cevahir.db")
            # Ensure parent directory exists
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {self.pool_size}")
        
        if self.max_overflow < 0:
            raise ValueError(f"max_overflow must be >= 0, got {self.max_overflow}")
    
    def get_connection_string(self) -> str:
        """
        Get database connection string.
        
        Returns:
            Connection string for SQLAlchemy
            
        Raises:
            ValueError: If database type is unsupported
        """
        if self.db_type == "postgresql":
            # PostgreSQL connection string
            return (
                f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        
        elif self.db_type == "sqlite":
            # SQLite connection string (development only)
            if not self.sqlite_path:
                raise ValueError("sqlite_path must be set for SQLite")
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{self.sqlite_path.absolute()}"
        
        elif self.db_type == "mysql":
            # MySQL connection string
            mysql_password = os.getenv("MYSQL_PASSWORD", "")
            mysql_host = os.getenv("MYSQL_HOST", "localhost")
            mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
            mysql_db = os.getenv("MYSQL_DB", "cevahir")
            mysql_user = os.getenv("MYSQL_USER", "cevahir")
            return (
                f"mysql+pymysql://{mysql_user}:{mysql_password}"
                f"@{mysql_host}:{mysql_port}/{mysql_db}"
            )
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary (password masked)
        """
        config = {
            "db_type": self.db_type,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "echo": self.echo,
            "echo_pool": self.echo_pool,
        }
        
        if self.db_type == "postgresql":
            config.update({
                "postgres_host": self.postgres_host,
                "postgres_port": self.postgres_port,
                "postgres_db": self.postgres_db,
                "postgres_user": self.postgres_user,
                "postgres_password": "***" if self.postgres_password else "",
            })
        
        return config

