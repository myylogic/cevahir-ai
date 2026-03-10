# -*- coding: utf-8 -*-
"""
Database Configuration Tests
=============================

Test Edilen Dosya: database/config.py
Test Edilen Sınıf: DatabaseConfig

Endüstri Standartları: pytest, comprehensive coverage
"""

import pytest
import os
from pathlib import Path
from database.config import DatabaseConfig
from database.exceptions import DatabaseConfigurationError


class TestDatabaseConfig:
    """DatabaseConfig test suite"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = DatabaseConfig()
        assert config.db_type in {"postgresql", "sqlite", "mysql"}
    
    def test_sqlite_config(self):
        """Test SQLite configuration"""
        os.environ["DB_TYPE"] = "sqlite"
        config = DatabaseConfig()
        config.db_type = "sqlite"
        config.sqlite_path = Path("test.db")
        
        assert config.db_type == "sqlite"
        assert config.sqlite_path == Path("test.db")
    
    def test_postgresql_config(self):
        """Test PostgreSQL configuration"""
        os.environ["DB_TYPE"] = "postgresql"
        os.environ["POSTGRES_HOST"] = "localhost"
        os.environ["POSTGRES_PORT"] = "5432"
        os.environ["POSTGRES_DB"] = "testdb"
        os.environ["POSTGRES_USER"] = "testuser"
        os.environ["POSTGRES_PASSWORD"] = "testpass"
        
        config = DatabaseConfig()
        assert config.db_type == "postgresql"
        assert config.postgres_host == "localhost"
        assert config.postgres_port == 5432
        assert config.postgres_db == "testdb"
    
    def test_invalid_db_type(self):
        """Test invalid database type validation"""
        config = DatabaseConfig()
        config.db_type = "invalid"
        
        with pytest.raises(ValueError, match="Invalid db_type"):
            config.validate()
    
    def test_sqlite_connection_string(self, temp_db_path):
        """Test SQLite connection string generation"""
        config = DatabaseConfig()
        config.db_type = "sqlite"
        config.sqlite_path = temp_db_path
        
        conn_str = config.get_connection_string()
        assert "sqlite" in conn_str.lower()
        assert str(temp_db_path) in conn_str
    
    def test_postgresql_connection_string(self):
        """Test PostgreSQL connection string generation"""
        config = DatabaseConfig()
        config.db_type = "postgresql"
        config.postgres_host = "localhost"
        config.postgres_port = 5432
        config.postgres_db = "testdb"
        config.postgres_user = "testuser"
        config.postgres_password = "testpass"
        
        conn_str = config.get_connection_string()
        assert "postgresql" in conn_str.lower()
        assert "testdb" in conn_str
        assert "testuser" in conn_str
    
    def test_config_dict(self):
        """Test configuration dictionary export"""
        config = DatabaseConfig()
        config.db_type = "sqlite"
        
        config_dict = config.get_config_dict()
        assert "db_type" in config_dict
        assert config_dict["db_type"] == "sqlite"
    
    def test_pool_size_validation(self):
        """Test pool size validation"""
        config = DatabaseConfig()
        config.pool_size = 0
        
        with pytest.raises(ValueError, match="pool_size must be >= 1"):
            config.validate()
    
    def test_max_overflow_validation(self):
        """Test max overflow validation"""
        config = DatabaseConfig()
        config.max_overflow = -1
        
        with pytest.raises(ValueError, match="max_overflow must be >= 0"):
            config.validate()
    
    def test_sqlite_path_auto_creation(self):
        """Test SQLite path auto-creation"""
        config = DatabaseConfig()
        config.db_type = "sqlite"
        config.sqlite_path = Path("test_data/test.db")
        
        conn_str = config.get_connection_string()
        # Path should be created
        assert config.sqlite_path.parent.exists()

