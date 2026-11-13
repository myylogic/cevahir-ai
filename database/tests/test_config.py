# -*- coding: utf-8 -*-
"""
Database Configuration Tests
=============================

Test Edilen Dosya: database/config.py
Test Edilen Sınıf: DatabaseConfig

Endüstri Standartları: pytest, comprehensive coverage
"""

import pytest
from database.config import DatabaseConfig
from database.exceptions import DatabaseConfigurationError


# DatabaseConfig'in okuduğu tüm ortam değişkenleri (database/config.py).
_DB_ENV_VARS = (
    "DB_TYPE",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "DB_POOL_SIZE",
    "DB_MAX_OVERFLOW",
    "DB_POOL_TIMEOUT",
    "DB_POOL_RECYCLE",
    "DB_SQLITE_PATH",
    "DB_ECHO",
    "DB_ECHO_POOL",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_DB",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
)


@pytest.fixture(autouse=True)
def isolated_db_env(monkeypatch, tmp_path):
    """
    Her testi temiz ve geçerli bir ortamla başlatır.

    DatabaseConfig tüm alanlarını ortam değişkenlerinden okur ve __post_init__
    içinde doğrular. Testler eskiden os.environ'u kalıcı olarak değiştiriyordu:
    sonuç test sırasına bağlıydı (tek başına çalıştırılan 10 testten 8'i
    "POSTGRES_PASSWORD must be set" hatasıyla düşüyordu) ve POSTGRES_PASSWORD
    gibi değerler aynı oturumdaki diğer testlere sızıyordu. monkeypatch tüm
    değişiklikleri test sonunda geri alır.
    """
    for name in _DB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Varsayılan veritabanı PostgreSQL'dir; geçerli olması için tek zorunlu değer şifredir.
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    # SQLite doğrulaması üst dizini oluşturur (mkdir). Varsayılan yol "data/cevahir.db"
    # çalışma dizinine (depoya) yazacağı için yol tmp_path altına yönlendirilir.
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "cevahir.db"))


class TestDatabaseConfig:
    """DatabaseConfig test suite"""

    def test_default_config(self, monkeypatch):
        """Test default configuration"""
        monkeypatch.delenv("DB_SQLITE_PATH", raising=False)
        # DB_TYPE verilmediğinde varsayılan tip PostgreSQL'dir. Ortam artık hermetik
        # olduğu için beklenen değer kesindir (eskiden üç tipten biri kabul ediliyordu).
        config = DatabaseConfig()
        assert config.db_type == "postgresql"
        assert config.sqlite_path.as_posix() == "data/cevahir.db"

    def test_default_config_requires_postgres_password(self, monkeypatch):
        """Test that the default (PostgreSQL) config fails fast without a password"""
        # Eski test_default_config şifresiz bir ortamda bu hatayla düşüyordu. Bu,
        # kasıtlı ürün davranışıdır: eksik gizli değerde başlatma anında hata (12-factor).
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        with pytest.raises(ValueError, match="POSTGRES_PASSWORD must be set"):
            DatabaseConfig()

    def test_sqlite_config(self, monkeypatch, tmp_path):
        """Test SQLite configuration"""
        db_file = tmp_path / "test.db"
        monkeypatch.setenv("DB_TYPE", "sqlite")
        monkeypatch.setenv("DB_SQLITE_PATH", str(db_file))
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)  # SQLite needs no PG secret
        config = DatabaseConfig()

        # Değerler ortam değişkenlerinden gelmeli. Eskiden alanlar elle atanıp aynı
        # değerler geri okunuyordu; bu, ortamdan yapılandırmayı hiç test etmiyordu.
        assert config.db_type == "sqlite"
        assert config.sqlite_path == db_file

    def test_postgresql_config(self, monkeypatch):
        """Test PostgreSQL configuration"""
        monkeypatch.setenv("DB_TYPE", "postgresql")
        monkeypatch.setenv("POSTGRES_HOST", "localhost")
        monkeypatch.setenv("POSTGRES_PORT", "5432")
        monkeypatch.setenv("POSTGRES_DB", "testdb")
        monkeypatch.setenv("POSTGRES_USER", "testuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")

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

    def test_sqlite_path_auto_creation(self, tmp_path):
        """Test SQLite path auto-creation"""
        config = DatabaseConfig()
        config.db_type = "sqlite"
        # Eskiden Path("test_data/test.db") depoda test_data/ oluşturuyordu ve dizin bir
        # kez oluşunca test hiçbir şey kanıtlamıyordu. Artık tmp_path altında, önceden
        # var olmadığı doğrulanan bir dizin kullanılıyor.
        config.sqlite_path = tmp_path / "test_data" / "test.db"
        assert not config.sqlite_path.parent.exists()

        conn_str = config.get_connection_string()
        # Path should be created
        assert config.sqlite_path.parent.exists()

