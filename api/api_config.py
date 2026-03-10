# -*- coding: utf-8 -*-
"""
API Configuration - Industry Standard
=====================================

12-Factor App Principles:
- Config via environment variables
- Environment-based configuration
- Type-safe configuration
- Validation on initialization

Endüstri Standartları:
- Production-ready configuration
- Security best practices
- Monitoring & logging
- Performance optimization
"""

import os
import logging
from pathlib import Path
from typing import Optional
from config.parameters import LOGGING_PATH, MODEL_SAVE_PATH, DEVICE


class Config:
    """
    Base configuration class.
    
    Endüstri Standardı: 12-Factor App
    - Configuration via environment variables
    - Type-safe configuration
    - Validation on initialization
    """
    # Flask Core
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv("SECRET_KEY") or os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
    
    # JSON Configuration
    JSONIFY_PRETTYPRINT_REGULAR = False
    JSON_SORT_KEYS = False
    JSON_AS_ASCII = False  # UTF-8 support
    
    # Request Configuration
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB
    SEND_FILE_MAX_AGE_DEFAULT = int(os.getenv("SEND_FILE_MAX_AGE", 3600))  # 1 hour
    
    # Application Paths
    LOGGING_PATH = LOGGING_PATH
    MODEL_SAVE_PATH = MODEL_SAVE_PATH
    DEVICE = DEVICE
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Log file paths
    PROCESS_LOG = os.path.join(LOGGING_PATH, 'api_process.log')
    ERROR_LOG = os.path.join(LOGGING_PATH, 'api_errors.log')
    ACCESS_LOG = os.path.join(LOGGING_PATH, 'api_access.log')
    
    # Security Configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") or SECRET_KEY
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_METHODS = os.getenv("CORS_METHODS", "GET,POST,PUT,PATCH,DELETE,OPTIONS").split(",")
    CORS_HEADERS = os.getenv("CORS_HEADERS", "Content-Type,Authorization,X-User-ID").split(",")
    
    # Rate Limiting
    RATELIMIT_ENABLED = os.getenv("RATELIMIT_ENABLED", "true").lower() == "true"
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200 per day, 50 per hour")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL")
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    
    # Cevahir Configuration
    CEVAHIR_DEVICE = os.getenv("CEVAHIR_DEVICE", DEVICE)
    CEVAHIR_MODEL_PATH = os.getenv("CEVAHIR_MODEL_PATH")
    CEVAHIR_VOCAB_PATH = os.getenv("CEVAHIR_VOCAB_PATH", "data/vocab_lib/vocab.json")
    CEVAHIR_MERGES_PATH = os.getenv("CEVAHIR_MERGES_PATH", "data/merges_lib/merges.txt")
    
    # ChattingManagement Configuration
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "450"))
    ENABLE_USER_MEMORY = os.getenv("ENABLE_USER_MEMORY", "true").lower() == "true"
    
    # Monitoring & Metrics
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PATH = os.getenv("METRICS_PATH", "/api/v3/metrics")
    
    # API Versioning
    API_VERSION = os.getenv("API_VERSION", "v3")
    API_PREFIX = f"/api/{API_VERSION}"
    
    # Request ID
    ENABLE_REQUEST_ID = os.getenv("ENABLE_REQUEST_ID", "true").lower() == "true"
    REQUEST_ID_HEADER = os.getenv("REQUEST_ID_HEADER", "X-Request-ID")
    
    # Security Headers
    ENABLE_SECURITY_HEADERS = os.getenv("ENABLE_SECURITY_HEADERS", "true").lower() == "true"
    
    @staticmethod
    def init_logging():
        """Loglama yapılandırmasını başlatır"""
        from logging.handlers import RotatingFileHandler
        
        os.makedirs(Config.LOGGING_PATH, exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(Config.LOG_FORMAT)
        
        # File handler (process log)
        file_handler = RotatingFileHandler(
            Config.PROCESS_LOG,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Error handler
        error_handler = RotatingFileHandler(
            Config.ERROR_LOG,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if Config.DEBUG else logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        logging.info("Logging configured successfully")

    @staticmethod
    def init_logging():
        """Loglama yapılandırmasını başlatır"""
        os.makedirs(Config.LOGGING_PATH, exist_ok=True)  # Log klasörünü oluştur
        logging.basicConfig(
            level=Config.LOG_LEVEL,
            format="%(asctime)s - [%(levelname)s] - %(message)s",
            handlers=[
                logging.FileHandler(Config.ERROR_LOG, mode='a'),
                logging.StreamHandler()
            ]
        )

class DevelopmentConfig(Config):
    """
    Development environment configuration.
    
    Endüstri Standardı: Development best practices
    - Debug mode enabled
    - Detailed logging
    - Pretty JSON responses
    """
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.getenv("DEVELOPMENT_SECRET_KEY", "dev-secret-key-change-in-production")
    JSONIFY_PRETTYPRINT_REGULAR = True
    LOG_LEVEL = "DEBUG"
    
    # Development-specific
    CORS_ORIGINS = ["*"]  # Allow all origins in development
    RATELIMIT_ENABLED = False  # Disable rate limiting in development


class TestingConfig(Config):
    """
    Testing environment configuration.
    
    Endüstri Standardı: Testing best practices
    - Test mode enabled
    - Minimal logging
    - Fast execution
    """
    DEBUG = True
    TESTING = True
    SECRET_KEY = os.getenv("TESTING_SECRET_KEY", "test-secret-key")
    MAX_CONTENT_LENGTH = int(os.getenv("TESTING_MAX_CONTENT_LENGTH", 8 * 1024 * 1024))  # 8 MB
    LOG_LEVEL = "WARNING"
    
    # Testing-specific
    RATELIMIT_ENABLED = False
    ENABLE_METRICS = False


class ProductionConfig(Config):
    """
    Production environment configuration.
    
    Endüstri Standardı: Production best practices
    - Debug mode disabled
    - Security hardened
    - Performance optimized
    - Error-only logging
    """
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv("PRODUCTION_SECRET_KEY")
    JSONIFY_PRETTYPRINT_REGULAR = False
    LOG_LEVEL = "ERROR"
    
    # Production-specific
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
    RATELIMIT_ENABLED = True
    ENABLE_SECURITY_HEADERS = True
    
    def __init__(self):
        super().__init__()
        # Production validation
        if not self.SECRET_KEY or self.SECRET_KEY == "change-me-in-production":
            raise ValueError("PRODUCTION_SECRET_KEY must be set in production environment")
        if not self.JWT_SECRET_KEY or self.JWT_SECRET_KEY == self.SECRET_KEY:
            raise ValueError("JWT_SECRET_KEY must be set and different from SECRET_KEY in production")

def get_config():
    """
    FLASK_ENV ortam değişkenine göre yapılandırmayı döndürür.
    Varsayılan olarak 'development' ortamını kullanır.
    """
    env = os.getenv("FLASK_ENV", "development").lower()
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Uygulama başlatıldığında yapılandırmayı başlat
config = get_config()
config.init_logging()
