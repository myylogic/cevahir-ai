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

Modes, selected by FLASK_ENV:
- unset: safe default. Debug off, rate limiting on, CORS limited to
  http(s)://localhost, 127.0.0.1 and [::1] origins (any port).
- development: explicit local development. Debug and pretty JSON on, rate
  limiting off, the same localhost CORS default. Only this mode and testing
  accept CEVAHIR_TRUST_X_USER_ID=1, which lets the X-User-ID header act as the
  caller's identity; otherwise identity comes only from a verified JWT.
- testing: test runs; rate limiting and metrics off.
- production: refuses to load without real PRODUCTION_SECRET_KEY and
  JWT_SECRET_KEY values; no cross-origin access unless CORS_ORIGINS lists it.
Unknown FLASK_ENV values are an error. Security settings are read from the
environment when get_config() creates the configuration, not at import time.
"""

import os
import logging
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
LOGGING_PATH = os.getenv("LOGGING_PATH", str(_ROOT / "logs"))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", str(_ROOT / "model" / "checkpoints"))
DEVICE = os.getenv("CEVAHIR_DEVICE", "cpu")

# Example values shipped in this repository; never acceptable as real secrets.
PLACEHOLDER_SECRETS = frozenset({
    "change-me",
    "change-me-in-production",
    "dev-secret-key-change-in-production",
    "test-secret-key",
})
MIN_PRODUCTION_SECRET_LENGTH = 32

# Default browser origins outside production: http(s) on localhost, any port.
# A regular expression (Flask-CORS compiles it), anchored at both ends and free of commas
# so that it can also be written into the comma-separated CORS_ORIGINS variable.
LOCALHOST_CORS_ORIGINS = (r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:[0-9]+)?$",)
DEFAULT_CORS_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
DEFAULT_CORS_HEADERS = ("Content-Type", "Authorization", "X-Request-ID")

# The only modes in which CEVAHIR_TRUST_X_USER_ID may make X-User-ID an identity source.
TRUSTED_USER_HEADER_ENVS = frozenset({"development", "testing"})


class ConfigurationError(ValueError):
    """The selected API configuration is unsafe or incomplete."""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be 1/0, true/false, yes/no or on/off, not {value!r}")


def _env_list(name: str, default) -> list:
    value = os.getenv(name, "").strip()
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _secret_problems(name: str, value: Optional[str]) -> list:
    value = (value or "").strip()
    if not value:
        return [f"{name} is not set"]
    if value in PLACEHOLDER_SECRETS:
        return [f"{name} uses the placeholder value {value!r} shipped with this repository"]
    if len(value) < MIN_PRODUCTION_SECRET_LENGTH:
        return [f"{name} is shorter than {MIN_PRODUCTION_SECRET_LENGTH} characters"]
    return []


class Config:
    """
    Base configuration; its values are the safe default (DefaultConfig, FLASK_ENV unset).
    
    Endüstri Standardı: 12-Factor App
    - Configuration via environment variables
    - Type-safe configuration
    - Validation on initialization

    SECRET_KEY, JWT_SECRET_KEY, RATELIMIT_ENABLED, TRUST_X_USER_ID and the
    CORS_* lists are resolved in __init__, i.e. when get_config() is called.
    """
    API_ENV = "default"

    # Flask Core
    DEBUG = False
    TESTING = False
    
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
    
    # Security Configuration (SECRET_KEY, JWT_SECRET_KEY and TRUST_X_USER_ID: see __init__)
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))
    
    # CORS Configuration: comma-separated CORS_ORIGINS, CORS_METHODS and CORS_HEADERS
    # environment variables replace these defaults (see __init__).
    cors_origins_default = LOCALHOST_CORS_ORIGINS
    
    # Rate Limiting (RATELIMIT_ENABLED: see __init__)
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

    def __init__(self):
        problems = []
        generic_secret = os.getenv("SECRET_KEY") or os.getenv("FLASK_SECRET_KEY") or "change-me-in-production"
        self.SECRET_KEY = self._secret_key(generic_secret)
        # api/security/jwt.py refuses placeholder secrets when signing or verifying, so a
        # configuration without a real JWT_SECRET_KEY answers 401 on authenticated endpoints.
        self.JWT_SECRET_KEY = self._jwt_secret_key(generic_secret)
        self.RATELIMIT_ENABLED = self._ratelimit_enabled()
        self.TRUST_X_USER_ID = _env_flag("CEVAHIR_TRUST_X_USER_ID")
        if self.TRUST_X_USER_ID and self.API_ENV not in TRUSTED_USER_HEADER_ENVS:
            problems.append(
                "CEVAHIR_TRUST_X_USER_ID=1 lets any caller choose its identity and is accepted only "
                "with FLASK_ENV=development or FLASK_ENV=testing"
            )
        self.CORS_ORIGINS = _env_list("CORS_ORIGINS", self.cors_origins_default)
        self.CORS_METHODS = _env_list("CORS_METHODS", DEFAULT_CORS_METHODS)
        identity_header = ("X-User-ID",) if self.TRUST_X_USER_ID else ()
        self.CORS_HEADERS = _env_list("CORS_HEADERS", DEFAULT_CORS_HEADERS + identity_header)
        self._validate(problems)
        if problems:
            raise ConfigurationError(
                f"Refusing to load the {self.API_ENV!r} API configuration: " + "; ".join(problems) + "."
            )

    def _secret_key(self, generic_secret):
        return generic_secret

    def _jwt_secret_key(self, generic_secret):
        return os.getenv("JWT_SECRET_KEY") or generic_secret

    def _ratelimit_enabled(self):
        return _env_flag("RATELIMIT_ENABLED", default=True)

    def _validate(self, problems):
        """Mode-specific checks append human-readable problems to ``problems``."""
    
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

class DefaultConfig(Config):
    """
    Safe default used when FLASK_ENV is unset.

    - Debug mode disabled
    - Rate limiting enabled (RATELIMIT_ENABLED=false turns it off)
    - CORS limited to localhost origins unless CORS_ORIGINS is set
    - Identity only from a verified JWT
    """


class DevelopmentConfig(Config):
    """
    Explicit development configuration (FLASK_ENV=development).
    
    Endüstri Standardı: Development best practices
    - Debug mode enabled
    - Detailed logging
    - Pretty JSON responses
    - Rate limiting disabled; CORS still limited to localhost unless CORS_ORIGINS is set
    - CEVAHIR_TRUST_X_USER_ID=1 additionally accepts the X-User-ID header as identity
    """
    API_ENV = "development"
    DEBUG = True
    TESTING = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    LOG_LEVEL = "DEBUG"
    
    def _secret_key(self, generic_secret):
        return os.getenv("DEVELOPMENT_SECRET_KEY", "dev-secret-key-change-in-production")

    def _ratelimit_enabled(self):
        return False  # Disable rate limiting in development


class TestingConfig(Config):
    """
    Testing environment configuration.
    
    Endüstri Standardı: Testing best practices
    - Test mode enabled
    - Minimal logging
    - Fast execution
    """
    API_ENV = "testing"
    DEBUG = True
    TESTING = True
    MAX_CONTENT_LENGTH = int(os.getenv("TESTING_MAX_CONTENT_LENGTH", 8 * 1024 * 1024))  # 8 MB
    LOG_LEVEL = "WARNING"
    
    # Testing-specific
    ENABLE_METRICS = False

    def _secret_key(self, generic_secret):
        return os.getenv("TESTING_SECRET_KEY", "test-secret-key")

    def _ratelimit_enabled(self):
        return False


class ProductionConfig(Config):
    """
    Production environment configuration.
    
    Endüstri Standardı: Production best practices
    - Debug mode disabled
    - Security hardened: real, distinct PRODUCTION_SECRET_KEY and JWT_SECRET_KEY
      values are required when the configuration is loaded
    - Rate limiting always enabled; no cross-origin access unless CORS_ORIGINS lists it
    - Performance optimized
    - Error-only logging
    """
    API_ENV = "production"
    DEBUG = False
    TESTING = False
    JSONIFY_PRETTYPRINT_REGULAR = False
    LOG_LEVEL = "ERROR"
    
    # Production-specific
    cors_origins_default = ()
    ENABLE_SECURITY_HEADERS = True
    
    def _secret_key(self, generic_secret):
        return os.getenv("PRODUCTION_SECRET_KEY")

    def _jwt_secret_key(self, generic_secret):
        return os.getenv("JWT_SECRET_KEY")  # no fallback to SECRET_KEY or FLASK_SECRET_KEY

    def _ratelimit_enabled(self):
        return True

    def _validate(self, problems):
        secrets = _secret_problems("PRODUCTION_SECRET_KEY", self.SECRET_KEY)
        secrets += _secret_problems("JWT_SECRET_KEY", self.JWT_SECRET_KEY)
        if not secrets and self.SECRET_KEY == self.JWT_SECRET_KEY:
            secrets.append("JWT_SECRET_KEY must differ from PRODUCTION_SECRET_KEY")
        if secrets:
            secrets[-1] += (
                f" (set both to different random values of at least {MIN_PRODUCTION_SECRET_LENGTH} "
                "characters, e.g. python -c \"import secrets; print(secrets.token_urlsafe(48))\")"
            )
        problems.extend(secrets)


_CONFIGS = {
    "": DefaultConfig,
    "default": DefaultConfig,
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config():
    """
    FLASK_ENV ortam değişkenine göre yapılandırmayı döndürür.
    Değişken yoksa güvenli varsayılan (DefaultConfig) kullanılır: debug kapalı, hız
    sınırı açık, CORS yalnız localhost kökenleri. Geliştirme modu yalnız açıkça
    FLASK_ENV=development ile seçilir; bilinmeyen değerler hata verir.
    """
    env = os.getenv("FLASK_ENV", "").strip().lower()
    if env not in _CONFIGS:
        raise ConfigurationError(
            f"Unknown FLASK_ENV {env!r}: use development, testing or production, or leave it unset"
        )
    return _CONFIGS[env]()

# Importing configuration must not create log files or reset application logging.
config = get_config()
