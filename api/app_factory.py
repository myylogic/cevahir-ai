# -*- coding: utf-8 -*-
"""
Flask App Factory - Complete System Integration
===============================================

Entegre edilen modüller:
- Database (PostgreSQL/SQLite)
- Cevahir (Model + Tokenizer + Cognitive)
- ChattingManagement (User/Session/Conversation)
- API Routes (v3)
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Proje kök dizinini ekle
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Config imports
from api.api_config import get_config
from config.parameters import DEVICE, LOGGING_PATH

# Database imports
from database import DatabaseConnection, UnitOfWork
from database.config import DatabaseConfig

# Cevahir imports
from model.cevahir import Cevahir, CevahirConfig

# ChattingManagement imports
from chatting_management import ChattingManager, ChattingConfig

# API imports
from api.middleware.error_handler import register_error_handlers
from api.middleware.security import register_security_middleware
from api.middleware.request_id import register_request_id_middleware
from api.monitoring.health import register_health_checks
from api.monitoring.metrics import register_metrics
from api.routes.v3 import v3_bp
from api.routes.v3.chat import init_chat_routes
from api.routes.v3.sessions import init_session_routes
from api.routes.v3.users import init_user_routes
from api.services import ChatService, SessionService, UserService

logger = logging.getLogger(__name__)


def create_cevahir_instance() -> Cevahir:
    """
    Cevahir instance oluştur.
    
    Returns:
        Cevahir instance
    
    Raises:
        Exception: If initialization fails
    """
    logger.info("=" * 60)
    logger.info("INITIALIZING CEVAHIR")
    logger.info("=" * 60)
    
    try:
        # Cevahir config
        cevahir_config = CevahirConfig(
            device=os.getenv("CEVAHIR_DEVICE", DEVICE),
            seed=42,
            log_level=os.getenv("CEVAHIR_LOG_LEVEL", "INFO"),
            
            # Tokenizer config
            tokenizer={
                "vocab_path": os.getenv(
                    "CEVAHIR_VOCAB_PATH",
                    "data/vocab_lib/vocab.json"
                ),
                "merges_path": os.getenv(
                    "CEVAHIR_MERGES_PATH",
                    "data/merges_lib/merges.txt"
                ),
                "data_dir": None,  # Inference için gerekli değil
                "use_gpu": DEVICE == "cuda",
                "batch_size": 32,
                "max_unk_ratio": 0.01,
            },
            
            # Model config (training ile uyumlu olmalı: 48 layers)
            model={
                "vocab_size": 60000,
                "embed_dim": 1024,
                "num_heads": 8,  # ✅ Training ile uyumlu (8 heads - Colab crash fix)
                "num_layers": 24,  # ✅ Training ile uyumlu (24 layers)
                "ff_dim": 4096,
                "max_seq_length": 512,
                "dropout": 0.1,
                "use_rmsnorm": True,
                "use_swiglu": True,
                "use_kv_cache": True,
                "max_cache_len": 2048,
            },
            
            # Model loading
            load_model_path=os.getenv("CEVAHIR_MODEL_PATH", None),  # None = auto-detect
        )
        
        # Cevahir instance oluştur
        cevahir = Cevahir(cevahir_config)
        
        logger.info("✅ Cevahir initialized successfully")
        return cevahir
        
    except Exception as e:
        logger.error(f"❌ Cevahir initialization failed: {e}", exc_info=True)
        raise


def create_chatting_manager(cevahir: Cevahir) -> ChattingManager:
    """
    ChattingManager instance oluştur.
    
    Args:
        cevahir: Cevahir instance
    
    Returns:
        ChattingManager instance
    
    Raises:
        Exception: If initialization fails
    """
    logger.info("=" * 60)
    logger.info("INITIALIZING CHATTING MANAGEMENT")
    logger.info("=" * 60)
    
    try:
        # ChattingManagement config
        chatting_config = ChattingConfig(
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "450")),
            max_recent_messages=int(os.getenv("MAX_RECENT_MESSAGES", "10")),
            enable_user_memory=os.getenv("ENABLE_USER_MEMORY", "True").lower() == "true",
            enable_semantic_search=os.getenv("ENABLE_SEMANTIC_SEARCH", "True").lower() == "true",
            default_max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "256")),
        )
        
        # ChattingManager oluştur
        chatting_manager = ChattingManager(
            config=chatting_config,
            cevahir=cevahir
        )
        
        logger.info("✅ ChattingManager initialized successfully")
        return chatting_manager
        
    except Exception as e:
        logger.error(f"❌ ChattingManager initialization failed: {e}", exc_info=True)
        raise


def initialize_database():
    """
    Database bağlantısını başlat ve test et.
    
    Raises:
        Exception: If database connection fails
    """
    logger.info("=" * 60)
    logger.info("INITIALIZING DATABASE")
    logger.info("=" * 60)
    
    try:
        # Database config
        db_config = DatabaseConfig()
        
        # Database connection (singleton pattern - otomatik bağlanır)
        db = DatabaseConnection()
        
        # Test connection
        with UnitOfWork() as uow:
            # Simple test query
            logger.info("Testing database connection...")
            # Connection başarılı ise devam eder
        
        logger.info(f"✅ Database initialized: {db_config.db_type}")
        logger.info(f"   Host: {db_config.postgres_host if db_config.db_type == 'postgresql' else 'SQLite'}")
        logger.info(f"   Database: {db_config.postgres_db if db_config.db_type == 'postgresql' else db_config.sqlite_path}")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}", exc_info=True)
        raise


def create_app() -> Flask:
    """
    Flask uygulamasını oluştur ve tüm modülleri entegre et.
    
    Returns:
        Flask application instance
    """
    logger.info("=" * 80)
    logger.info("CREATING FLASK APPLICATION")
    logger.info("=" * 80)
    
    # Flask app oluştur
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    
    # Config yükle
    config = get_config()
    app.config.from_object(config)
    
    # CORS (config-based)
    cors_origins = app.config.get("CORS_ORIGINS", ["*"])
    cors_methods = app.config.get("CORS_METHODS", ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    cors_headers = app.config.get("CORS_HEADERS", ["Content-Type", "Authorization", "X-User-ID", "X-Request-ID"])
    
    CORS(
        app,
        resources={r"/*": {
            "origins": cors_origins,
            "methods": cors_methods,
            "allow_headers": cors_headers,
            "expose_headers": ["X-Request-ID"]
        }}
    )
    logger.info(f"✅ CORS configured: origins={cors_origins}")
    
    # Rate Limiting (config-based)
    if app.config.get("RATELIMIT_ENABLED", True):
        limiter = Limiter(
            get_remote_address,
            default_limits=app.config.get("RATELIMIT_DEFAULT", "200 per day, 50 per hour").split(", "),
            storage_uri=app.config.get("RATELIMIT_STORAGE_URI", "memory://")
        )
        limiter.init_app(app)
        logger.info("✅ Rate limiting enabled")
    else:
        logger.info("⚠️ Rate limiting disabled")
    
    # ========================================================================
    # 1. DATABASE INITIALIZATION
    # ========================================================================
    try:
        initialize_database()
    except Exception as e:
        logger.warning(f"⚠️ Database initialization failed: {e}")
        logger.warning("   Continuing without database (some features may not work)")
    
    # ========================================================================
    # 2. CEVAHIR INITIALIZATION
    # ========================================================================
    try:
        cevahir = create_cevahir_instance()
        app.cevahir = cevahir  # Flask app'e attach et
        logger.info("✅ Cevahir attached to Flask app")
    except Exception as e:
        logger.error(f"❌ Cevahir initialization failed: {e}")
        logger.error("   Application cannot start without Cevahir")
        raise
    
    # ========================================================================
    # 3. CHATTING MANAGEMENT INITIALIZATION
    # ========================================================================
    try:
        chatting_manager = create_chatting_manager(cevahir)
        app.chatting_manager = chatting_manager  # Flask app'e attach et
        logger.info("✅ ChattingManager attached to Flask app")
    except Exception as e:
        logger.error(f"❌ ChattingManager initialization failed: {e}")
        logger.error("   Application cannot start without ChattingManager")
        raise
    
    # ========================================================================
    # 4. SERVICE LAYER INITIALIZATION
    # ========================================================================
    try:
        chat_service = ChatService(chatting_manager)
        session_service = SessionService(chatting_manager)
        user_service = UserService()
        
        app.chat_service = chat_service
        app.session_service = session_service
        app.user_service = user_service
        
        logger.info("✅ Services initialized and attached to Flask app")
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}", exc_info=True)
        raise
    
    # ========================================================================
    # 5. MIDDLEWARE REGISTRATION
    # ========================================================================
    # Error handlers
    register_error_handlers(app)
    logger.info("✅ Error handlers registered")
    
    # Security middleware
    register_security_middleware(app)
    logger.info("✅ Security middleware registered")
    
    # Request ID middleware
    register_request_id_middleware(app)
    logger.info("✅ Request ID middleware registered")
    
    # ========================================================================
    # 6. MONITORING & OBSERVABILITY
    # ========================================================================
    # Health checks
    register_health_checks(app)
    logger.info("✅ Health checks registered")
    
    # Metrics collection
    if app.config.get("ENABLE_METRICS", True):
        register_metrics(app)
        logger.info("✅ Metrics collection registered")
    
    # ========================================================================
    # 7. API ROUTES (v3)
    # ========================================================================
    try:
        # Register v3 blueprint
        app.register_blueprint(v3_bp)
        
        # Initialize routes with services
        init_chat_routes(chat_service)
        init_session_routes(session_service)
        init_user_routes(user_service)
        
        logger.info("✅ API routes (v3) registered")
    except Exception as e:
        logger.error(f"❌ Route registration failed: {e}", exc_info=True)
        raise
    
    # ========================================================================
    # 7. LEGACY ROUTES (Optional - deprecated)
    # ========================================================================
    # Eski route'ları isteğe bağlı olarak ekleyebilirsin
    # from api.routes import register_blueprints
    # register_blueprints(app)  # Eski route'lar
    
    # ========================================================================
    # 8. HEALTH CHECK ENDPOINT (Root level)
    # ========================================================================
    @app.route('/')
    def home():
        """Ana sayfa"""
        return {
            "message": "Cevahir API v3",
            "status": "running",
            "endpoints": {
                "health": "/api/v3/health",
                "chat": "/api/v3/chat/messages",
                "sessions": "/api/v3/sessions",
                "users": "/api/v3/users/me"
            }
        }
    
    logger.info("=" * 80)
    logger.info("✅ FLASK APPLICATION CREATED SUCCESSFULLY")
    logger.info("=" * 80)
    
    return app


def setup_logging():
    """Loglama yapılandırması"""
    if not os.path.exists(LOGGING_PATH):
        os.makedirs(LOGGING_PATH)
    
    log_file = os.path.join(LOGGING_PATH, 'app_process.log')
    handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    logging.info("Logging configured")


if __name__ == '__main__':
    setup_logging()
    
    PORT = int(os.getenv("PORT", 5000))
    app = create_app()
    
    logging.info(f"Starting Flask application on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=True)

