# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: config.py
Modül: chatting_management
Görev: Chatting Management Configuration - Chatting management için yapılandırma.
       Endüstri Standardı: 12-Factor App (config via environment). ChattingConfig
       dataclass'ını içerir. Environment variable based configuration, type-safe
       configuration ve validation on initialization sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (config yönetimi)
- Design Patterns: Config Pattern (yapılandırma yönetimi)
- Endüstri Standartları: 12-Factor App (config via environment)

KULLANIM:
- Config yönetimi için
- Environment variable configuration için
- Type-safe configuration için

BAĞIMLILIKLAR:
- dataclasses: Dataclass tanımları
- os: Environment variables

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ChattingConfig:
    """
    Chatting Management configuration.
    
    Endüstri Standardı:
    - Environment variable based configuration
    - Type-safe configuration
    - Validation on initialization
    """
    
    # Context Management
    max_context_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "450")))  # 512 - buffer
    max_recent_messages: int = field(default_factory=lambda: int(os.getenv("MAX_RECENT_MESSAGES", "10")))
    enable_semantic_search: bool = field(default_factory=lambda: os.getenv("ENABLE_SEMANTIC_SEARCH", "True").lower() == "true")
    semantic_search_top_k: int = field(default_factory=lambda: int(os.getenv("SEMANTIC_SEARCH_TOP_K", "5")))
    
    # Session Management
    session_timeout_hours: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_HOURS", "24")))
    max_sessions_per_user: int = field(default_factory=lambda: int(os.getenv("MAX_SESSIONS_PER_USER", "100")))
    auto_cleanup_inactive_days: int = field(default_factory=lambda: int(os.getenv("AUTO_CLEANUP_INACTIVE_DAYS", "30")))
    
    # Message Processing
    max_message_length: int = field(default_factory=lambda: int(os.getenv("MAX_MESSAGE_LENGTH", "2000")))
    min_message_length: int = field(default_factory=lambda: int(os.getenv("MIN_MESSAGE_LENGTH", "1")))
    enable_message_preprocessing: bool = field(default_factory=lambda: os.getenv("ENABLE_MESSAGE_PREPROCESSING", "True").lower() == "true")
    
    # Response Generation
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "256")))
    default_temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.7")))
    default_top_p: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TOP_P", "0.9")))
    enable_streaming: bool = field(default_factory=lambda: os.getenv("ENABLE_STREAMING", "False").lower() == "true")
    
    # User Memory
    enable_user_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_USER_MEMORY", "True").lower() == "true")
    memory_retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("MEMORY_RETRIEVAL_TOP_K", "5")))
    
    # Performance
    enable_caching: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHING", "True").lower() == "true")
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("CHATTING_LOG_LEVEL", "INFO"))
    enable_request_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_REQUEST_LOGGING", "True").lower() == "true")
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.max_context_tokens < 1:
            raise ValueError(f"max_context_tokens must be >= 1, got {self.max_context_tokens}")
        
        if self.max_recent_messages < 1:
            raise ValueError(f"max_recent_messages must be >= 1, got {self.max_recent_messages}")
        
        if not (0.0 <= self.default_temperature <= 2.0):
            raise ValueError(f"default_temperature must be in [0, 2], got {self.default_temperature}")
        
        if not (0.0 <= self.default_top_p <= 1.0):
            raise ValueError(f"default_top_p must be in [0, 1], got {self.default_top_p}")
        
        if self.max_message_length < self.min_message_length:
            raise ValueError(f"max_message_length ({self.max_message_length}) must be >= min_message_length ({self.min_message_length})")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            "max_context_tokens": self.max_context_tokens,
            "max_recent_messages": self.max_recent_messages,
            "enable_semantic_search": self.enable_semantic_search,
            "session_timeout_hours": self.session_timeout_hours,
            "max_message_length": self.max_message_length,
            "default_max_tokens": self.default_max_tokens,
            "enable_user_memory": self.enable_user_memory,
            "enable_caching": self.enable_caching,
        }

