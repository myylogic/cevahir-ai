# -*- coding: utf-8 -*-
"""
Chatting Management Module
==========================

Endüstri Standartları:
- SOLID Principles (Repository Pattern, Dependency Inversion)
- Clean Architecture (Layered design)
- Enterprise Features (Error handling, logging, monitoring)
- Database Integration (UnitOfWork pattern)

Mimari:
- ChattingManager: Ana koordinatör
- Components: UserManager, SessionManager, ConversationManager, ContextBuilder
- Storage: Database persistence layer
- Integrations: Cevahir API integration

Cevahir.py Entegrasyonu:
- Cevahir.process(text, state=CognitiveState) → CognitiveOutput
- ChattingManager context hazırlar, Cevahir inference yapar
"""

from chatting_management.chatting_manager import ChattingManager
from chatting_management.config import ChattingConfig
from chatting_management.exceptions import (
    ChattingManagementError,
    UserNotFoundError,
    SessionNotFoundError,
    SessionAccessDeniedError,
    InvalidMessageError,
    ContextBuildingError,
    CevahirIntegrationError,
    ConversationError,
)

__all__ = [
    "ChattingManager",
    "ChattingConfig",
    "ChattingManagementError",
    "UserNotFoundError",
    "SessionNotFoundError",
    "SessionAccessDeniedError",
    "InvalidMessageError",
    "ContextBuildingError",
    "CevahirIntegrationError",
    "ConversationError",
]

__version__ = "1.0.0"

