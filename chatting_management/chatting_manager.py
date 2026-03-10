# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: chatting_manager.py
Modül: chatting_management
Görev: Chatting Manager - Ana koordinatör: User/Session/Conversation yönetimi +
       Cevahir entegrasyonu. User, Session, Conversation ve Memory yönetimini
       koordine eder. Cevahir model entegrasyonu, context building ve response
       generation işlemlerini yapar. Clean Architecture (layered design) prensiplerine
       uygun.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (orchestration only),
                     Dependency Inversion (interface'lere bağımlı),
                     Open/Closed (component'ler ile genişletilebilir)
- Design Patterns: Manager Pattern (chatting management), Facade Pattern
- Endüstri Standartları: Clean Architecture, error handling, logging,
                         transaction management

KULLANIM:
- Chatting management orchestration için
- User/Session/Conversation yönetimi için
- Cevahir model entegrasyonu için
- Context building ve response generation için

BAĞIMLILIKLAR:
- Cevahir: Model sınıfı
- UserManager, SessionManager, ConversationManager: Component'ler
- UserStorage, SessionStorage, ConversationStorage, MemoryStorage: Storage layer
- ContextBuilder: Context building

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
from typing import Optional, Dict, Any,List
from model.cevahir import Cevahir
from cognitive_management.cognitive_types import CognitiveState, CognitiveOutput
from chatting_management.config import ChattingConfig
from chatting_management.storage import (
    UserStorage, SessionStorage, ConversationStorage, MemoryStorage
)
from chatting_management.components import (
    UserManager, SessionManager, ConversationManager, ContextBuilder
)
from chatting_management.exceptions import (
    UserNotFoundError, SessionNotFoundError, SessionAccessDeniedError,
    InvalidMessageError, CevahirIntegrationError
)

logger = logging.getLogger(__name__)


class ChattingManager:
    """
    Ana koordinatör: Chatting management operations.
    
    SOLID Principles:
    - Single Responsibility: Orchestration
    - Dependency Inversion: Depends on Cevahir interface, storage interfaces
    - Open/Closed: Extensible via components
    
    Cevahir Entegrasyonu:
    - Cevahir.process(text, state=CognitiveState) → CognitiveOutput
    - ChattingManager context hazırlar, Cevahir inference yapar
    """
    
    def __init__(
        self,
        config: ChattingConfig,
        cevahir: Cevahir
    ):
        """
        Initialize chatting manager.
        
        Args:
            config: Chatting configuration
            cevahir: Cevahir instance (shared, singleton)
        """
        self.config = config
        self.cevahir = cevahir
        
        # Initialize storage layer
        self.user_storage = UserStorage()
        self.session_storage = SessionStorage()
        self.conversation_storage = ConversationStorage()
        self.memory_storage = MemoryStorage() if config.enable_user_memory else None
        
        # Initialize components
        self.user_manager = UserManager(self.user_storage)
        self.session_manager = SessionManager(self.session_storage)
        self.conversation_manager = ConversationManager(self.conversation_storage)
        self.context_builder = ContextBuilder(
            config=config,
            conversation_storage=self.conversation_storage,
            memory_storage=self.memory_storage
        )
        
        logger.info("ChattingManager initialized")
    
    def send_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process user message and generate response.
        
        Flow:
        1. Validate session
        2. Validate message
        3. Load conversation history
        4. Build context
        5. Call Cevahir.process()
        6. Save messages to database
        7. Return response
        
        Args:
            user_id: User ID
            session_id: Session ID
            message: User message
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with:
            - response: Assistant response text
            - session_id: Session ID
            - message_id: Assistant message ID
            
        Raises:
            SessionNotFoundError: If session not found
            SessionAccessDeniedError: If user doesn't own session
            InvalidMessageError: If message is invalid
            CevahirIntegrationError: If Cevahir processing fails
        """
        try:
            # 1. Validate session
            session = self.session_manager.get_session(session_id, user_id=user_id)
            
            # 2. Validate message
            if not message or len(message.strip()) < self.config.min_message_length:
                raise InvalidMessageError("Message is too short")
            
            if len(message) > self.config.max_message_length:
                raise InvalidMessageError(f"Message exceeds maximum length ({self.config.max_message_length})")
            
            # 3. Build context
            state = self.context_builder.build_context(
                user_id=user_id,
                session_id=session_id,
                current_message=message
            )
            
            # 4. Call Cevahir.process()
            try:
                output: CognitiveOutput = self.cevahir.process(
                    text=message,
                    state=state,
                    **kwargs
                )
                response_text = output.text
            except Exception as e:
                logger.error(f"Cevahir processing error: {e}", exc_info=True)
                raise CevahirIntegrationError(f"Cevahir processing failed: {e}") from e
            
            # 5. Save messages to database
            user_message = self.conversation_manager.add_message(
                session_id=session_id,
                role="user",
                content=message,
                metadata={"cevahir_mode": output.used_mode}
            )
            
            assistant_message = self.conversation_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
                metadata={
                    "cevahir_mode": output.used_mode,
                    "tool_used": output.tool_used,
                    "revised_by_critic": output.revised_by_critic
                }
            )
            
            # 6. Update session last activity
            self.session_manager.update_last_activity(session_id)
            
            logger.info(f"Message processed: session={session_id}, user={user_id}")
            
            return {
                "response": response_text,
                "session_id": session_id,
                "message_id": assistant_message.message_id,
                "metadata": {
                    "mode": output.used_mode,
                    "tool_used": output.tool_used
                }
            }
            
        except (SessionNotFoundError, SessionAccessDeniedError, InvalidMessageError, CevahirIntegrationError):
            raise
        except Exception as e:
            logger.error(f"Error in send_message: {e}", exc_info=True)
            raise
    
    def get_conversation_history(
        self,
        session_id: str,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            session_id: Session ID
            user_id: User ID (for validation)
            limit: Maximum number of messages
            
        Returns:
            List of message dictionaries
        """
        # Validate session access
        self.session_manager.get_session(session_id, user_id=user_id)
        
        messages = self.conversation_manager.get_history(session_id, limit=limit)
        
        return [
            {
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "metadata": msg.metadata or {}
            }
            for msg in messages
        ]
    
    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create new session.
        
        Args:
            user_id: User ID
            title: Session title (optional)
            **kwargs: Additional session metadata
            
        Returns:
            Session dictionary
        """
        # Validate user exists
        self.user_manager.get_user(user_id)
        
        session = self.session_manager.create_session(
            user_id=user_id,
            title=title,
            metadata=kwargs
        )
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "title": session.title,
            "created_at": session.created_at.isoformat() if session.created_at else None
        }
    
    def list_sessions(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List user sessions.
        
        Args:
            user_id: User ID
            limit: Maximum number of sessions
            
        Returns:
            List of session dictionaries
        """
        sessions = self.session_manager.list_sessions(user_id, limit=limit)
        
        return [
            {
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "last_activity": session.last_activity.isoformat() if session.last_activity else None
            }
            for session in sessions
        ]

