# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: memory_storage.py
Modül: chatting_management/storage
Görev: Memory Storage - User memory persistence using database module. SOLID Principle:
       Dependency Inversion. User memory storage, memory persistence, memory retrieval,
       memory update ve memory deletion işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Dependency Inversion (database interface'e bağımlı)
- Design Patterns: Storage Pattern (memory storage)
- Endüstri Standartları: Data persistence best practices

KULLANIM:
- User memory storage için
- Memory persistence için
- Memory retrieval için

BAĞIMLILIKLAR:
- database.UnitOfWork: Database unit of work
- database.models: Database models

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
from typing import List, Optional, Dict, Any
from database import UnitOfWork
from database.models import UserMemory
from database.utils.helpers import generate_uuid

logger = logging.getLogger(__name__)


class MemoryStorage:
    """
    User memory storage using database module.
    
    SOLID Principle: Dependency Inversion
    """
    
    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        priority: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[UserMemory]:
        """
        Get user memories with optional filters.
        
        Args:
            user_id: User ID
            memory_type: Filter by memory type (optional)
            priority: Filter by priority (optional)
            limit: Maximum number of memories
            
        Returns:
            List of user memories
        """
        try:
            with UnitOfWork() as uow:
                return uow.user_memories.get_by_user_id(
                    user_id=user_id,
                    memory_type=memory_type,
                    priority=priority,
                    limit=limit
                )
        except Exception as e:
            logger.error(f"Error getting user memories: {e}", exc_info=True)
            raise
    
    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        priority: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserMemory:
        """
        Add user memory.
        
        Args:
            user_id: User ID
            memory_type: Memory type ('fact', 'preference', 'pattern', 'relationship', 'goal')
            content: Memory content
            priority: Priority ('high', 'medium', 'low')
            metadata: Memory metadata (optional)
            
        Returns:
            Created UserMemory instance
        """
        try:
            memory_id = generate_uuid()
            memory = UserMemory(
                memory_id=memory_id,
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                priority=priority,
                metadata=metadata or {}
            )
            
            with UnitOfWork() as uow:
                created_memory = uow.user_memories.create(memory)
                logger.debug(f"Memory added: {memory_id} ({memory_type}) for user {user_id}")
                return created_memory
        except Exception as e:
            logger.error(f"Error adding memory: {e}", exc_info=True)
            raise
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        priority: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserMemory:
        """
        Update user memory.
        
        Args:
            memory_id: Memory ID
            content: New content (optional)
            priority: New priority (optional)
            metadata: New metadata (optional)
            
        Returns:
            Updated UserMemory instance
        """
        try:
            with UnitOfWork() as uow:
                memory = uow.user_memories.get_by_id(memory_id)
                if memory is None:
                    raise ValueError(f"Memory not found: {memory_id}")
                
                if content is not None:
                    memory.content = content
                if priority is not None:
                    memory.priority = priority
                if metadata is not None:
                    # Merge metadata
                    current_metadata = memory.metadata or {}
                    current_metadata.update(metadata)
                    memory.metadata = current_metadata
                
                updated_memory = uow.user_memories.update(memory)
                logger.debug(f"Memory updated: {memory_id}")
                return updated_memory
        except Exception as e:
            logger.error(f"Error updating memory: {e}", exc_info=True)
            raise
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete user memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            with UnitOfWork() as uow:
                deleted = uow.user_memories.delete(memory_id)
                if deleted:
                    logger.debug(f"Memory deleted: {memory_id}")
                return deleted
        except Exception as e:
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            raise

