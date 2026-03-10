# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: async_middleware.py
Modül: cognitive_management/v2/middleware
Görev: Async Middleware Support - Async middleware protocol and base implementation.
       AsyncMiddleware, BaseAsyncMiddleware ve async middleware interface tanımlarını
       içerir. Async interceptor pattern için temel yapı sağlar. Async before/after
       hook'ları ve async error handling desteği sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (async middleware base),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Async Interceptor Pattern (async middleware)
- Endüstri Standartları: Async middleware best practices

KULLANIM:
- Async middleware interface tanımları için
- Base async middleware implementation için
- Async interceptor pattern için

BAĞIMLILIKLAR:
- BaseMiddleware: Base middleware
- asyncio: Async işlemler
- CognitiveTypes: Tip tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Protocol, Optional
from abc import ABC, abstractmethod
import asyncio

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
)
from .base import Middleware, BaseMiddleware


class AsyncMiddleware(Protocol):
    """
    Async middleware interface.
    Async interceptor pattern için.
    """
    
    async def before_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """
        Request processing öncesi (async).
        
        Args:
            state: Cognitive state
            request: Cognitive input
            
        Returns:
            Modified (state, request)
        """
        ...
    
    async def after_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """
        Request processing sonrası (async).
        
        Args:
            state: Cognitive state
            request: Cognitive input
            response: Cognitive output
            
        Returns:
            Modified response
        """
        ...
    
    async def on_error_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """
        Error handling (async).
        
        Args:
            state: Cognitive state
            request: Cognitive input
            error: Exception
            
        Returns:
            Error response (None = propagate error)
        """
        ...


class BaseAsyncMiddleware(ABC):
    """
    Base async middleware implementation.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._next: Optional[AsyncMiddleware] = None
    
    def set_next(self, middleware: AsyncMiddleware) -> None:
        """Set next middleware in chain"""
        self._next = middleware
    
    async def before_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Before processing (async)"""
        state, request = await self._before_async(state, request)
        if self._next:
            return await self._next.before_async(state, request)
        return state, request
    
    async def after_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """After processing (async)"""
        response = await self._after_async(state, request, response)
        if self._next:
            return await self._next.after_async(state, request, response)
        return response
    
    async def on_error_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """Error handling (async)"""
        result = await self._on_error_async(state, request, error)
        if result is not None:
            return result
        if self._next:
            return await self._next.on_error_async(state, request, error)
        return None
    
    @abstractmethod
    async def _before_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Override this"""
        return state, request
    
    async def _after_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """Override this"""
        return response
    
    async def _on_error_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """Override this"""
        return None


class SyncToAsyncMiddlewareAdapter(BaseAsyncMiddleware):
    """
    Adapter to convert sync middleware to async.
    Wraps sync middleware methods in async.
    """
    
    def __init__(self, sync_middleware: Middleware):
        super().__init__(f"AsyncAdapter({getattr(sync_middleware, 'name', 'Unknown')})")
        self.sync_middleware = sync_middleware
    
    async def _before_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Wrap sync before in async"""
        return await asyncio.to_thread(
            self.sync_middleware.before,
            state,
            request,
        )
    
    async def _after_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """Wrap sync after in async"""
        return await asyncio.to_thread(
            self.sync_middleware.after,
            state,
            request,
            response,
        )
    
    async def _on_error_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """Wrap sync on_error in async"""
        return await asyncio.to_thread(
            self.sync_middleware.on_error,
            state,
            request,
            error,
        )


__all__ = [
    "AsyncMiddleware",
    "BaseAsyncMiddleware",
    "SyncToAsyncMiddlewareAdapter",
]
