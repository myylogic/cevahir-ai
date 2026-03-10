# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: base.py
Modül: cognitive_management/v2/middleware
Görev: Base Middleware - Middleware base classes ve interfaces. Middleware,
       BaseMiddleware ve middleware interface tanımlarını içerir. Interceptor
       pattern için temel yapı sağlar. Before/after hook'ları ve error handling
       desteği sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (middleware base),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Interceptor Pattern (middleware)
- Endüstri Standartları: Middleware best practices

KULLANIM:
- Middleware interface tanımları için
- Base middleware implementation için
- Interceptor pattern için

BAĞIMLILIKLAR:
- CognitiveTypes: Tip tanımları
- abc: Abstract base classes

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Protocol, Optional, Any, Callable
from abc import ABC, abstractmethod

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
)


class Middleware(Protocol):
    """
    Middleware interface.
    Interceptor pattern için.
    """
    
    def before(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """
        Request processing öncesi.
        
        Args:
            state: Cognitive state
            request: Cognitive input
            
        Returns:
            Modified (state, request)
        """
        ...
    
    def after(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """
        Request processing sonrası.
        
        Args:
            state: Cognitive state
            request: Cognitive input
            response: Cognitive output
            
        Returns:
            Modified response
        """
        ...
    
    def on_error(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """
        Error handling.
        
        Args:
            state: Cognitive state
            request: Cognitive input
            error: Exception
            
        Returns:
            Error response (None = propagate error)
        """
        ...


class BaseMiddleware(ABC):
    """
    Base middleware implementation.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._next: Optional[Middleware] = None
    
    def set_next(self, middleware: Middleware) -> None:
        """Set next middleware in chain"""
        self._next = middleware
    
    def before(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Before processing"""
        state, request = self._before(state, request)
        if self._next:
            return self._next.before(state, request)
        return state, request
    
    def after(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """After processing"""
        response = self._after(state, request, response)
        if self._next:
            return self._next.after(state, request, response)
        return response
    
    def on_error(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """Error handling"""
        result = self._on_error(state, request, error)
        if result is not None:
            return result
        if self._next:
            return self._next.on_error(state, request, error)
        return None
    
    @abstractmethod
    def _before(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Override this"""
        return state, request
    
    def _after(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """Override this"""
        return response
    
    def _on_error(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """Override this"""
        return None


__all__ = ["Middleware", "BaseMiddleware"]

