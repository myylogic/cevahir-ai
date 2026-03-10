# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: validation.py
Modül: cognitive_management/v2/middleware
Görev: Validation Middleware - Input validation, type safety, data sanitization.
       ValidationMiddleware sınıfını içerir. Type validation, range validation,
       content validation ve sanitization işlemlerini yapar. Input validation
       ve data sanitization sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (validation middleware),
                     Dependency Inversion (BaseMiddleware interface'e bağımlı)
- Design Patterns: Middleware Pattern (validation middleware)
- Endüstri Standartları: Input validation best practices

KULLANIM:
- Input validation için
- Type safety için
- Data sanitization için

BAĞIMLILIKLAR:
- BaseMiddleware: Base middleware
- ValidationError: Exception tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Optional

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
)
from cognitive_management.exceptions import ValidationError
from .base import BaseMiddleware


class ValidationMiddleware(BaseMiddleware):
    """
    Input validation middleware.
    - Type validation
    - Range validation
    - Content validation
    - Sanitization
    """
    
    def __init__(
        self,
        max_message_length: int = 10000,
        min_message_length: int = 1,
    ):
        super().__init__("Validation")
        self.max_message_length = max_message_length
        self.min_message_length = min_message_length
    
    def _before(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Validate input before processing"""
        # Validate request
        if not isinstance(request, CognitiveInput):
            raise ValidationError("request must be CognitiveInput instance")
        
        # Validate user message
        user_message = request.user_message or ""
        user_message = user_message.strip()
        
        if len(user_message) < self.min_message_length:
            raise ValidationError(
                f"User message too short (min: {self.min_message_length})",
                details={"length": len(user_message)}
            )
        
        if len(user_message) > self.max_message_length:
            raise ValidationError(
                f"User message too long (max: {self.max_message_length})",
                details={"length": len(user_message)}
            )
        
        # Validate state
        if not isinstance(state, CognitiveState):
            raise ValidationError("state must be CognitiveState instance")
        
        # Sanitize (basic)
        user_message = self._sanitize(user_message)
        request.user_message = user_message
        
        return state, request
    
    def _sanitize(self, text: str) -> str:
        """Basic text sanitization"""
        # Remove control characters (except newlines and tabs)
        import re
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text


__all__ = ["ValidationMiddleware"]

