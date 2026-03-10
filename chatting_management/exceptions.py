# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: exceptions.py
Modül: chatting_management
Görev: Chatting Management Exceptions - Custom exceptions for chatting management
       operations. Endüstri Standardı: Specific, actionable error messages.
       ChattingManagementError, UserNotFoundError, SessionNotFoundError,
       SessionAccessDeniedError, ConversationError, ContextBuildingError ve
       diğer özel exception sınıflarını içerir.

MİMARİ:
- SOLID Prensipleri: Exception hierarchy (istisna hiyerarşisi)
- Design Patterns: Exception Pattern (özel istisnalar)
- Endüstri Standartları: Exception handling best practices

KULLANIM:
- Özel exception tanımları için
- Hata sınıflandırması için
- Specific, actionable error messages için

BAĞIMLILIKLAR:
- Modül içi bağımlılıklar

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""


class ChattingManagementError(Exception):
    """Base exception for chatting management operations"""
    pass


class UserNotFoundError(ChattingManagementError):
    """User not found in database"""
    pass


class SessionNotFoundError(ChattingManagementError):
    """Session not found in database"""
    pass


class SessionAccessDeniedError(ChattingManagementError):
    """User does not have access to this session"""
    pass


class InvalidMessageError(ChattingManagementError):
    """Invalid message format or content"""
    pass


class ContextBuildingError(ChattingManagementError):
    """Error building conversation context"""
    pass


class CevahirIntegrationError(ChattingManagementError):
    """Error in Cevahir API integration"""
    pass


class ConversationError(ChattingManagementError):
    """Error in conversation management"""
    pass

