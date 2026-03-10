# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: exceptions.py
Modül: cognitive_management
Görev: Cognitive Exceptions - Bilişsel yönetim katmanında kullanılacak özel istisnalar.
       Hataları sınıflandırıp tanı koymayı kolaylaştırmak, log/telemetri ve testlerde
       net ayrımlar yapabilmek için. CognitiveError, PolicyRoutingError, DeliberationError,
       MemoryError, ValidationError ve diğer özel exception sınıflarını içerir.

MİMARİ:
- SOLID Prensipleri: Exception hierarchy (istisna hiyerarşisi)
- Design Patterns: Exception Pattern (özel istisnalar)
- Endüstri Standartları: Exception handling best practices

KULLANIM:
- Özel exception tanımları için
- Hata sınıflandırması için
- Log/telemetri için

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

from __future__ import annotations
from typing import Any, Dict, Optional


class CognitiveError(Exception):
    """
    Tüm bilişsel katman hatalarının taban sınıfı.
    Ek bağlam taşıyabilmesi için 'details' ve 'cause' alanlarını destekler.
    """

    def __init__(
        self,
        message: str = "",
        *,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: Dict[str, Any] = details or {}
        self.cause: Optional[BaseException] = cause

    def __str__(self) -> str:
        base = self.message or self.__class__.__name__
        if self.details:
            base += f" | details={self.details}"
        if self.cause:
            base += f" | cause={type(self.cause).__name__}: {self.cause}"
        return base


# ==== Yapılandırma / Arayüz Hataları ========================================

class ConfigError(CognitiveError):
    """Geçersiz/eksik konfigürasyon veya çelişkili ayarlar."""


class ModelInterfaceError(CognitiveError):
    """ModelManager arayüzü beklenen metotları sağlamıyor veya hatalı davranıyor."""


class ValidationError(CognitiveError):
    """Girdi/çıktı doğrulama başarısızlığı (tip, şekil, aralık vb.)."""


# ==== Politika / Akış Yönlendirme ===========================================

class PolicyRoutingError(CognitiveError):
    """PolicyRouter içinde strateji seçiminde/özellik çıkarımında hata."""


class ToolPolicyError(CognitiveError):
    """Araç (tool) seçimi/izinleriyle ilgili hatalar."""


# ==== İç Ses (Deliberation) / Seçici ========================================

class DeliberationError(CognitiveError):
    """İç ses üretimi veya aday düşünceler oluşturulurken hata."""


class SelectionError(CognitiveError):
    """Üretilen adaylar arasından seçim yapılırken hata."""


# ==== Eleştirmen (Critic) ====================================================

class CriticError(CognitiveError):
    """Critic değerlendirmesi veya revizyon sürecinde hata."""


# ==== Bellek / Bağlam ========================================================

class MemoryError(CognitiveError):
    """Bellek servisi (özetleme, ekleme, geri çağırma) ile ilgili hata."""


class ContextBuildError(CognitiveError):
    """Bağlamın (context) kurulması veya kırpılması sırasında hata."""


# ==== Zaman Aşımı / Performans ==============================================

class CognitiveTimeout(CognitiveError):
    """Aşırı gecikme veya zaman aşımı durumları."""


__all__ = [
    "CognitiveError",
    "ConfigError",
    "ModelInterfaceError",
    "ValidationError",
    "PolicyRoutingError",
    "ToolPolicyError",
    "DeliberationError",
    "SelectionError",
    "CriticError",
    "MemoryError",
    "ContextBuildError",
    "CognitiveTimeout",
]
