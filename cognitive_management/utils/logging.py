# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: logging.py
Modül: cognitive_management/utils
Görev: Cognitive Logging - Üretim kullanımı için hafif ama güçlü bir log yardımcı
       modülü. Standart logging üzerine kurulu, bağımlılık yok. JSON ve satır-içi
       (human-readable) format desteği. contextvars ile istek/oturum bağlamsal
       alanları (trace_id, session_id vb.). StructuredAdapter, log_event() ve
       exception_to_dict() fonksiyonları sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (logging yönetimi)
- Design Patterns: Logger Pattern (logging yönetimi)
- Endüstri Standartları: Logging best practices

KULLANIM:
- Logging işlemleri için
- Structured logging için
- Context-aware logging için

BAĞIMLILIKLAR:
- logging: Python logging modülü
- contextvars: Context variables

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional, Mapping
from contextvars import ContextVar


# ==== Bağlamsal Alanlar (istek/oturum) =======================================

_ctx_fields: ContextVar[Dict[str, Any]] = ContextVar("_ctx_fields", default={})

def set_context(**kwargs: Any) -> None:
    """Bağlam alanlarını günceller (ör. trace_id, session_id, user_id)."""
    cur = dict(_ctx_fields.get())
    cur.update({k: v for k, v in kwargs.items() if v is not None})
    _ctx_fields.set(cur)

def clear_context(keys: Optional[list[str]] = None) -> None:
    """Bağlam alanlarını temizler; keys=None ise tümünü siler."""
    if keys is None:
        _ctx_fields.set({})
    else:
        cur = dict(_ctx_fields.get())
        for k in keys:
            cur.pop(k, None)
        _ctx_fields.set(cur)

def get_context() -> Dict[str, Any]:
    """Geçerli bağlam alanlarını döndürür."""
    return dict(_ctx_fields.get())


# ==== Formatter'lar ===========================================================

class JsonFormatter(logging.Formatter):
    """Yapılandırılmış JSON log formatter'ı."""
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Ek alanlar (extra) ve bağlam
        payload.update(getattr(record, "extra_fields", {}) or {})
        ctx = get_context()
        if ctx:
            payload.update(ctx)
        # Hata bilgisi
        if record.exc_info:
            payload["exception"] = _exc_info_to_dict(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

class LineFormatter(logging.Formatter):
    """İnsan okunabilir, tek satır key=value formatı."""
    default_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = "%Y-%m-%d %H:%M:%S"):
        super().__init__(fmt or self.default_fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        parts = [base]
        # bağlam ve ekstra alanları key=value olarak ekle
        ctx = get_context()
        extra_fields = getattr(record, "extra_fields", {}) or {}
        for k, v in {**ctx, **extra_fields}.items():
            parts.append(f"{k}={v}")
        # istisna varsa tek satır özeti
        if record.exc_info:
            exd = _exc_info_to_dict(record.exc_info)
            parts.append(f"exc_type={exd.get('type')} exc_msg={exd.get('message')}")
        return " | ".join(parts)


# ==== LoggerAdapter (structured) ==============================================

class StructuredAdapter(logging.LoggerAdapter):
    """
    logging.LoggerAdapter üzerine kuruludur; extra alanlarını 'extra_fields'
    anahtarına koyar, formatter bunları işler.
    """
    def process(self, msg: Any, kwargs: Dict[str, Any]):
        extra = kwargs.get("extra", {})
        extra_fields = dict(self.extra or {})
        # çağrıdaki 'extra_fields' önceliklidir
        passed = extra.get("extra_fields", {})
        if passed:
            extra_fields.update(passed)
        extra["extra_fields"] = extra_fields
        kwargs["extra"] = extra
        return msg, kwargs

    def event(self, event_name: str, **fields: Any) -> None:
        """Kısa olay kaydı: logger.event("decode_start", batch=4, device="cuda")"""
        self.info(event_name, extra={"extra_fields": fields})


# ==== Yardımcı Fonksiyonlar ===================================================

_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def parse_level(level: Any, default: int = logging.INFO) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return _LEVEL_MAP.get(level.strip().upper(), default)
    return default

def setup_logging(
    *,
    level: Any = None,
    json_output: Optional[bool] = None,
    log_to_file: Optional[str] = None,
    propagate: bool = False,
    root_logger_name: str = "",
) -> None:
    """
    Ana logging kurulumunu yapar. Tekrarlı çağrılarda idempotent davranır.
    Ortam değişkenleri:
      CM_LOG_LEVEL=INFO|DEBUG|...
      CM_LOG_JSON=1|0
    """
    env = os.environ
    if level is None:
        level = parse_level(env.get("CM_LOG_LEVEL", "INFO"))
    else:
        level = parse_level(level)

    if json_output is None:
        json_output = env.get("CM_LOG_JSON", "0").strip() in {"1", "true", "TRUE", "on", "ON"}

    logger = logging.getLogger(root_logger_name)
    # Mevcut handler'ları temizle (aynı akış iki kez loglanmasın)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.setLevel(level)
    logger.propagate = propagate

    formatter = JsonFormatter() if json_output else LineFormatter()

    # Konsol handler
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Dosya handler (opsiyonel)
    if log_to_file:
        fh = logging.FileHandler(log_to_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def get_logger(name: str, *, base_fields: Optional[Mapping[str, Any]] = None) -> StructuredAdapter:
    """
    Adapte logger döner; base_fields her log kaydına eklenecek sabit alanlardır.
    """
    lg = logging.getLogger(name)
    return StructuredAdapter(lg, dict(base_fields or {}))

def log_event(event: str, **fields: Any) -> None:
    """
    Hızlı tek satırlık olay kaydı (root logger üzerinden).
    Örn: log_event("cognitive_decision", mode="think1", temperature=0.7)
    """
    logger = get_logger("")
    logger.event(event, **fields)

def exception_to_dict(exc: BaseException) -> Dict[str, Any]:
    """İstisnayı güvenli bir sözlüğe çevirir."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }

def _exc_info_to_dict(exc_info) -> Dict[str, Any]:
    etype, evalue, etb = exc_info
    return {
        "type": getattr(etype, "__name__", str(etype)),
        "message": str(evalue),
        "traceback": "".join(traceback.format_exception(etype, evalue, etb)),
    }


# ==== Modül başlatımında güvenli varsayılan kurulum ==========================

# Eğer dışarıdan özel bir kurulum yapılmadıysa, makul bir varsayılan kur.
if not logging.getLogger("").handlers:
    try:
        setup_logging(level=os.environ.get("CM_LOG_LEVEL", "INFO"),
                      json_output=os.environ.get("CM_LOG_JSON", "0") in {"1", "true", "TRUE", "on", "ON"})
    except Exception:
        # Logging kurulumu asla uygulamayı düşürmemeli.
        pass


__all__ = [
    "set_context",
    "clear_context",
    "get_context",
    "JsonFormatter",
    "LineFormatter",
    "StructuredAdapter",
    "parse_level",
    "setup_logging",
    "get_logger",
    "log_event",
    "exception_to_dict",
]
