# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: timers.py
Modül: cognitive_management/utils
Görev: Cognitive Timers - Zaman ölçümü ve performans telemetrisi için yardımcı
       modüller. timer() contextmanager (kod bloğunun süresini ölçer ve loglar),
       timed() dekoratör (fonksiyon süresini otomatik ölçer ve loglar), Stopwatch
       (başla/dur/sıfırla arayüzü) ve TimerStats (ad bazlı kümülatif istatistik
       toplayıcı) sağlar. Basit ve bağımsız; ek bağımlılık yok.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (zaman ölçümü)
- Design Patterns: Timer Pattern (zaman ölçümü)
- Endüstri Standartları: Performance monitoring best practices

KULLANIM:
- Zaman ölçümü için
- Performans telemetrisi için
- Timer istatistikleri için

BAĞIMLILIKLAR:
- time: Zaman ölçümü
- threading: Thread-safe işlemler
- logging: Logging işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
import time
import math
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, Tuple, TypeVar

from .logging import get_logger

_T = TypeVar("_T")


# ==== Yardımcı Fonksiyonlar ===================================================

def _now_ms() -> float:
    # time.perf_counter: yüksek çözünürlük ve monotonik sayaç
    return time.perf_counter() * 1000.0


# ==== Stopwatch ===============================================================

class Stopwatch:
    """
    Basit kronometre. Tekrarlı start/stop destekler, toplam süreyi biriktirir.
    Thread-safe değildir; tek thread içinde kullanın.
    """
    def __init__(self) -> None:
        self._running: bool = False
        self._t0_ms: float = 0.0
        self._acc_ms: float = 0.0

    def start(self) -> None:
        if not self._running:
            self._t0_ms = _now_ms()
            self._running = True

    def stop(self) -> float:
        if self._running:
            dt = _now_ms() - self._t0_ms
            self._acc_ms += dt
            self._running = False
        return self.elapsed_ms

    def reset(self) -> None:
        self._running = False
        self._t0_ms = 0.0
        self._acc_ms = 0.0

    @property
    def elapsed_ms(self) -> float:
        if self._running:
            return self._acc_ms + (_now_ms() - self._t0_ms)
        return self._acc_ms


# ==== Kümülatif İstatistikler =================================================

class TimerStats:
    """
    Ad bazlı kümülatif zaman istatistikleri.
    - count, total_ms, avg_ms, min_ms, max_ms, p95_ms (yaklaşık)
    """
    __slots__ = ("_lock", "_stats")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, float]] = {}

    def add(self, name: str, duration_ms: float) -> None:
        with self._lock:
            s = self._stats.get(name)
            if s is None:
                s = {
                    "count": 0.0,
                    "total_ms": 0.0,
                    "min_ms": float("inf"),
                    "max_ms": 0.0,
                    # P95 için tek geçişli basit bir EMA yaklaşımı
                    "ema_ms": duration_ms,
                    "p95_ms": duration_ms,
                }
                self._stats[name] = s
            s["count"] += 1.0
            s["total_ms"] += duration_ms
            s["min_ms"] = min(s["min_ms"], duration_ms)
            s["max_ms"] = max(s["max_ms"], duration_ms)
            # EMA
            alpha = 0.1
            s["ema_ms"] = (1 - alpha) * s["ema_ms"] + alpha * duration_ms
            # P95 kaba tahmin: EMA + 1.645 * std_approx (burada std_approx ~ |dur - EMA|)
            std_approx = abs(duration_ms - s["ema_ms"])
            s["p95_ms"] = s["ema_ms"] + 1.645 * std_approx

    def snapshot(self, name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        with self._lock:
            if name is None:
                # kopya döndür
                return {k: dict(v) for k, v in self._stats.items()}
            return {name: dict(self._stats.get(name, {}))}

    def summary(self, name: str) -> Dict[str, float]:
        with self._lock:
            s = self._stats.get(name)
            if not s or s["count"] <= 0:
                return {}
            avg = s["total_ms"] / max(1.0, s["count"])
            return {
                "count": s["count"],
                "total_ms": s["total_ms"],
                "avg_ms": avg,
                "min_ms": s["min_ms"],
                "max_ms": s["max_ms"],
                "p95_ms": s["p95_ms"],
            }


_GLOBAL_STATS = TimerStats()
_DEFAULT_LOGGER = get_logger("timers")


# ==== Context Manager =========================================================

@contextmanager
def timer(name: str, *, logger=None, collect_stats: bool = True) -> Generator[None, None, None]:
    """
    Kullanım:
        with timer("encode_batch"):
            run()

    Parametreler:
        name:     Zamanlayıcı adı (log ve istatistiklerde görünecek).
        logger:   Özel logger (StructuredAdapter). None ise varsayılan logger kullanılır.
        collect_stats: True ise kümülatif istatistiklere eklenir.
    """
    lg = logger or _DEFAULT_LOGGER
    t0 = _now_ms()
    try:
        yield
    finally:
        dt = _now_ms() - t0
        lg.event("timer", name=name, duration_ms=round(dt, 3))
        if collect_stats:
            _GLOBAL_STATS.add(name, dt)


# ==== Dekoratör ===============================================================

def timed(name: Optional[str] = None, *, logger=None, collect_stats: bool = True) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """
    Fonksiyon süresini otomatik ölçen dekoratör.

    @timed("forward_pass")
    def forward(...):
        ...

    Eğer name None ise fonksiyon adı kullanılır.
    """
    def _wrap(fn: Callable[..., _T]) -> Callable[..., _T]:
        nm = name or fn.__name__

        def _inner(*args, **kwargs) -> _T:
            with timer(nm, logger=logger, collect_stats=collect_stats):
                return fn(*args, **kwargs)
        _inner.__name__ = fn.__name__
        _inner.__doc__ = fn.__doc__
        _inner.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
        return _inner
    return _wrap


# ==== Yardımcı API ============================================================

def get_timer_stats(name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Kümülatif zaman istatistiklerinin anlık görüntüsünü döndürür.
    name verilirse sadece o ada ait istatistik döner.
    """
    return _GLOBAL_STATS.snapshot(name)

def summary(name: str) -> Dict[str, float]:
    """
    Belirli bir zamanlayıcı için özet istatistikleri döndürür.
    """
    return _GLOBAL_STATS.summary(name)


__all__ = [
    "Stopwatch",
    "TimerStats",
    "timer",
    "timed",
    "get_timer_stats",
    "summary",
]
