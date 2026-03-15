# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ — Training Management V3
================================================================================
Dosya : training_management/v3/utils/training_logger.py
Modül : TrainingLogger
Görev : V3 eğitimi için yapılandırılmış logging.

Özellikler:
  • Renkli terminal çıktısı (ANSI kodlarıyla — colorama opsiyonel).
  • Her epoch için başlangıç/bitiş formatı.
  • Her N batch'te batch özeti.
  • Güvenlik olayları: NaN, spike, divergence.
  • Inference quality probe sonuçları.
  • Convergence durum değişiklikleri.
  • Yapılandırılmış JSON log satırları (her epoch bir satır, JSONL formatı).
  • Thread-safe (logging modülünün kilitlemesi ile).

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# ANSI renk kodları (colorama gerekmeden)
# ---------------------------------------------------------------------------

class _C:
    """ANSI renk kodları sabitleri."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Ön plan renkleri
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    # Arka plan
    BG_RED  = "\033[41m"
    BG_YEL  = "\033[43m"


def _colorize(text: str, *codes: str, force: bool = False) -> str:
    """
    Terminal renk desteği varsa metni renklendirir.
    Windows'ta veya pipe'ta renkler devre dışı kalır (force=True ile geçersiz kıl).
    """
    if not force and not _terminal_supports_color():
        return text
    return "".join(codes) + text + _C.RESET


def _terminal_supports_color() -> bool:
    """Terminal ANSI renk desteğini kontrol eder."""
    # colorama varsa ANSI desteklenir
    try:
        import colorama  # type: ignore
        colorama.init(autoreset=True)
        return True
    except ImportError:
        pass
    # Windows'ta CMD/PowerShell'de ANSI desteği Windows 10 1511+ ile var
    if sys.platform == "win32":
        return os.environ.get("ANSICON") is not None or os.environ.get("WT_SESSION") is not None
    # Linux/macOS: tty kontrolü
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ---------------------------------------------------------------------------
# JSONL log satır formatı
# ---------------------------------------------------------------------------

def _make_jsonl_entry(level: str, event: str, data: Dict[str, Any]) -> str:
    """JSON Lines formatında tek satır log girdisi oluşturur."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **data,
    }
    return json.dumps(entry, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# TrainingLogger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    V3 eğitimi için yapılandırılmış, renkli ve JSON çıktılı logger.

    Parametre açıklamaları:
        log_dir         : Log dosyalarının kaydedileceği dizin.
        experiment_name : Deney adı (log dosya ismi öneki).
        log_every_n_batches: Her N batch'te bir batch özeti yazar.
        use_color       : Renkli terminal çıktısını zorla (None → otomatik algıla).

    Dosyalar::

        log_dir/
            {experiment_name}_events.jsonl   ← Her satır bir JSON objesi
            {experiment_name}_training.log   ← İnsan okunabilir metin logu
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "cevahir_v3",
        log_every_n_batches: int = 50,
        use_color: Optional[bool] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_every_n_batches = max(1, log_every_n_batches)
        self._use_color: bool = (
            use_color if use_color is not None else _terminal_supports_color()
        )

        # Dizin oluştur
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Python standart logger
        self._logger = logging.getLogger(f"CevahirV3.{experiment_name}")
        if not self._logger.handlers:
            self._logger.setLevel(logging.DEBUG)
            # Konsol handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._logger.addHandler(ch)
            # Dosya handler (plain text)
            log_file = self.log_dir / f"{experiment_name}_training.log"
            fh = logging.FileHandler(str(log_file), encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._logger.addHandler(fh)

        # JSONL dosya handle'ı
        jsonl_path = self.log_dir / f"{experiment_name}_events.jsonl"
        self._jsonl_file = open(str(jsonl_path), "a", encoding="utf-8", buffering=1)

        # Batch sayacı (batch özeti için)
        self._batch_counter: int = 0
        # Epoch başlangıç zamanı
        self._epoch_start_time: float = 0.0

        self._logger.info(
            "%s Training logger başlatıldı | log_dir=%s",
            _colorize("[TrainingLogger]", _C.CYAN, _C.BOLD) if self._use_color else "[TrainingLogger]",
            log_dir,
        )

    # ------------------------------------------------------------------
    # Epoch logging
    # ------------------------------------------------------------------

    def log_epoch_start(
        self,
        epoch: int,
        total_epochs: int,
        lr: float,
        tf_prob: Optional[float] = None,
    ) -> None:
        """
        Epoch başlangıcını loglar.

        Args:
            epoch       : Mevcut epoch numarası (1-indeksli).
            total_epochs: Toplam epoch sayısı.
            lr          : Mevcut öğrenme oranı.
            tf_prob     : Teacher forcing olasılığı (scheduled sampling).
        """
        self._epoch_start_time = time.perf_counter()
        self._batch_counter = 0

        header = f"EPOCH {epoch}/{total_epochs}"
        tf_str = f" | TF={tf_prob:.3f}" if tf_prob is not None else ""
        msg = f"{'═' * 20} {header} {'═' * 20}  LR={lr:.2e}{tf_str}"

        if self._use_color:
            msg = _colorize(msg, _C.BOLD, _C.CYAN)

        self._logger.info(msg)
        self._write_jsonl("INFO", "epoch_start", {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "lr": lr,
            "teacher_forcing_prob": tf_prob,
        })

    def log_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        duration: float,
        convergence_status: str,
    ) -> None:
        """
        Epoch sonucunu loglar.

        Args:
            epoch             : Mevcut epoch numarası.
            train_metrics     : {"loss": float, "acc": float, "entropy": float, ...}
            val_metrics       : {"loss": float, "acc": float, "entropy": float, ...}
            duration          : Epoch süresi (saniye).
            convergence_status: "converging" | "stagnating" | "diverging" | "stable" vb.
        """
        t_loss = train_metrics.get("loss", float("nan"))
        v_loss = val_metrics.get("loss", float("nan"))
        t_acc  = train_metrics.get("acc", None)
        v_acc  = val_metrics.get("acc", None)
        t_ent  = train_metrics.get("entropy", None)
        v_ent  = val_metrics.get("entropy", None)

        # Kayıp satırı
        loss_line = f"  Loss  → Train: {t_loss:.4f} | Val: {v_loss:.4f}"
        acc_line  = ""
        if t_acc is not None and v_acc is not None:
            acc_line = f"  Acc   → Train: {t_acc:.4f} | Val: {v_acc:.4f}"
        ent_line = ""
        if t_ent is not None and v_ent is not None:
            ent_line = f"  Ent   → Train: {t_ent:.4f} | Val: {v_ent:.4f}"

        # Convergence rengi
        status_color = {
            "converging":  _C.GREEN,
            "stable":      _C.GREEN,
            "stagnating":  _C.YELLOW,
            "diverging":   _C.RED,
        }.get(convergence_status.lower(), _C.WHITE)

        status_str = convergence_status.upper()
        if self._use_color:
            status_str = _colorize(status_str, status_color, _C.BOLD)
            loss_line  = _colorize(loss_line, _C.WHITE)

        lines = [
            f"  Süre  → {duration:.1f}s | Durum: {status_str}",
            loss_line,
        ]
        if acc_line:
            lines.append(_colorize(acc_line, _C.WHITE) if self._use_color else acc_line)
        if ent_line:
            lines.append(_colorize(ent_line, _C.DIM) if self._use_color else ent_line)
        lines.append("  " + "─" * 58)

        for line in lines:
            self._logger.info(line)

        self._write_jsonl("INFO", "epoch_end", {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "duration_sec": duration,
            "convergence_status": convergence_status,
        })

    # ------------------------------------------------------------------
    # Batch logging
    # ------------------------------------------------------------------

    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        acc: float,
        entropy: float,
        grad_norm: float,
        tokens_per_sec: float,
    ) -> None:
        """
        Batch düzeyinde özet loglar (her log_every_n_batches batch'te bir).

        Args:
            epoch        : Mevcut epoch.
            batch        : Mevcut batch numarası (0-indeksli).
            total_batches: Toplam batch sayısı.
            loss         : Batch loss değeri.
            acc          : Batch accuracy.
            entropy      : Tahmin dağılımının entropisi.
            grad_norm    : Global gradient norm.
            tokens_per_sec: İşlenen token/saniye.
        """
        self._batch_counter += 1
        if self._batch_counter % self.log_every_n_batches != 0:
            return

        pct = 100.0 * (batch + 1) / max(total_batches, 1)
        msg = (
            f"  [{epoch}] Batch {batch+1:4d}/{total_batches} ({pct:5.1f}%)"
            f" | loss={loss:.4f} acc={acc:.3f} ent={entropy:.3f}"
            f" | grad={grad_norm:.3f} | {tokens_per_sec:.0f} tok/s"
        )

        if self._use_color:
            msg = _colorize(msg, _C.DIM)

        self._logger.debug(msg)
        # Batch olaylarını JSONL'e YAZMIYORUZ — çok fazla satır üretir.
        # Sadece çok yavaş durumlar (grad_norm > 10) kayıt edilir.
        if grad_norm > 10.0:
            self._write_jsonl("WARN", "high_grad_norm", {
                "epoch": epoch,
                "batch": batch,
                "grad_norm": grad_norm,
                "loss": loss,
            })

    # ------------------------------------------------------------------
    # Güvenlik olayı logging
    # ------------------------------------------------------------------

    def log_safety_event(
        self,
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        """
        Güvenlik olaylarını (NaN, spike, divergence) loglar.

        Args:
            event_type: "nan_detected" | "loss_spike" | "divergence" | "recovery" vb.
            details   : Olayla ilgili ek bilgi sözlüğü.
        """
        icon_map = {
            "nan_detected": "🔴 NaN",
            "loss_spike":   "⚠️  SPIKE",
            "divergence":   "💥 DIVERGE",
            "recovery":     "✅ RECOVERY",
            "grad_overflow":"⚠️  GRAD_OVF",
        }
        icon = icon_map.get(event_type, f"[{event_type.upper()}]")

        msg = f"  {icon} | {details}"

        if self._use_color:
            color = _C.RED if "nan" in event_type or "div" in event_type else _C.YELLOW
            if "recovery" in event_type:
                color = _C.GREEN
            msg = _colorize(msg, color, _C.BOLD)

        self._logger.warning(msg)
        self._write_jsonl("WARN", f"safety_{event_type}", details)

    # ------------------------------------------------------------------
    # Inference probe logging
    # ------------------------------------------------------------------

    def log_inference_probe(
        self,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Inference quality probe sonuçlarını loglar.

        Args:
            epoch  : Probe yapılan epoch.
            metrics: {"entropy": float, "eos_ratio": float, "ttr": float,
                      "avg_len": float, "production_ready": bool, ...}
        """
        entropy    = metrics.get("entropy", float("nan"))
        eos_ratio  = metrics.get("eos_ratio", float("nan"))
        avg_len    = metrics.get("avg_len", float("nan"))
        ttr        = metrics.get("ttr", float("nan"))
        ready      = metrics.get("production_ready", False)

        ready_str = "HAZIR" if ready else "HAZIR DEĞİL"
        if self._use_color:
            ready_str = _colorize(ready_str, _C.GREEN if ready else _C.YELLOW, _C.BOLD)

        msg = (
            f"  [PROBE @ E{epoch}] ent={entropy:.3f} eos={eos_ratio:.3f}"
            f" avg_len={avg_len:.1f} TTR={ttr:.3f} → {ready_str}"
        )

        if self._use_color:
            msg = _colorize(msg, _C.MAGENTA)

        self._logger.info(msg)
        self._write_jsonl("INFO", "inference_probe", {"epoch": epoch, **metrics})

    # ------------------------------------------------------------------
    # Checkpoint logging
    # ------------------------------------------------------------------

    def log_checkpoint(
        self,
        epoch: int,
        path: str,
        is_best: bool,
    ) -> None:
        """
        Checkpoint kaydını loglar.

        Args:
            epoch  : Kaydın yapıldığı epoch.
            path   : Kaydedilen dosya yolu.
            is_best: Bu checkpoint en iyi metriğe mi sahip.
        """
        tag = " [BEST]" if is_best else ""
        msg = f"  [CHECKPOINT]{tag} Epoch {epoch} → {path}"

        if self._use_color:
            color = _C.GREEN if is_best else _C.CYAN
            msg = _colorize(msg, color)

        self._logger.info(msg)
        self._write_jsonl("INFO", "checkpoint_saved", {
            "epoch": epoch,
            "path": path,
            "is_best": is_best,
        })

    # ------------------------------------------------------------------
    # LR değişim logging
    # ------------------------------------------------------------------

    def log_lr_change(
        self,
        old_lr: float,
        new_lr: float,
        reason: str,
    ) -> None:
        """
        Learning rate değişimini loglar.

        Args:
            old_lr: Eski öğrenme oranı.
            new_lr: Yeni öğrenme oranı.
            reason: Değişim sebebi (örn. "scheduler_step", "plateau", "warmup_end").
        """
        direction = "▼" if new_lr < old_lr else "▲"
        msg = (
            f"  [LR] {direction} {old_lr:.2e} → {new_lr:.2e}"
            f" | Sebep: {reason}"
        )

        if self._use_color:
            color = _C.BLUE if new_lr < old_lr else _C.GREEN
            msg = _colorize(msg, color, _C.BOLD)

        self._logger.info(msg)
        self._write_jsonl("INFO", "lr_change", {
            "old_lr": old_lr,
            "new_lr": new_lr,
            "reason": reason,
        })

    # ------------------------------------------------------------------
    # Convergence status logging
    # ------------------------------------------------------------------

    def log_convergence_change(
        self,
        old_status: str,
        new_status: str,
        epoch: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convergence durumu değişimini loglar.

        Args:
            old_status: Önceki durum ("converging", "stagnating", vb.).
            new_status: Yeni durum.
            epoch     : Değişimin gerçekleştiği epoch.
            details   : Ek bilgiler (opsiyonel).
        """
        msg = (
            f"  [CONVERGENCE] E{epoch}: {old_status.upper()} → {new_status.upper()}"
        )
        if details:
            msg += f" | {details}"

        if self._use_color:
            color_map = {
                "converging": _C.GREEN,
                "stable":     _C.GREEN,
                "stagnating": _C.YELLOW,
                "diverging":  _C.RED,
            }
            color = color_map.get(new_status.lower(), _C.WHITE)
            msg = _colorize(msg, color, _C.BOLD)

        self._logger.info(msg)
        self._write_jsonl("INFO", "convergence_change", {
            "epoch": epoch,
            "old_status": old_status,
            "new_status": new_status,
            **(details or {}),
        })

    # ------------------------------------------------------------------
    # Genel bilgi ve uyarı logging
    # ------------------------------------------------------------------

    def info(self, msg: str, **kwargs: Any) -> None:
        """Genel bilgi mesajı."""
        self._logger.info(msg)
        if kwargs:
            self._write_jsonl("INFO", "general", {"message": msg, **kwargs})

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Uyarı mesajı."""
        if self._use_color:
            msg = _colorize(msg, _C.YELLOW)
        self._logger.warning(msg)
        if kwargs:
            self._write_jsonl("WARN", "warning", {"message": msg, **kwargs})

    def error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Hata mesajı."""
        if self._use_color:
            msg = _colorize(msg, _C.RED, _C.BOLD)
        self._logger.error(msg, exc_info=exc_info)
        self._write_jsonl("ERROR", "error", {"message": msg, **kwargs})

    # ------------------------------------------------------------------
    # Kapatma
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Log dosyalarını temiz bir şekilde kapatır.
        Context manager kullanılıyorsa __exit__'te otomatik çağrılır.
        """
        try:
            self._jsonl_file.flush()
            self._jsonl_file.close()
        except Exception:
            pass
        for handler in self._logger.handlers[:]:
            try:
                handler.close()
                self._logger.removeHandler(handler)
            except Exception:
                pass
        self._logger.info("[TrainingLogger] Kapatıldı.")

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    def _write_jsonl(
        self,
        level: str,
        event: str,
        data: Dict[str, Any],
    ) -> None:
        """JSONL dosyasına tek satır yazar."""
        try:
            line = _make_jsonl_entry(level, event, data)
            self._jsonl_file.write(line + "\n")
        except Exception as exc:
            # JSONL yazım hatası diğer loglamayı engellememeli
            self._logger.debug("[TrainingLogger] JSONL yazım hatası: %s", exc)
