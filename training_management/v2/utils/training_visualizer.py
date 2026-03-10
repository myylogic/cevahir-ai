# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: training_visualizer.py
Modül: training_management/v2/utils
Görev: Training Visualizer - Eğitim ve doğrulama metriklerini görselleştirmek için
       gelişmiş görselleştirici. Headless ortam desteği (Agg backend), güvenli
       dosya adı üretimi, EMA yumuşatma, epoch ekseni, toplu çizim ve CSV/JSON
       çıktı desteği sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (görselleştirme)
- Design Patterns: Visualizer Pattern (metrik görselleştirme)
- Endüstri Standartları: Training visualization best practices

KULLANIM:
- Eğitim metriklerini görselleştirmek için
- Loss/accuracy grafikleri çizmek için
- CSV/JSON çıktı üretmek için
- Headless ortam desteği için

BAĞIMLILIKLAR:
- matplotlib: Görselleştirme
- csv, json: Çıktı formatları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import os
import csv
import json
import math
import logging
from typing import List, Optional, Dict, Any, Iterable, Tuple

# Headless destek: DISPLAY yoksa Agg
try:
    import matplotlib
    if os.environ.get("DISPLAY", "") == "" or os.environ.get("MPLBACKEND", "") == "Agg":
        matplotlib.use("Agg")  # GUI gerektirmeyen backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Opsiyonel TrainingLogger
try:
    from training_management.v2.utils.training_logger import TrainingLogger
except Exception:
    TrainingLogger = None  # type: ignore


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _slugify(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name.strip())
    return safe or "figure"


def _ema(values: Iterable[float], alpha: float) -> List[float]:
    out: List[float] = []
    last: Optional[float] = None
    for v in values:
        if last is None:
            last = float(v)
        else:
            last = alpha * float(v) + (1.0 - alpha) * last
        out.append(last)
    return out


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _align_series(*series: List[float]) -> Tuple[List[List[float]], int]:
    """Serileri en kısa uzunluğa hizalar ve kırpar."""
    if not series:
        return [], 0
    min_len = min(len(s) for s in series if isinstance(s, list))
    if min_len <= 0:
        return [s[:0] for s in series], 0
    return [s[:min_len] for s in series], min_len


class TrainingVisualizer:
    """
    Eğitim sürecinin grafiklerini oluşturan sınıf.
    """

    def __init__(
        self,
        save_dir: str = "visualizations",
        *,
        style: str = "default",
        run_name: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> None:
        """
        Args:
            save_dir: Grafiklerin kaydedileceği dizin.
            style: Matplotlib stil adı (örn. 'default', 'ggplot', 'seaborn-v0_8' vb.)
            run_name: (opsiyonel) log’larda kullanmak için isim etiketi.
            logger: TrainingLogger veya logging.Logger; yoksa basit logger oluşturulur.
        """
        self.save_dir = os.path.abspath(save_dir)
        _ensure_dir(self.save_dir)

        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(style)
            except Exception:
                # bilinmeyen stil adı verilirse default'a düş
                plt.style.use("default")

        # ✅ SOLID: Logger dependency injection (TrainingManager'dan geçirilir)
        if logger is not None:
            self.logger = logger
        else:
            # Fallback: Basit console logger (dosya logging yok)
            self.logger = logging.getLogger("TrainingVisualizer")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            self.logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------ public API

    def plot_loss(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        *,
        epochs: Optional[List[int]] = None,
        ema_alpha: Optional[float] = None,
        title: str = "Eğitim ve Doğrulama Kaybı",
        xlabel: str = "Epoch",
        ylabel: str = "Kayıp",
        save_filename: str = "loss_plot.png",
        show: bool = False,
    ) -> str:
        """
        Eğitim ve (opsiyonel) doğrulama loss grafiğini çizer ve kaydeder.
        """
        series = [train_losses] + ([val_losses] if val_losses is not None else [])
        series, n = _align_series(*series)
        if n == 0:
            raise ValueError("plot_loss: çizilecek veri yok.")

        train, = series[:1]
        val = series[1] if len(series) > 1 else None

        # EMA yumuşatma
        if ema_alpha is not None and 0.0 < float(ema_alpha) < 1.0:
            train = _ema(train, float(ema_alpha))
            if val is not None:
                val = _ema(val, float(ema_alpha))

        # X ekseni
        xs = epochs[:n] if isinstance(epochs, list) and len(epochs) >= n else list(range(1, n + 1))

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib mevcut değil, grafik oluşturulamıyor")
            return None

        fig = plt.figure(figsize=(10, 6))
        plt.plot(xs, train, label="Eğitim Kaybı", marker="o")
        if val is not None:
            plt.plot(xs, val, label="Doğrulama Kaybı", marker="x")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, _slugify(save_filename))
        self._finalize_figure(fig, save_path, show)
        self._log_info(f"Kayıp grafiği kaydedildi: {save_path}")
        return save_path

    def plot_accuracy(
        self,
        train_accuracies: List[float],
        val_accuracies: Optional[List[float]] = None,
        *,
        epochs: Optional[List[int]] = None,
        ema_alpha: Optional[float] = None,
        title: str = "Eğitim ve Doğrulama Doğruluğu",
        xlabel: str = "Epoch",
        ylabel: str = "Doğruluk",
        save_filename: str = "accuracy_plot.png",
        show: bool = False,
    ) -> str:
        """
        Eğitim ve (opsiyonel) doğrulama accuracy grafiğini çizer ve kaydeder.
        """
        series = [train_accuracies] + ([val_accuracies] if val_accuracies is not None else [])
        series, n = _align_series(*series)
        if n == 0:
            raise ValueError("plot_accuracy: çizilecek veri yok.")

        tr, = series[:1]
        va = series[1] if len(series) > 1 else None

        if ema_alpha is not None and 0.0 < float(ema_alpha) < 1.0:
            tr = _ema(tr, float(ema_alpha))
            if va is not None:
                va = _ema(va, float(ema_alpha))

        xs = epochs[:n] if isinstance(epochs, list) and len(epochs) >= n else list(range(1, n + 1))

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib mevcut değil, grafik oluşturulamıyor")
            return None

        fig = plt.figure(figsize=(10, 6))
        plt.plot(xs, tr, label="Eğitim Doğruluğu", marker="o")
        if va is not None:
            plt.plot(xs, va, label="Doğrulama Doğruluğu", marker="x")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, _slugify(save_filename))
        self._finalize_figure(fig, save_path, show)
        self._log_info(f"Doğruluk grafiği kaydedildi: {save_path}")
        return save_path

    def plot_custom_metric(
        self,
        metric_values: List[float],
        metric_name: str,
        *,
        epochs: Optional[List[int]] = None,
        ema_alpha: Optional[float] = None,
        save_filename: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """
        Belirtilen metriğin (tek seri) grafiğini çizer ve kaydeder.
        """
        values = list(metric_values or [])
        if not values:
            raise ValueError("plot_custom_metric: metric_values boş.")

        if ema_alpha is not None and 0.0 < float(ema_alpha) < 1.0:
            values = _ema(values, float(ema_alpha))

        n = len(values)
        xs = epochs[:n] if isinstance(epochs, list) and len(epochs) >= n else list(range(1, n + 1))

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib mevcut değil, grafik oluşturulamıyor")
            return None

        fig = plt.figure(figsize=(10, 6))
        plt.plot(xs, values, label=metric_name, marker="o")
        plt.title(title or f"{metric_name} Grafiği")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)

        filename = save_filename or f"{_slugify(metric_name.lower())}_plot.png"
        save_path = os.path.join(self.save_dir, _slugify(filename))
        self._finalize_figure(fig, save_path, show)
        self._log_info(f"{metric_name} grafiği kaydedildi: {save_path}")
        return save_path

    def plot_from_history(
        self,
        history: Dict[str, List[float]],
        *,
        save_prefix: str = "",
        ema_alpha: Optional[float] = None,
        show: bool = False,
    ) -> Dict[str, str]:
        """
        training_history sözlüğünden (örn. {'train_loss': [...], 'val_loss': [...], 'accuracy': [...]})
        otomatik grafikler üretir.

        Returns:
            {'loss': path, 'accuracy': path} (mevcut olanlar)
        """
        paths: Dict[str, str] = {}
        sp = (save_prefix + "_") if save_prefix else ""

        train_loss = history.get("train_loss") or history.get("loss") or []
        val_loss = history.get("val_loss") or []
        if train_loss:
            paths["loss"] = self.plot_loss(
                train_loss, val_loss,
                ema_alpha=ema_alpha,
                save_filename=f"{sp}loss.png",
                show=show,
            )

        train_acc = history.get("train_accuracy") or []
        val_acc = history.get("accuracy") or history.get("val_accuracy") or []
        # Eğer sadece val accuracy tutuluyorsa onu tek başına da çizebiliriz
        if train_acc or val_acc:
            if not train_acc:
                # tek seri çiz
                paths["accuracy"] = self.plot_custom_metric(
                    val_acc, "Accuracy",
                    ema_alpha=ema_alpha,
                    save_filename=f"{sp}accuracy.png",
                    show=show,
                )
            else:
                paths["accuracy"] = self.plot_accuracy(
                    train_acc, val_acc,
                    ema_alpha=ema_alpha,
                    save_filename=f"{sp}accuracy.png",
                    show=show,
                )

        return paths

    # -------------------------------------------------------------- export helpers

    def export_history_csv(
        self,
        history: Dict[str, List[Any]],
        filename: str = "metrics.csv",
    ) -> str:
        """
        History sözlüğünü CSV olarak kaydeder (anahtarlar sütun adına karşılık gelir).
        Farklı uzunluktaki sütunlar en kısa uzunluğa kırpılır.
        """
        keys = list(history.keys())
        if not keys:
            raise ValueError("export_history_csv: history boş.")

        # hizalama
        min_len = min(len(history[k]) for k in keys if isinstance(history[k], list))
        if min_len <= 0:
            raise ValueError("export_history_csv: yazılacak veri yok.")
        rows = []
        for i in range(min_len):
            row = {k: history[k][i] for k in keys}
            rows.append(row)

        path = os.path.join(self.save_dir, _slugify(filename))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        self._log_info(f"Metrikler CSV olarak kaydedildi: {path}")
        return path

    def export_history_json(
        self,
        history: Dict[str, Any],
        filename: str = "metrics.json",
        merge_if_exists: bool = True,
    ) -> str:
        """
        History sözlüğünü JSON olarak kaydeder. merge_if_exists=True ise mevcut dosya ile birleştirir.
        """
        path = os.path.join(self.save_dir, _slugify(filename))
        if merge_if_exists and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
            except Exception:
                prev = {}
            # basit birleştirme: aynı key varsa arka arkaya ekle
            for k, v in history.items():
                if isinstance(prev.get(k), list) and isinstance(v, list):
                    prev[k].extend(v)
                else:
                    prev[k] = v
            data = prev
        else:
            data = history

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._log_info(f"Metrikler JSON olarak kaydedildi: {path}")
        return path

    # --------------------------------------------------------------- internal util

    @staticmethod
    def _finalize_figure(fig: plt.Figure, save_path: str, show: bool = False) -> None:
        if not MATPLOTLIB_AVAILABLE:
            return
        _ensure_dir(os.path.dirname(save_path))
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            try:
                plt.show()
            except Exception:
                # headless ortamda show başarısız olabilir
                pass
        plt.close(fig)

    # --------------------------------------------------------------- logging shim

    def _log_info(self, msg: str) -> None:
        if hasattr(self.logger, "log_info"):
            self.logger.log_info(msg)
        else:
            self.logger.info(msg)

    def _log_warning(self, msg: str) -> None:
        if hasattr(self.logger, "log_warning"):
            self.logger.log_warning(msg)
        else:
            self.logger.warning(msg)

    def _log_error(self, msg: str) -> None:
        if hasattr(self.logger, "log_error"):
            self.logger.log_error(msg)
        else:
            self.logger.error(msg)
