# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ — Training Management V3
================================================================================
Dosya : training_management/v3/utils/metrics_tracker.py
Modül : MetricsTracker
Görev : Epoch/batch düzeyinde metrik geçmişi takibi, JSON kaydetme/yükleme
        ve matplotlib görselleştirmesi.

Takip edilen metrikler (tam liste):
  Kayıp : train_loss, val_loss
  Doğr. : train_acc, val_acc
  Entrp : train_entropy, val_entropy
  Grad. : gradient_norm, gradient_health_score
  Tok.  : eos_ratio, content_ratio, unigram_entropy
  Perf. : tokens_per_sec, epoch_duration_sec
  LR    : lr (her param group için ayrı key: lr_0, lr_1, ...)
  Çıkar.: inference_quality (her probe epoch'u için)
  Kayıp detayı: loss_breakdown.ce, loss_breakdown.entropy_reg,
                loss_breakdown.focal, loss_breakdown.auxiliary
  Zamanlama: teacher_forcing_prob
  Güvenlik : nan_count, spike_count

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Epoch ve batch düzeyinde metrik geçmişi yöneticisi.

    Kullanım örneği::

        tracker = MetricsTracker()
        tracker.update_epoch(
            epoch=1,
            train_metrics={"loss": 2.3, "acc": 0.4, "entropy": 1.8},
            val_metrics={"loss": 2.5, "acc": 0.38, "entropy": 1.6},
            gradient_norm=0.9,
            gradient_health_score=0.95,
            eos_ratio=0.12,
            tokens_per_sec=4200.0,
            epoch_duration_sec=93.5,
            lr={0: 2e-4},
            loss_breakdown={"ce": 2.1, "entropy_reg": 0.05},
            teacher_forcing_prob=0.9,
            nan_count=0,
            spike_count=0,
        )
        history = tracker.get_history()
        best_epoch, best_val = tracker.get_best_epoch("val_loss", "min")
    """

    # Tüm takip edilen skaler metrikler (epoch başına tek değer)
    _SCALAR_KEYS = [
        "train_loss", "val_loss",
        "train_acc",  "val_acc",
        "train_entropy", "val_entropy",
        "gradient_norm", "gradient_health_score",
        "eos_ratio", "content_ratio", "unigram_entropy",
        "tokens_per_sec", "epoch_duration_sec",
        "teacher_forcing_prob",
        "nan_count", "spike_count",
    ]

    def __init__(self) -> None:
        # Ana geçmiş: her key → liste (epoch başına bir değer)
        self._history: Dict[str, List[Any]] = defaultdict(list)
        # Epoch numarası listesi (sıra referansı)
        self._epochs: List[int] = []
        # Çıkarım kalitesi probe'ları: epoch → metrik sözlüğü
        self._inference_probes: Dict[int, Dict[str, Any]] = {}
        # Oluşturulma zamanı
        self._created_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Güncelleme
    # ------------------------------------------------------------------

    def update_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        *,
        gradient_norm: Optional[float] = None,
        gradient_health_score: Optional[float] = None,
        eos_ratio: Optional[float] = None,
        content_ratio: Optional[float] = None,
        unigram_entropy: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        epoch_duration_sec: Optional[float] = None,
        lr: Optional[Dict[int, float]] = None,
        loss_breakdown: Optional[Dict[str, float]] = None,
        inference_quality: Optional[Dict[str, Any]] = None,
        teacher_forcing_prob: Optional[float] = None,
        nan_count: Optional[int] = None,
        spike_count: Optional[int] = None,
        **extra_kwargs: Any,
    ) -> None:
        """
        Tek bir epoch'un tüm metriklerini kaydeder.

        Args:
            epoch            : Epoch numarası (0-indeksli veya 1-indeksli, tutarlı olsun).
            train_metrics    : {"loss": float, "acc": float, "entropy": float, ...}
            val_metrics      : {"loss": float, "acc": float, "entropy": float, ...}
            gradient_norm    : Global gradient norm.
            gradient_health_score: 0–1 arası gradient sağlık skoru.
            eos_ratio        : Batch'te EOS tokenlarının oranı.
            content_ratio    : İçerik tokenlarının oranı.
            unigram_entropy  : Token dağılımının unigram entropisi.
            tokens_per_sec   : İşlenen token/saniye.
            epoch_duration_sec: Epoch süresi (saniye).
            lr               : {param_group_index: lr_value} sözlüğü.
            loss_breakdown   : {"ce": float, "entropy_reg": float, "focal": float, "auxiliary": float}
            inference_quality: Probe epoch'larında çıkarım metrikleri.
            teacher_forcing_prob: O anki teacher forcing olasılığı.
            nan_count        : Epoch içindeki NaN sayısı.
            spike_count      : Epoch içindeki loss spike sayısı.
            **extra_kwargs   : Ek metrikler (dinamik genişletme için).
        """
        self._epochs.append(epoch)

        # --- Train ve val metrikler (loss, acc, entropy) ---
        for key, val in train_metrics.items():
            self._history[f"train_{key}"].append(self._to_float(val))
        for key, val in val_metrics.items():
            self._history[f"val_{key}"].append(self._to_float(val))

        # --- Skaler metrikler ---
        scalar_map = {
            "gradient_norm":         gradient_norm,
            "gradient_health_score": gradient_health_score,
            "eos_ratio":             eos_ratio,
            "content_ratio":         content_ratio,
            "unigram_entropy":       unigram_entropy,
            "tokens_per_sec":        tokens_per_sec,
            "epoch_duration_sec":    epoch_duration_sec,
            "teacher_forcing_prob":  teacher_forcing_prob,
            "nan_count":             nan_count,
            "spike_count":           spike_count,
        }
        for key, val in scalar_map.items():
            if val is not None:
                self._history[key].append(self._to_float(val))

        # --- LR (per param group) ---
        if lr is not None:
            for group_idx, lr_val in lr.items():
                self._history[f"lr_{group_idx}"].append(float(lr_val))
            # Genel LR: ilk param group
            if 0 in lr:
                self._history["lr"].append(float(lr[0]))

        # --- Loss breakdown ---
        if loss_breakdown is not None:
            for comp_key, comp_val in loss_breakdown.items():
                self._history[f"loss_breakdown_{comp_key}"].append(
                    self._to_float(comp_val)
                )

        # --- Inference quality ---
        if inference_quality is not None:
            self._inference_probes[epoch] = inference_quality
            for iq_key, iq_val in inference_quality.items():
                if isinstance(iq_val, (int, float)):
                    self._history[f"inference_{iq_key}"].append(float(iq_val))

        # --- Ekstra kwargs ---
        for key, val in extra_kwargs.items():
            if isinstance(val, (int, float)):
                self._history[key].append(float(val))

        logger.debug(
            "[MetricsTracker] Epoch %d güncellendi — train_loss=%.4f, val_loss=%.4f",
            epoch,
            self._history.get("train_loss", [float("nan")])[-1],
            self._history.get("val_loss", [float("nan")])[-1],
        )

    # ------------------------------------------------------------------
    # Sorgulama
    # ------------------------------------------------------------------

    def get_history(self) -> Dict[str, List[Any]]:
        """Tüm metrik geçmişini döner (her key → değer listesi)."""
        result = {"epochs": list(self._epochs)}
        result.update({k: list(v) for k, v in self._history.items()})
        return result

    def get_best_epoch(
        self,
        metric: str = "val_loss",
        mode: str = "min",
    ) -> Tuple[int, float]:
        """
        Belirli bir metrik için en iyi epoch'u döner.

        Args:
            metric: Takip edilecek metrik anahtarı (örn. "val_loss", "val_acc").
            mode  : "min" veya "max".

        Returns:
            (best_epoch_index, best_value) tuple'ı.
            Metrik bulunamazsa (-1, float('inf') veya float('-inf')).
        """
        values = self._history.get(metric)
        if not values:
            sentinel = float("inf") if mode == "min" else float("-inf")
            return -1, sentinel

        if mode == "min":
            best_idx = int(min(range(len(values)), key=lambda i: values[i]))
        else:
            best_idx = int(max(range(len(values)), key=lambda i: values[i]))

        best_epoch = (
            self._epochs[best_idx] if best_idx < len(self._epochs) else best_idx
        )
        return best_epoch, float(values[best_idx])

    def get_summary(self) -> Dict[str, Any]:
        """
        Son epoch'un özet metriklerini döner.
        Boş geçmişte boş sözlük döner.
        """
        if not self._epochs:
            return {}

        summary: Dict[str, Any] = {"epoch": self._epochs[-1]}
        for key, values in self._history.items():
            if values:
                summary[key] = values[-1]

        # Inference probe (son probe, son epoch ya da daha önceki)
        if self._inference_probes:
            last_probe_epoch = max(self._inference_probes.keys())
            summary["inference_quality"] = self._inference_probes[last_probe_epoch]

        return summary

    def get_inference_probe(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Belirli bir epoch'un inference probe sonucunu döner."""
        return self._inference_probes.get(epoch)

    # ------------------------------------------------------------------
    # JSON kaydet / yükle
    # ------------------------------------------------------------------

    def save_to_json(self, path: str) -> None:
        """
        Tüm geçmişi JSON dosyasına kaydeder.
        Dizin yoksa oluşturulur.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "created_at": self._created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "epochs": list(self._epochs),
            "history": {k: list(v) for k, v in self._history.items()},
            "inference_probes": {
                str(ep): data for ep, data in self._inference_probes.items()
            },
        }
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
            os.replace(tmp_path, path)
            logger.info("[MetricsTracker] Geçmiş kaydedildi: %s", path)
        except Exception as exc:
            logger.error("[MetricsTracker] JSON kayıt hatası: %s", exc)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def load_from_json(self, path: str) -> None:
        """
        JSON dosyasından geçmişi yükler. Mevcut geçmiş sıfırlanır.

        Args:
            path: Yüklenecek JSON dosyasının yolu.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[MetricsTracker] Geçmiş dosyası bulunamadı: {path}"
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload: dict = json.load(f)

            self._epochs = payload.get("epochs", [])
            raw_history = payload.get("history", {})
            self._history = defaultdict(list, {k: list(v) for k, v in raw_history.items()})
            raw_probes = payload.get("inference_probes", {})
            self._inference_probes = {int(ep): data for ep, data in raw_probes.items()}
            self._created_at = payload.get("created_at", self._created_at)
            logger.info(
                "[MetricsTracker] Geçmiş yüklendi: %s — %d epoch",
                path,
                len(self._epochs),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[MetricsTracker] JSON ayrıştırma hatası: {path} — {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Görselleştirme
    # ------------------------------------------------------------------

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Temel eğitim metrik grafiklerini çizer.

        Çizilecek grafikler (2×3 grid):
          1. Train/Val Loss
          2. Train/Val Accuracy
          3. Train/Val Entropy
          4. Gradient Norm
          5. Learning Rate
          6. Tokens/Sec

        Args:
            save_path: Grafik kaydedilecek PNG yolu. None ise plt.show() ile gösterilir.
        """
        try:
            import matplotlib
            matplotlib.use("Agg" if save_path else "TkAgg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "[MetricsTracker] matplotlib bulunamadı. Grafik çizilemiyor."
            )
            return

        epochs = self._epochs
        if not epochs:
            logger.warning("[MetricsTracker] Grafik için yeterli veri yok.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle("Cevahir V3 — Eğitim Geçmişi", fontsize=14, fontweight="bold")

        def _safe_plot(ax, key_pairs, title, ylabel, xlabel="Epoch", logy=False):
            has_data = False
            for label, key in key_pairs:
                vals = self._history.get(key, [])
                if vals:
                    x = epochs[: len(vals)]
                    ax.plot(x, vals, label=label, linewidth=1.5)
                    has_data = True
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if logy and has_data:
                try:
                    ax.set_yscale("log")
                except Exception:
                    pass
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        _safe_plot(
            axes[0, 0],
            [("Train Loss", "train_loss"), ("Val Loss", "val_loss")],
            "Kayıp (Loss)", "Loss", logy=False,
        )
        _safe_plot(
            axes[0, 1],
            [("Train Acc", "train_acc"), ("Val Acc", "val_acc")],
            "Doğruluk (Accuracy)", "Accuracy",
        )
        _safe_plot(
            axes[0, 2],
            [("Train Entropy", "train_entropy"), ("Val Entropy", "val_entropy")],
            "Entropi (Entropy)", "Entropy",
        )
        _safe_plot(
            axes[1, 0],
            [("Gradient Norm", "gradient_norm")],
            "Gradient Norm", "Norm",
        )
        # LR — ilk param group
        lr_keys = [(f"LR (g{i})", f"lr_{i}") for i in range(3)
                   if f"lr_{i}" in self._history]
        if not lr_keys:
            lr_keys = [("LR", "lr")]
        _safe_plot(axes[1, 1], lr_keys, "Learning Rate", "LR", logy=True)

        _safe_plot(
            axes[1, 2],
            [("Tokens/Sec", "tokens_per_sec")],
            "İşlem Hızı", "Tokens/Sec",
        )

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[MetricsTracker] Grafik kaydedildi: %s", save_path)
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(val: Any) -> float:
        """Güvenli float dönüşümü."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    def __len__(self) -> int:
        """Kayıtlı epoch sayısını döner."""
        return len(self._epochs)

    def __repr__(self) -> str:
        return (
            f"MetricsTracker(epochs={len(self._epochs)}, "
            f"keys={list(self._history.keys())[:5]}...)"
        )


# ---------------------------------------------------------------------------
# JSON serialization yardımcısı
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Standart olmayan tipleri JSON uyumlu formata çevirir."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
    # numpy int/float
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)
