"""
loss_spike_detector.py
======================
Cevahir Türkçe Dil Modeli - Loss Spike Tespiti ve Müdahale Sistemi

Eğitim sırasındaki ani loss artışlarını (spike) istatistiksel yöntemle
tespit eder ve otomatik müdahale uygular.

Spike tespiti:
    loss > mean(last_k) + n_sigma * std(last_k)

Müdahale stratejisi:
    1. Ani spike → LR geçici azalt (spike_lr_factor)
    2. Sürekli spike → Warmup yeniden başlat
    3. Her spike → spike log'una yaz (checkpoint entegrasyonu için)

Window-based statistics ile rolling mean/std hesaplama kullanılır.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class LossSpikeDetector:
    """
    Loss spike tespiti ve otomatik müdahale sistemi.

    Spike Tespiti:
        Rolling window içindeki mean ve std hesaplanır.
        loss > mean + n_sigma * std koşulu sağlanıyorsa spike sayılır.

    Müdahale Seviyeleri:
        1. İlk spike     → LR'ı spike_lr_factor ile geçici azalt
        2. Sürekli spike → Warmup yeniden başlatılır (simüle edilir)
        3. Her spike     → Spike log'u tutulur

    Args:
        optimizer:       PyTorch optimizer nesnesi.
        window_size:     İstatistik penceresi büyüklüğü.
        n_sigma:         Spike eşiği (standart sapma çarpanı).
        spike_lr_factor: Spike anında LR'a uygulanacak çarpan (0 < f < 1).
        recovery_epochs: Spike sonrası recovery süresi (epoch).
        min_history:     Spike tespiti için gereken minimum geçmiş sayısı.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        window_size: int = 10,
        n_sigma: float = 3.0,
        spike_lr_factor: float = 0.5,
        recovery_epochs: int = 3,
        min_history: int = 5,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size en az 2 olmalı.")
        if n_sigma <= 0:
            raise ValueError("n_sigma pozitif olmalı.")
        if not (0.0 < spike_lr_factor < 1.0):
            raise ValueError("spike_lr_factor 0 ile 1 arasında olmalı.")
        if min_history < 2:
            raise ValueError("min_history en az 2 olmalı.")
        if min_history > window_size:
            raise ValueError("min_history, window_size'dan büyük olamaz.")

        self.optimizer = optimizer
        self.window_size = window_size
        self.n_sigma = n_sigma
        self.spike_lr_factor = spike_lr_factor
        self.recovery_epochs = recovery_epochs
        self.min_history = min_history

        # --- Dahili durum ---
        self._loss_window: Deque[float] = deque(maxlen=window_size)
        self._loss_history_full: List[float] = []
        self._spike_count: int = 0
        self._last_spike_epoch: Optional[int] = None
        self._recovery_remaining: int = 0  # Recovery için kalan epoch sayısı
        self._spike_log: List[Dict[str, Any]] = []
        self._original_lrs: List[float] = []  # Spike öncesi LR değerleri
        self._consecutive_spikes: int = 0

        # Optimizer'daki başlangıç LR değerlerini kaydet
        self._base_lrs: List[float] = [
            pg["lr"] for pg in optimizer.param_groups
        ]

        logger.info(
            "LossSpikeDetector başlatıldı | window=%d | n_sigma=%.1f | "
            "lr_factor=%.3f | recovery_epochs=%d",
            window_size,
            n_sigma,
            spike_lr_factor,
            recovery_epochs,
        )

    # ------------------------------------------------------------------
    # Herkese Açık API
    # ------------------------------------------------------------------

    def update(self, loss: float, epoch: int) -> bool:
        """
        Yeni bir loss değerini sisteme ekler ve spike tespiti yapar.

        Args:
            loss:  Mevcut epoch/adım için hesaplanan loss değeri.
            epoch: Mevcut epoch numarası.

        Returns:
            True  → spike tespit edildi.
            False → normal durum.
        """
        if math.isnan(loss) or math.isinf(loss):
            logger.warning(
                "update(): NaN/Inf loss alındı (%.4f). Spike log'a yazılıyor.", loss
            )
            # NaN/Inf loss'u history'e ekleme ama spike say
            self._handle_spike(loss, epoch, reason="nan_inf_loss")
            return True

        # Geçmişe ekle
        self._loss_window.append(loss)
        self._loss_history_full.append(loss)
        # [MEM-FIX] _loss_history_full sınırsız büyümeyi önle: max 10000 entry tut
        if len(self._loss_history_full) > 10000:
            self._loss_history_full = self._loss_history_full[-10000:]

        # Recovery sayacını geri say
        if self._recovery_remaining > 0:
            self._recovery_remaining -= 1
            logger.debug(
                "Recovery modu | kalan: %d epoch", self._recovery_remaining
            )

        # Yeterli geçmiş yoksa spike tespiti yapma
        if len(self._loss_window) < self.min_history:
            return False

        # Spike tespiti
        is_spike = self._detect_spike(loss)
        if is_spike:
            self._consecutive_spikes += 1
            self._handle_spike(loss, epoch, reason="statistical")
            return True
        else:
            self._consecutive_spikes = 0
            return False

    def _detect_spike(self, loss: float) -> bool:
        """
        Rolling window istatistikleriyle spike tespiti yapar.

        Hesaplama:
            window'dan mevcut loss hariç tutularak mean/std hesaplanır,
            böylece spike kendi kendini maskelemiş olmaz.

        Args:
            loss: Kontrol edilecek loss değeri.

        Returns:
            True → spike tespit edildi.
        """
        # Son eklenen değer zaten window'da; önceki değerlere bak
        window_list = list(self._loss_window)

        # Mevcut loss'u çıkar (kendisiyle karşılaştırmamak için)
        if len(window_list) > 1:
            reference = window_list[:-1]
        else:
            return False

        if len(reference) < 2:
            return False

        mean_val = sum(reference) / len(reference)
        variance = sum((x - mean_val) ** 2 for x in reference) / len(reference)
        std_val = math.sqrt(variance) if variance > 0 else 0.0

        threshold = mean_val + self.n_sigma * std_val

        is_spike = loss > threshold

        if is_spike:
            logger.warning(
                "Loss spike tespit edildi: %.4f > %.4f (mean=%.4f, std=%.4f, n_sigma=%.1f)",
                loss,
                threshold,
                mean_val,
                std_val,
                self.n_sigma,
            )

        return is_spike

    def intervene(self) -> Dict[str, Any]:
        """
        Son spike için uygun müdahale eylemini uygular.

        Karar mantığı:
            consecutive_spikes >= 3 → warmup yeniden başlat
            diğer                   → LR geçici azalt

        Returns:
            Uygulanan eylem detaylarını içeren dict.
        """
        action_taken: Dict[str, Any] = {
            "timestamp": time.time(),
            "consecutive_spikes": self._consecutive_spikes,
            "recovery_epochs": self.recovery_epochs,
        }

        if self._consecutive_spikes >= 3:
            # Sürekli spike → warmup benzeri LR sıfırlama
            action_taken["action"] = "warmup_restart"
            self._warmup_restart()
        else:
            # Tek/az spike → LR geçici azalt
            action_taken["action"] = "reduce_lr_temporary"
            self._temporary_lr_reduce()

        # Recovery modunu etkinleştir
        self._recovery_remaining = self.recovery_epochs
        action_taken["recovery_epochs_remaining"] = self._recovery_remaining

        logger.info("Spike müdahalesi uygulandı: %s", action_taken)
        return action_taken

    def is_in_recovery(self) -> bool:
        """
        Sistemin spike sonrası recovery modunda olup olmadığını döndürür.

        Returns:
            True → recovery modunda (intervene sonrası bekleme süresi).
        """
        return self._recovery_remaining > 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Spike istatistiklerini döndürür.

        Returns:
            Dict içeriği:
                spike_count        : Toplam spike sayısı
                last_spike_epoch   : Son spike'ın gerçekleştiği epoch
                consecutive_spikes : Ardışık spike sayısı
                in_recovery        : Recovery modunda mı?
                recovery_remaining : Recovery için kalan epoch
                window_mean        : Mevcut pencere ortalaması
                window_std         : Mevcut pencere standart sapması
                window_size_current: Mevcut penceredeki eleman sayısı
                total_loss_steps   : Toplam güncelleme sayısı
                spike_log_last_5   : Son 5 spike kaydı
        """
        mean_val, std_val = self._window_stats()
        return {
            "spike_count": self._spike_count,
            "last_spike_epoch": self._last_spike_epoch,
            "consecutive_spikes": self._consecutive_spikes,
            "in_recovery": self.is_in_recovery(),
            "recovery_remaining": self._recovery_remaining,
            "window_mean": mean_val,
            "window_std": std_val,
            "window_size_current": len(self._loss_window),
            "total_loss_steps": len(self._loss_history_full),
            "spike_log_last_5": self._spike_log[-5:],
        }

    # ------------------------------------------------------------------
    # Özellikler (Properties)
    # ------------------------------------------------------------------

    @property
    def spike_count(self) -> int:
        """Toplam tespit edilen spike sayısı."""
        return self._spike_count

    @property
    def loss_history(self) -> List[float]:
        """
        Tüm loss geçmişinin kopyası (değiştirilemez referans vermemek için).
        """
        return list(self._loss_history_full)

    # ------------------------------------------------------------------
    # Dahili Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _window_stats(self) -> tuple[float, float]:
        """
        Mevcut pencere için mean ve std hesaplar.

        Returns:
            (mean, std) tuple'ı. Pencere boşsa (0.0, 0.0) döner.
        """
        if not self._loss_window:
            return 0.0, 0.0
        values = list(self._loss_window)
        mean_val = sum(values) / len(values)
        if len(values) < 2:
            return mean_val, 0.0
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = math.sqrt(variance)
        return mean_val, std_val

    def _handle_spike(
        self, loss: float, epoch: int, reason: str
    ) -> None:
        """
        Spike tespiti sonrası ortak işlemleri gerçekleştirir:
        sayaç güncelleme, log yazma.
        """
        self._spike_count += 1
        self._last_spike_epoch = epoch
        mean_val, std_val = self._window_stats()

        spike_record: Dict[str, Any] = {
            "epoch": epoch,
            "loss": loss,
            "window_mean": mean_val,
            "window_std": std_val,
            "reason": reason,
            "timestamp": time.time(),
        }
        self._spike_log.append(spike_record)

        # Log'u 5000 kayıtla sınırla
        if len(self._spike_log) > 5000:
            self._spike_log = self._spike_log[-5000:]

        logger.warning(
            "Spike #%d | epoch=%d | loss=%.4f | reason=%s",
            self._spike_count,
            epoch,
            loss,
            reason,
        )

    def _temporary_lr_reduce(self) -> None:
        """
        Spike anında LR'ı geçici olarak azaltır.
        Mevcut LR'ları kaydeder (recovery sonrası restore için).
        """
        self._original_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        for pg in self.optimizer.param_groups:
            old_lr = pg["lr"]
            pg["lr"] = old_lr * self.spike_lr_factor
            logger.warning(
                "Spike LR azaltması: %.6f → %.6f", old_lr, pg["lr"]
            )

    def _warmup_restart(self) -> None:
        """
        Sürekli spike durumunda LR'ı çok küçük bir değere çekerek
        warmup benzeri yeniden başlatma simüle eder.
        Başlangıç LR'ının %1'i ile başlar.
        """
        self._original_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        for i, pg in enumerate(self.optimizer.param_groups):
            restart_lr = self._base_lrs[i] * 0.01
            logger.warning(
                "Warmup restart: LR %.6f → %.6f", pg["lr"], restart_lr
            )
            pg["lr"] = restart_lr

    def __repr__(self) -> str:
        return (
            f"LossSpikeDetector("
            f"window={self.window_size}, "
            f"n_sigma={self.n_sigma}, "
            f"spikes={self._spike_count}, "
            f"in_recovery={self.is_in_recovery()})"
        )
