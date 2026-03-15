"""
divergence_detector.py
=======================
Cevahir Türkçe Dil Modeli - Divergence ve Convergence Takip Sistemi

Eğitim sırasında train/validation loss ilişkisini izler, overfitting,
divergence ve plateau durumlarını tespit eder. Early stopping uygular.

Divergence Tespiti:
    Overfitting  : train_loss / val_loss > overfit_ratio (varsayılan 3.0)
    Diverging    : val_loss son patience epoch içinde monoton artıyor
    Plateau      : |loss_change| < plateau_threshold son patience epoch içinde

Erken Durdurma (Early Stopping):
    val_loss > best_val_loss + min_delta durumunda patience sayacı artar.
    Patience dolduğunda eğitim durdurulmalıdır.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConvergenceStatus(Enum):
    """Modelin mevcut convergence durumunu temsil eden enum."""
    CONVERGING = "converging"      # Loss azalıyor, sağlıklı eğitim
    PLATEAU = "plateau"            # Loss değişmiyor, ilerleme durdu
    OVERFITTING = "overfitting"    # Train/val arasında büyük uçurum
    DIVERGING = "diverging"        # Val loss monoton artıyor
    HEALTHY = "healthy"            # Her şey normal, stabil


class DivergenceDetector:
    """
    Training/Validation divergence ve convergence takibi.

    Her epoch sonunda train_loss ve val_loss değerleri verilerek güncellenir.
    Sistem overfitting, divergence ve plateau durumlarını tespit eder
    ve uygun ConvergenceStatus döner.

    Args:
        patience:           Early stopping için maksimum kötüleşme epoch sayısı.
        min_delta:          Val loss iyileşmesi için minimum eşik.
        overfit_ratio:      train_loss/val_loss bu oranı geçerse overfitting sayılır.
        plateau_threshold:  Loss değişim eşiği (bu altında plateau sayılır).
        diverge_window:     Monoton artış için bakılacak epoch penceresi.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        overfit_ratio: float = 3.0,
        plateau_threshold: float = 0.0005,
        diverge_window: int = 5,
    ) -> None:
        if patience < 1:
            raise ValueError("patience en az 1 olmalı.")
        if min_delta < 0:
            raise ValueError("min_delta negatif olamaz.")
        if overfit_ratio <= 1.0:
            raise ValueError("overfit_ratio 1.0'dan büyük olmalı.")
        if plateau_threshold < 0:
            raise ValueError("plateau_threshold negatif olamaz.")
        if diverge_window < 2:
            raise ValueError("diverge_window en az 2 olmalı.")

        self.patience = patience
        self.min_delta = min_delta
        self.overfit_ratio = overfit_ratio
        self.plateau_threshold = plateau_threshold
        self.diverge_window = diverge_window

        # --- Dahili durum ---
        self._best_val_loss: float = math.inf
        self._best_epoch: int = 0
        self._patience_counter: int = 0
        self._current_status: ConvergenceStatus = ConvergenceStatus.HEALTHY
        self._total_epochs_tracked: int = 0

        # Geçmiş kayıtları
        self._train_loss_history: List[float] = []
        self._val_loss_history: List[float] = []
        self._epoch_history: List[int] = []

        # Divergence tespiti için kayan pencere
        self._val_loss_window: Deque[float] = deque(maxlen=diverge_window)
        self._overfit_ratio_current: float = 0.0

        logger.info(
            "DivergenceDetector başlatıldı | patience=%d | min_delta=%.4f | "
            "overfit_ratio=%.1f | plateau_threshold=%.4f | diverge_window=%d",
            patience,
            min_delta,
            overfit_ratio,
            plateau_threshold,
            diverge_window,
        )

    # ------------------------------------------------------------------
    # Herkese Açık API
    # ------------------------------------------------------------------

    def update(
        self, train_loss: float, val_loss: float, epoch: int
    ) -> ConvergenceStatus:
        """
        Her epoch sonunda train ve validation loss değerlerini günceller
        ve mevcut convergence durumunu hesaplar.

        Args:
            train_loss: Epoch'un ortalama eğitim kaybı.
            val_loss:   Epoch'un doğrulama kaybı.
            epoch:      Mevcut epoch numarası (0-indexed veya 1-indexed).

        Returns:
            Hesaplanan ConvergenceStatus.
        """
        # Geçersiz değer koruması
        if math.isnan(train_loss) or math.isinf(train_loss):
            logger.warning("Geçersiz train_loss: %.4f. DIVERGING olarak işaretlendi.", train_loss)
            self._current_status = ConvergenceStatus.DIVERGING
            return self._current_status
        if math.isnan(val_loss) or math.isinf(val_loss):
            logger.warning("Geçersiz val_loss: %.4f. DIVERGING olarak işaretlendi.", val_loss)
            self._current_status = ConvergenceStatus.DIVERGING
            return self._current_status

        # Geçmişe ekle
        self._train_loss_history.append(train_loss)
        self._val_loss_history.append(val_loss)
        self._epoch_history.append(epoch)
        self._val_loss_window.append(val_loss)
        self._total_epochs_tracked += 1
        # [MEM-FIX] Unbounded history listelerini sınırla: max 2000 epoch kaydı tut
        _MAX_HISTORY = 2000
        if len(self._train_loss_history) > _MAX_HISTORY:
            self._train_loss_history = self._train_loss_history[-_MAX_HISTORY:]
            self._val_loss_history = self._val_loss_history[-_MAX_HISTORY:]
            self._epoch_history = self._epoch_history[-_MAX_HISTORY:]

        # --- Early stopping güncellemesi ---
        if val_loss < self._best_val_loss - self.min_delta:
            # İyileşme var
            self._best_val_loss = val_loss
            self._best_epoch = epoch
            self._patience_counter = 0
            logger.debug(
                "Val loss iyileşti: %.4f | epoch=%d", val_loss, epoch
            )
        else:
            # Yeterli iyileşme yok
            self._patience_counter += 1
            logger.debug(
                "Val loss iyileşmedi | patience: %d/%d",
                self._patience_counter,
                self.patience,
            )

        # --- Overfit ratio hesapla ---
        if val_loss > 0:
            self._overfit_ratio_current = train_loss / val_loss
        else:
            self._overfit_ratio_current = 0.0

        # --- Durum tespiti (öncelik sırasıyla) ---
        status = self._determine_status(train_loss, val_loss)
        self._current_status = status

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | status=%s | patience=%d/%d",
            epoch,
            train_loss,
            val_loss,
            status.value,
            self._patience_counter,
            self.patience,
        )

        return status

    def should_stop_early(self) -> bool:
        """
        Early stopping koşulunun sağlanıp sağlanmadığını döndürür.

        Returns:
            True  → eğitim durdurulmalı.
            False → eğitime devam edilmeli.
        """
        should_stop = self._patience_counter >= self.patience
        if should_stop:
            logger.warning(
                "Early stopping tetiklendi | patience=%d/%d | "
                "best_val_loss=%.4f | best_epoch=%d",
                self._patience_counter,
                self.patience,
                self._best_val_loss,
                self._best_epoch,
            )
        return should_stop

    def get_status(self) -> ConvergenceStatus:
        """
        Mevcut convergence durumunu döndürür.

        Returns:
            En son hesaplanan ConvergenceStatus.
        """
        return self._current_status

    def get_stats(self) -> Dict[str, Any]:
        """
        Detaylı istatistik sözlüğü döndürür.

        Returns:
            Dict içeriği:
                best_val_loss          : En iyi validation loss
                best_epoch             : En iyi validation loss'un gerçekleştiği epoch
                patience_counter       : Mevcut patience sayacı
                patience_limit         : Maksimum patience
                status                 : Mevcut convergence durumu (string)
                overfit_ratio_current  : Mevcut train/val loss oranı
                total_epochs_tracked   : Takip edilen toplam epoch sayısı
                should_stop_early      : Early stopping koşulu sağlandı mı?
                train_loss_last        : Son train loss değeri
                val_loss_last          : Son val loss değeri
                val_loss_trend         : Son diverge_window içindeki trend (list)
        """
        train_last = self._train_loss_history[-1] if self._train_loss_history else None
        val_last = self._val_loss_history[-1] if self._val_loss_history else None

        return {
            "best_val_loss": self._best_val_loss if not math.isinf(self._best_val_loss) else None,
            "best_epoch": self._best_epoch,
            "patience_counter": self._patience_counter,
            "patience_limit": self.patience,
            "status": self._current_status.value,
            "overfit_ratio_current": round(self._overfit_ratio_current, 4),
            "total_epochs_tracked": self._total_epochs_tracked,
            "should_stop_early": self.should_stop_early(),
            "train_loss_last": train_last,
            "val_loss_last": val_last,
            "val_loss_trend": list(self._val_loss_window),
        }

    # ------------------------------------------------------------------
    # Özellikler (Properties)
    # ------------------------------------------------------------------

    @property
    def best_val_loss(self) -> float:
        """Şimdiye kadar görülen en düşük validation loss."""
        return self._best_val_loss

    @property
    def patience_counter(self) -> int:
        """Mevcut patience sayacı (iyileşme olmayan epoch sayısı)."""
        return self._patience_counter

    # ------------------------------------------------------------------
    # Dahili Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _determine_status(
        self, train_loss: float, val_loss: float
    ) -> ConvergenceStatus:
        """
        Mevcut train/val loss değerlerine göre convergence durumunu belirler.

        Öncelik sırası:
            1. DIVERGING  → Val loss monoton artıyor (en kritik)
            2. OVERFITTING → Train/val oranı aşırı yüksek
            3. PLATEAU    → Loss değişimi çok küçük
            4. CONVERGING → Loss azalıyor
            5. HEALTHY    → Stabil durum

        Returns:
            ConvergenceStatus enum değeri.
        """
        # 1. Divergence kontrolü (en yüksek öncelik)
        if self._is_diverging():
            return ConvergenceStatus.DIVERGING

        # 2. Overfitting kontrolü
        if self._is_overfitting(train_loss, val_loss):
            return ConvergenceStatus.OVERFITTING

        # 3. Plateau kontrolü
        if self._is_plateau():
            return ConvergenceStatus.PLATEAU

        # 4. Converging kontrolü
        if self._is_converging():
            return ConvergenceStatus.CONVERGING

        # 5. Varsayılan: sağlıklı
        return ConvergenceStatus.HEALTHY

    def _is_diverging(self) -> bool:
        """
        Val loss son diverge_window epoch içinde monoton artıyor mu?

        Monoton artış: her değer bir öncekinden büyük veya eşit.
        Minimum diverge_window kadar veri gerekir.
        """
        if len(self._val_loss_window) < self.diverge_window:
            return False

        window = list(self._val_loss_window)
        # Monoton artış: her eleman bir öncekinden büyük
        is_monoton_increasing = all(
            window[i] >= window[i - 1]
            for i in range(1, len(window))
        )

        if is_monoton_increasing:
            logger.warning(
                "Divergence tespit edildi: val_loss son %d epoch monoton artıyor | "
                "pencere=%s",
                self.diverge_window,
                [f"{v:.4f}" for v in window],
            )

        return is_monoton_increasing

    def _is_overfitting(self, train_loss: float, val_loss: float) -> bool:
        """
        Train/val loss oranı overfit_ratio'yu aşıyor mu?

        Not: val_loss çok küçükse (0'a yakın) bu oran yanıltıcı olabilir,
        bu yüzden minimum val_loss koruması uygulanır.
        """
        if val_loss <= 1e-8:
            return False

        ratio = train_loss / val_loss
        is_overfit = ratio > self.overfit_ratio

        if is_overfit:
            logger.warning(
                "Overfitting tespit edildi | train/val=%.2f > threshold=%.2f",
                ratio,
                self.overfit_ratio,
            )

        return is_overfit

    def _is_plateau(self) -> bool:
        """
        Son birkaç epoch içinde val_loss değişimi plateau_threshold'un altında mı?

        Minimum iki değer gerekir.
        """
        if len(self._val_loss_history) < 2:
            return False

        # Son birkaç değerdeki maksimum değişim
        recent = self._val_loss_history[-min(self.diverge_window, len(self._val_loss_history)):]
        if len(recent) < 2:
            return False

        max_change = max(abs(recent[i] - recent[i - 1]) for i in range(1, len(recent)))
        is_plateau = max_change < self.plateau_threshold

        if is_plateau:
            logger.info(
                "Plateau tespit edildi | maksimum değişim=%.6f < eşik=%.6f",
                max_change,
                self.plateau_threshold,
            )

        return is_plateau

    def _is_converging(self) -> bool:
        """
        Val loss azalıyor mu?

        Son iki değer arasında anlamlı bir düşüş var mı kontrol eder.
        """
        if len(self._val_loss_history) < 2:
            return False

        latest = self._val_loss_history[-1]
        previous = self._val_loss_history[-2]
        return (previous - latest) > self.min_delta

    def __repr__(self) -> str:
        return (
            f"DivergenceDetector("
            f"patience={self.patience}, "
            f"patience_counter={self._patience_counter}, "
            f"best_val_loss={self._best_val_loss:.4f}, "
            f"status={self._current_status.value})"
        )
