"""
nan_recovery.py
===============
Cevahir Türkçe Dil Modeli - NaN/Inf Kurtarma Sistemi

NaN ve Inf değerlerini eğitim sırasında tespit edip otomatik olarak
düzelten modül. Gradient patlamaları ve sayısal kararsızlıkları yakalar.

Strateji:
    1. NaN gradient tespit → gradientleri sıfırla (batch'i atla)
    2. Ardışık NaN sayısı > nan_tolerance → LR yarıya indir
    3. Ardışık NaN sayısı > critical_threshold → son checkpoint'e dön

Referans: Sundararajan et al. 2017 - Training stability best practices
"""

from __future__ import annotations

import logging
import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NaNRecoveryAction(Enum):
    """NaN/Inf tespiti sonrasında alınacak kurtarma eylemleri."""
    ZERO_GRADIENTS = "zero_grad"          # Gradientleri sıfırla, batch'i atla
    REDUCE_LR = "reduce_lr"               # Öğrenme hızını düşür
    RESTORE_CHECKPOINT = "restore_checkpoint"  # Son checkpoint'e dön
    SKIP_BATCH = "skip_batch"             # Batch'i tamamen atla


class NaNRecovery:
    """
    NaN/Inf tespiti ve otomatik kurtarma sistemi.

    Strateji:
        1. NaN gradient tespit → gradientleri sıfırla (skip batch)
        2. Ardışık NaN sayısı > nan_tolerance → LR yarıya indir
        3. Ardışık NaN sayısı > critical_threshold → son checkpoint'e dön

    Referans: Sundararajan et al. 2017 - training stability best practices

    Args:
        optimizer: PyTorch optimizer nesnesi.
        nan_tolerance: Ardışık NaN sayısı bu eşiği geçince LR azaltılır.
        critical_threshold: Bu eşiği geçince checkpoint'e dönülür.
        lr_reduction_factor: LR azaltma çarpanı (0 < f < 1).
        checkpoint_manager: Opsiyonel checkpoint yöneticisi (restore için).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        nan_tolerance: int = 3,
        critical_threshold: int = 10,
        lr_reduction_factor: float = 0.5,
        checkpoint_manager: Optional[Any] = None,
    ) -> None:
        if not (0.0 < lr_reduction_factor < 1.0):
            raise ValueError(
                f"lr_reduction_factor 0 ile 1 arasında olmalı, alınan: {lr_reduction_factor}"
            )
        if nan_tolerance >= critical_threshold:
            raise ValueError(
                "nan_tolerance, critical_threshold'dan küçük olmalı."
            )

        self.optimizer = optimizer
        self.nan_tolerance = nan_tolerance
        self.critical_threshold = critical_threshold
        self.lr_reduction_factor = lr_reduction_factor
        self.checkpoint_manager = checkpoint_manager

        # --- Dahili sayaçlar ve istatistikler ---
        self._consecutive_nans: int = 0
        self._total_recoveries: int = 0
        self._action_history: List[Dict[str, Any]] = []
        self._lr_reduction_count: int = 0
        self._checkpoint_restore_count: int = 0
        self._skip_batch_count: int = 0

        logger.info(
            "NaNRecovery başlatıldı | tolerance=%d | critical=%d | lr_factor=%.3f",
            nan_tolerance,
            critical_threshold,
            lr_reduction_factor,
        )

    # ------------------------------------------------------------------
    # Herkese Açık API
    # ------------------------------------------------------------------

    def check_loss(self, loss: torch.Tensor) -> bool:
        """
        Loss tensöründe NaN veya Inf var mı kontrol eder.

        Args:
            loss: Hesaplanan kayıp tensörü.

        Returns:
            True  → loss geçerli (NaN/Inf yok).
            False → loss geçersiz (NaN veya Inf içeriyor).
        """
        if not isinstance(loss, torch.Tensor):
            logger.warning("check_loss: Tensor olmayan değer alındı: %s", type(loss))
            return False

        loss_value = loss.item()
        if math.isnan(loss_value):
            logger.warning("NaN loss tespit edildi: %s", loss_value)
            return False
        if math.isinf(loss_value):
            logger.warning("Inf loss tespit edildi: %s", loss_value)
            return False
        return True

    def check_gradients(self, model: nn.Module) -> bool:
        """
        Modelin tüm gradientlerinde NaN veya Inf var mı kontrol eder.

        Args:
            model: Kontrol edilecek PyTorch modeli.

        Returns:
            True  → tüm gradientler geçerli.
            False → en az bir gradient NaN veya Inf içeriyor.
        """
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_data = param.grad.data
            if torch.isnan(grad_data).any():
                logger.warning("NaN gradient tespit edildi | parametre: %s", name)
                return False
            if torch.isinf(grad_data).any():
                logger.warning("Inf gradient tespit edildi | parametre: %s", name)
                return False
        return True

    def handle_nan_loss(
        self, model: nn.Module, loss: torch.Tensor
    ) -> NaNRecoveryAction:
        """
        NaN/Inf loss durumunda uygun kurtarma eylemini seçer ve uygular.

        Karar ağacı:
            consecutive_nans > critical_threshold → RESTORE_CHECKPOINT
            consecutive_nans > nan_tolerance      → REDUCE_LR + ZERO_GRADIENTS
            diğer                                 → SKIP_BATCH

        Args:
            model: Eğitilen PyTorch modeli.
            loss:  NaN/Inf içerdiği tespit edilmiş kayıp tensörü.

        Returns:
            Uygulanan NaNRecoveryAction.
        """
        self._consecutive_nans += 1
        self._total_recoveries += 1

        logger.warning(
            "NaN loss işleniyor | ardışık NaN: %d / tolerance: %d / critical: %d",
            self._consecutive_nans,
            self.nan_tolerance,
            self.critical_threshold,
        )

        # Kritik eşik aşıldı → checkpoint'e dön
        if self._consecutive_nans > self.critical_threshold:
            action = self._restore_checkpoint(model)
        # Tolerans eşiği aşıldı → LR azalt
        elif self._consecutive_nans > self.nan_tolerance:
            self._zero_gradients(model)
            self._reduce_lr()
            action = NaNRecoveryAction.REDUCE_LR
        else:
            # İlk birkaç NaN → sadece batch'i atla
            self._zero_gradients(model)
            action = NaNRecoveryAction.SKIP_BATCH
            self._skip_batch_count += 1

        self._record_action(action, context="nan_loss")
        return action

    def handle_nan_gradients(self, model: nn.Module) -> NaNRecoveryAction:
        """
        NaN/Inf gradient durumunda uygun kurtarma eylemini seçer ve uygular.

        Karar ağacı:
            consecutive_nans > critical_threshold → RESTORE_CHECKPOINT
            consecutive_nans > nan_tolerance      → REDUCE_LR + ZERO_GRADIENTS
            diğer                                 → ZERO_GRADIENTS

        Args:
            model: NaN gradient içeren PyTorch modeli.

        Returns:
            Uygulanan NaNRecoveryAction.
        """
        self._consecutive_nans += 1
        self._total_recoveries += 1

        logger.warning(
            "NaN gradient işleniyor | ardışık NaN: %d / tolerance: %d / critical: %d",
            self._consecutive_nans,
            self.nan_tolerance,
            self.critical_threshold,
        )

        # Kritik eşik aşıldı → checkpoint'e dön
        if self._consecutive_nans > self.critical_threshold:
            action = self._restore_checkpoint(model)
        elif self._consecutive_nans > self.nan_tolerance:
            self._zero_gradients(model)
            self._reduce_lr()
            action = NaNRecoveryAction.REDUCE_LR
        else:
            self._zero_gradients(model)
            action = NaNRecoveryAction.ZERO_GRADIENTS

        self._record_action(action, context="nan_gradients")
        return action

    def reset_counter(self) -> None:
        """
        Başarılı bir batch tamamlandığında ardışık NaN sayacını sıfırlar.
        Her başarılı ileri/geri geçişten sonra çağrılmalıdır.
        """
        if self._consecutive_nans > 0:
            logger.debug(
                "NaN sayacı sıfırlandı (önceki: %d)", self._consecutive_nans
            )
        self._consecutive_nans = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Kurtarma istatistiklerini döndürür.

        Returns:
            Dict içeriği:
                consecutive_nans       : Anlık ardışık NaN sayısı
                total_recoveries       : Toplam kurtarma girişimi
                lr_reduction_count     : LR azaltma sayısı
                checkpoint_restores    : Checkpoint geri yükleme sayısı
                skip_batch_count       : Atlanan batch sayısı
                current_lr             : Mevcut optimizer LR değerleri
                action_history_last_10 : Son 10 eylem kaydı
        """
        current_lrs = [
            pg["lr"] for pg in self.optimizer.param_groups
        ]
        return {
            "consecutive_nans": self._consecutive_nans,
            "total_recoveries": self._total_recoveries,
            "lr_reduction_count": self._lr_reduction_count,
            "checkpoint_restores": self._checkpoint_restore_count,
            "skip_batch_count": self._skip_batch_count,
            "current_lr": current_lrs,
            "action_history_last_10": self._action_history[-10:],
        }

    # ------------------------------------------------------------------
    # Özellikler (Properties)
    # ------------------------------------------------------------------

    @property
    def consecutive_nans(self) -> int:
        """Mevcut ardışık NaN/Inf sayısı."""
        return self._consecutive_nans

    @property
    def total_recoveries(self) -> int:
        """Toplam NaN/Inf kurtarma girişimi sayısı."""
        return self._total_recoveries

    # ------------------------------------------------------------------
    # Dahili Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _zero_gradients(self, model: nn.Module) -> None:
        """
        Tüm model gradientlerini sıfırlar.
        NaN gradient'in optimizer adımına taşınmasını önler.
        """
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        logger.debug("Tüm gradientler sıfırlandı.")

    def _reduce_lr(self) -> None:
        """
        Optimizer'daki tüm param gruplarının LR değerini
        lr_reduction_factor ile çarpar.
        """
        for pg in self.optimizer.param_groups:
            old_lr = pg["lr"]
            pg["lr"] = old_lr * self.lr_reduction_factor
            logger.warning(
                "LR azaltıldı: %.6f → %.6f", old_lr, pg["lr"]
            )
        self._lr_reduction_count += 1

    def _restore_checkpoint(self, model: nn.Module) -> NaNRecoveryAction:
        """
        checkpoint_manager mevcutsa son checkpoint'i yükler.
        Mevcut değilse yalnızca uyarı loglar.
        """
        if self.checkpoint_manager is not None:
            try:
                self.checkpoint_manager.load_last(model)
                logger.critical(
                    "Kritik NaN eşiği aşıldı (%d). Checkpoint geri yüklendi.",
                    self._consecutive_nans,
                )
                self._checkpoint_restore_count += 1
                self._consecutive_nans = 0  # Sıfırla — yeni başlangıç
            except Exception as exc:
                logger.error("Checkpoint geri yükleme başarısız: %s", exc)
        else:
            logger.critical(
                "Kritik NaN eşiği aşıldı (%d). checkpoint_manager tanımlı değil!",
                self._consecutive_nans,
            )
        return NaNRecoveryAction.RESTORE_CHECKPOINT

    def _record_action(self, action: NaNRecoveryAction, context: str) -> None:
        """Eylem geçmişine yeni kayıt ekler."""
        self._action_history.append(
            {
                "action": action.value,
                "context": context,
                "consecutive_nans": self._consecutive_nans,
                "timestamp": time.time(),
            }
        )
        # Geçmişi en fazla 1000 kayıtla sınırla (bellek tasarrufu)
        if len(self._action_history) > 1000:
            self._action_history = self._action_history[-1000:]

    def __repr__(self) -> str:
        return (
            f"NaNRecovery("
            f"tolerance={self.nan_tolerance}, "
            f"critical={self.critical_threshold}, "
            f"consecutive={self._consecutive_nans}, "
            f"total_recoveries={self._total_recoveries})"
        )
