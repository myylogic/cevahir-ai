"""
Cevahir V3 - Ana Eğitim Yöneticisi (Facade Pattern)
=====================================================
Bu modül, Cevahir Türkçe dil modelinin tüm eğitim altyapısını
tek bir orkestratör sınıfı üzerinden yöneten TrainingManager'ı implemente eder.

Tasarım Deseni:
    Facade Pattern: Karmaşık alt sistem bileşenlerini (kayıp, gradyan, izleme,
    güvenlik, curriculum vb.) tek bir yüksek seviyeli arayüzde birleştirir.

Bileşen Kategorileri:
    - Çekirdek     : model, optimizer, loss_manager, training_loop
    - Zamanlayıcı  : scheduler, checkpoint_manager
    - İzleme       : tensorboard_manager, training_logger, gradient_health_monitor,
                     token_distribution_monitor, inference_quality_probe
    - Güvenlik     : divergence_detector, loss_spike_detector, nan_recovery
    - Curriculum   : curriculum_manager
    - EMA          : Exponential Moving Average model ağırlıkları

Eğitim Akışı (her epoch):
    1. Curriculum veri ayarı
    2. Eğitim epoch'u (TrainingLoop)
    3. Doğrulama epoch'u (TrainingLoop)
    4. EMA güncelleme
    5. Öğrenme oranı zamanlayıcı adımı
    6. NaN kurtarma kontrolü
    7. Kayıp ani artış tespiti
    8. Iraksama tespiti
    9. Gradyan sağlık raporu
    10. Token dağılım raporu
    11. Çıkarım kalite testi (her probe_interval epoch'ta)
    12. TensorBoard loglama
    13. Checkpoint kaydetme
    14. Erken durdurma kontrolü
    15. Epoch özeti loglama
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from training_management.v3.core.loss_manager import CompositeLossManager, LossConfig
from training_management.v3.core.batch_processor import BatchProcessor
from training_management.v3.core.gradient_manager import GradientManager
from training_management.v3.core.training_loop import (
    TrainingLoop,
    TrainingLoopConfig,
    EpochMetrics,
)

logger = logging.getLogger(__name__)

__all__ = ["TrainingManager", "TrainingManagerConfig"]


# ---------------------------------------------------------------------------
# Konfigürasyon Veri Sınıfı
# ---------------------------------------------------------------------------

@dataclass
class TrainingManagerConfig:
    """
    TrainingManager için tüm konfigürasyon parametrelerini içerir.

    Eğitim Genel:
        total_epochs          : Toplam epoch sayısı
        device                : Hesaplama cihazı ('cuda', 'cpu')
        seed                  : Tekrar üretilebilirlik için rastgele tohum

    Checkpoint:
        checkpoint_dir        : Checkpoint kaydetme dizini
        save_best             : En iyi doğrulama kaybını kaydet
        save_last             : Her epoch sonunda son checkpoint kaydet
        save_every_n_epochs   : Bu kadar epoch'ta bir periyodik checkpoint

    Erken Durdurma:
        early_stopping_patience: Gelişme olmadığında kaç epoch beklenecek
        early_stopping_metric  : İzlenecek metrik ('val_loss', 'val_accuracy')
        early_stopping_mode    : 'min' (kayıp için) veya 'max' (doğruluk için)

    EMA:
        use_ema               : Exponential Moving Average aktif mi?
        ema_decay             : EMA bozunum faktörü (örn. 0.999)

    İzleme:
        probe_interval        : Kaç epoch'ta bir çıkarım kalite testi yapılır
        log_gradient_health   : Gradyan sağlığını logla
        log_token_dist        : Token dağılımını logla

    Güvenlik:
        nan_recovery_enabled  : NaN kurtarma mekanizması aktif mi?
        divergence_window     : Iraksama tespiti için bakış penceresi (epoch)
        spike_multiplier      : Kayıp ani artış eşik çarpanı
    """
    # Eğitim genel
    total_epochs: int = 100
    device: str = "cuda"
    seed: Optional[int] = 42

    # Checkpoint
    checkpoint_dir: str = "checkpoints/v3"
    save_best: bool = True
    save_last: bool = True
    save_every_n_epochs: int = 10

    # Erken durdurma
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"   # 'min' veya 'max'

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.999

    # İzleme
    probe_interval: int = 5            # Kaç epoch'ta bir çıkarım testi
    log_gradient_health: bool = True
    log_token_dist: bool = True
    tensorboard_log_dir: str = "runs/v3"

    # Güvenlik
    nan_recovery_enabled: bool = True
    divergence_window: int = 5
    spike_multiplier: float = 3.0

    # TrainingLoop konfigürasyonu (iç içe)
    loop_config: Optional[TrainingLoopConfig] = None


# ---------------------------------------------------------------------------
# Erken Durdurma Yardımcısı
# ---------------------------------------------------------------------------

class _EarlyStopper:
    """
    Erken durdurma mantığını kapsülleyen yardımcı sınıf.

    Parametreler:
        patience : Bu kadar epoch gelişme olmazsa eğitimi durdur
        mode     : 'min' (kaybı minimize et) veya 'max' (doğruluğu maksimize et)
        delta    : Gelişme sayılması için minimum değişim miktarı
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        delta: float = 1e-4,
    ):
        self.patience = patience
        self.mode = mode
        self.delta = delta

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_epoch = -1

    def step(self, value: float, epoch: int) -> bool:
        """
        Metrik değerini günceller ve erken durdurma gerekip gerekmediğini döndürür.

        Args:
            value: Mevcut epoch'taki metrik değeri
            epoch: Mevcut epoch numarası

        Returns:
            True → eğitimi durdur, False → devam et
        """
        improved = False

        if self.mode == "min":
            if value < self.best_value - self.delta:
                improved = True
        else:  # max
            if value > self.best_value + self.delta:
                improved = True

        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1

        should_stop = self.counter >= self.patience
        if should_stop:
            logger.info(
                f"Erken durdurma tetiklendi! "
                f"En iyi epoch: {self.best_epoch}, "
                f"En iyi değer: {self.best_value:.6f}, "
                f"Sabır sayacı: {self.counter}/{self.patience}"
            )
        return should_stop

    @property
    def is_best(self) -> bool:
        """Sayaç 0 ise son epoch en iyi sonuçtu."""
        return self.counter == 0


# ---------------------------------------------------------------------------
# Ana Eğitim Yöneticisi
# ---------------------------------------------------------------------------

class TrainingManager:
    """
    Cevahir V3 Ana Eğitim Yöneticisi
    ==================================
    Tüm eğitim altyapısını tek bir Facade arayüzünde birleştirir.

    Bu sınıf şu sorumluluklara sahiptir:
        - Epoch döngüsünü yönetmek
        - Tüm bileşenleri doğru sırayla koordine etmek
        - Güvenlik kontrollerini uygulamak
        - Metrikleri kaydetmek ve raporlamak
        - Checkpoint yönetimi
        - Erken durdurma

    Bağımlılık Enjeksiyonu:
        Tüm bileşenler constructor üzerinden enjekte edilir.
        Bu; test edilebilirlik, bileşen değiştirme ve
        mock kullanımı için esneklik sağlar.

    Kullanım:
        # Temel kullanım
        manager = TrainingManager(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_manager=loss_manager,
            config=config,
        )
        results = manager.train(start_epoch=0, end_epoch=100)

        # Factory method ile
        manager = TrainingManager.from_config(model, loaders, optimizer, config_dict)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        optimizer: torch.optim.Optimizer,
        loss_manager: CompositeLossManager,
        config: TrainingManagerConfig,
        # Opsiyonel bileşenler
        scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        ema: Optional[Any] = None,
        tensorboard_manager: Optional[Any] = None,
        training_logger: Optional[Any] = None,
        divergence_detector: Optional[Any] = None,
        loss_spike_detector: Optional[Any] = None,
        nan_recovery: Optional[Any] = None,
        inference_quality_probe: Optional[Any] = None,
        curriculum_manager: Optional[Any] = None,
        gradient_health_monitor: Optional[Any] = None,
        token_distribution_monitor: Optional[Any] = None,
        training_loop: Optional[TrainingLoop] = None,
    ):
        """
        Args:
            model                    : Eğitilecek PyTorch dil modeli
            train_loader             : Eğitim veri yükleyicisi
            val_loader               : Doğrulama veri yükleyicisi
            optimizer                : PyTorch optimizer
            loss_manager             : CompositeLossManager nesnesi
            config                   : TrainingManagerConfig nesnesi

            scheduler                : Öğrenme oranı zamanlayıcısı (v3.utils.TrainingScheduler)
            checkpoint_manager       : Checkpoint yöneticisi (v3.utils.CheckpointManager)
            ema                      : EMA yöneticisi (v3.utils.EMA)
            tensorboard_manager      : TensorBoard yöneticisi (v3.monitoring.TensorBoardManager)
            training_logger          : Eğitim logger'ı (v3.utils.TrainingLogger)
            divergence_detector      : Iraksama dedektörü (v3.safety.DivergenceDetector)
            loss_spike_detector      : Kayıp ani artış dedektörü (v3.safety.LossSpikeDetector)
            nan_recovery             : NaN kurtarma (v3.safety.NaNRecovery)
            inference_quality_probe  : Çıkarım kalite testi (v3.monitoring.InferenceQualityProbe)
            curriculum_manager       : Curriculum yöneticisi (v3.curriculum.CurriculumManager)
            gradient_health_monitor  : Gradyan sağlık monitörü (v3.monitoring.GradientHealthMonitor)
            token_distribution_monitor: Token dağılım monitörü (v3.monitoring.TokenDistributionMonitor)
            training_loop            : Hazır TrainingLoop (None → config'den otomatik oluşturulur)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_manager = loss_manager
        self.config = config

        # Opsiyonel bileşenler (None olabilir → güvenli çağrılar gerekir)
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.ema = ema
        self.tensorboard_manager = tensorboard_manager
        self.training_logger = training_logger
        self.divergence_detector = divergence_detector
        self.loss_spike_detector = loss_spike_detector
        self.nan_recovery = nan_recovery
        self.inference_quality_probe = inference_quality_probe
        self.curriculum_manager = curriculum_manager
        self.gradient_health_monitor = gradient_health_monitor
        self.token_distribution_monitor = token_distribution_monitor

        # Eğitim döngüsü
        if training_loop is not None:
            self.training_loop = training_loop
        else:
            loop_cfg = config.loop_config or TrainingLoopConfig(device=config.device)
            self.training_loop = TrainingLoop(
                model=model,
                optimizer=optimizer,
                loss_manager=loss_manager,
                config=loop_cfg,
            )

        # Erken durdurma
        self._early_stopper = _EarlyStopper(
            patience=config.early_stopping_patience,
            mode=config.early_stopping_mode,
        )

        # Dahili durum
        self._best_val_metric: float = float("inf") if config.early_stopping_mode == "min" else float("-inf")
        self._best_epoch: int = -1
        self._history: List[Dict] = []

        # Rastgele tohum
        if config.seed is not None:
            self._set_seed(config.seed)

        logger.info(
            f"TrainingManager başlatıldı | "
            f"total_epochs={config.total_epochs}, "
            f"device={config.device}, "
            f"early_stopping=patience:{config.early_stopping_patience}"
        )

    # ------------------------------------------------------------------
    # Yardımcı Metotlar
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Tekrar üretilebilirlik için rastgele tohumları ayarlar."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"Rastgele tohum ayarlandı: {seed}")

    def _get_metric_value(self, metrics: EpochMetrics, metric_name: str) -> float:
        """
        EpochMetrics'ten metrik değerini güvenli şekilde çıkarır.

        Args:
            metrics    : Epoch metrikleri
            metric_name: 'val_loss', 'val_accuracy' vb.

        Returns:
            Metrik değeri (float)
        """
        # 'val_loss' → 'loss', 'val_accuracy' → 'accuracy'
        key = metric_name.replace("val_", "").replace("train_", "")
        return metrics.get(key, float("inf"))

    def _safe_call(self, method_name: str, obj: Optional[Any], *args, **kwargs) -> Any:
        """
        Opsiyonel bileşen metotlarını güvenli şekilde çağırır.
        Bileşen None ise veya metot yoksa sessizce atlar.

        Args:
            method_name: Çağrılacak metot adı
            obj        : Bileşen nesnesi (None olabilir)
            *args      : Pozisyonel argümanlar
            **kwargs   : Anahtar kelime argümanları

        Returns:
            Metot dönüş değeri veya None
        """
        if obj is None:
            return None
        method = getattr(obj, method_name, None)
        if method is None:
            logger.debug(f"{type(obj).__name__}.{method_name} bulunamadı, atlanıyor")
            return None
        try:
            return method(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"{type(obj).__name__}.{method_name}() çağrısı başarısız: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Ana Eğitim Döngüsü
    # ------------------------------------------------------------------

    def train(
        self,
        start_epoch: int = 0,
        end_epoch: Optional[int] = None,
    ) -> Dict:
        """
        Ana eğitim döngüsü - tüm bileşenleri koordine eder.

        Her epoch şu adımları takip eder:
            1.  Curriculum veri ayarı (curriculum_manager.adjust_data)
            2.  Eğitim epoch'u      (TrainingLoop.train_epoch)
            3.  Doğrulama epoch'u   (TrainingLoop.validate_epoch)
            4.  EMA güncelleme      (ema.update)
            5.  Öğrenme oranı adımı (scheduler.step)
            6.  NaN kurtarma        (nan_recovery.check)
            7.  Kayıp ani artış     (loss_spike_detector.detect)
            8.  Iraksama tespiti    (divergence_detector.detect)
            9.  Gradyan sağlık      (gradient_health_monitor.log)
            10. Token dağılım       (token_distribution_monitor.log)
            11. Çıkarım kalite testi (inference_quality_probe, her probe_interval epoch)
            12. TensorBoard loglama  (tensorboard_manager.log_epoch)
            13. Checkpoint kaydetme  (checkpoint_manager.save)
            14. Erken durdurma       (EarlyStopper.step)
            15. Epoch özeti          (training_logger.log_epoch_summary)

        Args:
            start_epoch: Başlangıç epoch numarası (devam ettirme için)
            end_epoch  : Bitiş epoch numarası (None → config.total_epochs)

        Returns:
            Eğitim geçmişi ve en iyi sonuçları içeren sözlük:
                - history       : Her epoch'un train/val metriklerinin listesi
                - best_epoch    : En iyi doğrulama metriğinin epoch'u
                - best_metric   : En iyi doğrulama metriği değeri
                - total_epochs  : Tamamlanan epoch sayısı
                - early_stopped : Erken durdurma tetiklenip tetiklenmediği
        """
        _end_epoch = end_epoch if end_epoch is not None else self.config.total_epochs
        early_stopped = False
        cfg = self.config

        logger.info(
            f"Eğitim başlıyor: epoch {start_epoch} → {_end_epoch - 1}"
        )
        epoch_start_time = time.time()

        for epoch in range(start_epoch, _end_epoch):
            epoch_t0 = time.time()
            logger.info(f"{'='*60}")
            logger.info(f"EPOCH {epoch}/{_end_epoch - 1} başlıyor")

            # ==============================================================
            # 1. Curriculum: Veri ayarlaması
            # ==============================================================
            self._safe_call(
                "adjust_data",
                self.curriculum_manager,
                epoch=epoch,
                train_loader=self.train_loader,
            )

            # ==============================================================
            # 2. Eğitim Epoch'u
            # ==============================================================
            try:
                train_metrics: EpochMetrics = self.training_loop.train_epoch(
                    data_loader=self.train_loader,
                    epoch=epoch,
                )
            except Exception as e:
                logger.error(f"Eğitim epoch {epoch} başarısız: {e}", exc_info=True)
                # NaN kurtarma denenilebilir
                self._safe_call("trigger", self.nan_recovery, model=self.model)
                continue

            logger.info(
                f"Eğitim | Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f} | "
                f"PPL: {train_metrics['perplexity']:.2f} | "
                f"Hız: {train_metrics['tokens_per_sec']:.0f} tok/s"
            )

            # ==============================================================
            # 3. Doğrulama Epoch'u
            # ==============================================================
            try:
                val_metrics: EpochMetrics = self.training_loop.validate_epoch(
                    data_loader=self.val_loader,
                )
            except Exception as e:
                logger.error(f"Doğrulama epoch {epoch} başarısız: {e}", exc_info=True)
                val_metrics = self._empty_metrics()

            logger.info(
                f"Doğrulama | Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | "
                f"PPL: {val_metrics['perplexity']:.2f}"
            )

            # ==============================================================
            # 4. EMA Güncelleme
            # ==============================================================
            if cfg.use_ema and self.ema is not None:
                self._safe_call("update", self.ema, model=self.model)

            # ==============================================================
            # 5. Öğrenme Oranı Zamanlayıcı Adımı
            # ==============================================================
            if self.scheduler is not None:
                # Bazı zamanlayıcılar val_loss ister (ReduceLROnPlateau vb.)
                try:
                    sched_method = getattr(self.scheduler, "step", None)
                    if sched_method is not None:
                        import inspect
                        sig = inspect.signature(sched_method)
                        params = list(sig.parameters.keys())
                        if "metrics" in params or "val_loss" in params:
                            self.scheduler.step(val_metrics["loss"])
                        else:
                            self.scheduler.step()
                except Exception as e:
                    logger.warning(f"Scheduler adımı başarısız: {e}")

            current_lr = self._get_current_lr()
            logger.debug(f"Mevcut öğrenme oranı: {current_lr:.2e}")

            # ==============================================================
            # 6. NaN Kurtarma Kontrolü
            # ==============================================================
            if cfg.nan_recovery_enabled and self.nan_recovery is not None:
                nan_detected = (
                    not torch.isfinite(torch.tensor(train_metrics["loss"])).item()
                    or not torch.isfinite(torch.tensor(val_metrics["loss"])).item()
                )
                if nan_detected:
                    logger.error(f"Epoch {epoch}: NaN/Inf kayıp tespit edildi!")
                    recovery_success = self._safe_call(
                        "recover",
                        self.nan_recovery,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                    )
                    if not recovery_success:
                        logger.error("NaN kurtarma başarısız, eğitim durduruluyor")
                        break

            # ==============================================================
            # 7. Kayıp Ani Artış Tespiti
            # ==============================================================
            spike_detected = self._safe_call(
                "detect",
                self.loss_spike_detector,
                loss=train_metrics["loss"],
                epoch=epoch,
            )
            if spike_detected:
                logger.warning(
                    f"Epoch {epoch}: Kayıp ani artışı tespit edildi! "
                    f"Loss={train_metrics['loss']:.4f}"
                )
                # Müdahale: öğrenme oranını azalt (opsiyonel)
                self._safe_call(
                    "intervene",
                    self.loss_spike_detector,
                    optimizer=self.optimizer,
                    model=self.model,
                )

            # ==============================================================
            # 8. Iraksama Tespiti
            # ==============================================================
            diverged = self._safe_call(
                "detect",
                self.divergence_detector,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch=epoch,
            )
            if diverged:
                logger.warning(
                    f"Epoch {epoch}: Model ıraksaması tespit edildi! "
                    f"Eğitim durdurulabilir."
                )

            # ==============================================================
            # 9. Gradyan Sağlık Monitörü
            # ==============================================================
            if cfg.log_gradient_health:
                self._safe_call(
                    "log",
                    self.gradient_health_monitor,
                    model=self.model,
                    epoch=epoch,
                    grad_norm=train_metrics.get("gradient_norm", 0.0),
                )

            # ==============================================================
            # 10. Token Dağılım Monitörü
            # ==============================================================
            if cfg.log_token_dist:
                self._safe_call(
                    "log",
                    self.token_distribution_monitor,
                    token_dist=train_metrics.get("token_dist", {}),
                    epoch=epoch,
                )

            # ==============================================================
            # 11. Çıkarım Kalite Testi (her probe_interval epoch'ta)
            # ==============================================================
            probe_results = None
            if (
                self.inference_quality_probe is not None
                and (epoch + 1) % cfg.probe_interval == 0
            ):
                probe_results = self._safe_call(
                    "probe",
                    self.inference_quality_probe,
                    model=self.model,
                    epoch=epoch,
                )
                if probe_results:
                    logger.info(f"Çıkarım kalite testi (epoch {epoch}): {probe_results}")

            # ==============================================================
            # 12. TensorBoard Loglama
            # ==============================================================
            self._log_to_tensorboard(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                current_lr=current_lr,
                probe_results=probe_results,
            )

            # ==============================================================
            # 13. Checkpoint Kaydetme
            # ==============================================================
            val_metric_value = self._get_metric_value(
                val_metrics, cfg.early_stopping_metric
            )
            is_best = self._is_new_best(val_metric_value)

            if is_best:
                self._best_val_metric = val_metric_value
                self._best_epoch = epoch
                logger.info(
                    f"Yeni en iyi model! "
                    f"{cfg.early_stopping_metric}={val_metric_value:.6f} "
                    f"(epoch {epoch})"
                )

            self._save_checkpoints(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
            )

            # ==============================================================
            # 14. Erken Durdurma Kontrolü
            # ==============================================================
            should_stop = self._early_stopper.step(val_metric_value, epoch)
            if should_stop:
                early_stopped = True
                logger.info(
                    f"Erken durdurma: epoch {epoch}. "
                    f"En iyi: epoch {self._best_epoch}, "
                    f"değer={self._best_val_metric:.6f}"
                )
                break

            # ==============================================================
            # 15. Epoch Özeti Loglama
            # ==============================================================
            epoch_elapsed = time.time() - epoch_t0
            epoch_record = {
                "epoch": epoch,
                "train": dict(train_metrics),
                "val": dict(val_metrics),
                "lr": current_lr,
                "is_best": is_best,
                "elapsed_sec": epoch_elapsed,
            }
            self._history.append(epoch_record)

            self._safe_call(
                "log_epoch_summary",
                self.training_logger,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                elapsed=epoch_elapsed,
                is_best=is_best,
            )

            logger.info(
                f"Epoch {epoch} tamamlandı | "
                f"Süre: {epoch_elapsed:.1f}s | "
                f"En iyi epoch: {self._best_epoch}"
            )

        # ------------------------------------------------------------------
        # Eğitim Tamamlandı
        # ------------------------------------------------------------------
        total_elapsed = time.time() - epoch_start_time
        completed_epochs = len(self._history)

        final_results = {
            "history": self._history,
            "best_epoch": self._best_epoch,
            "best_metric": self._best_val_metric,
            "total_epochs": completed_epochs,
            "early_stopped": early_stopped,
            "total_time_sec": total_elapsed,
        }

        logger.info(
            f"{'='*60}\n"
            f"Eğitim tamamlandı!\n"
            f"  Toplam epoch   : {completed_epochs}\n"
            f"  En iyi epoch   : {self._best_epoch}\n"
            f"  En iyi değer   : {self._best_val_metric:.6f}\n"
            f"  Toplam süre    : {total_elapsed:.1f}s\n"
            f"  Erken durdurma : {'Evet' if early_stopped else 'Hayır'}\n"
            f"{'='*60}"
        )

        # TensorBoard'u kapat
        self._safe_call("close", self.tensorboard_manager)

        return final_results

    # ------------------------------------------------------------------
    # Yardımcı Metotlar
    # ------------------------------------------------------------------

    def _is_new_best(self, val_metric: float) -> bool:
        """En iyi metrik değerinin güncellenip güncellenmediğini kontrol eder."""
        if self.config.early_stopping_mode == "min":
            return val_metric < self._best_val_metric
        else:
            return val_metric > self._best_val_metric

    def _get_current_lr(self) -> float:
        """Optimizer'dan mevcut öğrenme oranını okur."""
        try:
            return self.optimizer.param_groups[0]["lr"]
        except (IndexError, KeyError):
            return 0.0

    def _log_to_tensorboard(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
        current_lr: float,
        probe_results: Optional[Dict] = None,
    ) -> None:
        """
        Tüm epoch metriklerini TensorBoard'a yazar.

        Args:
            epoch         : Mevcut epoch
            train_metrics : Eğitim metrikleri
            val_metrics   : Doğrulama metrikleri
            current_lr    : Mevcut öğrenme oranı
            probe_results : Çıkarım kalite testi sonuçları (opsiyonel)
        """
        if self.tensorboard_manager is None:
            return

        tb = self.tensorboard_manager

        # Temel kayıp ve doğruluk metrikleri
        metric_map = {
            "Loss/train": train_metrics["loss"],
            "Loss/val": val_metrics["loss"],
            "Accuracy/train": train_metrics["accuracy"],
            "Accuracy/val": val_metrics["accuracy"],
            "Perplexity/train": train_metrics["perplexity"],
            "Perplexity/val": val_metrics["perplexity"],
            "Entropy/train": train_metrics["entropy"],
            "Entropy/val": val_metrics["entropy"],
            "GradientNorm/train": train_metrics["gradient_norm"],
            "TokensPerSec/train": train_metrics["tokens_per_sec"],
            "LearningRate": current_lr,
        }

        # Kayıp bileşenleri (train)
        for comp_name, comp_val in train_metrics.get("loss_breakdown", {}).items():
            metric_map[f"LossBreakdown/train_{comp_name}"] = comp_val

        # Kayıp bileşenleri (val)
        for comp_name, comp_val in val_metrics.get("loss_breakdown", {}).items():
            metric_map[f"LossBreakdown/val_{comp_name}"] = comp_val

        # Token dağılımı
        for dist_key, dist_val in train_metrics.get("token_dist", {}).items():
            metric_map[f"TokenDist/{dist_key}"] = dist_val

        # Çıkarım kalite sonuçları
        if probe_results and isinstance(probe_results, dict):
            for pk, pv in probe_results.items():
                if isinstance(pv, (int, float)):
                    metric_map[f"InferenceQuality/{pk}"] = pv

        # TensorBoard'a yaz
        for tag, value in metric_map.items():
            try:
                self._safe_call("add_scalar", tb, tag, value, epoch)
            except Exception:
                pass

    def _save_checkpoints(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
        is_best: bool,
    ) -> None:
        """
        Checkpoint kaydetme stratejilerini uygular.

        Stratejiler:
            - En iyi model    : save_best=True ise yeni en iyi her zaman kaydedilir
            - Son model       : save_last=True ise her epoch kaydedilir
            - Periyodik       : Her save_every_n_epochs epoch'ta kaydedilir

        Args:
            epoch      : Mevcut epoch
            train_metrics: Eğitim metrikleri
            val_metrics  : Doğrulama metrikleri
            is_best    : Bu epoch en iyi doğrulama metriği mi?
        """
        if self.checkpoint_manager is None:
            return

        cfg = self.config
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_metrics": dict(train_metrics),
            "val_metrics": dict(val_metrics),
            "loop_state": self.training_loop.state_dict(),
        }

        # EMA ağırlıkları da kaydet
        if cfg.use_ema and self.ema is not None:
            ema_state = self._safe_call("state_dict", self.ema)
            if ema_state is not None:
                checkpoint_data["ema_state_dict"] = ema_state

        # Scheduler durumu
        if self.scheduler is not None:
            sched_state = self._safe_call("state_dict", self.scheduler)
            if sched_state:
                checkpoint_data["scheduler_state_dict"] = sched_state

        # En iyi model checkpoint'i
        if cfg.save_best and is_best:
            self._safe_call("save_best", self.checkpoint_manager, checkpoint_data)

        # Son model checkpoint'i
        if cfg.save_last:
            self._safe_call("save_last", self.checkpoint_manager, checkpoint_data)

        # Periyodik checkpoint
        if cfg.save_every_n_epochs > 0 and (epoch + 1) % cfg.save_every_n_epochs == 0:
            self._safe_call(
                "save_periodic",
                self.checkpoint_manager,
                checkpoint_data,
                epoch=epoch,
            )

    @staticmethod
    def _empty_metrics() -> EpochMetrics:
        """Boş/sıfır değerli EpochMetrics döndürür (hata durumu için)."""
        return EpochMetrics(
            loss=float("inf"),
            accuracy=0.0,
            perplexity=float("inf"),
            entropy=0.0,
            gradient_norm=0.0,
            tokens_per_sec=0.0,
            loss_breakdown={},
            token_dist={},
        )

    # ------------------------------------------------------------------
    # Factory Method
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        loaders: Tuple[Any, Any],
        optimizer: torch.optim.Optimizer,
        config_dict: Dict,
    ) -> "TrainingManager":
        """
        Konfigürasyon sözlüğünden TrainingManager oluşturur.

        Minimum bileşenlerle (loss_manager ve config) çalışır.
        Gelişmiş bileşenler daha sonra atanabilir.

        Args:
            model      : PyTorch dil modeli
            loaders    : (train_loader, val_loader) çifti
            optimizer  : PyTorch optimizer
            config_dict: Konfigürasyon parametrelerini içeren sözlük

        Returns:
            Yapılandırılmış TrainingManager nesnesi

        Örnek:
            config_dict = {
                "total_epochs": 50,
                "device": "cuda",
                "vocab_size": 32000,
                "eos_token_id": 2,
                "label_smoothing": 0.1,
                "use_amp": True,
                "grad_accum_steps": 4,
            }
            manager = TrainingManager.from_config(model, loaders, optimizer, config_dict)
            results = manager.train()
        """
        train_loader, val_loader = loaders

        # TrainingManagerConfig oluştur
        manager_kwargs = {
            k: v for k, v in config_dict.items()
            if k in TrainingManagerConfig.__dataclass_fields__
        }
        manager_config = TrainingManagerConfig(**manager_kwargs)

        # LossConfig oluştur
        loss_kwargs = {
            k: v for k, v in config_dict.items()
            if k in LossConfig.__dataclass_fields__
        }
        loss_config = LossConfig(**loss_kwargs)

        # CompositeLossManager oluştur
        loss_manager = CompositeLossManager(
            config=loss_config,
            device=config_dict.get("device", "cpu"),
        )

        # TrainingLoopConfig oluştur
        loop_kwargs = {
            k: v for k, v in config_dict.items()
            if k in TrainingLoopConfig.__dataclass_fields__
        }
        loop_config = TrainingLoopConfig(**loop_kwargs)
        manager_config.loop_config = loop_config

        logger.info(
            f"TrainingManager.from_config() ile oluşturuldu | "
            f"epochs={manager_config.total_epochs}, "
            f"device={manager_config.device}"
        )

        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_manager=loss_manager,
            config=manager_config,
        )

    # ------------------------------------------------------------------
    # Durum Yönetimi
    # ------------------------------------------------------------------

    def get_history(self) -> List[Dict]:
        """
        Tüm epoch kayıtlarını döndürür.

        Returns:
            Her epoch için train/val metriklerini içeren liste
        """
        return self._history

    def get_best_checkpoint_info(self) -> Dict:
        """
        En iyi checkpoint hakkında bilgi döndürür.

        Returns:
            best_epoch ve best_metric değerlerini içeren sözlük
        """
        return {
            "best_epoch": self._best_epoch,
            "best_metric": self._best_val_metric,
            "metric_name": self.config.early_stopping_metric,
        }

    def resume_from_checkpoint(
        self,
        checkpoint: Dict,
        strict: bool = True,
    ) -> int:
        """
        Checkpoint'ten eğitimi devam ettirir.

        Args:
            checkpoint: Checkpoint sözlüğü (torch.load ile yüklenen)
            strict    : Model ağırlık yükleme modu

        Returns:
            Devam edilecek epoch numarası (checkpoint epoch + 1)
        """
        # Model ağırlıkları
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            logger.info("Model ağırlıkları checkpoint'ten yüklendi")

        # Optimizer durumu
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer durumu checkpoint'ten yüklendi")

        # Döngü durumu (global_step, scaler vb.)
        if "loop_state" in checkpoint:
            self.training_loop.load_state_dict(checkpoint["loop_state"])

        # Scheduler durumu
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self._safe_call(
                "load_state_dict",
                self.scheduler,
                checkpoint["scheduler_state_dict"],
            )

        # EMA durumu
        if "ema_state_dict" in checkpoint and self.ema is not None:
            self._safe_call(
                "load_state_dict",
                self.ema,
                checkpoint["ema_state_dict"],
            )

        resume_epoch = checkpoint.get("epoch", -1) + 1
        logger.info(f"Checkpoint yüklendi. Eğitim epoch {resume_epoch}'tan devam edecek.")
        return resume_epoch

    def __repr__(self) -> str:
        cfg = self.config
        components = []
        if self.scheduler:
            components.append(f"scheduler={type(self.scheduler).__name__}")
        if self.ema:
            components.append("ema=ON")
        if self.tensorboard_manager:
            components.append("tensorboard=ON")
        if self.curriculum_manager:
            components.append("curriculum=ON")

        return (
            f"TrainingManager("
            f"epochs={cfg.total_epochs}, "
            f"device={cfg.device}, "
            f"early_stopping={cfg.early_stopping_patience}, "
            f"{', '.join(components)}"
            f")"
        )
