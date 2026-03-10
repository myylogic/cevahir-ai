# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: training_manager.py
Modül: training_management/v2/core
Görev: Training Manager V2 (Facade) - Eğitim yönetim sistemi için ana facade.
       Tüm alt sistemleri koordine eder. Facade Pattern ile eğitim sürecini
       yönetir, training loop, loss computation, gradient management ve batch
       processing işlemlerini koordine eder.

MİMARİ:
- SOLID Prensipleri: Facade Pattern (tüm alt sistemleri koordine eder),
                     Single Responsibility (koordinasyon), Dependency Inversion
- Design Patterns: Facade Pattern (eğitim sürecini koordine eder)
- Endüstri Standartları: PyTorch training orchestration

KULLANIM:
- Model eğitimi başlatmak için
- Eğitim sürecini koordine etmek için
- TrainingManager instance oluşturup eğitimi yönetmek için

BAĞIMLILIKLAR:
- TrainingLoop: Eğitim döngüsü
- LossComputation: Loss hesaplama
- GradientManager: Gradient yönetimi
- BatchProcessor: Batch işleme
- MetricsTracker: Metrik takibi
- TrainingScheduler: Scheduler yönetimi
- CheckpointManager: Checkpoint yönetimi

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import math
from typing import Tuple, Optional, Any, Dict
import torch
from torch.utils.data import DataLoader

from .training_loop import TrainingLoop
from .loss_computation import LossComputation
from .gradient_manager import GradientManager
from .batch_processor import BatchProcessor

# Utils imports
from ..metrics.metrics_tracker import MetricsTracker
from ..utils.training_scheduler import TrainingScheduler
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.training_logger import TrainingLogger

# Monitoring imports
from ..monitoring.tensorboard_manager import TensorBoardManager
from ..monitoring.memory_tracker import MemoryTracker
from ..monitoring.performance_tracker import PerformanceTracker

# Metrics imports
from ..metrics.advanced_metrics import AdvancedMetrics


class TrainingManager:
    """
    Main training manager (Facade pattern).
    
    Responsibilities:
    - Coordinate all training subsystems
    - Initialize components
    - Provide high-level training API
    - Manage training lifecycle (epochs, early stopping, checkpointing)
    
    SOLID: Facade Pattern - coordinates but doesn't implement
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        config: Dict[str, Any],
        start_epoch: int = 1,
        logger: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        tensorboard_manager: Optional[Any] = None,
    ):
        """
        Initialize TrainingManager.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            config: Configuration dictionary
            start_epoch: Starting epoch number
            logger: Optional logger instance (creates default if None)
            scheduler: Optional scheduler instance (creates from config if None)
            checkpoint_manager: Optional checkpoint manager (creates from config if None)
            tensorboard_manager: Optional TensorBoard manager (creates from config if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.start_epoch = int(start_epoch)
        
        # Device setup
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device_str, str):
            self.device = torch.device(device_str)
        else:
            self.device = device_str
        
        self.model = model.to(self.device)
        
        # Logger setup
        if logger is None:
            enable_file_logging = config.get("enable_file_logging", True)
            self.logger = TrainingLogger(enable_file_logging=enable_file_logging)
        else:
            self.logger = logger
        
        # Configuration
        self.epochs = int(config.get("epochs", 1))
        self.vocab_size = int(config.get("vocab_size"))
        self.early_stopping_patience = int(config.get("early_stopping_patience", 3))
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
        
        # Monitoring configuration
        self.track_memory = bool(config.get("track_memory", True))
        self.track_performance = bool(config.get("track_performance", True))
        self.enable_training_analytics = bool(config.get("enable_training_analytics", True))
        
        # Initialize core components
        self.batch_processor = BatchProcessor(logger=self.logger)
        self.loss_computation = LossComputation(criterion, logger=self.logger)
        self.gradient_manager = GradientManager(
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            logger=self.logger
        )
        
        # Initialize monitoring components (for TrainingLoop)
        memory_tracker = MemoryTracker(enabled=self.track_memory, logger=self.logger) if self.track_memory else None
        performance_tracker = PerformanceTracker(enabled=self.track_performance, logger=self.logger) if self.track_performance else None
        
        # Initialize training analytics (for detailed logging)
        training_analytics = None
        if self.enable_training_analytics:
            try:
                from ..utils.training_analytics import TrainingAnalytics
                log_every_n_batches = int(config.get("analytics_log_every_n_batches", 10))
                log_gradients = bool(config.get("analytics_log_gradients", True))
                log_weight_updates = bool(config.get("analytics_log_weight_updates", True))
                log_loss_details = bool(config.get("analytics_log_loss_details", True))
                log_activations = bool(config.get("analytics_log_activations", False))
                
                training_analytics = TrainingAnalytics(
                    model=self.model,
                    logger=self.logger,
                    log_every_n_batches=log_every_n_batches,
                    log_gradients=log_gradients,
                    log_weight_updates=log_weight_updates,
                    log_loss_details=log_loss_details,
                    log_activations=log_activations,
                )
                self.logger.log_info("[TrainingAnalytics] Enabled - Detaylı training loglama aktif")
            except Exception as e:
                self.logger.log_warning(f"[TrainingAnalytics] Failed to initialize: {e}")
                training_analytics = None
        
        self.training_analytics = training_analytics
        
        # ✅ KRİTİK: Scheduler'ı TrainingLoop'dan ÖNCE set et (warmup için batch-based step)
        # Scheduler setup
        if scheduler is None:
            scheduler_type = config.get("scheduler_type", "ReduceLROnPlateau")
            scheduler_kwargs = config.get("scheduler_kwargs", {})
            # Warmup parametreleri config'ten al
            warmup_steps = config.get("warmup_steps", 0)
            warmup_start_factor = config.get("warmup_start_factor", 0.1)
            embedding_warmup_factor = config.get("embedding_warmup_factor", 1.0)  # [OK] Embedding warmup yok (sabit LR)
            self.scheduler = TrainingScheduler(
                optimizer=self.optimizer,
                scheduler_type=scheduler_type,
                logger=self.logger,
                warmup_steps=warmup_steps,
                warmup_start_factor=warmup_start_factor,
                embedding_warmup_factor=embedding_warmup_factor,  # [OK] GRADIENT FIX
                **scheduler_kwargs
            )
        else:
            self.scheduler = scheduler
        
        self.training_loop = TrainingLoop(
            model=self.model,
            optimizer=self.optimizer,
            loss_computation=self.loss_computation,
            gradient_manager=self.gradient_manager,
            batch_processor=self.batch_processor,
            device=self.device,
            config=config,
            logger=self.logger,
            memory_tracker=memory_tracker,
            performance_tracker=performance_tracker,
            training_analytics=training_analytics,
            scheduler=self.scheduler,  # ✅ YENİ: Scheduler warmup için batch-based step
        )
        
        # Initialize utils components
        self.metrics_tracker = MetricsTracker(logger=self.logger)
        
        # Advanced metrics (optional)
        self.calculate_advanced_metrics = bool(config.get("calculate_advanced_metrics", False))
        if self.calculate_advanced_metrics:
            try:
                self.advanced_metrics = AdvancedMetrics(logger=self.logger)
                self.logger.log_info("[AdvancedMetrics] Enabled (opsiyonel)")
            except Exception as e:
                self.logger.log_warning(f"[AdvancedMetrics] Failed to initialize: {e}")
                self.advanced_metrics = None
                self.calculate_advanced_metrics = False
        else:
            self.advanced_metrics = None
        
        # TensorBoard manager setup
        if tensorboard_manager is None:
            tb_log_dir = config.get("tensorboard_log_dir", "./runs")
            tb_enabled = config.get("enable_tensorboard", True)
            self.tensorboard_manager = TensorBoardManager(
                log_dir=tb_log_dir,
                enabled=tb_enabled,
                logger=self.logger
            )
        else:
            self.tensorboard_manager = tensorboard_manager
        
        # Checkpoint manager setup
        if checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_model_dir=self.checkpoint_dir,
                max_checkpoints=int(config.get("max_checkpoints", 5)),
                device=str(self.device),
                logger=self.logger
            )
        else:
            self.checkpoint_manager = checkpoint_manager
        
        # Training state
        self.current_epoch = self.start_epoch
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
        }
        self.global_step = 0  # For TensorBoard step tracking
    
    def _is_warmup_finished(self) -> bool:
        """
        Warmup bitmiş mi kontrol et.
        
        Returns:
            True if warmup is finished, False otherwise
        """
        if self.scheduler is None:
            return True  # Scheduler yok, warmup yok
        
        # TrainingScheduler'un scheduler attribute'una eriş
        if not hasattr(self.scheduler, 'scheduler'):
            return True  # Warmup wrapper yok
        
        warmup_wrapper = self.scheduler.scheduler
        if not hasattr(warmup_wrapper, 'step_count') or not hasattr(warmup_wrapper, 'warmup_steps'):
            return True  # Warmup wrapper attributes yok
        
        return warmup_wrapper.step_count >= warmup_wrapper.warmup_steps
    
    def train(
        self,
        epoch_callback: Optional[Any] = None
    ) -> Tuple[float, float]:
        """
        Execute training loop.
        
        Coordinates:
        - Training/validation epochs (via TrainingLoop)
        - Metrics tracking (via MetricsTracker)
        - Learning rate scheduling (via TrainingScheduler)
        - Checkpointing (via CheckpointManager)
        - TensorBoard logging (via TensorBoardManager)
        - Early stopping
        
        Args:
            epoch_callback: Optional callback function (epoch, train_loss, val_loss) -> None
            
        Returns:
            Tuple of (final_train_loss, final_val_loss)
        """
        # Start logging
        self.logger.log_info("=" * 60)
        self.logger.log_info("TrainingManager.train() BAŞLADI")
        self.logger.log_info("=" * 60)
        
        try:
            self.vocab_size = int(self.config["vocab_size"])
            self.logger.log_info(f"[OK] TM start — vocab_size={self.vocab_size}")
            
            # Epoch count validation
            if self.epochs == 0:
                self.logger.log_warning("⚠️ Epoch sayısı 0! Eğitim başlamayacak.")
                return float("inf"), float("inf")
        except Exception as e:
            self.logger.log_error(f"Vocab size alınamadı! Eğitim başlatılamıyor: {e}", exc_info=True)
            raise
        
        # Training state
        train_loss = val_loss = None
        val_accuracy = None
        early_stopping_counter = 0
        
        # Main training loop
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.current_epoch = epoch
            
            self.logger.log_info(f"Epoch {epoch}/{self.start_epoch + self.epochs - 1} başladı.")
            
            # Epoch info logging — ana model LR'ı logla (param_groups[0] embedding olabilir, en düşük LR)
            all_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups]
            current_lr = max(all_lrs) if all_lrs else self.optimizer.param_groups[0].get("lr", 0.0)
            total_batches = max(1, len(self.train_loader))
            val_batches = max(1, len(self.val_loader))
            
            self.logger.log_info("=" * 80)
            self.logger.log_info(f"EPOCH {epoch}/{self.start_epoch + self.epochs - 1} BAŞLIYOR")
            self.logger.log_info("=" * 80)
            self.logger.log_info(f"  Training Batches: {total_batches} | Validation Batches: {val_batches}")
            self.logger.log_info(f"  Learning Rate (main): {current_lr:.8f}")
            # Epoch 0'da warmup bilgisi: LR düşük başlayıp peak'e çıkacak (kontrol için)
            if epoch == 0 and self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
                try:
                    sched_lr = self.scheduler.get_last_lr()
                    lr_val = sched_lr[0] if isinstance(sched_lr, (list, tuple)) else float(sched_lr)
                    warmup_steps = getattr(
                        getattr(self.scheduler, "scheduler", None), "warmup_steps", None
                    ) or self.config.get("warmup_steps", 0)
                    base_lr = self.config.get("learning_rate", 0.0002)
                    start_factor = self.config.get("warmup_start_factor", 0.1)
                    self.logger.log_info(
                        f"  [Warmup] Başlangıç LR={lr_val:.2e} (peak={base_lr:.2e}), "
                        f"warmup_steps={warmup_steps}, start_factor={start_factor}"
                    )
                except Exception:
                    pass
            self.logger.log_info("-" * 80)
            
            try:
                # Training epoch (returns metrics including performance/memory stats)
                train_result = self.training_loop.train_epoch(self.train_loader, epoch=epoch)
                train_loss, train_accuracy, avg_gradient_norm = train_result[:3]
                
                # Validation epoch
                val_result = self.training_loop.validate_epoch(self.val_loader, epoch=epoch)
                val_loss, val_accuracy = val_result[:2]
                
                # Validate losses
                train_loss = float(train_loss)
                val_loss = float(val_loss)
                if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                    raise ValueError(f"[Epoch {epoch}] Geçersiz train/val loss tespit edildi.")
            
            except Exception as e:
                self.logger.log_error(f"[Epoch {epoch}] Eğitim/validasyon adımında hata: {e}", exc_info=True)
                train_loss = float("inf")
                val_loss = float("inf")
                val_accuracy = 0.0
                break
            
            # Analytics: Log epoch summary
            if self.training_analytics:
                self.training_analytics.log_epoch_summary(epoch, train_loss, train_accuracy)
            
            # Epoch summary
            loss_diff = train_loss - val_loss
            train_acc_pct = train_accuracy * 100 if train_accuracy else 0.0
            val_acc_pct = val_accuracy * 100 if val_accuracy else 0.0
            
            self.logger.log_info("=" * 80)
            self.logger.log_info(f"EPOCH {epoch} ÖZET")
            self.logger.log_info("=" * 80)
            self.logger.log_info(f"  Training:")
            self.logger.log_info(f"    Loss:     {train_loss:.6f}")
            self.logger.log_info(f"    Accuracy: {train_accuracy:.4f} ({train_acc_pct:.2f}%)")
            self.logger.log_info(f"    GradNorm: {avg_gradient_norm:.4f}" if math.isfinite(avg_gradient_norm) else "    GradNorm: (NaN — bir batch'te gradient sayısal patlama)")
            self.logger.log_info(f"  Validation:")
            self.logger.log_info(f"    Loss:     {val_loss:.6f}")
            self.logger.log_info(f"    Accuracy: {val_accuracy:.4f} ({val_acc_pct:.2f}%)")
            self.logger.log_info(f"  Fark (Train - Val): {loss_diff:.6f}")
            
            if loss_diff < 0:
                self.logger.log_warning("  ⚠️  UYARI: Val Loss < Train Loss (validation set sorunlari olabilir)")
            elif loss_diff > 0.2:
                self.logger.log_warning("  ⚠️  UYARI: Fark cok buyuk (overfitting riski)")
            
            self.logger.log_info("=" * 80)
            
            # Update metrics tracker
            self.metrics_tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                accuracy=val_accuracy
            )
            
            # Update training history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["accuracy"].append(val_accuracy)
            
            # TensorBoard logging (epoch level)
            if self.tensorboard_manager.enabled:
                try:
                    self.tensorboard_manager.log_scalar("Loss/Train", train_loss, epoch)
                    self.tensorboard_manager.log_scalar("Loss/Validation", val_loss, epoch)
                    
                    # Perplexity
                    if math.isfinite(train_loss):
                        train_ppl = math.exp(min(20.0, train_loss))
                        self.tensorboard_manager.log_scalar("Perplexity/Train", train_ppl, epoch)
                    if math.isfinite(val_loss):
                        val_ppl = math.exp(min(20.0, val_loss))
                        self.tensorboard_manager.log_scalar("Perplexity/Validation", val_ppl, epoch)
                    
                    # Accuracy
                    if val_accuracy is not None:
                        self.tensorboard_manager.log_scalar("Accuracy/Validation", val_accuracy, epoch)
                    
                    # Gradient norm
                    if math.isfinite(avg_gradient_norm):
                        self.tensorboard_manager.log_scalar("GradNorm/Avg", avg_gradient_norm, epoch)
                except Exception as e:
                    if self.logger:
                        self.logger.log_warning(f"[TensorBoard] Failed to log epoch metrics: {e}")
            
            # Learning rate scheduling
            # ✅ YENİ: Warmup bitmişse scheduler.step() çağır (epoch-based, metric ile)
            lr_value: Optional[float] = None
            if self.scheduler is not None:
                try:
                    # Warmup bitmiş mi kontrol et
                    warmup_finished = self._is_warmup_finished()
                    if warmup_finished:
                        # Warmup bitmiş → ReduceLROnPlateau için metric-based step (epoch-based)
                        grad_norm_for_scheduler = avg_gradient_norm if math.isfinite(avg_gradient_norm) else None
                        self.scheduler.step(metric=val_loss, gradient_norm=grad_norm_for_scheduler)
                        all_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups]
                        lr_value = max(all_lrs) if all_lrs else 0.0
                        self.logger.log_info(f"LR güncellendi (main): {float(lr_value):.8f} (AvgGradNorm={avg_gradient_norm:.4f})")
                    else:
                        # Warmup devam ediyor → LR TrainingLoop'da batch-based güncelleniyor
                        all_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups]
                        lr_value = max(all_lrs) if all_lrs else 0.0
                        self.logger.log_info(f"LR (warmup, main): {float(lr_value):.8f} (warmup devam ediyor)")
                    
                    # Log LR to TensorBoard
                    if self.tensorboard_manager.enabled:
                        self.tensorboard_manager.log_scalar("LR", lr_value, epoch)
                except Exception as e:
                    self.logger.log_warning(f"[Epoch {epoch}] LR güncellenemedi: {e}")
            if lr_value is None and self.tensorboard_manager.enabled:
                all_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups]
                lr_value = max(all_lrs) if all_lrs else 0.0
                self.tensorboard_manager.log_scalar("LR", float(lr_value), epoch)
            
            # Memory tracking summary (epoch level)
            if self.track_memory and self.training_loop.memory_tracker:
                try:
                    mem_stats = self.training_loop.memory_tracker.track()
                    if mem_stats and self.tensorboard_manager.enabled:
                        if "gpu_allocated_mb" in mem_stats:
                            self.tensorboard_manager.log_scalar("Memory/GPUAllocatedMB", mem_stats["gpu_allocated_mb"], epoch)
                        if "gpu_reserved_mb" in mem_stats:
                            self.tensorboard_manager.log_scalar("Memory/GPUReservedMB", mem_stats["gpu_reserved_mb"], epoch)
                except Exception as e:
                    if self.logger:
                        self.logger.log_debug(f"[Memory] Tracking error: {e}")
            
            # Performance tracking summary (epoch level)
            if self.track_performance and self.training_loop.performance_tracker:
                try:
                    perf_tracker = self.training_loop.performance_tracker
                    if perf_tracker.batch_times and self.tensorboard_manager.enabled:
                        avg_batch_time = sum(perf_tracker.batch_times) / len(perf_tracker.batch_times)
                        samples_per_sec = self.config.get("batch_size", 1) / avg_batch_time if avg_batch_time > 0 else 0
                        seq_len = self.config.get("seq_len", 1)
                        tokens_per_sec = samples_per_sec * seq_len
                        
                        self.tensorboard_manager.log_scalar("Performance/AvgBatchTime", avg_batch_time, epoch)
                        self.tensorboard_manager.log_scalar("Performance/SamplesPerSec", samples_per_sec, epoch)
                        self.tensorboard_manager.log_scalar("Performance/TokensPerSec", tokens_per_sec, epoch)
                except Exception as e:
                    if self.logger:
                        self.logger.log_debug(f"[Performance] Tracking error: {e}")
            
            # Flush TensorBoard
            if self.tensorboard_manager.enabled:
                self.tensorboard_manager.flush()
            
            # Epoch callback
            if epoch_callback is not None:
                try:
                    epoch_callback(epoch, train_loss, val_loss)
                except Exception as e:
                    self.logger.log_warning(f"[Epoch {epoch}] Epoch callback hatası: {e}")
            
            # Checkpointing and early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                try:
                    # [DEBUG] Checkpoint kaydetmeden önce model instance kontrolü
                    if self.model is not None:
                        model_state_dict = self.model.state_dict()
                        model_keys = list(model_state_dict.keys())
                        model_type = type(self.model).__name__
                        is_simple_model = (
                            len(model_keys) == 3 and 
                            all(k in model_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
                        )
                        self.logger.log_info("=" * 60)
                        self.logger.log_info(f"[CHECKPOINT DEBUG] [Epoch {epoch}] Checkpoint kaydetme öncesi model kontrolü:")
                        self.logger.log_info(f"  Model Type: {model_type}")
                        self.logger.log_info(f"  State Dict Keys: {len(model_keys)}")
                        self.logger.log_info(f"  İlk 10 Key: {model_keys[:10]}")
                        self.logger.log_info(f"  SimpleModel mi? {is_simple_model}")
                        if is_simple_model:
                            self.logger.log_error("  ⚠️ KRİTİK UYARI: SimpleModel instance'ı kaydediliyor!")
                        else:
                            self.logger.log_info("  ✅ CevahirNeuralNetwork instance'ı kaydediliyor")
                        self.logger.log_info("=" * 60)
                    
                    # Save best model
                    self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        training_history=self.training_history,
                        metric=val_loss,
                        is_best=True
                    )
                    self.logger.log_info(f"[Epoch {epoch}] [OK] Yeni en iyi val loss: {val_loss:.6f}. Model kaydedildi.")
                    early_stopping_counter = 0
                except Exception as e:
                    self.logger.log_error(f"[Epoch {epoch}] Model kaydedilemedi: {e}")
            else:
                early_stopping_counter += 1
                self.logger.log_info(f"[Epoch {epoch}] Early stopping counter: {early_stopping_counter}/{self.early_stopping_patience}")
                
                if early_stopping_counter >= self.early_stopping_patience:
                    stop_msg = f"[Epoch {epoch}] Erken durdurma: {self.early_stopping_patience} epoch boyunca gelişme yok."
                    self.logger.log_info(stop_msg)
                    break
        
        # Close TensorBoard
        if self.tensorboard_manager.enabled:
            self.tensorboard_manager.close()
        
        # Final summary
        self.logger.log_info("=" * 60)
        self.logger.log_info("Eğitim tamamlandı")
        self.logger.log_info("=" * 60)
        
        if self.training_history["val_loss"]:
            final_train_loss = self.training_history["train_loss"][-1] if self.training_history["train_loss"] else float("inf")
            final_val_loss = self.training_history["val_loss"][-1]
            self.logger.log_info(f"Final Train Loss: {final_train_loss:.6f}")
            self.logger.log_info(f"Final Val Loss: {final_val_loss:.6f}")
            self.logger.log_info(f"Best Val Loss: {self.best_val_loss:.6f}")
        
        return (
            float(train_loss if train_loss is not None else "inf"),
            float(val_loss if val_loss is not None else "inf")
        )
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
        """
        return self.metrics_tracker.get_history()
