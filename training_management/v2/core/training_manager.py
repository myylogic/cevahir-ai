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
from ..utils.training_scheduler import (
    TrainingScheduler, apply_continuation_learning_rate, find_plateau_scheduler, main_learning_rate,
    mark_embedding_param_groups,
)
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
        from training_management.contracts import validate_training_backend
        validate_training_backend(config)
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
            mark_embedding_param_groups(self.optimizer, self.model)  # embedding grupları açık "name" etiketiyle tanınır
            warmup_steps = config.get("warmup_steps", 0)
            warmup_start_factor = config.get("warmup_start_factor", 0.1)
            embedding_warmup_factor = config.get("embedding_warmup_factor")  # None: embedding grupları da normal warmup alır
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
        # What best_val_loss (and the plateau scheduler's best) measures: "nll", or "objective" while
        # a run resumed from an older checkpoint has not yet beaten the objective-scale best recorded there.
        self.best_val_metric = "nll"
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
        }
        self.global_step = 0  # For TensorBoard step tracking
        self.early_stopping_counter = 0

    def _checkpoint_extra_state(self):
        import random
        import numpy as np
        numpy_state = np.random.get_state()
        return {
            "format_version": 1,
            "tokenizer_identity": self.config.get("tokenizer_identity"),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "loop": self.training_loop.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_metric": self.best_val_metric,
            "early_stopping_counter": self.early_stopping_counter,
            "rng": {
                "python": random.getstate(),
                "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }

    def resume_from_checkpoint(self, path):
        """Resume at the next epoch; legacy weights remain usable with an explicit warning.

        ``train()`` then runs epochs ``saved_epoch + 1`` .. ``config["epochs"]`` (a total target).
        ``config["continuation_learning_rate"]``, if set, is applied last: the optimizer and
        scheduler state restored here carry the checkpointed run's learning rate.

        A checkpoint without ``best_val_metric`` was written while the reported validation loss
        was the training objective, so its ``best_val_loss`` and plateau best are objective values.
        Model selection then compares the validation objective until an epoch beats that best,
        and validation NLL afterwards; a new NLL is never compared with an old objective value.
        """
        import random
        import numpy as np
        state = torch.load(path, map_location=self.device, weights_only=True)
        extra = state.get("extra_state") or {}
        best_val_metric = extra.get("best_val_metric", "objective")
        if best_val_metric not in ("nll", "objective"):
            raise ValueError(f"Unknown best_val_metric in checkpoint: {best_val_metric!r}")
        from model_management.checkpoint_contract import unpack_checkpoint, validate_model_identity, validate_state_shapes
        model_state, _, _, metadata = unpack_checkpoint(state)
        from model_management.checkpoint_contract import validate_tokenizer_identity
        validate_tokenizer_identity(metadata.get("tokenizer_identity"), self.config.get("tokenizer_identity"))
        validate_model_identity(self.model, metadata.get("config"))
        validate_state_shapes(self.model, model_state)
        self.model.load_state_dict(model_state, strict=True)
        if hasattr(self.model, "clear_kv_cache"):
            self.model.clear_kv_cache()
        if "optimizer_state_dict" in state:
            # load_state_dict replaces every param group with the saved one; keep the group names
            # (e.g. the "embedding" warmup tag) that a checkpoint written without them would drop.
            names = [group.get("name") for group in self.optimizer.param_groups]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            for group, name in zip(self.optimizer.param_groups, names):
                if name is not None and "name" not in group:
                    group["name"] = name
        self.start_epoch = int(state.get("epoch", 0)) + 1
        self.training_history = state.get("training_history", self.training_history)
        if not extra:
            self.logger.log_warning("Legacy checkpoint has no RNG/scheduler/scaler state; exact resume is unavailable")
            self._apply_continuation_learning_rate()
            return self.start_epoch
        if self.scheduler is not None and extra.get("scheduler") is not None:
            self.scheduler.load_state_dict(extra["scheduler"])
        self.training_loop.load_state_dict(extra.get("loop", {}))
        self.global_step = self.training_loop.optimizer_steps
        self.best_val_loss = float(extra.get("best_val_loss", float("inf")))
        self.best_val_metric = best_val_metric
        if best_val_metric == "objective":
            self.logger.log_warning(
                f"Checkpoint records best_val_loss={self.best_val_loss:.6f} as a validation objective value "
                "(recorded before validation NLL drove model selection). The best checkpoint, early stopping "
                "and the plateau scheduler compare the validation objective until an epoch beats it, "
                "then validation NLL."
            )
        self.early_stopping_counter = int(extra.get("early_stopping_counter", 0))
        rng = extra.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            ns = rng["numpy"]
            np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
            torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([r.cpu() for r in rng["cuda"]])
        self._apply_continuation_learning_rate()
        return self.start_epoch

    def _apply_continuation_learning_rate(self):
        """Override the restored learning rate with ``config["continuation_learning_rate"]``.

        Building the optimizer with a different LR has no effect on a resumed run, because
        ``optimizer.load_state_dict`` restores the checkpointed LR; the override therefore runs
        after the optimizer and scheduler state are loaded. Returns the factor applied, or None.
        """
        value = self.config.get("continuation_learning_rate")
        if value is None:
            return None
        factor = apply_continuation_learning_rate(self.optimizer, value, self.scheduler)
        lrs = ", ".join(f"{float(group['lr']):.2e}" for group in self.optimizer.param_groups)
        self.logger.log_info(
            f"[Resume] continuation_learning_rate={float(value):.2e} applied after the optimizer/scheduler "
            f"state was restored (x{factor:.4g}); group LRs: {lrs}"
        )
        return factor

    def _switch_model_selection_to_nll(self, val_nll: float) -> None:
        """An epoch beat an older checkpoint's objective-scale best: compare validation NLL from now on.

        The plateau scheduler's best is rebased to this epoch's NLL too, so that its next step
        does not compare an NLL with an objective value.
        """
        self.best_val_metric = "nll"
        plateau = find_plateau_scheduler(self.scheduler)
        if plateau is not None:
            plateau.best = float(val_nll)
            plateau.num_bad_epochs = 0
        self.logger.log_info(
            f"[Model selection] The checkpoint's objective-scale best was beaten; the best checkpoint, "
            f"early stopping and the plateau scheduler now compare validation NLL (NLL={val_nll:.6f})"
        )
    
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
        early_stopping_counter = self.early_stopping_counter
        
        # Main training loop. `epochs` is the total target epoch count: a fresh run trains
        # 1..epochs, a run resumed from checkpoint epoch k trains k+1..epochs.
        last_epoch = self.epochs
        if self.start_epoch > last_epoch:
            self.logger.log_warning(
                f"Checkpoint epoch {self.start_epoch - 1} already reaches the target epochs={last_epoch}; "
                "nothing to train. Raise 'epochs' to continue training."
            )
            if self.tensorboard_manager.enabled:
                self.tensorboard_manager.close()
            history_train = self.training_history.get("train_loss") or []
            history_val = self.training_history.get("val_loss") or []
            return (
                float(history_train[-1]) if history_train else float("inf"),
                float(history_val[-1]) if history_val else float("inf"),
            )
        for epoch in range(self.start_epoch, last_epoch + 1):
            self.current_epoch = epoch
            sampler = getattr(self.train_loader, "batch_sampler", None)
            if not hasattr(sampler, "set_epoch"):
                sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            
            self.logger.log_info(f"Epoch {epoch}/{last_epoch} başladı.")
            
            # Epoch info logging — ana model LR'ı: "embedding" etiketi olmayan ilk grup (warmup'ın izlediği
            # LR; warmup dışında tutulan bir embedding grubu bundan yüksek olabilir)
            current_lr = main_learning_rate(self.optimizer)
            total_batches = max(1, len(self.train_loader))
            val_batches = max(1, len(self.val_loader))
            
            self.logger.log_info("=" * 80)
            self.logger.log_info(f"EPOCH {epoch}/{last_epoch} BAŞLIYOR")
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
                raise RuntimeError(f"Training failed at epoch {epoch}") from e
            
            # Analytics: Log epoch summary
            if self.training_analytics:
                self.training_analytics.log_epoch_summary(epoch, train_loss, train_accuracy)
            
            # Reported train/val loss is the plain masked NLL (perplexity = exp(NLL)); the
            # optimized objective (label smoothing, entropy term, MoE auxiliary loss) is logged apart.
            train_stats = getattr(self.training_loop, "last_train_stats", None) or {}
            val_stats = getattr(self.training_loop, "last_val_stats", None) or {}
            train_objective = train_stats.get("objective")
            val_objective = val_stats.get("objective")
            skipped_steps = int(train_stats.get("skipped_steps", 0) or 0)
            # Model selection (best checkpoint, early stopping, plateau scheduler) compares validation NLL.
            # After resuming an older checkpoint whose best is a validation objective value, it compares the
            # validation objective until an epoch beats that best (see resume_from_checkpoint).
            selection_loss = val_loss
            if self.best_val_metric == "objective" and val_objective is not None:
                selection_loss = float(val_objective)

            # Epoch summary
            loss_diff = train_loss - val_loss
            train_acc_pct = train_accuracy * 100 if train_accuracy else 0.0
            val_acc_pct = val_accuracy * 100 if val_accuracy else 0.0
            
            self.logger.log_info("=" * 80)
            self.logger.log_info(f"EPOCH {epoch} ÖZET")
            self.logger.log_info("=" * 80)
            self.logger.log_info(f"  Training:")
            self.logger.log_info(f"    Loss (NLL): {train_loss:.6f} | Perplexity: {math.exp(min(20.0, train_loss)):.2f}")
            if train_objective is not None:
                self.logger.log_info(f"    Objective:  {train_objective:.6f} (optimize edilen amaç)")
            self.logger.log_info(f"    Accuracy: {train_accuracy:.4f} ({train_acc_pct:.2f}%)")
            if math.isfinite(avg_gradient_norm):
                self.logger.log_info(f"    GradNorm: {avg_gradient_norm:.4f}")
            elif skipped_steps:
                self.logger.log_info("    GradNorm: (yok — bu epoch'taki bütün optimizer adımları fp16 taşması nedeniyle atlandı)")
            else:
                self.logger.log_info("    GradNorm: (NaN — bir batch'te gradient sayısal patlama)")
            if skipped_steps:
                self.logger.log_warning(f"    Atlanan fp16 adımı: {skipped_steps} (GradScaler ölçeği küçülttü)")
            self.logger.log_info(f"  Validation:")
            self.logger.log_info(f"    Loss (NLL): {val_loss:.6f} | Perplexity: {math.exp(min(20.0, val_loss)):.2f}")
            if val_objective is not None:
                self.logger.log_info(f"    Objective:  {val_objective:.6f}")
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
            
            # Update training history (train_loss / val_loss: plain NLL)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["accuracy"].append(val_accuracy)
            if train_objective is not None:
                self.training_history.setdefault("train_objective", []).append(train_objective)
            if val_objective is not None:
                self.training_history.setdefault("val_objective", []).append(val_objective)
            
            # TensorBoard logging (epoch level)
            if self.tensorboard_manager.enabled:
                try:
                    self.tensorboard_manager.log_scalar("Loss/Train", train_loss, epoch)
                    self.tensorboard_manager.log_scalar("Loss/Validation", val_loss, epoch)
                    if train_objective is not None:
                        self.tensorboard_manager.log_scalar("Objective/Train", train_objective, epoch)
                    if val_objective is not None:
                        self.tensorboard_manager.log_scalar("Objective/Validation", val_objective, epoch)
                    if skipped_steps:
                        self.tensorboard_manager.log_scalar("Train/SkippedSteps", skipped_steps, epoch)
                    
                    # Perplexity of the plain NLL (not of the regularized objective)
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
                        if hasattr(self.scheduler, "step_epoch"):
                            self.scheduler.step_epoch(metric=selection_loss)
                        else:
                            self.scheduler.step(metric=selection_loss, gradient_norm=grad_norm_for_scheduler)
                        lr_value = main_learning_rate(self.optimizer)
                        self.logger.log_info(f"LR güncellendi (main): {float(lr_value):.8f} (AvgGradNorm={avg_gradient_norm:.4f})")
                    else:
                        # Warmup devam ediyor → LR TrainingLoop'da batch-based güncelleniyor
                        lr_value = main_learning_rate(self.optimizer)
                        self.logger.log_info(f"LR (warmup, main): {float(lr_value):.8f} (warmup devam ediyor)")
                    
                    # Log LR to TensorBoard
                    if self.tensorboard_manager.enabled:
                        self.tensorboard_manager.log_scalar("LR", lr_value, epoch)
                except Exception as e:
                    raise RuntimeError(f"Scheduler failed at epoch {epoch}") from e
            if lr_value is None and self.tensorboard_manager.enabled:
                lr_value = main_learning_rate(self.optimizer)
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
            
            # Performance tracking summary (epoch level): this epoch's training batches only,
            # measured with the samples and non-pad target tokens they actually contained
            # (not the configured batch_size x seq_len).
            if self.track_performance and self.training_loop.performance_tracker:
                try:
                    perf = train_stats.get("performance")
                    if perf and self.tensorboard_manager.enabled:
                        self.tensorboard_manager.log_scalar("Performance/AvgBatchTime", perf["avg_batch_time_sec"], epoch)
                        self.tensorboard_manager.log_scalar("Performance/SamplesPerSec", perf["samples_per_sec"], epoch)
                        self.tensorboard_manager.log_scalar("Performance/TokensPerSec", perf["tokens_per_sec"], epoch)
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
            
            # Save the completed epoch, including the state needed by the next epoch.
            is_best = selection_loss < self.best_val_loss
            if self.best_val_metric == "objective":
                self.logger.log_info(
                    f"  Model selection: validation objective {selection_loss:.6f} vs the checkpoint's "
                    f"objective-scale best {self.best_val_loss:.6f}"
                )
            if is_best:
                if self.best_val_metric != "nll":
                    self._switch_model_selection_to_nll(val_loss)
                self.best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            self.early_stopping_counter = early_stopping_counter
            self.global_step = self.training_loop.optimizer_steps
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                training_history=self.training_history,
                metric=val_loss,
                is_best=is_best,
                extra_state=self._checkpoint_extra_state(),
            )
            if early_stopping_counter >= self.early_stopping_patience:
                self.logger.log_info(f"Early stopping after epoch {epoch}")
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
            self.logger.log_info(f"Best Val Loss ({self.best_val_metric}): {self.best_val_loss:.6f}")
        
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
