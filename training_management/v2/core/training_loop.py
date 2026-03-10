# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: training_loop.py
Modül: training_management/v2/core
Görev: Training Loop - Eğitim ve doğrulama epoch döngüleri. Training ve validation
       loop execution işlemlerini yönetir. Loss computation, gradient management
       ve batch processing ile entegre çalışır.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (training/validation loop execution),
                     Dependency Inversion (abstraksiyonlara bağımlı)
- Design Patterns: Loop Pattern (eğitim döngüsü)
- Endüstri Standartları: PyTorch training loop best practices

KULLANIM:
- Eğitim epoch döngüsü için
- Validation epoch döngüsü için
- Training ve validation işlemlerini yönetmek için

BAĞIMLILIKLAR:
- LossComputation: Loss hesaplama
- GradientManager: Gradient yönetimi
- BatchProcessor: Batch işleme
- tqdm: Progress bar (opsiyonel)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import contextlib
import math
from typing import Tuple, Optional, Any, Dict
import torch
from torch.utils.data import DataLoader

from .loss_computation import LossComputation
from .gradient_manager import GradientManager
from .batch_processor import BatchProcessor

# [NEW] Advanced token metrics
from ..metrics.advanced_token_metrics import AdvancedTokenMetrics
from ..utils.enhanced_training_logger import EnhancedTrainingLogger

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


class TrainingLoop:
    """
    Training and validation loop executor.
    
    Responsibilities:
    - Execute training epoch
    - Execute validation epoch
    - Process batches
    - Handle gradient accumulation
    - Coordinate with loss computation, gradient management, etc.
    
    SOLID: Single Responsibility Principle (loop execution only)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_computation: LossComputation,
        gradient_manager: GradientManager,
        batch_processor: BatchProcessor,
        device: torch.device,
        config: Dict[str, Any],
        logger: Optional[Any] = None,
        memory_tracker: Optional[Any] = None,
        performance_tracker: Optional[Any] = None,
        training_analytics: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ):
        """
        Initialize TrainingLoop.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_computation: LossComputation instance
            gradient_manager: GradientManager instance
            batch_processor: BatchProcessor instance
            device: Device to run on
            config: Configuration dictionary
            logger: Optional logger instance
            memory_tracker: Optional MemoryTracker instance
            performance_tracker: Optional PerformanceTracker instance
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_computation = loss_computation
        self.gradient_manager = gradient_manager
        self.batch_processor = batch_processor
        self.device = device
        self.config = config
        self.logger = logger
        
        # Configuration
        self.grad_accum_steps = int(config.get("grad_accum_steps", 1))
        self.pad_token_id = config.get("pad_token_id")
        self.use_amp = bool(config.get("use_amp", False))
        self.use_progress_bar = bool(config.get("use_progress_bar", True))
        self.batch_size = int(config.get("batch_size", 1))
        self.seq_len = int(config.get("seq_len", 1))
        
        # AMP setup
        self._amp_device_type = "cuda" if (device.type == "cuda" and torch.cuda.is_available()) else "cpu"
        if self.use_amp and self._amp_device_type == "cuda":
            # ✅ FIX: torch.cuda.amp.GradScaler() deprecated, torch.amp.GradScaler('cuda') kullan
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Safety: NaN/Inf detector
        try:
            from ..safety.nan_inf_detector import NaNInfDetector
            self.nan_inf_detector = NaNInfDetector(logger=logger)
        except ImportError:
            self.nan_inf_detector = None
        
        # Safety: Gradient explosion detector
        try:
            from ..safety.gradient_explosion_detector import GradientExplosionDetector
            max_grad_norm = float(config.get("max_grad_norm", 1.0))
            explosion_threshold = float(config.get("gradient_explosion_threshold", 10.0))
            self.gradient_explosion_detector = GradientExplosionDetector(
                threshold=explosion_threshold,
                logger=logger
            )
            self.max_grad_norm = max_grad_norm  # Store for explosion detection
        except ImportError:
            self.gradient_explosion_detector = None
            self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        
        # [NEW] Advanced token metrics for EOS/BOS/PAD tracking
        vocab_size = config.get("vocab_size", 8000)
        special_tokens_dict = {
            config.get("pad_token_id", 0): "PAD",
            config.get("bos_token_id", 1): "BOS",
            config.get("eos_token_id", 2): "EOS",
            config.get("unk_token_id", 3): "UNK",
        }
        self.advanced_metrics = AdvancedTokenMetrics(vocab_size, special_tokens_dict)
        self.enhanced_logger = EnhancedTrainingLogger("TrainingLoopLogger")
        self.prev_train_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = config.get("early_stopping_patience", 20)
        
        # Store last batch metrics for epoch summary
        self.last_batch_metrics = {
            "overall_acc": 0.0,
            "special_accs": {},
            "top5_acc": 0.0,
            "entropy": 0.0
        }
        
        # Safety: Validation manager
        try:
            from ..safety.validation_manager import ValidationManager
            self.validation_manager = ValidationManager(logger=logger)
        except ImportError:
            self.validation_manager = None
        
        # Monitoring: Memory and Performance trackers
        self.memory_tracker = memory_tracker
        self.performance_tracker = performance_tracker
        
        # Analytics: Training analytics for detailed logging
        self.training_analytics = training_analytics
        
        # Scheduler: For batch-based warmup
        self.scheduler = scheduler
    
    def _autocast_ctx(self):
        """AMP autocast context (CUDA varsa), yoksa no-op."""
        if self.use_amp and self._amp_device_type == "cuda":
            return torch.amp.autocast("cuda", enabled=True)
        return contextlib.nullcontext()
    
    def _step_scheduler_for_warmup(self):
        """
        Warmup devam ediyorsa scheduler.step() çağır (batch-based).
        
        Endüstri Standardı:
        - Warmup: Batch-based (her optimizer.step() sonrası)
        - ReduceLROnPlateau: Epoch-based (epoch sonunda, metric ile)
        """
        if self.scheduler is None:
            return
        
        # TrainingScheduler'un scheduler attribute'una eriş
        if not hasattr(self.scheduler, 'scheduler'):
            return
        
        warmup_wrapper = self.scheduler.scheduler
        if not hasattr(warmup_wrapper, 'step_count') or not hasattr(warmup_wrapper, 'warmup_steps'):
            return
        
        # Warmup devam ediyor mu?
        if warmup_wrapper.step_count < warmup_wrapper.warmup_steps:
            # Warmup devam ediyor → batch-based step (metric gerekmez)
            try:
                self.scheduler.step()
            except Exception as e:
                if self.logger:
                    self.logger.log_debug(f"[Warmup] Scheduler step error (ignored): {e}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Execute one training epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (avg_loss, avg_accuracy, avg_gradient_norm)
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        running_loss = 0.0
        running_acc = 0.0
        processed_batches = 0
        total_gradient_norm = 0.0
        gradient_norm_batches = 0
        micro = 0  # Gradient accumulation counter
        
        total_batches = max(1, len(train_loader))
        
        # Progress bar
        if self.use_progress_bar and TQDM_AVAILABLE:
            epoch_pbar = tqdm(
                train_loader,
                desc=f"Training",
                leave=False,
                ncols=100,
                unit="batch"
            )
        else:
            epoch_pbar = train_loader
        
        for batch_idx, batch in enumerate(epoch_pbar, start=1):
            try:
                # Analytics: Log batch start
                if self.training_analytics:
                    self.training_analytics.log_batch_start(batch_idx, epoch or 0, total_batches)
                    self.training_analytics.snapshot_weights(batch_idx)
                
                # Performance tracking: Start batch timing
                if self.performance_tracker:
                    self.performance_tracker.start_batch()
                
                # Parse batch
                inputs, targets = self.batch_processor.parse_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass with AMP
                with self._autocast_ctx():
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                
                # Validation: Check logits shape (if enabled)
                if self.validation_manager:
                    try:
                        vocab_size = self.config.get("vocab_size")
                        if vocab_size:
                            expected_shape = (inputs.shape[0], inputs.shape[1] if inputs.dim() > 1 else 1)
                            self.validation_manager.validate_logits_shape(
                                logits, expected_shape, vocab_size
                            )
                    except Exception as e:
                        if self.logger:
                            self.logger.log_error(f"Validation error in batch {batch_idx}: {e}")
                        continue  # Skip invalid batch
                
                # Compute loss (CRITICAL: Uses criterion with EOS weight, label smoothing)
                loss, acc, ppl = self.loss_computation.compute_loss(
                    logits=logits,
                    targets=targets,
                    pad_id=self.pad_token_id
                )
                
                # [CRITICAL FIX] Advanced Token Metrics - EOS/BOS/PAD tracking
                if batch_idx % 10 == 0:  # Her 10 batch'te log
                    try:
                        overall_acc, special_accs, top5_acc, entropy = self.advanced_metrics.compute_per_token_accuracy(
                            logits=logits,
                            targets=targets
                        )
                        
                        # Store for epoch summary
                        self.last_batch_metrics = {
                            "overall_acc": overall_acc,
                            "special_accs": special_accs,
                            "top5_acc": top5_acc,
                            "entropy": entropy
                        }
                        
                        # Check for critical issues
                        critical_issues = self.advanced_metrics.check_critical_issues(
                            overall_acc, special_accs, entropy
                        )
                        
                        # Log batch with token metrics
                        self.enhanced_logger.log_batch_with_token_metrics(
                            epoch=epoch,
                            batch_idx=batch_idx,
                            loss=float(loss.item()),
                            overall_acc=overall_acc,
                            special_accs=special_accs,
                            top5_acc=top5_acc,
                            entropy=entropy,
                            grad_norm=None  # Gradient norm'u optimizer.step() sonra yapacağız
                        )
                        
                        # Log critical warnings
                        if critical_issues:
                            self.enhanced_logger.log_critical_warnings(
                                warnings=critical_issues,
                                epoch=epoch,
                                batch_idx=batch_idx
                            )
                    except Exception as e:
                        if self.logger:
                            self.logger.log_warning(f"[METRICS] Token metrics computation failed at batch {batch_idx}: {e}")
                
                # Analytics: Log loss details
                if self.training_analytics:
                    self.training_analytics.log_loss(batch_idx, loss, acc, ppl, logits)
                
                # NaN/Inf detection
                if self.nan_inf_detector:
                    if not self.nan_inf_detector.detect_loss(loss):
                        if self.logger:
                            self.logger.log_error(
                                f"Batch {batch_idx}: Loss contains NaN/Inf, skipping"
                            )
                        continue
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / self.grad_accum_steps
                
                # Backward pass with AMP
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                micro += 1
                
                # Gradient accumulation: step only every N batches
                if micro >= self.grad_accum_steps:
                    # Unscale gradients (for clipping)
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Clip gradients
                    grad_norm = self.gradient_manager.clip_gradients(self.model)
                    if grad_norm is not None and (grad_norm == grad_norm and abs(grad_norm) != float("inf")):
                        total_gradient_norm += grad_norm
                        gradient_norm_batches += 1
                        
                        # Analytics: Log gradient summary
                        if self.training_analytics:
                            total_grad_norm = self.training_analytics.get_total_gradient_norm()
                            self.training_analytics.log_gradient_summary(batch_idx, total_grad_norm)
                            self.training_analytics.log_layer_gradient_stats(batch_idx)
                        
                        # Gradient explosion detection (after clipping)
                        if self.gradient_explosion_detector and grad_norm is not None:
                            explosion_result = self.gradient_explosion_detector.detect(
                                model=self.model,
                                max_grad_norm=self.max_grad_norm
                            )
                            if explosion_result["has_explosion"]:
                                if self.logger:
                                    self.logger.log_warning(
                                        f"Batch {batch_idx}: {explosion_result['recommendation']}"
                                    )
                    
                    # Optimizer step with AMP
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # ✅ YENİ: Warmup için batch-based scheduler step
                    # Warmup devam ediyorsa scheduler.step() çağır (batch-based)
                    if self.scheduler is not None:
                        self._step_scheduler_for_warmup()
                    
                    # Analytics: Track weight updates after optimizer step
                    if self.training_analytics:
                        self.training_analytics.track_weight_updates(batch_idx)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    micro = 0
                
                # Performance tracking: End batch timing
                if self.performance_tracker:
                    perf_stats = self.performance_tracker.end_batch(
                        batch_size=inputs.shape[0],
                        seq_len=inputs.shape[1] if inputs.dim() > 1 else 1
                    )
                
                # Memory tracking (every 10 batches)
                if self.memory_tracker and batch_idx % 10 == 0:
                    self.memory_tracker.track()
                
                # Accumulate metrics
                running_loss += float(loss.item())
                running_acc += float(acc)
                processed_batches += 1
                
                # Progress bar update — ana model LR (embedding param_groups[0] en düşük olabilir)
                if self.use_progress_bar and TQDM_AVAILABLE:
                    all_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups]
                    current_lr = max(all_lrs) if all_lrs else 0.0
                    epoch_pbar.set_postfix({
                        "loss": f"{float(loss.item()):.4f}",
                        "acc": f"{float(acc):.4f}",
                        "ppl": f"{float(ppl):.2f}",
                        "lr": f"{current_lr:.2e}"
                    })
                
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        f"Batch {batch_idx} sırasında hata: {e}",
                        exc_info=True
                    )
                raise RuntimeError(f"Training stopped at batch {batch_idx}: {e}") from e
        
        # Handle remaining gradient accumulation
        if micro > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = self.gradient_manager.clip_gradients(self.model)
            if grad_norm is not None and (grad_norm == grad_norm and abs(grad_norm) != float("inf")):
                total_gradient_norm += grad_norm
                gradient_norm_batches += 1
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad(set_to_none=True)
        
        # Close progress bar
        if self.use_progress_bar and TQDM_AVAILABLE:
            epoch_pbar.close()
        
        # Analytics: Log activation summary at end of epoch
        if self.training_analytics:
            self.training_analytics.log_activation_summary(processed_batches)
        
        # Calculate averages
        denom = max(1, processed_batches)
        avg_loss = running_loss / denom
        avg_acc = running_acc / denom
        avg_gradient_norm = total_gradient_norm / gradient_norm_batches if gradient_norm_batches > 0 else 0.0
        
        # [NEW] Epoch summary with advanced token metrics (from last batch)
        overall_epoch_acc = self.last_batch_metrics.get("overall_acc", avg_acc)
        special_epoch_accs = self.last_batch_metrics.get("special_accs", {})
        top5_epoch_acc = self.last_batch_metrics.get("top5_acc", 0.0)
        epoch_entropy = self.last_batch_metrics.get("entropy", 0.0)
        
        # Log epoch summary
        loss_change = None
        convergence_status = "Unknown"
        if self.prev_train_loss != float('inf'):
            loss_change = avg_loss - self.prev_train_loss
            if loss_change >= -0.0001:  # Not improving
                self.patience_counter += 1
                convergence_status = f"No improvement ({self.patience_counter}/{self.max_patience})"
            else:
                self.patience_counter = 0
                convergence_status = f"Improving (loss_delta: {loss_change:.4f})"
        
        self.enhanced_logger.log_epoch_summary(
            epoch=epoch,
            train_loss=avg_loss,
            val_loss=None,  # Will be set when validation runs
            overall_acc=overall_epoch_acc,
            special_accs=special_epoch_accs,
            top5_acc=top5_epoch_acc,
            entropy=epoch_entropy,
            loss_change=loss_change,
            convergence_status=convergence_status
        )
        
        self.prev_train_loss = avg_loss
        
        if self.logger:
            self.logger.log_info(
                f"Epoch tamamlandı. Ortalama Loss: {avg_loss:.4f}, "
                f"Accuracy: {avg_acc:.4f}, AvgGradNorm: {avg_gradient_norm:.4f}"
            )
        
        return float(avg_loss), float(avg_acc), float(avg_gradient_norm)
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Execute one validation epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number (for logging)
            
        Returns:
            Tuple of (avg_loss, avg_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        processed_batches = 0
        
        total_batches = max(1, len(val_loader))
        
        # Progress bar
        if self.use_progress_bar and TQDM_AVAILABLE:
            epoch_pbar = tqdm(
                val_loader,
                desc=f"Validation",
                leave=False,
                ncols=100,
                unit="batch"
            )
        else:
            epoch_pbar = val_loader
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(epoch_pbar, start=1):
                try:
                    # Parse batch
                    inputs, targets = self.batch_processor.parse_batch(batch)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass with AMP
                    with self._autocast_ctx():
                        outputs = self.model(inputs)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                    
                    # Validation: Check logits shape (if enabled)
                    if self.validation_manager:
                        try:
                            vocab_size = self.config.get("vocab_size")
                            if vocab_size:
                                expected_shape = (inputs.shape[0], inputs.shape[1] if inputs.dim() > 1 else 1)
                                self.validation_manager.validate_logits_shape(
                                    logits, expected_shape, vocab_size
                                )
                        except Exception as e:
                            if self.logger:
                                self.logger.log_error(f"Validation error in val batch {batch_idx}: {e}")
                            continue  # Skip invalid batch
                    
                    # Compute loss (CRITICAL: Uses criterion with EOS weight, label smoothing)
                    loss, acc, ppl = self.loss_computation.compute_loss(
                        logits=logits,
                        targets=targets,
                        pad_id=self.pad_token_id
                    )
                    
                    # [CRITICAL FIX] Advanced Token Metrics - EOS/BOS/PAD tracking
                    if batch_idx % 10 == 0:  # Her 10 batch'te log
                        try:
                            overall_acc, special_accs, top5_acc, entropy = self.advanced_metrics.compute_per_token_accuracy(
                                logits=logits,
                                targets=targets
                            )
                            
                            # Store for epoch summary
                            self.last_batch_metrics = {
                                "overall_acc": overall_acc,
                                "special_accs": special_accs,
                                "top5_acc": top5_acc,
                                "entropy": entropy
                            }
                            
                            # Check for critical issues in validation
                            critical_issues = self.advanced_metrics.check_critical_issues(
                                overall_acc, special_accs, entropy
                            )
                            
                            # Log validation batch with token metrics
                            self.enhanced_logger.log_batch_with_token_metrics(
                                epoch=epoch,
                                batch_idx=batch_idx,
                                loss=float(loss.item()),
                                overall_acc=overall_acc,
                                special_accs=special_accs,
                                top5_acc=top5_acc,
                                entropy=entropy,
                                grad_norm=None
                            )
                            
                            # Log critical warnings
                            if critical_issues:
                                self.enhanced_logger.log_critical_warnings(
                                    warnings=critical_issues,
                                    epoch=epoch,
                                    batch_idx=batch_idx
                                )
                        except Exception as e:
                            if self.logger:
                                self.logger.log_warning(f"[METRICS] Token metrics computation failed at val batch {batch_idx}: {e}")
                    
                    # Performance tracking: End batch timing
                    if self.performance_tracker:
                        self.performance_tracker.end_batch(
                            batch_size=inputs.shape[0],
                            seq_len=inputs.shape[1] if inputs.dim() > 1 else 1
                        )
                    
                    # NaN/Inf detection
                    if self.nan_inf_detector:
                        if not self.nan_inf_detector.detect_loss(loss):
                            if self.logger:
                                self.logger.log_error(
                                    f"Val batch {batch_idx}: Loss contains NaN/Inf, skipping"
                                )
                            continue
                    
                    # Performance tracking: End batch timing
                    if self.performance_tracker:
                        self.performance_tracker.end_batch(
                            batch_size=inputs.shape[0],
                            seq_len=inputs.shape[1] if inputs.dim() > 1 else 1
                        )
                    
                    # Accumulate metrics
                    running_loss += float(loss.item())
                    running_acc += float(acc)
                    processed_batches += 1
                    
                    # Progress bar update
                    if self.use_progress_bar and TQDM_AVAILABLE:
                        epoch_pbar.set_postfix({
                            "loss": f"{float(loss.item()):.4f}",
                            "acc": f"{float(acc):.4f}",
                            "ppl": f"{float(ppl):.2f}"
                        })
                
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(
                            f"Val batch {batch_idx} sırasında hata: {e}",
                            exc_info=True
                        )
                    raise RuntimeError(f"Validation stopped at batch {batch_idx}: {e}") from e
        
        # Close progress bar
        if self.use_progress_bar and TQDM_AVAILABLE:
            epoch_pbar.close()
        
        # Calculate averages
        denom = max(1, processed_batches)
        avg_loss = running_loss / denom
        avg_acc = running_acc / denom
        
        # [NEW] Validation epoch summary with advanced token metrics (from last batch)
        overall_val_acc = self.last_batch_metrics.get("overall_acc", avg_acc)
        special_val_accs = self.last_batch_metrics.get("special_accs", {})
        top5_val_acc = self.last_batch_metrics.get("top5_acc", 0.0)
        val_entropy = self.last_batch_metrics.get("entropy", 0.0)
        
        # Log validation epoch summary (use prev_train_loss for reference)
        self.enhanced_logger.log_epoch_summary(
            epoch=epoch,
            train_loss=self.prev_train_loss if self.prev_train_loss != float('inf') else None,
            val_loss=avg_loss,
            overall_acc=overall_val_acc,
            special_accs=special_val_accs,
            top5_acc=top5_val_acc,
            entropy=val_entropy,
            loss_change=None,
            convergence_status="Validation"
        )
        
        if self.logger:
            self.logger.log_info(
                f"Validation tamamlandı. Ortalama Loss: {avg_loss:.4f}, "
                f"Accuracy: {avg_acc:.4f}"
            )
        
        return float(avg_loss), float(avg_acc)
