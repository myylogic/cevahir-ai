# -*- coding: utf-8 -*-
"""
Enhanced Training Logger - Advanced logging with token-level details
EOS, BOS, PAD gibi token'ları öğrenip öğrenmediğini track eder.
"""

import logging
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger("EnhancedTrainingLogger")


class EnhancedTrainingLogger:
    """Advanced logging with token-level metrics"""
    
    def __init__(self, logger_name: str = "EnhancedTrainingLogger"):
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_batch_with_token_metrics(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        overall_acc: float,
        special_accs: Dict[str, Tuple[float, int]],
        top5_acc: float,
        entropy: float,
        grad_norm: Optional[float] = None
    ) -> None:
        """Log detailed batch metrics including special tokens."""
        
        log_msg = f"\n[EPOCH {epoch}] Batch {batch_idx:4d}\n"
        log_msg += f"|- Loss: {loss:.6f}\n"
        log_msg += f"|- Overall Accuracy: {overall_acc:.2%}\n"
        log_msg += f"|- Special Tokens:\n"
        
        for token_name, (acc, count) in special_accs.items():
            if count > 0:
                if token_name == "PAD":
                    # PAD loss'ta ignore_index ile maskelendiği için model PAD tahmin etmek üzere eğitilmez; 0% beklenir.
                    status = "[MASKED]" if acc <= 0.1 else "[OK]"
                elif acc > 0.5:
                    status = "[OK]"
                elif acc > 0.1:
                    status = "[LOW]"
                else:
                    status = "[CRITICAL]"
                log_msg += (
                    f"|  |- {token_name:6s}: {acc:6.2%} ({count:4d} samples) {status}\n"
                )
        
        log_msg += f"|- Top-5 Accuracy: {top5_acc:.2%}\n"
        log_msg += f"|- Entropy (content only): {entropy:.4f}\n"
        
        if grad_norm is not None:
            log_msg += f"`- Gradient Norm: {grad_norm:.6f}\n"
        
        self.logger.info(log_msg)
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        overall_acc: float,
        special_accs: Dict[str, Tuple[float, int]],
        top5_acc: float,
        entropy: float,
        loss_change: Optional[float] = None,
        convergence_status: str = "Unknown"
    ) -> None:
        """Log epoch-level summary."""
        
        log_msg = f"\n{'='*80}\n"
        log_msg += f"[EPOCH {epoch} SUMMARY]\n"
        
        # Handle None values for train/val loss
        train_loss_str = f"{train_loss:.6f}" if train_loss is not None else "N/A"
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        log_msg += f"|- Train Loss: {train_loss_str} | Val Loss: {val_loss_str}\n"
        
        if loss_change is not None and train_loss is not None:
            improvement = -loss_change / train_loss * 100 if train_loss > 0 else 0
            log_msg += f"|- Loss Change: {loss_change:+.6f} ({improvement:+.2f}%)\n"
        
        log_msg += f"|- Overall Accuracy: {overall_acc:.2%}\n"
        log_msg += f"|- Special Token Accuracies:\n"
        
        for token_name, (acc, count) in special_accs.items():
            if count > 0:
                if token_name == "PAD":
                    marker = "[MASKED]" if acc <= 0.1 else "[OK]"  # PAD loss'ta ignore
                elif acc > 0.5:
                    marker = "[OK]"
                elif acc > 0.1:
                    marker = "[LOW]"
                else:
                    marker = "[CRIT]"
                log_msg += f"|  |- {token_name:6s}: {acc:6.2%} {marker}\n"
        
        log_msg += f"|- Top-5 Accuracy: {top5_acc:.2%}\n"
        log_msg += f"|- Entropy (content only): {entropy:.4f}\n"
        log_msg += f"`- Convergence: {convergence_status}\n"
        log_msg += f"{'='*80}\n"
        
        self.logger.info(log_msg)
    
    def log_critical_warnings(self, warnings: List[str], epoch: int, batch_idx: int) -> None:
        """Log critical training warnings."""
        if not warnings:
            return
        
        for warning in warnings:
            self.logger.warning(
                f"[EPOCH {epoch}] [BATCH {batch_idx}] {warning}"
            )
    
    def log_convergence_status(
        self,
        epoch: int,
        loss_change: float,
        patience: int,
        max_patience: int,
        ema_loss: float
    ) -> None:
        """Log convergence detection."""
        if patience > max_patience * 0.5:
            log_level = self.logger.warning
        else:
            log_level = self.logger.info
        
        status = "Stalled" if patience >= max_patience else f"Patience {patience}/{max_patience}"
        log_level(
            f"[CONVERGENCE] Epoch {epoch}: Loss change={loss_change:+.6f}, "
            f"EMA={ema_loss:.6f}, {status}"
        )
    
    def log_checkpoint_save(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        eos_acc: float,
        checkpoint_path: str
    ) -> None:
        """Log checkpoint saving."""
        self.logger.info(
            f"[CHECKPOINT] Epoch {epoch} saved | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"EOS Accuracy: {eos_acc:.2%} | Path: {checkpoint_path}"
        )
    
    def log_training_start(self, config: Dict) -> None:
        """Log training initialization."""
        log_msg = f"\n{'='*80}\n"
        log_msg += "[TRAINING START]\n"
        log_msg += f"|- Total Epochs: {config.get('num_epochs', 'N/A')}\n"
        log_msg += f"|- Batch Size: {config.get('batch_size', 'N/A')}\n"
        log_msg += f"|- Learning Rate: {config.get('learning_rate', 'N/A')}\n"
        log_msg += f"|- EOS Weight: {config.get('eos_token_weight', 'N/A')}\n"
        log_msg += f"|- Label Smoothing: {config.get('label_smoothing', 'N/A')}\n"
        log_msg += f"`- Special Tokens: EOS, BOS, PAD, UNK\n"
        log_msg += f"{'='*80}\n"
        
        self.logger.info(log_msg)
    
    def log_training_end(self, final_train_loss: float, final_val_loss: float) -> None:
        """Log training completion."""
        log_msg = f"\n{'='*80}\n"
        log_msg += "[TRAINING COMPLETE]\n"
        log_msg += f"|- Final Train Loss: {final_train_loss:.6f}\n"
        log_msg += f"|- Final Val Loss: {final_val_loss:.6f}\n"
        log_msg += f"`- Check 'best_model.pth' for best checkpoint\n"
        log_msg += f"{'='*80}\n"
        
        self.logger.info(log_msg)
