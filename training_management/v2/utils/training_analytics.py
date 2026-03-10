# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: training_analytics.py
Modül: training_management/v2/utils
Görev: Training Analytics - Detaylı training loop analitiği ve loglama.
       Gradient flow, weight updates, loss değerleri ve activation'ları
       izler ve loglar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (analytics ve logging),
                     Dependency Inversion (abstraksiyonlara bağımlı)
- Design Patterns: Observer Pattern (modeli izler ve loglar)
- Endüstri Standartları: PyTorch training monitoring best practices

KULLANIM:
- Training loop'ta detaylı analitik için
- Gradient flow monitoring için
- Weight update tracking için
- Loss ve activation monitoring için

BAĞIMLILIKLAR:
- torch: Neural network operations
- typing: Type hints

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from collections import defaultdict
import math


class TrainingAnalytics:
    """
    Training loop analytics and detailed logging.
    
    Responsibilities:
    - Track gradient flow (norms, vanishing/exploding)
    - Track weight updates
    - Monitor loss values
    - Monitor activation values (optional)
    - Log detailed training statistics
    
    SOLID: Single Responsibility Principle (analytics only)
    """
    
    def __init__(
        self,
        model: nn.Module,
        logger: Optional[Any] = None,
        log_every_n_batches: int = 100,
        log_gradients: bool = True,
        log_weight_updates: bool = True,
        log_loss_details: bool = True,
        log_activations: bool = False,
    ):
        """
        Initialize TrainingAnalytics.
        
        Args:
            model: Model to monitor
            logger: Logger instance
            log_every_n_batches: Log frequency (every N batches)
            log_gradients: Enable gradient logging
            log_weight_updates: Enable weight update logging
            log_loss_details: Enable detailed loss logging
            log_activations: Enable activation logging (can be slow)
        """
        self.model = model
        self.logger = logger
        self.log_every_n_batches = log_every_n_batches
        self.log_gradients = log_gradients
        self.log_weight_updates = log_weight_updates
        self.log_loss_details = log_loss_details
        self.log_activations = log_activations
        
        # Gradient tracking
        self.gradient_norms: Dict[str, List[float]] = defaultdict(list)
        self.gradient_hooks: List[Any] = []
        
        # Weight update tracking
        self.weight_snapshots: Dict[str, torch.Tensor] = {}
        self.weight_update_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Loss tracking
        self.loss_history: List[float] = []
        self.loss_details_history: List[Dict[str, Any]] = []
        
        # Activation tracking (optional)
        self.activation_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.activation_hooks: List[Any] = []
        
        # Register hooks if enabled
        if self.log_gradients:
            self._register_gradient_hooks()
        
        if self.log_activations:
            self._register_activation_hooks()
    
    def _register_gradient_hooks(self):
        """Register gradient hooks for monitoring."""
        def make_gradient_hook(name: str):
            def hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    self.gradient_norms[name].append(grad_norm)
                return grad
            return hook
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook_handle = param.register_hook(make_gradient_hook(name))
                self.gradient_hooks.append((name, hook_handle))
    
    def _register_activation_hooks(self):
        """Register forward hooks for activation monitoring."""
        def make_activation_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    with torch.no_grad():
                        mean = output.mean().item()
                        std = output.std().item()
                        min_val = output.min().item()
                        max_val = output.max().item()
                        has_nan = torch.isnan(output).any().item()
                        has_inf = torch.isinf(output).any().item()
                        
                        self.activation_stats[name].append({
                            "mean": mean,
                            "std": std,
                            "min": min_val,
                            "max": max_val,
                            "has_nan": has_nan,
                            "has_inf": has_inf,
                        })
            return hook
        
        # Register hooks for key modules
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm, nn.RMSNorm)):
                hook_handle = module.register_forward_hook(make_activation_hook(name))
                self.activation_hooks.append((name, hook_handle))
    
    def log_batch_start(self, batch_idx: int, epoch: int, total_batches: int):
        """Log batch start information."""
        if self.logger and batch_idx % self.log_every_n_batches == 0:
            self.logger.log_info(f"[Batch {batch_idx}/{total_batches}] (Epoch {epoch}) Processing...")
    
    def log_loss(
        self,
        batch_idx: int,
        loss: torch.Tensor,
        accuracy: Optional[float] = None,
        perplexity: Optional[float] = None,
        logits: Optional[torch.Tensor] = None,
    ):
        """Log loss and related metrics."""
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
        self.loss_history.append(loss_value)
        
        if self.log_loss_details and self.logger:
            loss_details = {
                "loss": loss_value,
                "accuracy": accuracy,
                "perplexity": perplexity,
            }
            
            if logits is not None:
                with torch.no_grad():
                    logits_mean = logits.mean().item()
                    logits_std = logits.std().item()
                    logits_min = logits.min().item()
                    logits_max = logits.max().item()
                    has_nan = torch.isnan(logits).any().item()
                    has_inf = torch.isinf(logits).any().item()
                    
                    loss_details.update({
                        "logits_mean": logits_mean,
                        "logits_std": logits_std,
                        "logits_min": logits_min,
                        "logits_max": logits_max,
                        "logits_has_nan": has_nan,
                        "logits_has_inf": has_inf,
                    })
            
            self.loss_details_history.append(loss_details)
    
    def snapshot_weights(self, batch_idx: int):
        """Take a snapshot of current weights for update tracking."""
        if not self.log_weight_updates:
            return
        
        # Snapshot a subset of weights (every N batches or first/last)
        if batch_idx == 1 or batch_idx % (self.log_every_n_batches * 5) == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.data.numel() < 1e6:  # Skip very large params
                    self.weight_snapshots[name] = param.data.clone().detach()
    
    def track_weight_updates(self, batch_idx: int):
        """Track weight updates since last snapshot."""
        if not self.log_weight_updates or not self.weight_snapshots:
            return
        
        updates_found = False
        
        for name, param in self.model.named_parameters():
            if name in self.weight_snapshots:
                old_weight = self.weight_snapshots[name]
                new_weight = param.data
                
                # Compute update statistics
                weight_diff = (new_weight - old_weight).abs()
                update_norm = weight_diff.norm().item()
                update_mean = weight_diff.mean().item()
                update_max = weight_diff.max().item()
                
                # Relative update (as fraction of weight magnitude)
                weight_norm = old_weight.norm().item()
                relative_update = update_norm / (weight_norm + 1e-8)
                
                self.weight_update_stats[name].append({
                    "batch": batch_idx,
                    "update_norm": update_norm,
                    "update_mean": update_mean,
                    "update_max": update_max,
                    "relative_update": relative_update,
                })
                
                updates_found = True
        
        if updates_found and self.logger and batch_idx % self.log_every_n_batches == 0:
            self._log_weight_updates_summary(batch_idx)
    
    def _log_weight_updates_summary(self, batch_idx: int):
        """Log summary of weight updates."""
        if not self.weight_update_stats:
            return
        
        # Compute average updates
        avg_update_norm = 0.0
        avg_relative_update = 0.0
        count = 0
        
        for name, stats_list in self.weight_update_stats.items():
            if stats_list:
                latest = stats_list[-1]
                avg_update_norm += latest["update_norm"]
                avg_relative_update += latest["relative_update"]
                count += 1
        
        if count > 0:
            avg_update_norm /= count
            avg_relative_update /= count
            
            self.logger.log_info(
                f"[Batch {batch_idx}] Weight Updates: "
                f"avg_norm={avg_update_norm:.6e}, "
                f"avg_relative={avg_relative_update:.6e}"
            )
    
    def log_gradient_summary(
        self,
        batch_idx: int,
        total_gradient_norm: Optional[float] = None,
    ):
        """Log gradient flow summary."""
        if not self.log_gradients or not self.logger:
            return
        
        if batch_idx % self.log_every_n_batches != 0:
            return
        
        # Compute statistics
        all_norms = []
        for name, norms in self.gradient_norms.items():
            if norms:
                all_norms.extend(norms[-self.log_every_n_batches:])  # Last N batches
        
        if not all_norms:
            return
        
        mean_norm = sum(all_norms) / len(all_norms)
        min_norm = min(all_norms)
        max_norm = max(all_norms)
        
        # Count vanishing/exploding
        vanishing = sum(1 for n in all_norms if n < 1e-6)
        exploding = sum(1 for n in all_norms if n > 1000)
        
        self.logger.log_info(
            f"[Batch {batch_idx}] Gradient Flow: "
            f"mean={mean_norm:.6e}, min={min_norm:.6e}, max={max_norm:.6e}, "
            f"vanishing={vanishing}, exploding={exploding}"
        )
        
        if total_gradient_norm is not None:
            self.logger.log_info(f"[Batch {batch_idx}] Total Gradient Norm: {total_gradient_norm:.6f}")
    
    def log_layer_gradient_stats(self, batch_idx: int):
        """Log gradient statistics by layer."""
        if not self.log_gradients or not self.logger:
            return
        
        if batch_idx % (self.log_every_n_batches * 2) != 0:  # Less frequent
            return
        
        # Group by layer
        layer_stats = defaultdict(lambda: {"norms": [], "count": 0})
        
        for name, norms in self.gradient_norms.items():
            if norms:
                layer_name = name.split('.')[0]  # First part
                layer_stats[layer_name]["norms"].extend(norms[-self.log_every_n_batches:])
                layer_stats[layer_name]["count"] += 1
        
        if layer_stats:
            self.logger.log_info(f"[Batch {batch_idx}] Layer Gradient Norms:")
            for layer_name, stats in sorted(layer_stats.items()):
                if stats["norms"]:
                    mean_norm = sum(stats["norms"]) / len(stats["norms"])
                    max_norm = max(stats["norms"])
                    min_norm = min(stats["norms"])
                    
                    self.logger.log_info(
                        f"  {layer_name}: mean={mean_norm:.6e}, "
                        f"min={min_norm:.6e}, max={max_norm:.6e} "
                        f"({stats['count']} params)"
                    )
    
    def log_activation_summary(self, batch_idx: int):
        """Log activation statistics summary."""
        if not self.log_activations or not self.logger:
            return
        
        if batch_idx % (self.log_every_n_batches * 3) != 0:  # Less frequent
            return
        
        if not self.activation_stats:
            return
        
        self.logger.log_info(f"[Batch {batch_idx}] Activation Statistics:")
        for name, stats_list in list(self.activation_stats.items())[:5]:  # First 5 modules
            if stats_list:
                latest = stats_list[-1]
                self.logger.log_info(
                    f"  {name}: mean={latest['mean']:.6f}, std={latest['std']:.6f}, "
                    f"range=[{latest['min']:.6f}, {latest['max']:.6f}]"
                )
                if latest['has_nan'] or latest['has_inf']:
                    self.logger.log_error(
                        f"  ⚠️ {name}: NaN={latest['has_nan']}, Inf={latest['has_inf']}"
                    )
    
    def log_epoch_summary(self, epoch: int, avg_loss: float, avg_accuracy: float):
        """Log epoch summary with detailed analytics."""
        if not self.logger:
            return
        
        self.logger.log_info("=" * 80)
        self.logger.log_info(f"EPOCH {epoch} ANALYTICS SUMMARY")
        self.logger.log_info("=" * 80)
        
        # Loss summary
        if self.loss_history:
            min_loss = min(self.loss_history)
            max_loss = max(self.loss_history)
            recent_losses = self.loss_history[-100:] if len(self.loss_history) > 100 else self.loss_history
            recent_avg = sum(recent_losses) / len(recent_losses)
            
            self.logger.log_info(f"Loss: avg={avg_loss:.6f}, min={min_loss:.6f}, max={max_loss:.6f}")
            self.logger.log_info(f"  Recent 100 batches avg: {recent_avg:.6f}")
        
        # Gradient summary
        if self.gradient_norms:
            all_norms = []
            for norms in self.gradient_norms.values():
                all_norms.extend(norms)
            
            if all_norms:
                mean_norm = sum(all_norms) / len(all_norms)
                min_norm = min(all_norms)
                max_norm = max(all_norms)
                vanishing = sum(1 for n in all_norms if n < 1e-6)
                exploding = sum(1 for n in all_norms if n > 1000)
                
                self.logger.log_info(
                    f"Gradients: mean={mean_norm:.6e}, min={min_norm:.6e}, max={max_norm:.6e}"
                )
                self.logger.log_info(f"  Vanishing: {vanishing}, Exploding: {exploding}")
        
        # Weight update summary
        if self.weight_update_stats:
            all_relative_updates = []
            for stats_list in self.weight_update_stats.values():
                if stats_list:
                    all_relative_updates.extend([s["relative_update"] for s in stats_list])
            
            if all_relative_updates:
                avg_update = sum(all_relative_updates) / len(all_relative_updates)
                max_update = max(all_relative_updates)
                self.logger.log_info(
                    f"Weight Updates: avg_relative={avg_update:.6e}, max_relative={max_update:.6e}"
                )
        
        self.logger.log_info("=" * 80)
        
        # Clear history for next epoch (keep recent for trending)
        self.loss_history = self.loss_history[-1000:] if len(self.loss_history) > 1000 else self.loss_history
        self.gradient_norms = defaultdict(list)  # Clear but keep structure
    
    def get_total_gradient_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                total_norm += param_norm ** 2
        return math.sqrt(total_norm) if total_norm > 0 else 0.0
    
    def cleanup(self):
        """Remove all hooks."""
        for _, hook_handle in self.gradient_hooks:
            hook_handle.remove()
        self.gradient_hooks.clear()
        
        for _, hook_handle in self.activation_hooks:
            hook_handle.remove()
        self.activation_hooks.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()

