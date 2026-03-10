# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: loss_computation.py
Modül: training_management/v2/core
Görev: Loss Computation - Criterion desteği, masking ve padding handling ile
       loss computation. EOS weight ve label smoothing ile criterion kullanır.
       KRİTİK: Bu modül V1'deki bug'ları önlemek için criterion (EOS weight,
       label smoothing) kullanmalıdır.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (loss computation),
                     Dependency Inversion (criterion abstraksiyonuna bağımlı),
                     Open/Closed (farklı criterion türleri ile genişletilebilir)
- Design Patterns: Strategy Pattern (farklı criterion türleri için)
- Endüstri Standartları: Loss computation best practices

KULLANIM:
- Loss hesaplama için
- Criterion ile loss computation için
- Masking ve padding handling için

BAĞIMLILIKLAR:
- torch.nn.functional: Loss fonksiyonları
- torch: Tensor işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import math
from typing import Tuple, Optional, Any
import torch
import torch.nn.functional as F


class LossComputation:
    """
    Loss computation utility.
    
    Responsibilities:
    - Compute loss using criterion (with weight, label_smoothing support)
    - Apply padding masks
    - Calculate accuracy and perplexity
    
    SOLID: Single Responsibility Principle
    
    CRITICAL: Always uses criterion.weight and criterion.label_smoothing
    to ensure EOS weight and label smoothing are applied.
    """
    
    def __init__(
        self,
        criterion: torch.nn.Module,
        logger: Optional[Any] = None
    ):
        """
        Initialize LossComputation.
        
        Args:
            criterion: Loss function (e.g., CrossEntropyLoss)
            logger: Optional logger instance
        """
        self.criterion = criterion
        self.logger = logger
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pad_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute masked loss, accuracy, and perplexity.
        
        CRITICAL: This method uses criterion.weight and criterion.label_smoothing
        to ensure EOS weight and label smoothing are properly applied.
        
        Args:
            logits: Model output logits [B, T, V]
            targets: Target token IDs [B, T]
            pad_id: Padding token ID (None if no padding)
            
        Returns:
            Tuple of (loss_tensor, accuracy_float, perplexity_float)
            
        Raises:
            ValueError: If shapes don't match
        """
        B, T, V = logits.shape
        
        # Validate shapes
        if targets.shape != (B, T):
            raise ValueError(
                f"Targets shape expected {(B, T)}, got {tuple(targets.shape)}"
            )
        
        if logits.shape[1] != targets.shape[1]:
            error_msg = (
                f"CRITICAL ERROR: Input and Target have different lengths! "
                f"Logits seq_len={logits.shape[1]}, Targets seq_len={targets.shape[1]}. "
                f"This is WRONG for autoregressive training! "
                f"Input and Target must have the same length."
            )
            if self.logger:
                self.logger.log_error(error_msg)
            raise ValueError(error_msg)
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)
        
        # [OK] KRİTİK: AMP dtype uyumsuzluğu düzeltmesi
        # AMP kullanıldığında logits Half (float16) olabilir, ama targets Float (float32)
        # Cross entropy loss için logits'i float32'ye cast et (loss computation'da float32 daha stabil)
        if logits_flat.dtype != torch.float32:
            logits_flat = logits_flat.float()
        
        # CRITICAL: Extract weight and label_smoothing from criterion
        # This ensures EOS weight and label smoothing are applied
        weight = None
        if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
            weight = self.criterion.weight.to(logits_flat.device)
            # Weight'ın dtype'ı da logits ile uyumlu olmalı
            if weight.dtype != logits_flat.dtype:
                weight = weight.to(logits_flat.dtype)
        
        label_smoothing = getattr(self.criterion, 'label_smoothing', 0.0)
        
        # ignore_index: Use -100 to include all tokens, then mask
        # (Masking is handled separately for flexibility)
        ignore_index = -100
        
        # Compute loss per token (reduction="none")
        # CRITICAL: Use weight and label_smoothing from criterion
        loss_flat = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=weight,
            reduction="none",
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
        
        # Create padding mask
        if pad_id is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask = (targets != pad_id)
        
        # Apply mask and compute mean loss
        mask_flat = mask.view(-1).float()
        denom = mask_flat.sum().clamp_min(1.0)
        loss = (loss_flat * mask_flat).sum() / denom
        
        # Compute accuracy (masked)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == targets) & mask
            acc = correct.float().sum().item() / float(denom.item())
        
        # Compute perplexity
        loss_val = float(loss.item())
        ppl = math.exp(min(20.0, loss_val)) if math.isfinite(loss_val) else float("inf")
        
        return loss, float(acc), float(ppl)
    
    def compute_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pad_id: Optional[int] = None,
    ) -> float:
        """
        Compute accuracy only (without loss computation).
        
        Args:
            logits: Model output logits [B, T, V]
            targets: Target token IDs [B, T]
            pad_id: Padding token ID (None if no padding)
            
        Returns:
            Accuracy as float (0.0 to 1.0)
        """
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            
            if pad_id is None:
                mask = torch.ones_like(targets, dtype=torch.bool)
            else:
                mask = (targets != pad_id)
            
            correct = (preds == targets) & mask
            acc = correct.float().sum().item() / mask.float().sum().clamp_min(1.0).item()
            
            return float(acc)

