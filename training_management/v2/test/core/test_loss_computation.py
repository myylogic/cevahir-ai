"""
Test LossComputation
====================

Unit tests for LossComputation module.

CRITICAL: Test criterion usage (EOS weight, label smoothing)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from training_management.v2.core.loss_computation import LossComputation


class TestLossComputation:
    """Test LossComputation class"""
    
    @pytest.fixture
    def vocab_size(self):
        """Vocabulary size for tests"""
        return 128
    
    @pytest.fixture
    def batch_size(self):
        """Batch size for tests"""
        return 4
    
    @pytest.fixture
    def seq_len(self):
        """Sequence length for tests"""
        return 16
    
    @pytest.fixture
    def simple_criterion(self, vocab_size):
        """Simple criterion without weights"""
        return nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    
    @pytest.fixture
    def eos_weighted_criterion(self, vocab_size):
        """Criterion with EOS weight and label smoothing"""
        eos_id = 3
        device = torch.device("cpu")
        loss_weights = torch.ones(vocab_size, device=device)
        loss_weights[eos_id] = 0.1  # EOS weight
        
        return nn.CrossEntropyLoss(
            weight=loss_weights,
            ignore_index=-100,
            reduction="mean",
            label_smoothing=0.1
        )
    
    def test_compute_loss_basic(
        self, 
        simple_criterion, 
        vocab_size, 
        batch_size, 
        seq_len
    ):
        """Test basic loss computation"""
        # Arrange
        loss_computation = LossComputation(simple_criterion)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        pad_id = None
        
        # Act
        loss, acc, ppl = loss_computation.compute_loss(
            logits=logits,
            targets=targets,
            pad_id=pad_id
        )
        
        # Assert
        assert isinstance(loss, torch.Tensor)
        assert isinstance(acc, float)
        assert isinstance(ppl, float)
        assert 0.0 <= acc <= 1.0
        assert ppl > 0.0
        assert torch.isfinite(loss)
    
    def test_compute_loss_with_padding(
        self,
        simple_criterion,
        vocab_size,
        batch_size,
        seq_len
    ):
        """Test loss computation with padding tokens"""
        # Arrange
        loss_computation = LossComputation(simple_criterion)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(1, vocab_size, (batch_size, seq_len))
        targets[0, -1] = 0  # Add padding token
        pad_id = 0
        
        # Act
        loss, acc, ppl = loss_computation.compute_loss(
            logits=logits,
            targets=targets,
            pad_id=pad_id
        )
        
        # Assert
        assert torch.isfinite(loss)
        assert 0.0 <= acc <= 1.0
    
    def test_compute_loss_criterion_usage(
        self,
        eos_weighted_criterion,
        vocab_size,
        batch_size,
        seq_len
    ):
        """
        CRITICAL TEST: Criterion (EOS weight'li) kullanılıyor mu?
        
        Bu test, V1'deki kritik hatayı yakalayacak şekilde yazılmıştır.
        LossComputation, criterion.weight ve criterion.label_smoothing'i 
        kullanmalıdır.
        """
        # Arrange
        loss_computation = LossComputation(eos_weighted_criterion)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets[:, -1] = 3  # EOS token (id=3) son pozisyona
        pad_id = 0
        
        # Act - LossComputation kullan
        loss_with_computation, acc, ppl = loss_computation.compute_loss(
            logits=logits,
            targets=targets,
            pad_id=pad_id
        )
        
        # Manually compute with criterion (EOS weight'li) - MASK UYGULANMIŞ
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Mask uygula
        mask = (targets_flat != pad_id)
        mask_float = mask.float()
        denom = mask_float.sum().clamp_min(1.0)
        
        # Criterion'ın weight ve label_smoothing'ini al
        weight = eos_weighted_criterion.weight.to(logits_flat.device)
        label_smoothing = getattr(eos_weighted_criterion, 'label_smoothing', 0.0)
        
        # Criterion ile loss hesapla (reduction="none" ile)
        loss_flat_with_criterion = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=weight,
            reduction="none",
            label_smoothing=label_smoothing,
            ignore_index=-100  # Mask ile filtreleme yapıyoruz
        )
        loss_with_criterion_masked = (loss_flat_with_criterion * mask_float).sum() / denom
        
        # F.cross_entropy ile loss hesapla (EOS weight YOK) - MASK UYGULANMIŞ
        loss_flat_with_f_cross_entropy = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
            ignore_index=-100
        )
        loss_with_f_cross_entropy_masked = (loss_flat_with_f_cross_entropy * mask_float).sum() / denom
        
        # Assert
        # KRİTİK TEST: LossComputation, criterion loss ile AYNI olmalı
        loss_diff_vs_criterion = abs(loss_with_computation.item() - loss_with_criterion_masked.item())
        loss_diff_vs_f_cross_entropy = abs(loss_with_computation.item() - loss_with_f_cross_entropy_masked.item())
        
        # Criterion kullanılıyorsa: computation loss ≈ criterion loss olmalı
        criterion_is_used = loss_diff_vs_criterion < 1e-3
        
        # F.cross_entropy kullanılıyorsa: computation loss ≈ f_cross_entropy loss olur (HATA!)
        f_cross_entropy_is_used = loss_diff_vs_f_cross_entropy < 1e-3
        
        # TEST: Criterion kullanılmalı
        assert criterion_is_used, (
            f"KRİTİK HATA: LossComputation criterion kullanmıyor! "
            f"Computation loss={loss_with_computation.item():.6f}, "
            f"Criterion loss (masked)={loss_with_criterion_masked.item():.6f}, "
            f"F.cross_entropy loss (masked)={loss_with_f_cross_entropy_masked.item():.6f}. "
            f"Criterion (EOS weight'li) kullanılmalı!"
        )
        
        # F.cross_entropy kullanılmamalı (criterion kullanılmalı)
        assert not f_cross_entropy_is_used or criterion_is_used, (
            f"KRİTİK HATA: F.cross_entropy kullanılıyor, criterion kullanılmıyor! "
            f"EOS weight uygulanmıyor!"
        )
    
    def test_compute_loss_shape_validation(
        self,
        simple_criterion,
        vocab_size,
        batch_size,
        seq_len
    ):
        """Test loss computation with invalid shapes"""
        # Arrange
        loss_computation = LossComputation(simple_criterion)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len + 1))  # Wrong shape
        pad_id = 0
        
        # Act & Assert
        with pytest.raises(ValueError):
            loss_computation.compute_loss(
                logits=logits,
                targets=targets,
                pad_id=pad_id
            )
    
    def test_compute_accuracy(
        self,
        simple_criterion,
        vocab_size,
        batch_size,
        seq_len
    ):
        """Test accuracy computation"""
        # Arrange
        loss_computation = LossComputation(simple_criterion)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        pad_id = None
        
        # Act
        acc = loss_computation.compute_accuracy(
            logits=logits,
            targets=targets,
            pad_id=pad_id
        )
        
        # Assert
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
