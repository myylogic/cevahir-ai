"""
Test MetricsCalculator
======================

Unit tests for MetricsCalculator module.
"""

import pytest
import torch
from training_management.v2.metrics.metrics_calculator import MetricsCalculator


class TestMetricsCalculator:
    """Test MetricsCalculator class"""
    
    @pytest.fixture
    def calculator(self):
        """MetricsCalculator instance"""
        return MetricsCalculator()
    
    @pytest.fixture
    def vocab_size(self):
        """Vocabulary size"""
        return 128
    
    @pytest.fixture
    def batch_size(self):
        """Batch size"""
        return 4
    
    @pytest.fixture
    def seq_len(self):
        """Sequence length"""
        return 16
    
    def test_calculate_accuracy(
        self,
        calculator,
        vocab_size,
        batch_size,
        seq_len
    ):
        """Test accuracy calculation"""
        # Arrange
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        pad_id = None
        
        # Act
        acc = calculator.calculate_accuracy(
            logits=logits,
            targets=targets,
            pad_id=pad_id
        )
        
        # Assert
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
    
    def test_calculate_accuracy_with_padding(
        self,
        calculator,
        vocab_size,
        batch_size,
        seq_len
    ):
        """Test accuracy calculation with padding"""
        # Arrange
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(1, vocab_size, (batch_size, seq_len))
        targets[0, -1] = 0  # Padding token
        pad_id = 0
        
        # Act
        acc = calculator.calculate_accuracy(
            logits=logits,
            targets=targets,
            pad_id=pad_id
        )
        
        # Assert
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
    
    def test_calculate_perplexity_finite(self, calculator):
        """Test perplexity calculation with finite loss"""
        # Arrange
        loss = 2.0
        
        # Act
        ppl = calculator.calculate_perplexity(loss)
        
        # Assert
        assert isinstance(ppl, float)
        assert ppl > 0.0
        assert ppl < float('inf')
        # exp(2.0) ≈ 7.39
        assert abs(ppl - 7.389) < 1.0
    
    def test_calculate_perplexity_inf(self, calculator):
        """Test perplexity calculation with infinite loss"""
        # Arrange
        loss = float('inf')
        
        # Act
        ppl = calculator.calculate_perplexity(loss)
        
        # Assert
        assert ppl == float('inf')
    
    def test_calculate_perplexity_nan(self, calculator):
        """Test perplexity calculation with NaN loss"""
        # Arrange
        loss = float('nan')
        
        # Act
        ppl = calculator.calculate_perplexity(loss)
        
        # Assert
        assert ppl == float('inf')
    
    def test_calculate_perplexity_large_loss(self, calculator):
        """Test perplexity calculation with large loss (should be capped)"""
        # Arrange
        loss = 100.0  # Very large loss
        
        # Act
        ppl = calculator.calculate_perplexity(loss, max_exp=20.0)
        
        # Assert
        # Should be capped at exp(20.0) ≈ 485165195.41
        assert ppl == pytest.approx(485165195.41, rel=1e-3)
