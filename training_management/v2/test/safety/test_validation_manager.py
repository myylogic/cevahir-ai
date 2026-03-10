"""
Test ValidationManager
======================

Unit tests for ValidationManager module.
"""

import pytest
import torch
from training_management.v2.safety.validation_manager import ValidationManager


class TestValidationManager:
    """Test ValidationManager class"""
    
    @pytest.fixture
    def validator(self):
        """ValidationManager instance"""
        return ValidationManager()
    
    def test_validate_logits_shape_valid(self, validator):
        """Test logits shape validation with valid shape"""
        # Arrange
        logits = torch.randn(4, 16, 128)  # [B, T, V]
        expected_shape = (4, 16)
        vocab_size = 128
        
        # Act
        result = validator.validate_logits_shape(
            logits=logits,
            expected_shape=expected_shape,
            vocab_size=vocab_size
        )
        
        # Assert
        assert result is True
    
    def test_validate_logits_shape_wrong_dim(self, validator):
        """Test logits shape validation with wrong dimension"""
        # Arrange
        logits = torch.randn(4, 16)  # 2D instead of 3D
        expected_shape = (4, 16)
        vocab_size = 128
        
        # Act & Assert
        with pytest.raises(ValueError, match="must be 3D"):
            validator.validate_logits_shape(
                logits=logits,
                expected_shape=expected_shape,
                vocab_size=vocab_size
            )
    
    def test_validate_logits_shape_batch_mismatch(self, validator):
        """Test logits shape validation with batch size mismatch"""
        # Arrange
        logits = torch.randn(5, 16, 128)  # Batch size 5
        expected_shape = (4, 16)  # Expected batch size 4
        vocab_size = 128
        
        # Act & Assert
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validator.validate_logits_shape(
                logits=logits,
                expected_shape=expected_shape,
                vocab_size=vocab_size
            )
    
    def test_validate_logits_shape_seq_len_mismatch(self, validator):
        """Test logits shape validation with sequence length mismatch"""
        # Arrange
        logits = torch.randn(4, 20, 128)  # Seq len 20
        expected_shape = (4, 16)  # Expected seq len 16
        vocab_size = 128
        
        # Act & Assert
        with pytest.raises(ValueError, match="Sequence length mismatch"):
            validator.validate_logits_shape(
                logits=logits,
                expected_shape=expected_shape,
                vocab_size=vocab_size
            )
    
    def test_validate_logits_shape_vocab_mismatch(self, validator):
        """Test logits shape validation with vocab size mismatch"""
        # Arrange
        logits = torch.randn(4, 16, 256)  # Vocab size 256
        expected_shape = (4, 16)
        vocab_size = 128  # Expected vocab size 128
        
        # Act & Assert
        with pytest.raises(ValueError, match="Vocab size mismatch"):
            validator.validate_logits_shape(
                logits=logits,
                expected_shape=expected_shape,
                vocab_size=vocab_size
            )
