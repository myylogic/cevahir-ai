"""
Test BatchProcessor
===================

Unit tests for BatchProcessor module.
"""

import pytest
import torch
from training_management.v2.core.batch_processor import BatchProcessor


class TestBatchProcessor:
    """Test BatchProcessor class"""
    
    @pytest.fixture
    def batch_processor(self):
        """BatchProcessor instance"""
        return BatchProcessor()
    
    @pytest.fixture
    def batch_size(self):
        """Batch size for tests"""
        return 4
    
    @pytest.fixture
    def seq_len(self):
        """Sequence length for tests"""
        return 16
    
    def test_parse_batch_tensor(self, batch_processor, batch_size, seq_len):
        """Test parsing tensor batch (LM default)"""
        # Arrange
        batch = torch.randint(0, 100, (batch_size, seq_len))
        
        # Act
        inputs, targets = batch_processor.parse_batch(batch)
        
        # Assert
        assert torch.equal(inputs, batch)
        assert torch.equal(targets, batch)
    
    def test_parse_batch_tuple(self, batch_processor, batch_size, seq_len):
        """Test parsing tuple batch"""
        # Arrange
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))
        batch = (inputs, targets)
        
        # Act
        parsed_inputs, parsed_targets = batch_processor.parse_batch(batch)
        
        # Assert
        assert torch.equal(parsed_inputs, inputs)
        assert torch.equal(parsed_targets, targets)
    
    def test_parse_batch_dict(self, batch_processor, batch_size, seq_len):
        """Test parsing dict batch"""
        # Arrange
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))
        batch = {"input_ids": inputs, "labels": targets}
        
        # Act
        parsed_inputs, parsed_targets = batch_processor.parse_batch(batch)
        
        # Assert
        assert torch.equal(parsed_inputs, inputs)
        assert torch.equal(parsed_targets, targets)
    
    def test_parse_batch_dict_alternative_keys(self, batch_processor, batch_size, seq_len):
        """Test parsing dict batch with alternative keys"""
        # Arrange
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))
        batch = {"inputs": inputs, "targets": targets}
        
        # Act
        parsed_inputs, parsed_targets = batch_processor.parse_batch(batch)
        
        # Assert
        assert torch.equal(parsed_inputs, inputs)
        assert torch.equal(parsed_targets, targets)
    
    def test_parse_batch_invalid_format(self, batch_processor):
        """Test parsing invalid batch format"""
        # Arrange
        batch = "invalid"
        
        # Act & Assert
        with pytest.raises(ValueError):
            batch_processor.parse_batch(batch)
    
    def test_validate_batch_valid(self, batch_processor, batch_size, seq_len):
        """Test batch validation with valid batch"""
        # Arrange
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))
        
        # Act
        result = batch_processor.validate_batch(
            inputs=inputs,
            targets=targets,
            expected_input_dim=2,
            expected_target_dim=2
        )
        
        # Assert
        assert result is True
    
    def test_validate_batch_invalid_input_type(self, batch_processor):
        """Test batch validation with invalid input type"""
        # Arrange
        inputs = "invalid"
        targets = torch.randint(0, 100, (4, 16))
        
        # Act & Assert
        with pytest.raises(TypeError):
            batch_processor.validate_batch(inputs=inputs, targets=targets)
    
    def test_validate_batch_dimension_mismatch(self, batch_processor, batch_size):
        """Test batch validation with dimension mismatch"""
        # Arrange
        inputs = torch.randint(0, 100, (batch_size, 16))
        targets = torch.randint(0, 100, (batch_size, 16))
        
        # Act & Assert
        with pytest.raises(ValueError):
            batch_processor.validate_batch(
                inputs=inputs,
                targets=targets,
                expected_input_dim=3  # Expecting 3D but got 2D
            )
    
    def test_validate_batch_batch_size_mismatch(self, batch_processor, seq_len):
        """Test batch validation with batch size mismatch"""
        # Arrange
        inputs = torch.randint(0, 100, (4, seq_len))
        targets = torch.randint(0, 100, (5, seq_len))  # Different batch size
        
        # Act & Assert
        with pytest.raises(ValueError):
            batch_processor.validate_batch(
                inputs=inputs,
                targets=targets,
                expected_input_dim=2
            )
