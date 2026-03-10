"""
Test GradientExplosionDetector
==============================

Unit tests for GradientExplosionDetector module.
"""

import pytest
import torch
import torch.nn as nn
from training_management.v2.safety.gradient_explosion_detector import GradientExplosionDetector


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class TestGradientExplosionDetector:
    """Test GradientExplosionDetector class"""
    
    @pytest.fixture
    def detector(self):
        """GradientExplosionDetector instance"""
        return GradientExplosionDetector(threshold=10.0)
    
    @pytest.fixture
    def model(self):
        """Simple model for testing"""
        return SimpleModel()
    
    @pytest.fixture
    def dummy_data(self):
        """Dummy data"""
        return torch.randn(4, 10)
    
    @pytest.fixture
    def dummy_targets(self):
        """Dummy targets"""
        return torch.randint(0, 5, (4,))
    
    def test_detect_no_explosion(
        self,
        detector,
        model,
        dummy_data,
        dummy_targets
    ):
        """Test detection with normal gradients"""
        # Arrange
        criterion = nn.CrossEntropyLoss()
        outputs = model(dummy_data)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        
        max_grad_norm = 1.0
        
        # Act
        result = detector.detect(model, max_grad_norm)
        
        # Assert
        assert isinstance(result, dict)
        assert "has_explosion" in result
        assert "max_grad_value" in result
        assert "total_norm" in result
        assert "recommendation" in result
        
        # Normal gradients shouldn't explode
        assert result["has_explosion"] is False
    
    def test_detect_explosion(
        self,
        detector,
        model,
        dummy_data,
        dummy_targets
    ):
        """Test detection with exploding gradients"""
        # Arrange
        criterion = nn.CrossEntropyLoss()
        outputs = model(dummy_data)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        
        # Make gradients explode (100x larger)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(100.0)
        
        max_grad_norm = 1.0
        
        # Act
        result = detector.detect(model, max_grad_norm)
        
        # Assert
        # Should detect explosion (norm > 1.0 * 10.0 = 10.0)
        assert result["has_explosion"] is True
        assert result["total_norm"] > max_grad_norm * detector.threshold
