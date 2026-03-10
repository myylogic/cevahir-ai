"""
Test GradientManager
====================

Unit tests for GradientManager module.
"""

import pytest
import torch
import torch.nn as nn
from training_management.v2.core.gradient_manager import GradientManager


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TestGradientManager:
    """Test GradientManager class"""
    
    @pytest.fixture
    def gradient_manager(self):
        """GradientManager instance"""
        return GradientManager(max_grad_norm=1.0)
    
    @pytest.fixture
    def model(self):
        """Simple model for testing"""
        return SimpleModel()
    
    @pytest.fixture
    def dummy_data(self):
        """Dummy data for forward pass"""
        batch_size = 4
        input_size = 10
        return torch.randn(batch_size, input_size)
    
    @pytest.fixture
    def dummy_targets(self):
        """Dummy targets"""
        batch_size = 4
        output_size = 5
        return torch.randint(0, output_size, (batch_size,))
    
    def test_calculate_gradient_norm(
        self,
        gradient_manager,
        model,
        dummy_data,
        dummy_targets
    ):
        """Test gradient norm calculation"""
        # Arrange
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Forward and backward
        outputs = model(dummy_data)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        
        # Act
        grad_norm = gradient_manager.calculate_gradient_norm(model)
        
        # Assert
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0
        assert grad_norm < float('inf')
    
    def test_calculate_gradient_norm_no_gradients(
        self,
        gradient_manager,
        model
    ):
        """Test gradient norm calculation when no gradients exist"""
        # Arrange
        # No backward call, so no gradients
        
        # Act
        grad_norm = gradient_manager.calculate_gradient_norm(model)
        
        # Assert
        assert grad_norm == 0.0
    
    def test_clip_gradients(
        self,
        gradient_manager,
        model,
        dummy_data,
        dummy_targets
    ):
        """Test gradient clipping"""
        # Arrange
        criterion = nn.CrossEntropyLoss()
        
        # Forward and backward
        outputs = model(dummy_data)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        
        # Get original gradient norm
        original_norm = gradient_manager.calculate_gradient_norm(model)
        
        # Create large gradients manually to test clipping
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(10.0)  # Make gradients 10x larger
        
        large_norm = gradient_manager.calculate_gradient_norm(model)
        assert large_norm > gradient_manager.max_grad_norm  # Should exceed threshold
        
        # Act
        clipped_norm = gradient_manager.clip_gradients(model)
        
        # Assert
        assert clipped_norm is not None
        final_norm = gradient_manager.calculate_gradient_norm(model)
        assert final_norm <= gradient_manager.max_grad_norm * 1.1  # Allow small tolerance
    
    def test_detect_gradient_issues_no_issues(
        self,
        gradient_manager,
        model,
        dummy_data,
        dummy_targets
    ):
        """Test gradient issue detection with normal gradients"""
        # Arrange
        criterion = nn.CrossEntropyLoss()
        
        # Forward and backward
        outputs = model(dummy_data)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        
        # Act
        issues = gradient_manager.detect_gradient_issues(model)
        
        # Assert
        assert isinstance(issues, dict)
        assert "has_nan" in issues
        assert "has_inf" in issues
        assert "has_explosion" in issues
        assert issues["has_nan"] is False
        assert issues["has_inf"] is False
    
    def test_detect_gradient_issues_nan(
        self,
        gradient_manager,
        model,
        dummy_data,
        dummy_targets
    ):
        """Test gradient issue detection with NaN gradients"""
        # Arrange
        criterion = nn.CrossEntropyLoss()
        
        # Forward and backward
        outputs = model(dummy_data)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        
        # Inject NaN
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data[0] = float('nan')
                break
        
        # Act
        issues = gradient_manager.detect_gradient_issues(model)
        
        # Assert
        assert issues["has_nan"] is True
