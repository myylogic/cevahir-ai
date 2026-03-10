"""
Test NaNInfDetector
===================

Unit tests for NaNInfDetector module.
"""

import pytest
import torch
import numpy as np
from training_management.v2.safety.nan_inf_detector import NaNInfDetector


class TestNaNInfDetector:
    """Test NaNInfDetector class"""
    
    @pytest.fixture
    def detector(self):
        """NaNInfDetector instance"""
        return NaNInfDetector()
    
    def test_detect_normal_tensor(self, detector):
        """Test detection with normal tensor"""
        # Arrange
        tensor = torch.randn(4, 8)
        
        # Act
        result = detector.detect(tensor, "test_tensor")
        
        # Assert
        assert isinstance(result, dict)
        assert result["has_nan"] is False
        assert result["has_inf"] is False
        assert result["is_finite"] is True
    
    def test_detect_nan(self, detector):
        """Test NaN detection"""
        # Arrange
        tensor = torch.randn(4, 8)
        tensor[0, 0] = float('nan')
        
        # Act
        result = detector.detect(tensor, "test_tensor")
        
        # Assert
        assert result["has_nan"] is True
        assert result["is_finite"] is False
    
    def test_detect_inf(self, detector):
        """Test Inf detection"""
        # Arrange
        tensor = torch.randn(4, 8)
        tensor[0, 0] = float('inf')
        
        # Act
        result = detector.detect(tensor, "test_tensor")
        
        # Assert
        assert result["has_inf"] is True
        assert result["is_finite"] is False
    
    def test_detect_loss_finite(self, detector):
        """Test loss detection with finite loss"""
        # Arrange
        loss = torch.tensor(1.5)
        
        # Act
        result = detector.detect_loss(loss)
        
        # Assert
        assert result is True
    
    def test_detect_loss_nan(self, detector):
        """Test loss detection with NaN loss"""
        # Arrange
        loss = torch.tensor(float('nan'))
        
        # Act
        result = detector.detect_loss(loss)
        
        # Assert
        assert result is False
    
    def test_detect_loss_inf(self, detector):
        """Test loss detection with Inf loss"""
        # Arrange
        loss = torch.tensor(float('inf'))
        
        # Act
        result = detector.detect_loss(loss)
        
        # Assert
        assert result is False
