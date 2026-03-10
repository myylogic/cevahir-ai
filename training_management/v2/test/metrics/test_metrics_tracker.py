"""
Test MetricsTracker
===================

Unit tests for MetricsTracker module.
"""

import pytest
from training_management.v2.metrics.metrics_tracker import MetricsTracker


class TestMetricsTracker:
    """Test MetricsTracker class"""
    
    @pytest.fixture
    def tracker(self):
        """MetricsTracker instance"""
        return MetricsTracker()
    
    def test_update(self, tracker):
        """Test metrics update"""
        # Arrange
        epoch = 1
        train_loss = 1.5
        val_loss = 1.6
        accuracy = 0.85
        
        # Act
        tracker.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=accuracy
        )
        
        # Assert
        assert len(tracker.history["train_loss"]) == 1
        assert len(tracker.history["val_loss"]) == 1
        assert len(tracker.history["accuracy"]) == 1
        assert tracker.history["train_loss"][0] == train_loss
        assert tracker.history["val_loss"][0] == val_loss
        assert tracker.history["accuracy"][0] == accuracy
    
    def test_update_multiple(self, tracker):
        """Test multiple metrics updates"""
        # Arrange
        metrics = [
            (1, 1.5, 1.6, 0.85),
            (2, 1.3, 1.4, 0.87),
            (3, 1.1, 1.2, 0.89),
        ]
        
        # Act
        for epoch, train_loss, val_loss, accuracy in metrics:
            tracker.update(epoch, train_loss, val_loss, accuracy)
        
        # Assert
        assert len(tracker.history["train_loss"]) == 3
        assert len(tracker.history["val_loss"]) == 3
        assert len(tracker.history["accuracy"]) == 3
        assert tracker.history["train_loss"] == [1.5, 1.3, 1.1]
        assert tracker.history["val_loss"] == [1.6, 1.4, 1.2]
        assert tracker.history["accuracy"] == [0.85, 0.87, 0.89]
    
    def test_get_history(self, tracker):
        """Test history retrieval"""
        # Arrange
        tracker.update(1, 1.5, 1.6, 0.85)
        tracker.update(2, 1.3, 1.4, 0.87)
        
        # Act
        history = tracker.get_history()
        
        # Assert
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        assert "accuracy" in history
        assert len(history["train_loss"]) == 2
        
        # Should be a copy, not reference
        history["train_loss"].append(999.0)
        assert len(tracker.history["train_loss"]) == 2  # Original unchanged
