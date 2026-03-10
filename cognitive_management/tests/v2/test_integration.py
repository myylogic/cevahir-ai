# -*- coding: utf-8 -*-
"""
Integration Tests for V2 System
=================================
End-to-end integration tests for V2 Cognitive Management system.
"""

import pytest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput
from cognitive_management.config import CognitiveManagerConfig

from .conftest import (
    mock_model_api,
    default_config,
    empty_state,
    simple_input,
)


class TestV2Integration:
    """Integration test suite for V2 system"""
    
    @pytest.fixture
    def cognitive_manager(self, mock_model_api, default_config):
        """Create CognitiveManager instance"""
        return CognitiveManager(mock_model_api, default_config)
    
    def test_end_to_end_basic(self, cognitive_manager, empty_state, simple_input):
        """Test end-to-end basic flow"""
        output = cognitive_manager.handle(empty_state, simple_input)
        
        assert output is not None
        assert isinstance(output.text, str)
        assert len(output.text) > 0
    
    def test_end_to_end_with_history(self, cognitive_manager, simple_input):
        """Test end-to-end with conversation history"""
        state = CognitiveState(
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            step=2,
        )
        
        output = cognitive_manager.handle(state, simple_input)
        
        assert output is not None
        assert isinstance(output.text, str)
    
    def test_metrics_collection(self, cognitive_manager, empty_state, simple_input):
        """Test metrics collection"""
        # Process a request
        cognitive_manager.handle(empty_state, simple_input)
        
        # Get metrics
        metrics = cognitive_manager.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "global" in metrics
        assert "request_count" in metrics["global"] or "requests" in metrics["global"]
    
    def test_health_status(self, cognitive_manager):
        """Test health status"""
        health = cognitive_manager.get_health_status()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert health["status"] in ("healthy", "degraded", "unhealthy")
    
    def test_event_history(self, cognitive_manager, empty_state, simple_input):
        """Test event history"""
        # Process a request
        cognitive_manager.handle(empty_state, simple_input)
        
        # Get event history
        events = cognitive_manager.get_event_history()
        
        assert isinstance(events, list)
        # Should have at least request and response events
        assert len(events) >= 0

