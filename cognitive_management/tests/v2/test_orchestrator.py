# -*- coding: utf-8 -*-
"""
Tests for CognitiveOrchestrator
================================
Unit tests for V2 Orchestrator component.
"""

import pytest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.v2.core.orchestrator import CognitiveOrchestrator
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput, CognitiveOutput
from cognitive_management.v2.components import (
    PolicyRouterV2,
    MemoryServiceV2,
    CriticV2,
    DeliberationEngineV2,
)
from cognitive_management.v2.adapters.backend_adapter import ModelAPIAdapter
from cognitive_management.v2.events.event_bus import EventBus

from .conftest import (
    default_config,
    mock_model_api,
    empty_state,
    simple_input,
)


class TestCognitiveOrchestrator:
    """Test suite for CognitiveOrchestrator"""
    
    @pytest.fixture
    def orchestrator(self, default_config, mock_model_api):
        """Create orchestrator instance"""
        backend = ModelAPIAdapter(mock_model_api)
        policy_router = PolicyRouterV2(default_config)
        memory_service = MemoryServiceV2(default_config)
        critic = CriticV2(default_config, mock_model_api)
        event_bus = EventBus()
        
        return CognitiveOrchestrator(
            backend=backend,
            policy_router=policy_router,
            memory_service=memory_service,
            critic=critic,
            event_bus=event_bus,
        )
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.backend is not None
        assert orchestrator.policy_router is not None
        assert orchestrator.memory_service is not None
        assert orchestrator.critic is not None
        assert orchestrator.event_bus is not None
        assert orchestrator.pipeline is not None
    
    def test_handle_basic(self, orchestrator, empty_state, simple_input):
        """Test basic handle operation"""
        output = orchestrator.handle(empty_state, simple_input)
        
        assert isinstance(output, CognitiveOutput)
        assert isinstance(output.text, str)
        assert output.used_mode in ("direct", "think1", "debate2")
        assert isinstance(output.revised_by_critic, bool)
    
    def test_handle_with_middleware(self, orchestrator, empty_state, simple_input):
        """Test handle with middleware"""
        from cognitive_management.v2.middleware import ValidationMiddleware
        
        middleware = ValidationMiddleware(max_message_length=1000, min_message_length=1)
        orchestrator.middleware_chain = orchestrator._build_middleware_chain([middleware])
        
        output = orchestrator.handle(empty_state, simple_input)
        
        assert isinstance(output, CognitiveOutput)
    
    def test_handle_event_publishing(self, orchestrator, empty_state, simple_input):
        """Test event publishing"""
        event_count_before = len(orchestrator.event_bus._event_history)
        
        output = orchestrator.handle(empty_state, simple_input)
        
        # Should publish events
        event_count_after = len(orchestrator.event_bus._event_history)
        assert event_count_after > event_count_before
    
    def test_handle_error_handling(self, orchestrator, empty_state):
        """Test error handling"""
        # Invalid input
        invalid_input = None
        
        with pytest.raises(Exception):
            orchestrator.handle(empty_state, invalid_input)

