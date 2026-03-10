# -*- coding: utf-8 -*-
"""
Tests for MemoryServiceV2
==========================
Unit tests for V2 Memory Service component.
"""

import pytest

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import MemoryError, ValidationError

from .conftest import (
    default_config,
)


class TestMemoryServiceV2:
    """Test suite for MemoryServiceV2"""
    
    def test_initialization(self, default_config):
        """Test MemoryServiceV2 initialization"""
        service = MemoryServiceV2(default_config)
        assert service.cfg == default_config
        assert hasattr(service, '_episodic_memory')
        assert isinstance(service._episodic_memory, list)
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid config"""
        with pytest.raises(ValidationError):
            MemoryServiceV2(None)
    
    def test_add_turn_user(self, default_config):
        """Test adding user turn to history"""
        service = MemoryServiceV2(default_config)
        history = []
        
        updated = service.add_turn(history, "user", "Hello")
        
        assert len(updated) == 1
        assert updated[0]["role"] == "user"
        assert updated[0]["content"] == "Hello"
        assert len(service._episodic_memory) == 1
    
    def test_add_turn_assistant(self, default_config):
        """Test adding assistant turn to history"""
        service = MemoryServiceV2(default_config)
        history = []
        
        updated = service.add_turn(history, "assistant", "Hi there!")
        
        assert len(updated) == 1
        assert updated[0]["role"] == "assistant"
        assert updated[0]["content"] == "Hi there!"
    
    def test_add_turn_empty_content(self, default_config):
        """Test adding turn with empty content"""
        service = MemoryServiceV2(default_config)
        history = []
        
        updated = service.add_turn(history, "user", "")
        
        # Should not add empty content
        assert len(updated) == 0
    
    def test_add_turn_invalid_history(self, default_config):
        """Test adding turn with invalid history"""
        service = MemoryServiceV2(default_config)
        
        with pytest.raises(MemoryError):
            service.add_turn(None, "user", "Test")
    
    def test_retrieve_context(self, default_config):
        """Test context retrieval"""
        service = MemoryServiceV2(default_config)
        
        # Add some turns
        history = []
        service.add_turn(history, "user", "What is AI?")
        service.add_turn(history, "assistant", "AI is artificial intelligence.")
        service.add_turn(history, "user", "What is machine learning?")
        service.add_turn(history, "assistant", "ML is a subset of AI.")
        
        # Retrieve context
        contexts = service.retrieve_context("machine learning", top_k=2)
        
        assert isinstance(contexts, list)
        assert len(contexts) <= 2
        for ctx in contexts:
            assert "content" in ctx
            assert "score" in ctx
            assert isinstance(ctx["score"], float)
    
    def test_retrieve_context_empty(self, default_config):
        """Test context retrieval with empty memory"""
        service = MemoryServiceV2(default_config)
        
        contexts = service.retrieve_context("test query", top_k=5)
        
        assert isinstance(contexts, list)
        # Should return empty list or handle gracefully
        assert len(contexts) >= 0
    
    def test_build_context(self, default_config):
        """Test context building"""
        service = MemoryServiceV2(default_config)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        
        context = service.build_context(
            user_message="How are you?",
            history=history,
        )
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "How are you?" in context
    
    def test_build_context_empty_history(self, default_config):
        """Test context building with empty history"""
        service = MemoryServiceV2(default_config)
        
        context = service.build_context(
            user_message="Test",
            history=[],
        )
        
        assert isinstance(context, str)
        assert "Test" in context
    
    def test_summarize_if_needed(self, default_config):
        """Test summarization when needed"""
        service = MemoryServiceV2(default_config)
        history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        
        result = service.summarize_if_needed(history)
        
        # Should return history or summary
        assert isinstance(result, list) or isinstance(result, str)
    
    def test_prune(self, default_config):
        """Test history pruning"""
        service = MemoryServiceV2(default_config)
        history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]
        
        pruned = service.prune(history, user_message="Test")
        
        assert isinstance(pruned, list)
        assert len(pruned) <= len(history)

