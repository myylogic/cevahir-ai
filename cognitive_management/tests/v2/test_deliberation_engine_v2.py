# -*- coding: utf-8 -*-
"""
Tests for DeliberationEngineV2
===============================
Unit tests for V2 Deliberation Engine component.
"""

import pytest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.v2.components.deliberation_engine_v2 import DeliberationEngineV2
from cognitive_management.cognitive_types import ThoughtCandidate, DecodingConfig
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError

from .conftest import (
    default_config,
    config_with_think_mode,
    mock_model_api,
    decoding_config,
)


class TestDeliberationEngineV2:
    """Test suite for DeliberationEngineV2"""
    
    def test_initialization(self, default_config, mock_model_api):
        """Test DeliberationEngineV2 initialization"""
        engine = DeliberationEngineV2(default_config, mock_model_api)
        assert engine.cfg == default_config
        assert engine.mm == mock_model_api
    
    def test_initialization_invalid_config(self, mock_model_api):
        """Test initialization with invalid config"""
        with pytest.raises(ValidationError):
            DeliberationEngineV2(None, mock_model_api)
    
    def test_initialization_invalid_model_api(self, default_config):
        """Test initialization with invalid model API"""
        invalid_api = Mock()
        # Missing generate method
        with pytest.raises(ValidationError):
            DeliberationEngineV2(default_config, invalid_api)
    
    def test_generate_thoughts_single(self, default_config, mock_model_api, decoding_config):
        """Test generating single thought (think1 mode)"""
        engine = DeliberationEngineV2(default_config, mock_model_api)
        mock_model_api.generate.return_value = "This is a test thought about the question."
        
        thoughts = engine.generate_thoughts(
            prompt="What is AI?",
            num_thoughts=1,
            decoding_config=decoding_config,
        )
        
        assert isinstance(thoughts, list)
        assert len(thoughts) == 1
        assert isinstance(thoughts[0], ThoughtCandidate)
        assert thoughts[0].text != ""
        assert isinstance(thoughts[0].score, float)
        mock_model_api.generate.assert_called()
    
    def test_generate_thoughts_multiple(self, default_config, mock_model_api, decoding_config):
        """Test generating multiple thoughts (debate2 mode)"""
        engine = DeliberationEngineV2(default_config, mock_model_api)
        mock_model_api.generate.side_effect = [
            "First perspective on the question.",
            "Second perspective on the question.",
        ]
        mock_model_api.score.return_value = 0.8
        
        thoughts = engine.generate_thoughts(
            prompt="What is consciousness?",
            num_thoughts=2,
            decoding_config=decoding_config,
        )
        
        assert isinstance(thoughts, list)
        assert len(thoughts) == 2
        for thought in thoughts:
            assert isinstance(thought, ThoughtCandidate)
            assert thought.text != ""
            assert isinstance(thought.score, float)
        assert mock_model_api.generate.call_count == 2
    
    def test_generate_thoughts_cot_pattern(self, default_config, mock_model_api, decoding_config):
        """Test Chain of Thought pattern implementation"""
        engine = DeliberationEngineV2(default_config, mock_model_api)
        mock_model_api.generate.return_value = "Step 1: Analyze the question. Step 2: Consider options. Step 3: Conclude."
        
        thoughts = engine.generate_thoughts(
            prompt="Solve this problem step by step.",
            num_thoughts=1,
            decoding_config=decoding_config,
        )
        
        assert len(thoughts) == 1
        # Check that CoT prompt was used (indirectly via generate call)
        call_args = mock_model_api.generate.call_args
        assert call_args is not None
        prompt = call_args[0][0]
        assert "step" in prompt.lower() or "think" in prompt.lower()
    
    def test_generate_thoughts_scoring(self, default_config, mock_model_api, decoding_config):
        """Test thought scoring"""
        engine = DeliberationEngineV2(default_config, mock_model_api)
        mock_model_api.generate.return_value = "A well-reasoned thought."
        mock_model_api.score.return_value = 0.9
        
        thoughts = engine.generate_thoughts(
            prompt="Test question",
            num_thoughts=1,
            decoding_config=decoding_config,
        )
        
        assert len(thoughts) == 1
        assert thoughts[0].score > 0.0
        mock_model_api.score.assert_called()
    
    def test_generate_thoughts_empty_response(self, default_config, mock_model_api, decoding_config):
        """Test handling of empty response"""
        engine = DeliberationEngineV2(default_config, mock_model_api)
        mock_model_api.generate.return_value = ""
        
        thoughts = engine.generate_thoughts(
            prompt="Test",
            num_thoughts=1,
            decoding_config=decoding_config,
        )
        
        # Should handle empty response gracefully
        assert isinstance(thoughts, list)
        if len(thoughts) > 0:
            # If thought is created, it should have very low score
            assert thoughts[0].score < 0.0
    
    def test_generate_thoughts_validation(self, default_config, mock_model_api, decoding_config):
        """Test input validation"""
        engine = DeliberationEngineV2(default_config, mock_model_api)
        
        # Empty prompt
        with pytest.raises(ValidationError):
            engine.generate_thoughts("", num_thoughts=1, decoding_config=decoding_config)
        
        # Invalid num_thoughts
        with pytest.raises(ValidationError):
            engine.generate_thoughts("Test", num_thoughts=0, decoding_config=decoding_config)
        
        with pytest.raises(ValidationError):
            engine.generate_thoughts("Test", num_thoughts=5, decoding_config=decoding_config)
    
    def test_generate_thoughts_adaptive_length(self, default_config, mock_model_api, decoding_config):
        """Test adaptive thought length"""
        engine = DeliberationEngineV2(default_config, mock_model_api)
        long_thought = "A" * 1000
        mock_model_api.generate.return_value = long_thought
        
        thoughts = engine.generate_thoughts(
            prompt="Test",
            num_thoughts=1,
            decoding_config=decoding_config,
        )
        
        assert len(thoughts) == 1
        # Thought should be truncated if too long
        assert len(thoughts[0].text) <= 800  # Max thought length for think1

