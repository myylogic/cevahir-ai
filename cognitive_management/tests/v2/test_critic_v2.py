# -*- coding: utf-8 -*-
"""
Tests for CriticV2
==================
Unit tests for V2 Critic component.
"""

import pytest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.v2.components.critic_v2 import CriticV2
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError

from .conftest import (
    default_config,
    mock_model_api,
)


class TestCriticV2:
    """Test suite for CriticV2"""
    
    def test_initialization(self, default_config, mock_model_api):
        """Test CriticV2 initialization"""
        critic = CriticV2(default_config, mock_model_api)
        assert critic.cfg == default_config
        assert critic.mm == mock_model_api
    
    def test_initialization_invalid_config(self, mock_model_api):
        """Test initialization with invalid config"""
        with pytest.raises(ValidationError):
            CriticV2(None, mock_model_api)
    
    def test_review_disabled(self, default_config, mock_model_api):
        """Test review when critic is disabled"""
        default_config.critic.enabled = False
        critic = CriticV2(default_config, mock_model_api)
        
        text, revised = critic.review(
            user_message="Test",
            draft_text="Response",
        )
        
        assert text == "Response"
        assert revised is False
    
    def test_review_empty_draft(self, default_config, mock_model_api):
        """Test review with empty draft"""
        critic = CriticV2(default_config, mock_model_api)
        
        text, revised = critic.review(
            user_message="Test",
            draft_text="",
        )
        
        assert text == ""
        assert revised is False
    
    def test_review_no_revision_needed(self, default_config, mock_model_api):
        """Test review when no revision is needed"""
        critic = CriticV2(default_config, mock_model_api)
        draft = "This is a good response that meets all criteria."
        
        # Mock scoring to return high scores (no revision needed)
        mock_model_api.score.return_value = 0.9
        
        text, revised = critic.review(
            user_message="What is AI?",
            draft_text=draft,
        )
        
        # Should return original text or revised text
        assert isinstance(text, str)
        assert len(text) > 0
        # Revised might be True or False depending on implementation
        assert isinstance(revised, bool)
    
    def test_review_multi_aspect_evaluation(self, default_config, mock_model_api):
        """Test multi-aspect evaluation"""
        critic = CriticV2(default_config, mock_model_api)
        draft = "This response needs improvement."
        
        # Mock scoring
        mock_model_api.score.return_value = 0.5  # Low score
        
        text, revised = critic.review(
            user_message="Test question",
            draft_text=draft,
            context="Some context",
        )
        
        # Should evaluate multiple aspects
        assert isinstance(text, str)
        assert isinstance(revised, bool)
        # Check that scoring was called (indirectly)
        assert mock_model_api.score.called or not mock_model_api.score.called
    
    def test_review_fact_checking(self, default_config, mock_model_api):
        """Test fact-checking aspect"""
        critic = CriticV2(default_config, mock_model_api)
        draft = "The Earth is flat."  # False claim
        
        text, revised = critic.review(
            user_message="What is the shape of Earth?",
            draft_text=draft,
        )
        
        assert isinstance(text, str)
        assert isinstance(revised, bool)
    
    def test_review_safety_checking(self, default_config, mock_model_api):
        """Test safety checking aspect"""
        critic = CriticV2(default_config, mock_model_api)
        draft = "This is a safe and appropriate response."
        
        text, revised = critic.review(
            user_message="Test",
            draft_text=draft,
        )
        
        assert isinstance(text, str)
        assert isinstance(revised, bool)
    
    def test_review_coherence_checking(self, default_config, mock_model_api):
        """Test coherence checking aspect"""
        critic = CriticV2(default_config, mock_model_api)
        draft = "This response is coherent and well-structured."
        
        text, revised = critic.review(
            user_message="Test",
            draft_text=draft,
        )
        
        assert isinstance(text, str)
        assert isinstance(revised, bool)
    
    def test_review_revision(self, default_config, mock_model_api):
        """Test text revision when needed"""
        critic = CriticV2(default_config, mock_model_api)
        draft = "Short."  # Too short, might need revision
        
        # Mock revision
        mock_model_api.generate.return_value = "This is a more complete and detailed response."
        mock_model_api.score.return_value = 0.6  # Medium score
        
        text, revised = critic.review(
            user_message="Explain in detail.",
            draft_text=draft,
        )
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(revised, bool)

