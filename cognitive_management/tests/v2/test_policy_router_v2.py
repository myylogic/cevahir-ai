# -*- coding: utf-8 -*-
"""
Tests for PolicyRouterV2
========================
Unit tests for V2 Policy Router component.
"""

import pytest
from typing import Dict, Any

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import to avoid circular import with built-in types
import cognitive_management.types  # noqa: F401

from cognitive_management.v2.components.policy_router_v2 import PolicyRouterV2
from cognitive_management.cognitive_types import PolicyOutput, CognitiveState, Mode
from cognitive_management.config import CognitiveManagerConfig

from .conftest import (
    default_config,
    config_with_think_mode,
    empty_state,
    state_with_history,
    create_features,
    assert_policy_output_valid,
)


class TestPolicyRouterV2:
    """Test suite for PolicyRouterV2"""
    
    def test_initialization(self, default_config):
        """Test PolicyRouterV2 initialization"""
        router = PolicyRouterV2(default_config)
        assert router.cfg == default_config
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid config"""
        with pytest.raises(TypeError):
            PolicyRouterV2(None)
    
    def test_route_direct_mode(self, default_config, empty_state):
        """Test routing to direct mode for simple input"""
        router = PolicyRouterV2(default_config)
        features = create_features(entropy=0.3, input_length=5)
        
        output = router.route(features, empty_state)
        
        assert_policy_output_valid(output)
        assert output.mode == "direct"
        assert output.inner_steps == 0
    
    def test_route_think_mode(self, config_with_think_mode, empty_state):
        """Test routing to think mode for high entropy input"""
        router = PolicyRouterV2(config_with_think_mode)
        features = create_features(entropy=1.5, input_length=50)
        
        output = router.route(features, empty_state)
        
        assert_policy_output_valid(output)
        # Should route to think1 if entropy is high enough
        if config_with_think_mode.policy.allow_inner_steps:
            assert output.mode in ("direct", "think1")
    
    def test_route_debate_mode(self, config_with_think_mode, empty_state):
        """Test routing to debate mode for very high entropy"""
        router = PolicyRouterV2(config_with_think_mode)
        features = create_features(entropy=2.0, input_length=200)
        
        output = router.route(features, empty_state)
        
        assert_policy_output_valid(output)
        # Should route to debate2 if entropy is very high
        if config_with_think_mode.policy.debate_enabled:
            assert output.mode in ("direct", "think1", "debate2")
    
    def test_route_with_risk_score(self, default_config, empty_state):
        """Test routing with risk score"""
        router = PolicyRouterV2(default_config)
        features = create_features(
            entropy=0.5,
            input_length=20,
            has_claims=True,
            is_sensitive_domain=True,
        )
        
        output = router.route(features, empty_state)
        
        assert_policy_output_valid(output)
        assert output.tool in ("none", "maybe", "must")
    
    def test_route_adaptive_decoding(self, default_config, empty_state):
        """Test adaptive decoding based on features"""
        router = PolicyRouterV2(default_config)
        features = create_features(entropy=1.0, input_length=30)
        
        output = router.route(features, empty_state)
        
        assert_policy_output_valid(output)
        assert isinstance(output.decoding.max_new_tokens, int)
        assert output.decoding.max_new_tokens > 0
        assert 0.0 <= output.decoding.temperature <= 2.0
    
    def test_route_context_aware(self, default_config, state_with_history):
        """Test context-aware routing"""
        router = PolicyRouterV2(default_config)
        features = create_features(entropy=0.8, input_length=25)
        
        output = router.route(features, state_with_history)
        
        assert_policy_output_valid(output)
        # Should consider previous mode and entropy
        assert output.mode in ("direct", "think1", "debate2")
    
    def test_route_tool_selection(self, default_config, empty_state):
        """Test tool selection in routing"""
        router = PolicyRouterV2(default_config)
        
        # Test with needs_recent_info
        features = create_features(needs_recent_info=True)
        output = router.route(features, empty_state)
        assert output.tool in ("maybe", "must")
        
        # Test with needs_calc_or_parse
        features = create_features(needs_calc_or_parse=True)
        output = router.route(features, empty_state)
        assert output.tool in ("none", "maybe", "must")
    
    def test_route_edge_cases(self, default_config, empty_state):
        """Test edge cases"""
        router = PolicyRouterV2(default_config)
        
        # Very low entropy
        features = create_features(entropy=0.0, input_length=1)
        output = router.route(features, empty_state)
        assert_policy_output_valid(output)
        
        # Very high entropy
        features = create_features(entropy=3.0, input_length=1000)
        output = router.route(features, empty_state)
        assert_policy_output_valid(output)
        
        # Empty features
        features = {}
        output = router.route(features, empty_state)
        assert_policy_output_valid(output)

