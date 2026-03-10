# -*- coding: utf-8 -*-
"""
Pytest Configuration and Fixtures
==================================
Shared fixtures and configuration for V2 tests.
"""

import pytest
from typing import Dict, Any, Protocol
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import with explicit path to avoid circular import
from cognitive_management import types as cm_types
from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
    DecodingConfig,
    PolicyOutput,
    ThoughtCandidate,
)
from cognitive_management.config import CognitiveManagerConfig


# =============================================================================
# Mock Model API
# =============================================================================

class MockModelAPI(Protocol):
    """Mock Model API for testing"""
    def generate(self, prompt: str, decoding_cfg: DecodingConfig) -> str: ...
    def score(self, prompt: str, candidate: str) -> float: ...
    def entropy_estimate(self, text: str) -> float: ...


@pytest.fixture
def mock_model_api() -> MockModelAPI:
    """Create a mock model API"""
    mock = MagicMock(spec=MockModelAPI)
    mock.generate = Mock(return_value="Mocked response")
    mock.score = Mock(return_value=0.8)
    mock.entropy_estimate = Mock(return_value=1.5)
    return mock


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_config() -> CognitiveManagerConfig:
    """Create default configuration"""
    return CognitiveManagerConfig()


@pytest.fixture
def config_with_think_mode() -> CognitiveManagerConfig:
    """Create configuration with think mode enabled"""
    cfg = CognitiveManagerConfig()
    cfg.policy.allow_inner_steps = True
    cfg.policy.debate_enabled = True
    cfg.policy.entropy_gate_think = 0.8
    cfg.policy.entropy_gate_debate = 1.2
    return cfg


# =============================================================================
# State Fixtures
# =============================================================================

@pytest.fixture
def empty_state() -> CognitiveState:
    """Create empty cognitive state"""
    return CognitiveState(
        history=[],
        step=0,
        last_entropy=None,
        last_mode=None,
    )


@pytest.fixture
def state_with_history() -> CognitiveState:
    """Create cognitive state with conversation history"""
    return CognitiveState(
        history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        step=2,
        last_entropy=0.5,
        last_mode="direct",
    )


# =============================================================================
# Input/Output Fixtures
# =============================================================================

@pytest.fixture
def simple_input() -> CognitiveInput:
    """Create simple cognitive input"""
    return CognitiveInput(
        user_message="What is AI?",
        system_prompt=None,
    )


@pytest.fixture
def complex_input() -> CognitiveInput:
    """Create complex cognitive input with high entropy"""
    return CognitiveInput(
        user_message="Can you explain quantum mechanics and how it relates to consciousness?",
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def decoding_config() -> DecodingConfig:
    """Create decoding configuration"""
    return DecodingConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )


# =============================================================================
# Component Fixtures
# =============================================================================

@pytest.fixture
def mock_policy_output() -> PolicyOutput:
    """Create mock policy output"""
    return PolicyOutput(
        mode="direct",
        tool="none",
        decoding=DecodingConfig(),
        inner_steps=0,
    )


@pytest.fixture
def mock_thought_candidate() -> ThoughtCandidate:
    """Create mock thought candidate"""
    return ThoughtCandidate(
        text="This is a test thought.",
        score=0.85,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def create_features(
    entropy: float = 0.5,
    input_length: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """Create feature dictionary for testing"""
    features = {
        "entropy_est": entropy,
        "input_length": input_length,
        "needs_recent_info": False,
        "needs_calc_or_parse": False,
        "has_claims": False,
        "is_sensitive_domain": False,
    }
    features.update(kwargs)
    return features


def assert_policy_output_valid(output: PolicyOutput) -> None:
    """Assert that policy output is valid"""
    assert output.mode in ("direct", "think1", "debate2")
    assert output.tool in ("none", "maybe", "must")
    assert isinstance(output.decoding, DecodingConfig)
    assert isinstance(output.inner_steps, int)
    assert output.inner_steps >= 0


def assert_cognitive_output_valid(output: CognitiveOutput) -> None:
    """Assert that cognitive output is valid"""
    assert isinstance(output.text, str)
    assert output.used_mode in ("direct", "think1", "debate2")
    assert output.tool_used in (None, "none", "maybe", "must")
    assert isinstance(output.revised_by_critic, bool)

