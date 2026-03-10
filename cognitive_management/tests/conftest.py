# -*- coding: utf-8 -*-
"""
Pytest Configuration and Fixtures
==================================
Test fixtures ve konfigürasyon dosyası.

Endüstri Standartları:
- pytest fixtures kullanımı
- Mock objects ve test doubles
- Test isolation
- Fixture scope management
"""

import pytest
from typing import Generator, Optional
from unittest.mock import Mock, MagicMock, patch
import asyncio
import sys
import types as _stdlib_types  # Import Python's standard types module first

# Ensure we import from cognitive_management.types, not stdlib types
from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput, CognitiveOutput, DecodingConfig


# ============================================================================
# Mock Model API
# ============================================================================

class MockModelAPI:
    """
    Mock Model API for testing.
    
    Test Edilen Dosya: cognitive_manager.py (ModelAPI Protocol)
    """
    
    def __init__(self):
        self.call_count = 0
        self.generate_calls = []
        self.score_calls = []
    
    def generate(self, prompt: str, decoding_cfg: Optional[DecodingConfig] = None) -> str:
        """Mock generate method."""
        self.call_count += 1
        self.generate_calls.append((prompt, decoding_cfg))
        return f"Mock response for: {prompt[:50]}..."
    
    def score(self, prompt: str, candidate: str) -> float:
        """Mock score method."""
        self.call_count += 1
        self.score_calls.append((prompt, candidate))
        return 0.85
    
    def estimate_entropy(self, text: str) -> float:
        """Mock entropy estimation."""
        return 0.5
    
    def process_audio(self, audio: bytes) -> str:
        """Mock audio processing."""
        return "Transcribed audio text"
    
    def process_image(self, image: bytes) -> str:
        """Mock image processing."""
        return "Image description"
    
    def process_multimodal(self, text: str = None, audio: bytes = None, image: bytes = None) -> str:
        """Mock multimodal processing."""
        return "Multimodal response"


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_model_api() -> MockModelAPI:
    """
    Mock Model API fixture.
    
    Test Edilen Dosya: cognitive_manager.py (ModelAPI Protocol)
    """
    return MockModelAPI()


@pytest.fixture(scope="function")
def default_config() -> CognitiveManagerConfig:
    """
    Default configuration fixture.
    
    Test Edilen Dosya: config.py (CognitiveManagerConfig)
    """
    return CognitiveManagerConfig()


@pytest.fixture(scope="function")
def cognitive_manager(mock_model_api: MockModelAPI, default_config: CognitiveManagerConfig) -> CognitiveManager:
    """
    CognitiveManager instance fixture.
    
    Test Edilen Dosya: cognitive_manager.py (CognitiveManager.__init__)
    """
    return CognitiveManager(
        model_manager=mock_model_api,
        cfg=default_config
    )


@pytest.fixture(scope="function")
def cognitive_state() -> CognitiveState:
    """
    CognitiveState fixture.
    
    Test Edilen Dosya: types.py (CognitiveState)
    """
    return CognitiveState()


@pytest.fixture(scope="function")
def cognitive_input() -> CognitiveInput:
    """
    CognitiveInput fixture.
    
    Test Edilen Dosya: types.py (CognitiveInput)
    """
    return CognitiveInput(user_message="Test message")


@pytest.fixture(scope="function")
def decoding_config() -> DecodingConfig:
    """
    DecodingConfig fixture.
    
    Test Edilen Dosya: types.py (DecodingConfig)
    """
    return DecodingConfig()


# ============================================================================
# Async Test Support
# ============================================================================

@pytest.fixture(scope="function")
def event_loop():
    """
    Event loop fixture for async tests.
    
    Test Edilen Dosya: cognitive_manager.py (handle_async)
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Test Utilities
# ============================================================================

def assert_cognitive_output(output: CognitiveOutput) -> None:
    """
    Assert CognitiveOutput validity.
    
    Test Edilen Dosya: types.py (CognitiveOutput)
    """
    assert output is not None
    assert hasattr(output, 'text')
    assert isinstance(output.text, str)
    assert hasattr(output, 'metadata')
    assert isinstance(output.metadata, dict)

#pytest cognitive_management/tests/ --cov=cognitive_management --cov-report=html
def create_test_state() -> CognitiveState:
    """
    Create test CognitiveState.
    
    Test Edilen Dosya: types.py (CognitiveState)
    """
    return CognitiveState()


def create_test_input(message: str = "Test message") -> CognitiveInput:
    """
    Create test CognitiveInput.
    
    Test Edilen Dosya: types.py (CognitiveInput)
    """
    return CognitiveInput(user_message=message)

