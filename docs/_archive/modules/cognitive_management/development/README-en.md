# Cognitive Management V2 - Development Guide

**Version:** 2.0  
**Last Updated:** 2025-01-27

---

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [Development Workflow](#development-workflow)
3. [Testing Guide](#testing-guide)
4. [Contribution Guidelines](#contribution-guidelines)

---

## Setup Instructions

### 1. Prerequisites

```bash
# Python 3.8+
python --version

# Required packages
pip install -r requirements.txt
```

### 2. Optional Dependencies

```bash
# For hot reload (config management)
pip install watchdog

# For YAML config support
pip install pyyaml
```

### 3. Project Structure

```
cognitive_management/
├── v2/                    # V2 implementation
│   ├── core/             # Core components
│   ├── components/        # V2 components
│   ├── processing/       # Processing pipeline
│   ├── middleware/       # Middleware
│   ├── monitoring/       # Monitoring
│   ├── config/           # Config management
│   └── ...
├── config.py             # Configuration classes
├── types.py              # Type definitions
├── cognitive_manager.py  # Main entry point
└── docs/                 # Documentation
```

---

## Development Workflow

### 1. Code Style

```python
# Use type hints
def handle(
    self,
    state: CognitiveState,
    request: CognitiveInput
) -> CognitiveOutput:
    ...

# Add docstrings
def process(self, context: ProcessingContext) -> ProcessingContext:
    """
    Process request through pipeline.
    
    Args:
        context: Processing context
        
    Returns:
        Updated context
    """
    ...
```

### 2. SOLID Principles

- **Single Responsibility:** Each class has a single responsibility
- **Open/Closed:** Open for extension, closed for modification
- **Liskov Substitution:** Interfaces must be implemented correctly
- **Interface Segregation:** Small, focused interfaces
- **Dependency Inversion:** Depend on abstractions

### 3. Testing

```bash
# Run tests
cd cognitive_management
python -m pytest tests/v2/ -v

# Coverage
python -m pytest tests/v2/ --cov=. --cov-report=html
```

### 4. Linting

```bash
# Lint check
pylint cognitive_management/

# Type check
mypy cognitive_management/
```

---

## Testing Guide

### 1. Unit Tests

```python
# tests/v2/test_policy_router_v2.py
import pytest
from cognitive_management.v2.components.policy_router_v2 import PolicyRouterV2
from cognitive_management.config import CognitiveManagerConfig

def test_route_direct_mode():
    cfg = CognitiveManagerConfig()
    router = PolicyRouterV2(cfg)
    
    features = {"entropy_est": 0.5, "input_length": 50}
    state = CognitiveState()
    
    output = router.route(features, state)
    assert output.mode == "direct"
```

### 2. Integration Tests

```python
# tests/v2/test_integration.py
def test_full_request_flow():
    cm = CognitiveManager(MockModelManager())
    state = CognitiveState()
    request = CognitiveInput(user_message="Test")
    
    response = cm.handle(state, request)
    assert response.text
    assert response.used_mode in ["direct", "think1", "debate2"]
```

### 3. Mock Objects

```python
# tests/v2/conftest.py
@pytest.fixture
def mock_backend():
    class MockBackend:
        def generate(self, prompt, decoding_config):
            return "Mock response"
        def score(self, prompt, candidate):
            return 0.8
    return MockBackend()
```

---

## Contribution Guidelines

### 1. Code Standards

- Type hints required
- Docstrings required
- SOLID principles compliant
- Test coverage 80%+

### 2. Commit Messages

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
test: Add tests
refactor: Code refactoring
```

### 3. Pull Request Process

1. Create feature branch
2. Make changes
3. Add tests
4. Pass lint check
5. Open PR

---

## Related Documentation

- [API Reference](../api/README-en.md)
- [Architecture Documentation](../architecture/README-en.md)
- [Usage Guides](../guides/README-en.md)
- [Main Documentation](../README-en.md)

---

**Version:** 2.0
