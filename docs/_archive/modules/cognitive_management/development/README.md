# Cognitive Management V2 - Development Guide

**Versiyon:** 2.0  
**Son Güncelleme:** 2025-01-27

---

## 📋 İÇİNDEKİLER

1. [Setup Instructions](#setup-instructions)
2. [Development Workflow](#development-workflow)
3. [Testing Guide](#testing-guide)
4. [Contribution Guidelines](#contribution-guidelines)

---

## 🛠️ Setup Instructions

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

## 🔄 Development Workflow

### 1. Code Style

```python
# Type hints kullan
def handle(
    self,
    state: CognitiveState,
    request: CognitiveInput
) -> CognitiveOutput:
    ...

# Docstrings ekle
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

- **Single Responsibility:** Her class tek bir sorumluluğa sahip
- **Open/Closed:** Extension için açık, modification için kapalı
- **Liskov Substitution:** Interface'ler doğru implement edilmeli
- **Interface Segregation:** Küçük, odaklı interface'ler
- **Dependency Inversion:** Abstractions'a bağımlılık

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

## 🧪 Testing Guide

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

## 📝 Contribution Guidelines

### 1. Code Standards

- Type hints zorunlu
- Docstrings zorunlu
- SOLID principles uyumlu
- Test coverage %80+

### 2. Commit Messages

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
test: Add tests
refactor: Code refactoring
```

### 3. Pull Request Process

1. Feature branch oluştur
2. Değişiklikleri yap
3. Tests ekle
4. Lint check geç
5. PR oluştur

---

## 🔗 İlgili Dokümantasyon

- [API Reference](../api/README.md)
- [Architecture Documentation](../architecture/README.md)
- [Usage Guides](../guides/README.md)
- [Main Documentation](../README.md)

---

**Hazırlayan:** AI Assistant (Auto)  
**Versiyon:** 2.0

