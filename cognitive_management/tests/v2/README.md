# V2 Test Suite

Comprehensive test suite for V2 Cognitive Management system.

## Structure

```
tests/v2/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_policy_router_v2.py
├── test_deliberation_engine_v2.py
├── test_critic_v2.py
├── test_memory_service_v2.py
├── test_orchestrator.py
├── test_integration.py
└── README.md
```

## Running Tests

### Run all tests
```bash
pytest cognitive_management/tests/v2/ -v
```

### Run specific test file
```bash
pytest cognitive_management/tests/v2/test_policy_router_v2.py -v
```

### Run with coverage
```bash
pytest cognitive_management/tests/v2/ --cov=cognitive_management.v2 --cov-report=html
```

## Test Coverage Goals

- **Unit Tests:** 80%+ coverage for all components
- **Integration Tests:** All major flows covered
- **Edge Cases:** Comprehensive edge case testing

## Fixtures

See `conftest.py` for available fixtures:
- `mock_model_api`: Mock model API
- `default_config`: Default configuration
- `config_with_think_mode`: Configuration with think mode enabled
- `empty_state`: Empty cognitive state
- `state_with_history`: State with conversation history
- `simple_input`: Simple cognitive input
- `complex_input`: Complex cognitive input
- `decoding_config`: Decoding configuration

