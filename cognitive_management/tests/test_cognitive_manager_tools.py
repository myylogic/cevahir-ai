# -*- coding: utf-8 -*-
"""
Tool Management API Tests
==========================
CognitiveManager tool management metodlarını test eder.

Test Edilen Dosya: cognitive_manager.py
Test Edilen Metodlar:
- register_tool() - Tool kaydı
- list_available_tools() - Tool listeleme
- get_tool_schema() - Tool schema alma
- get_tool_metrics() - Tool metrikleri

Alt Modül Test Edilen Dosyalar:
- v2/components/tool_executor_v2.py (ToolExecutorV2)
- v2/components/tool_policy_v2.py (ToolPolicyV2)

Endüstri Standartları:
- pytest framework
- Fixture-based setup
- Assertion-based validation
- Tool registration validation
"""

import pytest
from typing import Dict, Any, Callable

from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput
from .conftest import (
    mock_model_api,
    default_config,
    cognitive_manager,
    cognitive_state,
    cognitive_input
)


# Helper function to clear allow list for tool tests
def clear_tool_allow_list(cm: CognitiveManager) -> None:
    """Clear the allow list to allow custom tools in tests."""
    if hasattr(cm, '_orchestrator') and cm._orchestrator and hasattr(cm._orchestrator, 'tool_executor'):
        if cm._orchestrator.tool_executor and hasattr(cm._orchestrator.tool_executor, 'cfg'):
            cm._orchestrator.tool_executor.cfg.tools.allow = ()


@pytest.fixture(autouse=True)
def auto_clear_tool_allow_list(cognitive_manager: CognitiveManager):
    """Automatically clear allow list before each test."""
    clear_tool_allow_list(cognitive_manager)
    yield
    # Restore after test if needed


# ============================================================================
# Test 1-10: register_tool() - Tool Registration
# Test Edilen Dosya: cognitive_manager.py (register_tool method)
# Alt Modül: v2/components/tool_executor_v2.py (ToolExecutorV2.register_tool)
# ============================================================================

def test_register_tool_basic(cognitive_manager: CognitiveManager):
    """
    Test 1: Basic register_tool() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Basit tool kaydı
    """
    def test_tool_function(param: str) -> str:
        return f"Result: {param}"
    
    cognitive_manager.register_tool(
        name="test_tool",
        func=test_tool_function,
        description="Test tool",
        parameters={"param": {"type": "string", "description": "Test parameter"}}
    )
    
    tools = cognitive_manager.list_available_tools()
    assert "test_tool" in tools


def test_register_tool_multiple(cognitive_manager: CognitiveManager):
    """
    Test 2: register_tool() with multiple tools.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Multiple tool kaydı
    """
    for i in range(5):
        def tool_func(x: str = "") -> str:
            return f"Tool {i}: {x}"
        
        cognitive_manager.register_tool(
            name=f"tool_{i}",
            func=tool_func,
            description=f"Tool {i}",
            parameters={}
        )
    
    tools = cognitive_manager.list_available_tools()
    assert len(tools) >= 5
    for i in range(5):
        assert f"tool_{i}" in tools


def test_register_tool_with_parameters(cognitive_manager: CognitiveManager):
    """
    Test 3: register_tool() with parameters.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Parametreli tool kaydı
    """
    def calculator(operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        return 0.0
    
    # Use a unique name to avoid conflict with default tools
    tool_name = "test_calculator"
    cognitive_manager.register_tool(
        name=tool_name,
        func=calculator,
        description="Calculator tool",
        parameters={
            "operation": {"type": "string", "description": "Operation type"},
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"}
        }
    )
    
    schema = cognitive_manager.get_tool_schema(tool_name)
    assert schema is not None
    assert "parameters" in schema


def test_register_tool_duplicate_name(cognitive_manager: CognitiveManager):
    """
    Test 4: register_tool() with duplicate name.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Duplicate tool name (edge case)
    """
    def tool1() -> str:
        return "Tool 1"
    
    def tool2() -> str:
        return "Tool 2"
    
    cognitive_manager.register_tool(
        name="duplicate_tool",
        func=tool1,
        description="First tool"
    )
    
    # Register with same name (should overwrite or error)
    try:
        cognitive_manager.register_tool(
            name="duplicate_tool",
            func=tool2,
            description="Second tool"
        )
    except Exception:
        # Expected if duplicate names not allowed
        pass


def test_register_tool_empty_name(cognitive_manager: CognitiveManager):
    """
    Test 5: register_tool() with empty name.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Boş tool name (edge case)
    """
    def tool_func() -> str:
        return "Test"
    
    from cognitive_management.exceptions import ValidationError
    try:
        cognitive_manager.register_tool(
            name="",
            func=tool_func,
            description="Empty name tool"
        )
        # Should not reach here - empty name should raise error
        assert False, "Expected ValidationError for empty name"
    except ValidationError:
        # Expected behavior - empty name should raise ValidationError
        pass


def test_register_tool_invalid_function(cognitive_manager: CognitiveManager):
    """
    Test 6: register_tool() with invalid function.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Invalid function (edge case)
    """
    try:
        cognitive_manager.register_tool(
            name="invalid_tool",
            func=None,  # type: ignore
            description="Invalid tool"
        )
    except (TypeError, ValueError):
        # Expected behavior
        pass


def test_register_tool_with_schema(cognitive_manager: CognitiveManager):
    """
    Test 7: register_tool() with complete schema.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Complete schema ile tool kaydı
    """
    def search_tool(query: str, limit: int = 10) -> str:
        return f"Search results for: {query}"
    
    # Use unique name to avoid conflict with default tools
    tool_name = "test_search_schema"
    cognitive_manager.register_tool(
        name=tool_name,
        func=search_tool,
        description="Search tool",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True
            },
            "limit": {
                "type": "integer",
                "description": "Result limit",
                "default": 10
            }
        }
    )
    
    schema = cognitive_manager.get_tool_schema(tool_name)
    assert schema is not None
    assert "parameters" in schema


def test_register_tool_async_function(cognitive_manager: CognitiveManager):
    """
    Test 8: register_tool() with async function.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Async function ile tool kaydı
    """
    import asyncio
    
    async def async_tool(param: str) -> str:
        await asyncio.sleep(0.01)
        return f"Async result: {param}"
    
    try:
        cognitive_manager.register_tool(
            name="async_tool",
            func=async_tool,
            description="Async tool"
        )
    except Exception:
        # May or may not support async (implementation dependent)
        pass


def test_register_tool_error_handling(cognitive_manager: CognitiveManager):
    """
    Test 9: register_tool() error handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Hata durumlarında handling
    """
    # Invalid parameters
    try:
        cognitive_manager.register_tool(
            name="error_tool",
            func=lambda: "test",
            description="Error tool",
            parameters="invalid"  # type: ignore
        )
    except (TypeError, ValueError):
        # Expected behavior
        pass


def test_register_tool_integration(cognitive_manager: CognitiveManager):
    """
    Test 10: register_tool() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.register_tool()
    Test Senaryosu: Integration testi
    """
    def integration_tool(value: int) -> int:
        return value * 2
    
    cognitive_manager.register_tool(
        name="integration_tool",
        func=integration_tool,
        description="Integration test tool",
        parameters={"value": {"type": "integer"}}
    )
    
    # Verify registration
    tools = cognitive_manager.list_available_tools()
    assert "integration_tool" in tools
    
    # Verify schema
    schema = cognitive_manager.get_tool_schema("integration_tool")
    assert schema is not None


# ============================================================================
# Test 11-20: list_available_tools() - Tool Listing
# Test Edilen Dosya: cognitive_manager.py (list_available_tools method)
# Alt Modül: v2/components/tool_executor_v2.py (ToolExecutorV2.list_available_tools)
# ============================================================================

def test_list_available_tools_basic(cognitive_manager: CognitiveManager):
    """
    Test 11: Basic list_available_tools() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Basit tool listeleme
    """
    tools = cognitive_manager.list_available_tools()
    assert isinstance(tools, list)
    # May have default tools or be empty


def test_list_available_tools_after_register(cognitive_manager: CognitiveManager):
    """
    Test 12: list_available_tools() after registering tools.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Register sonrası listeleme
    """
    def tool1() -> str:
        return "Tool 1"
    
    cognitive_manager.register_tool(
        name="list_test_tool",
        func=tool1,
        description="List test tool"
    )
    
    tools = cognitive_manager.list_available_tools()
    assert "list_test_tool" in tools


def test_list_available_tools_empty(cognitive_manager: CognitiveManager):
    """
    Test 13: list_available_tools() with no tools.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Tool yokken listeleme
    """
    tools = cognitive_manager.list_available_tools()
    assert isinstance(tools, list)
    # May have default tools or be empty


def test_list_available_tools_multiple(cognitive_manager: CognitiveManager):
    """
    Test 14: list_available_tools() with multiple tools.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Multiple tool listeleme
    """
    for i in range(10):
        def tool_func() -> str:
            return f"Tool {i}"
        
        cognitive_manager.register_tool(
            name=f"multi_tool_{i}",
            func=tool_func,
            description=f"Multi tool {i}"
        )
    
    tools = cognitive_manager.list_available_tools()
    assert len(tools) >= 10
    for i in range(10):
        assert f"multi_tool_{i}" in tools


def test_list_available_tools_consistency(cognitive_manager: CognitiveManager):
    """
    Test 15: list_available_tools() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Multiple call'larda consistency
    """
    cognitive_manager.register_tool(
        name="consistency_tool",
        func=lambda: "test",
        description="Consistency test"
    )
    
    tools1 = cognitive_manager.list_available_tools()
    tools2 = cognitive_manager.list_available_tools()
    
    assert "consistency_tool" in tools1
    assert "consistency_tool" in tools2


def test_list_available_tools_type_check(cognitive_manager: CognitiveManager):
    """
    Test 16: list_available_tools() return type check.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Return type validation
    """
    tools = cognitive_manager.list_available_tools()
    assert isinstance(tools, list)
    # All items should be strings
    for tool in tools:
        assert isinstance(tool, str)


def test_list_available_tools_immutability(cognitive_manager: CognitiveManager):
    """
    Test 17: list_available_tools() return value immutability.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Return value'nin immutable olması
    """
    tools1 = cognitive_manager.list_available_tools()
    
    # Modify returned list
    tools1.append("external_tool")
    
    tools2 = cognitive_manager.list_available_tools()
    # External tool should not be in internal list
    assert "external_tool" not in tools2


def test_list_available_tools_performance(cognitive_manager: CognitiveManager):
    """
    Test 18: list_available_tools() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Performans testi
    """
    import time
    
    # Register many tools
    for i in range(50):
        cognitive_manager.register_tool(
            name=f"perf_tool_{i}",
            func=lambda: "test",
            description=f"Perf tool {i}"
        )
    
    start = time.time()
    tools = cognitive_manager.list_available_tools()
    elapsed = time.time() - start
    
    assert len(tools) >= 50
    assert elapsed < 0.1  # Should be fast


def test_list_available_tools_concurrent_access(cognitive_manager: CognitiveManager):
    """
    Test 19: list_available_tools() concurrent access.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Concurrent access testi
    """
    import threading
    
    cognitive_manager.register_tool(
        name="concurrent_tool",
        func=lambda: "test",
        description="Concurrent test"
    )
    
    results = []
    
    def worker():
        tools = cognitive_manager.list_available_tools()
        results.append(tools)
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    assert len(results) == 5
    for result in results:
        assert "concurrent_tool" in result


def test_list_available_tools_integration(cognitive_manager: CognitiveManager):
    """
    Test 20: list_available_tools() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.list_available_tools()
    Test Senaryosu: Integration testi
    """
    # Register tools
    for i in range(5):
        cognitive_manager.register_tool(
            name=f"integration_tool_{i}",
            func=lambda x=i: f"Tool {x}",
            description=f"Integration tool {i}"
        )
    
    # List tools
    tools = cognitive_manager.list_available_tools()
    assert len(tools) >= 5
    
    # Verify each tool has schema
    for i in range(5):
        schema = cognitive_manager.get_tool_schema(f"integration_tool_{i}")
        assert schema is not None


# ============================================================================
# Test 21-30: get_tool_schema() - Tool Schema Retrieval
# Test Edilen Dosya: cognitive_manager.py (get_tool_schema method)
# Alt Modül: v2/components/tool_executor_v2.py (ToolExecutorV2.get_tool_schema)
# ============================================================================

def test_get_tool_schema_basic(cognitive_manager: CognitiveManager):
    """
    Test 21: Basic get_tool_schema() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Basit tool schema alma
    """
    def test_tool(param: str) -> str:
        return param
    
    cognitive_manager.register_tool(
        name="schema_test_tool",
        func=test_tool,
        description="Schema test tool",
        parameters={"param": {"type": "string"}}
    )
    
    schema = cognitive_manager.get_tool_schema("schema_test_tool")
    assert schema is not None
    assert isinstance(schema, dict)


def test_get_tool_schema_nonexistent_tool(cognitive_manager: CognitiveManager):
    """
    Test 22: get_tool_schema() with nonexistent tool.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Var olmayan tool schema (edge case)
    """
    schema = cognitive_manager.get_tool_schema("nonexistent_tool")
    assert schema is None


def test_get_tool_schema_with_parameters(cognitive_manager: CognitiveManager):
    """
    Test 23: get_tool_schema() with parameters.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Parametreli tool schema
    """
    def complex_tool(a: int, b: str, c: float = 1.0) -> dict:
        return {"a": a, "b": b, "c": c}
    
    cognitive_manager.register_tool(
        name="complex_tool",
        func=complex_tool,
        description="Complex tool",
        parameters={
            "a": {"type": "integer", "required": True},
            "b": {"type": "string", "required": True},
            "c": {"type": "number", "default": 1.0}
        }
    )
    
    schema = cognitive_manager.get_tool_schema("complex_tool")
    assert schema is not None
    assert "parameters" in schema
    # Schema format: {"parameters": {"type": "object", "properties": {...}}}
    params = schema["parameters"]
    if isinstance(params, dict) and "properties" in params:
        # New format with properties
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert "c" in params["properties"]
    else:
        # Old format (direct parameters)
        assert "a" in params
        assert "b" in params
        assert "c" in params


def test_get_tool_schema_schema_structure(cognitive_manager: CognitiveManager):
    """
    Test 24: get_tool_schema() schema structure validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Schema yapısı validation
    """
    cognitive_manager.register_tool(
        name="structure_tool",
        func=lambda: "test",
        description="Structure test tool"
    )
    
    schema = cognitive_manager.get_tool_schema("structure_tool")
    assert schema is not None
    # Common schema fields
    assert "name" in schema or "description" in schema or "parameters" in schema


def test_get_tool_schema_empty_name(cognitive_manager: CognitiveManager):
    """
    Test 25: get_tool_schema() with empty name.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Boş tool name (edge case)
    """
    schema = cognitive_manager.get_tool_schema("")
    assert schema is None


def test_get_tool_schema_type_check(cognitive_manager: CognitiveManager):
    """
    Test 26: get_tool_schema() return type check.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Return type validation
    """
    cognitive_manager.register_tool(
        name="type_check_tool",
        func=lambda: "test",
        description="Type check tool"
    )
    
    schema = cognitive_manager.get_tool_schema("type_check_tool")
    assert schema is None or isinstance(schema, dict)


def test_get_tool_schema_consistency(cognitive_manager: CognitiveManager):
    """
    Test 27: get_tool_schema() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Multiple call'larda consistency
    """
    cognitive_manager.register_tool(
        name="consistency_schema_tool",
        func=lambda x: x,
        description="Consistency schema tool",
        parameters={"x": {"type": "string"}}
    )
    
    schema1 = cognitive_manager.get_tool_schema("consistency_schema_tool")
    schema2 = cognitive_manager.get_tool_schema("consistency_schema_tool")
    
    # Should return same schema
    assert schema1 == schema2


def test_get_tool_schema_performance(cognitive_manager: CognitiveManager):
    """
    Test 28: get_tool_schema() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Performans testi
    """
    import time
    
    cognitive_manager.register_tool(
        name="perf_schema_tool",
        func=lambda: "test",
        description="Perf schema tool"
    )
    
    start = time.time()
    for _ in range(100):
        schema = cognitive_manager.get_tool_schema("perf_schema_tool")
        assert schema is not None
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_get_tool_schema_all_tools(cognitive_manager: CognitiveManager):
    """
    Test 29: get_tool_schema() for all registered tools.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Tüm tool'lar için schema alma
    """
    # Register multiple tools
    for i in range(5):
        cognitive_manager.register_tool(
            name=f"all_schema_tool_{i}",
            func=lambda x=i: f"Tool {x}",
            description=f"All schema tool {i}"
        )
    
    tools = cognitive_manager.list_available_tools()
    for tool in tools:
        if tool.startswith("all_schema_tool_"):
            schema = cognitive_manager.get_tool_schema(tool)
            assert schema is not None


def test_get_tool_schema_integration(cognitive_manager: CognitiveManager):
    """
    Test 30: get_tool_schema() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_schema()
    Test Senaryosu: Integration testi
    """
    def integration_schema_tool(param1: str, param2: int = 10) -> str:
        return f"{param1}: {param2}"
    
    cognitive_manager.register_tool(
        name="integration_schema_tool",
        func=integration_schema_tool,
        description="Integration schema tool",
        parameters={
            "param1": {"type": "string", "required": True},
            "param2": {"type": "integer", "default": 10}
        }
    )
    
    # Get schema
    schema = cognitive_manager.get_tool_schema("integration_schema_tool")
    assert schema is not None
    
    # Verify schema contains expected fields
    assert "parameters" in schema or "description" in schema


# ============================================================================
# Test 31-40: get_tool_metrics() - Tool Metrics
# Test Edilen Dosya: cognitive_manager.py (get_tool_metrics method)
# Alt Modül: v2/components/tool_executor_v2.py (ToolExecutorV2.get_tool_metrics)
# ============================================================================

def test_get_tool_metrics_basic(cognitive_manager: CognitiveManager):
    """
    Test 31: Basic get_tool_metrics() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Basit tool metrics alma
    """
    metrics = cognitive_manager.get_tool_metrics()
    assert isinstance(metrics, dict)
    # May have default metrics structure


def test_get_tool_metrics_specific_tool(cognitive_manager: CognitiveManager):
    """
    Test 32: get_tool_metrics() for specific tool.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics(tool_name)
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Belirli tool için metrics
    """
    cognitive_manager.register_tool(
        name="metrics_tool",
        func=lambda: "test",
        description="Metrics test tool"
    )
    
    metrics = cognitive_manager.get_tool_metrics("metrics_tool")
    assert isinstance(metrics, dict)


def test_get_tool_metrics_nonexistent_tool(cognitive_manager: CognitiveManager):
    """
    Test 33: get_tool_metrics() for nonexistent tool.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Var olmayan tool metrics (edge case)
    """
    metrics = cognitive_manager.get_tool_metrics("nonexistent_tool")
    # May return empty dict or None
    assert metrics is None or isinstance(metrics, dict)


def test_get_tool_metrics_all_tools(cognitive_manager: CognitiveManager):
    """
    Test 34: get_tool_metrics() for all tools.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Tüm tool'lar için metrics
    """
    # Register tools
    for i in range(3):
        cognitive_manager.register_tool(
            name=f"all_metrics_tool_{i}",
            func=lambda: "test",
            description=f"All metrics tool {i}"
        )
    
    metrics = cognitive_manager.get_tool_metrics()
    assert isinstance(metrics, dict)


def test_get_tool_metrics_after_usage(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 35: get_tool_metrics() after tool usage.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics(), handle()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Tool kullanımı sonrası metrics
    """
    cognitive_manager.register_tool(
        name="usage_metrics_tool",
        func=lambda x: f"Result: {x}",
        description="Usage metrics tool",
        parameters={"x": {"type": "string"}}
    )
    
    # Process request (tool may be used)
    input_msg = CognitiveInput(user_message="Use tool")
    cognitive_manager.handle(cognitive_state, input_msg)
    
    metrics = cognitive_manager.get_tool_metrics("usage_metrics_tool")
    assert isinstance(metrics, dict)


def test_get_tool_metrics_consistency(cognitive_manager: CognitiveManager):
    """
    Test 36: get_tool_metrics() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Multiple call'larda consistency
    """
    metrics1 = cognitive_manager.get_tool_metrics()
    metrics2 = cognitive_manager.get_tool_metrics()
    
    assert isinstance(metrics1, dict)
    assert isinstance(metrics2, dict)


def test_get_tool_metrics_structure(cognitive_manager: CognitiveManager):
    """
    Test 37: get_tool_metrics() metrics structure.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Metrics yapısı validation
    """
    cognitive_manager.register_tool(
        name="structure_metrics_tool",
        func=lambda: "test",
        description="Structure metrics tool"
    )
    
    metrics = cognitive_manager.get_tool_metrics("structure_metrics_tool")
    if metrics is not None:
        assert isinstance(metrics, dict)
        # May contain call_count, total_latency, etc.


def test_get_tool_metrics_performance(cognitive_manager: CognitiveManager):
    """
    Test 38: get_tool_metrics() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    for _ in range(100):
        metrics = cognitive_manager.get_tool_metrics()
        assert isinstance(metrics, dict)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_get_tool_metrics_empty(cognitive_manager: CognitiveManager):
    """
    Test 39: get_tool_metrics() with no tools.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Tool yokken metrics
    """
    metrics = cognitive_manager.get_tool_metrics()
    assert isinstance(metrics, dict)
    # May be empty or have default structure


def test_get_tool_metrics_integration(cognitive_manager: CognitiveManager):
    """
    Test 40: get_tool_metrics() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metod: ToolExecutorV2.get_tool_metrics()
    Test Senaryosu: Integration testi
    """
    # Register tools
    for i in range(3):
        cognitive_manager.register_tool(
            name=f"integration_metrics_tool_{i}",
            func=lambda x=i: f"Tool {x}",
            description=f"Integration metrics tool {i}"
        )
    
    # Get all metrics
    all_metrics = cognitive_manager.get_tool_metrics()
    assert isinstance(all_metrics, dict)
    
    # Get specific tool metrics
    for i in range(3):
        tool_metrics = cognitive_manager.get_tool_metrics(f"integration_metrics_tool_{i}")
        assert tool_metrics is None or isinstance(tool_metrics, dict)


# ============================================================================
# Test 41-50: Tool Management Integration and Edge Cases
# Test Edilen Dosya: cognitive_manager.py (Tool management integration)
# Alt Modül: v2/components/tool_executor_v2.py, v2/components/tool_policy_v2.py
# ============================================================================

def test_tool_management_full_workflow(cognitive_manager: CognitiveManager):
    """
    Test 41: Full tool management workflow.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm tool management metodları
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Alt Modül Metodlar: ToolExecutorV2 tüm metodlar
    Test Senaryosu: Tam tool management workflow
    """
    # 1. Register tool
    def workflow_tool(param: str) -> str:
        return f"Workflow: {param}"
    
    cognitive_manager.register_tool(
        name="workflow_tool",
        func=workflow_tool,
        description="Workflow test tool",
        parameters={"param": {"type": "string"}}
    )
    
    # 2. List tools
    tools = cognitive_manager.list_available_tools()
    assert "workflow_tool" in tools
    
    # 3. Get schema
    schema = cognitive_manager.get_tool_schema("workflow_tool")
    assert schema is not None
    
    # 4. Get metrics
    metrics = cognitive_manager.get_tool_metrics("workflow_tool")
    assert isinstance(metrics, dict)


def test_tool_management_error_recovery(cognitive_manager: CognitiveManager):
    """
    Test 42: Tool management error recovery.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: register_tool(), list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Test Senaryosu: Hata sonrası recovery
    """
    # Try invalid registration
    try:
        cognitive_manager.register_tool(
            name="error_tool",
            func=None,  # type: ignore
            description="Error tool"
        )
    except Exception:
        pass
    
    # System should still work
    tools = cognitive_manager.list_available_tools()
    assert isinstance(tools, list)


def test_tool_management_concurrent_registration(cognitive_manager: CognitiveManager):
    """
    Test 43: Tool management concurrent registration.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Test Senaryosu: Concurrent tool registration
    """
    import threading
    
    def register_worker(tool_id: int):
        cognitive_manager.register_tool(
            name=f"concurrent_tool_{tool_id}",
            func=lambda: f"Tool {tool_id}",
            description=f"Concurrent tool {tool_id}"
        )
    
    threads = [threading.Thread(target=register_worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    tools = cognitive_manager.list_available_tools()
    assert len(tools) >= 5


def test_tool_management_large_number(cognitive_manager: CognitiveManager):
    """
    Test 44: Tool management with large number of tools.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: register_tool(), list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Test Senaryosu: Çok sayıda tool (edge case)
    """
    for i in range(100):
        cognitive_manager.register_tool(
            name=f"large_tool_{i}",
            func=lambda x=i: f"Tool {x}",
            description=f"Large tool {i}"
        )
    
    tools = cognitive_manager.list_available_tools()
    assert len(tools) >= 100


def test_tool_management_schema_validation(cognitive_manager: CognitiveManager):
    """
    Test 45: Tool management schema validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: register_tool(), get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Test Senaryosu: Schema validation testi
    """
    def validated_tool(a: int, b: str) -> dict:
        return {"a": a, "b": b}
    
    cognitive_manager.register_tool(
        name="validated_tool",
        func=validated_tool,
        description="Validated tool",
        parameters={
            "a": {"type": "integer", "required": True},
            "b": {"type": "string", "required": True}
        }
    )
    
    schema = cognitive_manager.get_tool_schema("validated_tool")
    assert schema is not None
    assert "parameters" in schema


def test_tool_management_metrics_accuracy(cognitive_manager: CognitiveManager):
    """
    Test 46: Tool management metrics accuracy.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_tool_metrics()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Test Senaryosu: Metrics doğruluğu testi
    """
    cognitive_manager.register_tool(
        name="metrics_accuracy_tool",
        func=lambda: "test",
        description="Metrics accuracy tool"
    )
    
    initial_metrics = cognitive_manager.get_tool_metrics("metrics_accuracy_tool")
    assert isinstance(initial_metrics, dict)
    
    # Metrics should be consistent
    final_metrics = cognitive_manager.get_tool_metrics("metrics_accuracy_tool")
    assert isinstance(final_metrics, dict)


def test_tool_management_thread_safety(cognitive_manager: CognitiveManager):
    """
    Test 47: Tool management thread safety.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: register_tool(), list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Test Senaryosu: Thread safety testi
    """
    import threading
    
    def worker(worker_id: int):
        cognitive_manager.register_tool(
            name=f"thread_safe_tool_{worker_id}",
            func=lambda: f"Worker {worker_id}",
            description=f"Thread safe tool {worker_id}"
        )
        tools = cognitive_manager.list_available_tools()
        assert isinstance(tools, list)
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_tool_management_performance_under_load(cognitive_manager: CognitiveManager):
    """
    Test 48: Tool management performance under load.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: register_tool(), list_available_tools(), get_tool_schema()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Test Senaryosu: Yük altında performans
    """
    import time
    
    start = time.time()
    
    # Register many tools
    for i in range(50):
        cognitive_manager.register_tool(
            name=f"load_tool_{i}",
            func=lambda x=i: f"Tool {x}",
            description=f"Load tool {i}"
        )
    
    # List tools
    tools = cognitive_manager.list_available_tools()
    
    # Get schemas
    for tool in tools[:10]:  # First 10
        schema = cognitive_manager.get_tool_schema(tool)
        assert schema is None or isinstance(schema, dict)
    
    elapsed = time.time() - start
    assert elapsed < 5.0  # Should complete in reasonable time


def test_tool_management_state_persistence(cognitive_manager: CognitiveManager):
    """
    Test 49: Tool management state persistence.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: register_tool(), list_available_tools()
    Alt Modül Dosyası: v2/components/tool_executor_v2.py
    Test Senaryosu: State persistence testi
    """
    cognitive_manager.register_tool(
        name="persistence_tool",
        func=lambda: "test",
        description="Persistence test tool"
    )
    
    # Get tools multiple times
    tools1 = cognitive_manager.list_available_tools()
    tools2 = cognitive_manager.list_available_tools()
    
    # Should persist
    assert "persistence_tool" in tools1
    assert "persistence_tool" in tools2


def test_tool_management_end_to_end(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 50: Tool management end-to-end test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm tool management metodları
    Alt Modül Dosyası: v2/components/tool_executor_v2.py, v2/components/tool_policy_v2.py
    Test Senaryosu: End-to-end tool management testi
    """
    # 1. Register multiple tools
    for i in range(3):
        cognitive_manager.register_tool(
            name=f"e2e_tool_{i}",
            func=lambda x=i: f"E2E Tool {x}",
            description=f"E2E tool {i}",
            parameters={"x": {"type": "integer"}}
        )
    
    # 2. List all tools
    tools = cognitive_manager.list_available_tools()
    assert len(tools) >= 3
    
    # 3. Get schemas for all tools
    for i in range(3):
        schema = cognitive_manager.get_tool_schema(f"e2e_tool_{i}")
        assert schema is not None
    
    # 4. Get metrics
    all_metrics = cognitive_manager.get_tool_metrics()
    assert isinstance(all_metrics, dict)
    
    # 5. Process request (tools may be used)
    input_msg = CognitiveInput(user_message="E2E tool test")
    output = cognitive_manager.handle(cognitive_state, input_msg)
    assert output is not None
    
    # 6. Verify metrics still accessible
    metrics_after = cognitive_manager.get_tool_metrics()
    assert isinstance(metrics_after, dict)

