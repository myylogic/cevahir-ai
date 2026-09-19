"""Small deterministic probes of the existing cognitive runtime contracts."""
import asyncio
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.cognitive_types import CognitiveInput, CognitiveOutput, CognitiveState
from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
from cognitive_management.v2.components.tool_executor_v2 import ToolExecutorV2
from cognitive_management.v2.components.tool_policy_v2 import ToolPolicyV2
from cognitive_management.v2.processing.handlers import ContextBuildingHandler
from cognitive_management.v2.processing.pipeline import ProcessingContext
from cognitive_management.v2.middleware.cache import CacheMiddleware
from cognitive_management.v2.middleware.tracing import TracingMiddleware
from cognitive_management.v2.utils.request_scope import memory_scope


def config():
    cfg = CognitiveManagerConfig()
    cfg.memory.enable_vector_memory = False
    cfg.memory.enable_rag = False
    return cfg


def test_memory_retrieval_is_scoped_and_identifiers_do_not_repeat():
    memory = MemoryServiceV2(config())
    ids = []
    for i in range(50):
        with memory_scope("alice"):
            memory.add_turn([], "user", f"private apple {i}")
            ids.append(memory._episodic_memory[-1]["id"])
    assert len(set(ids)) == 50
    with memory_scope("bob"):
        assert not memory.retrieve_context("apple")
        memory.add_turn([], "user", "private orange")
    with memory_scope("alice"):
        assert memory.retrieve_context("apple")
        assert not memory.retrieve_context("orange")


def test_tool_handler_calls_executor_and_adds_actual_result():
    cfg = config(); cfg.tools.enable_tools = True; cfg.tools.allow = ["calculator", "search", "file"]
    executor = ToolExecutorV2(cfg)
    assert executor.list_available_tools() == ["calculator"]
    policy = ToolPolicyV2(cfg, executor)
    context = ProcessingContext(CognitiveState(), CognitiveInput("2+2"))
    context.features = {"needs_calc_or_parse": True, "tool_decision": "must"}
    result = ContextBuildingHandler(MemoryServiceV2(cfg), policy)._process(context)
    assert result.tool_name == "calculator"
    assert "[ARAÇ SONUCU: calculator]\n4" in result.context_text
    assert executor.get_tool_metrics("calculator")["success_count"] == 1


def test_failed_tool_is_not_reported_as_used():
    cfg = config(); cfg.tools.enable_tools = True; cfg.tools.allow = ["calculator"]
    executor = ToolExecutorV2(cfg)
    context = ProcessingContext(CognitiveState(), CognitiveInput("2/0"))
    context.features = {"needs_calc_or_parse": True, "tool_decision": "must"}
    result = ContextBuildingHandler(MemoryServiceV2(cfg), ToolPolicyV2(cfg, executor))._process(context)
    assert result.tool_name is None
    assert "tool_error" in result.request.metadata
    with pytest.raises(Exception):
        executor.execute("calculator", {"operation": "2**100000000"})


def test_response_cache_keys_include_complete_history_identity_and_decoding():
    cache = CacheMiddleware()
    a = CognitiveState(session_id="a", history=[{"role": "user", "content": "secret"}] + [{"role": "assistant", "content": "same"}] * 3)
    b = CognitiveState(session_id="a", history=[{"role": "user", "content": "different"}] + a.history[1:])
    request = CognitiveInput("hello")
    key = cache._generate_response_key(a, request)
    assert cache._generate_response_key(b, request) != key
    b.history = a.history; b.session_id = "b"
    assert cache._generate_response_key(b, request) != key
    request.metadata["_runtime_context"] = {"decoding": {"temperature": 0}, "scope": "a"}
    assert cache._generate_response_key(a, request) != key


def test_cache_returns_independent_response_objects():
    cache = CacheMiddleware()
    state = CognitiveState()
    request = CognitiveInput("hello")
    cache._before(state, request)
    cache._after(state, request, CognitiveOutput(text="ok", used_mode="direct"))
    second = CognitiveInput("hello")
    cache._before(state, second)
    second.metadata["_cached_response"].text = "changed"
    third = CognitiveInput("hello")
    cache._before(state, third)
    assert third.metadata["_cached_response"].text == "ok"


def test_concurrent_traces_do_not_overwrite_each_other():
    tracing = TracingMiddleware()
    state = CognitiveState()
    first, second = CognitiveInput("one"), CognitiveInput("two")
    a, b = copy_context(), copy_context()
    a.run(tracing._before, state, first)
    b.run(tracing._before, state, second)
    out_a = a.run(tracing._after, state, first, CognitiveOutput(text="a", used_mode="direct"))
    out_b = b.run(tracing._after, state, second, CognitiveOutput(text="b", used_mode="direct"))
    assert out_a.metadata["_trace_id"] == first.metadata["_trace_id"]
    assert out_b.metadata["_trace_id"] == second.metadata["_trace_id"]
    assert out_a.metadata["_span_id"] != out_b.metadata["_span_id"]
    assert a.run(lambda: tracing._current_span) is None


def test_vector_scope_is_filtered_before_top_k():
    from cognitive_management.v2.components.vector_store.memory_vector_store import MemoryVectorStore
    cfg = config()
    cfg.memory.hybrid_search_alpha = 1.0
    cfg.memory.rag_score_threshold = 0.0
    memory = MemoryServiceV2(cfg)
    store = MemoryVectorStore(cfg, dimension=2)
    store.add(texts=["other user's exact match", "my related context"],
              embeddings=[[1., 0.], [.8, .2]],
              metadata=[{"scope": "other"}, {"scope": "mine"}], ids=["a", "b"])
    memory._vector_memory_enabled = True
    memory._vector_store = store
    memory._embedding_adapter = SimpleNamespace(encode_single=lambda text: [1., 0.])
    with memory_scope("mine"):
        results = memory.retrieve_context("query", top_k=1)
    assert len(results) == 1
    assert results[0]["content"] == "my related context"
