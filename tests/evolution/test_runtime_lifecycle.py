"""Compiler, checkpoint and concurrent request lifecycle regression probes."""
import asyncio
import copy
from types import SimpleNamespace

import pytest
import torch

from model_management.compilation import configure_compilation
from model_management.config_schema import tiny_model_config
from model_management.model_manager import ModelManager
from cognitive_management.cognitive_types import CognitiveInput, CognitiveOutput, CognitiveState
from cognitive_management.v2.core.orchestrator import CognitiveOrchestrator
from cognitive_management.v2.middleware.base import BaseMiddleware
from cognitive_management.v2.middleware.tracing import TracingMiddleware
from cognitive_management.v2.middleware.async_middleware import SyncToAsyncMiddlewareAdapter
from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
from cognitive_management.config import CognitiveManagerConfig


def test_compiler_setup_failure_preserves_model_and_keys(monkeypatch):
    model = torch.nn.Linear(3, 2)
    keys = list(model.state_dict())
    def fail(*args, **kwargs):
        raise RuntimeError("compiler unavailable")
    monkeypatch.setattr(torch, "compile", fail)
    assert configure_compilation(model, {"torch_compile": True}) is model
    assert list(model.state_dict()) == keys
    assert not model.compilation_status["active"]
    assert "compiler unavailable" in model.compilation_status["fallback_reason"]
    assert model(torch.zeros(1, 3)).shape == (1, 2)


def test_lazy_compiler_failure_returns_to_original_forward(monkeypatch):
    from torch._dynamo.exc import Unsupported
    model = torch.nn.Linear(3, 2)
    x = torch.ones(1, 3)
    expected = model(x)
    calls = []
    def fake_compile(original, **kwargs):
        def compiled(*args, **kwargs):
            calls.append(1)
            raise Unsupported("test backend failure")
        return compiled
    monkeypatch.setattr(torch, "compile", fake_compile)
    configure_compilation(model, {"torch_compile": True})
    for _ in range(2):
        torch.testing.assert_close(model(x), expected)
    assert calls == [1]
    assert not model.compilation_status["active"]


def test_compile_does_not_hide_model_errors(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda forward, **kwargs: forward)
    model = configure_compilation(torch.nn.Linear(3, 2), {"torch_compile": True})
    with pytest.raises(RuntimeError, match="shapes"):
        model(torch.zeros(1, 5))
    assert model.compilation_status["active"]


@pytest.mark.parametrize("format", ["raw", "versioned", "trusted_module"])
def test_manager_checkpoint_formats_roundtrip(tmp_path, format):
    cfg = tiny_model_config(num_layers=1)
    first = ModelManager(cfg).initialize()
    first.eval_mode()
    x = torch.tensor([[2, 3, 4]])
    expected = first.forward(x, inference=True)[0]
    path = tmp_path / "model.pth"
    if format == "versioned":
        first.save(str(path), epoch=7)
        saved = torch.load(path, weights_only=True)
        assert saved["additional_info"]["config"]["config_version"] == 1
    else:
        torch.save(first.model if format == "trusted_module" else first.model.state_dict(), path)
    restored = ModelManager(copy.deepcopy(cfg)).initialize()
    restored.load(str(path), weights_only=False if format == "trusted_module" else True)
    restored.eval_mode()
    torch.testing.assert_close(restored.forward(x, inference=True)[0], expected)
    if format == "trusted_module":
        assert restored.optimizer is None and restored.scheduler is None


def test_async_adapters_execute_each_hook_once_and_keep_task_traces():
    class Count(BaseMiddleware):
        def __init__(self):
            super().__init__("count")
            self.before_count = self.after_count = 0
        def _before(self, state, request):
            self.before_count += 1
            return state, request
        def _after(self, state, request, response):
            self.after_count += 1
            return response
    tracing, count = TracingMiddleware(), Count()
    tracing.set_next(count)
    chain = SyncToAsyncMiddlewareAdapter(tracing)
    chain.set_next(SyncToAsyncMiddlewareAdapter(count))
    async def run(text):
        state, request = CognitiveState(), CognitiveInput(text)
        await chain.before_async(state, request)
        await asyncio.sleep(0)
        response = await chain.after_async(state, request, CognitiveOutput(text=text, used_mode="direct"))
        assert response.metadata["_trace_id"] == request.metadata["_trace_id"]
        return response.metadata["_span_id"]
    async def both():
        return await asyncio.gather(run("one"), run("two"))
    a, b = asyncio.run(both())
    assert a != b
    assert count.before_count == count.after_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_cached_response_updates_history_and_runs_after_hooks(asynchronous):
    response = CognitiveOutput(text="cached", used_mode="direct")
    class Hit(BaseMiddleware):
        def __init__(self):
            super().__init__("hit")
            self.finished = False
        def _before(self, state, request):
            request.metadata.update(_cache_hit=True, _cached_response=response)
            return state, request
        def _after(self, state, request, output):
            self.finished = True
            return output
    owner = object.__new__(CognitiveOrchestrator)
    cfg = CognitiveManagerConfig()
    cfg.memory.enable_vector_memory = cfg.memory.enable_rag = False
    owner.memory_service = MemoryServiceV2(cfg)
    owner.policy_router = SimpleNamespace(cfg=cfg)
    owner.middleware_chain = Hit()
    owner.performance_monitor = None
    owner._async_pipeline = SimpleNamespace()
    owner._async_middleware_chain = None
    state = CognitiveState()
    request = CognitiveInput("hello")
    output = asyncio.run(owner.handle_async(state, request)) if asynchronous else owner.handle(state, request)
    assert output.text == "cached"
    assert [turn["content"] for turn in state.history] == ["hello", "cached"]
    assert state.step == state.turn_count == 1
    assert owner.middleware_chain.finished
