"""Request-local memory and response-cache scope for sync and async runtimes."""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import wraps
import hashlib
import json
from contextlib import nullcontext

_scope = ContextVar("cevahir_memory_scope", default="local")


def current_scope():
    return _scope.get()


@contextmanager
def memory_scope(scope):
    token = _scope.set(str(scope))
    try:
        yield
    finally:
        _scope.reset(token)


def _prepare(owner, state, request, kwargs):
    # Identity comes from the authenticated caller's state, not user prompt text.
    user = state.metadata.get("user_id", "local")
    session = state.session_id
    scope = json.dumps([str(user), str(session)], separators=(",", ":"))
    cfg = getattr(getattr(owner, "policy_router", None), "cfg", None)
    decoding = kwargs.get("decoding_config")
    for internal in ("_research_run", "_cache_hit", "_cached_response"):
        request.metadata.pop(internal, None)
    request.metadata["_runtime_context"] = {
        "scope": scope,
        "decoding": asdict(decoding) if is_dataclass(decoding) else str(decoding),
        "configuration": asdict(cfg) if is_dataclass(cfg) else str(cfg),
        "memory_revision": getattr(owner.memory_service, "revision", 0),
    }
    return scope


def scoped_request(func):
    @wraps(func)
    def wrapped(self, state, request, *args, **kwargs):
        with memory_scope(_prepare(self, state, request, kwargs)):
            research = getattr(self, "research", None)
            with research.session(state, request) if research else nullcontext() as runtime:
                response = func(self, state, request, *args, **kwargs)
                return research.finish(runtime, request, response) if research else response
    return wrapped


def scoped_async_request(func):
    @wraps(func)
    async def wrapped(self, state, request, *args, **kwargs):
        with memory_scope(_prepare(self, state, request, kwargs)):
            research = getattr(self, "research", None)
            with research.session(state, request) if research else nullcontext() as runtime:
                response = await func(self, state, request, *args, **kwargs)
                return research.finish(runtime, request, response) if research else response
    return wrapped
