"""Request-local accounting for bounded cognitive model calls.

This module imports no model runtime. Token costs are conservative reservations,
not measurements of generated tokens or FLOPs. Failed calls keep their costs.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from collections.abc import Mapping
from math import isfinite
from numbers import Real
from threading import RLock
from uuid import uuid4


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class BudgetLimits:
    max_generate_calls: int = 6
    max_score_calls: int = 12
    max_entropy_calls: int = 1
    max_output_tokens: int = 1024
    max_input_chars: int = 64000
    reserve_answer_tokens: int = 128
    reserve_answer_input_chars: int = 8192

    def __post_init__(self):
        for name, value in asdict(self).items():
            _integer(name, value)
        if self.reserve_answer_tokens > self.max_output_tokens:
            raise ValueError("reserve_answer_tokens cannot exceed max_output_tokens")
        if self.reserve_answer_input_chars > self.max_input_chars:
            raise ValueError("reserve_answer_input_chars cannot exceed max_input_chars")


class BudgetExhausted(Exception):
    """A call was denied before reaching the model; never retry without accounting."""

    def __init__(self, reason, *, stage=None, kind=None):
        self.reason = reason
        self.stage = stage
        self.kind = kind
        super().__init__(f"Research compute budget denied {kind or 'call'}: {reason}")


_runtime = ContextVar("cevahir_research_runtime", default=None)
_stage = ContextVar("cevahir_research_stage", default="other")


def current_runtime():
    return _runtime.get()


@contextmanager
def request_runtime(runtime):
    """Bind one runtime; nested requests restore both runtime and stage."""
    if runtime is not None and not isinstance(runtime, RequestRuntime):
        raise TypeError("runtime must be RequestRuntime or None")
    runtime_token = _runtime.set(runtime)
    stage_token = _stage.set("other")
    try:
        yield runtime
    finally:
        _stage.reset(stage_token)
        _runtime.reset(runtime_token)


@contextmanager
def runtime_stage(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("stage must be a nonempty string")
    token = _stage.set(name)
    try:
        yield current_runtime()
    finally:
        _stage.reset(token)


class RequestRuntime:
    """A shared, thread-safe ledger with context-local stage selection.

    An answer's call/token/input allowance is protected until its first attempt.
    Its reservation is consumed even when that attempt raises or is cancelled.
    A final prompt exceeding the protected input allowance is not guaranteed to
    fit; callers must still bound context size. Beam generation multiplies both
    reserved output tokens and input characters by beam count.
    ``plan`` and ``features`` are caller-owned request metadata; set them before
    dispatching concurrent calls, or use ``set_metadata`` under the ledger lock.
    """

    def __init__(self, scope, identity, limits, request_id=None):
        if not isinstance(limits, BudgetLimits):
            raise TypeError("limits must be BudgetLimits")
        self.scope = str(scope)
        self.identity = deepcopy(identity)
        self.limits = limits
        self.request_id = str(request_id) if request_id is not None else uuid4().hex
        self.plan = {}
        self.features = {}
        self._lock = RLock()
        self._calls = {"generate": 0, "score": 0, "entropy": 0}
        self._output_tokens = 0
        self._input_chars = 0
        self._denied = 0
        self._errors = 0
        self._stages = {}
        self._attempts = []
        self._answer_reserved = True
        self._cancelled = False
        self._cancel_reason = None
        self._last_answer = None

    @property
    def stage(self):
        return _stage.get() if current_runtime() is self else "other"

    @property
    def cancelled(self):
        with self._lock:
            return self._cancelled

    @property
    def last_answer(self):
        with self._lock:
            return self._last_answer

    def cancel(self, reason="cancelled"):
        with self._lock:
            self._cancelled = True
            self._cancel_reason = str(reason)

    def set_metadata(self, *, plan=None, features=None):
        with self._lock:
            if plan is not None:
                self.plan = deepcopy(plan)
            if features is not None:
                self.features = deepcopy(features)

    def _stage_usage(self, stage):
        return self._stages.setdefault(stage, {
            "generate_calls": 0, "score_calls": 0, "entropy_calls": 0,
            "reserved_output_tokens": 0, "input_chars": 0,
            "denied": 0, "errors": 0,
        })

    def _append_attempt(self, record):
        self._attempts.append(record)
        # Repeated denied calls must not turn budget enforcement into a memory leak.
        if len(self._attempts) > 256:
            del self._attempts[:-256]

    def _deny(self, reason, kind, stage):
        self._denied += 1
        self._stage_usage(stage)["denied"] += 1
        self._append_attempt({"kind": kind, "stage": stage, "status": "denied", "reason": reason})
        raise BudgetExhausted(reason, stage=stage, kind=kind)

    def _reserve(self, kind, input_chars, requested_tokens=0, num_beams=1):
        stage = self.stage
        with self._lock:
            if self._cancelled:
                self._deny("cancelled", kind, stage)
            protect_answer = self._answer_reserved and not (kind == "generate" and stage == "answer")
            input_limit = self.limits.max_input_chars
            if protect_answer:
                input_limit -= self.limits.reserve_answer_input_chars
            if self._input_chars + input_chars > input_limit:
                self._deny("input_chars_reserved_for_answer" if protect_answer else "input_chars", kind, stage)
            call_limit = getattr(self.limits, f"max_{kind}_calls")
            held_for_answer = kind == "generate" and self._answer_reserved and stage != "answer"
            if self._calls[kind] >= call_limit - int(held_for_answer):
                self._deny("generate_calls_reserved_for_answer" if held_for_answer else f"{kind}_calls", kind, stage)
            granted_tokens = 0
            if kind == "generate":
                available = self.limits.max_output_tokens - self._output_tokens
                if held_for_answer:
                    available -= self.limits.reserve_answer_tokens
                granted_tokens = min(requested_tokens, available // num_beams)
                if granted_tokens < 1:
                    self._deny("output_tokens_reserved_for_answer" if held_for_answer else "output_tokens", kind, stage)
            cost = granted_tokens * num_beams
            self._calls[kind] += 1
            self._input_chars += input_chars
            self._output_tokens += cost
            if kind == "generate" and stage == "answer":
                self._answer_reserved = False
            usage = self._stage_usage(stage)
            usage[f"{kind}_calls"] += 1
            usage["input_chars"] += input_chars
            usage["reserved_output_tokens"] += cost
            record = {
                "kind": kind, "stage": stage, "status": "reserved",
                "input_chars": input_chars, "reserved_output_tokens": cost,
                "max_new_tokens": granted_tokens, "num_beams": num_beams,
            }
            self._append_attempt(record)
            return record, granted_tokens

    def _finish(self, record, *, error=None, result=None):
        with self._lock:
            if error is not None:
                record["status"] = "error"
                record["error_type"] = type(error).__name__
                self._errors += 1
                self._stage_usage(record["stage"])["errors"] += 1
            else:
                record["status"] = "completed_after_cancel" if self._cancelled else "completed"
                if (not self._cancelled and record["kind"] == "generate"
                        and record["stage"] in {"answer", "refinement"}
                        and isinstance(result, str) and result.strip()):
                    self._last_answer = result

    def snapshot(self):
        with self._lock:
            return {
                "schema_version": 1,
                "request_id": self.request_id, "scope": self.scope,
                "identity": deepcopy(self.identity), "stage": self.stage,
                "limits": asdict(self.limits),
                "calls": dict(self._calls),
                "reserved_output_tokens": self._output_tokens,
                "token_accounting": "upper_bound_reservation_times_num_beams",
                "input_accounting": "characters_times_num_beams_for_generation",
                "input_chars": self._input_chars,
                "denied": self._denied, "errors": self._errors,
                "answer_reservation_held": self._answer_reserved,
                "cancelled": self._cancelled, "cancel_reason": self._cancel_reason,
                "stages": deepcopy(self._stages), "attempts": deepcopy(self._attempts),
                "plan": deepcopy(self.plan), "features": deepcopy(self.features),
            }


def _config_value(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _copy_config(config, **changes):
    copied = deepcopy(config)
    if isinstance(copied, Mapping):
        return {**copied, **changes}
    if is_dataclass(copied) and not isinstance(copied, type):
        return replace(copied, **changes)
    for name, value in changes.items():
        setattr(copied, name, value)
    return copied


class BudgetedModelAPI:
    """Account at the common model boundary, preserving the unscoped API."""

    def __init__(self, raw):
        self._raw = raw

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def _routing_kwargs(self, runtime, kwargs):
        with runtime._lock:
            bias = deepcopy(runtime.plan.get("routing_bias"))
        if bias is None:
            return kwargs
        if not isinstance(bias, (list, tuple)) or not bias:
            raise ValueError("routing_bias must be a nonempty flat list of finite numbers")
        if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) for value in bias):
            raise ValueError("routing_bias must be a nonempty flat list of finite numbers")
        if not getattr(self._raw, "supports_routing_bias", False):
            raise ValueError("The model backend does not support the configured routing_bias")
        if "routing_bias" in kwargs:
            raise ValueError("routing_bias is controlled by the request plan")
        return {**kwargs, "routing_bias": [[float(value) for value in bias]]}

    def generate(self, prompt, decoding_config, **kwargs):
        runtime = current_runtime()
        if runtime is None:
            return self._raw.generate(prompt, decoding_config, **kwargs)
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if {"max_new_tokens", "min_new_tokens", "num_beams"}.intersection(kwargs):
            raise ValueError("budget-controlled decoding options must be supplied in decoding_config")
        maximum = _integer("max_new_tokens", _config_value(decoding_config, "max_new_tokens"), 1)
        minimum = _config_value(decoding_config, "min_new_tokens")
        if minimum is not None:
            _integer("min_new_tokens", minimum)
            if minimum > maximum:
                raise ValueError("min_new_tokens cannot exceed max_new_tokens")
        beams = _integer("num_beams", _config_value(decoding_config, "num_beams", 1), 1)
        kwargs = self._routing_kwargs(runtime, kwargs)
        # Copy before reservation so unsupported/mutable caller configs cannot leak.
        config = _copy_config(decoding_config, max_new_tokens=maximum)
        record, granted = runtime._reserve("generate", len(prompt) * beams, maximum, beams)
        try:
            changes = {"max_new_tokens": granted}
            if minimum is not None:
                changes["min_new_tokens"] = min(minimum, granted)
            config = _copy_config(config, **changes)
            result = self._raw.generate(prompt, config, **kwargs)
        except BaseException as exc:
            runtime._finish(record, error=exc)
            raise
        runtime._finish(record, result=result)
        return result

    def score(self, prompt, candidate, **kwargs):
        runtime = current_runtime()
        if runtime is None:
            return self._raw.score(prompt, candidate, **kwargs)
        if not isinstance(prompt, str) or not isinstance(candidate, str):
            raise TypeError("prompt and candidate must be strings")
        kwargs = self._routing_kwargs(runtime, kwargs)
        record, _ = runtime._reserve("score", len(prompt) + len(candidate))
        try:
            result = self._raw.score(prompt, candidate, **kwargs)
        except BaseException as exc:
            runtime._finish(record, error=exc)
            raise
        runtime._finish(record, result=result)
        return result

    def entropy_estimate(self, prompt, **kwargs):
        runtime = current_runtime()
        estimate = getattr(self._raw, "entropy_estimate", None)
        if estimate is None:
            estimate = getattr(self._raw, "estimate_entropy", None)
        if runtime is None:
            return estimate(prompt, **kwargs)
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        record, _ = runtime._reserve("entropy", len(prompt))
        try:
            details_fn = getattr(self._raw, "entropy_details", None)
            if callable(details_fn):
                details = details_fn(prompt, **kwargs)
                runtime.features["entropy_probe"] = {
                    "available": bool(details.get("available", False)),
                    "kind": str(details.get("kind", "unspecified")),
                    "calibrated": bool(details.get("calibrated", False)),
                }
                if not details.get("available", False):
                    raise ValueError("Model entropy is unavailable")
                result = details["value"]
            else:
                result = estimate(prompt, **kwargs)
            if isinstance(result, bool) or not isinstance(result, Real) or not isfinite(result) or not 0 <= result <= 1:
                raise ValueError("entropy_estimate must return a finite normalized value in [0, 1]")
        except BaseException as exc:
            runtime._finish(record, error=exc)
            raise
        runtime._finish(record, result=result)
        return result

    def estimate_entropy(self, prompt, **kwargs):
        return self.entropy_estimate(prompt, **kwargs)


def request_model_api(raw):
    """Account each acquired raw connection, without nesting budget proxies."""
    if current_runtime() is not None and not isinstance(raw, BudgetedModelAPI):
        return BudgetedModelAPI(raw)
    return raw


__all__ = [
    "BudgetLimits", "BudgetExhausted", "RequestRuntime", "BudgetedModelAPI",
    "current_runtime", "request_runtime", "runtime_stage",
]
