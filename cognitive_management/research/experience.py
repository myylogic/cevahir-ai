"""Bounded, externally verified route experience; no model or content storage.

An observation is *not* evidence of answer quality. Only explicit external
feedback verifies it. This store never trains weights and never labels a route
that did not execute. Its scores describe retained observations, not causal or
counterfactual estimates. The confidence penalty is a conservative heuristic,
not a calibrated statistical confidence interval.

Callers must authenticate feedback sources and supply an opaque scope, exact
model/tokenizer revisions, a comparable task bucket, and a stable sample_id
(e.g. a hash of the input). Task/sample keys are hashed before storage. Without
sample_id, all observations of one task bucket count as only one sample.
No prompt/response fields exist. Caller-supplied scope/source/request/revision
IDs must themselves be free of content. Snapshots are trusted local artifacts,
not an authorization boundary or an untrusted feedback ingestion endpoint.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import math
from threading import RLock
import time
from uuid import uuid4


_STATUSES = frozenset({"observed", "budget_limited", "error", "cache"})
_SOURCES = frozenset({"user", "evaluator", "tool_verified"})
_IDENTITY_KEYS = {"model_revision", "tokenizer_revision"}
_CONFIG_KEYS = {
    "max_records_per_scope", "min_support", "cost_weight",
    "decay_half_life_seconds", "max_scopes", "max_feedback_events_per_record",
}


def _string(name, value, maximum=512):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    return value


def _integer(name, value, minimum=1, maximum=1000000):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(name, value, minimum=0.0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its finite range")
    return result


def _identity(value):
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        raise ValueError("identity requires exactly model_revision and tokenizer_revision")
    return (_string("model_revision", value["model_revision"]),
            _string("tokenizer_revision", value["tokenizer_revision"]))


def _fingerprint(value):
    return sha256(_string("task/sample key", value, 8192).encode("utf-8")).hexdigest()


def _digest(name, value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value


def _source_ids(value):
    if not isinstance(value, (tuple, list)) or len(value) > 32:
        raise ValueError("source_ids must be a list/tuple of at most 32 opaque IDs")
    result = tuple(_string("source_id", item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError("source_ids must be unique")
    return result


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid {name} fields")


@dataclass(frozen=True)
class FeedbackEvent:
    event_id: str
    source: str
    value: float
    created_at: float
    supersedes: str | None = None


@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    scope: str
    identity: tuple[str, str]
    task_fingerprint: str
    sample_fingerprint: str
    planned_strategy: str
    executed_strategy: str
    status: str
    cost: float
    source_ids: tuple[str, ...]
    request_id: str | None
    created_at: float
    feedback: tuple[FeedbackEvent, ...] = ()


class ExperienceStore:
    """Thread-safe bounded storage and conservative, scoped route preferences.

    Scope capacity rejects new scopes; per-scope retention evicts the oldest
    observation. Event history is bounded too: further corrections are rejected
    when full rather than silently discarding replay protection. Replay IDs are
    unique within the retained store; forgetting/eviction removes their history.
    ``feedback`` returns False for an identical replay, raises ValueError for
    conflicts/invalid ownership, and returns True for a new event. A newer event
    from the same source supersedes that source's previous assessment. Current
    assessments from different sources combine by their minimum, not their sum.

    Only ``observed`` records with feedback count toward support. A negative
    (< 0.5) assessment of budget_limited/error/cache records vetoes promotion of
    that executed route while retained; their positive assessments never help
    promote any route or weaken the baseline. This explicit veto is not decayed;
    correct the feedback or forget the record to retract it. Other evidence is
    exponentially decayed from observation time. Stale support is conservative.
    """

    def __init__(self, max_records_per_scope=256, min_support=3, cost_weight=0.1,
                 decay_half_life_seconds=604800, max_scopes=128,
                 max_feedback_events_per_record=32):
        self._lock = RLock()
        self._config = self._validate_config({
            "max_records_per_scope": max_records_per_scope, "min_support": min_support,
            "cost_weight": cost_weight, "decay_half_life_seconds": decay_half_life_seconds,
            "max_scopes": max_scopes,
            "max_feedback_events_per_record": max_feedback_events_per_record,
        })
        self._records = OrderedDict()
        self._scopes = {}
        self._requests = {}
        self._events = {}
        self._revision = 0

    @staticmethod
    def _validate_config(config):
        _fields(config, _CONFIG_KEYS, "config")
        result = dict(config)
        for name in ("max_records_per_scope", "min_support", "max_scopes", "max_feedback_events_per_record"):
            result[name] = _integer(name, result[name])
        if result["min_support"] > result["max_records_per_scope"]:
            raise ValueError("min_support exceeds per-scope retention")
        result["cost_weight"] = _number("cost_weight", result["cost_weight"], maximum=1.0)
        result["decay_half_life_seconds"] = _number("decay_half_life_seconds", result["decay_half_life_seconds"], minimum=1e-6)
        return result

    @property
    def revision(self):
        with self._lock:
            return self._revision

    def _remove(self, experience_id):
        record = self._records.pop(experience_id)
        scoped = self._scopes[record.scope]
        del scoped[experience_id]
        if not scoped:
            del self._scopes[record.scope]
        if record.request_id is not None:
            del self._requests[(record.scope, record.identity, record.request_id)]
        for event in record.feedback:
            del self._events[event.event_id]

    def _insert(self, record):
        self._records[record.experience_id] = record
        self._scopes.setdefault(record.scope, OrderedDict())[record.experience_id] = None
        if record.request_id is not None:
            self._requests[(record.scope, record.identity, record.request_id)] = record.experience_id
        for event in record.feedback:
            self._events[event.event_id] = (record.experience_id, event)

    def observe(self, *, scope, identity, task_key, planned_strategy, executed_strategy,
                status, cost, source_ids=(), request_id=None, sample_id=None):
        """Record an unverified execution; identical request retries are idempotent."""
        scope = _string("scope", scope, 2048)
        identity = _identity(identity)
        task_fingerprint = _fingerprint(task_key)
        sample_fingerprint = task_fingerprint if sample_id is None else _fingerprint(sample_id)
        planned_strategy = _string("planned_strategy", planned_strategy, 128)
        executed_strategy = _string("executed_strategy", executed_strategy, 128)
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("status must be observed, budget_limited, error, or cache")
        cost = _number("cost", cost)
        source_ids = _source_ids(source_ids)
        request_id = None if request_id is None else _string("request_id", request_id)
        record = ExperienceRecord(uuid4().hex, scope, identity, task_fingerprint,
                                  sample_fingerprint, planned_strategy, executed_strategy,
                                  status, cost, source_ids, request_id, time.time())
        with self._lock:
            if request_id is not None:
                previous_id = self._requests.get((scope, identity, request_id))
                if previous_id is not None:
                    previous = self._records[previous_id]
                    retry = replace(record, experience_id=previous.experience_id,
                                    created_at=previous.created_at, feedback=previous.feedback)
                    if retry != previous:
                        raise ValueError("request_id replay conflicts with its original observation")
                    return previous_id
            if scope not in self._scopes and len(self._scopes) >= self._config["max_scopes"]:
                raise ValueError("scope capacity reached; forget an existing scope first")
            scoped = self._scopes.get(scope)
            if scoped is not None and len(scoped) >= self._config["max_records_per_scope"]:
                self._remove(next(iter(scoped)))
            self._insert(record)
            self._revision += 1
            return record.experience_id

    def feedback(self, *, experience_id, scope, identity, value, source, event_id):
        """Apply authenticated external feedback; caller owns source authentication."""
        experience_id = _string("experience_id", experience_id)
        scope = _string("scope", scope, 2048)
        identity = _identity(identity)
        value = _number("value", value, maximum=1.0)
        if not isinstance(source, str) or source not in _SOURCES:
            raise ValueError("source must be user, evaluator, or tool_verified")
        event_id = _string("event_id", event_id)
        with self._lock:
            record = self._records.get(experience_id)
            if record is None or record.scope != scope or record.identity != identity:
                raise ValueError("experience does not belong to this scope and identity")
            previous = self._events.get(event_id)
            if previous is not None:
                previous_id, event = previous
                if previous_id != experience_id or event.source != source or event.value != value:
                    raise ValueError("event_id replay conflicts with its original feedback")
                return False
            if len(record.feedback) >= self._config["max_feedback_events_per_record"]:
                raise ValueError("feedback event capacity reached for this experience")
            supersedes = next((e.event_id for e in reversed(record.feedback) if e.source == source), None)
            previous_time = record.feedback[-1].created_at if record.feedback else record.created_at
            event = FeedbackEvent(event_id, source, value, max(time.time(), previous_time), supersedes)
            self._records[experience_id] = replace(record, feedback=record.feedback + (event,))
            self._events[event_id] = (experience_id, event)
            self._revision += 1
            return True

    @staticmethod
    def _quality(record):
        latest = {}
        for event in record.feedback:
            latest[event.source] = event.value
        return min(latest.values()) if latest else None

    def recommend(self, *, scope, identity, task_key, baseline, allowed):
        """Return baseline unless supported observed alternatives clearly dominate.

        Cost is normalized as cost/(1+cost). Each unique sample contributes once
        per executed action, using its latest retained observation (including an
        unverified latest observation, which retracts that sample's support).
        Root routing must include budget/configuration in task_key for comparison.
        Both baseline and candidate require min_support independent samples; only
        permitted actions are returned, and baseline must itself be permitted.
        """
        scope = _string("scope", scope, 2048)
        identity = _identity(identity)
        task_fingerprint = _fingerprint(task_key)
        baseline = _string("baseline", baseline, 128)
        if not isinstance(allowed, (list, tuple, set, frozenset)) or not 1 <= len(allowed) <= 64:
            raise ValueError("allowed must contain 1 to 64 strategy names")
        actions = sorted({_string("allowed strategy", action, 128) for action in allowed})
        if baseline not in actions:
            raise ValueError("baseline must be allowed")
        with self._lock:
            # Derive from records every time: correction, forget, and eviction
            # cannot leave a stale learned preference in a separate cache.
            latest = {}
            negatives = {action: set() for action in actions}
            for experience_id in self._scopes.get(scope, ()):
                record = self._records[experience_id]
                if (record.identity != identity or record.task_fingerprint != task_fingerprint
                        or record.executed_strategy not in negatives):
                    continue
                latest[(record.executed_strategy, record.sample_fingerprint)] = record
                value = self._quality(record)
                if record.status != "observed" and value is not None and value < 0.5:
                    negatives[record.executed_strategy].add(record.sample_fingerprint)
            now = time.time()
            groups = {action: [] for action in actions}
            for (action, _), record in latest.items():
                value = self._quality(record)
                if record.status != "observed" or value is None:
                    continue
                age = max(0.0, now - record.created_at)
                weight = 2.0 ** (-age / self._config["decay_half_life_seconds"])
                groups[action].append((weight, value, record.cost / (1.0 + record.cost)))
            scores = {}
            for action, samples in groups.items():
                effective_support = math.fsum(row[0] for row in samples)
                if effective_support > 0:
                    quality = math.fsum(w * value for w, value, _ in samples) / effective_support
                    cost = math.fsum(w * cost for w, _, cost in samples) / effective_support
                    utility = quality - self._config["cost_weight"] * cost
                    penalty = min(2.0, 0.5 / math.sqrt(effective_support))
                    lower, upper = utility - penalty, utility + penalty
                else:
                    quality = cost = utility = penalty = lower = upper = None
                scores[action] = {
                    "support": len(samples), "effective_support": effective_support,
                    "mean_quality": quality, "mean_cost": cost, "utility": utility,
                    "confidence_penalty": penalty, "lower": lower, "upper": upper,
                    "negative_evidence": len(negatives[action]),
                }
            reference = scores[baseline]
            result = {"strategy": baseline, "support": reference["support"],
                      "reason": "insufficient_baseline_support", "scores": scores,
                      "revision": self._revision}
            if reference["support"] < self._config["min_support"] or reference["upper"] is None:
                return result
            candidates = [action for action in actions if action != baseline
                          and scores[action]["support"] >= self._config["min_support"]
                          and not scores[action]["negative_evidence"]
                          and scores[action]["lower"] is not None
                          and scores[action]["lower"] > reference["upper"]]
            if candidates:
                selected = max(candidates, key=lambda action: (scores[action]["lower"], action))
                result.update(strategy=selected, support=scores[selected]["support"],
                              reason="supported_observed_preference")
            else:
                result["reason"] = "no_supported_advantage"
            return result

    def forget(self, *, scope, experience_id=None):
        """Remove a scoped observation or entire scope, including its feedback."""
        scope = _string("scope", scope, 2048)
        if experience_id is not None:
            experience_id = _string("experience_id", experience_id)
        with self._lock:
            ids = list(self._scopes.get(scope, ())) if experience_id is None else [experience_id]
            removed = 0
            for item in ids:
                record = self._records.get(item)
                if record is not None and record.scope == scope:
                    self._remove(item)
                    removed += 1
            if removed:
                self._revision += 1
            return removed

    def snapshot(self):
        """Return an independent versioned JSON-compatible value, never raw content."""
        with self._lock:
            records = []
            for r in self._records.values():
                records.append({
                    "experience_id": r.experience_id, "scope": r.scope,
                    "identity": dict(zip(("model_revision", "tokenizer_revision"), r.identity)),
                    "task_fingerprint": r.task_fingerprint, "sample_fingerprint": r.sample_fingerprint,
                    "planned_strategy": r.planned_strategy, "executed_strategy": r.executed_strategy,
                    "status": r.status, "cost": r.cost, "source_ids": list(r.source_ids),
                    "request_id": r.request_id, "created_at": r.created_at,
                    "feedback": [{"event_id": e.event_id, "source": e.source,
                                  "value": e.value, "created_at": e.created_at,
                                  "supersedes": e.supersedes} for e in r.feedback],
                })
            return {"schema_version": 1, "config": dict(self._config),
                    "revision": self._revision, "records": records}

    @staticmethod
    def _parse_record(raw, max_events):
        _fields(raw, {"experience_id", "scope", "identity", "task_fingerprint", "sample_fingerprint",
                      "planned_strategy", "executed_strategy", "status", "cost", "source_ids",
                      "request_id", "created_at", "feedback"}, "record")
        if not isinstance(raw["status"], str) or raw["status"] not in _STATUSES:
            raise ValueError("invalid observation status")
        created_at = _number("created_at", raw["created_at"])
        if not isinstance(raw["feedback"], list) or len(raw["feedback"]) > max_events:
            raise ValueError("feedback history exceeds bound")
        events, latest = [], {}
        previous_time = created_at
        for data in raw["feedback"]:
            _fields(data, {"event_id", "source", "value", "created_at", "supersedes"}, "feedback")
            source = data["source"]
            if not isinstance(source, str) or source not in _SOURCES:
                raise ValueError("invalid feedback source")
            event_id = _string("event_id", data["event_id"])
            event_time = _number("feedback created_at", data["created_at"], minimum=previous_time)
            if data["supersedes"] != latest.get(source):
                raise ValueError("invalid feedback supersession chain")
            events.append(FeedbackEvent(event_id, source, _number("value", data["value"], maximum=1),
                                        event_time, data["supersedes"]))
            latest[source] = event_id
            previous_time = event_time
        return ExperienceRecord(
            _string("experience_id", raw["experience_id"]), _string("scope", raw["scope"], 2048),
            _identity(raw["identity"]), _digest("task_fingerprint", raw["task_fingerprint"]),
            _digest("sample_fingerprint", raw["sample_fingerprint"]),
            _string("planned_strategy", raw["planned_strategy"], 128),
            _string("executed_strategy", raw["executed_strategy"], 128), raw["status"],
            _number("cost", raw["cost"]), _source_ids(raw["source_ids"]),
            None if raw["request_id"] is None else _string("request_id", raw["request_id"]),
            created_at, tuple(events))

    def restore(self, snapshot):
        """Atomically replace state after complete schema/retention validation.

        Persisted bounds/configuration are restored. Revision always advances,
        even when loading an older snapshot, to invalidate caller-side caches.
        Parse JSON with duplicate-key rejection before calling this method when
        snapshots arrive as text; this API receives an already-decoded value.
        """
        _fields(snapshot, {"schema_version", "config", "revision", "records"}, "snapshot")
        if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1:
            raise ValueError("unsupported experience snapshot version")
        config = self._validate_config(snapshot["config"])
        saved_revision = _integer("revision", snapshot["revision"], minimum=0, maximum=2**63 - 1)
        rows = snapshot["records"]
        if not isinstance(rows, list) or len(rows) > config["max_scopes"] * config["max_records_per_scope"]:
            raise ValueError("snapshot exceeds total retention bound")
        candidate = ExperienceStore(**config)
        for raw in rows:
            record = self._parse_record(raw, config["max_feedback_events_per_record"])
            if record.experience_id in candidate._records:
                raise ValueError("duplicate experience_id in snapshot")
            if record.scope not in candidate._scopes and len(candidate._scopes) >= config["max_scopes"]:
                raise ValueError("snapshot exceeds scope bound")
            if len(candidate._scopes.get(record.scope, ())) >= config["max_records_per_scope"]:
                raise ValueError("snapshot exceeds per-scope retention bound")
            if record.request_id is not None and (record.scope, record.identity, record.request_id) in candidate._requests:
                raise ValueError("duplicate request_id in snapshot")
            seen = set(candidate._events)
            for event in record.feedback:
                if event.event_id in seen:
                    raise ValueError("duplicate event_id in snapshot")
                seen.add(event.event_id)
            candidate._insert(record)
        with self._lock:
            self._config = config
            self._records = candidate._records
            self._scopes = candidate._scopes
            self._requests = candidate._requests
            self._events = candidate._events
            self._revision = max(self._revision, saved_revision) + 1
