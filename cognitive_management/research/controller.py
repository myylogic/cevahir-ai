"""Connect externally verified experience to bounded cognitive execution.

No weights are trained here. Recommendations compare supported actions within
coarse, explicit feature strata; they are not causal estimates of unseen actions.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .experience import ExperienceStore
from .runtime import RequestRuntime, current_runtime, request_runtime


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state_scope(state):
    return json.dumps([str(state.metadata.get("user_id", "local")), str(state.session_id)],
                      separators=(",", ":"))


class ResearchController:
    VERSION = "experience-compute-1"

    def __init__(self, cfg, *, allow_deliberation=False, allow_tot=False):
        self._live_cfg = cfg
        self.cfg = deepcopy(cfg)
        self.settings = self.cfg.research
        self.settings.validate()
        self.enabled = self.settings.mode != "off"
        self._configuration_id = fingerprint(asdict(cfg))
        self.identity = {"model_revision": self.settings.model_revision,
                         "tokenizer_revision": self.settings.tokenizer_revision}
        self.allowed = ["direct"]
        if allow_deliberation:
            self.allowed.extend(["think1", "debate2"])
        if allow_tot:
            self.allowed.append("tot")
        self.store = ExperienceStore(
            max_records_per_scope=self.settings.max_records_per_scope,
            min_support=self.settings.min_support,
            cost_weight=self.settings.cost_weight,
            decay_half_life_seconds=self.settings.decay_half_life_seconds,
        ) if self.enabled else None
        # Fixed controls define which outcomes are comparable across ablations.
        fixed = {"policy": asdict(cfg.policy), "critic": asdict(cfg.critic),
                 "decoding": asdict(cfg.default_decoding),
                 "limits": asdict(self.settings.limits),
                 "routing": self.settings.routing_bias_by_domain if self.settings.enable_neural_routing else {},
                 "system_prompt": cfg.default_system_prompt, "version": self.VERSION}
        self.profile_id = fingerprint(fixed)

    @contextmanager
    def session(self, state, request):
        if not self.enabled:
            with request_runtime(None):
                yield None
            return
        if fingerprint(asdict(self._live_cfg)) != self._configuration_id:
            raise ValueError("Research configuration changed; rebuild the cognitive manager for a new experiment")
        runtime = RequestRuntime(state_scope(state), self.identity, self.settings.limits)
        runtime.sample_id = fingerprint(request.user_message)
        runtime.source_ids = []
        # Trials must perform their own work. A cache reuse is not new evidence.
        request.metadata["_research_run"] = True
        with request_runtime(runtime):
            try:
                yield runtime
            except BaseException as exc:
                runtime.cancel(type(exc).__name__)
                self._observe(runtime, "error", "direct")
                raise

    def route(self, context, policy):
        runtime = current_runtime()
        if runtime is None:
            return policy
        features = context.features
        def number(key, default=0.0):
            value = features.get(key, default)
            return float(value) if isinstance(value, (float, int)) and math.isfinite(value) else default
        # Retrieval scores from different providers are not calibrated uncertainty.
        evidence = {
            "query_type": str(features.get("query_type", "unknown")),
            "domain": str(features.get("domain", "general")),
            "complexity_bin": min(2, max(0, int(number("complexity_score") * 3))),
            "uncertainty_bin": min(2, max(0, int(number("entropy_est")))),
            "uncertainty_source": features.get("uncertainty_source", "unspecified"),
            "memory_present": bool(features.get("has_relevant_memory", False)),
        }
        if "entropy_probe" in runtime.features:
            evidence["entropy_probe"] = runtime.features["entropy_probe"]
        runtime.source_ids = list(dict.fromkeys(str(item["id"]) for item in context.retrieved_contexts
                                               if item.get("id")))[:32]
        baseline = policy.mode if policy.mode in self.allowed else "direct"
        profile = fingerprint([self.profile_id, asdict(policy.decoding),
                               context.request.system_prompt, evidence])
        recommendation = {"strategy": baseline, "support": 0, "reason": "budget_only", "scores": {}}
        if self.settings.mode in {"shadow", "adaptive"}:
            recommendation = self.store.recommend(scope=runtime.scope, identity=self.identity,
                task_key=profile, baseline=baseline, allowed=self.allowed)
        chosen = recommendation["strategy"] if self.settings.mode == "adaptive" else baseline
        # Optional randomized coverage is explicit and reproducible from request ID.
        # It collects actual outcomes; it does not invent counterfactual rewards.
        rate = self.settings.exploration_rate if self.settings.mode == "adaptive" else 0.0
        draw = int(fingerprint([runtime.request_id, "explore"])[:13], 16) / float(16 ** 13)
        exploiting = chosen
        if draw < rate:
            index = int(fingerprint([runtime.request_id, "action"])[:8], 16) % len(self.allowed)
            chosen = self.allowed[index]
        propensity = rate / len(self.allowed) + ((1 - rate) if chosen == exploiting else 0)
        plan = {"version": self.VERSION, "mode": self.settings.mode,
                "task_key": profile, "baseline_strategy": baseline,
                "planned_strategy": chosen, "executed_strategy": chosen,
                "recommendation": recommendation, "selection_probability": propensity,
                "exploration": draw < rate, "fallbacks": []}
        if self.settings.enable_neural_routing:
            vector = self.settings.routing_bias_by_domain.get(evidence["domain"])
            if vector is not None:
                plan["routing_bias"] = list(vector)
                plan["routing_profile_id"] = fingerprint(vector)
        runtime.set_metadata(plan=plan, features=evidence)
        policy.mode = chosen
        return policy

    def _observe(self, runtime, status, strategy):
        if self.store is None or self.settings.mode == "budget":
            return None
        snapshot = runtime.snapshot()
        # Unitless bounded cost, consistent only within the fixed budget profile.
        limits = snapshot["limits"]
        cost = sum((snapshot["calls"]["generate"] / max(1, limits["max_generate_calls"]),
                    snapshot["calls"]["score"] / max(1, limits["max_score_calls"]),
                    snapshot["reserved_output_tokens"] / max(1, limits["max_output_tokens"]),
                    snapshot["input_chars"] / max(1, limits["max_input_chars"]))) / 4
        try:
            return self.store.observe(scope=runtime.scope, identity=self.identity,
                task_key=runtime.plan.get("task_key", "unrouted"),
                planned_strategy=runtime.plan.get("planned_strategy", "direct"),
                executed_strategy=strategy, status=status, cost=cost,
                source_ids=tuple(runtime.source_ids), request_id=runtime.request_id,
                sample_id=runtime.sample_id)
        except ValueError as exc:
            runtime.plan["experience_record_error"] = str(exc)
            return None

    def finish(self, runtime, request, response):
        if runtime is None:
            return response
        snapshot = runtime.snapshot()
        status = "observed"
        if snapshot["denied"]:
            status = "budget_limited"
        elif snapshot["errors"] or response.metadata.get("processing_errors") or not response.text:
            status = "error"
        strategy = runtime.plan.get("executed_strategy", response.used_mode)
        experience_id = self._observe(runtime, status, strategy)
        execution = runtime.snapshot()
        execution.update(status=status, experience_id=experience_id,
                         quality_verified=False, profile_id=self.profile_id)
        response.metadata["execution"] = execution
        return response

    def feedback(self, state, experience_id, *, value, source, event_id):
        if self.store is None:
            raise ValueError("Research experience is disabled")
        return self.store.feedback(experience_id=experience_id, scope=state_scope(state),
                                  identity=self.identity, value=value, source=source, event_id=event_id)

    def forget(self, state, experience_id=None):
        if self.store is None:
            return 0
        return self.store.forget(scope=state_scope(state), experience_id=experience_id)

    def save(self, path):
        """Explicit local persistence, separate from neural weight checkpoints."""
        if self.store is None:
            raise ValueError("Research experience is disabled")
        payload = {"schema_version": 1, "identity": self.identity,
                   "profile_id": self.profile_id, "store": self.store.snapshot()}
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                             prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load(self, path):
        if self.store is None:
            raise ValueError("Research experience is disabled")
        def unique_keys(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("Duplicate snapshot key")
                value[key] = item
            return value
        with Path(path).open("rb") as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError("Experience snapshot exceeds the 32 MiB local load limit")
        payload = json.loads(data, object_pairs_hook=unique_keys)
        if (not isinstance(payload, dict) or set(payload) != {"schema_version", "identity", "profile_id", "store"}
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or payload["identity"] != self.identity or payload["profile_id"] != self.profile_id):
            raise ValueError("Experience snapshot identity/profile mismatch")
        if not isinstance(payload["store"], dict) or payload["store"].get("config") != self.store.snapshot()["config"]:
            raise ValueError("Experience snapshot retention/policy configuration mismatch")
        self.store.restore(payload["store"])
