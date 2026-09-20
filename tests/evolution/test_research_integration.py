"""Cheap causal integration probes: fake model only, no training/downloads."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.cognitive_types import CognitiveInput, CognitiveState, PolicyOutput, DecodingConfig
from cognitive_management.research.runtime import BudgetLimits, current_runtime
from cognitive_management.research.controller import ResearchController
from cognitive_management.v2.processing.pipeline import ProcessingContext
from cognitive_management.v2.components.critic_v2 import CriticV2
from cognitive_management.v2.utils.context_pruning import build_context


class FakeModel:
    supports_routing_bias = True

    def __init__(self):
        self.calls = []

    def generate(self, prompt, decoding, **kwargs):
        self.calls.append((current_runtime().stage if current_runtime() else "baseline", kwargs))
        return "Yerel kontrollü yanıt."

    def score(self, prompt, candidate, **kwargs):
        self.calls.append(("score", kwargs))
        return 0.8

    def entropy_estimate(self, prompt):
        return 0.2


def configuration(mode="shadow"):
    cfg = CognitiveManagerConfig()
    cfg.memory.enable_vector_memory = False
    cfg.memory.enable_rag = False
    cfg.critic.enabled = False
    cfg.critic.enable_external_fact_checking = False
    cfg.runtime.enable_auto_anomaly_check = False
    cfg.runtime.enable_logging = False
    cfg.runtime.enable_telemetry = False
    cfg.policy.allow_inner_steps = False
    cfg.research.mode = mode
    cfg.research.model_revision = "fake-model-1"
    cfg.research.tokenizer_revision = "fake-tokenizer-1"
    return cfg


def test_disabled_path_has_no_research_accounting():
    model = FakeModel()
    manager = CognitiveManager(model, configuration("off"))
    out = manager.handle(CognitiveState(), CognitiveInput("Merhaba"))
    assert out.text and "execution" not in out.metadata
    assert model.calls == [("baseline", {})]
    assert current_runtime() is None


def test_feedback_lifecycle_and_cache_bypass():
    manager = CognitiveManager(FakeModel(), configuration())
    state = CognitiveState(metadata={"user_id": "a"}, session_id="s")
    out = manager.handle(state, CognitiveInput("Merhaba"))
    execution = out.metadata["execution"]
    eid = execution["experience_id"]
    assert execution["status"] == "observed"
    assert not execution["quality_verified"]
    assert execution["calls"]["generate"] == 1
    assert manager.record_experience_feedback(state, eid, value=1, source="user", event_id="event-1")
    assert not manager.record_experience_feedback(state, eid, value=1, source="user", event_id="event-1")
    with pytest.raises(ValueError):
        manager.record_experience_feedback(CognitiveState(session_id="other"), eid,
                                           value=1, source="user", event_id="bad")
    # Identical fresh state normally permits a cached response; a research trial runs.
    repeated = manager.handle(CognitiveState(metadata={"user_id": "a"}, session_id="s"), CognitiveInput("Merhaba"))
    assert repeated.metadata["execution"]["calls"]["generate"] == 1
    assert repeated.metadata["execution"]["experience_id"] != eid
    assert manager.forget_experience(state, eid) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_deliberation_cannot_consume_final_answer_reservation(asynchronous):
    cfg = configuration()
    cfg.policy.allow_inner_steps = True
    cfg.research.limits = BudgetLimits(max_generate_calls=1, max_output_tokens=32,
        reserve_answer_tokens=32, max_input_chars=2000, reserve_answer_input_chars=1800)
    model = FakeModel()
    manager = CognitiveManager(model, cfg)
    manager._orchestrator.policy_router.route = lambda **kw: PolicyOutput(
        mode="debate2", tool="none", decoding=DecodingConfig(max_new_tokens=100))
    args = CognitiveState(), CognitiveInput("Bir plan oluştur")
    out = asyncio.run(manager.handle_async(*args)) if asynchronous else manager.handle(*args)
    execution = out.metadata["execution"]
    assert out.text == "Yerel kontrollü yanıt."
    assert execution["calls"]["generate"] == 1
    assert execution["reserved_output_tokens"] == 32
    assert execution["status"] == "budget_limited"
    assert execution["plan"]["planned_strategy"] == "debate2"
    assert execution["plan"]["executed_strategy"] == out.used_mode == "direct"
    assert execution["stages"]["answer"]["generate_calls"] == 1
    assert out.reasoning_chain and "processing_errors" in out.metadata


def test_sync_async_tot_factory_and_metadata_match():
    cfg = configuration()
    cfg.policy.allow_inner_steps = True
    cfg.policy.tot_enabled = True
    manager = CognitiveManager(FakeModel(), cfg)
    def find(first, prefix):
        while first:
            if first.name == prefix + "Deliberation":
                return first
            first = first._next
    sync = find(manager._orchestrator.pipeline._first_handler, "")
    asynchronous = find(manager._orchestrator._build_async_pipeline()._first_handler, "Async")._sync_handler
    assert sync.tree_of_thoughts is not None and asynchronous.tree_of_thoughts is not None
    assert sync.cfg is asynchronous.cfg


def test_adaptive_selection_requires_verified_diverse_samples_and_retracts():
    cfg = configuration("adaptive")
    controller = ResearchController(cfg, allow_deliberation=True)
    state = CognitiveState(session_id="test")
    request = CognitiveInput("novel held out question")
    def route():
        with controller.session(state, request) as runtime:
            context = ProcessingContext(state, request, features={"domain": "general"})
            policy = controller.route(context, PolicyOutput("direct", "none", DecodingConfig()))
            return policy, runtime
    _, initial = route()
    for action, value in [("direct", 0), ("think1", 1)]:
        for i in range(8):
            eid = controller.store.observe(scope=initial.scope, identity=controller.identity,
                task_key=initial.plan["task_key"], planned_strategy=action, executed_strategy=action,
                status="observed", cost=0.1, sample_id=f"training-{i}")
            controller.store.feedback(scope=initial.scope, identity=controller.identity,
                experience_id=eid, value=value, source="evaluator", event_id=f"{action}-{i}")
    chosen, after = route()
    assert chosen.mode == "think1"
    assert after.plan["recommendation"]["support"] == 8
    controller.forget(state)
    assert route()[0].mode == "direct"


def test_routing_profile_reaches_generation_and_score_but_not_entropy():
    cfg = configuration()
    cfg.research.enable_neural_routing = True
    cfg.research.routing_bias_by_domain = {"general": [0.2, -0.2]}
    model = FakeModel()
    manager = CognitiveManager(model, cfg)
    out = manager.handle(CognitiveState(), CognitiveInput("Merhaba"))
    assert model.calls[0] == ("answer", {"routing_bias": [[0.2, -0.2]]})
    assert out.metadata["execution"]["plan"]["routing_profile_id"]


def test_disabled_or_unavailable_entropy_is_not_validated_quality():
    cfg = configuration()
    model = FakeModel()
    model.entropy_details = lambda prompt: {"value": .5, "available": False,
                                           "kind": "next_token_entropy", "calibrated": False}
    out = CognitiveManager(model, cfg).handle(CognitiveState(), CognitiveInput("Merhaba"))
    execution = out.metadata["execution"]
    assert execution["features"]["uncertainty_source"] == "unavailable"
    assert not execution["features"]["entropy_probe"]["available"]
    assert execution["errors"] == 1


def test_critic_review_details_are_call_local():
    cfg = configuration()
    cfg.critic.enabled = True
    cfg.critic.enable_constitutional_ai = False
    critic = CriticV2(cfg, FakeModel())
    critic._evaluate_all = lambda user, draft, context: [user]
    critic._should_revise = lambda feedback: False
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: critic.review_detailed(name, "draft"), ["A", "B"]))
    assert [r.feedback for r in results] == [("A",), ("B",)]
    assert [r.passes for r in results] == [1, 1]


def test_system_instruction_and_current_question_survive_history_pruning():
    cfg = configuration()
    cfg.memory.max_history_tokens = 128
    text = build_context(cfg, system_prompt="Özel sistem talimatı", user_message="Güncel soru",
                         history=[{"role": "assistant", "content": "eski " * 200}])
    assert "[SYSTEM]\nÖzel sistem talimatı" in text
    assert text.endswith("[USER]\nGüncel soru")


def test_retrieved_evidence_is_reused_without_second_lookup():
    cfg = configuration()
    cfg.memory.enable_rag = True
    model = FakeModel()
    manager = CognitiveManager(model, cfg)
    lookups = []
    manager._memory_service.retrieve_context = lambda **kw: lookups.append(kw) or [
        {"id": "source-1", "content": "Geçmiş bilgi", "score": .8, "role": "user"}]
    out = manager.handle(CognitiveState(), CognitiveInput("Merhaba"))
    assert len(lookups) == 1
    assert out.memory_hits == 1 and out.context_sources == ["source-1"]
    record = manager._orchestrator.research.store.snapshot()["records"][0]
    assert record["source_ids"] == ["source-1"]


def test_research_config_roundtrip_and_identity_requirement():
    cfg = configuration()
    copy = CognitiveManagerConfig().override_with(cfg.to_dict())
    copy.validate()
    assert copy.to_dict() == cfg.to_dict()
    copy.research.model_revision = ""
    with pytest.raises(ValueError, match="revisions"):
        copy.validate()


def test_persistent_experience_rejects_mismatched_artifacts_transactionally(tmp_path):
    manager = CognitiveManager(FakeModel(), configuration())
    state = CognitiveState(session_id="persist")
    out = manager.handle(state, CognitiveInput("Merhaba"))
    eid = out.metadata["execution"]["experience_id"]
    manager.record_experience_feedback(state, eid, value=1, source="evaluator", event_id="verify")
    path = tmp_path / "experience.json"
    manager.save_experience(path)
    manager.forget_experience(state)
    manager.load_experience(path)
    assert manager.forget_experience(state, eid) == 1
    cfg = configuration()
    cfg.research.model_revision = "another-model"
    other = CognitiveManager(FakeModel(), cfg)
    before = other._orchestrator.research.store.snapshot()
    with pytest.raises(ValueError, match="identity/profile"):
        other.load_experience(path)
    assert other._orchestrator.research.store.snapshot() == before


def test_live_configuration_changes_cannot_mix_experiments():
    cfg = configuration()
    manager = CognitiveManager(FakeModel(), cfg)
    cfg.critic.max_passes += 1
    with pytest.raises(ValueError, match="configuration changed"):
        manager.handle(CognitiveState(), CognitiveInput("Merhaba"))


def test_nested_disabled_controller_does_not_inherit_outer_research_state():
    outer = ResearchController(configuration())
    inner = ResearchController(configuration("off"))
    state, request = CognitiveState(), CognitiveInput("nested")
    with outer.session(state, request) as runtime:
        with inner.session(state, CognitiveInput("inner")):
            assert current_runtime() is None
        assert current_runtime() is runtime


def test_direct_orchestrator_construction_also_counts_raw_backend():
    from cognitive_management.v2.core.orchestrator import CognitiveOrchestrator
    from cognitive_management.v2.components.policy_router_v2 import PolicyRouterV2
    from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
    cfg = configuration()
    model = FakeModel()
    orchestrator = CognitiveOrchestrator(model, PolicyRouterV2(cfg), MemoryServiceV2(cfg), CriticV2(cfg, model))
    output = orchestrator.handle(CognitiveState(), CognitiveInput("Merhaba"))
    assert output.metadata["execution"]["calls"] == {"generate": 1, "score": 0, "entropy": 1}


def test_pooled_backend_failure_retry_cannot_escape_budget():
    from cognitive_management.research.runtime import RequestRuntime, request_runtime, runtime_stage, BudgetExhausted
    from cognitive_management.v2.adapters.backend_adapter import ModelAPIAdapter
    calls = []
    model = FakeModel()
    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("backend failed after starting")
    model.generate = fail
    pool = SimpleNamespace(acquire=lambda: model, release=lambda conn: None)
    adapter = ModelAPIAdapter(model, connection_pool=pool)
    limits = BudgetLimits(max_generate_calls=1)
    with request_runtime(RequestRuntime("s", {}, limits)) as runtime, runtime_stage("answer"):
        with pytest.raises(BudgetExhausted):
            adapter.generate("prompt", DecodingConfig())
    assert calls == [1]
    assert runtime.snapshot()["calls"]["generate"] == 1
