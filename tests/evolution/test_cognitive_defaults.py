"""Cost, routing and resilience contracts of the cognitive layer's defaults.

Counting fake backends and a tiny random model only: no training, no downloads,
no network (``urllib.request.urlopen`` is replaced and counted).

Measured before this change (commit 3aa1e47, library defaults, the same
``Backend`` fakes and ``PLAIN_MESSAGES`` as below): 59-64 ``generate`` calls per
message with ``Backend("untrained")`` (1 answer, 28 constitutional evaluations,
28 suggestions, 1 constitutional revision, 1 self-refine, plus 3
self-consistency samples and 2 debate thoughts on routed messages) and 31-34
with ``Backend("compliant")``, which also made 2 outbound Wikipedia attempts per
message. Now: 2 calls per message for the lean default; 31-33 and 17 for the
full profile. ``test_calls_per_message_lean_default_and_full_profile`` measures
and prints the current numbers.
"""
import collections
import hashlib
import time
import urllib.request

import pytest
import torch

from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.cognitive_types import (
    CognitiveInput, CognitiveState, DecodingConfig, PolicyOutput,
)
from cognitive_management.config import COGNITIVE_PROFILES, CognitiveManagerConfig
from cognitive_management.exceptions import CognitiveError
from cognitive_management.v2.adapters.backend_adapter import ModelAPIAdapter
from cognitive_management.v2.components.constitutional_critic import ConstitutionalCritic
from cognitive_management.v2.components.critic_v2 import CriticV2
from cognitive_management.v2.components.fact_checkers import WikipediaFactChecker
from cognitive_management.v2.components.policy_router_v2 import PolicyRouterV2
from cognitive_management.v2.config.constitutional_principles import (
    CONSTITUTIONAL_PRINCIPLES, detect_principle_language, get_principles,
)
from cognitive_management.v2.middleware.error_handler import (
    CircuitBreakerConfig, CircuitState, ErrorHandlingMiddleware,
)
from cognitive_management.v2.processing.handlers import SelfConsistencyHandler
from cognitive_management.v2.processing.pipeline import ProcessingContext
from cognitive_management.v2.utils.heuristics import build_features, mode_gates

WORDS = "ve bir bu için ile da de çok daha gibi olarak kadar sonra ancak ama zaman yıl gün insan".split()
PLAIN_MESSAGES = [
    "Merhaba, nasılsın?",
    "Türkiye'nin başkenti neresidir?",
    "Neden gökyüzü mavi görünür, açıklar mısın?",
    "Bana kısa bir kitap önerir misin?",
]
EVALUATION_PREFIX = "Aşağıdaki metnin şu prensibe"


class Backend:
    """Counting ModelAPI fake.

    ``untrained`` mimics a random-initialised model: token salad, score 0 and
    near-maximal entropy. ``compliant`` follows every judge prompt and answers
    with factual sentences. ``entropy`` may be a float, ``"raise"`` or
    ``"unavailable"`` (an ``entropy_details`` probe reporting available=False
    while ``entropy_estimate`` returns the 0.5 placeholder, as CevahirModelAPI does).
    """

    def __init__(self, kind="compliant", entropy=0.3, answer=None):
        self.kind, self.entropy, self.answer = kind, entropy, answer
        self.calls = collections.Counter()
        self.prompts = []

    def generate(self, prompt, decoding_cfg, **kwargs):
        self.calls["generate"] += 1
        self.prompts.append(prompt)
        if self.kind == "compliant":
            if "Sadece sayıyı yaz" in prompt:
                return "0.0"
            if "Tek kelimeyle" in prompt:
                return "DOĞRU"
            return self.answer or "Ankara Türkiye'nin başkentidir. Bu bilgi genel olarak doğrudur."
        digest = int(hashlib.sha256((prompt + str(self.calls["generate"])).encode()).hexdigest(), 16)
        return " ".join(WORDS[(digest >> (5 * i)) % len(WORDS)] for i in range(8 + digest % 12))

    def score(self, prompt, candidate, **kwargs):
        self.calls["score"] += 1
        return 0.0 if self.kind == "untrained" else 0.8

    def entropy_estimate(self, text, **kwargs):
        self.calls["entropy"] += 1
        if self.entropy == "raise":
            raise RuntimeError("entropy probe failed")
        if self.entropy == "unavailable":
            return 0.5
        return self.entropy

    def __getattr__(self, name):
        if name == "entropy_details" and self.__dict__.get("entropy") == "unavailable":
            return lambda text, **kwargs: {"value": 0.5, "available": False,
                                           "kind": "next_token_entropy", "calibrated": False}
        raise AttributeError(name)


class NoEntropyBackend:
    """A valid ModelAPI without the optional entropy probe."""

    def __init__(self):
        self.inner = Backend("compliant")

    def generate(self, prompt, decoding_cfg, **kwargs):
        return self.inner.generate(prompt, decoding_cfg, **kwargs)

    def score(self, prompt, candidate, **kwargs):
        return self.inner.score(prompt, candidate, **kwargs)


@pytest.fixture(autouse=True)
def network(monkeypatch):
    """Record every outbound urlopen attempt and fail it like a blocked network."""
    attempts = []

    def blocked(request, *args, **kwargs):
        attempts.append({"url": getattr(request, "full_url", str(request)), "timeout": kwargs.get("timeout")})
        raise OSError("network disabled in tests")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    return attempts


def offline(cfg):
    """Keep tests independent of optional embedding packages and their warnings."""
    cfg.memory.enable_vector_memory = False
    cfg.memory.enable_rag = False
    cfg.runtime.enable_auto_anomaly_check = False
    return cfg


def run(cfg, backend, message):
    manager = CognitiveManager(backend, offline(cfg))
    state = CognitiveState(session_id="s", metadata={"user_id": "u"})
    return manager.handle(state, CognitiveInput(user_message=message)), state


# --------------------------------------------------------------------------- profiles

def test_lean_profile_is_the_documented_default():
    default = CognitiveManagerConfig()
    assert default.profile_name() == "lean"
    assert CognitiveManagerConfig.lean() == default
    for section, switches in COGNITIVE_PROFILES["lean"].items():
        for key, value in switches.items():
            assert getattr(getattr(default, section), key) == value
    assert default.critic.enabled and default.critic.max_passes == 1

    full = CognitiveManagerConfig.full()
    assert full.profile_name() == "full"
    assert full.policy.allow_inner_steps and full.policy.debate_enabled and full.policy.tot_enabled
    assert full.policy.self_consistency_enabled
    assert full.critic.enable_constitutional_ai and full.critic.enable_external_fact_checking
    assert full.critic.enable_llm_fact_verification
    # Only the profile switches differ; every other field keeps its default.
    lean_dict, full_dict = default.to_dict(), full.to_dict()
    for section, switches in COGNITIVE_PROFILES["full"].items():
        for key in switches:
            lean_dict[section].pop(key)
            full_dict[section].pop(key)
    assert lean_dict == full_dict

    # One switch from a config file or the environment.
    assert CognitiveManagerConfig().override_with({"profile": "full"}).profile_name() == "full"
    mixed = CognitiveManagerConfig().override_with({"profile": "full", "critic": {"enable_external_fact_checking": False}})
    assert mixed.profile_name() == "custom" and mixed.critic.enable_constitutional_ai
    assert CognitiveManagerConfig.from_env({"CM_PROFILE": "full"}).profile_name() == "full"
    assert CognitiveManagerConfig.from_env({}).profile_name() == "lean"
    with pytest.raises(ValueError):
        CognitiveManagerConfig().apply_profile("fast")
    with pytest.raises(ValueError):
        CognitiveManagerConfig.from_env({"CM_PROFILE": "ful"})
    bad = CognitiveManagerConfig()
    bad.critic.fact_checking_time_budget = 0
    with pytest.raises(ValueError):
        bad.validate()


# ---------------------------------------------------------------- calls per message

@pytest.mark.parametrize("kind,entropy", [("untrained", 0.99), ("compliant", 0.3)])
def test_calls_per_message_lean_default_and_full_profile(kind, entropy, network, capsys):
    report = []
    for message in PLAIN_MESSAGES:
        lean_backend, full_backend = Backend(kind, entropy), Backend(kind, entropy)
        network.clear()
        lean_out, _ = run(CognitiveManagerConfig(), lean_backend, message)
        lean_net = len(network)
        network.clear()
        run(CognitiveManagerConfig.full(), full_backend, message)
        full_net = len(network)
        report.append((message, lean_backend.calls["generate"], full_backend.calls["generate"], full_net))

        # Lean: one answer generation plus at most one lexical-critic revision.
        assert lean_backend.calls["generate"] <= 2
        assert lean_backend.calls["score"] == 0
        assert lean_backend.calls["entropy"] == 1
        assert lean_out.used_mode == "direct"
        assert lean_net == 0
        assert not any(p.startswith(EVALUATION_PREFIX) for p in lean_backend.prompts)
        # Full restores the multi-call behaviour, with each principle evaluated once.
        evaluations = sum(p.startswith(EVALUATION_PREFIX) for p in full_backend.prompts)
        assert evaluations == len(CONSTITUTIONAL_PRINCIPLES) == 14
        assert full_backend.calls["generate"] >= 8 * lean_backend.calls["generate"]
        if kind == "compliant":
            assert 0 < full_net <= CognitiveManagerConfig().critic.fact_checking_max_claims
    with capsys.disabled():
        print(f"\n[{kind}] generate calls per message (lean / full, full network attempts):")
        for message, lean_calls, full_calls, full_net in report:
            print(f"  {message!r}: {lean_calls} / {full_calls} ({full_net})")


def test_tree_of_thoughts_route_only_runs_in_the_full_profile(capsys):
    message = "Bu denklemi çöz ve adım adım hesapla: " + " ".join(["x + 2y = 10"] * 20)
    lean, full = Backend("compliant"), Backend("compliant")
    lean_out, _ = run(CognitiveManagerConfig(), lean, message)
    full_out, _ = run(CognitiveManagerConfig.full(), full, message)
    assert lean_out.used_mode == "direct"
    assert lean.calls["generate"] <= 2 and lean.calls["score"] == 0
    assert full_out.used_mode == "tot"
    # Depth 3 x branching 3: 1 + 3 + 9 expansions, four scored candidates each.
    assert full.calls["score"] == 52
    assert full.calls["generate"] >= 52 + 1 + len(CONSTITUTIONAL_PRINCIPLES)
    with capsys.disabled():
        print(f"\n[tot route] lean: {dict(lean.calls)}; full: {dict(full.calls)}")


def test_tiny_model_backend_lean_default_costs_at_most_two_generations():
    """The real CevahirModelAPI with a tiny random model, counted at the adapter."""
    from model.cevahir import CevahirModelAPI
    from model_management.config_schema import tiny_model_config
    from model_management.model_manager import ModelManager

    class ToyTokenizer:
        def encode(self, text, **kwargs):
            return list(text[:4]), [5 + (ord(c) % 100) for c in text[:4]]

        def get_vocab(self):
            return {"<EOS>": 3}

        def decode(self, ids, **kwargs):
            return " ".join(str(i) for i in ids if i != 3)

    torch.manual_seed(0)
    manager = ModelManager(tiny_model_config(num_layers=1))
    api = CevahirModelAPI(manager, ToyTokenizer())
    counts = collections.Counter()
    original_generate = api.generate

    def counted(prompt, cfg, **kwargs):
        counts["generate"] += 1
        return original_generate(prompt, cfg, **kwargs)

    api.generate = counted
    cfg = CognitiveManagerConfig()
    cfg.decoding_bounds.max_new_tokens_bounds = (2, 4)
    out, state = run(cfg, api, "Merhaba, nasılsın?")
    assert 1 <= counts["generate"] <= 2
    assert out.used_mode == "direct"
    assert state.last_entropy is not None  # the tiny model's entropy probe is a real measurement


# ------------------------------------------------------------ constitutional principles

def test_principles_are_deduplicated_and_selected_by_response_language():
    assert len(CONSTITUTIONAL_PRINCIPLES) == 14
    tr = get_principles(languages=("tr",))
    en = get_principles(languages=("en",))
    assert tr == [turkish for _, turkish in CONSTITUTIONAL_PRINCIPLES]
    assert en == [english for english, _ in CONSTITUTIONAL_PRINCIPLES]
    assert len(set(tr)) == len(tr) == 14
    assert len(get_principles()) == 28  # historical flat list unchanged for importers
    assert get_principles(("Kısa ol.",), languages=("tr", "tr"))[-1] == "Kısa ol."
    assert len(get_principles(languages=("tr", "en"))) == 28
    with pytest.raises(ValueError):
        get_principles(languages=("de",))
    assert detect_principle_language("Ankara Türkiye'nin başkentidir.") == "tr"
    assert detect_principle_language("The capital of Turkey is Ankara.") == "en"
    assert detect_principle_language("12345") is None

    cfg = CognitiveManagerConfig.full()
    cfg.critic.custom_principles = ("Kısa ol.",)
    for answer, language in [("Ankara Türkiye'nin başkentidir.", "tr"),
                             ("The capital of Turkey is Ankara.", "en"),
                             ("12345", "tr")]:
        backend = Backend("compliant")
        critic = ConstitutionalCritic(cfg, backend)
        text, details = critic.review_with_constitution("Başkent neresi?", answer)
        evaluated = [p for p in backend.prompts if p.startswith(EVALUATION_PREFIX)]
        expected = get_principles(("Kısa ol.",), languages=(language,))
        assert len(evaluated) == backend.calls["generate"] == len(expected) == 15
        assert all(f"PRENSİP: {principle}\n" in prompt for principle, prompt in zip(expected, evaluated))
        assert text == answer and details["revised"] is False


# ------------------------------------------------------------------ self-consistency

@pytest.mark.parametrize("entropy,samples", [(0.3, 0), (0.6, 3), (None, 0), (0.9, 0)])
def test_self_consistency_obeys_the_router_gate(entropy, samples):
    cfg = CognitiveManagerConfig.full()
    backend = Backend("compliant")
    message = "Türkiye'nin başkenti neresidir?"
    scaled = None if entropy is None else entropy * 3.0
    features = build_features(cfg, user_message=message, entropy_est=scaled)
    assert features["query_type"] not in ("creative", "conversational")
    gate = mode_gates(cfg, features)["self_consistency_gate"]
    assert gate is (samples > 0)
    context = ProcessingContext(CognitiveState(), CognitiveInput(message))
    context.features = features
    context.context_text = "[USER]\n" + message
    context.draft_text = "taslak"
    context.policy_output = PolicyOutput(mode="direct", tool="none", decoding=DecodingConfig(max_new_tokens=8))
    SelfConsistencyHandler(ModelAPIAdapter(backend), cfg)._process(context)
    assert backend.calls["generate"] == samples
    assert (context.self_consistency_result is not None) is (samples > 0)


# --------------------------------------------------------------- unavailable entropy

@pytest.mark.parametrize("probe", ["raise", "unavailable", "missing"])
def test_unavailable_entropy_routes_direct(probe):
    backend = NoEntropyBackend() if probe == "missing" else Backend("compliant", entropy=probe)
    message = "Türkiye'nin başkenti neresidir?"
    out, state = run(CognitiveManagerConfig.full(), backend, message)
    assert out.used_mode == "direct"
    assert state.last_entropy is None
    assert "[Özellik Çıkarımı]" in out.reasoning_chain[0].content
    assert "entropy=yok" in out.reasoning_chain[0].content
    # Control: the same message with a measured entropy of 0.5 (1.5 on the router
    # scale, exactly the think threshold) still takes the deliberation path.
    measured, measured_state = run(CognitiveManagerConfig.full(), Backend("compliant", entropy=0.5), message)
    assert measured.used_mode == "think1" and measured_state.last_entropy == pytest.approx(1.5)


def test_adapter_reports_none_instead_of_placeholder_entropy():
    from model.cevahir import CevahirModelAPI
    from model_management.config_schema import tiny_model_config
    from model_management.model_manager import ModelManager

    class EmptyTokenizer:
        def encode(self, text, **kwargs):
            return [], []

        def get_vocab(self):
            return {"<EOS>": 3}

        def decode(self, ids, **kwargs):
            return ""

    api = CevahirModelAPI(ModelManager(tiny_model_config(num_layers=1)), EmptyTokenizer())
    assert api.entropy_details("metin")["available"] is False
    assert api.entropy_estimate("metin") == 0.5  # placeholder, not a measurement
    assert ModelAPIAdapter(api).estimate_entropy("metin") is None
    assert ModelAPIAdapter(Backend(entropy="raise")).estimate_entropy("x") is None
    assert ModelAPIAdapter(Backend(entropy=float("nan"))).estimate_entropy("x") is None
    assert ModelAPIAdapter(Backend(entropy=0.25)).estimate_entropy("x") == 0.25

    # Routing on explicit None: a math query would otherwise always think first.
    router = PolicyRouterV2(CognitiveManagerConfig.full())
    math_features = build_features(router.cfg, user_message="12*7 kaç eder?", entropy_est=None)
    assert math_features["entropy_est"] is None
    assert router.route(math_features, CognitiveState()).mode == "direct"
    assert router.route({**math_features, "entropy_est": 0.3}, CognitiveState()).mode == "think1"


# ------------------------------------------------------------------ network egress

FACTUAL_ANSWER = ("Ankara 1923 yılında başkent oldu. İstanbul nüfusu 15 milyon kişidir. "
                  "Türkiye 81 ilden oluşur. Everest 8848 metre yüksekliğindedir.")


def test_wikipedia_egress_is_off_by_default(network):
    cfg = CognitiveManagerConfig()
    critic = CriticV2(cfg, Backend("compliant", answer=FACTUAL_ANSWER))
    assert critic._fact_checkers == []
    critic.review("Türkiye hakkında bilgi ver.", FACTUAL_ANSWER)
    out, _ = run(CognitiveManagerConfig(), Backend("compliant", answer=FACTUAL_ANSWER), "Türkiye hakkında bilgi ver.")
    assert out.text and network == []


def test_opt_in_wikipedia_checks_are_bounded_per_response(network, monkeypatch):
    cfg = CognitiveManagerConfig()
    cfg.critic.enable_external_fact_checking = True
    critic = CriticV2(cfg, Backend("compliant"))
    assert [type(c) for c in critic._fact_checkers] == [WikipediaFactChecker]

    # Claim cap: four claims in the answer, three requests.
    critic._check_facts(FACTUAL_ANSWER)
    assert len(network) == cfg.critic.fact_checking_max_claims == 3
    assert all(0 < attempt["timeout"] <= 5.0 for attempt in network)

    # Time budget: each request "takes" 4 s on a fake clock; no request starts
    # after the 10 s budget and none gets a timeout longer than what is left.
    cfg.critic.fact_checking_max_claims = 10
    network.clear()
    clock = {"now": 1000.0}

    def slow(request, *args, **kwargs):
        network.append({"timeout": kwargs.get("timeout")})
        clock["now"] += 4.0
        raise OSError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", slow)
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    critic._check_facts(FACTUAL_ANSWER)
    assert [attempt["timeout"] for attempt in network] == [5.0, 5.0, pytest.approx(2.0)]


# ------------------------------------------------------------ error-handler honesty

def test_error_handler_neither_retries_nor_claims_a_timeout(monkeypatch):
    with pytest.raises(TypeError):
        ErrorHandlingMiddleware(retry_config=object())
    with pytest.raises(TypeError):
        ErrorHandlingMiddleware(timeout=30.0)
    middleware = ErrorHandlingMiddleware(CircuitBreakerConfig(failure_threshold=2, failure_window=60.0))
    assert not hasattr(middleware, "retry_config") and not hasattr(middleware, "timeout")
    monkeypatch.setattr(time, "sleep", lambda *_: pytest.fail("error handling must not sleep"))
    response = middleware.on_error(CognitiveState(), CognitiveInput("x"), CognitiveError("bozuk"))
    assert "bozuk" in response.text and middleware.circuit_state.failures == 1

    # failure_window is real: an old failure does not count toward opening the circuit.
    from datetime import datetime, timedelta
    middleware.circuit_state.last_failure_time = datetime.now() - timedelta(seconds=61)
    middleware.on_error(CognitiveState(), CognitiveInput("x"), RuntimeError("yeni"))
    assert middleware.circuit_state.failures == 1
    assert middleware.circuit_state.state == CircuitState.CLOSED
    middleware.on_error(CognitiveState(), CognitiveInput("x"), RuntimeError("yine"))
    assert middleware.circuit_state.state == CircuitState.OPEN


def test_failed_pipeline_runs_once_and_answers_with_an_error(monkeypatch):
    manager = CognitiveManager(Backend("compliant"), offline(CognitiveManagerConfig()))
    runs = []

    def broken(*args, **kwargs):
        runs.append(1)
        raise RuntimeError("pipeline failure")

    monkeypatch.setattr(manager._orchestrator.pipeline, "process", broken)
    monkeypatch.setattr(time, "sleep", lambda *_: pytest.fail("no retry back-off expected"))
    out = manager.handle(CognitiveState(), CognitiveInput("Merhaba"))
    assert runs == [1]
    assert out.text == "Beklenmeyen bir hata oluştu. Lütfen tekrar deneyin."
