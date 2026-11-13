"""Correctness of the cognitive layer's judges, classifiers, memory and prompts.

Each test pins a defect that was reproduced on the code before the fix:
judge replies read without negation ("Yanlış, doğru değil" scored as true),
label/answer replies read from the label alone ("İhlal: yok" as a violation),
negators found inside other words ("medikal"), keyword substrings inside
unrelated Turkish words ("yaz" in "yazılım") and the correct matches a stricter
word matcher lost ("suçlu", "önerirsin"), cross-user eviction and duplicated
retrieval hits in the episodic memory, a score threshold that meant a different
cosine in each vector store, notes that search could not find, and a debate
prompt with a literal "{perspective}". No network and no neural model are used.
"""
import math
from types import SimpleNamespace

import pytest

from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.cognitive_types import CognitiveInput, CognitiveState
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.v2.components.constitutional_critic import ConstitutionalCritic
from cognitive_management.v2.components.critic_v2 import CriticV2
from cognitive_management.v2.components.deliberation_engine_v2 import DeliberationEngineV2
from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
from cognitive_management.v2.components.rag_enhancer import RAGEnhancer
from cognitive_management.v2.components.vector_store.base import (
    BaseVectorStore,
    cosine_to_score,
    score_threshold_to_cosine,
)
from cognitive_management.v2.components.vector_store.chroma_vector_store import ChromaVectorStore
from cognitive_management.v2.components.vector_store.memory_vector_store import MemoryVectorStore
from cognitive_management.v2.utils.heuristics import build_features
from cognitive_management.v2.utils.request_scope import memory_scope
from cognitive_management.v2.utils.turkish_text import keyword_in_text, turkish_lower, turkish_upper
from cognitive_management.v2.utils.verdict_parsing import (
    FALSE,
    PARTIAL,
    TRUE,
    UNCERTAIN,
    UNPARSEABLE,
    parse_truth_verdict,
    parse_violation_score,
)


def offline_config():
    """No network and no vector store. The judges under test are switched on here, so
    the tests do not depend on which cognitive stages are on by default."""
    cfg = CognitiveManagerConfig()
    cfg.memory.enable_vector_memory = False
    cfg.critic.enable_external_fact_checking = False
    cfg.critic.enable_llm_fact_verification = True
    cfg.critic.enable_constitutional_ai = True
    return cfg


class FixedReply:
    """Model stub that answers every prompt with the same text and records prompts."""

    def __init__(self, reply=""):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, decoding_cfg=None, **kwargs):
        self.prompts.append(prompt)
        return self.reply

    def score(self, prompt, candidate, **kwargs):
        return 0.5


# --- P2: judge verdicts --------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ("Yanlış, doğru değil", FALSE),
    ("Kısmen doğru", PARTIAL),
    ("KISMEN DOĞRU", PARTIAL),
    ("Kısmen doğru, kısmen yanlış", PARTIAL),
    ("Doğru değil", FALSE),
    ("Doğru", TRUE),
    ("DOĞRU.", TRUE),
    ("Doğrudur. Yanlış anlaşılacak bir yanı yok.", TRUE),  # only the first verdict sentence
    ("YANLIŞ", FALSE),          # str.lower() gives "yanliş"; the old parser missed it
    ("yanlis", FALSE),
    ("Yanlış değil", TRUE),
    ("BELİRSİZ", UNCERTAIN),
    ("Emin değilim", UNCERTAIN),
    ("Muhtemelen doğru", UNCERTAIN),
    ("Not true", FALSE),
    ("It isn't correct", FALSE),
    ("Partially true", PARTIAL),
    ("", UNPARSEABLE),
    ("kitap yol ve su", UNPARSEABLE),
    ("Doğru ve yanlış", UNPARSEABLE),
    ("DOĞRU, YANLIŞ, KISMEN DOĞRU veya BELİRSİZ mi?", UNPARSEABLE),  # echoed options
    ("DOĞRU, YANLIŞ veya BELİRSİZ mi? Tek kelimeyle yanıtla. Yanlış", FALSE),
    # Negators are whole words or word endings, not letters inside other words.
    ("Doğru medikal bilgi.", TRUE),
    ("Doğru komedi.", TRUE),
    ("Correct immediately.", TRUE),
    ("Bu iddia doğru görünmüyor.", FALSE),
    ("Bu doğru olmayabilir.", UNCERTAIN),
    ("Doğruluk: yüksek", UNPARSEABLE),          # a measure noun, not a verdict
])
def test_truth_verdicts_respect_negation_and_hedges(reply, expected):
    assert parse_truth_verdict(reply).label == expected


@pytest.mark.parametrize("reply, expected", [
    # A label or question takes its value or answer from the next segment.
    ("Doğru: hayır", FALSE),
    ("Doğru: evet", TRUE),
    ("Doğru: değil", FALSE),
    ("Doğru mu? Hayır, yanlış.", FALSE),
    ("doğru mu? yanlış.", FALSE),
    ("Doğru mu? Evet.", TRUE),
    ("Doğru mu? Kısmen.", PARTIAL),
    ("Doğru mu? Evet, hayır.", UNPARSEABLE),
    ("Correct? No.", FALSE),
    ("Karar: doğru değil", FALSE),
    ("Yanlış: nüfus 2020'de artmadı.", FALSE),   # the label is the verdict
    ("Doğru mu?", UNPARSEABLE),                  # a question is not its own answer
    ("Cevap: Evet", UNPARSEABLE),                # yes to nothing
])
def test_truth_verdicts_read_labels_and_answers(reply, expected):
    assert parse_truth_verdict(reply).label == expected


@pytest.mark.parametrize("reply, expected", [
    ("Yanlış, doğru değil", 0.20),
    ("Kısmen doğru", 0.70),
    ("DOĞRU", 0.90),
    ("Belirsiz", 0.55),
    ("Doğru: hayır", 0.20),
    ("Doğru mu? Hayır, yanlış.", 0.20),
    ("", None),
    ("kitap yol ve su", None),
])
def test_llm_fact_check_scores_and_abstains(reply, expected):
    critic = CriticV2(offline_config(), FixedReply(reply))
    claim = SimpleNamespace(claim="Türkiye'nin nüfusu 2020 yılında yüzde 50 arttı")
    assert critic._check_facts_llm("metin", [claim]) == expected


def test_unreadable_fact_judge_leaves_the_checker_score_alone():
    class AlwaysVerified:
        def verify(self, claim, **kwargs):   # kwargs: e.g. a per-request timeout
            return SimpleNamespace(is_verified=True)

    text = "Türkiye'nin nüfusu 2020 yılında yüzde 50 arttı."
    unreadable = CriticV2(offline_config(), FixedReply(""))
    unreadable._fact_checkers = [AlwaysVerified()]
    assert unreadable._check_facts_enhanced(text) == pytest.approx(1.0)
    refuted = CriticV2(offline_config(), FixedReply("Yanlış, doğru değil"))
    refuted._fact_checkers = [AlwaysVerified()]
    assert refuted._check_facts_enhanced(text) == pytest.approx(0.6 * 1.0 + 0.4 * 0.20)


@pytest.mark.parametrize("reply, expected", [
    ("uygun değil, ihlal var", 0.8),
    ("ihlal yok", 0.0),           # the old parser returned 0.8: "ihlal" without the negation
    ("Uygun.", 0.0),
    ("Uygunsuz", 0.8),
    ("Prensibe aykırı değil", 0.0),
    ("Metin hiçbir ilkeyi ihlal etmiyor", 0.0),
    ("Kısmen uygun", 0.5),
    ("No violation", 0.0),
    ("This does not violate the principle", 0.0),
    ("non-compliant", 0.8),
    ("0,8", 0.8),                 # Turkish decimal comma; the old parser read 0.0
    ("0.2", 0.2),
    ("%80", 0.8),
    ("8/10", 0.8),
    ("Skor: 7", 0.7),
    ("Bu metin 2. prensibe uygundur", 0.0),  # an ordinal is not a score
    ("", None),
    ("kitap yol ve su", None),
    ("uygun ama ihlal var", None),
    # Label: value and question? answer.
    ("İhlal: yok", 0.0),
    ("İhlal: Yok.", 0.0),
    ("İhlal var mı? Yok.", 0.0),
    ("Violation: none", 0.0),
    ("İhlal: var", 0.8),
    ("İhlal var mı? Hayır.", 0.0),
    ("Uygun: hayır", 0.8),
    ("Uygun mu? Hayır, uygun.", None),          # contradicts itself
    ("Değerlendirme: İhlal: yok", 0.0),
    # Negators and measure nouns.
    ("Uygun medikal öneri.", 0.0),
    ("İhlalsiz.", 0.0),
    ("Metin ihlal içermiyor.", 0.0),
    ("Uygunluk düşük.", None),
    # A percentage or ratio attached to a compliance word measures compliance.
    ("%100 uygun", 0.0),
    ("%20 uygun", 0.8),
    ("100% compliant", 0.0),
    ("Uygun: %90", 0.1),
    ("Uygunluk: 0,9", 0.1),
    ("%20 ihlal", 0.2),
    ("%100 uygun değil", None),
    ("0.0 (tamamen uygun)", 0.0),               # a plain number keeps the asked scale
    ("0.0 uygun", 0.0),
])
def test_violation_scores_respect_negation_and_abstain(reply, expected):
    score = parse_violation_score(reply)
    if expected is None:
        assert score is None
    else:
        assert score == pytest.approx(expected)


class ConstitutionalModel:
    """Answers evaluation prompts with ``verdict``; counts the other prompts."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.evaluations = 0
        self.other_calls = 0

    def generate(self, prompt, decoding_cfg=None, **kwargs):
        if "Sadece sayıyı yaz" in prompt:
            self.evaluations += 1
            return self.verdict
        self.other_calls += 1
        return "Düzeltilmiş metin."

    def score(self, prompt, candidate, **kwargs):
        return 0.5


# The counts are taken from the prompts the model received, not from the principle
# list, so the tests hold however many principles a review evaluates.

@pytest.mark.parametrize("verdict", [
    "", "kitap yol ve su", "ihlal yok", "0,1",
    "İhlal: yok", "İhlal var mı? Yok.", "Violation: none", "%100 uygun",
])
def test_unreadable_or_clean_verdicts_do_not_revise(verdict):
    model = ConstitutionalModel(verdict)
    critic = ConstitutionalCritic(offline_config(), model)
    text, info = critic.review_with_constitution("Soru?", "Taslak cevap.")
    assert (text, info["revised"], info["violations"]) == ("Taslak cevap.", False, [])
    assert model.evaluations > 0 and model.other_calls == 0
    assert info["unparseable"] == (model.evaluations if verdict in ("", "kitap yol ve su") else 0)


@pytest.mark.parametrize("verdict", ["uygun değil, ihlal var", "Uygun: hayır", "İhlal var mı? Evet."])
def test_negated_compliance_is_a_violation_and_revises(verdict):
    model = ConstitutionalModel(verdict)
    critic = ConstitutionalCritic(offline_config(), model)
    text, info = critic.review_with_constitution("Soru?", "Taslak cevap.")
    assert info["revised"] and text == "Düzeltilmiş metin."
    assert model.evaluations > 0 and len(info["violations"]) == model.evaluations
    assert all(v["score"] == pytest.approx(0.8) for v in info["violations"])


# --- P4: Turkish keyword classification ---------------------------------------

@pytest.mark.parametrize("message, query_type, domain, recent", [
    # Old result: creative / creative ("yaz" inside "yazılım").
    ("Yazılım mühendisliğine nasıl başlarım?", "reasoning", "general", False),
    # Old result: domain law ("hak" inside "hakkında").
    ("Hakkında bilgi verir misin?", "factual", "general", False),
    # Old result: math / math ("kaç" inside "kaçırdığım").
    ("Kaçırdığım treni nasıl yakalarım?", "reasoning", "general", False),
    # Old result: domain science ("gen" inside "genel"). "bugün" is a real word of this
    # weather-today question, so needs_recent_info stays true.
    ("Genel olarak bugün hava nasıl?", "reasoning", "general", True),
])
def test_keyword_substrings_inside_other_words_do_not_classify(message, query_type, domain, recent):
    features = build_features(CognitiveManagerConfig(), user_message=message, entropy_est=0.0)
    assert (features["query_type"], features["domain"], features["needs_recent_info"]) == \
        (query_type, domain, recent)


@pytest.mark.parametrize("message, query_type, domain", [
    ("Bana kısa bir şiir yaz.", "creative", "creative"),
    ("Bir hikaye yazar mısın?", "creative", "creative"),  # was conversational: "hi" in "hikaye"
    ("3 çarpı 4 kaç eder?", "math", "math"),
    ("İLAÇ nedir?", "factual", "medical"),   # was general: str.lower() leaves "i" + U+0307
    ("API nedir?", None, "technology"),                  # capital I in an English acronym
    ("Python'da bir fonksiyon yazar mısın?", "code", "technology"),
    ("Kodlama öğrenmek istiyorum", None, "technology"),
    ("Bölümü bitirdim, şimdi ne yapmalıyım?", None, "general"),   # bölüm is not "böl"
    ("Dün okula gittim", None, "general"),                 # neither "git" nor "ok"
    # Matched by the old substring test; derived words are now listed themselves,
    # and the aorist takes its person endings.
    ("Suçlu bulunursam ne olur?", None, "law"),
    ("Suçlamalar doğru mu?", None, "law"),
    ("Suçsuz olduğumu nasıl kanıtlarım?", "reasoning", "law"),
    ("Hakkımı nasıl ararım?", "reasoning", "law"),         # hak doubles its k: hakkım
    ("Bana ne önerirsin?", "reasoning", "general"),
    ("Bu ilaç etkili mi?", "reasoning", "medical"),
    ("GitHub'a projemi nasıl yüklerim?", "code", "technology"),
    ("Okey, anlaştık.", "conversational", "general"),
])
def test_inflected_keywords_still_classify(message, query_type, domain):
    features = build_features(CognitiveManagerConfig(), user_message=message, entropy_est=0.0)
    if query_type is not None:
        assert features["query_type"] == query_type
    assert features["domain"] == domain


@pytest.mark.parametrize("message", ["Bana yardım edebilir misin?", "Kontrol edebilir misin?"])
def test_edebilir_is_not_literary(message):
    # "edebi" (literary) takes any continuation but "edebilir" (can do); both
    # requests were classified creative/creative and answered at the creative temperature.
    features = build_features(CognitiveManagerConfig(), user_message=message, entropy_est=0.0)
    assert "creative" not in (features["query_type"], features["domain"])


@pytest.mark.parametrize("text, keyword, expected", [
    ("Yazılım mühendisliği", "yaz", False),
    ("Hakkında bilgi verir misin?", "hak", False),
    ("Kaçırdığım tren", "kaç", False),
    ("Genel olarak", "gen", False),
    ("Hakkında bilgi verir misin?", "bilgi ver", True),
    ("Lütfen yazın", "yaz", True),
    ("Saat kaçta?", "kaç", True),
    ("İlacı ne zaman almalıyım?", "ilaç", True),   # final consonant softening
    ("İstatistiği hesapla", "istatistik", True),
    ("Işığın hızı", "hız", True),
    ("Hızlı bir cevap", "hız", False),
    ("Farklı bir yöntem", "fark", False),
    ("Haklarımız", "hak", True),
    ("son haberler", "son haber", True),
    ("bugünkü durum", "bugün", True),
    ("TARIH dersi", "tarih", True),
    ("%50 arttı", "%", True),
    ("c++ öğren", "c++", True),
    ("Ne önerirsiniz?", "öner", True),
    ("Kaçar tane alalım?", "kaç", True),
    ("Neden kaçarsın?", "kaç", False),           # to flee, not "how many"
    ("Onu tanırım", "tanı", False),               # to know someone, not "diagnosis"
    ("Hakkını ara", "hak", True),
    ("Hakkımda konuştular", "hak", False),        # about me
    ("Edebiyat dersi", "edebi", True),
    ("Yardım edebilirim", "edebi", False),
])
def test_keyword_matching_table(text, keyword, expected):
    assert keyword_in_text(text, keyword) is expected


def test_turkish_casing_helpers():
    assert turkish_lower("KISMEN İLAÇ") == "kısmen ilaç"
    assert turkish_upper("konservatif ve güvenli") == "KONSERVATİF VE GÜVENLİ"


# --- P6: memory ---------------------------------------------------------------

def test_one_users_traffic_does_not_evict_another_users_memory():
    memory = MemoryServiceV2(offline_config())
    with memory_scope("alice"):
        memory.add_turn([], "user", "Benim fıstık alerjim var.")
    with memory_scope("bob"):
        for i in range(25):
            memory.add_turn([], "user", f"Bob mesaj {i}")
        assert 10 <= len(memory._episodic_memory) <= 20      # bob's own list is trimmed
    with memory_scope("alice"):
        assert [r["content"] for r in memory.retrieve_context("alerjim", top_k=3)] == \
            ["Benim fıstık alerjim var."]
        for i in range(20):
            memory.add_turn([], "user", f"Alice mesaj {i}")
        assert len(memory._episodic_memory) == 10            # her own traffic still trims hers
        assert not memory.retrieve_context("alerjim", top_k=3)


def test_number_of_scopes_kept_in_process_is_bounded():
    assert CognitiveManagerConfig().memory.episodic_max_scopes == 1024
    cfg = offline_config()
    cfg.memory.episodic_max_scopes = 2
    memory = MemoryServiceV2(cfg)
    assert memory._max_scopes == 2
    for scope in ("a", "b", "c"):
        with memory_scope(scope):
            memory.add_turn([], "user", f"kayıt {scope}")
    with memory_scope("a"):
        assert not memory.retrieve_context("kayıt")             # least recently written
    with memory_scope("b"):
        memory.add_turn([], "user", "yeni kayıt b")
    with memory_scope("d"):
        memory.add_turn([], "user", "kayıt d")
    with memory_scope("c"):
        assert not memory.retrieve_context("kayıt")
    with memory_scope("b"):
        assert memory.retrieve_context("kayıt")
    cfg.memory.episodic_max_scopes = 0
    with pytest.raises(ValueError):
        cfg.validate()


def test_retrieval_returns_each_record_once():
    memory = MemoryServiceV2(offline_config())
    with memory_scope("s"):
        memory.add_turn([], "user", "Benim adım Ayşe")
        memory.add_turn([], "assistant", "Memnun oldum.")
        memory.add_turn([], "user", "Benim adım Ayşe")     # same text, another turn
        hits = memory.retrieve_context("Benim adım", top_k=5)
    assert [h["content"] for h in hits] == ["Benim adım Ayşe"]
    assert memory.retrieve_context("Benim adım", top_k=0) == []


def test_second_turn_prompt_carries_each_message_once():
    cfg = offline_config()
    cfg.critic.enabled = False
    cfg.policy.self_consistency_enabled = False
    cfg.policy.allow_inner_steps = False
    model = FixedReply("Memnun oldum, not aldım.")
    manager = CognitiveManager(model, cfg=cfg)
    state = CognitiveState(session_id="s1", metadata={"user_id": "u1"})
    first, second = "Benim adım Ayşe ve en sevdiğim renk turkuaz.", "Benim en sevdiğim renk neydi?"
    manager.handle(state, CognitiveInput(user_message=first))
    model.prompts.clear()
    manager.handle(state, CognitiveInput(user_message=second))
    prompt = model.prompts[0]
    # Before the fix: first message 3x (history + two identical references), question 2x.
    assert prompt.count(first) == 1 and prompt.count(second) == 1
    assert prompt.rstrip().endswith(f"[USER]\n{second}")


def test_rag_adds_only_new_references_before_the_question():
    cfg = offline_config()
    rag = RAGEnhancer(MemoryServiceV2(cfg), cfg)
    context = "[SYSTEM]\nSistem.\n\n[USER]\nEski soru\n\n[ASSISTANT]\nEski cevap\n\n[USER]\nYeni soru?"
    items = [{"role": "user", "content": "Eski soru", "score": 0.9},
             {"role": "note", "content": "Kullanıcı turkuazı sever.", "score": 0.8},
             {"role": "note", "content": "Kullanıcı  turkuazı sever.", "score": 0.7}]
    enhanced = rag.enhance_context("Yeni soru?", existing_context=context, retrieved_items=items)
    assert enhanced.count("Eski soru") == 1 and enhanced.count("Yeni soru?") == 1
    assert enhanced.count("turkuazı sever") == 1
    assert enhanced.index("[RETRIEVED CONTEXT FROM MEMORY]") < enhanced.index("[USER]\nYeni soru?")
    assert enhanced.rstrip().endswith("[USER]\nYeni soru?")
    only_history = rag.enhance_context("Yeni soru?", existing_context=context, retrieved_items=items[:1])
    assert only_history == context


class FakeChromaCollection:
    """The part of a Chroma collection that ChromaVectorStore.search uses."""

    def __init__(self, ids, embeddings, metadatas, space=None, return_embeddings=True):
        self.ids, self.embeddings, self.metadatas = ids, embeddings, metadatas
        self.metadata = {"hnsw:space": space} if space else None
        self.return_embeddings = return_embeddings

    def _distance(self, query, vector):
        space = (self.metadata or {}).get("hnsw:space", "l2")
        dot = sum(a * b for a, b in zip(query, vector))
        if space in ("cosine", "ip"):
            return 1.0 - dot                                   # unit vectors
        return sum((a - b) ** 2 for a, b in zip(query, vector))  # squared L2, Chroma's default

    def query(self, query_embeddings, n_results, where=None, include=()):
        query = query_embeddings[0]
        rows = sorted(range(len(self.ids)), key=lambda i: self._distance(query, self.embeddings[i]))
        rows = [i for i in rows if not where or all(self.metadatas[i].get(k) == v for k, v in where.items())]
        rows = rows[:n_results]
        result = {"ids": [[self.ids[i] for i in rows]],
                  "documents": [[f"doc {self.ids[i]}" for i in rows]],
                  "metadatas": [[self.metadatas[i] for i in rows]],
                  "distances": [[self._distance(query, self.embeddings[i]) for i in rows]]}
        if self.return_embeddings and "embeddings" in include:
            result["embeddings"] = [[self.embeddings[i] for i in rows]]
        return result


def chroma_store(collection):
    store = ChromaVectorStore.__new__(ChromaVectorStore)   # no chromadb needed
    BaseVectorStore.__init__(store, dimension=2)
    store.cfg, store._collection, store._collection_name = offline_config(), collection, "test"
    store._count = len(collection.ids)
    return store


COSINES = {"a": 0.9, "b": 0.6, "c": 0.45, "d": 0.35, "e": 0.0}
VECTORS = {key: [c, math.sqrt(1.0 - c * c)] for key, c in COSINES.items()}


@pytest.mark.parametrize("space, return_embeddings", [
    (None, True), (None, False), ("cosine", False), ("ip", False),
])
def test_score_threshold_means_the_same_cosine_in_both_stores(space, return_embeddings):
    threshold = 0.7
    assert score_threshold_to_cosine(threshold) == pytest.approx(0.4)
    ids = list(VECTORS)
    memory_store = MemoryVectorStore(offline_config(), dimension=2)
    memory_store.add(texts=[f"doc {i}" for i in ids], embeddings=[VECTORS[i] for i in ids],
                     metadata=[{"scope": "s"} for _ in ids], ids=ids)
    chroma = chroma_store(FakeChromaCollection(ids, [VECTORS[i] for i in ids],
                                               [{"scope": "s"} for _ in ids], space, return_embeddings))
    query = [1.0, 0.0]
    from_memory = memory_store.search(query, top_k=5, score_threshold=threshold, filter_metadata={"scope": "s"})
    from_chroma = chroma.search(query, top_k=5, score_threshold=threshold, filter_metadata={"scope": "s"})
    # cos >= 0.40 in both; the former 1 / (1 + d) kept only "a" (cos >= 0.786) in Chroma.
    assert [r.id for r in from_memory] == [r.id for r in from_chroma] == ["a", "b", "c"]
    for mine, theirs in zip(from_memory, from_chroma):
        assert theirs.score == pytest.approx(mine.score, abs=1e-9)
        assert theirs.score == pytest.approx(cosine_to_score(COSINES[theirs.id]), abs=1e-9)


def test_default_threshold_keeps_the_former_chroma_strictness():
    # Chroma is the default provider. Its old score 1 / (1 + d), d = 2 - 2cos for unit
    # vectors, kept cos >= 0.786 at the old default 0.7; the default on the shared
    # scale keeps that cosine instead of loosening it to 0.40.
    old_chroma_cosine = 1.0 - (1.0 / 0.7 - 1.0) / 2.0
    default = CognitiveManagerConfig().memory
    assert default.vector_store_provider == "chroma"
    assert score_threshold_to_cosine(default.rag_score_threshold) == pytest.approx(old_chroma_cosine, abs=1e-3)
    ids = list(VECTORS) + ["f"]
    vectors = dict(VECTORS, f=[0.8, 0.6])                 # cos 0.8: kept before and now
    chroma = chroma_store(FakeChromaCollection(ids, [vectors[i] for i in ids], [{"scope": "s"} for _ in ids]))
    hits = chroma.search([1.0, 0.0], top_k=6, score_threshold=default.rag_score_threshold,
                         filter_metadata={"scope": "s"})
    assert [r.id for r in hits] == ["a", "f"]


def test_notes_are_searchable_scoped_and_clearable():
    memory = MemoryServiceV2(offline_config())
    note = "Kullanıcının adı Ayşe ve Python öğreniyor."
    with memory_scope("alice"):
        memory.add_note(note)
        assert [h["content"] for h in memory.retrieve_context("Ayşe")] == [note]
    with memory_scope("bob"):
        assert memory.retrieve_context("Ayşe") == []
    with memory_scope("alice"):
        memory.clear_notes()
        assert memory.retrieve_context("Ayşe") == []


def test_notes_are_found_with_vector_memory_on_and_stay_out_of_the_store():
    cfg = offline_config()
    cfg.memory.hybrid_search_alpha = 0.7
    cfg.memory.rag_score_threshold = 0.0
    memory = MemoryServiceV2(cfg)
    memory._vector_store = MemoryVectorStore(cfg, dimension=2)
    memory._embedding_adapter = SimpleNamespace(encode_single=lambda text: [1.0, 0.0])
    memory._vector_memory_enabled = True
    with memory_scope("alice"):
        memory.add_turn([], "user", "Bugün hava çok güzel.")   # embedded and stored
        memory.add_note("Kullanıcının adı Ayşe.")               # kept in process only
        hits = memory.retrieve_context("Ayşe", top_k=3)
    assert memory._vector_store.get_stats()["items_count"] == 1
    assert [(h["content"], h["role"]) for h in hits if h["role"] == "note"] == \
        [("Kullanıcının adı Ayşe.", "note")]


def test_cevahir_search_memory_finds_an_added_memory():
    from model.cevahir import Cevahir

    manager = CognitiveManager(FixedReply("cevap"), cfg=offline_config())
    facade = SimpleNamespace(_cognitive_manager=manager)
    Cevahir.add_memory(facade, "Kullanıcının adı Ayşe ve Python öğreniyor.")
    results = Cevahir.search_memory(facade, "Ayşe")
    assert [r["content"] for r in results] == ["Kullanıcının adı Ayşe ve Python öğreniyor."]


# --- P9: debate prompt -----------------------------------------------------------

def test_debate_prompts_name_their_perspective():
    model = FixedReply("Bir düşünce.")
    engine = DeliberationEngineV2(offline_config(), model)
    engine.generate_thoughts("Hangi yöntemi seçmeliyim?", num_thoughts=2)
    first, second = model.prompts
    assert "{perspective}" not in first + second
    assert "konservatif ve güvenli bir yaklaşım düşün" in first
    assert "yaratıcı ve esnek bir yaklaşım düşün" in second
    assert "KONSERVATİF VE GÜVENLİ PERSPECTIVE" in first and "YARATICI VE ESNEK PERSPECTIVE" in second
