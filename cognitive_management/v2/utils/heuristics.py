# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: heuristics.py
Modül: cognitive_management/v2/utils
Görev: V3 Heuristics — Gelişmiş özellik çıkarımı, sorgu sınıflandırması,
       alan tespiti ve politika yönlendirme yardımcıları.

       V3 yenilikleri:
         • classify_query_type()    — 6 sınıflı sorgu tipi tespiti
         • detect_domain()          — 8 alan sınıfı tespiti
         • compute_complexity_score()— çok-faktörlü karmaşıklık puanı
         • build_features()         — 14 özellik ile zenginleştirilmiş çıktı
         • mode_gates()             — query_type + domain farkında kapılar
         • estimate_risk()          — domain-aware çok-faktörlü risk skoru

MİMARİ:
- SOLID Prensipleri: Single Responsibility (heuristic utilities)
- Design Patterns: Utility Pattern (pure functions)
- Akademik Referanslar:
    • Wei et al. 2022 — CoT aktivasyonu için karmaşıklık eşiği
    • Wang et al. 2022 — Self-Consistency için entropy >= 2.0 önerisi
    • Yao et al. 2023 — ToT için uzun/karmaşık girdiler eşiği

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import re
from typing import Dict, Any, Tuple

from cognitive_management.config import CognitiveManagerConfig


# =============================================================================
# Sabitler — Anahtar Kelime Kümeleri
# =============================================================================

_MATH_KW = frozenset([
    # Türkçe
    "hesapla", "hesaplama", "kaç", "toplam", "fark", "çarp", "böl",
    "türev", "integral", "denklem", "matris", "logaritma", "karekök",
    "olasılık", "istatistik", "ortalama", "standart sapma", "yüzde",
    "oran", "kesir", "mutlak değer", "trigonometri", "geometri",
    # İngilizce
    "calculate", "compute", "sum", "multiply", "divide", "how many",
    "solve", "equation", "integral", "derivative", "matrix", "vector",
    "probability", "statistics", "mean", "variance", "logarithm",
    "square root", "trigonometry", "geometry", "algebra", "calculus",
])

_SCIENCE_KW = frozenset([
    # Türkçe
    "atom", "molekül", "hücre", "evrim", "fizik", "kimya", "biyoloji",
    "foton", "elektron", "nötron", "proton", "manyetik", "elektrik",
    "enerji", "kuvvet", "hız", "ivme", "reaksiyon", "enzim", "dna",
    "gen", "protein", "ekosistem", "termodinamik", "kuantum",
    # İngilizce
    "atom", "molecule", "cell", "evolution", "physics", "chemistry",
    "biology", "quantum", "relativity", "dna", "protein", "enzyme",
    "energy", "force", "velocity", "acceleration", "thermodynamics",
])

_LAW_KW = frozenset([
    # Türkçe
    "hukuk", "kanun", "yasa", "mahkeme", "dava", "suç", "ceza",
    "anayasa", "sözleşme", "hak", "özgürlük", "mevzuat", "tüzük",
    "yönetmelik", "avukat", "hakim", "savcı", "savunma", "dava",
    # İngilizce
    "law", "legal", "court", "case", "crime", "punishment", "contract",
    "constitution", "rights", "legislation", "regulation", "attorney",
    "judge", "prosecution", "defense", "liability", "jurisdiction",
])

_MEDICAL_KW = frozenset([
    # Türkçe
    "hastalık", "tedavi", "ilaç", "doktor", "sağlık", "belirti",
    "teşhis", "ameliyat", "terapi", "virüs", "bakteri", "enfeksiyon",
    "tanı", "semptom", "kronik", "akut", "alerji", "bağışıklık",
    # İngilizce
    "disease", "treatment", "medicine", "doctor", "health", "symptom",
    "diagnosis", "surgery", "therapy", "medication", "virus", "bacteria",
    "infection", "chronic", "acute", "allergy", "immunity", "vaccine",
])

_HISTORY_KW = frozenset([
    # Türkçe
    "tarih", "osmanlı", "cumhuriyet", "savaş", "imparatorluk", "antik",
    "yüzyıl", "dönem", "devrim", "uygarlık", "arkeoloji", "miras",
    # İngilizce
    "history", "ottoman", "empire", "ancient", "century", "period",
    "revolution", "civilization", "archaeology", "heritage", "dynasty",
    "medieval", "renaissance", "wwi", "wwii", "cold war",
])

_CREATIVE_KW = frozenset([
    # Türkçe
    "yaz", "oluştur", "tasarla", "hikaye", "şiir", "roman", "senaryo",
    "hayal", "kurgula", "tasvir", "anlatı", "karakter", "yaratıcı",
    "edebi", "kompozisyon", "öykü", "masal", "deneme",
    # İngilizce
    "write", "create", "design", "story", "poem", "novel", "script",
    "imagine", "narrative", "character", "creative", "literary",
    "compose", "tale", "essay", "fiction", "brainstorm",
])

_CODE_KW = frozenset([
    # Türkçe
    "kod", "program", "fonksiyon", "algoritma", "hata", "değişken",
    "döngü", "koşul", "sınıf", "metot", "veritabanı", "api",
    # İngilizce
    "code", "python", "javascript", "java", "function", "algorithm",
    "debug", "error", "class", "method", "api", "database", "sql",
    "html", "css", "react", "git", "deploy", "framework", "library",
    "c++", "rust", "golang", "typescript", "bash", "linux",
])

_FACTUAL_KW = frozenset([
    # Türkçe — soru kalıpları
    "nedir", "kimdir", "neredir", "ne zaman", "nasıl çalışır",
    "açıkla", "tanımla", "bilgi ver", "söyle", "anlat",
    "farkı nedir", "tarihçesi", "özellikleri",
    # İngilizce
    "what is", "who is", "where is", "when did", "how does",
    "explain", "define", "tell me about", "describe", "difference between",
])

_REASONING_KW = frozenset([
    # Türkçe — analiz / nedensellik
    "neden", "niçin", "nasıl", "analiz", "karşılaştır", "değerlendir",
    "sonuç", "öner", "tavsiye", "mantık", "sebep", "etki", "ilişki",
    "kanıtla", "destekle", "eleştir", "avantaj", "dezavantaj",
    # İngilizce
    "why", "analyze", "compare", "evaluate", "because", "therefore",
    "reason", "cause", "effect", "argue", "justify", "implications",
    "pros and cons", "advantages", "disadvantages", "critique",
])

_CONVERSATIONAL_KW = frozenset([
    # Türkçe
    "merhaba", "selam", "nasılsın", "teşekkür", "tamam", "anladım",
    "evet", "hayır", "lütfen", "rica", "güzel", "iyi", "harika",
    # İngilizce
    "hello", "hi", "hey", "how are you", "thanks", "thank you",
    "ok", "okay", "sure", "got it", "great", "nice", "awesome",
])

# Bağlaçlar — karmaşıklık hesabı için
_CONNECTIVES = frozenset([
    # Türkçe
    "ve", "veya", "ama", "çünkü", "ancak", "hem", "ne", "ya",
    "fakat", "oysa", "rağmen", "karşın", "dolayısıyla", "böylece",
    # İngilizce
    "and", "or", "but", "because", "however", "although",
    "unless", "whereas", "therefore", "consequently", "moreover",
])

# İç-içe yapı işaretçileri — karmaşıklık hesabı için
_CLAUSE_MARKERS = frozenset([
    "ki", "that", "which", "who", "where", "when", "while",
    "although", "whereas", "even if", "so that",
])

# Sayısal ifade deseni — matematiksel sorgu tespiti için
_MATH_EXPR = re.compile(
    r'\b\d+\s*[\+\-\*/\^=<>]\s*\d+\b'   # aritmetik: 3+4, x=5
    r'|√\d+'                              # karekök: √9
    r'|\d+\s*[²³]'                        # üs: x²
    r'|[a-z]\s*=\s*\d'                    # değişken ataması: x=3
    r'|\d+%',                             # yüzde: 50%
    re.IGNORECASE,
)


# =============================================================================
# Yardımcı Fonksiyonlar
# =============================================================================

def _lower(s: str | None) -> str:
    return (s or "").strip().lower()


def _contains_any(text: str, needles: frozenset | Tuple[str, ...]) -> bool:
    if not text or not needles:
        return False
    t = _lower(text)
    return any(n in t for n in needles)


def _count_matches(text: str, needles: frozenset) -> int:
    t = _lower(text)
    return sum(1 for n in needles if n in t)


def _token_len(text: str) -> int:
    return len((text or "").split())


# =============================================================================
# Sorgu Sınıflandırması — classify_query_type()
# =============================================================================

def classify_query_type(text: str) -> str:
    """
    Kullanıcı mesajını 6 sorgu tipinden birine atar.

    Yöntem: Her kategori için anahtar kelime sayımı + ağırlıklı puanlama.
    Matematiksel ifade regex'i matematik kategorisine 3 puan bonus verir.

    Returns:
        QueryType literal: "factual" | "reasoning" | "creative" |
                           "conversational" | "math" | "code" | "unknown"
    """
    if not text or not text.strip():
        return "unknown"

    # Ağırlıklı puanlar (yüksek ağırlık = daha güçlü sinyal)
    scores: Dict[str, float] = {
        "math":           _count_matches(text, _MATH_KW) * 2.0,
        "code":           _count_matches(text, _CODE_KW) * 2.0,
        "creative":       _count_matches(text, _CREATIVE_KW) * 1.5,
        "factual":        _count_matches(text, _FACTUAL_KW) * 1.0,
        "reasoning":      _count_matches(text, _REASONING_KW) * 1.0,
        "conversational": _count_matches(text, _CONVERSATIONAL_KW) * 2.0,
    }

    # Matematiksel ifade içeriyorsa güçlü sinyal
    if _MATH_EXPR.search(text):
        scores["math"] += 3.0

    # Sorgu kısaysa ve sohbet kelimesi varsa → conversational baskın olsun
    if _token_len(text) <= 5 and scores["conversational"] > 0:
        scores["conversational"] += 1.5

    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "unknown"


# =============================================================================
# Alan Tespiti — detect_domain()
# =============================================================================

def detect_domain(text: str) -> str:
    """
    Kullanıcı mesajının ait olduğu alanı tespit eder.

    Returns:
        DomainType literal: "math" | "science" | "law" | "medical" |
                            "technology" | "history" | "creative" | "general"
    """
    if not text or not text.strip():
        return "general"

    domains: Dict[str, float] = {
        "math":       _count_matches(text, _MATH_KW) * 2.0,
        "science":    _count_matches(text, _SCIENCE_KW) * 1.5,
        "law":        _count_matches(text, _LAW_KW) * 1.5,
        "medical":    _count_matches(text, _MEDICAL_KW) * 1.5,
        "technology": _count_matches(text, _CODE_KW) * 1.5,
        "history":    _count_matches(text, _HISTORY_KW) * 1.0,
        "creative":   _count_matches(text, _CREATIVE_KW) * 1.0,
    }

    if _MATH_EXPR.search(text):
        domains["math"] += 2.0

    best = max(domains, key=lambda k: domains[k])
    return best if domains[best] > 0 else "general"


# =============================================================================
# Karmaşıklık Puanı — compute_complexity_score()
# =============================================================================

def compute_complexity_score(text: str, features: Dict[str, Any]) -> float:
    """
    Çok-faktörlü karmaşıklık puanı (0.0–1.0).

    Faktörler:
        • Kelime sayısı        (0.30 ağırlık)
        • Birden fazla soru    (0.15 ağırlık)
        • Bağlaç zenginliği   (0.20 ağırlık)
        • İç-içe cümle        (0.15 ağırlık)
        • Hassas alan         (0.10 ağırlık)
        • İddia içeriği       (0.10 ağırlık)

    Akademik not: Wei et al. 2022, karmaşık görevlerde CoT aktivasyonunun
    doğruluğu artırdığını; Yao et al. 2023 ise özellikle çok-adımlı
    akıl yürütme gerektiren sorularda ToT'un üstün olduğunu gösterir.
    """
    score = 0.0
    text_lower = _lower(text)
    words = text.split()

    # 1. Kelime sayısı faktörü
    n_words = len(words)
    length_factor = min(1.0, n_words / 60.0)  # 60+ kelime = maksimum
    score += length_factor * 0.30

    # 2. Birden fazla soru
    n_questions = text.count("?")
    if n_questions > 1:
        score += min(0.15, (n_questions - 1) * 0.05)

    # 3. Bağlaç zenginliği
    n_connectives = sum(1 for c in _CONNECTIVES if f" {c} " in text_lower)
    score += min(0.20, n_connectives * 0.04)

    # 4. İç-içe yapı (ki, that, which…)
    n_clauses = sum(1 for m in _CLAUSE_MARKERS if f" {m} " in text_lower)
    score += min(0.15, n_clauses * 0.05)

    # 5. Hassas alan
    if features.get("is_sensitive_domain"):
        score += 0.10

    # 6. İddia içeriği
    if features.get("has_claims"):
        score += 0.10

    return min(1.0, score)


# =============================================================================
# Özellik Vektörü — build_features()
# =============================================================================

def build_features(
    cfg: CognitiveManagerConfig,
    *,
    user_message: str,
    entropy_est: float | None = None,
) -> Dict[str, Any]:
    """
    Kullanıcı mesajından zenginleştirilmiş özellik sözlüğü üretir.

    V3 Yenilikleri (önceki → şimdi):
        6 özellik → 14 özellik
        Entropi-only → QueryType + DomainType + complexity_score eklendi

    Döndürülen anahtarlar:
        input_length          : kelime sayısı
        entropy_est           : belirsizlik kestirimi (0.0–3.0 ölçek)
        needs_recent_info     : güncel bilgi tetikleyicisi
        needs_calc_or_parse   : hesaplama/ayrıştırma tetikleyicisi
        has_claims            : iddia marker'ı
        is_sensitive_domain   : hassas alan (hukuk, tıp, siyaset…)
        query_type            : QueryType string
        domain                : DomainType string
        complexity_score      : 0.0–1.0 karmaşıklık puanı
        is_math_query         : bool kısayol
        is_creative_query     : bool kısayol
        is_conversational     : bool kısayol
        is_code_query         : bool kısayol
        n_question_marks      : soru işareti sayısı
    """
    msg = _lower(user_message)
    f_rules = cfg.features.tool_rules
    s_rules = cfg.safety

    # Araç tetikleyicileri
    needs_recent = _contains_any(msg, tuple(f_rules.get("needs_recent_info_triggers", ())))
    needs_calc   = _contains_any(msg, tuple(f_rules.get("needs_calc_or_parse_triggers", ())))

    # Risk / iddia sinyalleri
    has_claim    = _contains_any(msg, s_rules.claim_markers)
    is_sensitive = _contains_any(msg, s_rules.risk_keywords_sensitive)

    # Temel metrikler
    tokens = _token_len(user_message)
    n_q    = user_message.count("?")

    # Temel özellik sözlüğü (karmaşıklık için gerekli)
    base_feats: Dict[str, Any] = {
        "input_length":        tokens,
        "entropy_est":         float(entropy_est or 0.0),
        "needs_recent_info":   needs_recent,
        "needs_calc_or_parse": needs_calc,
        "has_claims":          has_claim,
        "is_sensitive_domain": is_sensitive,
        "n_question_marks":    n_q,
    }

    # V3: Sorgu tipi ve alan tespiti
    qtype  = classify_query_type(user_message)
    domain = detect_domain(user_message)

    # V3: Karmaşıklık puanı
    complexity = compute_complexity_score(user_message, base_feats)

    return {
        **base_feats,
        # V3 yeni alanlar
        "query_type":       qtype,
        "domain":           domain,
        "complexity_score": complexity,
        # Pratik bool kısayollar (router'da şartlı kullanım için)
        "is_math_query":       (qtype == "math"),
        "is_creative_query":   (qtype == "creative"),
        "is_conversational":   (qtype == "conversational"),
        "is_code_query":       (qtype == "code"),
    }


# =============================================================================
# Risk Skoru — estimate_risk()
# =============================================================================

def estimate_risk(cfg: CognitiveManagerConfig, features: Dict[str, Any]) -> float:
    """
    Domain-aware çok-faktörlü risk skoru (0.0–1.0).

    Faktörler:
        • İddia marker'ı       → +0.40 (gerçek doğrulama gerekir)
        • Hassas alan          → +0.40 (hukuk, tıp, siyaset)
        • Tıbbi/hukuki alan   → +0.20 ek bonus
        • Entropi yüksekliği   → +0.10 * normalize(entropy)
        • Karmaşıklık skoru    → +0.10 * complexity

    Akademik not: Tıbbi ve hukuki alanlar hata maliyeti yüksek olduğundan
    risk skoru yüksek tutulur; bu da critic katmanını daha sık aktive eder.
    """
    r     = cfg.features.risk_rules
    base  = float(r.get("base_risk", 0.0))
    score = base

    # İddia riski
    if features.get("has_claims"):
        score += float(r.get("claim_bonus", 0.40))

    # Hassas alan riski
    if features.get("is_sensitive_domain"):
        score += float(r.get("sensitive_bonus", 0.40))

    # Domain ek riski: tıp ve hukuk özellikle hassas
    domain = features.get("domain", "general")
    if domain in ("medical", "law"):
        score += 0.20

    # Entropi belirsizlik riski
    try:
        ent    = float(features.get("entropy_est", 0.0))
        score += 0.10 * max(0.0, min(ent, 3.0)) / 3.0
    except Exception:
        pass

    # Karmaşıklık riski
    try:
        complexity = float(features.get("complexity_score", 0.0))
        score     += 0.10 * complexity
    except Exception:
        pass

    return max(0.0, min(float(r.get("max_risk", 1.0)), score))


# =============================================================================
# Araç Kararı — should_tool()
# =============================================================================

def should_tool(cfg: CognitiveManagerConfig, features: Dict[str, Any]) -> str:
    """
    Araç kullanımına yönelik öneri: "none" | "maybe" | "must"

    Kural:
        güncel bilgi gerekiyor → "must"
        hesaplama / parsing   → "maybe" (veya math query ise "must")
        diğer                 → config default
    """
    if not cfg.tools.enable_tools:
        return "none"

    if features.get("needs_recent_info"):
        return "must"

    # Matematik sorgusu + hesaplama tetikleyicisi → must
    if features.get("is_math_query") and features.get("needs_calc_or_parse"):
        return "must"

    if features.get("needs_calc_or_parse"):
        return "maybe"

    return str(cfg.features.tool_rules.get("default_decision", "none"))


# =============================================================================
# Mod Kapıları — mode_gates()
# =============================================================================

def mode_gates(cfg: CognitiveManagerConfig, features: Dict[str, Any]) -> Dict[str, bool]:
    """
    PolicyRouter'ın kullandığı binary kapılar.

    V3 Yenilikleri:
        • query_type farkındası: math/code → think_gate her zaman açık
        • domain farkındası: medical/law → risk skoru düşük olsa da think_gate açılır
        • creative → debate_gate kapatılır (yaratıcı içerikte debate anlamsız)
        • self_consistency_gate: entropy ortasında, birden fazla tur gerekebilir

    Kapılar:
        think_gate:           Chain-of-Thought aktifleştirmesi (Wei et al. 2022)
        debate_gate:          İki aday + Self-Consistency (Wang et al. 2022)
        tot_gate:             Tree of Thoughts (Yao et al. 2023)
        self_consistency_gate:N örneklem, çoğunluk oylaması

    Akademik not:
        • think_gate  → entropy >= 1.5 (Wang et al. önerir)
        • debate_gate → entropy >= 2.5 veya input_length >= 200
        • tot_gate    → entropy >= 3.0 veya input_length >= 300
    """
    ent        = float(features.get("entropy_est", 0.0))
    length     = int(features.get("input_length", 0))
    query_type = features.get("query_type", "unknown")
    domain     = features.get("domain", "general")
    complexity = float(features.get("complexity_score", 0.0))

    # --- Think Gate ---
    # Standart eşik: entropy >= threshold
    think_gate = ent >= cfg.policy.entropy_gate_think
    # Math/code → her zaman CoT (akıl yürütme adımları gerekir)
    if query_type in ("math", "code"):
        think_gate = True
    # Tıp/hukuk → riski düşük olsa da CoT aktif
    if domain in ("medical", "law"):
        think_gate = True
    # Yüksek karmaşıklık → CoT
    if complexity >= 0.5:
        think_gate = True

    # --- Debate Gate ---
    # Yaratıcı sorular için debate anlamsız (doğru/yanlış yok)
    if query_type == "creative":
        debate_gate = False
    else:
        debate_gate = (
            cfg.policy.debate_enabled
            and (
                ent    >= cfg.policy.entropy_gate_debate
                or length >= cfg.policy.length_gate_debate
                or complexity >= 0.65
            )
        )

    # --- ToT Gate ---
    tot_gate = (
        cfg.policy.tot_enabled
        and (
            ent    >= cfg.policy.entropy_gate_tot
            or length >= cfg.policy.length_gate_tot
            or complexity >= 0.80
        )
    )
    # Math veya code ile çok uzun girdi → ToT
    if query_type in ("math", "code") and length >= 50:
        tot_gate = cfg.policy.tot_enabled

    # --- Self-Consistency Gate ---
    # Orta entropy aralığında (1.5–2.5) özellikle yararlı
    sc_enabled = getattr(cfg.policy, "self_consistency_enabled", False)
    self_consistency_gate = (
        sc_enabled
        and cfg.policy.entropy_gate_think <= ent < cfg.policy.entropy_gate_debate
        and not (query_type in ("conversational", "creative"))
    )

    return {
        "think_gate":            think_gate,
        "debate_gate":           debate_gate,
        "tot_gate":              tot_gate,
        "self_consistency_gate": self_consistency_gate,
    }


# =============================================================================
# Dışa Aktarım
# =============================================================================

__all__ = [
    "classify_query_type",
    "detect_domain",
    "compute_complexity_score",
    "build_features",
    "estimate_risk",
    "should_tool",
    "mode_gates",
]
