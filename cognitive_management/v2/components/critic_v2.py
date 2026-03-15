# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: critic_v2.py
Modül: cognitive_management/v2/components
Görev: V3 Critic — Akademik ve endüstri standardında çok-boyutlu çıktı
       değerlendirmesi ve Self-Refine döngüsü.

       V3 Yenilikleri:
         • _check_task_match()    — Kullanıcı niyeti ile yanıt örtüşmesi
                                    (intent keyword recall)
         • _check_coherence()    — N-gram örtüşme tabanlı cümle tutarlılığı
         • _check_relevance()    — TF-benzeri ağırlıklı kelime benzerliği
                                    (stopword filtrelemeli)
         • _check_length()       — Adaptif uzunluk (sorgu tipine göre)
         • CriticFeedback list   — Yapılandırılmış geribildirim
         • _last_feedback / _last_passes — Handler entegrasyonu için izleme
         • İyileştirilmiş self-refine prompt (few-shot tarzı, rollü)
         • Constitutional AI entegrasyonu korundu (Bai et al. 2022)

       Akademik Referanslar:
         • Madaan et al. 2023 — Self-Refine: Iterative Refinement with Self-Feedback
         • Bai et al. 2022    — Constitutional AI: Harmlessness from AI Feedback
         • Gou et al. 2023    — CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing

MİMARİ:
- SOLID Prensipleri: SRP, Dependency Inversion (interface'lere bağımlı)
- Akademik Standart: Çok-boyutlu değerlendirme + model tabanlı revizyon

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Protocol, Tuple

from cognitive_management.cognitive_types import CriticFeedback, DecodingConfig
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError
from cognitive_management.v2.interfaces.component_protocols import Critic as ICritic

# Phase 7.2: External Fact-Checking
from cognitive_management.v2.components.fact_checkers import (
    create_fact_checkers,
    FactChecker,
    FactCheckResult,
)
from cognitive_management.v2.utils.claim_extraction import (
    extract_claims,
    ExtractedClaim,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model API Protocol
# =============================================================================

class ModelAPI(Protocol):
    def generate(self, prompt: str, decoding_cfg: DecodingConfig) -> str: ...
    def score(self, prompt: str, candidate: str) -> float: ...


# =============================================================================
# Sabitler — Türkçe + İngilizce Stopword Listesi
# =============================================================================

_STOPWORDS = frozenset([
    # Türkçe
    "bir", "bu", "şu", "o", "ve", "ile", "de", "da", "ki", "mi",
    "mu", "mü", "mı", "için", "ama", "ya", "veya", "hem", "ne",
    "gibi", "kadar", "daha", "çok", "en", "her", "hiç", "bazı",
    "olan", "olur", "oldu", "olsa", "ise", "idi", "iyi", "kötü",
    "var", "yok", "ben", "sen", "biz", "siz",
    # İngilizce
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "may", "might", "shall", "should", "must",
    "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "by", "with", "from", "that", "this", "it", "its",
])

# Zararlı içerik uyarı kelimeleri (Türkçe + İngilizce)
_HARM_PATTERNS = re.compile(
    r'\b(zararlı|tehlikeli|yasadışı|şiddet|nefret|saldırı|taciz'
    r'|harmful|dangerous|illegal|violence|hate|attack|harassment'
    r'|bombala|öldür|zarar ver|incit)\b',
    re.IGNORECASE | re.UNICODE,
)

# Soru tipi tespit
_MATH_Q = re.compile(r'\b(kaç|hesapla|toplam|çarp|böl|türev|integrate|calculate|how many|solve)\b', re.IGNORECASE)
_CREATIVE_Q = re.compile(r'\b(yaz|oluştur|hikaye|şiir|roman|write|create|story|poem|script)\b', re.IGNORECASE)


# =============================================================================
# Yardımcı Fonksiyonlar
# =============================================================================

def _tokenize(text: str) -> List[str]:
    """Noktalama işaretlerini temizle, küçük harf yap, token listesi döndür."""
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return [w for w in text.lower().split() if w and w not in _STOPWORDS and len(w) > 1]


def _bigrams(tokens: List[str]) -> Counter:
    return Counter(zip(tokens, tokens[1:]))


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _tf_idf_sim(text_a: str, text_b: str) -> float:
    """
    Basitleştirilmiş TF-ağırlıklı kelime örtüşme benzerliği.
    İki metin arasındaki anlamsal benzerlik için kullanılır.
    Tam TF-IDF gerektirmez; corpussuz yaklaşım yeterlidir.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)

    # Normalize TF
    max_a = max(tf_a.values(), default=1)
    max_b = max(tf_b.values(), default=1)

    sim = 0.0
    all_words = set(tf_a.keys()) | set(tf_b.keys())
    for w in all_words:
        wa = tf_a.get(w, 0) / max_a
        wb = tf_b.get(w, 0) / max_b
        sim += min(wa, wb)

    norm = sum(v / max_a for v in tf_a.values())
    if norm == 0:
        return 0.0

    return min(1.0, sim / norm)


# =============================================================================
# CriticV2 — V3 Çok-Boyutlu Critic
# =============================================================================

class CriticV2(ICritic):
    """
    V3 Critic: Akademik standartlarda çok-boyutlu yanıt değerlendirmesi.

    Değerlendirme Boyutları (ağırlıklı):
        task_match  (0.30) — Kullanıcı niyeti ile yanıt örtüşmesi
        relevance   (0.25) — TF-ağırlıklı semantik ilgililik
        coherence   (0.20) — N-gram örtüşme tabanlı cümle akışı
        safety      (0.15) — Zararlı içerik tespiti
        factuality  (0.10) — Gerçek doğrulama (temel veya harici)

    Self-Refine (Madaan et al. 2023):
        • max_passes iterasyon: değerlendir → geri bildirim → revize
        • Model tabanlı revizyon (model generate() çağrısı)
        • Yapılandırılmış ve rollü prompt mühendisliği

    Constitutional AI (Bai et al. 2022):
        • İlk geçişte constitutional kontrol yapılır
        • Kural ihlali varsa revize edilir, sonra Self-Refine başlar
    """

    # Boyut ağırlıkları (toplamı 1.0)
    _WEIGHTS: Dict[str, float] = {
        "task_match": 0.30,
        "relevance":  0.25,
        "coherence":  0.20,
        "safety":     0.15,
        "factuality": 0.10,
    }

    def __init__(self, cfg: CognitiveManagerConfig, model_api: ModelAPI):
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        for fn in ("generate", "score"):
            if not hasattr(model_api, fn):
                raise ValidationError(f"ModelAPI '{fn}' metodunu sağlamıyor.")

        self.cfg = cfg
        self.mm = model_api

        # External fact-checkers
        self._fact_checkers: List[FactChecker] = []
        if cfg.critic.enable_external_fact_checking:
            try:
                self._fact_checkers = create_fact_checkers(cfg)
            except Exception as e:
                logger.warning(f"Fact checker başlatılamadı: {e}")

        # Constitutional AI (lazy)
        self._constitutional_critic = None

        # Handler entegrasyonu için izleme
        self._last_feedback: Optional[List[CriticFeedback]] = None
        self._last_passes: int = 0

    # =========================================================================
    # Ana Giriş Noktası — review()
    # =========================================================================

    def review(
        self,
        user_message: str,
        draft_text: str,
        context: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Self-Refine döngüsü (Madaan et al. 2023).

        Akış:
            1. Constitutional AI kontrolü (ihlal → revizyon)
            2. Çok-boyutlu değerlendirme
            3. Revizyon gerekirrse: geribildirim üret → model revize et
            4. max_passes dolduktan veya yeterince iyi olunca dur

        Args:
            user_message: Kullanıcının orijinal sorusu.
            draft_text:   Değerlendirilecek taslak yanıt.
            context:      Opsiyonel bağlam metni (geçmiş + RAG).

        Returns:
            (final_text, was_revised)
        """
        if not self.cfg.critic.enabled:
            return draft_text, False

        if not draft_text or not draft_text.strip():
            return draft_text, False

        current_text = draft_text
        was_revised   = False
        self._last_feedback = None
        self._last_passes   = 0

        # --- Adım 1: Constitutional AI ---
        if self.cfg.critic.enable_constitutional_ai:
            try:
                if self._constitutional_critic is None:
                    from cognitive_management.v2.components.constitutional_critic import ConstitutionalCritic
                    self._constitutional_critic = ConstitutionalCritic(self.cfg, self.mm)

                constitutional_result, violations = self._constitutional_critic.review_with_constitution(
                    user_message=user_message,
                    draft_text=current_text,
                )
                if violations.get("revised", False) and constitutional_result:
                    current_text = constitutional_result
                    was_revised  = True
                    logger.debug("Constitutional revision uygulandı.")
            except Exception as e:
                logger.warning(f"Constitutional review başarısız: {e}")

        # --- Adım 2–N: Self-Refine Döngüsü ---
        max_iters = max(1, self.cfg.critic.max_passes)

        for iteration in range(max_iters):
            feedback_list = self._evaluate_all(user_message, current_text, context)
            self._last_feedback = feedback_list
            self._last_passes   = iteration + 1

            needs_revision = self._should_revise(feedback_list)

            if not needs_revision:
                break

            revised_text = self._revise_text(
                user_message, current_text, feedback_list, context
            )

            if revised_text and revised_text.strip() and revised_text != current_text:
                current_text = revised_text
                was_revised  = True
            else:
                break  # Değişim yok → dur

        return current_text, was_revised

    # =========================================================================
    # Çok-Boyutlu Değerlendirme
    # =========================================================================

    def _evaluate_all(
        self,
        user_message: str,
        draft_text: str,
        context: Optional[str],
    ) -> List[CriticFeedback]:
        """
        Tüm boyutlar için CriticFeedback listesi üretir.
        """
        feedback_list: List[CriticFeedback] = []

        # 1. Task Match
        tm_score = self._check_task_match(user_message, draft_text)
        feedback_list.append(CriticFeedback(
            aspect="task_match",
            score=tm_score,
            message=self._tm_message(tm_score),
            needs_revision=(tm_score < 0.45),
        ))

        # 2. Relevance
        rel_score = self._check_relevance(user_message, draft_text)
        feedback_list.append(CriticFeedback(
            aspect="relevance",
            score=rel_score,
            message=self._rel_message(rel_score),
            needs_revision=(rel_score < 0.35),
        ))

        # 3. Coherence
        coh_score = self._check_coherence(draft_text)
        feedback_list.append(CriticFeedback(
            aspect="coherence",
            score=coh_score,
            message=self._coh_message(coh_score),
            needs_revision=(coh_score < 0.50),
        ))

        # 4. Safety
        saf_score = self._check_safety(user_message, draft_text)
        feedback_list.append(CriticFeedback(
            aspect="safety",
            score=saf_score,
            message=self._saf_message(saf_score),
            needs_revision=(saf_score < 0.70),
            constitutional=(saf_score < 0.40),
        ))

        # 5. Factuality
        fact_score = self._check_facts(draft_text)
        feedback_list.append(CriticFeedback(
            aspect="factuality",
            score=fact_score,
            message=self._fact_message(fact_score),
            needs_revision=(fact_score < 0.75),
        ))

        return feedback_list

    # =========================================================================
    # Boyut 1: Task Match
    # =========================================================================

    def _check_task_match(self, user_message: str, draft_text: str) -> float:
        """
        Kullanıcı sorusunun ana niyet kelimelerinin yanıtta ne kadar
        karşılandığını ölçer (intent keyword recall).

        Yöntem:
            • Sorgudan stopword filtreli token seti oluştur
            • Bu tokenların yanıtta kaçı geçiyor → recall
            • Bonus: Yanıt soru tipiyle uyumlu mu?
                     (math sorusu → sayı var mı? creative → paragraf var mı?)
        """
        q_tokens = set(_tokenize(user_message))
        a_tokens = set(_tokenize(draft_text))

        if not q_tokens:
            return 0.7  # Soru belirsiz, varsayılan

        recall = len(q_tokens & a_tokens) / len(q_tokens)

        # Yapısal uyum bonusu
        bonus = 0.0
        if _MATH_Q.search(user_message):
            # Matematik sorusu: yanıtta rakam olmalı
            if re.search(r'\d', draft_text):
                bonus += 0.15
        elif _CREATIVE_Q.search(user_message):
            # Yaratıcı yazı: yanıt en az 3 cümle olmalı
            n_sent = len(re.findall(r'[.!?]+', draft_text))
            if n_sent >= 3:
                bonus += 0.10

        return min(1.0, recall * 0.85 + bonus)

    def _tm_message(self, score: float) -> str:
        if score >= 0.75:
            return "Yanıt kullanıcının sorusunu iyi karşılıyor."
        elif score >= 0.45:
            return "Yanıt kısmen ilgili; kullanıcının asıl sorusuna daha doğrudan odaklan."
        else:
            return "Yanıt kullanıcının sorusundan belirgin şekilde sapıyor; soruyu yeniden oku ve odaklan."

    # =========================================================================
    # Boyut 2: Relevance
    # =========================================================================

    def _check_relevance(self, user_message: str, draft_text: str) -> float:
        """
        TF-ağırlıklı kelime benzerliği + bigram örtüşmesi.

        Stopword filtreli, böylece "bir", "ve" gibi boş sözcükler
        sahte yüksek benzerlik puanı vermez.
        """
        if not user_message or not draft_text:
            return 0.50

        tf_sim  = _tf_idf_sim(user_message, draft_text)

        # Bigram örtüşmesi (sözdizimsel tutarlılık)
        bg_q = _bigrams(_tokenize(user_message))
        bg_a = _bigrams(_tokenize(draft_text))
        bg_overlap = 0.0
        if bg_q:
            common = sum((bg_q & bg_a).values())
            bg_overlap = common / max(sum(bg_q.values()), 1)

        # Ağırlıklı birleşim
        return min(1.0, tf_sim * 0.70 + bg_overlap * 0.30)

    def _rel_message(self, score: float) -> str:
        if score >= 0.60:
            return "Yanıt kullanıcı sorusuyla yeterince ilgili."
        elif score >= 0.35:
            return "Yanıt kısmen ilgili; sorunun ana kavramlarına daha çok değin."
        else:
            return "Yanıt soruyla çok az ilgili; tamamen farklı bir konuya kayılmış olabilir."

    # =========================================================================
    # Boyut 3: Coherence
    # =========================================================================

    def _check_coherence(self, text: str) -> float:
        """
        Cümle-içi ve cümleler-arası tutarlılık.

        Yöntem:
            • Cümleleri böl
            • Ardışık cümleler arasında kelime örtüşmesi hesapla
            • Düşük örtüşme → sert konu değişimi → tutarsızlık
            • Ekstra: Min cümle uzunluğu, noktalama varlığı
        """
        if not text or len(text.strip()) < 10:
            return 0.30

        # Cümlelere böl
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        if len(sentences) == 1:
            # Tek cümle: noktalama + uzunluk kontrolü
            score = 0.55
            if any(c in text for c in ".!?"):
                score += 0.20
            if 10 <= len(text.split()) <= 80:
                score += 0.15
            return min(1.0, score)

        # Ardışık cümle örtüşmesi
        overlaps = []
        for i in range(len(sentences) - 1):
            tok_a = set(_tokenize(sentences[i]))
            tok_b = set(_tokenize(sentences[i + 1]))
            if tok_a and tok_b:
                overlaps.append(_jaccard(tok_a, tok_b))

        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

        # Çok yüksek örtüşme (tekrar), çok düşük (kopukluk) her ikisi de kötü
        # Optimal: 0.05–0.35 arası
        if 0.05 <= mean_overlap <= 0.35:
            coh_score = 0.75 + (0.25 * (mean_overlap / 0.35))
        elif mean_overlap > 0.35:
            # Aşırı tekrar
            coh_score = max(0.50, 1.0 - (mean_overlap - 0.35) * 1.5)
        else:
            # Çok kopuk
            coh_score = max(0.30, mean_overlap * 5.0)

        # Noktalama varlığı bonusu
        has_punct = any(c in text for c in ".!?")
        if has_punct:
            coh_score += 0.05

        return min(1.0, coh_score)

    def _coh_message(self, score: float) -> str:
        if score >= 0.70:
            return "Yanıt tutarlı ve iyi organize edilmiş."
        elif score >= 0.50:
            return "Yanıtta küçük tutarsızlıklar var; cümleler arasındaki geçişleri pürüzsüzleştir."
        else:
            return "Yanıt tutarsız veya parçalı; fikirler arasında mantıksal bir bağ kur."

    # =========================================================================
    # Boyut 4: Safety
    # =========================================================================

    def _check_safety(self, user_message: str, draft_text: str) -> float:
        """
        Zararlı içerik tespiti (regex tabanlı, genişletilebilir).
        """
        harm_count = len(_HARM_PATTERNS.findall(draft_text))
        if harm_count == 0:
            return 1.0
        elif harm_count == 1:
            return 0.65
        elif harm_count <= 3:
            return 0.35
        else:
            return 0.10

    def _saf_message(self, score: float) -> str:
        if score >= 0.80:
            return "Yanıt güvenli ve uygun içerik barındırıyor."
        elif score >= 0.50:
            return "Yanıt bazı hassas ifadeler içeriyor; ifadeyi daha dikkatli ve nötr bir dille yeniden yaz."
        else:
            return "Yanıt zararlı veya uygunsuz içerik barındırıyor; tamamen uygun içerik üret."

    # =========================================================================
    # Boyut 5: Factuality
    # =========================================================================

    def _check_facts(self, text: str) -> float:
        """
        Gerçek doğrulama:
            • Harici fact-checker varsa → _check_facts_enhanced()
            • Yoksa → belirsizlik belirteci heuristiği
        """
        if self.cfg.critic.enable_external_fact_checking and self._fact_checkers:
            return self._check_facts_enhanced(text)
        return self._check_facts_basic(text)

    def _check_facts_basic(self, text: str) -> float:
        text_lower = text.lower()
        # Doğrulanmamış sinyal
        unverified = ["kaynak yok", "doğrulanmamış", "unverified", "no source"]
        if any(u in text_lower for u in unverified):
            return 0.50

        # İddia + belirsizlik = iyi (hedging)
        claim_markers  = ["%", "oran", "istatistik", "araştırma", "çalışma",
                          "percent", "statistics", "study", "research"]
        uncert_markers = ["belki", "muhtemelen", "sanırım", "olabilir",
                          "maybe", "perhaps", "possibly", "might"]

        has_claims  = any(m in text_lower for m in claim_markers)
        has_uncert  = any(m in text_lower for m in uncert_markers)

        if has_claims and has_uncert:
            return 0.88   # İyi: iddialı ama temkinli
        elif has_claims:
            return 0.72   # Kabul edilebilir: iddialı, kanıtlanmamış
        return 1.00       # İddia yok: nötr metin

    def _check_facts_enhanced(self, text: str) -> float:
        """Harici fact-checker + LLM doğrulama (Phase 7.2)."""
        claims = extract_claims(text, min_confidence=self.cfg.critic.claim_extraction_min_confidence)
        if not claims:
            return 1.0

        verified_count = 0
        for claim_obj in claims:
            for checker in self._fact_checkers:
                try:
                    result = checker.verify(claim_obj.claim)
                    if result and result.is_verified is True:
                        verified_count += 1
                        break
                except Exception:
                    continue

        base_score = (
            0.70 + (verified_count / len(claims)) * 0.30
            if verified_count > 0
            else 0.60
        )

        if self.cfg.critic.enable_llm_fact_verification:
            llm_score = self._check_facts_llm(text, claims)
            return (base_score * 0.60) + (llm_score * 0.40)

        return base_score

    def _check_facts_llm(self, text: str, claims: List[ExtractedClaim]) -> float:
        """LLM tabanlı gerçek doğrulama."""
        if not claims:
            return 1.0
        try:
            claims_text = "\n".join(f"- {c.claim}" for c in claims[:3])
            prompt = (
                f"Aşağıdaki metinde geçen iddiaları değerlendir.\n\n"
                f"Metin (ilk 400 karakter):\n{text[:400]}\n\n"
                f"İddialar:\n{claims_text}\n\n"
                f"Bu iddialar genel olarak DOĞRU, YANLIŞ, KISMEN DOĞRU veya BELİRSİZ mi?\n"
                f"Tek kelimeyle yanıtla."
            )
            response = self.mm.generate(
                prompt,
                DecodingConfig(max_new_tokens=20, temperature=0.20, top_p=0.80),
            )
            rl = (response or "").lower()
            if "doğru" in rl or "correct" in rl or "true" in rl:
                return 0.90
            if "kısmen" in rl or "partial" in rl:
                return 0.70
            if "belirsiz" in rl or "uncertain" in rl or "unknown" in rl:
                return 0.55
            if "yanlış" in rl or "false" in rl or "incorrect" in rl:
                return 0.20
            return 0.60
        except Exception as e:
            logger.debug(f"LLM fact verification başarısız: {e}")
            return 0.60

    def _fact_message(self, score: float) -> str:
        if score >= 0.85:
            return "Yanıt gerçek içerik açısından yeterli görünüyor."
        elif score >= 0.65:
            return "Yanıt doğrulanmamış iddialar içerebilir; belirsizliği belirt veya 'muhtemelen' gibi hedging ekle."
        else:
            return "Yanıt doğruluğu şüpheli iddialara sahip; varsayımlarını açıkça ifade et veya yanıtı yeniden yaz."

    # =========================================================================
    # Revizyon Kararı
    # =========================================================================

    def _should_revise(self, feedback_list: List[CriticFeedback]) -> bool:
        """
        Herhangi bir boyut revizyon gerektiriyorsa True döner.

        Ek kural: Ağırlıklı ortalama < 0.65 ise revize et.
        """
        for fb in feedback_list:
            if fb.needs_revision:
                return True

        weights = self._WEIGHTS
        w_total = sum(weights.get(fb.aspect, 0.10) for fb in feedback_list)
        if w_total == 0:
            return False
        w_score = sum(
            fb.score * weights.get(fb.aspect, 0.10)
            for fb in feedback_list
        ) / w_total

        return w_score < 0.65

    # =========================================================================
    # Geribildirim Metni İnşası
    # =========================================================================

    def _build_feedback_prompt(self, feedback_list: List[CriticFeedback]) -> str:
        """
        Yapılandırılmış geribildirim metnini oluşturur.

        Self-Refine (Madaan et al. 2023) önerir ki geri bildirim:
            1. Spesifik olmalı (hangi boyut, neden?)
            2. Yapıcı olmalı (ne yapmalı?)
            3. Kısa olmalı (model dikkatini dağıtmamak için)
        """
        parts = []
        for fb in feedback_list:
            if fb.needs_revision:
                parts.append(f"• [{fb.aspect.upper()}] {fb.message}")

        if not parts:
            return "Yanıtı genel kalite, netlik ve özlük açısından iyileştir."

        return "\n".join(parts)

    # =========================================================================
    # Model Tabanlı Revizyon — Self-Refine Döngüsü Adımı
    # =========================================================================

    def _revise_text(
        self,
        user_message: str,
        draft_text: str,
        feedback_list: List[CriticFeedback],
        context: Optional[str] = None,
    ) -> str:
        """
        Model tabanlı Self-Refine revizyonu (Madaan et al. 2023).

        Prompt yapısı:
            [GÖREV]       — Revizyonun amacı
            [BAĞLAM]      — Konuşma bağlamı (opsiyonel, ilk 300 karakter)
            [KULLANICI]   — Orijinal kullanıcı sorusu
            [TASLAK]      — Revize edilecek metin
            [ELEŞTİRİ]   — Boyut bazlı yapılandırılmış geribildirim
            [TALİMAT]     — Modele ne yapması gerektiği

        Fallback: Model başarısız olursa heuristik düzeltme uygular.
        """
        feedback_str = self._build_feedback_prompt(feedback_list)
        ctx_section  = (
            f"\n[BAĞLAM]\n{context[:300]}\n"
            if context and context.strip()
            else ""
        )

        revision_prompt = (
            f"[GÖREV] Aşağıdaki taslak yanıtı eleştiri doğrultusunda iyileştir. "
            f"Kullanıcının sorusunu tam olarak yanıtla, açık ve özlü yaz.\n"
            f"{ctx_section}"
            f"\n[KULLANICI SORUSU]\n{user_message}\n"
            f"\n[TASLAK YANIT]\n{draft_text}\n"
            f"\n[ELEŞTİRİ]\n{feedback_str}\n"
            f"\n[TALİMAT] Eleştiriyi göz önünde bulundurarak kullanıcıya "
            f"daha iyi ve doğru bir yanıt üret. Sadece yanıtı yaz, açıklama ekleme.\n"
            f"\n[YENİ YANIT]"
        )

        try:
            revised = self.mm.generate(
                revision_prompt,
                DecodingConfig(
                    max_new_tokens=min(
                        512,
                        max(64, len(draft_text.split()) * 2),  # Taslak uzunluğuna göre adaptif
                    ),
                    temperature=0.60,  # Düşük → daha tutarlı revizyon
                    top_p=0.90,
                    repetition_penalty=1.15,
                ),
            )
            revised = (revised or "").strip()
            if revised:
                return revised
        except Exception as e:
            logger.debug(f"Self-Refine model revizyonu başarısız: {e}")

        # -------- Heuristik Fallback --------
        revised = draft_text

        # Safety ihlali var mı?
        saf_fb = next((f for f in feedback_list if f.aspect == "safety" and f.needs_revision), None)
        if saf_fb:
            revised = f"[Uyarı: Bu konuda dikkatli olunmalıdır.] {revised}"

        # Coherence sorunu: noktalama eksik
        coh_fb = next((f for f in feedback_list if f.aspect == "coherence" and f.needs_revision), None)
        if coh_fb and not revised.endswith((".", "!", "?")):
            revised += "."

        # Factuality sorunu: hedging ekle
        fact_fb = next((f for f in feedback_list if f.aspect == "factuality" and f.needs_revision), None)
        if fact_fb:
            hedge_words = ["belki", "muhtemelen", "sanırım", "possibly", "perhaps"]
            if not any(w in revised.lower() for w in hedge_words):
                revised = f"Muhtemelen {revised}"

        return revised


# =============================================================================
# Dışa Aktarım
# =============================================================================

__all__ = ["CriticV2", "ModelAPI"]
