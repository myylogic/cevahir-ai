# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: handlers.py
Modül: cognitive_management/v2/processing
Görev: V3 Processing Handlers — Chain of Responsibility pattern.

       V3 Yenilikleri:
         • FeatureExtractionHandler : query_type + domain + complexity entegrasyonu;
                                      ReasoningTrace başlatma
         • SelfConsistencyHandler  : Wang et al. 2022 — N örneklem, çoğunluk/hibrit seçim
         • ContextBuildingHandler  : yapılandırılmış prompt (system | memory | cot | user)
         • GenerationHandler       : akıl yürütme izi kaydı
         • CriticHandler           : CriticFeedback entegrasyonu
         • MemoryUpdateHandler     : CognitiveState.query_type/domain güncellemesi

       Akademik Referanslar:
         • Wei et al. 2022     — Chain-of-Thought (think1 modu)
         • Wang et al. 2022    — Self-Consistency (SelfConsistencyHandler)
         • Yao et al. 2023     — Tree of Thoughts (DeliberationHandler)
         • Madaan et al. 2023  — Self-Refine (CriticHandler)

MİMARİ:
- SOLID Prensipleri: Single Responsibility (her handler tek sorumluluk),
                     Chain of Responsibility Pattern

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import List, Optional

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.cognitive_types import (
    DecodingConfig,
    PolicyOutput,
    ThoughtCandidate,
    ReasoningTrace,
    SelfConsistencyResult,
    CriticFeedback,
)
from cognitive_management.v2.utils.heuristics import build_features
from .pipeline import BaseProcessingHandler, ProcessingContext

# İsteğe bağlı: InMemoryCache
try:
    from ..utils.cache import InMemoryCache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Yardımcı — N-gram tabanlı metin benzerliği
# =============================================================================

def _bigram_overlap(a: str, b: str) -> float:
    """
    İki metin arasında bigram örtüşme oranı (Precision tabanlı).
    Self-Consistency'de aday cevapların benzerliğini ölçmek için kullanılır.
    """
    def bigrams(text: str):
        words = text.lower().split()
        return Counter(zip(words, words[1:]))

    bg_a = bigrams(a)
    bg_b = bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    common = sum((bg_a & bg_b).values())
    return common / max(len(bg_a), len(bg_b))


def _majority_select(candidates: List[str]) -> tuple[str, float]:
    """
    Çoğunluk oyu ile en desteklenen yanıtı seç.

    Wang et al. 2022 (Self-Consistency) yöntemi:
        Her çift arası benzerlik hesaplanır; en yüksek ortalama benzerliğe
        sahip aday 'merkezi' yanıt olarak seçilir.

    Returns:
        (seçilen_metin, ortalama_uyum_skoru)
    """
    if not candidates:
        return "", 0.0
    if len(candidates) == 1:
        return candidates[0], 1.0

    n = len(candidates)
    scores = []
    for i, c in enumerate(candidates):
        sim = sum(
            _bigram_overlap(c, candidates[j])
            for j in range(n) if j != i
        ) / (n - 1)
        scores.append(sim)

    best_idx = scores.index(max(scores))
    return candidates[best_idx], float(scores[best_idx])


def _score_select(
    candidates: List[str],
    user_message: str,
    backend,
) -> tuple[str, float]:
    """
    Model-score tabanlı seçim: backend.score() kullanılır.
    Sadece backend.score() mevcutsa çalışır; aksi hâlde majority'ye düşer.
    """
    if not hasattr(backend, "score"):
        return _majority_select(candidates)

    scores = []
    for c in candidates:
        try:
            s = float(backend.score(user_message, c))
        except Exception:
            s = 0.0
        scores.append(s)

    best_idx = scores.index(max(scores))
    # Normalize agreement score
    mean_s = sum(scores) / len(scores) if scores else 0.0
    agreement = (scores[best_idx] - mean_s) / (max(scores) - min(scores) + 1e-8)
    return candidates[best_idx], float(max(0.0, min(1.0, agreement)))


# =============================================================================
# 1. Feature Extraction Handler
# =============================================================================

class FeatureExtractionHandler(BaseProcessingHandler):
    """
    Girdi özelliklerini çıkarır.

    V3 Yenilikleri:
        • Model logit entropy (Phase 9): backend.estimate_entropy() → 0-3 ölçek
        • QueryType + DomainType tespiti (heuristics.classify_query_type/detect_domain)
        • Complexity score hesaplaması
        • ReasoningTrace başlatma (context'e eklenir)
        • Bellek geri çağırma skoru özeti

    SOLID: SRP — yalnızca özellik çıkarımı.
    """

    def __init__(self, memory_service, backend=None):
        super().__init__("FeatureExtraction")
        self.memory_service = memory_service
        self._backend = backend

    def _process(self, context: ProcessingContext) -> ProcessingContext:
        from cognitive_management.config import CognitiveManagerConfig
        cfg = getattr(self.memory_service, "cfg", CognitiveManagerConfig())
        user_msg = context.request.user_message

        # --- Entropy Tahmini ---
        entropy_raw = 0.8
        try:
            if self._backend is not None and hasattr(self._backend, "estimate_entropy"):
                logit_entropy = self._backend.estimate_entropy(user_msg)
                entropy_raw = float(logit_entropy) * 3.0
            else:
                # Heuristik yedek: soru işareti + belirsizlik belirteçleri
                q_count = user_msg.count("?")
                unc_words = ["belki", "muhtemelen", "sanırım", "olabilir",
                             "maybe", "perhaps", "possibly", "not sure"]
                u_count = sum(1 for w in unc_words if w in user_msg.lower())
                entropy_raw = (q_count * 0.5) + (u_count * 0.3)
        except Exception:
            entropy_raw = 0.8

        entropy = min(3.0, max(0.0, entropy_raw))

        # --- Bellek Geri Çağırma ---
        retrieved: list = []
        try:
            retrieved = self.memory_service.retrieve_context(
                query=user_msg,
                top_k=cfg.memory.rag_top_k if hasattr(cfg.memory, "rag_top_k") else 3,
            )
            context.retrieved_contexts = retrieved
        except Exception:
            context.retrieved_contexts = []

        # --- Özellik Vektörü (V3 zenginleştirilmiş) ---
        features = build_features(cfg, user_message=user_msg, entropy_est=entropy)

        # Bellek geri çağırma özeti
        if retrieved:
            features["has_relevant_memory"] = True
            features["memory_relevance_score"] = max(
                (c.get("score", 0.0) for c in retrieved), default=0.0
            )
            features["memory_hit_count"] = len(retrieved)
        else:
            features["has_relevant_memory"] = False
            features["memory_relevance_score"] = 0.0
            features["memory_hit_count"] = 0

        context.features = features

        # --- ReasoningTrace Başlatma ---
        # FeatureExtraction aşaması ilk adımdır (step=0)
        trace0 = ReasoningTrace(
            step=0,
            content=f"[Özellik Çıkarımı] query_type={features.get('query_type','unknown')} "
                    f"domain={features.get('domain','general')} "
                    f"complexity={features.get('complexity_score', 0.0):.2f} "
                    f"entropy={entropy:.2f}",
            score=1.0,
            source="feature_extraction",
        )
        if not hasattr(context, "reasoning_traces"):
            context.reasoning_traces = []
        context.reasoning_traces.append(trace0)

        return context


# =============================================================================
# 2. Policy Routing Handler
# =============================================================================

class PolicyRoutingHandler(BaseProcessingHandler):
    """
    Özellik vektörüne göre bilişsel mod seçer.
    SOLID: SRP — yalnızca politika yönlendirmesi.
    """

    def __init__(self, policy_router):
        super().__init__("PolicyRouting")
        self.policy_router = policy_router

    def _process(self, context: ProcessingContext) -> ProcessingContext:
        policy_output: PolicyOutput = self.policy_router.route(
            features=context.features,
            state=context.state,
        )

        # Kullanıcıdan gelen decoding parametresi her zaman önceliklidir
        if context.decoding_config:
            policy_output.decoding = context.decoding_config

        context.policy_output = policy_output

        # Trace ekle
        if hasattr(context, "reasoning_traces"):
            context.reasoning_traces.append(ReasoningTrace(
                step=len(context.reasoning_traces),
                content=f"[Politika Yönlendirme] mode={policy_output.mode} "
                        f"tool={policy_output.tool} "
                        f"complexity={policy_output.complexity:.2f}",
                score=1.0,
                source="policy_routing",
            ))

        return context


# =============================================================================
# 3. Deliberation Handler (CoT / Debate / ToT)
# =============================================================================

class DeliberationHandler(BaseProcessingHandler):
    """
    İç düşünce üretimi: think1 (CoT), debate2, tot modlarını destekler.

    Akademik:
        think1  → Wei et al. 2022 (CoT)
        debate2 → Wang et al. 2022 (Self-Consistency çoklu yol)
        tot     → Yao et al. 2023 (Tree of Thoughts)

    SOLID: SRP — yalnızca deliberation.
    """

    def __init__(self, engine, backend, cfg=None, tree_of_thoughts=None):
        super().__init__("Deliberation")
        self.engine = engine
        self.backend = backend
        self.cfg = cfg
        self.tree_of_thoughts = tree_of_thoughts

    def _process(self, context: ProcessingContext) -> ProcessingContext:
        policy = context.policy_output
        if not policy or policy.mode not in ("think1", "debate2", "tot"):
            return context

        if policy.mode == "tot":
            return self._process_tot(context, policy)

        # CoT / Debate
        try:
            n_thoughts = 1 if policy.mode == "think1" else 2
            thoughts = self.engine.generate_thoughts(
                prompt=context.request.user_message,
                num_thoughts=n_thoughts,
                decoding_config=policy.decoding,
            )

            if thoughts:
                from cognitive_management.v2.utils.selectors import (
                    pick_best_by_score,
                    diversify_topk,
                    select_topk,
                )

                if policy.mode == "debate2" and len(thoughts) >= 2:
                    diverse = diversify_topk(thoughts, k=2, jaccard_threshold=0.7)
                    context.selected_thought = pick_best_by_score(diverse or thoughts)
                elif len(thoughts) > 1:
                    top = select_topk(thoughts, k=min(3, len(thoughts)))
                    context.selected_thought = pick_best_by_score(top)
                else:
                    context.selected_thought = pick_best_by_score(thoughts)

                # Trace
                if hasattr(context, "reasoning_traces") and context.selected_thought:
                    context.reasoning_traces.append(ReasoningTrace(
                        step=len(context.reasoning_traces),
                        content=f"[{policy.mode.upper()}] {context.selected_thought.text[:200]}",
                        score=getattr(context.selected_thought, "score", 0.5),
                        source=policy.mode,
                    ))
        except Exception as e:
            logger.warning(f"DeliberationHandler hata: {e}")
            context.selected_thought = None

        return context

    def _process_tot(self, context: ProcessingContext, policy: PolicyOutput) -> ProcessingContext:
        """Tree of Thoughts (Yao et al. 2023) işlemcisi."""
        if not self.tree_of_thoughts:
            logger.warning("TreeOfThoughts başlatılmadı, think1 moduna geçiliyor")
            try:
                thoughts = self.engine.generate_thoughts(
                    prompt=context.request.user_message,
                    num_thoughts=1,
                    decoding_config=policy.decoding,
                )
                if thoughts:
                    from cognitive_management.v2.utils.selectors import pick_best_by_score
                    context.selected_thought = pick_best_by_score(thoughts)
            except Exception:
                context.selected_thought = None
            return context

        try:
            best_paths = self.tree_of_thoughts.solve(
                problem=context.request.user_message,
                system_prompt=context.request.system_prompt or (
                    self.cfg.default_system_prompt if self.cfg else None
                ),
                decoding_config=policy.decoding,
            )

            if best_paths:
                best_path, path_score = best_paths[0]
                combined = " → ".join(best_path[-3:])
                context.selected_thought = ThoughtCandidate(
                    text=combined,
                    score=path_score,
                    depth=len(best_path),
                    path=best_path,
                )

                if hasattr(context, "reasoning_traces"):
                    context.reasoning_traces.append(ReasoningTrace(
                        step=len(context.reasoning_traces),
                        content=f"[ToT] derinlik={len(best_path)}, skor={path_score:.3f} → {combined[:150]}",
                        score=path_score,
                        source="tot",
                    ))
        except Exception as e:
            logger.warning(f"ToT hata, think1'e geçiliyor: {e}")
            try:
                thoughts = self.engine.generate_thoughts(
                    prompt=context.request.user_message,
                    num_thoughts=1,
                    decoding_config=policy.decoding,
                )
                if thoughts:
                    from cognitive_management.v2.utils.selectors import pick_best_by_score
                    context.selected_thought = pick_best_by_score(thoughts)
            except Exception:
                context.selected_thought = None

        return context


# =============================================================================
# 4. Context Building Handler (RAG + yapılandırılmış prompt)
# =============================================================================

class ContextBuildingHandler(BaseProcessingHandler):
    """
    Üretim için bağlam metni oluşturur.

    V3 Yenilikleri:
        • Yapılandırılmış prompt bölümleri:
            [SİSTEM] | [HAFIZA] | [İLGİLİ BAĞLAM (RAG)] | [DÜŞÜNCE] | [KULLANICI]
        • RAG entegrasyonu (lazy initialization)
        • CoT düşüncesi varsa [DÜŞÜNCE] bölümüne dahil edilir

    SOLID: SRP — yalnızca bağlam inşası.
    """

    def __init__(self, memory_service, tool_policy=None):
        super().__init__("ContextBuilding")
        self.memory_service = memory_service
        self.tool_policy = tool_policy
        self._rag_enhancer = None

    def _process(self, context: ProcessingContext) -> ProcessingContext:
        cfg = getattr(self.memory_service, "cfg", None)

        # Araç seçimi
        if self.tool_policy:
            try:
                context.tool_name = self.tool_policy.choose_tool(context.features)
            except Exception:
                context.tool_name = None

        # Geçmiş özetleme ve budama
        history = self.memory_service.summarize_if_needed(context.state.history)
        history = self.memory_service.prune(history, user_message=context.request.user_message)

        # Temel bağlam
        context_text = self.memory_service.build_context(
            user_message=context.request.user_message,
            history=history,
            system_prompt=None,
        )

        # RAG zenginleştirme (lazy)
        try:
            if self._rag_enhancer is None and cfg and cfg.memory.enable_rag:
                from cognitive_management.v2.components.rag_enhancer import RAGEnhancer
                self._rag_enhancer = RAGEnhancer(self.memory_service, cfg)

            if self._rag_enhancer and self._rag_enhancer.enabled:
                context_text = self._rag_enhancer.enhance_context(
                    user_message=context.request.user_message,
                    existing_context=context_text,
                )
        except Exception as e:
            logger.warning(f"RAG zenginleştirme başarısız: {e}")

        # CoT düşüncesi ekle
        if (
            context.selected_thought
            and hasattr(context.selected_thought, "text")
            and context.selected_thought.text
        ):
            context_text = (
                f"{context_text}\n\n"
                f"[DÜŞÜNCE ADIMI]\n{context.selected_thought.text}"
            )

        # Araç isteği ekle
        if context.tool_name:
            context_text = f"{context_text}\n\n[ARAÇ İSTEĞİ] {context.tool_name}"

        context.context_text = context_text
        return context


# =============================================================================
# 5. Generation Handler
# =============================================================================

class GenerationHandler(BaseProcessingHandler):
    """
    Model backend'i çağırarak metin üretir.

    V3 Yenilikleri:
        • Üretim başarısını ve süresini ReasoningTrace'e kaydeder.

    SOLID: SRP — yalnızca text generation.
    """

    def __init__(self, backend):
        super().__init__("Generation")
        self.backend = backend

    def _process(self, context: ProcessingContext) -> ProcessingContext:
        policy = context.policy_output

        if not policy or not context.context_text:
            context.errors.append("Generation: Politika veya bağlam eksik")
            return context

        t0 = time.time()
        try:
            draft = self.backend.generate(
                prompt=context.context_text,
                decoding_config=policy.decoding,
            )
            context.draft_text = (draft or "").strip()
        except Exception as e:
            context.errors.append(f"Generation hatası: {e}")
            context.draft_text = ""

        elapsed = (time.time() - t0) * 1000.0

        if hasattr(context, "reasoning_traces"):
            context.reasoning_traces.append(ReasoningTrace(
                step=len(context.reasoning_traces),
                content=f"[Üretim] {elapsed:.0f}ms — "
                        f"{len(context.draft_text)} karakter",
                score=1.0 if context.draft_text else 0.0,
                source="generation",
            ))

        return context


# =============================================================================
# 6. Self-Consistency Handler  (Wang et al. 2022)
# =============================================================================

class SelfConsistencyHandler(BaseProcessingHandler):
    """
    Self-Consistency Decoding — Wang et al. 2022.

    N farklı yanıt örnekler (yüksek temperature) ve en çok desteklenen
    yanıtı çoğunluk oyu veya model skoru ile seçer.

    Aktivasyon: features["query_type"] reasoning/math/factual VE
                self_consistency_gate açık (entropy orta aralık).

    Yöntemler:
        "majority" — bigram örtüşme tabanlı merkezi aday seçimi
        "score"    — backend.score() tabanlı seçim
        "hybrid"   — ağırlıklı ortalama (0.6·majority + 0.4·score)

    SOLID: SRP — yalnızca self-consistency örneklemesi.
    """

    def __init__(self, backend, cfg=None):
        super().__init__("SelfConsistency")
        self.backend = backend
        self.cfg = cfg

    def _process(self, context: ProcessingContext) -> ProcessingContext:
        # Gate: sadece uygun koşullarda aktifleşir
        if not self._should_activate(context):
            return context

        n = getattr(self.cfg.policy, "self_consistency_n", 3) if self.cfg else 3
        method = getattr(self.cfg.policy, "self_consistency_method", "hybrid") if self.cfg else "hybrid"
        policy = context.policy_output

        # Yüksek temperature ile çeşitli örnekler
        sc_decoding = DecodingConfig(
            max_new_tokens=policy.decoding.max_new_tokens,
            temperature=min(0.95, (policy.decoding.temperature or 0.7) + 0.15),
            top_p=0.95,
            repetition_penalty=1.05,
        )

        candidates: List[str] = []
        for i in range(n):
            try:
                text = self.backend.generate(
                    prompt=context.context_text,
                    decoding_config=sc_decoding,
                )
                if text and text.strip():
                    candidates.append(text.strip())
            except Exception as e:
                logger.debug(f"SC örnekleme {i+1} başarısız: {e}")

        if not candidates:
            return context  # Mevcut taslağı koru

        # Seçim
        if method == "majority" or not hasattr(self.backend, "score"):
            selected, agreement = _majority_select(candidates)
        elif method == "score":
            selected, agreement = _score_select(
                candidates, context.request.user_message, self.backend
            )
        else:  # hybrid
            maj_text, maj_score = _majority_select(candidates)
            sc_text, sc_score = _score_select(
                candidates, context.request.user_message, self.backend
            )
            # Hangisi daha yüksek skor aldıysa onu seç
            if sc_score >= maj_score:
                selected, agreement = sc_text, 0.6 * sc_score + 0.4 * maj_score
            else:
                selected, agreement = maj_text, 0.6 * maj_score + 0.4 * sc_score

        # Sonucu kaydet
        sc_result = SelfConsistencyResult(
            candidates=candidates,
            selected=selected,
            agreement_score=float(agreement),
            method=method,
        )
        context.draft_text = selected
        context.self_consistency_result = sc_result

        # Trace
        if hasattr(context, "reasoning_traces"):
            context.reasoning_traces.append(ReasoningTrace(
                step=len(context.reasoning_traces),
                content=f"[Self-Consistency] n={len(candidates)} aday, "
                        f"yöntem={method}, uyum={agreement:.2f}",
                score=float(agreement),
                source="self_consistency",
            ))

        return context

    def _should_activate(self, context: ProcessingContext) -> bool:
        """Self-Consistency aktivasyon koşulları."""
        # draft_text henüz üretilmemiş olmalı (bu handler GenerationHandler'dan ÖNCE çalışmaz)
        # Veya draft_text varsa iyileştirmek için çalışır.
        features = context.features or {}
        query_type = features.get("query_type", "unknown")
        # Yaratıcı ve konuşma modları için anlamsız
        if query_type in ("creative", "conversational"):
            return False
        # Bağlam hazır olmalı
        if not context.context_text:
            return False
        return True


# =============================================================================
# 7. Critic Handler  (Madaan et al. 2023 — Self-Refine)
# =============================================================================

class CriticHandler(BaseProcessingHandler):
    """
    Self-Refine ile çıktı kalitesini denetler.

    V3 Yenilikleri:
        • CriticFeedback listesini context'e kaydeder
        • Trace'e critic geçiş sayısını ve genel skoru yazar

    Akademik: Self-Refine (Madaan et al. 2023).
    SOLID: SRP — yalnızca critic review.
    """

    def __init__(self, critic):
        super().__init__("Critic")
        self.critic = critic

    def _process(self, context: ProcessingContext) -> ProcessingContext:
        if not context.draft_text:
            return context

        try:
            final_text, revised = self.critic.review(
                user_message=context.request.user_message,
                draft_text=context.draft_text,
                context=context.context_text,
            )
            context.final_text = final_text
            context.revised = revised

            # CriticFeedback listesini critic'ten al (varsa)
            feedback_list = getattr(self.critic, "_last_feedback", None)
            if feedback_list:
                context.critic_feedback = feedback_list

            # Self-Refine geçiş sayısını kaydet
            if hasattr(context, "critic_passes"):
                context.critic_passes = int(getattr(self.critic, "_last_passes", 1))

            if hasattr(context, "reasoning_traces"):
                context.reasoning_traces.append(ReasoningTrace(
                    step=len(context.reasoning_traces),
                    content=f"[Critic] revize={'evet' if revised else 'hayır'} "
                            f"geçiş={getattr(self.critic, '_last_passes', 1)}",
                    score=1.0 if not revised else 0.7,
                    source="critic",
                ))

        except Exception as e:
            logger.warning(f"CriticHandler hata: {e}")
            context.final_text = context.draft_text
            context.revised = False

        return context


# =============================================================================
# 8. Memory Update Handler
# =============================================================================

class MemoryUpdateHandler(BaseProcessingHandler):
    """
    Konuşma geçmişini ve episodik belleği günceller.

    V3 Yenilikleri:
        • CognitiveState.query_type ve domain güncellenir.
        • turn_count arttırılır.
        • reasoning_traces state'e yazılır.

    SOLID: SRP — yalnızca bellek güncelleme.
    """

    def __init__(self, memory_service):
        super().__init__("MemoryUpdate")
        self.memory_service = memory_service

    def _process(self, context: ProcessingContext) -> ProcessingContext:
        final_text = context.final_text or context.draft_text or ""

        if not final_text:
            return context

        # Konuşma geçmişine ekle
        self.memory_service.add_turn(
            history=context.state.history,
            role="user",
            content=context.request.user_message,
        )
        self.memory_service.add_turn(
            history=context.state.history,
            role="assistant",
            content=final_text,
        )

        # Oturum özeti enjeksiyonu
        try:
            self.memory_service.inject_session_summary_if_needed(context.state.history)
        except Exception:
            pass

        # Durum güncelleme
        context.state.step += 1
        context.state.turn_count += 1

        if context.policy_output:
            context.state.last_mode = context.policy_output.mode

        if context.features:
            context.state.last_entropy = float(
                context.features.get("entropy_est", 0.0)
            )
            # V3: query_type ve domain state'e yansıt
            qt = context.features.get("query_type")
            dm = context.features.get("domain")
            if qt:
                context.state.query_type = qt
            if dm:
                context.state.domain = dm

        # V3: Reasoning trace'leri state'e kaydet (opsiyonel)
        if hasattr(context, "reasoning_traces") and context.reasoning_traces:
            context.state.reasoning_traces.extend(context.reasoning_traces)
            # Maksimum 50 trace tut (bellek yönetimi)
            if len(context.state.reasoning_traces) > 50:
                context.state.reasoning_traces = context.state.reasoning_traces[-50:]

        return context


# =============================================================================
# Dışa Aktarım
# =============================================================================

__all__ = [
    "FeatureExtractionHandler",
    "PolicyRoutingHandler",
    "DeliberationHandler",
    "ContextBuildingHandler",
    "GenerationHandler",
    "SelfConsistencyHandler",
    "CriticHandler",
    "MemoryUpdateHandler",
]
