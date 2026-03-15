# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: cognitive_types.py
Modül: cognitive_management
Görev: Cognitive Types - Bilişsel yönetim katmanı için tip tanımları.
       V3 (Akademik+Endüstri): QueryType, DomainType, ReasoningTrace,
       CriticFeedback, SelfConsistencyResult; CognitiveState ve
       CognitiveOutput'un zenginleştirilmiş yapısı.

MİMARİ:
- SOLID Prensipleri: Type definitions (tip tanımları)
- Design Patterns: Type Pattern (tip tanımları)
- Akademik Referanslar:
    • Wei et al. 2022 — Chain-of-Thought (CoT) → ReasoningTrace
    • Wang et al. 2022 — Self-Consistency → SelfConsistencyResult
    • Yao et al. 2023 — Tree of Thoughts (ToT) → ThoughtCandidate genişletmesi
    • Madaan et al. 2023 — Self-Refine → CriticFeedback
    • Bai et al. 2022 — Constitutional AI → CriticFeedback.constitutional

KULLANIM:
- Tip tanımları için
- Dataclass tanımları için
- Type hints için

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


# =============================================================================
# Mod kipleri ve kararlar
# =============================================================================

Mode = Literal["direct", "think1", "debate2", "tot", "react", "self_consistency"]
"""
Bilişsel işleme modu:
  direct           — Tek geçişli hızlı üretim
  think1           — Chain-of-Thought (Wei et al. 2022)
  debate2          — İki aday + Self-Consistency (Wang et al. 2022)
  tot              — Tree of Thoughts (Yao et al. 2023)
  react            — Reason+Act interleaving (Yao et al. 2022)
  self_consistency — N örneklem, çoğunluk oylaması (Wang et al. 2022)
"""

ToolDecision = Literal["none", "maybe", "must"]

# =============================================================================
# Sorgu ve Alan Sınıflandırması
# =============================================================================

QueryType = Literal[
    "factual",        # "Türkiye'nin başkenti nedir?"
    "reasoning",      # "Neden…?", "Nasıl etkiler…?", analiz
    "creative",       # Hikaye yaz, şiir yaz, senaryo oluştur
    "conversational", # Selamlama, genel sohbet
    "math",           # Sayısal hesaplama, denklem
    "code",           # Kod yazma, debug, açıklama
    "unknown",        # Sınıflandırılamadı
]
"""
Sorgu tipi – politika yönlendirme ve decoding parametrelerini etkiler.
"""

DomainType = Literal[
    "math",        # Matematik, istatistik
    "science",     # Fizik, kimya, biyoloji
    "law",         # Hukuk, mevzuat
    "medical",     # Sağlık, hastalık, ilaç
    "technology",  # Yazılım, donanım, programlama
    "history",     # Tarih, kronoloji
    "creative",    # Edebiyat, sanat, yaratıcı üretim
    "general",     # Genel bilgi / tanımlanamadı
]
"""
Alan tipi – risk skoru, critic katılığı ve decoding sıcaklığını etkiler.
"""


# =============================================================================
# Decoding / Üretim Ayarları
# =============================================================================

@dataclass
class DecodingConfig:
    """
    Metin üretim parametreleri. ModelManager.generate() / backend tarafından tüketilir.

    Alanlar:
        max_new_tokens:      Üretilecek maksimum yeni token sayısı.
        min_new_tokens:      EOS'un erken gelmesini önlemek için minimum token.
        temperature:         Softmax sıcaklığı (0.0 → greedy, >1.0 → daha rastgele).
        top_p:               Nucleus sampling eşiği (0.0–1.0).
        top_k:               Top-k sampling; 0 → devre dışı.
        repetition_penalty:  Tekrar cezası (>1.0 → tekrarı azaltır).
    """
    max_new_tokens: int = 256
    min_new_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = 0
    repetition_penalty: float = 1.1


# =============================================================================
# Akıl Yürütme İzleme Tipleri  (Wei et al. 2022 — CoT)
# =============================================================================

@dataclass
class ReasoningTrace:
    """
    Akıl yürütme zincirindeki tek bir adım.

    Akademik bağlam: Chain-of-Thought (Wei et al. 2022) çalışması,
    modelin ara adımlar üretmesini sağlamanın doğruluğu artırdığını gösterir.
    Bu dataclass, her adımı izlenebilir biçimde kaydeder.

    Alanlar:
        step:    Adım numarası (0'dan başlar).
        content: Bu adımın metin içeriği.
        score:   Bu adımın kalite skoru (0.0–1.0).
        source:  Adımı hangi modun ürettiği ("cot", "tot", "debate", "direct").
    """
    step: int
    content: str
    score: float = 0.0
    source: str = "direct"  # "cot" | "tot" | "debate" | "direct" | "react"


# =============================================================================
# Critic / Self-Refine Tipleri  (Madaan et al. 2023)
# =============================================================================

@dataclass
class CriticFeedback:
    """
    Critic bileşeninden gelen yapılandırılmış geribildirim.

    Akademik bağlam: Self-Refine (Madaan et al. 2023) çerçevesinde
    model çıktısını tek bir genel puana indirgemek yerine, çok-boyutlu
    geri bildirim yapısı daha yönlendirici revizyonlara yol açar.

    Alanlar:
        aspect:           Değerlendirilen boyut (örn. "coherence", "relevance").
        score:            0.0–1.0 arası puan (yüksek = iyi).
        message:          İnsan-okunabilir geri bildirim metni.
        needs_revision:   Bu boyut revizyon gerektiriyor mu?
        constitutional:   Constitutional AI ihlali mi? (Bai et al. 2022)
    """
    aspect: str
    score: float
    message: str
    needs_revision: bool = False
    constitutional: bool = False   # Constitutional AI ihlali


# =============================================================================
# Self-Consistency Tipi  (Wang et al. 2022)
# =============================================================================

@dataclass
class SelfConsistencyResult:
    """
    Self-Consistency örneklemesinin sonucu.

    Akademik bağlam: Self-Consistency (Wang et al. 2022) çalışması,
    aynı soruya N farklı akıl yürütme yolu üretip çoğunluk oylaması
    ile en güvenilir cevabı seçmenin doğruluğu artırdığını kanıtlar.

    Alanlar:
        candidates:       N örneklenmiş yanıt metni.
        selected:         Seçilen (en çok desteklenen) yanıt.
        agreement_score:  Adaylar arasındaki uyum oranı (0.0–1.0).
        method:           Seçim yöntemi ("majority" | "score" | "hybrid").
    """
    candidates: List[str]
    selected: str
    agreement_score: float
    method: str = "majority"


# =============================================================================
# Politika Çıkışı
# =============================================================================

@dataclass
class PolicyOutput:
    """
    PolicyRouter kararının bütünsel çıktısı.

    Alanlar:
        mode:         Seçilen bilişsel mod.
        tool:         Araç kullanım önerisi.
        decoding:     Üretim parametreleri.
        inner_steps:  Think/debate/tot için iç adım sayısı.
        query_type:   Algılanan sorgu tipi.
        domain:       Algılanan alan.
        complexity:   Sorgu karmaşıklık skoru (0.0–1.0).
    """
    mode: Mode
    tool: ToolDecision
    decoding: DecodingConfig
    inner_steps: int = 0
    query_type: Optional[QueryType] = None
    domain: Optional[DomainType] = None
    complexity: float = 0.0


# =============================================================================
# İç Ses Adayları  (Yao et al. 2023 — ToT)
# =============================================================================

@dataclass
class ThoughtCandidate:
    """
    Tek bir düşünce adayı.

    Akademik bağlam: Tree of Thoughts (Yao et al. 2023) çalışmasında
    her düğüm bir ThoughtCandidate'e karşılık gelir.

    Alanlar:
        text:   Düşünce metni.
        score:  Değerlendirme skoru (yüksek = daha iyi).
        depth:  ToT'da ağaç derinliği (kök = 0).
        path:   Bu noktaya gelen önceki düşünceler zinciri.
    """
    text: str
    score: float
    depth: int = 0
    path: List[str] = field(default_factory=list)


# =============================================================================
# Bilişsel Durum
# =============================================================================

def _make_session_id() -> str:
    """Benzersiz oturum kimliği üretir."""
    return str(uuid.uuid4())[:8]


@dataclass
class CognitiveState:
    """
    Chat/oturum sırasında kümülatif durum.

    Alanlar:
        history:          {"role": "user"|"assistant"|"system_summary", "content": "..."} listesi.
        step:             Kaçıncı bilişsel tur.
        last_entropy:     Bir önceki belirsizlik kestirimi (opsiyonel).
        last_mode:        Son karar verilen kip (opsiyonel).
        session_id:       Oturum kimliği (UUID tabanlı, 8 karakter).
        turn_count:       Toplam kullanıcı-asistan tur sayısı.
        query_type:       Son algılanan sorgu tipi.
        domain:           Son algılanan alan.
        reasoning_traces: Bu oturumda kaydedilen akıl yürütme adımları.
        metadata:         Özel oturum bilgileri (ör. kullanıcı tercihleri).
    """
    history: List[Dict[str, Any]] = field(default_factory=list)
    step: int = 0
    last_entropy: Optional[float] = None
    last_mode: Optional[Mode] = None
    # V3: Yeni alanlar
    session_id: str = field(default_factory=_make_session_id)
    turn_count: int = 0
    query_type: Optional[QueryType] = None
    domain: Optional[DomainType] = None
    reasoning_traces: List[ReasoningTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# İstek / Veri Girişi
# =============================================================================

@dataclass
class CognitiveInput:
    """
    Kullanıcıdan / üst katmandan gelen ham istek.

    Alanlar:
        user_message:  Zorunlu kullanıcı mesajı.
        system_prompt: Sistem davranışını sabitleyen üst yönerge (opsiyonel).
        metadata:      Sinyaller, tip etiketleri, risk bayrakları (opsiyonel).
    """
    user_message: str
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Nihai Çıktı
# =============================================================================

@dataclass
class CognitiveOutput:
    """
    CognitiveManager.handle(...) dönüşü.

    Temel alanlar:
        text:               Üretilen yanıt metni.
        used_mode:          Kullanılan bilişsel mod.
        tool_used:          Kullanılan araç adı (opsiyonel).
        revised_by_critic:  Critic tarafından revize edildi mi?
        metadata:           İzleme verileri (opsiyonel).

    V3 Genişletilmiş alanlar:
        reasoning_chain:         Üretim sürecindeki ReasoningTrace listesi.
        critic_passes:           Kaç Self-Refine turu yapıldı.
        critic_feedback:         Yapılandırılmış CriticFeedback listesi.
        memory_hits:             RAG'dan kaç bellek öğesi çekildi.
        latency_ms:              Toplam işlem süresi (milisaniye).
        query_type:              Algılanan sorgu tipi.
        domain:                  Algılanan alan.
        self_consistency_result: Self-Consistency örnekleme sonucu (opsiyonel).
        context_sources:         Kullanılan bağlam kaynakları (ör. "vector", "history").
    """
    text: str
    used_mode: Mode
    tool_used: Optional[str] = None
    revised_by_critic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    # V3: Zenginleştirilmiş çıktı alanları
    reasoning_chain: List[ReasoningTrace] = field(default_factory=list)
    critic_passes: int = 0
    critic_feedback: Optional[List[CriticFeedback]] = None
    memory_hits: int = 0
    latency_ms: float = 0.0
    query_type: Optional[QueryType] = None
    domain: Optional[DomainType] = None
    self_consistency_result: Optional[SelfConsistencyResult] = None
    context_sources: List[str] = field(default_factory=list)


# =============================================================================
# Dışa Aktarım
# =============================================================================

__all__ = [
    # Literaller
    "Mode",
    "ToolDecision",
    "QueryType",
    "DomainType",
    # Dataclass'lar
    "DecodingConfig",
    "ReasoningTrace",
    "CriticFeedback",
    "SelfConsistencyResult",
    "PolicyOutput",
    "ThoughtCandidate",
    "CognitiveState",
    "CognitiveInput",
    "CognitiveOutput",
]
