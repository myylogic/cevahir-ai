# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: config.py
Modül: cognitive_management
Görev: Cognitive Manager Config - CognitiveManager ve alt modüller için merkezi
       konfigürasyon. Dataclass tabanlı, tip güvenli yapı. Varsayılanlar üretim için
       makul ve temkinli. Kurallar (policy/rule set) ayrı sözlüklerle yönetilir.
       Ortam değişkenleri veya dış sözlükle override edilebilir. Geçerlilik (validate)
       ve birleştirme (merge/override_with) yardımcıları içerir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (config yönetimi)
- Design Patterns: Config Pattern (yapılandırma yönetimi)
- Endüstri Standartları: Configuration management best practices

KULLANIM:
- Config yönetimi için
- Config validation için
- Config merge/override için

BAĞIMLILIKLAR:
- dataclasses: Dataclass tanımları
- cognitive_types: Tip tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple

# Yerel tipler
from .cognitive_types import DecodingConfig

# =========================
# Alt-Konfigürasyon Sınıfları
# =========================

@dataclass
class CriticConfig:
    """
    Çıktı sonrası kısa 'eleştirel kontrol' katmanı ayarları.
    checks: Uygulanacak kontrol isimleri.
    strictness: 0.0–1.0 (yüksek değer = daha katı revizyon).
    
    Phase 7.2: Advanced Critic System Enhancement
    - External fact-checking support
    - Constitutional AI patterns
    - LLM-based fact verification
    """
    enabled: bool = True
    checks: Tuple[str, ...] = ("task_match", "safety", "claim_density")
    strictness: float = 0.5
    max_passes: int = 1  # gerekirse 1 kez revizyon
    
    # Phase 7.2: External Fact-Checking Configuration
    enable_external_fact_checking: bool = True  # External fact-checking APIs
    fact_checking_providers: Tuple[str, ...] = ("wikipedia",)  # "wikipedia" | "google" | "wolfram" | "llm"
    
    # Wikipedia fact-checking
    enable_wikipedia: bool = True
    wikipedia_max_results: int = 3  # Number of Wikipedia pages to check
    
    # Google Fact Check API (opsiyonel)
    enable_google_fact_check: bool = False
    google_fact_check_api_key: Optional[str] = None
    
    # Wolfram Alpha API (opsiyonel)
    enable_wolfram: bool = False
    wolfram_api_key: Optional[str] = None
    
    # LLM-based fact verification
    enable_llm_fact_verification: bool = True  # Model-based fact checking
    
    # Phase 7.2: Constitutional AI Configuration
    enable_constitutional_ai: bool = True  # Constitutional AI patterns
    constitutional_strictness: float = 0.7  # Strictness for constitutional principles (0.0-1.0)
    custom_principles: Tuple[str, ...] = ()  # Custom constitutional principles
    
    # Fact-checking thresholds
    fact_checking_score_threshold: float = 0.7  # Minimum score for verified facts
    claim_extraction_min_confidence: float = 0.5  # Minimum confidence for claim extraction

@dataclass
class MemoryConfig:
    """
    Oturum belleği ve özetleme ayarları.
    max_history_tokens: Modelin bağlam penceresine göre ayarlanmalı.
    
    Phase 7.1: Vector Memory Enhancement
    - Vector embeddings ve RAG (Retrieval-Augmented Generation) desteği
    """
    session_summary_every: int = 6
    salient_topk: int = 8
    max_history_tokens: int = 3072
    enable_session_summary: bool = True
    enable_salient_pruning: bool = True
    
    # Phase 7.1: Vector Memory & RAG Configuration
    enable_vector_memory: bool = True  # Vector embeddings kullanımı
    embedding_provider: str = "sentence-transformers"  # "sentence-transformers" | "openai" | "none"
    embedding_model: Optional[str] = None  # None = default model (all-MiniLM-L6-v2 for sentence-transformers)
    openai_api_key: Optional[str] = None  # OpenAI embeddings için API key
    
    vector_store_provider: str = "chroma"  # "chroma" | "pinecone" | "weaviate" | "qdrant" | "milvus" | "memory"
    vector_store_path: Optional[str] = "./memory/episodic_store"  # Local storage path (Chroma için)
    # Default: disk-persisted episodic memory that survives across sessions.
    # Set to None for in-memory (session-only) mode.
    vector_store_collection_name: str = "cognitive_memory"  # Collection/database name
    
    # Pinecone configuration (if using Pinecone)
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    
    # RAG Configuration
    enable_rag: bool = True  # Retrieval-Augmented Generation
    rag_top_k: int = 3  # Number of retrieved documents for RAG
    rag_score_threshold: float = 0.7  # Minimum similarity score for retrieval
    
    # Vector search configuration
    vector_search_top_k: int = 5  # Default top-k for vector search
    hybrid_search_alpha: float = 0.7  # Alpha for hybrid search (0.0 = keyword only, 1.0 = vector only)

@dataclass
class PolicyConfig:
    """
    Strateji seçimi (direct/think/debate/tot/self_consistency) için kapılar.

    Akademik Referanslar:
        • Wei et al. 2022     — CoT: think1 modu
        • Wang et al. 2022    — Self-Consistency: debate2 + self_consistency
        • Yao et al. 2023     — Tree of Thoughts: tot modu
        • Madaan et al. 2023  — Self-Refine: critic max_passes

    Eşik Kılavuzu (normalize edilmiş 0–3 entropi ölçeği):
        entropy_gate_think   : 1.5  → Hafif belirsizlik → CoT
        entropy_gate_debate  : 2.5  → Orta belirsizlik  → Self-Consistency
        entropy_gate_tot     : 3.0  → Yüksek belirsizlik → ToT
    """
    debate_enabled: bool = True
    entropy_gate_think: float = 1.5
    entropy_gate_debate: float = 2.5
    entropy_gate_tot: float = 3.0
    length_gate_debate: int = 200
    length_gate_tot: int = 300
    allow_inner_steps: bool = True
    tot_enabled: bool = True
    tot_max_depth: int = 3
    tot_branching_factor: int = 3
    tot_top_k: int = 5
    # V3: Self-Consistency (Wang et al. 2022)
    self_consistency_enabled: bool = True    # N örneklem → çoğunluk oyu
    self_consistency_n: int = 3             # Örneklem sayısı (3–5 önerilir)
    self_consistency_method: str = "hybrid" # "majority" | "score" | "hybrid"

@dataclass
class ToolsConfig:
    """
    Araç kullanımı politikası.
    allow: İzin verilen tool adları.
    policy: 'heuristic' veya ileride 'learned'.
    """
    allow: Tuple[str, ...] = ("calculator", "search", "file")
    policy: str = "heuristic"
    enable_tools: bool = True

@dataclass
class DecodingBounds:
    """
    Dinamik üretim parametreleri için güvenli aralıklar.

    Domain-aware decoding kılavuzu:
        Konuşma / basit sorgu : temperature → 0.5–0.65
        Genel bilgi           : temperature → 0.60–0.70
        Yaratıcı içerik       : temperature → 0.75–0.90
        Matematik / mantık    : temperature → 0.40–0.55 (düşük = tutarlı)
    """
    max_new_tokens_bounds: Tuple[int, int] = (32, 512)
    temperature_bounds: Tuple[float, float] = (0.40, 0.90)
    top_p_default: float = 0.90
    repetition_penalty_default: float = 1.15
    # Domain-specific temperature overrides
    math_temperature: float = 0.45      # Deterministik çıktı
    creative_temperature: float = 0.85  # Çeşitlilik
    code_temperature: float = 0.50      # Tutarlı ve doğru kod

@dataclass
class SafetyConfig:
    """
    Basit güvenlik/guardrail ayarları.
    risk_keywords_sensitive: Hassas alan anahtar kelimeleri.
    claim_markers: İddia/sayı içeren tetikleyiciler.
    """
    risk_keywords_sensitive: Tuple[str, ...] = ("siyaset", "tıbbi", "hukuki")
    claim_markers: Tuple[str, ...] = ("istatistik", "oran", "kanıt", "%")
    raise_on_high_risk: bool = False  # True ise yüksek riskte critic mutlaka devreye girer
    high_risk_threshold: float = 0.6

@dataclass
class FeatureRules:
    """
    Özellik çıkarımı için heuristik kurallar.
    Bu sözlükler PolicyRouter ve ToolPolicy tarafından kullanılır.
    """
    # Araç kullanım tercihleri
    tool_rules: Dict[str, Any] = field(default_factory=lambda: {
        "needs_recent_info_triggers": ("bugün", "güncel", "son haber", "en yeni"),
        "needs_calc_or_parse_triggers": ("hesapla", "toplam", "adet", "oran"),
        "default_decision": "none",  # none|maybe|must
    })
    # Risk/iddia kestirimi
    risk_rules: Dict[str, Any] = field(default_factory=lambda: {
        "base_risk": 0.0,
        "claim_bonus": 0.5,      # iddia algılanırsa
        "sensitive_bonus": 0.5,  # hassas alan algılanırsa
        "max_risk": 1.0,
    })

@dataclass
class RuntimeToggles:
    """
    Koşu zamanı açık/kapalı anahtarları.
    
    Phase 5-6: AIOps, Performance Optimization settings added.
    """
    enable_logging: bool = True
    enable_telemetry: bool = True
    enable_internal_thought_logging: bool = False  # iç sesler loglansın mı (gizlilik!)
    fail_fast: bool = False  # hata durumunda hızlıca istisna at
    # Phase 5: AIOps settings
    aiops_sensitivity: str = "medium"  # "low", "medium", "high"
    enable_auto_anomaly_check: bool = True  # Automatic anomaly detection after requests
    # Phase 6: Performance optimization settings
    enable_profiling: bool = False  # Performance profiling
    enable_semantic_cache: bool = False  # Semantic cache for similar queries
    semantic_cache_threshold: float = 0.85  # Similarity threshold for semantic cache (0.0-1.0)
    semantic_cache_max_size: int = 1000  # Maximum entries in semantic cache
    enable_cache_warming: bool = False  # Cache warming on startup
    # Phase 8: Advanced features
    enable_connection_pool: bool = False  # Connection pooling for backend
    connection_pool_size: int = 10  # Maximum connections in pool
    enable_batch_processing: bool = False  # Batch request processing
    batch_size: int = 10  # Batch size for request batching
    batch_timeout: float = 1.0  # Batch timeout in seconds

# =========================
# Ana Konfigürasyon
# =========================

@dataclass
class CognitiveManagerConfig:
    """
    CognitiveManager genel konfigürasyonu.
    """
    # Alt modüller
    critic: CriticConfig = field(default_factory=CriticConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    decoding_bounds: DecodingBounds = field(default_factory=DecodingBounds)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    features: FeatureRules = field(default_factory=FeatureRules)
    runtime: RuntimeToggles = field(default_factory=RuntimeToggles)

    # Varsayılan decoding ayarları (policy bunu güncelleyebilir)
    default_decoding: DecodingConfig = field(
        default_factory=lambda: DecodingConfig(
            max_new_tokens=128,
            temperature=0.65,
            top_p=0.90,
            top_k=0,
            repetition_penalty=1.15,
        )
    )

    # Sistem prompt'u gibi üst seviye davranış ayarları
    default_system_prompt: str = (
        "Sen Cevahir, Türkçe konuşan bir yapay zeka asistanısın. "
        "Doğru, yararlı ve anlaşılır cevaplar ver. "
        "Emin olmadığında belirsizliğini belirt."
    )

    # =========================
    # Yardımcı Metotlar
    # =========================

    def override_with(self, updates: Dict[str, Any]) -> "CognitiveManagerConfig":
        """
        Dışarıdan gelen sözlükle alanları günceller (sığ/derin birleştirme).
        Dataclass içindeki dataclass alanları için rekürsif güncelleme yapar.
        """
        def _merge(dc_obj, upd):
            if not isinstance(upd, dict):
                return dc_obj
            for f in getattr(dc_obj, "__dataclass_fields__", {}).keys():  # type: ignore
                if f not in upd:
                    continue
                cur = getattr(dc_obj, f)
                new = upd[f]
                if hasattr(cur, "__dataclass_fields__") and isinstance(new, dict):
                    _merge(cur, new)
                else:
                    setattr(dc_obj, f, new)
            return dc_obj

        return _merge(self, dict(updates))  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        """Konfigürasyonu sözlüğe döndür (serileştirme/telemetri için)."""
        return asdict(self)

    # Basit geçerlilik kontrolleri
    def validate(self) -> None:
        lo, hi = self.decoding_bounds.max_new_tokens_bounds
        if not (1 <= lo < hi):
            raise ValueError(f"max_new_tokens_bounds geçersiz: {lo}, {hi}")

        tlo, thi = self.decoding_bounds.temperature_bounds
        if not (0.0 <= tlo < thi <= 2.0):
            raise ValueError(f"temperature_bounds geçersiz: {tlo}, {thi}")

        if not (0.0 <= self.critic.strictness <= 1.0):
            raise ValueError(f"critic.strictness aralık dışı: {self.critic.strictness}")
        
        # Phase 7.2: Critic configuration validation
        if not (0.0 <= self.critic.constitutional_strictness <= 1.0):
            raise ValueError(f"critic.constitutional_strictness aralık dışı: {self.critic.constitutional_strictness}")
        
        if not (0.0 <= self.critic.fact_checking_score_threshold <= 1.0):
            raise ValueError(f"critic.fact_checking_score_threshold aralık dışı: {self.critic.fact_checking_score_threshold}")
        
        if not (0.0 <= self.critic.claim_extraction_min_confidence <= 1.0):
            raise ValueError(f"critic.claim_extraction_min_confidence aralık dışı: {self.critic.claim_extraction_min_confidence}")
        
        if self.critic.enable_google_fact_check and not self.critic.google_fact_check_api_key:
            raise ValueError("critic.google_fact_check_api_key gerekli (enable_google_fact_check=True)")
        
        if self.critic.enable_wolfram and not self.critic.wolfram_api_key:
            raise ValueError("critic.wolfram_api_key gerekli (enable_wolfram=True)")

        if self.policy.length_gate_debate < 0:
            raise ValueError("policy.length_gate_debate negatif olamaz.")

        # Phase 7.1: Vector memory validation
        if self.memory.embedding_provider not in {"sentence-transformers", "openai", "none"}:
            raise ValueError(f"memory.embedding_provider geçersiz: {self.memory.embedding_provider}")
        
        if self.memory.embedding_provider == "openai" and not self.memory.openai_api_key:
            raise ValueError("memory.openai_api_key gerekli (embedding_provider='openai')")
        
        if self.memory.vector_store_provider not in {"chroma", "pinecone", "weaviate", "qdrant", "milvus", "memory"}:
            raise ValueError(f"memory.vector_store_provider geçersiz: {self.memory.vector_store_provider}")
        
        if self.memory.vector_store_provider == "pinecone":
            if not self.memory.pinecone_api_key:
                raise ValueError("memory.pinecone_api_key gerekli (vector_store_provider='pinecone')")
            if not self.memory.pinecone_index_name:
                raise ValueError("memory.pinecone_index_name gerekli (vector_store_provider='pinecone')")
        
        if not (0.0 <= self.memory.rag_score_threshold <= 1.0):
            raise ValueError(f"memory.rag_score_threshold [0,1] aralığında olmalı: {self.memory.rag_score_threshold}")
        
        if not (0.0 <= self.memory.hybrid_search_alpha <= 1.0):
            raise ValueError(f"memory.hybrid_search_alpha [0,1] aralığında olmalı: {self.memory.hybrid_search_alpha}")

        # default_decoding alanlarını da doğrula
        dd = self.default_decoding
        if dd.max_new_tokens <= 0:
            raise ValueError("default_decoding.max_new_tokens > 0 olmalı.")
        if dd.temperature < 0.0:
            raise ValueError("default_decoding.temperature negatif olamaz.")
        if not (0.0 <= dd.top_p <= 1.0):
            raise ValueError("default_decoding.top_p [0,1] aralığında olmalı.")
        if dd.top_k is not None and dd.top_k < 0:
            raise ValueError("default_decoding.top_k negatif olamaz.")
        if dd.repetition_penalty <= 0.0:
            raise ValueError("default_decoding.repetition_penalty > 0 olmalı.")

        # V3: Self-Consistency validation
        if self.policy.self_consistency_n < 1:
            raise ValueError("policy.self_consistency_n >= 1 olmalı.")
        if self.policy.self_consistency_method not in {"majority", "score", "hybrid"}:
            raise ValueError("policy.self_consistency_method geçersiz (majority|score|hybrid).")

    # Ortam değişkenlerinden hızlı yükleme (opsiyonel, hafif)
    @staticmethod
    def from_env(env: Optional[Dict[str, str]] = None) -> "CognitiveManagerConfig":
        """
        Minimal ortam değişkeni desteği (yoksa varsayılan). Sadece birkaç kritik anahtar okunur.
        Örn:
          CM_CRITIC_ENABLED=true
          CM_TOOLS_ENABLE=true
          CM_DEBATE_ENABLED=false
        """
        import os
        e = env or os.environ

        def _b(key: str, default: bool) -> bool:
            val = e.get(key, "").strip().lower()
            if val in {"1", "true", "yes", "on"}:  return True
            if val in {"0", "false", "no", "off"}: return False
            return default

        cfg = CognitiveManagerConfig()
        cfg.critic.enabled = _b("CM_CRITIC_ENABLED", cfg.critic.enabled)
        cfg.tools.enable_tools = _b("CM_TOOLS_ENABLE", cfg.tools.enable_tools)
        cfg.policy.debate_enabled = _b("CM_DEBATE_ENABLED", cfg.policy.debate_enabled)

        # Sınırlar (sayısal)
        try:
            lo = int(e.get("CM_MAX_NEW_TOKENS_LO", cfg.decoding_bounds.max_new_tokens_bounds[0]))
            hi = int(e.get("CM_MAX_NEW_TOKENS_HI", cfg.decoding_bounds.max_new_tokens_bounds[1]))
            cfg.decoding_bounds.max_new_tokens_bounds = (lo, hi)
        except Exception:
            pass
        try:
            tlo = float(e.get("CM_TEMP_LO", cfg.decoding_bounds.temperature_bounds[0]))
            thi = float(e.get("CM_TEMP_HI", cfg.decoding_bounds.temperature_bounds[1]))
            cfg.decoding_bounds.temperature_bounds = (tlo, thi)
        except Exception:
            pass

        return cfg

# =========================
# Kural Sözlüğü (Örnek/Varsayılan)
# =========================

DEFAULT_RULES: Dict[str, Any] = {
    "tool_policy": {
        "priority": ["search", "calculator", "file"],
        "force_search_if_recent": True,
        "force_calc_if_numeric": True,
    },
    "mode_selection": {
        "prefer_direct_for_short_inputs": True,
        "debate_on_long_or_uncertain": True,
    },
    "critic_policy": {
        "require_on_high_risk": True,
        "light_on_low_risk": True,
        "revise_once_max": True,
    },
}

__all__ = [
    "CriticConfig",
    "MemoryConfig",
    "PolicyConfig",
    "ToolsConfig",
    "DecodingBounds",
    "SafetyConfig",
    "FeatureRules",
    "RuntimeToggles",
    "CognitiveManagerConfig",
    "DEFAULT_RULES",
]
