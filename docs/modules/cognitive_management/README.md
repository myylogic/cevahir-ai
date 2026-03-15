# Cognitive Management — Kullanım Kılavuzları

**Modül:** `cognitive_management`
**Son Güncelleme:** 2026-03-16

---

## İçindekiler

1. [Akademik Temeller](#akademik-temeller)
2. [Tipik Kullanım Senaryoları](#tipik-kullanım-senaryoları)
3. [Config Kılavuzu](#config-kılavuzu)
4. [Vektör Bellek ve RAG Kurulumu](#vektör-bellek-ve-rag-kurulumu)
5. [Özel Bileşen Yazma](#özel-bileşen-yazma)
6. [AIOps İzleme Entegrasyonu](#aiops-i̇zleme-entegrasyonu)
7. [Performans Optimizasyonu](#performans-optimizasyonu)
8. [Sorun Giderme](#sorun-giderme)
9. [Sürüm Geçmişi](#sürüm-geçmişi)

---

## Akademik Temeller

Cognitive Management'ta uygulanan bilişsel stratejiler doğrudan akademik çalışmalardan esinlenmiştir:

### Chain-of-Thought (CoT) — `think1` modu

> Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.

Modele yanıt vermeden önce adım adım düşünme fırsatı verilir. "Adım adım düşünelim: ..." gibi bir önizleme cümlesiyle başlayan iç monolog, ardından nihai yanıt gelir.

**Ne zaman kullanılır:**
- Sorgu karmaşıklığı orta düzeyde (entropi 1.5–2.5)
- Çok adımlı mantık gerektiren sorular
- "Neden?", "Nasıl?", analitik sorular

```python
# Manuel olarak zorlamak:
config.policy.entropy_gate_think = 0.0   # Her soruda CoT devreye girsin
config.policy.entropy_gate_debate = 999  # debate asla devreye girmesin
```

---

### Self-Consistency — `self_consistency` modu

> Wang, X. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.

Aynı soruya N farklı akıl yürütme yolu üretilir ve çoğunluk/hibrit oylama ile en güvenilir yanıt seçilir. Özellikle matematik ve mantık sorularında doğruluğu belirgin biçimde artırır.

**Parametreler:**
- `self_consistency_n = 3` → 3 farklı akıl yürütme yolu
- `self_consistency_method = "hybrid"` → örtüşme + puan birleşimi

**Takas:**
- Doğruluk artar ama 3x model çağrısı = 3x gecikme
- `n = 5` daha iyi sonuç verir ama 5x gecikme

---

### Tree of Thoughts (ToT) — `tot` modu

> Yao, S. et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS.

Yanıt üretimi ağaç arama olarak modellenir. Her düğümde birden fazla düşünce adayı üretilir, puanlanır ve en iyiler seçilerek devam edilir.

**Parametreler:**
- `tot_max_depth = 3` → 3 tur derinlik
- `tot_branching_factor = 3` → Her turda 3 dal
- `tot_top_k = 5` → En iyi 5 dal tutulur

**Ne zaman kullanılır:**
- Çok adımlı planlama gerektiren sorular
- Yaratıcı içerik üretimi
- Yüksek belirsizlik (entropi ≥ 3.0)

---

### Self-Refine — CriticHandler içinde

> Madaan, A. et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS.

Model önce bir yanıt üretir, ardından kendi ürettiği yanıtı eleştirir ve gerekirse revize eder. `max_passes` parametresiyle döngü sayısı sınırlandırılır.

```python
config.critic.max_passes = 2     # Maksimum 2 revizyon turu
config.critic.strictness = 0.7   # Yüksek katılık = daha sık revizyon
```

---

### Constitutional AI — ConstitutionalCritic

> Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic.

Model çıktıları önceden tanımlanmış "anayasal ilkeler" kümesiyle karşılaştırılır. İhlal bulunanlar revizyon için işaretlenir.

Varsayılan ilkeler (`constitutional_principles.py`):
- Fiziksel zarar verme içeriği yok
- Kişisel bilgi sızdırma yok
- Aldatıcı veya yanıltıcı içerik yok
- Nefret söylemi veya ayrımcılık yok
- Yasa dışı faaliyet yönlendirmesi yok
- ... (toplam 16 ilke)

Özel ilke eklemek:

```python
config.critic.custom_principles = (
    "Cevahir asla kesin tıbbi teşhis koymaz.",
    "Cevahir asla hisse senedi tavsiyesi vermez.",
)
```

---

### ReAct — `react` modu

> Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.

Düşünce (Reason) ve Eylem (Act) adımları iç içe örülür:
```
Düşünce: Bu soruyu yanıtlamak için web araması gerekiyor.
Eylem: search("güncel hava durumu İstanbul")
Gözlem: Bugün İstanbul'da 18°C, parçalı bulutlu.
Düşünce: Artık yanıtlayabilirim.
Yanıt: İstanbul'da bugün 18°C ve parçalı bulutlu hava bekleniyor.
```

---

## Tipik Kullanım Senaryoları

### Senaryo 1: Sohbet Botu (Hızlı Yanıt Öncelikli)

Gecikmeyi minimuma indirmek, çoğunlukla `direct` modunu kullanmak:

```python
config = CognitiveManagerConfig()

# Eşikleri yükselt — neredeyse her zaman direct
config.policy.entropy_gate_think = 2.8
config.policy.entropy_gate_debate = 99.0
config.policy.tot_enabled = False
config.policy.self_consistency_enabled = False

# Critic'i kapat (gecikme kaynağı)
config.critic.enabled = False

# Bellek basit oturum, disk yazma yok
config.memory.enable_vector_memory = False
config.memory.enable_rag = False

manager = CognitiveManager(model_manager=model, config=config)
```

---

### Senaryo 2: Araştırma Asistanı (Doğruluk Öncelikli)

```python
config = CognitiveManagerConfig()

# CoT'u düşük eşikle her zaman devreye al
config.policy.entropy_gate_think = 0.5
config.policy.entropy_gate_debate = 1.5
config.policy.self_consistency_enabled = True
config.policy.self_consistency_n = 5    # 5 örneklem

# Critic çok sıkı
config.critic.strictness = 0.9
config.critic.max_passes = 2
config.critic.enable_external_fact_checking = True
config.critic.enable_wikipedia = True

# Epizodik bellek aktif
config.memory.enable_vector_memory = True
config.memory.enable_rag = True
config.memory.rag_top_k = 5

config.validate()
manager = CognitiveManager(model_manager=model, config=config)
```

---

### Senaryo 3: Matematik / Kod Çözücü

```python
config = CognitiveManagerConfig()

# CoT zorunlu
config.policy.entropy_gate_think = 0.0
# Düşük sıcaklık — hesap makinesi gibi tutarlı
config.decoding_bounds.math_temperature = 0.3
config.decoding_bounds.code_temperature = 0.3

# Araç: calculator zorunlu (sayı içeren her sorguda)
config.features.tool_rules["force_calc_if_numeric"] = True

# Bellek: sadece oturum (vektör belleğe gerek yok)
config.memory.enable_vector_memory = False

manager = CognitiveManager(model_manager=model, config=config)
output = manager.handle("153 × 47 = ?")
# expected: mode="think1", tool_used="calculator"
```

---

### Senaryo 4: Yaratıcı İçerik Üretimi

```python
config = CognitiveManagerConfig()

# Yüksek sıcaklık — çeşitlilik
config.decoding_bounds.creative_temperature = 0.92
config.default_decoding.temperature = 0.88

# ToT — yaratıcı dallarda arama
config.policy.entropy_gate_tot = 1.5
config.policy.tot_branching_factor = 4
config.policy.tot_top_k = 8

# Critic'i hafif tut (yaratıcı içerikte fact-check anlamsız)
config.critic.checks = ("task_match",)
config.critic.enable_external_fact_checking = False

manager = CognitiveManager(model_manager=model, config=config)
output = manager.handle("Kış gecesinde geçen kısa bir hikaye yaz.")
# expected: mode="tot", used_mode="tot"
```

---

## Config Kılavuzu

### Entropy Gate Ayarlama

Model logit entropisi normalize edilmiş 0–3 ölçeğindedir:

| Entropi Değeri | Anlam | Önerilen Tepki |
|---------------|-------|----------------|
| 0.0 – 1.0 | Çok düşük belirsizlik (açık sorgu) | `direct` |
| 1.0 – 1.5 | Düşük belirsizlik | `direct` veya `think1` |
| 1.5 – 2.5 | Orta belirsizlik | `think1` |
| 2.5 – 3.0 | Yüksek belirsizlik | `debate2` / `self_consistency` |
| 3.0+ | Çok yüksek belirsizlik | `tot` |

Sisteminizin tipik entropi değerlerini öğrenmek için:

```python
output = manager.handle("test mesajı")
print(manager.state.last_entropy)
```

### Token Üretim Sınırları

```python
# Kısa yanıtlar için (sohbet)
config.decoding_bounds.max_new_tokens_bounds = (16, 128)
config.default_decoding.max_new_tokens = 64

# Uzun yanıtlar için (makale, rapor)
config.decoding_bounds.max_new_tokens_bounds = (64, 1024)
config.default_decoding.max_new_tokens = 512
```

### Tekrar Cezası

`repetition_penalty > 1.0` tekrarlayan ifadeleri baskılar:

```python
config.default_decoding.repetition_penalty = 1.2   # Daha güçlü baskı
config.default_decoding.repetition_penalty = 1.05  # Hafif baskı (yaratıcı içerik için)
```

---

## Vektör Bellek ve RAG Kurulumu

### ChromaDB (Yerleşik, Önerilen)

```bash
pip install chromadb sentence-transformers
```

```python
config.memory.enable_vector_memory = True
config.memory.embedding_provider = "sentence-transformers"
config.memory.embedding_model = None           # None = all-MiniLM-L6-v2 (varsayılan)
config.memory.vector_store_provider = "chroma"
config.memory.vector_store_path = "./memory/episodic_store"  # Disk'e kalıcı
config.memory.enable_rag = True
config.memory.rag_top_k = 3
config.memory.rag_score_threshold = 0.7
```

### In-Memory (Geliştirme / Test)

```python
config.memory.vector_store_provider = "memory"
config.memory.vector_store_path = None
# Oturum kapanınca tüm bellek kaybolur
```

### RAG Nasıl Çalışır?

```
Kullanici: "Peki ya onceki oturumda konustugumuz konular?"
                |
                v
EmbeddingAdapter.embed(kullanici_mesaji)
                |
                v
VectorStore.similarity_search(embed, top_k=3, threshold=0.7)
                |
                v
Benzer onceki oturum parcalari:
    - "Gecen hafta Cevahir eğitimini konustuk..."
    - "Tokenizer yapılandırması hakkında..."
                |
                v
ContextBuildingHandler --> [MEMORY] bölümüne ekle
                |
                v
Model, geçmiş bağlamla daha tutarlı yanıt verir
```

### Bellek Konsolidasyon (Her 6 Turda)

```python
# session_summary_every = 6 (varsayılan)
# Her 6 turda otomatik çalışır:

def _consolidate_memory():
    summary = model.generate(
        f"Bu oturum özetini çıkar: {history[-6:]}"
    )
    embedding = embed_adapter.embed(summary)
    vector_store.add(
        text=summary,
        embedding=embedding,
        metadata={"session_id": state.session_id, "turn": state.turn_count}
    )
    # Eski history kısalır — özet vektörde
```

---

## Özel Bileşen Yazma

### Özel PolicyRouter

```python
from cognitive_management.v2.interfaces.component_protocols import PolicyRouter
from cognitive_management.cognitive_types import PolicyOutput, DecodingConfig, CognitiveState

class MyDomainPolicyRouter:
    """Tıbbi sorgular için her zaman ToT kullanan özel router."""

    def route(self, features: dict, state: CognitiveState) -> PolicyOutput:
        domain = features.get("domain", "general")

        if domain == "medical":
            return PolicyOutput(
                mode="tot",
                tool="none",
                decoding=DecodingConfig(temperature=0.3, max_new_tokens=512),
                inner_steps=9,
                query_type=features.get("query_type"),
                domain=domain,
            )
        # Varsayılan
        return PolicyOutput(
            mode="direct",
            tool="none",
            decoding=DecodingConfig(),
        )
```

### Özel FactChecker

```python
from cognitive_management.v2.components.fact_checkers.base import BaseFactChecker

class MyCustomChecker(BaseFactChecker):
    def verify(self, claim: str) -> tuple[bool, float]:
        """
        Returns:
            (verified: bool, confidence: float 0.0-1.0)
        """
        # Kendi dogrulama mantiginiz
        result = my_api.check(claim)
        return result.is_true, result.confidence
```

### Özel VectorStore

```python
from cognitive_management.v2.components.vector_store.base import BaseVectorStore

class MyVectorStore(BaseVectorStore):
    def add(self, text: str, embedding: list[float], metadata: dict) -> str:
        # ID dondur
        ...

    def search(self, embedding: list[float], top_k: int, threshold: float) -> list[dict]:
        # [{"text": ..., "score": ..., "metadata": ...}] dondur
        ...

    def delete(self, doc_id: str) -> bool:
        ...
```

---

## AIOps İzleme Entegrasyonu

### Temel İzleme

```python
from cognitive_management.v2.monitoring.performance_monitor import PerformanceMonitor
from cognitive_management.v2.monitoring.health_check import HealthCheck

perf_mon = PerformanceMonitor()

# Orchestrator'a bağla
orchestrator = CognitiveOrchestrator(
    ...,
    performance_monitor=perf_mon,
)

# Metrikleri oku
metrics = perf_mon.get_metrics()
print(f"Ortalama gecikme: {metrics['avg_latency_ms']:.1f} ms")
print(f"İstek/dakika: {metrics['requests_per_minute']:.1f}")
print(f"p95 gecikme: {metrics['p95_latency_ms']:.1f} ms")
```

### Anomali Tespiti

```python
from cognitive_management.v2.monitoring.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(sensitivity="high")

# Her istek sonrası kontrol (enable_auto_anomaly_check=True ile otomatik)
anomalies = detector.check(metrics)
if anomalies:
    for anomaly in anomalies:
        print(f"Anomali: {anomaly.type} — {anomaly.description}")
```

### Sağlık Kontrolü

```python
from cognitive_management.v2.monitoring.health_check import HealthCheck

health = HealthCheck(backend=backend, memory_service=memory_service)
status = health.check()

print(status.overall)         # "healthy" | "degraded" | "unhealthy"
print(status.backend_ok)      # True/False
print(status.vector_store_ok) # True/False
print(status.latency_ok)      # True/False
```

### Alerting

```python
from cognitive_management.v2.monitoring.alerting import AlertingManager

alerting = AlertingManager()
alerting.set_threshold("avg_latency_ms", max_value=2000)  # 2 saniye üstü uyarı
alerting.set_threshold("error_rate", max_value=0.05)      # %5 hata oranı uyarı

# Callback kayıt
alerting.on_alert(lambda alert: print(f"UYARI: {alert.message}"))
```

---

## Performans Optimizasyonu

### Semantik Cache

Anlamsal olarak benzer sorgulara (cosine ≥ 0.85) cache'den yanıt ver:

```python
config.runtime.enable_semantic_cache = True
config.runtime.semantic_cache_threshold = 0.85  # 0.85 = yüksek benzerlik gerekli
config.runtime.semantic_cache_max_size = 1000   # Maksimum cache girdisi

# "İstanbul'un nüfusu nedir?" ve "İstanbul kaç kişilik?"
# → Benzerlik 0.88 → cache hit → direkt yanıt, model çağrısı yok
```

### Cache Ön Isıtma

Sık sorulan sorular için başlangıçta cache doldur:

```python
config.runtime.enable_cache_warming = True

# Isıtma listesi (cache_warming.py'de tanımlanır):
warmup_queries = [
    "Merhaba, nasıl yardımcı olabilirim?",
    "Cevahir nedir?",
    "Türkçe yardım eder misin?",
]
```

### Request Batching

Yoğun yük senaryolarında toplu istek işleme:

```python
config.runtime.enable_batch_processing = True
config.runtime.batch_size = 10
config.runtime.batch_timeout = 1.0  # Saniye; timeout dolunca grup işlenir
```

### Profiling

Hangi handler'ın en çok zaman aldığını öğren:

```python
config.runtime.enable_profiling = True

output = manager.handle("test")
profile = output.metadata.get("handler_times", {})
for handler, ms in sorted(profile.items(), key=lambda x: -x[1]):
    print(f"  {handler}: {ms:.1f} ms")
# Örnek çıktı:
#   GenerationHandler: 1240.3 ms
#   DeliberationHandler: 312.1 ms
#   CriticHandler: 89.5 ms
#   MemoryRetrievalHandler: 45.2 ms
```

---

## Sorun Giderme

### Sorun: Yanıtlar Çok Yavaş

**Kontrol listesi:**
1. `used_mode` kontrol et — `tot` veya `self_consistency` seçilmiş olabilir
2. `critic.max_passes` değeri kaçta? 2+ ise gecikme artar
3. `enable_external_fact_checking = True` mu? Wikipedia çağrısı ~500ms
4. `rag_top_k` kaçta? Yüksekse embedding maliyeti artar

**Hızlı düzeltme:**
```python
config.policy.entropy_gate_think = 2.0     # CoT daha az devreye girsin
config.policy.tot_enabled = False          # ToT kapat
config.critic.max_passes = 1              # Tek revizyon turu
config.critic.enable_external_fact_checking = False
config.memory.rag_top_k = 2
```

---

### Sorun: Çok Fazla Self-Refine Revizyonu

Critic çok sık `needs_revision=True` dönüyorsa:

```python
config.critic.strictness = 0.3        # Daha toleranslı
config.critic.checks = ("task_match",) # Sadece görev uyumunu kontrol et
config.critic.enable_constitutional_ai = True   # Bunu kapat
config.safety.high_risk_threshold = 0.9  # Risk eşiğini yükselt
```

---

### Sorun: Vektör Bellek Çalışmıyor

```python
# Bağımlılıklar yüklü mü?
try:
    import chromadb
    import sentence_transformers
    print("Bagimliliklar tamam")
except ImportError as e:
    print(f"Eksik bagimlilik: {e}")
    # Cozum: pip install chromadb sentence-transformers
```

```python
# Path yazılabilir mi?
import os
path = "./memory/episodic_store"
os.makedirs(path, exist_ok=True)
assert os.access(path, os.W_OK), "Yazma izni yok!"
```

---

### Sorun: Araç Kullanılmıyor

```python
# Aracın allow listesinde olduğundan emin ol
config.tools.allow = ("calculator", "search", "file")
config.tools.enable_tools = True

# Tetikleyici kelimeleri kontrol et
features = config.features.tool_rules
print(features["needs_calc_or_parse_triggers"])
# ("hesapla", "toplam", "adet", "oran")
# Kullanıcı mesajında bu kelimeler olmalı
```

---

### Sorun: Oturum Belleği Taşıyor

```python
# Token sınırını artır
config.memory.max_history_tokens = 4096

# Özetleme sıklığını artır (daha sık özetle)
config.memory.session_summary_every = 4   # Varsayılan 6, daha sık yap

# Salient pruning açık mı?
config.memory.enable_salient_pruning = True
config.memory.salient_topk = 6   # Özette tutulacak mesaj sayısı
```

---

## Sürüm Geçmişi

### V2 (Mevcut — 2025)

**Yeni Özellikler:**
- CognitiveOrchestrator: Tam enterprise entegrasyon
- ProcessingPipeline: Chain of Responsibility mimarisi
- EventBus: Pub/Sub event yönetimi
- Middleware zinciri: Validation, Metrics, Tracing
- DependencyContainer: Dependency Injection
- MemoryServiceV2: Çift katmanlı bellek (oturum + vektör)
- CriticV2 + ConstitutionalCritic: Self-Refine + Constitutional AI
- SelfConsistencyHandler: Wang et al. 2022
- TreeOfThoughts: Yao et al. 2023
- AIOps monitoring: PerformanceMonitor, AnomalyDetector, TrendAnalyzer
- Semantik cache: Anlamsal benzerlik bazlı cache
- Request batching: Toplu istek işleme
- Asenkron pipeline: AsyncProcessingPipeline

**Bug Düzeltmeleri:**
- Oturum belleği taşma sorunları
- Vektör bellek konsolidasyon tutarsızlığı
- Critic re-entry döngüsü

### V1 (Arşiv — 2024)

- CognitiveManager temel yapısı
- PolicyRouter (entropi bazlı)
- MemoryService (sadece oturum geçmişi)
- Basit Critic (heuristik kontroller)
- Tool executor (calculator, search, file)
- CoT / debate2 desteği
