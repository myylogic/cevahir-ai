# Phase 1 — Refactoring Plan · Cognitive

> **Kanonik Birim #7 · Cognitive** — `cognitive_management/` (v2)
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P1: sync/async birleştirme).
>
> **Bu sürüm derinleştirilmiştir:** her kalem `dosya:satır` çapası ve mimari kesitle bağlıdır.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `cognitive_management/` (~3.046 üst + ~17.581 v2) |
| Mevcut test | **24 dosya / ~19.500 LOC — ağırlıklı facade seviyesi** |
| Refactor hedefleri | 2070 LOC facade, sync/async ikizliği, çoklu `ModelAPI`, cache çeşitliliği, izleme yükü |
| Kritik sözleşmeler | `ModelAPI` protokolü · 8-handler zinciri sırası · `CognitiveInput/Output` (`text`) |

> ⚠️ Ortam: `torch`/ChromaDB kurulu değil; canlı koşu geliştirme ortamında.

### 0.1 Mimari Referans Haritası

| Referans | Ne için |
|----------|---------|
| [master-architecture §5](../../master-architecture.md#5-uçtan-uca-akış--çıkarım-inference) | Çıkarımda bilişsel katmanın yeri |
| [master-architecture §9](../../master-architecture.md#9-mimari-kararlar-ve-uygulanan-desenler) | Uygulanan desenler (CoR, Strategy, Observer, DI) |
| [research §4](README.md#4-i̇ç-mimari-katmanlı-desenler) | Facade→Orchestrator→Pipeline katmanları |
| [research §5](README.md#5-veri--kontrol-akışı--i̇şleme-zinciri) | Gerçek 8-handler zinciri (hedefin temeli) |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri |
| [model-engine research §4.2](../model-engine/README.md#42-adapter-köprüsü-cevahirmodelapi) | Backend'e giriş: `ModelAPI` protokolü (asla somut model) |

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Kaynak | Durum |
|------|------|--------|-------|
| Facade (manager) | `test_cognitive_manager_{performance,monitoring,aiops,tracing,events,tools,config,...}` (~19.5k LOC) | [research §3.1](README.md#31-üst-düzey) | ✅ Çok kapsamlı |
| Handler zinciri (bileşen) | dağınık | [research §3.2](README.md#32-çekirdek--i̇şleme-v2core-v2processing) | 🟡 İnce |
| Critic / ToT / memory (birim) | — | [research §3.3](README.md#33-bileşenler-v2components) | 🟡 İnce |
| Sync↔async parite | — | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | ❌ Yok |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **8-handler zinciri** birim testleri | `v2/processing/handlers.py:151/243/283/413/490/540/654/713` (8 handler) | [research §5](README.md#5-veri--kontrol-akışı--i̇şleme-zinciri) | her handler'ın `handle(ctx)` giriş→çıkış sözleşmesi |
| **T-02** | **Sync↔async parite** | `v2/processing/pipeline.py` vs `async_pipeline.py`; `handlers.py` vs `async_handlers.py` | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | aynı girdi → sync ve async aynı çıktı |
| **T-03** | **Politika yönlendirme eşikleri** | `v2/components/policy_router_v2.py:57` (`PolicyRouterV2`) | [research §5.1](README.md#51-politika-kararı-policyrouterv2--strategy) | entropi→mod sınır değerleri |
| **T-04** | **RAG bellek** getirme/güncelleme | `v2/components/memory_service_v2.py:67`; `vector_store/chroma_vector_store.py` | [research §5.2](README.md#52-bellek-memoryservicev2--iki-katman) | vektör getirme + periyodik özet |
| **T-05** | **Critic revizyon** (self-refine) | `v2/components/critic_v2.py:164`; `constitutional_critic.py` | [research §3.3](README.md#33-bileşenler-v2components) | ihlal→revizyon akışı |
| **T-06** | **Çoklu `ModelAPI` protokolü** | `cognitive_manager.py:96` **ve** `critic_v2.py:69` | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | tek protokol uyum testi |
| **T-07** | **ToT** BFS/beam | `v2/components/tree_of_thoughts.py:106` | [research §3.3](README.md#33-bileşenler-v2components) | ağaç genişletme/budama |

### A.3 Olması Gereken Test Durumu
Facade kapsamı korunur; handler/critic/ToT/memory **birim** testleriyle güçlenir;
sync↔async parite (T-02) ve tek `ModelAPI` (T-06) doğrulanır. T-02/T-06, Geliştirme
Fazı D1/D2'nin önkoşuludur.

### A.4 Test Sprint'leri
**T1** facade yeşil taban + bileşen kapsam boşluğu haritası.
**T2** handler birim (T-01) + politika eşikleri (T-03) + protokol (T-06).
**T3** bellek/critic/ToT (T-04, T-05, T-07) + sync-async parite (T-02).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
cognitive_management/
  ├── interfaces/  → TEK ModelAPI protokolü (facade + critic buradan alır)
  ├── processing/  → tek pipeline çekirdeği; sync ve async ince adaptörler
  ├── cognitive_manager.py → yalın facade (event/cache/trace ayrı modüle)
  └── cache/       → birleşik cache stratejisi
```
> 8-handler sırası [research §5](README.md#5-veri--kontrol-akışı--i̇şleme-zinciri) ile
> **aynı kalır**; yalnız sync/async ikizliği ve protokol dağınıklığı giderilir.

### B.2 Geliştirme Sprint'leri *(P1)*
**D1 — Tek `ModelAPI` protokolü** *(düşük risk)*: `cognitive_manager.py:96` ve
`critic_v2.py:69` çift tanımını `v2/interfaces/`'te birleştir. Önkoşul: T-06.
**Kabul:** protokol tek yerde; uyum testi yeşil.
**D2 — Sync/async birleştirme** *(yüksek risk)*: `pipeline`+`handlers` ile
`async_pipeline`+`async_handlers` ortak çekirdeğe. Önkoşul: T-02 parite. **Kabul:**
parite korunur; ikiz kod biter.
**D3 — Facade ayrıştırma** *(orta risk)*: `cognitive_manager.py` (2070) event/cache/
trace API'lerini ayrı modüllere çıkar. **Kabul:** `handle` imzası aynı; LOC düşer.
**D4 — Cache birleştirme** *(orta risk)*: `middleware/cache`, `utils/cache`,
`semantic_cache`, `cache_warming` tek stratejide.
**D5 — İzleme yükü denetimi** *(düşük risk)*: 6 monitoring modülünün aktifliğini
netleştir; pasif olanı işaretle.

### B.3 Korunacak Sözleşmeler
| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| `ModelAPI` protokolü | `v2/interfaces/` (D1 sonrası) | Model/Engine adapter buna uyar |
| 8-handler pipeline sırası | `v2/core/orchestrator.py:154` | akıl yürütme davranışı |
| `CognitiveInput/Output` alanları (`text`) | `cognitive_types.py:311` | [master §5](../../master-architecture.md#5-uçtan-uca-akış--çıkarım-inference) tüketiciler |
| `handle` genel imzası | `cognitive_manager.py:509` | facade tüketicileri |

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + §D.
D1/D3 research §3/§9 anchor'larını etkiler → aynı PR'da güncellenir.

## D. Durum Tablosu
| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 facade taban + boşluk | 24 test dosyası | ⏳ |
| A | T2 handler/politika/protokol | `handlers.py`, `policy_router_v2.py:57`, `critic_v2.py:69` | ⏳ |
| A | T3 bellek/critic/ToT/parite | `memory_service_v2.py:67`, `tree_of_thoughts.py:106`, `async_*` | ⏳ |
| B | D1 tek ModelAPI | `cognitive_manager.py:96`, `critic_v2.py:69` | ⏳ |
| B | D2 sync/async birleştirme | `processing/{,async_}pipeline.py` | ⏳ |
| B | D3 facade ayrıştırma | `cognitive_manager.py` | ⏳ |
| B | D4 cache birleştirme | `middleware/cache`, `utils/*cache*` | ⏳ |
| B | D5 izleme denetimi | `v2/monitoring/*` | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
