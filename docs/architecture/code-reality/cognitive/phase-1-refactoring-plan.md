# Phase 1 — Refactoring Plan · Cognitive

> **Kanonik Birim #7 · Cognitive** — `cognitive_management/` (v2)
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md) (P1: sync/async birleştirme).

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `cognitive_management/` (üst + v2) |
| Boyut | ~20.600 LOC |
| Mevcut test | **24 dosya / ~19.500 LOC — ağırlıklı facade seviyesi** (`test_cognitive_manager_*`) |
| Refactor hedefleri | 2070 LOC facade, sync/async ikizliği, çoklu `ModelAPI`, cache çeşitliliği, izleme yükü |
| Kritik sözleşmeler | `ModelAPI` protokolü · 8-handler zinciri · `CognitiveInput/Output` · pipeline sırası |

> ⚠️ Testler facade'ı yoğun kapsıyor (performance/monitoring/aiops/tracing/events/
> tools/config); **bileşen (handler/critic/ToT/memory) birim testi** görece ince.

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| Facade (manager) | `test_cognitive_manager_{performance,monitoring,aiops,tracing,events,tools,config,...}` | ✅ Çok kapsamlı |
| Handler zinciri (bileşen) | dağınık | 🟡 İnce |
| Critic / ToT / memory (birim) | — | 🟡 İnce |
| Sync↔async parite | — | ❌ Yok |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **8-handler zinciri** birim testleri | [§3.2 / §5](README.md#5-veri--kontrol-akışı--i̇şleme-zinciri) | her handler'ın `handle(ctx)` sözleşmesi |
| T-02 | **Sync↔async parite** | [§8](README.md#8-refactor-sinyalleri--tech-debt) | aynı girdi → sync ve async aynı çıktı |
| T-03 | **Politika yönlendirme eşikleri** | [§5.1](README.md#51-politika-kararı-policyrouterv2--strategy) | entropi→mod seçimi sınır testleri |
| T-04 | **RAG bellek** getirme/güncelleme | [§5.2](README.md#52-bellek-memoryservicev2--iki-katman) | vektör getirme + periyodik özet |
| T-05 | **Critic revizyon** — self-refine tetikleme | [§3.3](README.md#33-bileşenler-v2components) | ihlal→revizyon akışı |
| T-06 | **Çoklu `ModelAPI` protokolü** tutarlılığı | [§8](README.md#8-refactor-sinyalleri--tech-debt) | tek protokol uyum testi |
| T-07 | **ToT** BFS/beam doğruluğu | [§3.3](README.md#33-bileşenler-v2components) | ağaç genişletme/budama |

### A.3 Olması Gereken Test Durumu
Facade kapsamı korunur; handler/critic/ToT/memory **birim** testleriyle
güçlendirilir; sync↔async parite ve tek `ModelAPI` protokolü doğrulanır.

### A.4 Test Sprint'leri
**T1** facade yeşil taban + bileşen kapsam boşluğu haritası.
**T2** handler birim + politika eşikleri (T-01, T-03).
**T3** bellek/critic/ToT + sync-async parite (T-02, T-04, T-05, T-07).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
cognitive_management/
  ├── interfaces/  → TEK ModelAPI protokolü (facade + critic buradan alır)
  ├── processing/  → tek pipeline çekirdeği; sync ve async ince adaptörler
  ├── cognitive_manager.py → yalın facade (event/cache/trace API'leri ayrı modüle)
  └── cache/       → birleşik cache stratejisi
```

### B.2 Geliştirme Sprint'leri *(P1)*
**D1 — Tek `ModelAPI` protokolü** *(düşük risk)*: `cognitive_manager.py` ve
`critic_v2.py`'deki çift tanımı `interfaces/`'te birleştir. Önkoşul: T-06.
**D2 — Sync/async birleştirme** *(yüksek risk)*: `pipeline`+`handlers` ve
`async_pipeline`+`async_handlers`'ı ortak çekirdeğe indir. Önkoşul: T-02 parite.
**D3 — Facade ayrıştırma** *(orta risk)*: 2070 LOC `cognitive_manager.py`'den
event/cache/trace API'lerini ayrı modüllere çıkar.
**D4 — Cache birleştirme** *(orta risk)*: `middleware/cache`, `utils/cache`,
`semantic_cache`, `cache_warming` tek stratejide.
**D5 — İzleme yükü denetimi** *(düşük risk)*: 6 monitoring modülünün aktifliğini
netleştir (ölü/pasif olanı işaretle).

### B.3 Korunacak Sözleşmeler
`ModelAPI` protokolü; 8-handler pipeline sırası; `CognitiveInput/Output` alanları
(özellikle `text`); `handle` genel imzası.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + tablo.

## D. Durum Tablosu
| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 facade taban + boşluk | ⏳ |
| A | T2 handler/politika | ⏳ |
| A | T3 bellek/critic/ToT/parite | ⏳ |
| B | D1 tek ModelAPI | ⏳ |
| B | D2 sync/async birleştirme | ⏳ |
| B | D3 facade ayrıştırma | ⏳ |
| B | D4 cache birleştirme | ⏳ |
| B | D5 izleme denetimi | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
