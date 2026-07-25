# Phase 1 — Refactoring Plan · Model / Engine

> **Kanonik Birim #3 · Model / Engine** — `model/`, `model_management/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md).

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `model/` (facade) + `model_management/` (yaşam döngüsü) |
| Boyut | ~7.700 LOC |
| Mevcut test | **6 dosya / ~8.391 LOC** (cevahir + model_manager, comprehensive + academic) |
| Refactor hedefleri | 2114 LOC facade bölme, `ModelManager` çift-rol, config çift kaynak, auto-load sabit yol, ChatPipeline örtüşme |
| Kritik sözleşmeler | composition-root wiring · `CevahirModelAPI` protokolü · checkpoint save/load · `tie_weights` kuralı |

> ⚠️ Ortam: `torch`/`pytest` yok; canlı koşu geliştirme ortamında.

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Durum |
|------|------|-------|
| `Cevahir` facade | `model/tests/test_cevahir.py` (1460), `_comprehensive.py` (1608), `_part2.py` (2055), `_academic.py` (852) | ✅ Çok kapsamlı |
| `ModelManager` | `model_management/test/test_model_manager.py` (369), `_comprehensive.py` (2047) | ✅ Kapsamlı |
| Gerçek vocab | `model/tests/verify_real_vocab_usage.py` | 🟡 Betik |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | research | Test hedefi |
|---|--------------|----------|-------------|
| T-01 | **Composition root wiring** — tokenizer→model→adapter→cognitive doğru kuruluyor mu | [§4.1](README.md#41-facade-kompozisyonu-cevahir) | init sonrası bileşen bağ testi |
| T-02 | **generate determinizmi** — autoregressive + beam aynı seed'de tekrarlanabilir | [§4.2](README.md#42-adapter-köprüsü-cevahirmodelapi) | seed'li üretim testi |
| T-03 | **Checkpoint round-trip** — save→load aynı ağırlık/çıktı | [§4.3](README.md#43-model-yaşam-döngüsü-modelmanager) | save/load eşdeğerlik |
| T-04 | **`tie_weights` kuralı** doğrulaması | [config_schema](README.md#31-engine--facade-model) | seq_proj_dim==embed_dim ihlali hata veriyor mu |
| T-05 | **Config çift kaynak** — cevahir.py ↔ train.py tutarsızlığı checkpoint kırıyor | [§8](README.md#8-refactor-sinyalleri--tech-debt) | tutarsız config → net hata testi |
| T-06 | **Adapter protokol uyumu** — `CevahirModelAPI` `CognitiveModelAPI`'yi tam karşılıyor mu | [§4.2](README.md#42-adapter-köprüsü-cevahirmodelapi) | protokol uyum testi |
| T-07 | **Multimodal stub** — `process_audio/image/multimodal` gerçek mi | [§8](README.md#8-refactor-sinyalleri--tech-debt) | stub/gerçek durum testi |

### A.3 Olması Gereken Test Durumu
Composition wiring, üretim determinizmi, checkpoint round-trip, config sözleşme
doğrulaması, adapter protokol uyumu tam kapsanır; multimodal durumu netleşir.

### A.4 Test Sprint'leri
**T1** yeşil taban (6 dosya koş, `verify_real_vocab_usage`'ı pytest'e al).
**T2** wiring + protokol + tying (T-01, T-04, T-06).
**T3** üretim determinizmi + checkpoint + config sözleşmesi (T-02, T-03, T-05).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
model/
  ├── config.py        → CevahirConfig (ayrı)
  ├── model_api.py     → CevahirModelAPI (ayrı, protokol tek kaynak)
  └── cevahir.py       → Cevahir facade (yalnız kompozisyon + genel API)
model_management/
  ├── (çıkarım yolu)   → build/load/generate
  └── (eğitim yolu)    → optimizer/criterion/scheduler  (ayrı sorumluluk)
config: tek şema kaynağı (config_schema) → cevahir + train.py ondan türer
```

### B.2 Geliştirme Sprint'leri
**D1 — Facade bölme** *(orta risk)*: `cevahir.py`'yi `CevahirConfig` / `CevahirModelAPI`
/ `Cevahir` dosyalarına ayır (import geriye uyumlu). Koruyan: T2/T3.
**D2 — Config tek kaynağı** *(orta risk, roadmap P1)*: `model/cevahir.py` ve
`training_system/train.py` çift parametresini `config_schema` tek kaynağına bağla.
README'deki "iki yerde güncelle" uyarısı kalkar. Önkoşul: T-05.
**D3 — ModelManager rol ayrımı** *(orta risk)*: eğitim vs çıkarım bileşenlerini
ayır (ISP). Koruyan: model_manager comprehensive testleri.
**D4 — Auto-load yolu config'e** *(düşük risk)*: `saved_models/cevahir_model.pth`
sabitini config alanına taşı.
**D5 — Multimodal netleştirme** *(düşük risk)*: stub'ları ya uygula ya açıkça
"not implemented" işaretle.

### B.3 Korunacak Sözleşmeler
Genel `Cevahir` API imzaları; `CevahirModelAPI` protokolü; checkpoint formatı;
`tie_weights` kuralı; `vocab_size` akışı.

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + tablo.

## D. Durum Tablosu
| Faz | Sprint | Durum |
|-----|--------|-------|
| A | T1 yeşil taban | ⏳ |
| A | T2 wiring/protokol/tying | ⏳ |
| A | T3 determinizm/checkpoint/config | ⏳ |
| B | D1 facade bölme | ⏳ |
| B | D2 config tek kaynağı | ⏳ |
| B | D3 ModelManager rol ayrımı | ⏳ |
| B | D4 auto-load config | ⏳ |
| B | D5 multimodal | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
