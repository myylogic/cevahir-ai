# Phase 1 — Refactoring Plan · Model / Engine

> **Kanonik Birim #3 · Model / Engine** — `model/`, `model_management/`
> [research](README.md) ile eşleşir. Akış: **Test Fazı (A)** → **Geliştirme Fazı (B)**.
> Üst plan: [development-roadmap](../../development-roadmap.md).
>
> **Bu sürüm derinleştirilmiştir:** her kalem `dosya:satır` çapası ve mimari doküman
> kesitiyle ilişkilendirilmiştir.

---

## 0. Kapsam

| Alan | Değer |
|------|-------|
| Kaynak | `model/cevahir.py` (2114) + `model_management/` (~5.570) |
| Mevcut test | **6 dosya / ~8.391 LOC** |
| Refactor hedefleri | facade bölme, `ModelManager` çift-rol, config çift kaynak, auto-load sabit yol, ChatPipeline örtüşme, multimodal stub |
| Kritik sözleşmeler | composition-root wiring · `CevahirModelAPI` protokolü · checkpoint save/load · `tie_weights` kuralı · `vocab_size` akışı |

> ⚠️ Ortam: `torch`/`pytest` yok; canlı koşu geliştirme ortamında.

### 0.1 Mimari Referans Haritası

| Referans | Ne için |
|----------|---------|
| [master-architecture §4](../../master-architecture.md#4-bileşen-etkileşim-haritası-component-interaction) | Composition root + adapter köprüsü (bu birimin kalbi) |
| [master-architecture §5](../../master-architecture.md#5-uçtan-uca-akış--çıkarım-inference) | Çıkarım akışı (generate/process) |
| [master-architecture §7](../../master-architecture.md#7-çalışma-zamanı-yolları-özeti) | Çıkarım vs eğitim yolu (facade yalnız çıkarım) |
| [research §4](README.md#4-i̇ç-mimari) | Facade kompozisyonu + adapter + yaşam döngüsü |
| [research §8](README.md#8-refactor-sinyalleri--tech-debt) | Refactor sinyalleri (bu planın kaynağı) |
| [neural-network research §7](../neural-network/README.md#7-bağımlılıklar) | İnşa edilen ağ (bu birim `CevahirNeuralNetwork`'ü kurar) |
| [cognitive research §1](../cognitive/README.md#1-kimlik) | `ModelAPI` protokolü (adapter buna uymalı) |

---

## A. TEST FAZI

### A.1 Test Reality
| Alan | Test | Kaynak (research §3) | Durum |
|------|------|----------------------|-------|
| `Cevahir` facade | `model/tests/test_cevahir.py` (1460), `_comprehensive.py` (1608), `_part2.py` (2055), `_academic.py` (852) | [§3.1](README.md#31-engine--facade-model) | ✅ Çok kapsamlı |
| `ModelManager` | `model_management/test/test_model_manager.py` (369), `_comprehensive.py` (2047) | [§3.2](README.md#32-model-yaşam-döngüsü-model_management) | ✅ Kapsamlı |
| Gerçek vocab | `model/tests/verify_real_vocab_usage.py` | — | 🟡 Betik (pytest değil) |

### A.2 Kusur Envanteri
| # | Kusur/Boşluk | Kod çapası | Mimari ref | Test hedefi |
|---|--------------|-----------|-----------|-------------|
| **T-01** | **Composition root wiring** doğru mu | `cevahir.py:1093` (`__init__`), `:1197` (`_init_tokenizer`), `:1229` (`_init_model`), `:1282` (`_init_cognitive`) | [master §4](../../master-architecture.md#4-bileşen-etkileşim-haritası-component-interaction) · [research §4.1](README.md#41-facade-kompozisyonu-cevahir) | init sonrası tokenizer/model/adapter/cognitive bağ testi |
| **T-02** | **generate determinizmi** (autoregressive + beam) | `cevahir.py:555` (`_autoregressive_generate`), `:754` (`_generate_with_beam_search`), giriş `:493` (`generate`) | [research §4.2](README.md#42-adapter-köprüsü-cevahirmodelapi) | seed'li tekrar → aynı çıktı |
| **T-03** | **Checkpoint round-trip** | `model_management/model_saver.py:245`, `model_loader.py:238` | [research §4.3](README.md#43-model-yaşam-döngüsü-modelmanager) | save→load aynı ağırlık/çıktı |
| **T-04** | **`tie_weights` kuralı** | `model_management/config_schema.py:205` | [neural-network §4](../neural-network/README.md#4-i̇ç-mimari--tam-model-montajı) | `seq_proj_dim!=embed_dim` → hata |
| **T-05** | **Config çift kaynak** tutarsızlığı checkpoint kırıyor | `cevahir.py:365` (`CevahirConfig`) ↔ `training_system/train.py` `TRAIN_CONFIG` | [research §8](README.md#8-refactor-sinyalleri--tech-debt) · [master §7](../../master-architecture.md#7-çalışma-zamanı-yolları-özeti) | tutarsız config → net erken hata |
| **T-06** | **Adapter protokol uyumu** | `cevahir.py:468` (`CevahirModelAPI`) | [cognitive §1](../cognitive/README.md#1-kimlik) (`ModelAPI`) | `generate/embed/forward/entropy_estimate/score` uyum |
| **T-07** | **Multimodal stub** gerçek mi | `cevahir.py:1014` (`process_audio`), `:1020` (`process_image`), `:1026` (`process_multimodal`) | [research §8](README.md#8-refactor-sinyalleri--tech-debt) | stub/gerçek durum testi |

### A.3 Olması Gereken Test Durumu
Composition wiring (T-01), üretim determinizmi (T-02), checkpoint round-trip (T-03),
config sözleşmesi (T-04/T-05), adapter protokol uyumu (T-06) tam kapsanır;
multimodal durumu netleşir (T-07). T-01/T-04/T-06 Geliştirme Fazı önkoşuludur.

### A.4 Test Sprint'leri
**T1** yeşil taban (6 dosya; `verify_real_vocab_usage`'ı pytest'e al).
**T2** wiring + protokol + tying (T-01, T-04, T-06).
**T3** determinizm + checkpoint + config sözleşmesi (T-02, T-03, T-05).

---

## B. GELİŞTİRME FAZI

### B.1 Hedef Mimari
```
model/
  ├── config.py     → CevahirConfig            (cevahir.py:365'ten ayrılır)
  ├── model_api.py  → CevahirModelAPI          (cevahir.py:468'ten ayrılır — protokol tek kaynak)
  └── cevahir.py    → Cevahir facade           (yalnız kompozisyon + genel API)
model_management/
  ├── inference/    → build/load/generate
  └── training/     → optimizer/criterion/scheduler  (ISP: eğitim ayrı)
config: config_schema TEK kaynak → cevahir + train.py ondan türer
```
> Kompozisyon akışı [master §4](../../master-architecture.md#4-bileşen-etkileşim-haritası-component-interaction)
> ile aynı kalır; yalnız dosya sınırları netleşir.

### B.2 Geliştirme Sprint'leri
**D1 — Facade bölme** *(orta risk)*: `cevahir.py`'yi `CevahirConfig`/`CevahirModelAPI`/
`Cevahir` dosyalarına ayır (import geriye uyumlu). Önkoşul: T2/T3. **Kabul:** dış API
imzaları aynı; testler yeşil.
**D2 — Config tek kaynağı** *(orta risk, roadmap P1)*: `cevahir.py:365` ↔ `train.py`
çift parametresini `config_schema` tek kaynağına bağla. **Neural-network D4 ile ortak.**
Önkoşul: T-05. **Kabul:** README'deki "iki yerde güncelle" uyarısı kalkar.
**D3 — ModelManager rol ayrımı** *(orta risk)*: eğitim vs çıkarım bileşenlerini ayır
([research §4.3 notu](README.md#43-model-yaşam-döngüsü-modelmanager)). **Kabul:** çıkarım
yolu optimizer kurmadan çalışır.
**D4 — Auto-load yolu config'e** *(düşük risk)*: `cevahir.py:1229`'daki sabit
`saved_models/cevahir_model.pth` → config alanı. **Kabul:** yol config'ten okunur.
**D5 — Multimodal netleştirme** *(düşük risk)*: stub'ları uygula veya "not implemented"
işaretle. Önkoşul: T-07.

### B.3 Korunacak Sözleşmeler
| Sözleşme | Kaynak | Neden |
|----------|--------|-------|
| `Cevahir` genel API imzaları | `cevahir.py:1050+` | Chatting/API/terminal bağlı |
| `CevahirModelAPI` protokolü | `cevahir.py:468` | Cognitive bu protokole bağlı ([cognitive §7](../cognitive/README.md#7-bağımlılıklar)) |
| checkpoint formatı | `model_saver.py`/`model_loader.py` | eğitilmiş ağırlık uyumu |
| `tie_weights` + `vocab_size` akışı | `config_schema.py:205`, `cevahir.py:1229` | shape uyumu ([master §10](../../master-architecture.md#10-bağımlılık-yönü-ve-kural-i̇hlalleri)) |

---

## C. Kod ↔ Doküman Tutarlılığı
Her sprint: kod + [research §8](README.md#8-refactor-sinyalleri--tech-debt) + §D +
gerekiyorsa [roadmap](../../development-roadmap.md#5-i̇zleme). D1 facade bölme,
research §3/§9 anchor'larını etkiler → aynı PR'da güncellenir.

## D. Durum Tablosu
| Faz | Sprint | Kod çapası | Durum |
|-----|--------|-----------|-------|
| A | T1 yeşil taban | 6 test dosyası | ⏳ |
| A | T2 wiring/protokol/tying | `cevahir.py:1093/468`, `config_schema.py:205` | ⏳ |
| A | T3 determinizm/checkpoint/config | `cevahir.py:555/754`, `model_saver/loader` | ⏳ |
| B | D1 facade bölme | `cevahir.py:365/468/1050` | ⏳ |
| B | D2 config tek kaynağı | `cevahir.py:365` ↔ `train.py` | ⏳ |
| B | D3 ModelManager rol ayrımı | `model_manager.py` | ⏳ |
| B | D4 auto-load config | `cevahir.py:1229` | ⏳ |
| B | D5 multimodal | `cevahir.py:1014-1026` | ⏳ |

*✅ tamam · 🔄 devam · ⏳ planlandı · ⛔ engelli*
