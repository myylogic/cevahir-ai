# Cevahir — Master Architecture

> **Amaç:** Bu doküman, Cevahir'in **mevcut (as-is)** yazılım mimarisini mühendislik
> dilinde, uçtan uca tarif eder. Sonraki büyük güncelleme başlamadan önce sistemin
> tam bir referans haritasını sağlamak için yazılmıştır. Buradaki her ifade **gerçek
> koddan** çıkarılmıştır; tasarım temennisi değil, kodun kendisidir.
>
> **Kapsam:** Sistemin bütünsel görünümü, katmanlar, kanonik birimler, uçtan uca akışlar,
> bağımlılık yönü, çalışma zamanı yolları (eğitim vs. çıkarım).
> Her kanonik birimin dosya-satır düzeyindeki ayrıntısı `code-reality/` altındadır.
>
> **Kardeş dokümanlar:**
> - [`architecture-search-index.md`](architecture-search-index.md) — tüm birimlerin arama/gezinme indeksi
> - [`code-reality/`](code-reality/) — kodun mimarideki birebir karşılığı

---

## 1. Cevahir Nedir? (Mühendislik Tanımı)

Cevahir, **tek bir depo içinde uçtan uca çalışan bir dil modeli (LLM) motorudur.**
Tokenizer eğitiminden başlayıp, transformer decoder'ın sıfırdan inşasına, eğitim
orkestrasyonuna, çıkarıma (inference) ve bilişsel akıl yürütme (cognitive reasoning)
katmanına kadar tüm zinciri kapsar. Mühendislik açısından sistem **üç farklı sorumluluk
ekseninde** ele alınabilir:

| Eksen | Sorumluluk | Baskın çalışma zamanı |
|-------|------------|-----------------------|
| **Veri → Sözlük** | Ham metnin toplanması, işlenmesi ve BPE sözlüğüne dönüştürülmesi | Batch / offline |
| **Model → Ağırlık** | Transformer decoder'ın tanımı, eğitimi, kaydı/yüklenmesi | Batch (eğitim) |
| **Sorgu → Yanıt** | Çıkarım, bilişsel strateji, bellek, araç kullanımı, sohbet, API | Online / etkileşimli |

Sistemin **birleştirici cephe (facade)** noktası `model/cevahir.py` içindeki `Cevahir`
sınıfıdır; ancak bu sınıf **yalnızca çıkarım** içindir. Eğitim tamamen ayrı bir yol
izler (bkz. §6). Bu ayrım, sistemin en önemli mimari kararlarından biridir ve
kodda açıkça belgelenmiştir.

---

## 2. Katmanlı Görünüm (Layered View)

Sistem, aşağıdan yukarıya doğru şu katmanlardan oluşur. Her katman yalnızca kendisinden
alttaki katmanlara bağımlıdır (bağımlılık yönü yukarıdan aşağıya).

```
┌───────────────────────────────────────────────────────────────────────────┐
│  L6  SUNUM / ERİŞİM KATMANI                                                  │
│      api/ (FastAPI routes v3, services)   ·   chatting_management/           │
│      REST uçları, oturum, sohbet akışı, kullanıcı yönetimi                   │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────────┐
│  L5  BİLİŞSEL KATMAN (Cognitive)                                             │
│      cognitive_management/ (v2)                                              │
│      Strateji (direct/think/debate/tot), bellek (RAG), critic, araçlar,     │
│      middleware, event bus, DI container, izleme (AIOps)                     │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────────┐
│  L4  BİRLEŞTİRİCİ MOTOR (Unified Engine — Facade)                            │
│      model/cevahir.py  →  Cevahir                                            │
│      encode/decode · forward · generate · process · batch · tools · memory  │
│      CevahirModelAPI (adapter: NN ↔ Cognitive protokolü)                    │
└──────────────┬───────────────────────────────┬────────────────────────────┘
               │                               │
┌──────────────▼──────────────┐   ┌────────────▼───────────────────────────────┐
│  L3a  MODEL YAŞAM DÖNGÜSÜ    │   │  L3b  ÇEKİRDEK SİNİR AĞI (Neural Network)   │
│  model_management/          │   │  src/neural_network_module/                │
│  ModelManager, loader,      │   │  Transformer decoder: embedding, RoPE,     │
│  saver, initializer,        │   │  attention, FFN/SwiGLU, RMSNorm, MoE,      │
│  updater, profiler,         │   │  KV cache, quantization, checkpointing     │
│  chat_pipeline, health      │   │                                            │
└──────────────┬──────────────┘   └────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────────────────┐
│  L2  EĞİTİM KATMANI (yalnızca eğitim yolunda aktif)                          │
│      training_system/ (v2/v3)   ·   training_management/ (v2/v3)             │
│      Eğitim döngüsü, cache, curriculum, optimizasyon, safety, metrics        │
└──────────────┬──────────────────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────┐   ┌────────────────────────────────────────────┐
│  L1  TOKENIZER              │   │  L0  VERİ KATMANI                          │
│  tokenizer_management/      │   │  data_loader_management/ · data_processing/│
│  TokenizerCore, BPE         │◄──│  database/ (repositories, schemas)         │
│  (encode/decode/train),     │   │  Ham metin toplama, temizleme, yükleme,    │
│  morfoloji, heceleme        │   │  kalıcılık                                  │
└─────────────────────────────┘   └────────────────────────────────────────────┘
```

> **Not:** Katman numaraları sorumluluk seviyesini gösterir, çağrı sırasını değil.
> Örneğin çıkarımda akış L6 → L5 → L4 → (L3a + L3b) → L1 şeklinde aşağı iner.

---

## 3. Kanonik Birimler (Canonical Units)

Sistem, aşağıdaki **kanonik birimlere** ayrılır. Her birim, tek bir sorumluluk
alanına sahip bir kod kümesidir ve `code-reality/` altında kendi klasöründe
ayrıntılı olarak haritalanır.

| # | Kanonik Birim | Kaynak Dizin(ler) | Ana Sorumluluk | Çalışma Zamanı |
|---|---------------|-------------------|----------------|----------------|
| 1 | **Tokenizer** | `tokenizer_management/` | Türkçe-odaklı BPE sözlük eğitimi + encode/decode | Batch + Online |
| 2 | **Neural Network** | `src/neural_network_module/` | Transformer decoder çekirdek bileşenleri | Batch + Online |
| 3 | **Model / Engine** | `model/`, `model_management/` | `Cevahir` facade + model yaşam döngüsü | Online |
| 4 | **Training Management** | `training_management/` (v2/v3) | Eğitim orkestrasyonu, metrik, güvenlik | Batch |
| 5 | **Training System** | `training_system/` (v2/v3) | Eğitim runtime, cache, veri boru hattı | Batch |
| 6 | **Data Loader** | `data_loader_management/` | Eğitim verisinin yüklenmesi | Batch |
| 7 | **Data Processing** | `data_processing/` | Scraping, altyazı/Wikipedia veri hazırlığı | Offline |
| 8 | **Cognitive** | `cognitive_management/` (v2) | Akıl yürütme, bellek, critic, araç kullanımı | Online |
| 9 | **Chatting** | `chatting_management/` | Oturum, geçmiş, sohbet akışı | Online |
| 10 | **API** | `api/` | REST erişim katmanı (routes v3, services) | Online |
| 11 | **Database** | `database/` | Kalıcılık (repository, schema, interface) | Online |

**Yardımcı/çevresel dizinler** (kanonik birim sayılmaz ama haritada anılır):
`scripts/` (yardımcı betikler), `tests/` (üst düzey testler), `education/`,
`image/`, `data/`, `dataset_subtitle/`.

---

## 4. Bileşen Etkileşim Haritası (Component Interaction)

Aşağıdaki diyagram, **çıkarım (inference)** yolundaki gerçek nesne kompozisyonunu
gösterir. `Cevahir.__init__` bu grafiği bizzat kurar (composition root).

```
                         ┌─────────────────────────┐
                         │      CevahirConfig       │
                         │  (validate → alt config) │
                         └────────────┬────────────┘
                                      │ compose
                                      ▼
        ┌──────────────────────────────────────────────────────────┐
        │                      Cevahir  (Facade)                     │
        │              model/cevahir.py :: class Cevahir             │
        └───────┬──────────────────┬───────────────────┬───────────┘
                │ _init_tokenizer   │ _init_model        │ _init_cognitive
                ▼                   ▼                    ▼
      ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
      │  TokenizerCore   │  │   ModelManager   │  │  CognitiveManager    │
      │  tokenizer_mgmt/ │  │  model_mgmt/     │  │  cognitive_mgmt/     │
      │  core            │  │  (V-4 NN sarar)  │  │  v2                  │
      └──────────────────┘  └────────┬─────────┘  └──────────┬───────────┘
                                     │                        │ ihtiyaç: ModelAPI
                                     │ wraps                  │
                                     ▼                        │
                          ┌──────────────────────┐            │
                          │ src/neural_network_  │            │
                          │ module (Transformer) │            │
                          └──────────────────────┘            │
                                                              │
                ┌─────────────────────────────────────────────┘
                ▼  (adapter köprüsü)
      ┌──────────────────────────────────────────────┐
      │  CevahirModelAPI  (implements CognitiveModelAPI)│
      │  ModelManager + TokenizerCore → generate/embed/ │
      │  forward/entropy_estimate/score  arayüzü        │
      └──────────────────────────────────────────────┘
```

**Kilit mimari nokta — Adapter köprüsü:** `CognitiveManager` doğrudan `ModelManager`'a
bağımlı değildir. `Cevahir`, `ModelManager` + `TokenizerCore`'u `CevahirModelAPI`
adaptörüyle sarar ve bu adaptörü bilişsel katmana enjekte eder. Böylece bilişsel katman
somut sinir ağına değil, `CognitiveModelAPI` **protokolüne** bağımlı olur
(Dependency Inversion). Refactor açısından bu, sinir ağını değiştirirken bilişsel
katmanın el değmeden kalabileceği anlamına gelir.

---

## 5. Uçtan Uca Akış — Çıkarım (Inference)

`cevahir.process("...")` çağrısının izlediği yol:

```
Kullanıcı metni (str)
   │
   ▼
[L6] API / ChattingManager           (opsiyonel — doğrudan da çağrılabilir)
   │
   ▼
[L4] Cevahir.process()               model/cevahir.py
   │   └─ CognitiveManager.handle()   'e devreder
   ▼
[L5] CognitiveManager (v2)           cognitive_management/
   │   ├─ Feature extraction (entropi, domain, karmaşıklık)
   │   ├─ Memory retrieval (RAG / oturum geçmişi)
   │   ├─ Policy routing (mod seçimi: direct/think/debate/tot)
   │   ├─ Deliberation (CoT / ToT / debate adımları)
   │   ├─ Generation  ─────────────┐
   │   ├─ Critic (güvenlik/doğrulama/revizyon)
   │   └─ Memory update            │
   ▼                               │ backend.generate()
[L4] CevahirModelAPI.generate() ◄──┘
   │   └─ ModelManager + TokenizerCore
   ▼
[L3] ModelManager.forward/generate() model_management/
   │   └─ Transformer decoder ileri geçiş
   ▼
[L3b] src/neural_network_module      embedding → RoPE → attention → FFN → RMSNorm → logits
   │
   ▼
[L1] TokenizerCore.decode()          token id'leri → metin
   │
   ▼
CognitiveOutput (text, mode, metadata) → kullanıcıya
```

**Saf üretim (cognitive'siz):** `cevahir.generate("...")` bilişsel katmanı atlar ve
doğrudan `CevahirModelAPI` üzerinden otoregresif/beam-search üretim yapar
(`_autoregressive_generate`, `_generate_with_beam_search`).

---

## 6. Uçtan Uca Akış — Eğitim (Training)

Eğitim **`Cevahir` facade'ını kullanmaz.** Ayrı giriş noktaları vardır:

```
① SÖZLÜK EĞİTİMİ
   Ham metin  →  tokenizer_management/train_bpe.py
                  └─ BPETrainer → vocab + merges  →  disk
                     (prepare_bpe_cache.py ile cache hızlandırma)

② VERİ HAZIRLIĞI
   data_processing/ (scraping, altyazı, Wikipedia)
       →  data_loader_management/ (yükleme)
          →  training_system/ cache (data_cache.py, prepare_cache.py)

③ MODEL EĞİTİMİ
   training_system/train.py            (giriş noktası)
       →  training_system/v3 (runtime, veri, config doğrulama)
          →  training_management/v3    (orkestrasyon)
             ├─ core        (eğitim döngüsü)
             ├─ curriculum  (müfredat)
             ├─ optimizers  (optimizasyon stratejileri)
             ├─ monitoring  (metrik, TensorBoard)
             └─ safety      (kararlılık/koruma)
                 │
                 ▼ kullanır
             model_management/ModelManager
                 │
                 ▼ eğitir
             src/neural_network_module (Transformer decoder)
                 │
                 ▼ kaydeder
             model_management/model_saver → checkpoint (.pth)
```

**v2 vs v3:** Hem `training_system` hem `training_management` altında paralel `v2/`
ve `v3/` uygulamaları bulunur. Bu, sistemin en belirgin **refactor sinyalidir** ve
`code-reality/` ilgili birimlerde ayrıntılı ele alınır (hangi sürüm aktif, hangi
sorumluluklar tekrar ediyor).

---

## 7. Çalışma Zamanı Yolları Özeti

| Yol | Giriş Noktası | Kullanılan Birimler | `Cevahir` facade? |
|-----|---------------|---------------------|-------------------|
| **Sözlük eğitimi** | `tokenizer_management/train_bpe.py` | Tokenizer | ❌ |
| **Model eğitimi** | `training_system/train.py` | Training System/Mgmt, Model Mgmt, NN, Tokenizer | ❌ |
| **Saf üretim** | `Cevahir.generate()` | Engine, Model Mgmt, NN, Tokenizer | ✅ |
| **Bilişsel yanıt** | `Cevahir.process()` | Engine, Cognitive, Model Mgmt, NN, Tokenizer | ✅ |
| **REST/sohbet** | `api/routes/v3` → `chatting_management` | API, Chatting, Engine, Cognitive | ✅ (dolaylı) |

---

## 8. Kesişen İlgiler (Cross-Cutting Concerns)

Bu ilgiler tek bir birime ait değildir; sistemin geneline yayılır.

| İlgi | Nerede | Not |
|------|--------|-----|
| **Konfigürasyon** | `CevahirConfig` + her birimin kendi `config.py`'si | Merkezi facade config + birim-yerel config'ler |
| **Loglama** | Python `logging`, birim bazlı logger'lar | Standart kütüphane |
| **İzleme (Observability)** | `cognitive_management/v2/monitoring`, `api/monitoring`, `model_management/health_monitor` | AIOps: anomali, trend, alert |
| **Hata yönetimi** | Her birimde özel exception hiyerarşisi (`*Error`) | `CevahirError`, `TokenizerCoreError`, vb. |
| **Test** | Her birim içinde `test/`+`tests/`, kök `tests/` | Birim-yerel + entegrasyon |
| **Cihaz yönetimi (CPU/CUDA)** | `CevahirConfig.device` → alt birimlere yayılır | Seed + device tek noktadan set edilir |
| **Reprodüksiyon** | `Cevahir.__init__` içinde seed yönetimi | random + torch + cuda seed |

---

## 9. Mimari Kararlar ve Uygulanan Desenler

Kodda **fiilen gözlemlenen** desenler (temenni değil):

| Desen | Uygulandığı Yer | Amaç |
|-------|-----------------|------|
| **Facade** | `Cevahir` | Üç alt sistemi tek API arkasında birleştirir |
| **Adapter** | `CevahirModelAPI` | NN'i bilişsel protokole uyarlar |
| **Composition Root** | `Cevahir.__init__` | Tüm bağımlılıkları tek noktada kurar |
| **Dependency Injection** | `Cevahir(tokenizer_core=..., model_manager=..., cognitive_manager=...)` | Bileşenler dışarıdan enjekte edilebilir |
| **Protocol-based interface** | `CognitiveModelAPI`, cognitive `interfaces/` | Somut sınıf yerine protokole bağımlılık |
| **Chain of Responsibility** | Cognitive `processing/` handler zinciri | Her handler tek sorumluluk |
| **Strategy** | Cognitive policy router (mod seçimi) | Runtime'da algoritma seçimi |
| **Repository** | `database/repositories`, cognitive VectorStore | Kalıcılık soyutlaması |
| **Manager per domain** | `*Manager` sınıfları (Tokenizer, Model, Cognitive, Chatting) | Her alanın kendi yöneticisi |
| **Versioned modules** | `v2/`, `v3/` (training, cognitive) | Kademeli evrim (refactor borcu içerir) |

---

## 10. Bağımlılık Yönü ve Kural İhlalleri

**Sağlıklı bağımlılık yönü** (yukarıdan aşağıya): API → Chatting → Engine → Cognitive
→ Model Mgmt → NN → Tokenizer → Data.

Refactor sırasında dikkat edilecek **potansiyel kırılma noktaları** (`code-reality/`
birimlerinde teyit edilecek):

1. **v2/v3 ikizliği** — `training_system` ve `training_management` altında paralel
   sürümler; hangisinin kanonik olduğu netleştirilmeli.
2. **Facade yalnızca inference** — eğitim yolunun facade dışında olması bilinçli bir
   karardır; ancak eğitim tarafında `model_management` hem eğitim hem çıkarım
   sorumluluğu taşıyabilir (teyit edilecek).
3. **Bilişsel katmanın backend'e erişimi** — yalnızca `CevahirModelAPI` adaptörü
   üzerinden olmalı; doğrudan `ModelManager` sızıntısı aranacak.

---

## 11. Bu Dokümanı Nasıl Okumalı?

1. **Genel resim** için bu dokümanı okuyun (§2 katmanlar, §4 kompozisyon, §5–6 akışlar).
2. **Bir birimde çalışacaksanız** [`architecture-search-index.md`](architecture-search-index.md)
   üzerinden ilgili `code-reality/<birim>/` klasörüne gidin.
3. **Refactor planlarken** her birimin `README.md`'sindeki "Refactor Sinyalleri /
   Tech-Debt" bölümüne bakın; oradaki bulgular §10 ile birlikte değerlendirilir.

---

*Bu doküman kodun mevcut halini yansıtır. Kod değiştikçe güncellenmelidir.
Dosya-satır referansları için `code-reality/` alt dokümanlarına bakın.*
