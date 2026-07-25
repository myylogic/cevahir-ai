# Code Reality — Model / Engine

> Kanonik Birim #3 · Kaynak: `model/` + `model_management/`
> Birleştirici çıkarım motoru (`Cevahir` facade) ve model yaşam döngüsünün
> **kodundan çıkarılmış** mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L4 + L3a), [search index](../../architecture-search-index.md).

> 🔧 **Eşleşen plan:** Bu research ile birlikte okunacak test+geliştirme planı →
> [`phase-1-refactoring-plan.md`](phase-1-refactoring-plan.md)

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Model / Engine |
| **Kaynak dizin** | `model/` (facade), `model_management/` (yaşam döngüsü) |
| **Toplam boyut** | ~7.700 LOC (2.114 `cevahir.py` + ~5.570 `model_management/`; testler hariç) |
| **Facade sınıfı** | `Cevahir` — `model/cevahir.py:1050` (composition root) |
| **Adaptör** | `CevahirModelAPI(CognitiveModelAPI)` — `model/cevahir.py:468` |
| **Yaşam döngüsü** | `ModelManager` — `model_management/model_manager.py:107` |
| **Çalışma zamanı** | Online (çıkarım) — eğitim tarafı `ModelManager` üzerinden de kullanılır |
| **Dış bağımlılık** | Tokenizer, Neural Network, Cognitive; `torch` |

> **Kritik ayrım:** `model/cevahir.py` **yalnızca çıkarım** içindir (kodda açıkça
> belgelenmiş). Eğitim `training_system/` üzerinden gider ama `ModelManager`'ı
> paylaşır. Yani `model_management` **hem eğitim hem çıkarım** tarafından kullanılır;
> `Cevahir` facade'ı ise sadece çıkarımdır.

---

## 2. Sorumluluk

İki ayrı ama komşu sorumluluk:

1. **`model/` (Engine/Facade):** Tokenizer + Model + Cognitive'i tek bir `Cevahir`
   API'si arkasında birleştirmek; encode/decode, forward, generate (autoregressive +
   beam search), process (bilişsel), batch işlemler, bellek ve araç API'leri.
2. **`model_management/` (Lifecycle):** Sinir ağı modelinin yaşam döngüsü — inşa
   (model+optimizer+criterion+scheduler), kaydetme/yükleme (checkpoint), güncelleme,
   sağlık izleme, profil çıkarma, config şeması doğrulama.

**Kapsam dışı:** Transformer matematiği (Neural Network), tokenizasyon (Tokenizer),
akıl yürütme (Cognitive), eğitim döngüsü (Training System).

---

## 3. Dosya Envanteri

### 3.1 Engine / Facade (`model/`)

| Dosya | LOC | Sınıf/rol | Anahtar üyeler |
|-------|-----|-----------|----------------|
| `model/cevahir.py` | 2114 | **`Cevahir`** (facade), **`CevahirModelAPI`** (adapter), **`CevahirConfig`** (config) | `__init__` (composition root), `_init_tokenizer/_init_model/_init_cognitive`, `encode`, `decode`, `forward`, `generate`, `process`, `generate_batch`, `process_batch`, `train`, `save_model`, `load_model`, `register_tool`, `add_memory`, `get_health_status` |

`CevahirModelAPI` içindeki üretim çekirdeği: `generate`, `_autoregressive_generate`,
`_generate_with_beam_search`, `score`, `entropy_estimate`, `process_multimodal`.

`model/tests/` — `test_cevahir*.py` (kapsamlı + akademik + gerçek vocab doğrulama).

### 3.2 Model Yaşam Döngüsü (`model_management/`)

| Dosya | LOC | Sınıf | Rol / anahtar üyeler |
|-------|-----|-------|----------------------|
| `model_manager.py` | 1082 | **`ModelManager`** | Orkestratör: `build_model`, `build_optimizer`, `build_criterion`, `build_scheduler`, `initialize`, `configure_tensorboard`, `load`, `save`, device yönetimi |
| `model_initializer.py` | 765 | **`ModelInitializer`** | `build_model` (CevahirNeuralNetwork), `initialize_optimizer/criterion/scheduler`, `build_training_components` |
| `model_loader.py` | 466 | **`ModelLoader`** | `load_model`, `load_optimizer`, `load_scheduler`, `load_checkpoint_raw`, `load_all` |
| `model_saver.py` | 433 | **`ModelSaver`** | `save_checkpoint`, `save_weights_only`, `save_full_model`, `save_additional_info`, `save_model` |
| `model_updater.py` | 496 | **`ModelUpdater`**, **`UpdateReport`** | `update_model`, `update_optimizer`, `update_scheduler`, `update_learning_rate`, `bulk_update` |
| `chat_pipeline.py` | 248 | **`ChatPipeline`** | Terminal sohbet: `generate`, `process_input`, `run`, `main` |
| `health_monitor.py` | 581 | **`ModelHealthMonitor`** + `GradientHealth`/`WeightHealth`/`AttentionHealth`/`HealthReport` | Eğitim sağlığı: gradyan/ağırlık/dikkat anomalileri |
| `profiler.py` | 542 | **`ParamStats`/`MemorySnapshot`/`FlopEstimate`/`TimingResult`** | Parametre/bellek/FLOP/zaman profili |
| `config_schema.py` | 486 | **`ModelArchConfig`/`TrainingConfig`/`CheckpointConfig`** (`_SchemaBase`) | Tipli config + `validate` (tie_weights kuralı) |
| `exceptions.py` | 303 | `CevahirModelError` hiyerarşisi | `ModelNotInitializedError`, `ForwardError`, `ShapeError`, `CheckpointError`, `QuantizationError`, `DeviceError`, ... |

---

## 4. İç Mimari

### 4.1 Facade Kompozisyonu (`Cevahir`)

`Cevahir.__init__` sistemin **composition root**'udur — tüm alt sistemleri kurar
ve birbirine bağlar:

```
Cevahir(config)                              model/cevahir.py:1050
  │
  ├─ CevahirConfig.validate()
  ├─ seed set (random + torch + cuda)
  │
  ├─ _init_tokenizer()  → TokenizerCore              [Tokenizer birimi]
  ├─ _init_model()      → ModelManager.initialize()  [bu birim → NN birimi]
  │                        · vocab_size = tokenizer.get_vocab_size()   ← kontrat
  │                        · (ops.) auto-load saved_models/cevahir_model.pth
  ├─ CevahirModelAPI(ModelManager, TokenizerCore)   ← ADAPTER köprüsü
  └─ _init_cognitive()  → CognitiveManager(model_api) [Cognitive birimi]
```

### 4.2 Adapter Köprüsü (`CevahirModelAPI`)

`CognitiveModelAPI` protokolünü uygular; bilişsel katmanın ihtiyaç duyduğu
`generate/embed/forward/entropy_estimate/score` metotlarını `ModelManager` +
`TokenizerCore` üzerinden sağlar. Böylece **Cognitive somut modele değil protokole
bağımlıdır** (Dependency Inversion). Üretim mantığının çekirdeği burada yaşar:

```
CevahirModelAPI.generate()
  ├─ _autoregressive_generate()   (greedy/sampling; KV cache ile)
  └─ _generate_with_beam_search() (beam search)
score() / entropy_estimate()      (kalite + belirsizlik ölçümü — Cognitive kullanır)
```

### 4.3 Model Yaşam Döngüsü (`ModelManager`)

```
ModelManager.initialize(build_optimizer, build_criterion, build_scheduler)
  │
  ├─ build_model()      → ModelInitializer.build_model(CevahirNeuralNetwork, cfg)
  ├─ build_optimizer()  → ModelInitializer.initialize_optimizer(...)   (eğitimde)
  ├─ build_criterion()  → ...initialize_criterion(...)                 (eğitimde)
  └─ build_scheduler()  → ...initialize_scheduler(...)                 (eğitimde)

ModelManager.save/load ↔ ModelSaver / ModelLoader (checkpoint IO)
ModelManager + ModelUpdater  → runtime parametre/lr güncelleme
```

> **Çıkarım vs eğitim:** `Cevahir._init_model` optimizer/criterion/scheduler'ı
> `False` ile atlar (çıkarım). Eğitim yolu (`training_system`) bunları `True` ile
> kurar. Aynı `ModelManager`, iki modda farklı bileşen kümesiyle çalışır.

---

## 5. Veri / Kontrol Akışı

### 5.1 `cevahir.generate()` (saf üretim — cognitive'siz)

```
prompt → TokenizerCore.encode → CevahirModelAPI._autoregressive_generate
   → ModelManager.model.forward (CevahirNeuralNetwork) → logits
   → sampling/greedy döngüsü (KV cache) → id'ler → TokenizerCore.decode → text
```

### 5.2 `cevahir.process()` (bilişsel yanıt)

```
prompt → CognitiveManager.handle()  [Cognitive birimi]
   → (feature/memory/policy/deliberation) → backend.generate()
   → CevahirModelAPI.generate() → ModelManager → NN → decode
   → critic/memory update → CognitiveOutput
```

### 5.3 Terminal sohbet (`chat_pipeline.py`)

`ChatPipeline` bağımsız bir terminal döngüsüdür (`main()` → `run()`); checkpoint'ten
model yükler, `tie_weights` varsayılanını set eder, `generate` ile yanıt üretir.
Cevahir + ChattingManager hattını sarar.

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Yeni üretim stratejisi (ör. nucleus) | `CevahirModelAPI._autoregressive_generate` | Beam ile aynı imza |
| Yeni optimizer/scheduler | `ModelInitializer.initialize_optimizer/scheduler` | Config şemasına ekle |
| Checkpoint formatı | `ModelSaver`/`ModelLoader` | `save_*`/`load_*` simetrisi |
| Yeni config alanı | `config_schema.py` (`_SchemaBase` + `validate`) | tie_weights kuralını koru |
| Sağlık metriği | `health_monitor.py` (`*Health` + `HealthReport`) | severity sözleşmesi |
| Composition değişikliği | `Cevahir.__init__` | Tüm sistemin tek montaj noktası |

---

## 7. Bağımlılıklar

**Bağımlı olduğu:** Tokenizer (`TokenizerCore`), Neural Network (`CevahirNeuralNetwork`),
Cognitive (`CognitiveManager`, `CognitiveModelAPI`, `CognitiveManagerConfig`), `torch`.

**Buna bağımlı olanlar:** API/Chatting (dolaylı), Training System (`ModelManager`'ı
eğitim için kullanır).

> **Örtük kontratlar:**
> - `vocab_size` Tokenizer → Model (`_init_model`).
> - `tie_weights=True` → `seq_proj_dim == embed_dim` (`config_schema.validate`).
> - Auto-load yolu: `saved_models/cevahir_model.pth`.

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Konum | Risk | Not |
|--------|-------|------|-----|
| **Dev facade dosyası** | `model/cevahir.py` 2114 LOC, 3 sınıf bir arada | Yüksek | `CevahirConfig`, `CevahirModelAPI`, `Cevahir` ayrı dosyalara bölünebilir |
| **`ModelManager` çift-rol** | Hem eğitim hem çıkarım bileşenlerini yönetir | Orta | Eğitim/çıkarım sorumlulukları ayrılabilir (ISP) |
| **Örtük default doldurma** | `ModelInitializer.build_model` eksik NN parametrelerini default'lar | Orta | Şema doğrulaması tek noktaya çekilebilir |
| **Metot-içi import** | `cevahir.py` içinde `import torch/os/...` dağınık | Düşük | — |
| **Auto-load sabit yolu** | `saved_models/cevahir_model.pth` hardcoded | Düşük | Config'e taşınabilir |
| **`ChatPipeline` ayrık hat** | Cevahir'i sarıyor ama bağımsız `main` | Düşük | Facade ile örtüşme netleştirilmeli |
| **Multimodal iskeleti** | `process_audio/image/multimodal` stub görünümlü | Düşük | Uygulanma durumu teyit edilmeli |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| Composition root | `model/cevahir.py:1093` (`Cevahir.__init__`) |
| Model init + auto-load | `model/cevahir.py:1229` (`_init_model`) |
| Adapter köprüsü | `model/cevahir.py:468` (`CevahirModelAPI`) |
| Autoregressive üretim | `model/cevahir.py:555` (`_autoregressive_generate`) |
| Beam search | `model/cevahir.py:754` (`_generate_with_beam_search`) |
| Bilişsel süreç | `model/cevahir.py:1708` (`process`) |
| Config doğrulama | `model/cevahir.py:432` (`CevahirConfig.validate`) |
| Model inşası | `model_management/model_manager.py:206` (`build_model`) |
| Model init detayı | `model_management/model_initializer.py:167` |
| Checkpoint kaydet | `model_management/model_saver.py:245` (`save_checkpoint`) |
| Checkpoint yükle | `model_management/model_loader.py:238` (`load_model`) |
| tie_weights kuralı | `model_management/config_schema.py:205` |
| Terminal sohbet | `model_management/chat_pipeline.py:213` (`main`) |

---

*Kaynak: `model/`, `model_management/` — analiz kodun mevcut halinden çıkarılmıştır.*
