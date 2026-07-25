# Code Reality — Training System

> Kanonik Birim #5 · Kaynak: `training_system/` (v2 + v3 + kök)
> Eğitim **servis/koşu** katmanının (giriş noktası, veri/cache, kurulum) kodundan
> çıkarılmış mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L2), [search index](../../architecture-search-index.md).
> Komşu birim: [training-management](../training-management/) (bu servisin çağırdığı motor).

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Training System (eğitim servisi + koşu) |
| **Kaynak dizin** | `training_system/` (kök betikler + `v2/` + `v3/`) |
| **Toplam boyut** | ~8.320 LOC (testler hariç) |
| **Giriş noktası** | `training_system/train.py:1` (koşu betiği) |
| **Servis sınıfları** | `TrainingService` (v2: `v2/core/training_service.py:90`), `TrainingServiceV3` (`v3/core/training_service_v3.py:70`) |
| **Çalışma zamanı** | Batch (eğitim) |
| **Dış bağımlılık** | `torch`; Training Mgmt, Model Mgmt, Tokenizer, Data Loader |

---

## 2. Sorumluluk

Bir eğitim koşusunu **baştan sona kurmak ve çalıştırmak**: seed/dizin/ortam
hazırlığı, tokenizer config yükleme, cache'ten veri okuma, veri hizalama/doğrulama,
train/val ayrımı, tensöre çevirme, model başlatma, ardından motoru
(`training_management.TrainingManager`) çağırıp koşuyu yönetme; epoch sonrası
inline test ve koşu özeti.

**Kapsam dışı:** Eğitim döngüsünün iç mekaniği (Training Management), model tanımı
(Neural Network), ham veri toplama (Data Processing).

---

## 3. Dosya Envanteri

### 3.1 Kök (koşu + cache + doğrulama)

| Dosya | LOC | Rol / anahtar üyeler |
|-------|-----|----------------------|
| `train.py` | 807 | **Giriş noktası**: `set_seed`, `ensure_dirs`, `log_env_info`, `load_tokenizer_config`; v2 `TrainingService`'i import eder, v3'ü opsiyonel dener |
| `data_cache.py` | 680 | Eğitim verisi cache üretimi/okuması |
| `prepare_cache.py` | 651 | Cache hazırlama boru hattı |
| `config_validator.py` | 639 | Eğitim config doğrulama |
| `health_check.py` | 763 | Koşu öncesi/sırası sistem sağlık kontrolü |
| `clear_cache.py`, `debug_all_cache.py`, `test_cache_overlap.py`, `test_overlap.py` | — | Cache bakım/hata ayıklama/testleri |

### 3.2 v3 (güncel) — `training_system/v3/`

| Alt sistem | Dosya(lar) | Rol |
|------------|-----------|-----|
| **core** | `training_service_v3.py` (724), `config_manager_v3.py` (291) | v3 servis + config |
| **data** | `cache_v3.py` (533), `dataloader_v3.py` (282), `dataset_v3.py` (138), `sampler_v3.py` (148), `collator_v3.py` (107) | v3 veri boru hattı (kaynak-id farkında) |

### 3.3 v2 (önceki) — `training_system/v2/`

| Alt sistem | Dosya(lar) | Rol |
|------------|-----------|-----|
| **core** | `training_service.py` (918), `config_manager.py`, `criterion_manager.py`, `data_preparator.py`, `bpe_validator.py` | v2 servis + yardımcılar |
| **utils** | `warmup_calculator.py`, `data_loader_wrapper.py` | Warmup, loader sarma |
| **docs** | `ARCHITECTURE.md`, `README.md`, `*_STATUS.md`, `FIXES_APPLIED.md` | Eski repo-içi dokümanlar |

---

## 4. İç Mimari

```
   python training_system/train.py            (giriş noktası)
        │  set_seed / ensure_dirs / log_env_info / load_tokenizer_config
        │  config seçimi → v2 TrainingService (default) | v3 TrainingServiceV3 (varsa)
        ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  TrainingService(V3)   (training_system/*/core)                    │
   │                                                                    │
   │   _setup_device()                                                  │
   │   load_data_from_cache() ──► cache_v3 / data_cache                 │
   │        · _validate_alignment / _source_id_aware_split / _to_tensors│
   │   _initialize_model() ──► ModelManager (build + optional load)     │
   │        │                    [Model/Engine birimi]                  │
   │   train():                                                         │
   │        └─ TrainingManager(...)   [Training Management birimi]      │
   │             · epoch_callback ile geri bildirim                     │
   │        └─ _test_model_after_epoch / _test_model_inline (inline QA) │
   └──────────────────────────────────────────────────────────────────┘
        │
        ▼
   koşu özeti (best loss, checkpoint yolu) → disk/log
```

---

## 5. Veri / Kontrol Akışı

```
① Cache hazırlığı (ayrı adım)
   prepare_cache.py / data_cache.py
      · Data Loader'dan ham veri → tokenize → cache dosyaları

② Koşu
   train.py
      → TrainingService.load_data_from_cache()
          · cache oku → (kaynak-id farkında) train/val split → tensörler
      → TrainingService._initialize_model()  (ModelManager)
      → TrainingService.train()
          → TrainingManager.train()  [motor]
              · her epoch sonrası epoch_callback → inline test
      → özet yaz
```

**v3 veri farkı:** v3 `data/` boru hattı **kaynak-id farkındadır**
(`_source_id_aware_split`, `_split_by_source_id`) — aynı kaynaktan örneklerin
train/val'e sızmasını önler (veri sızıntısı koruması). v2'de bu daha basittir.

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Yeni veri kaynağı/format | `v3/data/dataset_v3.py` + `collator_v3.py` | tensör sözleşmesi `_to_tensors` |
| Cache stratejisi | `data_cache.py` / `v3/data/cache_v3.py` | `prepare_cache.py` ile senkron |
| Split politikası | `_source_id_aware_split` / `_simple_random_split` | sızıntı koruması |
| Config alanı | `config_validator.py` + `v3/core/config_manager_v3.py` | doğrulama |
| Koşu öncesi kontrol | `health_check.py` | — |
| Epoch-sonu QA | `_test_model_after_epoch` / `_test_model_inline` | üretim örneklemesi |

---

## 7. Bağımlılıklar

**Bağımlı olduğu:** Training Management (`TrainingManager` + v2 util'ler),
Model Mgmt (`ModelManager`), Tokenizer (config + vocab), Data Loader (ham veri), `torch`.
**Buna bağımlı olanlar:** Yok (en üst eğitim giriş noktası; `Cevahir` facade'ı
kullanmaz).

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Kanıt | Risk | Not |
|--------|-------|------|-----|
| **v2/v3 servis ikizliği** | `TrainingService` (918) ve `TrainingServiceV3` (724) paralel | **Yüksek** | Kanonik servis kararı |
| **v3'ün v2'ye bağımlılığı** | `training_service_v3.py:500-508` v2 checkpoint/tensorboard/logger'ı import eder; v3 manager yoksa v2'ye düşer | **Yüksek** | v3 tam bağımsız değil |
| **Karışık import stratejisi** | `train.py` v2'yi kesin, v3'ü `try/except` ile import eder | Orta | Sürüm seçimi config'e taşınabilir |
| **Kökte dağınık betikler** | `debug_all_cache.py`, `test_*_overlap.py`, `clear_cache.py` kökte | Düşük | `tests/`/`scripts/` altına |
| **Eski doc dosyaları** | `v2/ARCHITECTURE.md`, `*_STATUS.md` | Orta | Bu code-reality güncel referans; eskiler arşivlenmeli |
| **Cache formatı örtük** | `data_cache` ↔ `cache_v3` iki format | Orta | Format sürümlemesi netleştirilmeli |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| Koşu giriş noktası | `training_system/train.py:1` |
| v2 servis | `training_system/v2/core/training_service.py:90` (`TrainingService`) |
| v2 train | `training_system/v2/core/training_service.py:264` |
| v3 servis | `training_system/v3/core/training_service_v3.py:70` (`TrainingServiceV3`) |
| v3 cache'ten veri | `training_system/v3/core/training_service_v3.py:212` (`load_data_from_cache`) |
| Kaynak-id farkında split | `training_system/v3/core/training_service_v3.py:288` |
| Motora bağlanma | `training_system/v3/core/training_service_v3.py:489` (`_run_training`) |
| Cache hazırlama | `training_system/prepare_cache.py`, `data_cache.py` |
| Config doğrulama | `training_system/config_validator.py` |
| Sağlık kontrolü | `training_system/health_check.py` |

---

*Kaynak: `training_system/` — analiz kodun mevcut halinden çıkarılmıştır.*
