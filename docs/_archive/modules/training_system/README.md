# 🎯 Training System - Kapsamlı Dokümantasyon

**Versiyon:** V-3 (Current)
**Son Güncelleme:** 2026-03-16
**Durum:** ✅ Production-Ready | V2 + V3 Dual Stack

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Dizin Yapısı](#dizin-yapısı)
3. [Zorunlu İş Akışı](#zorunlu-iş-akışı)
4. [Kök Araçlar](#kök-araçlar)
   - [train.py](#trainpy)
   - [prepare_cache.py](#prepare_cachepy)
   - [config_validator.py](#config_validatorpy)
   - [health_check.py](#health_checkpy)
   - [data_cache.py](#data_cachepy)
5. [V3 Sistemi](#v3-sistemi)
   - [TrainingServiceV3](#trainingservicev3)
   - [ConfigManagerV3](#configmanagerv3)
   - [DataCacheV3](#datacachev3)
   - [CevahirDataset](#cevahirdataset)
   - [BucketBatchSampler](#bucketbatchsampler)
   - [DynamicPaddingCollator](#dynamicpaddingcollator)
   - [DataLoader V3](#dataloader-v3)
6. [V2 Sistemi](#v2-sistemi)
   - [TrainingService (V2)](#trainingservice-v2)
   - [ConfigManager (V2)](#configmanager-v2)
   - [CriterionManager](#criterionmanager)
   - [BPEValidator](#bpevalidator)
   - [DataPreparator (Deprecated)](#datapreparator-deprecated)
   - [DataLoaderWrapper (V2)](#dataloaderwrapper-v2)
   - [WarmupCalculator](#warmupcalculator)
7. [V2 → V3 Otomatik Seçim](#v2--v3-otomatik-seçim)
8. [Konfigürasyon Parametreleri](#konfigürasyon-parametreleri)
9. [GPU Optimizasyonları](#gpu-optimizasyonları)
10. [Cache Sistemi Karşılaştırması](#cache-sistemi-karşılaştırması)

---

## Genel Bakış

Training System, Cevahir-AI modelinin eğitimini yöneten end-to-end sistemdir. İki paralel stack içerir:

- **V2 Stack:** Kararlı, production-ready temel eğitim sistemi
- **V3 Stack:** GPU-optimize edilmiş gelişmiş eğitim sistemi — Strict Cache, BucketBatchSampler, DynamicPadding

```
train_bpe.py → prepare_cache.py → train.py
     ↓               ↓               ↓
  BPE vocab      V3 Cache        V2 veya V3
  oluştur        hazırla         otomatik seçim
```

> ⚠️ **Bu sıra zorunludur.** `prepare_cache.py` çalıştırılmadan `train.py` başlatılmamalıdır.

---

## Dizin Yapısı

```
training_system/
├── train.py                    # Ana giriş noktası (80+ parametre TRAIN_CONFIG)
├── train_bpe.py                # BPE tokenizer eğitimi
├── prepare_cache.py            # Veri ön-işleme ve cache hazırlama
├── config_validator.py         # 5 aşamalı config validasyonu
├── health_check.py             # Eğitim sonrası model kalite kontrolü
├── data_cache.py               # V2 DataCache (graceful fallback)
│
├── v2/
│   ├── core/
│   │   ├── training_service.py     # V2 TrainingService (936 satır)
│   │   ├── config_manager.py       # V2 ConfigManager (21 parametre)
│   │   ├── criterion_manager.py    # CriterionManager + EntropyRegCriterion
│   │   ├── bpe_validator.py        # BPE dosya varlık kontrolü
│   │   └── data_preparator.py      # DEPRECATED stub
│   ├── data/
│   │   └── data_loader_wrapper.py  # SimpleDataset + create_dataloaders()
│   └── utils/
│       └── warmup_calculator.py    # Dinamik warmup adımı hesaplama
│
└── v3/
    ├── core/
    │   ├── training_service_v3.py  # V3 TrainingService (725 satır)
    │   └── config_manager_v3.py    # V3 ConfigManager (55+ parametre, 11 grup)
    └── data/
        ├── cache_v3.py             # DataCacheV3 (strict mode, SHA-256)
        ├── dataset_v3.py           # CevahirDataset (uzunluk indeksi)
        ├── sampler_v3.py           # BucketBatchSampler
        ├── collator_v3.py          # DynamicPaddingCollator
        └── dataloader_v3.py        # create_dataloaders_v3() factory
```

---

## Zorunlu İş Akışı

### Adım 1 — BPE Eğitimi
```bash
python training_system/train_bpe.py
```
Çıktı: `bpe_vocab.json`, `bpe_merges.txt`

### Adım 2 — Cache Hazırlama
```bash
python training_system/prepare_cache.py \
  --data-dir ./data/raw \
  --cache-dir ./training_system/cache \
  --max-seq-length 512
```
Çıktı: `cache/*.pkl` + `cache/*.sha256` + `cache/*.meta.json` (V3 üçlü yapısı)

### Adım 3 — Eğitim
```bash
python training_system/train.py
```
`TRAIN_CONFIG` içindeki `use_v3_training_system: True/False` parametresine göre V3 veya V2 otomatik seçilir.

### Adım 4 — Sağlık Kontrolü (Opsiyonel)
```bash
python training_system/health_check.py \
  --model-path saved_models/cevahir_model.pth
```

---

## Kök Araçlar

### train.py

Ana giriş noktası. `TRAIN_CONFIG` dict içinde 80+ parametre barındırır.

**Önemli işlevler:**

| Fonksiyon | Açıklama |
|---|---|
| `normalize_config()` | `gradient_clip → max_grad_norm`, BPE yolları, scheduler_kwargs, TensorBoard defaults |
| `log_env_info()` | PyTorch sürümü, CUDA kullanılabilirliği, GPU adı |
| `ensure_dirs()` | `saved_models/`, `logs/`, `cache/` dizinlerini oluşturur |
| `main()` | OOM fix → seed → normalize_config → V3/V2 seçim → eğitim |

**OOM Fix (Kritik):**
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```
GPU bellek parçalanmasını önler; eğitim başlamadan önce set edilir.

**V5 Mimari (TRAIN_CONFIG):**
```python
"num_kv_heads": 2,                # GQA — %75 KV cache azalması
"rope_scaling_type": "yarn",      # YaRN — uzun context genişletme
"rope_scaling_factor": 2.0,
"sliding_window": 512,            # Sliding Window Attention
```

---

### prepare_cache.py

Eğitim verisini autoregressive formata dönüştürür ve V3 cache yapısına kaydeder.

**4 Adımlı Süreç:**

```
1. Cache temizle (--no-clear-cache ile atlanabilir)
2. TokenizerCore yükle (BPE vocab + merges)
3. Veri encode et → format_data_func()
4. DataCacheV3.save() → pkl + sha256 + meta.json
```

**`format_data_func()` — Autoregressive Format:**
```
inp: [BOS] + encoded_input   (truncate sağdan, BOS korunur)
tgt: encoded_target + [EOS]  (truncate sağdan, EOS korunur)
```

**Deduplication:** `hash(inp_tuple) + source_id` ile duplicate tespiti; kaynak ayrımı korunur.

**Overlap Analizi:** Pre-format ve post-format hash collision oranı loglanır (veri sızıntısı tespiti).

**CLI Parametreleri:**

| Parametre | Default | Açıklama |
|---|---|---|
| `--data-dir` | — | Ham veri dizini |
| `--cache-dir` | `./cache` | Cache çıktı dizini |
| `--max-seq-length` | 512 | Maksimum token uzunluğu |
| `--include-whole-words` | False | Tam kelime encoding |
| `--include-syllables` | False | Hece bazlı encoding |
| `--include-sep` | False | SEP token ekle |
| `--no-clear-cache` | False | Mevcut cache'i temizleme |

---

### config_validator.py

`TRAIN_CONFIG` dict'ini eğitimden önce doğrular. 5 aşamalı validasyon sistemi.

**`ValidationResult` Dataclass:**
```python
@dataclass
class ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

    def print_report() -> None  # Renkli terminal çıktısı
```

**5 Validasyon Aşaması:**

| Aşama | Kontrol |
|---|---|
| 1. Zorunlu Alan | 11 zorunlu alan varlığı (`vocab_size`, `embed_dim`, `num_heads`, `num_layers`, `batch_size`, `epochs`, `lr`, `device`, `data_dir`, `bpe_vocab_path`, `bpe_merges_path`) |
| 2. Tür | 50+ alan tür uyumu (`int`, `float`, `bool`, `str`, `list`, `Optional`) |
| 3. Aralık | 30+ alan değer aralığı (`lr ∈ [1e-7, 1.0]`, `dropout ∈ [0,1]`, `label_smoothing ∈ [0,0.5]`) |
| 4. Tutarlılık | 8 çapraz-alan kuralı (aşağıya bakın) |
| 5. Best-Practice | 6 öneri (label_smoothing=0.0 uyarısı, vb.) |

**Tutarlılık Kuralları (Aşama 4):**
- `swa_start_epoch < epochs`
- LLRD + AdamW uyum kontrolü
- `tie_weights=True` → `embed_dim == seq_proj_dim`
- `grad_accum_steps > batch_size` → WARNING (olağandışı yapılandırma)
- `use_amp=True + device="cpu"` → WARNING
- `moe_top_k < num_experts`
- `embed_dim % num_heads == 0`
- SAM + Lookahead birlikte kullanım uyarısı

**Kullanım:**
```python
from training_system.config_validator import ConfigValidator

result = ConfigValidator.validate(config)
result.print_report()

# Hata varsa ValueError fırlatır, uyarıları loglar
ConfigValidator.validate_and_raise(config)
```

---

### health_check.py

Eğitim sonrası model kalite kontrolü. 8 sabit Türkçe/İngilizce prompt ile inference testi.

**Ölçülen Metrikler:**

| Metrik | Production Kriteri | Açıklama |
|---|---|---|
| `entropy` | > 2.0 | Shannon entropy — düşük = tekrarcı çıktı |
| `eos_ratio` | < 0.3 | Erken EOS oranı — yüksek = yetersiz üretim |
| `avg_len` | > 5 | Ortalama yanıt uzunluğu (token) |
| `ttr` | > 0.3 | Type-Token Ratio — kelime çeşitliliği |

**Çıktı:** JSON raporu + terminal özeti

**CLI:**
```bash
python health_check.py \
  --model-path saved_models/cevahir_model.pth \
  --config-path training_system/train.py \
  --verbose
```

---

### data_cache.py

V2 DataCache — graceful fallback ile cache yönetimi.

| Metod | Açıklama |
|---|---|
| `get_cached_data()` | Cache'den yükle, bulunamazsa `None` döndür (V3'ten farklı!) |
| `save_cached_data()` | Atomic write ile cache'e kaydet |
| `get_or_process()` | Cache varsa yükle, yoksa işle ve kaydet |
| `get_or_process_corpus()` | BPE eğitim corpus cache'i |

**Cache Key:** `MD5(data_dir + encode_mode + vocab_path + merges_path + max_seq + ...)`

**V3'ten Farkı:**
- V2: `get_cached_data()` → `None` döndürür (fallback desteği)
- V3: `load_strict()` → `CacheNotFoundError` fırlatır (strict mode)

---

## V3 Sistemi

### TrainingServiceV3

**Dosya:** `v3/core/training_service_v3.py` (725 satır)

V3 eğitim orkestratörü. Strict Cache Mode, Source-ID aware split ve gelişmiş GPU optimizasyonu sunar.

**`__init__()` — 10 Adım:**

```
1.  BPE yollarını al (config'ten veya varsayılan)
2.  Device kurulumu (CUDA/CPU)
3.  data_dir doğrulama
4.  BPEValidator.validate_files()
5.  TokenizerCore yükle
6.  vocab_size al
7.  DataCacheV3 oluştur (strict_mode=True)
8.  ModelManager oluştur
9.  CriterionManager oluştur
10. ConfigManagerV3 oluştur
```

**`train()` — 5 Adım:**

```
1. Model initialize (_initialize_model)
2. Cache'den veri yükle (load_data_from_cache — STRICT)
3. Source-ID aware split (_source_id_aware_split)
4. V3 DataLoader oluştur (create_dataloaders_v3)
5. V3 Config hazırla + TrainingManager başlat
```

**`_source_id_aware_split()` — Veri Sızıntısı Önleme:**

Aynı `source_id`'ye ait kayıtların train ve val'e bölünmesini önler:
```
source_id=1 → tüm örnekleri train'e
source_id=2 → tüm örnekleri val'e
```
Fallback: source_id yoksa basit random split + WARNING.

**`_initialize_model()` — Checkpoint Fallback Zinciri:**
```
last.pth → best.pth → checkpoint_*.pth (en yeni) → sıfırdan
```

**`_test_model_inline()` — Epoch Sonu Testi:**
- `top_k=80` ile generation
- Min 5 token EOS koruma
- Sadece loglama amaçlı (eğitimi etkilemez)

---

### ConfigManagerV3

**Dosya:** `v3/core/config_manager_v3.py`

55+ parametre, 11 grup halinde V3 TrainingManager'a iletir.

**11 Parametre Grubu:**

| Grup | Temel Parametreler |
|---|---|
| **Temel** | `vocab_size`, `epochs`, `batch_size`, `device`, `seed` |
| **Optimizer** | `lr`, `weight_decay`, `optimizer_type`, `use_adagrad`, `use_adamw8bit` |
| **Scheduler** | `scheduler_type`, `scheduler_kwargs`, `warmup_steps` |
| **Regularizasyon** | `dropout`, `label_smoothing`, `entropy_coeff`, `focal_loss_gamma` |
| **Gradient** | `max_grad_norm`, `grad_accum_steps`, `use_amp`, `agc_clip_val` |
| **Optimizasyon** | `use_sam`, `sam_rho`, `use_lookahead`, `lookahead_k`, `lookahead_alpha` |
| **EMA/SWA** | `use_ema`, `ema_decay`, `use_swa`, `swa_start_epoch`, `swa_lr` |
| **LLRD** | `use_llrd`, `llrd_decay` |
| **Curriculum** | `use_curriculum`, `curriculum_strategy`, `scheduled_sampling_start` |
| **Güvenlik** | `nan_recovery_enabled`, `spike_detection_enabled`, `spike_threshold` |
| **Token** | `pad_id`, `bos_id`, `eos_id`, `unk_id` |

**Dahili Validasyon:**
```python
assert 0.0 <= label_smoothing <= 0.5
assert 0.0 <= entropy_coeff <= 1.0
assert 0.0 < ema_decay < 1.0
assert batch_size > 0
assert epochs > 0
```

---

### DataCacheV3

**Dosya:** `v3/data/cache_v3.py`

**Strict Cache Mode** — Cache yoksa eğitim durur, graceful fallback yok.

**Özel Exception'lar:**
```python
class CacheNotFoundError(Exception): ...  # Cache bulunamadı
class CacheIntegrityError(Exception): ... # SHA-256 uyuşmazlığı
```

**Üçlü Dosya Yapısı:**
```
cache/
├── {cache_key}.pkl       # Asıl veri (pickle)
├── {cache_key}.sha256    # SHA-256 checksum
└── {cache_key}.meta.json # Metadata (tarih, parametre, boyut)
```

**Cache Key Bileşenleri (MD5, 9 bileşen):**
```
data_dir + encode_mode + vocab_hash + max_seq + include_whole_words
+ include_syllables + include_sep + bpe_vocab_path + bpe_merges_path
```

**`load_strict()` Akışı:**
```
1. cache_key eşleşmesi ara
2. Eşleşme yoksa → CacheNotFoundError (detaylı mesaj: mevcut cache'ler listelenir)
3. Eşleşme varsa → SHA-256 checksum doğrula
4. Checksum hatalı → CacheIntegrityError
5. Başarılı → veri döndür
```

**`save()` — Atomic Write:**
```
1. tmp dosyaya yaz
2. SHA-256 hesapla
3. tmp → asıl dosya rename (atomic)
4. .sha256 ve .meta.json yaz
```

**`load_for_training()` — Üst Seviye API:**
```python
cache.load_for_training(
    tokenizer=tokenizer,
    data_dir=config["data_dir"],
    config=config
)
# vocab_hash + data_hash hesapla → cache_key → load_strict()
```

---

### CevahirDataset

**Dosya:** `v3/data/dataset_v3.py`

`torch.utils.data.Dataset` alt sınıfı. BucketBatchSampler entegrasyonu için uzunluk indeksi tutar.

**Özellikler:**
- **Uzunluk indeksi:** Her sequence için PAD hariç gerçek uzunluk (BucketBatchSampler için)
- **Lazy tensor:** Veri liste ise `__getitem__`'da tensor'a çevir
- **Source-ID drop:** 3-tuple `(inp, tgt, source_id)` → `(inp, tgt)`
- **`get_length_stats()`:** `min / max / mean / median / p25 / p75 / p90 / p99`

**Uzunluk Hesaplama (`_compute_lengths`):**
```python
# PAD token'larını geriye doğru say; son non-PAD pozisyonu bul
for i in range(len(inp_list) - 1, -1, -1):
    if inp_list[i] != pad_id:
        real_len = i + 1; break
```

---

### BucketBatchSampler

**Dosya:** `v3/data/sampler_v3.py`

Sequence uzunluklarına göre gruplama yaparak padding waste'i minimize eder.

**Referans:** Schwartz et al. 2020, "Right Tool for the Job"

**Algoritma:**
```
1. Sequence'ları uzunluğa göre sırala (sorted_indices)
2. num_buckets adet eşit büyüklükte bucket'a böl
3. Her epoch: bucket içi shuffle → batch oluştur → batch'leri shuffle
4. Epoch bazlı seed: rng = Random(seed + epoch)
```

**Padding Waste Karşılaştırması:**
```
Statik Padding:  GPU'nun %70-90'ı PAD token hesaplar
Bucket Batching: Padding waste %20-40'a düşer
```

**Parametreler:**

| Parametre | Default | Açıklama |
|---|---|---|
| `lengths` | — | Her örnek için gerçek uzunluk listesi |
| `batch_size` | — | Batch başına örnek sayısı |
| `num_buckets` | 32 | Bucket sayısı (↑ = daha az padding, ↓ randomness) |
| `shuffle_buckets` | True | Epoch başında batch shuffle |
| `shuffle_within_bucket` | True | Bucket içi shuffle |
| `drop_last` | False | Son incomplete batch'i at |
| `seed` | 42 | Tekrarlanabilirlik için temel seed |

**Epoch Güncellemesi:**
```python
sampler.set_epoch(epoch)  # Her epoch başında çağrılmalı
```

---

### DynamicPaddingCollator

**Dosya:** `v3/data/collator_v3.py`

Her batch'i kendi içindeki maksimum sequence uzunluğuna pad eder.

**Referans:** Ott et al. 2019, "Scaling Neural Machine Translation" (fairseq)

**V2 Karşılaştırması:**
```
V2 (custom_collate): torch.stack() → global max_seq_length'e pad
V3 (DynamicPaddingCollator): batch içi max uzunluğa pad
```

**GPU Etkisi (Örnek):**
```
Batch {len=10, len=12, len=15}:
  V2: 512 token'a pad → 97% PAD hesabı
  V3: 15 token'a pad  → sıfır gereksiz PAD hesabı
```

**`__call__()` Akışı:**
```python
max_len = max(item[0].size(-1) for item in batch)
if max_seq_length: max_len = min(max_len, max_seq_length)

inputs  = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
targets = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

for i, (inp, tgt) in enumerate(batch):
    inp_len = min(inp.size(-1), max_len)
    inputs[i, :inp_len] = inp[:inp_len]
    # ... targets aynı
```

**Geriye Uyumluluk:**
```python
# V2 static collate (pre-padded veriler için):
collate_fn = create_static_collate(pad_id=0)
```

---

### DataLoader V3

**Dosya:** `v3/data/dataloader_v3.py`

**6 GPU Optimizasyon Katmanı:**

| # | Optimizasyon | Etki |
|---|---|---|
| 1 | `pin_memory=True` | Sabitlenmiş RAM → PCIe DMA hızlı transfer |
| 2 | `non_blocking=True` | Async GPU transfer (hesaplama ile örtüşür) |
| 3 | `prefetch_factor=2` | N batch önceden hazırla |
| 4 | `persistent_workers=True` | Worker process'leri epoch'lar arası canlı |
| 5 | `BucketBatchSampler` | Padding waste minimize |
| 6 | `DynamicPaddingCollator` | Batch başına dinamik pad uzunluğu |

**`create_dataloaders_v3()` — Train / Val Farkı:**

| | Train | Val |
|---|---|---|
| Sampler | `BucketBatchSampler` (shuffle) | `SequentialSampler` |
| Bucket | `use_bucket_batching=True` | `False` |
| Shuffle | `True` | `False` |

**Windows Uyarısı:**
```python
if num_workers > 0 and os.name == "nt":
    logger.warning("Windows'ta num_workers>0 sorunlu olabilir...")
```

---

## V2 Sistemi

### TrainingService (V2)

**Dosya:** `v2/core/training_service.py` (936 satır)

**`__init__()` Adımları:**
- BPEValidator → TokenizerCore → vocab_size → DataCache (optional) → CriterionManager → ConfigManager → DataPreparator

**`train()` Ana Akış:**
```
1. Model initialize (_initialize_model)
2. Cache'den veri yükle (graceful fallback: None döndürür)
3. Random split (source_id dikkate almaz)
4. V2 DataLoader (create_dataloaders)
5. V2 Config hazırla + TrainingManager
```

**`prepare_from_cache()` — Next-Token Alignment Doğrulama:**
```python
# inp[i+1] == tgt[i] kontrolü — autoregressive format doğrulaması
_validate_alignment(data)
```

**`_test_model_after_epoch()` — Epoch Sonu Testi:**
- Cevahir.generate() kullanmaz (standalone test)
- Model doğrudan çağrılır

---

### ConfigManager (V2)

**Dosya:** `v2/core/config_manager.py` (109 satır)

`TRAIN_CONFIG` → V2 TrainingManager formatına dönüştürür.

**`prepare_training_config()` — 21 Parametre:**

Özel token ID çıkarımı:
```python
pad_id = tokenizer.special_tokens["<PAD>"]
bos_id = tokenizer.special_tokens["<BOS>"]
eos_id = tokenizer.special_tokens["<EOS>"]
unk_id = tokenizer.special_tokens["<UNK>"]
```

**V3 ConfigManagerV3 ile Farkı:**

| | V2 ConfigManager | V3 ConfigManagerV3 |
|---|---|---|
| Parametre sayısı | ~21 | 55+ |
| Gruplar | Yok (düz dict) | 11 grup |
| Entropy coeff | ✗ | ✓ |
| SAM/Lookahead | ✗ | ✓ |
| EMA/SWA | ✗ | ✓ |
| LLRD | ✗ | ✓ |
| Curriculum | ✗ | ✓ |
| NaN Recovery | ✗ | ✓ |

---

### CriterionManager

**Dosya:** `v2/core/criterion_manager.py`

Loss fonksiyonu oluşturma ve yapılandırma.

**`EntropyRegCriterion(nn.Module)` — Entropy Regularization:**

> Referans: Pereyra et al. 2017, "Regularizing Neural Networks by Penalizing Confident Output Distributions"

```python
loss = CE_loss + entropy_coeff * (-mean_entropy)
# Yüksek confidence → negatif entropy → penalize
# Model daha "düşük emin" çıktılar üretmeye teşvik edilir
```

**Memory-Safe Chunk Hesaplama:**
```python
_CHUNK = 512  # Her forward'da max 512 örnek işle
# Büyük vocab_size'da OOM riskini önler
```

**`LossComputation` Uyumluluk Property'leri:**
```python
@property
def weight(self) -> Optional[Tensor]: ...
@property
def label_smoothing(self) -> float: ...
@property
def ignore_index(self) -> int: ...
```

**`CriterionManager.create_criterion()` Akışı:**
```
1. Vocab ağırlık tensor oluştur (EOS özel ağırlığı)
2. CrossEntropyLoss(label_smoothing, ignore_index=pad_id)
3. entropy_coeff > 0 ise → EntropyRegCriterion ile sarmala
4. Döndür
```

---

### BPEValidator

**Dosya:** `v2/core/bpe_validator.py` (99 satır)

BPE vocab ve merges dosyalarının varlığını ve içeriğini doğrular.

**Strateji:** Fixed vocab (vocab sadece `train_bpe.py` ile oluşturulur, otomatik oluşturma yok)

```python
validator = BPEValidator(
    vocab_path="bpe_vocab.json",
    merges_path="bpe_merges.txt"
)
validator.validate_files()
# Eksik veya boşsa → RuntimeError (train_bpe.py çalıştır mesajı)
```

---

### DataPreparator (Deprecated)

**Dosya:** `v2/core/data_preparator.py` (62 satır)

> ⛔ **DEPRECATED** — Artık kullanılmıyor.

```python
# Örnekleme sırasında uyarı:
DeprecationWarning: "DataPreparator kullanımdan kaldırıldı.
TrainingService.prepare_from_cache() kullanın."
```

İşlevler `training_service.py` ve `prepare_cache.py`'ye taşındı.

---

### DataLoaderWrapper (V2)

**Dosya:** `v2/data/data_loader_wrapper.py` (130 satır)

**`SimpleDataset`:**
```python
class SimpleDataset(Dataset):
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.data[idx]  # (inp, tgt) tuple
```

**`custom_collate()`:**
```python
inputs  = torch.stack([item[0] for item in batch])
targets = torch.stack([item[1] for item in batch])
# Tüm sequence'lar aynı uzunlukta varsayılır (pre-padded)
```

**`create_dataloaders()`:**
```python
DataLoader(
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=(prefetch_factor if num_workers > 0 else None),
)
```

**V3 ile Farkı:**

| | V2 DataLoader | V3 DataLoader |
|---|---|---|
| Sampler | RandomSampler / SequentialSampler | BucketBatchSampler |
| Collate | `torch.stack()` (statik) | DynamicPaddingCollator (dinamik) |
| Padding | Global max_seq_length | Batch içi max uzunluk |
| Padding Waste | %70-90 | %20-40 |

---

### WarmupCalculator

**Dosya:** `v2/utils/warmup_calculator.py` (68 satır)

```python
def calculate_warmup_steps(
    batches_per_epoch: int,
    grad_accum_steps: int = 1,
    warmup_epochs: float = 1.0,
) -> int:
    steps = (batches_per_epoch // grad_accum_steps) * warmup_epochs
    return max(1, int(steps))  # Minimum 1 warmup step
```

V2 TrainingService içinde `train()` akışında çağrılır.

---

## V2 → V3 Otomatik Seçim

`train.py → main()` içinde:

```python
if config.get("use_v3_training_system", False):
    from training_system.v3.core.training_service_v3 import TrainingServiceV3
    service = TrainingServiceV3(config)
else:
    from training_system.v2.core.training_service import TrainingService
    service = TrainingService(config)
```

**V3 için TRAIN_CONFIG:**
```python
"use_v3_training_system": True,
"use_bucket_batching": True,
"num_buckets": 32,
"use_dynamic_padding": True,
"prefetch_factor": 2,
"persistent_workers": True,
```

**Ne Zaman V3 Kullanılmalı?**

| Durum | Öneri |
|---|---|
| GPU ile eğitim | V3 (pin_memory + async transfer) |
| Değişken uzunluklu sequences | V3 (BucketBatchSampler) |
| Büyük dataset (>100K örnek) | V3 (DynamicPadding) |
| Cache bütünlüğü kritik | V3 (SHA-256 strict mode) |
| Source bazlı data split | V3 (source_id_aware_split) |
| CPU ile hızlı prototip | V2 (daha basit) |

---

## Konfigürasyon Parametreleri

### Kritik V5 Mimari Parametreleri

```python
TRAIN_CONFIG = {
    # Temel mimari
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 8,
    "ffn_dim": 2048,

    # GQA (Grouped Query Attention)
    "num_kv_heads": 2,           # 4 grup → %75 KV cache azalması

    # RoPE + YaRN
    "rope_scaling_type": "yarn",
    "rope_scaling_factor": 2.0,

    # Sliding Window Attention
    "sliding_window": 512,

    # Optimizer
    "use_adamw8bit": True,       # bitsandbytes — ~8 GB VRAM tasarrufu

    # V3 Training System
    "use_v3_training_system": True,
    "use_bucket_batching": True,
    "num_buckets": 32,
}
```

### Gelişmiş Eğitim Parametreleri (V3 ConfigManagerV3)

```python
# Entropy Regularization (Pereyra et al. 2017)
"entropy_coeff": 0.01,          # 0 = devre dışı

# Focal Loss (Lin et al. 2017)
"focal_loss_gamma": 2.0,        # 0 = standart CrossEntropy

# SAM (Sharpness-Aware Minimization)
"use_sam": False,
"sam_rho": 0.05,

# Lookahead
"use_lookahead": False,
"lookahead_k": 5,
"lookahead_alpha": 0.5,

# AGC (Adaptive Gradient Clipping)
"agc_clip_val": 0.01,

# EMA (Exponential Moving Average)
"use_ema": True,
"ema_decay": 0.999,

# SWA (Stochastic Weight Averaging)
"use_swa": False,
"swa_start_epoch": 5,
"swa_lr": 1e-4,

# LLRD (Layer-wise Learning Rate Decay)
"use_llrd": False,
"llrd_decay": 0.9,

# Scheduled Sampling
"scheduled_sampling_start": 0.0,

# NaN Recovery
"nan_recovery_enabled": True,
"spike_detection_enabled": True,
"spike_threshold": 5.0,
```

---

## GPU Optimizasyonları

### Bellek Yönetimi

```python
# train.py başında — OOM parçalanmasını önler
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# bitsandbytes ile 8-bit AdamW
"use_adamw8bit": True  # ~8 GB VRAM tasarrufu
```

### DataLoader GPU Optimizasyonları

```python
DataLoader(
    pin_memory=True,          # RAM → GPU DMA hızlı transfer
    num_workers=4,            # Parallel veri yükleme
    prefetch_factor=2,        # 2 batch önceden hazırla
    persistent_workers=True,  # Process overhead sıfırla
)
```

### Gradient Optimizasyonları

```python
"use_amp": True,              # Mixed precision (FP16/BF16)
"grad_accum_steps": 4,        # Effective batch_size = batch * 4
"max_grad_norm": 1.0,         # Gradient clipping
"agc_clip_val": 0.01,         # Adaptive Gradient Clipping
```

---

## Cache Sistemi Karşılaştırması

| Özellik | V2 DataCache | V3 DataCacheV3 |
|---|---|---|
| Bulunamama durumu | `None` döndür | `CacheNotFoundError` fırlat |
| Bütünlük doğrulama | Yok | SHA-256 checksum |
| Metadata | Yok | `.meta.json` (tarih, param, boyut) |
| Atomic write | Yok (doğrudan yaz) | tmp → rename |
| Cache key bileşeni | ~5 bileşen | 9 bileşen |
| İzin verilen hata | Evet (fallback) | Hayır (strict) |
| Kullanım senaryosu | Prototip/geliştirme | Production eğitimi |

---

*Yazar: Muhammed Yasin Yılmaz — Cevahir-AI Projesi*
*© 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.*
