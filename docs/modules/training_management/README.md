# Training System V3 — Modül Dokümantasyonu

**Versiyon:** V3 (Strict Cache + Advanced GPU Batching)
**Dizin:** `training_system/v3/`
**Giriş Noktası:** `training_system/train.py`
**Son Güncelleme:** 2026-03-16

---

## İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Zorunlu Eğitim Akışı](#zorunlu-eğitim-akışı)
3. [V3 Dizin Yapısı](#v3-dizin-yapısı)
4. [V2 → V3 Kritik Değişiklikler](#v2--v3-kritik-değişiklikler)
5. [Bileşenler](#bileşenler)
   - [TrainingServiceV3](#trainingservicev3)
   - [ConfigManagerV3](#configmanagerv3)
   - [DataCacheV3](#datacachev3)
   - [CevahirDataset](#cevahirdataset)
   - [BucketBatchSampler](#bucketbatchsampler)
   - [DynamicPaddingCollator](#dynamicpaddingcollator)
   - [DataLoader Factory](#dataloader-factory)
6. [train.py — TRAIN\_CONFIG](#trainpy--train_config)
7. [55+ Parametre Referansı](#55-parametre-referansı)
8. [GPU Optimizasyonları](#gpu-optimizasyonları)
9. [Cache Sistemi](#cache-sistemi)
10. [Entegrasyon](#entegrasyon)

---

## Genel Bakış

Training System V3, V2'nin tam yeniden yazımıdır. Üç temel yenilik taşır:

| Yenilik | V2 | V3 |
|---|---|---|
| Cache yönetimi | Optional fallback (raw data işlenebilir) | **Strict mode** — cache yoksa `CacheNotFoundError` |
| GPU batching | Statik pad (global `max_seq_length`) | **BucketSampler + DynamicPad** |
| Config aktarımı | ~20 parametre | **55+ parametre** |
| Train/val split | Basit random split | **Source-ID aware split** (data leakage yok) |
| Cache doğrulama | Yok | **SHA-256 checksum + JSON metadata** |

```
training_system/
├── train.py                       ← Giriş noktası (TRAIN_CONFIG burada)
├── prepare_cache.py               ← Cache hazırlama (adım 2)
├── v2/                            ← Geriye dönük uyumluluk
│   └── core/training_service.py
└── v3/                            ← Yeni pipeline
    ├── core/
    │   ├── training_service_v3.py ← Orchestrator
    │   └── config_manager_v3.py   ← 55+ parametre aktarımı
    └── data/
        ├── cache_v3.py            ← Strict cache manager
        ├── dataset_v3.py          ← CevahirDataset (uzunluk indeksi)
        ├── sampler_v3.py          ← BucketBatchSampler
        ├── collator_v3.py         ← DynamicPaddingCollator
        └── dataloader_v3.py       ← DataLoader factory
```

---

## Zorunlu Eğitim Akışı

```
ADIM 1 → python tokenizer_management/train_bpe.py
           BPE vocab/merges dosyaları oluşturulur

ADIM 2 → python training_system/prepare_cache.py
           Eğitim verisi tokenize edilir ve cache'e yazılır
           (SHA-256 checksum + JSON metadata ile)

ADIM 3 → python training_system/train.py
           Model eğitimi başlar
           Cache yoksa: CacheNotFoundError → eğitim başlamaz
```

> **⚠️ ÖNEMLİ:** Adım 3 adım 2 çıktısı olmadan çalışmaz. Bu izolasyon kasıtlıdır — veri hazırlama ve model eğitimini birbirinden ayırır (MLOps best practice).

---

## V3 Dizin Yapısı

```
training_system/v3/
├── core/
│   ├── __init__.py
│   ├── training_service_v3.py     ← TrainingServiceV3
│   └── config_manager_v3.py       ← ConfigManagerV3
├── data/
│   ├── __init__.py
│   ├── cache_v3.py                ← DataCacheV3, CacheNotFoundError
│   ├── dataset_v3.py              ← CevahirDataset
│   ├── sampler_v3.py              ← BucketBatchSampler
│   ├── collator_v3.py             ← DynamicPaddingCollator
│   └── dataloader_v3.py           ← create_dataloaders_v3
└── utils/
    └── __init__.py
```

---

## V2 → V3 Kritik Değişiklikler

### 1. Strict Cache Mode

```
V2: Cache yoksa → raw data okunur ve işlenir (implicit, yavaş)
V3: Cache yoksa → CacheNotFoundError (explicit, hızlı hata)
```

`CacheNotFoundError` içinde:
- Aranan `cache_key` ve `data_hash`
- Mevcut cache dosyaları listesi (varsa boyut + metadata)
- Key uyuşmazlığının olası nedenleri (4 madde)
- Adım adım çözüm talimatı

### 2. Source-ID Aware Train/Val Split

V2'de basit random shuffle yapılıyordu. Aynı belgeden (source) gelen chunk'lar train ve val'a dağılabiliyordu — **data leakage**.

V3'te:
```
source_id çıkarım → unique source'lar %80/%20 bölünür →
aynı source'un TÜM chunk'ları aynı split'e gider
```

Source ID yoksa: basit random split yapılır ve WARNING logu yazılır.

### 3. GPU Batching Stack

```
V2: torch.stack() → tüm sequence'lar max_seq_length'e pad (statik)
    GPU zamanının %70-90'ı PAD token hesaplar

V3: BucketBatchSampler → benzer uzunluktakiler aynı batch'e
    DynamicPaddingCollator → batch içi maksimum uzunluğa pad
    → padding waste %20-40'a düşer (Schwartz et al. 2020)
```

### 4. Config: 55+ Parametre

V2 ConfigManager ~20 parametre geçiriyordu. V3, entropy regularization'dan SWA'ya, curriculum learning'den loss spike detection'a kadar tüm parametreleri TrainingManager'a eksiksiz aktarır.

### 5. Cache Integrity

Her cache dosyası yazılırken:
- `cached_data_<key>_<hash>.pkl` → ana veri
- `cached_data_<key>_<hash>.sha256` → SHA-256 checksum
- `cached_data_<key>_<hash>.meta.json` → human-readable metadata

Yüklemede checksum doğrulanır. Uyuşmazlık → `CacheIntegrityError`.

---

## Bileşenler

### TrainingServiceV3

**Dosya:** `v3/core/training_service_v3.py`
**Pattern:** Facade Pattern — tüm V3 pipeline'ını orkestre eder.

```python
from training_system.v3 import TrainingServiceV3

service = TrainingServiceV3(config=TRAIN_CONFIG)
train_loss, val_loss = service.train()
```

**`__init__` adımları (sırayla):**

1. BPE dosya yolları → dizin oluşturma
2. Device seçimi (GPU/CPU)
3. `data_dir` varlığı kontrolü
4. `BPEValidator` — vocab/merges formatı doğrulama
5. `TokenizerCore` başlatma
6. Vocab size TokenizerCore'dan al → config override
7. `DataCacheV3` strict mode ile başlat
8. `ModelManager.initialize(optimizer=True, criterion=False, scheduler=True)`
9. `CriterionManager` → `entropy_coeff` destekli loss oluştur
10. `ConfigManagerV3` başlat

**`train()` pipeline (5 adım):**

```
1. Model initialize (checkpoint yükle — last.pth → best.pth → en yeni .pth)
2. load_data_from_cache() → CacheNotFoundError fırlatabilir
3. create_dataloaders_v3() (BucketSampler + DynamicPad)
4. ConfigManagerV3.prepare_training_config() → 55+ parametre
5. TrainingManager.train(epoch_callback=...) → (train_loss, val_loss)
```

**Epoch sonu test:** Her epoch sonunda model `.eval()` moduna alınır, test prompt'ları için top-k=80 örnekleme ile inference yapılır ve sonuçlar loglanır.

**Checkpoint arama sırası:**
```
resume_from_path → load_checkpoint_path → last.pth → best.pth → checkpoint_*.pth (en yeni)
```

---

### ConfigManagerV3

**Dosya:** `v3/core/config_manager_v3.py`
**Pattern:** Adapter Pattern — `TRAIN_CONFIG` → TrainingManager config sözlüğü

```python
config = config_manager.prepare_training_config(
    base_config=TRAIN_CONFIG,
    tokenizer_core=tok,
    device="cuda"
)
# → 55+ parametreli dict
```

**11 Parametre Grubu:**

| Grup | İçerik |
|---|---|
| 1. Temel | `epochs`, `batch_size`, `max_grad_norm`, `grad_accum_steps`, `use_amp` |
| 2. Loss | `label_smoothing`, `entropy_coeff`, `use_focal_loss`, `focal_gamma`, `aux_loss_weight` |
| 3. Optimizer | SAM, Lookahead, AGC, Gradient Noise |
| 4. EMA / SWA | `use_ema`, `ema_decay`, `use_swa`, `swa_start_epoch`, `swa_lr` |
| 5. LR Schedule | LLRD, Cosine Restarts |
| 6. Scheduled Sampling | `use_scheduled_sampling`, `ss_start_epoch`, `ss_decay_rate`, `min_teacher_forcing` |
| 7. Curriculum | `use_curriculum`, `curriculum_strategy`, `curriculum_max_len_start` |
| 8. Güvenlik | NaN tolerance, NaN LR reduction, spike detection |
| 9. Monitoring | `inference_probe_interval`, `log_gradient_health`, `log_token_dist` |
| 10. GPU Batching | `use_bucket_batching`, `num_buckets`, `use_dynamic_padding`, workers |
| 11. Cache | `cache_dir`, `cache_strict_mode`, `cache_verify_integrity` |

**Validasyon (config üretildikten sonra çalışır):**

```python
# Hata fırlatan kontroller:
label_smoothing  ∈ [0, 0.5]
entropy_coeff    ∈ [0, 1.0]
ema_decay        ∈ (0, 1)
batch_size       > 0
epochs           > 0
```

---

### DataCacheV3

**Dosya:** `v3/data/cache_v3.py`
**Pattern:** Cache Pattern + Fail-Fast Pattern

```python
cache = DataCacheV3(
    data_dir="education/",
    cache_dir=".cache/preprocessed_data",
    strict_mode=True,          # V3: cache yoksa hata
    verify_integrity=True,     # V3: SHA-256 checksum
)
```

**Cache Key Bileşenleri:**

```
cache_key = MD5(
    data_dir_normalized |
    encode_mode |
    include_whole_words |
    include_syllables |
    include_sep |
    max_seq_length |
    vocab_hash |
    alignment_format |
    formatted_True
)
```

Cache key uyuşmazlığının tipik nedenleri:
- `max_seq_length` değişti
- Vocab dosyası güncellendi (vocab_hash değişti)
- `alignment_format` değişti
- Eğitim verisi değişti (data_hash değişti)

**Cache Dosyası Yapısı:**

```
.cache/preprocessed_data/
├── cached_data_<key16>_<hash8>.pkl        ← pickle veri
├── cached_data_<key16>_<hash8>.sha256     ← SHA-256 checksum
└── cached_data_<key16>_<hash8>.meta.json  ← human-readable metadata
```

**Metadata içeriği:**
```json
{
  "version": "v3",
  "created_at": "2026-03-16 10:30:00",
  "cache_key": "...",
  "data_hash": "...",
  "encode_mode": "train",
  "max_seq_length": 768,
  "sample_count": 560000,
  "file_size_mb": 1240.5
}
```

**Public API:**

| Metod | Açıklama |
|---|---|
| `load_for_training(tokenizer_core, ...)` | Strict yükleme — hata fırlatır |
| `save(cache_key, data_hash, data, ...)` | Atomic write + checksum + metadata |
| `clear()` | Tüm cache dosyalarını sil |
| `list_caches()` | Mevcut cache'leri metadata ile listele |

---

### CevahirDataset

**Dosya:** `v3/data/dataset_v3.py`
**Temel:** `torch.utils.data.Dataset`

```python
dataset = CevahirDataset(
    data=train_data,           # List[(inp_tensor, tgt_tensor)]
    pad_id=0,
    precompute_lengths=True,   # BucketBatchSampler için uzunlukları önceden hesapla
)
```

**Özellikler:**

- **Uzunluk indeksi:** Her sequence'ın gerçek uzunluğu (PAD'lar sayılmaz) `BucketBatchSampler`'a sağlanır
- **Lazy tensor:** Zaten tensor ise dönüşüm yapılmaz
- **Source ID yönetimi:** 3-tuple `(inp, tgt, source_id)` → `__getitem__` source_id'yi atar
- **İstatistik raporu:** `get_length_stats()` → min/max/mean/median/p25/p75/p90/p99

```python
stats = dataset.get_length_stats()
# → {"count": 560000, "min": 4, "max": 768, "mean": 312.5, ...}
```

---

### BucketBatchSampler

**Dosya:** `v3/data/sampler_v3.py`
**Temel:** `torch.utils.data.Sampler[List[int]]`

**Algoritma:**

```
1. Tüm sequence'ları uzunluğa göre sırala
2. num_buckets adet gruba (bucket) böl
3. Epoch başında bucket içlerini karıştır
4. Her bucket'tan batch'ler oluştur
5. Tüm batch'leri karıştır (bucket sıralaması belli olmasın)
```

```python
sampler = BucketBatchSampler(
    lengths=dataset.lengths,    # Her sequence'ın gerçek uzunluğu
    batch_size=64,
    num_buckets=32,             # Daha fazla → daha iyi gruplama, daha az çeşitlilik
    shuffle_buckets=True,       # Epoch bazlı randomness
    shuffle_within_bucket=True,
    drop_last=False,
    seed=42,
)
sampler.set_epoch(epoch)        # Epoch bazlı seed değişimi
```

**Padding Tasarrufu (Schwartz et al. 2020):**

```
Statik padding: Tüm batch → max_seq_length=768
  → GPU zamanının %70-90'ı PAD token'a gider

BucketSampler + DynamicPad: Benzer uzunluktakiler → batch içi max pad
  → padding waste %20-40'a düşer
  → GPU throughput artar
```

---

### DynamicPaddingCollator

**Dosya:** `v3/data/collator_v3.py`

Batch içindeki sequence'ların maksimum uzunluğuna kadar pad eder.

```python
collator = DynamicPaddingCollator(
    pad_id=0,
    max_seq_length=768,   # Absolut üst limit (güvenlik)
    non_blocking=True,    # Async GPU transfer
)
```

**Karşılaştırma:**

```
V2 (statik):   Batch {len=10, len=12, len=15} → pad to 768
V3 (dinamik):  Batch {len=10, len=12, len=15} → pad to 15
```

Kısa sequence batchleri için GPU %97 daha az PAD token hesaplar.

**Akademik referans:** Ott et al. 2019, fairseq — dynamic padding ile %2-3x throughput artışı.

Geriye dönük uyumluluk için `create_static_collate()` factory fonksiyonu da mevcuttur (V2 arayüzü).

---

### DataLoader Factory

**Dosya:** `v3/data/dataloader_v3.py`

```python
from training_system.v3.data.dataloader_v3 import create_dataloaders_v3

train_loader, val_loader = create_dataloaders_v3(
    train_data=train_data,
    val_data=val_data,
    batch_size=64,
    pad_id=0,
    device="cuda",
    use_bucket_batching=True,
    num_buckets=32,
    use_dynamic_padding=True,
    max_seq_length=768,
    num_workers=4,             # Linux/Colab; Windows'ta 0 önerilir
    pin_memory=True,           # CUDA: sabitlenmiş RAM → DMA transfer
    prefetch_factor=2,
    persistent_workers=True,
)
```

**GPU Optimizasyon Katmanları:**

| Optimizasyon | Parametre | Etki |
|---|---|---|
| `pin_memory=True` | Sabitlenmiş RAM | PCIe üzerinden async DMA transfer |
| `num_workers=4` | Paralel prefetch | CPU/GPU örtüşümü |
| `prefetch_factor=2` | Worker başına 2 batch | Veri bant genişliği kullanımı |
| `persistent_workers=True` | Worker'lar canlı | Epoch arası başlatma overhead'i yok |
| `BucketBatchSampler` | Uzunluk gruplaması | Padding waste ↓ |
| `DynamicPaddingCollator` | Batch-aware pad | GPU memory ↓ |

**Train vs Val farklılıkları:**

```
Train: shuffle=True, BucketBatchSampler, DynamicPad
Val:   shuffle=False, SequentialSampler, DynamicPad (bucket yok)
```

---

## train.py — TRAIN_CONFIG

`training_system/train.py` içindeki `TRAIN_CONFIG` tüm eğitim parametrelerini içerir.

**`main()` Akışı:**

```
1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True → fragmentation azaltma
2. set_seed(42)
3. log_env_info() → GPU adı, CC, VRAM bilgisi
4. ensure_dirs() → dizinleri oluştur
5. normalize_config(TRAIN_CONFIG) → BPE/tokenizer ayarları config'ten yükle
6. continuation_lr kontrolü → checkpoint varsa LR override
7. V3 mevcut ise TrainingServiceV3 başlat, değilse V2 fallback
8. service.train() → eğitim başlar
```

**Config Normalizasyonu (`normalize_config`):**

- `gradient_clip` → `max_grad_norm` alias
- BPE/tokenizer ayarları `tokenizer_management/config.py`'den otomatik yüklenir (hardcoded değer yok)
- Scheduler kwargs oluşturulur (`lr_decay_factor`, `lr_decay_patience`, `lr_threshold`, `lr_min`)
- TensorBoard varsayılanları set edilir
- `torch.set_float32_matmul_precision("high")` (PyTorch 2.x)

---

## 55+ Parametre Referansı

### Temel Eğitim

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `epochs` | 100 | Epoch sayısı |
| `batch_size` | 64 | Mini-batch boyutu |
| `learning_rate` | 0.0002 | Öğrenme hızı |
| `grad_accum_steps` | 8 | Gradient accumulation adımları (efektif batch = 64×8=512) |
| `max_grad_norm` | 1.0 | Gradient clipping üst sınırı |
| `use_amp` | True | Mixed precision (AMP) |
| `early_stopping_patience` | 10 | Epoch sayısı sabır |
| `dropout` | 0.2 | Dropout oranı |
| `weight_decay` | 0.01 | L2 regularizasyon |

### Optimizer

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `optimizer` | `"adamw8bit"` | `adamw` / `adamw8bit` / `adam` / `radam` / `sgd` |
| `use_sam` | `False` | Sharpness-Aware Minimization (Foret et al. 2021) |
| `sam_rho` | 0.05 | SAM pertürbation büyüklüğü |
| `use_lookahead` | `False` | Lookahead (Zhang et al. 2019) |
| `lookahead_k` | 5 | Slow weights güncelleme sıklığı |
| `lookahead_alpha` | 0.5 | Slow weights interpolasyon faktörü |
| `use_agc` | `False` | Adaptive Gradient Clipping (Brock et al. 2021) |
| `use_gradient_noise` | `False` | Gradient Noise (Neelakantan et al. 2015) |

### Loss Fonksiyonu

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `label_smoothing` | 0.1 | Label Smoothing (Szegedy et al. 2016) |
| `eos_token_weight` | 1.0 | EOS token ağırlığı (1.0 = standart) |
| `entropy_coeff` | 0.01 | Entropy regularization (Pereyra et al. 2017) — overconfidence cezası |
| `use_focal_loss` | `False` | Focal Loss (Lin et al. 2017) |
| `focal_gamma` | 2.0 | Focal Loss gamma |
| `aux_loss_weight` | 0.01 | MoE auxiliary loss ağırlığı |

### Scheduler & LR

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `scheduler_type` | `"reduce_on_plateau"` | `plateau` / `cosine` / `cawr` / `step` / `onecycle` |
| `lr_decay_factor` | 0.75 | Plateau faktörü |
| `lr_decay_patience` | 15 | Plateau sabır (epoch) |
| `lr_min` | 1e-6 | Minimum LR |
| `warmup_steps` | 1500 | Warmup adım sayısı (dinamik hesaplanır) |
| `warmup_start_factor` | 0.1 | Warmup başlangıç LR çarpanı |
| `use_llrd` | `False` | Layer-wise LR Decay |
| `llrd_decay_factor` | 0.9 | Katman başına LR çarpanı |
| `use_cosine_restarts` | `False` | SGDR (Loshchilov & Hutter 2016) |

### EMA & SWA

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `use_ema` | `True` | Exponential Moving Average |
| `ema_decay` | 0.999 | EMA bozunum faktörü |
| `ema_update_after_step` | 100 | İlk N adımdan sonra EMA güncelle |
| `ema_update_every` | 10 | Her N adımda bir güncelle |
| `use_swa` | `False` | Stochastic Weight Averaging (Izmailov et al. 2018) |
| `swa_start_epoch` | 80 | SWA başlangıç epoch'u |
| `swa_lr` | 1e-5 | SWA sabit LR |
| `swa_anneal_epochs` | 10 | Annealing epoch sayısı |

### Exposure Bias & Curriculum

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `use_scheduled_sampling` | `True` | Scheduled Sampling (Bengio et al. 2015) |
| `ss_start_epoch` | 10 | Scheduled sampling başlangıç epoch'u |
| `ss_decay_rate` | 0.05 | Her epoch teacher forcing düşüşü |
| `min_teacher_forcing` | 0.3 | Teacher forcing alt sınırı |
| `use_curriculum` | `False` | Curriculum Learning (Bengio et al. 2009) |
| `curriculum_strategy` | `"length_based"` | `length_based` / `loss_based` |
| `curriculum_max_len_start` | 64 | Başlangıçta max sequence uzunluğu |
| `curriculum_warmup_epochs` | 20 | Tam veriyi görene kadar epoch |

### Güvenlik

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `nan_tolerance` | 3 | Art arda NaN sayısı (checkpoint'e geri dön) |
| `nan_lr_reduction` | 0.5 | NaN sonrası LR çarpanı |
| `spike_n_sigma` | 3.0 | Loss spike eşiği (N-sigma) |
| `spike_window_size` | 20 | Referans pencere büyüklüğü (batch) |
| `spike_lr_reduction` | 0.8 | Spike sonrası LR çarpanı |

### Model (V5 Mimari)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `embed_dim` | 512 | Embedding boyutu |
| `num_heads` | 8 | Attention head sayısı |
| `num_kv_heads` | 2 | GQA KV head sayısı (%75 KV cache azalması) |
| `num_layers` | 8 | Transformer layer sayısı |
| `ffn_dim` | `None` | `None` → otomatik `embed_dim × 4` |
| `pe_mode` | `"rope"` | Positional encoding türü |
| `rope_scaling_type` | `"yarn"` | `"none"` / `"yarn"` / `"linear"` |
| `rope_scaling_factor` | 2.0 | YaRN context uzatma faktörü |
| `sliding_window` | 512 | Sliding window attention boyutu |
| `use_rmsnorm` | `True` | RMSNorm |
| `use_swiglu` | `True` | SwiGLU activation |
| `use_kv_cache` | `True` | KV Cache |
| `use_moe` | `False` | Mixture of Experts |
| `num_experts` | 8 | Expert sayısı (MoE aktifse) |
| `moe_top_k` | 2 | Her token için seçilen expert sayısı |

### GPU Batching & Cache

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `use_bucket_batching` | `True` | BucketBatchSampler |
| `num_buckets` | 32 | Bucket sayısı |
| `use_dynamic_padding` | `True` | DynamicPaddingCollator |
| `data_loader_num_workers` | 4 (Linux) / 0 (Win) | DataLoader worker sayısı |
| `data_loader_pin_memory` | `True` | Sabitlenmiş RAM |
| `prefetch_factor` | 2 | Worker başına prefetch batch |
| `persistent_workers` | `True` | Epoch arası worker canlılığı |
| `cache_strict_mode` | `True` | Cache zorunlu (hata fırlat) |
| `cache_verify_integrity` | `True` | SHA-256 checksum doğrulama |

---

## GPU Optimizasyonları

### OOM Koruması

```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

Büyük contiguous blok bulunamadığında allocator parçalı segment kullanır → fragmentation %30-60 azalır.

### AMP (Mixed Precision)

```python
"use_amp": True   # FP16/BF16 forward pass, FP32 gradient
```

A100'de ~2x hız artışı. `torch.set_float32_matmul_precision("high")` ile matmul hassasiyeti optimize edilir.

### Gradient Accumulation

```python
"batch_size": 64,
"grad_accum_steps": 8,
# Efektif batch = 64 × 8 = 512
```

Büyük efektif batch'i düşük GPU memory ile simüle eder.

---

## Cache Sistemi

### Cache Neden Zorunlu?

Tokenizasyon, model eğitiminden çok daha uzun sürer. Raw veriden her eğitimde tokenize etmek:
- Eğitim başlangıcını 10-30 dakika geciktirir
- Tokenizasyon tekrarlanabilir olmazdı (vocab değişebilir)
- Veri bütünlüğü doğrulanamaz

V3 strict mode ile bu sorunlar ortadan kalkar.

### Cache Geçersizleşme Koşulları

Cache key bileşenlerinden herhangi biri değişirse cache geçersiz sayılır:

```
1. max_seq_length değişti → farklı seq uzunluğu
2. vocab_hash değişti → BPE vocab/merges güncellendi
3. alignment_format değişti → autoregressive format değişti
4. data_hash değişti → eğitim verisi değişti (dosya adı/boyutu)
5. encode_mode, include_whole_words, include_syllables, include_sep değişti
```

Çözüm:
```bash
python training_system/prepare_cache.py
```

---

## Entegrasyon

### train.py'de V3/V2 Otomatik Seçimi

```python
try:
    from training_system.v3 import TrainingServiceV3
    _TRAINING_SYSTEM_V3_AVAILABLE = True
except ImportError:
    TrainingServiceV3 = None
    _TRAINING_SYSTEM_V3_AVAILABLE = False

# main() içinde:
if _TRAINING_SYSTEM_V3_AVAILABLE:
    service = TrainingServiceV3(config=effective_cfg)
else:
    service = TrainingService(config=effective_cfg)   # V2 fallback
```

### TrainingManager Seçimi (V3 Service içinde)

```python
try:
    from training_management.v3 import TrainingManager as V3TrainingManager
    _has_v3 = True
except ImportError:
    _has_v3 = False

# Hata durumunda V2 TrainingManager kullanılır
from training_management.v2.core.training_manager import TrainingManager as V2TrainingManager
```

### V2 ile Geriye Dönük Uyumluluk

V3 şu V2 bileşenlerini hâlâ kullanır:
- `training_management.v2.utils.checkpoint_manager.CheckpointManager`
- `training_management.v2.monitoring.tensorboard_manager.TensorBoardManager`
- `training_management.v2.utils.training_logger.TrainingLogger`
- `training_management.v2.utils.training_scheduler.TrainingScheduler`
- `training_system.v2.core.bpe_validator.BPEValidator`
- `training_system.v2.core.criterion_manager.CriterionManager`
- `training_system.v2.utils.warmup_calculator.calculate_warmup_steps`

---

*Yazar: Muhammed Yasin Yılmaz | Telif Hakkı © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.*
