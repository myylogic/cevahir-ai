# Training System V3 — Module Documentation

**Version:** V3 (Strict Cache + Advanced GPU Batching)
**Directory:** `training_system/v3/`
**Entry Point:** `training_system/train.py`
**Last Updated:** 2026-03-16

---

## Table of Contents

1. [Overview](#overview)
2. [Mandatory Training Flow](#mandatory-training-flow)
3. [V3 Directory Structure](#v3-directory-structure)
4. [V2 → V3 Critical Changes](#v2--v3-critical-changes)
5. [Components](#components)
   - [TrainingServiceV3](#trainingservicev3)
   - [ConfigManagerV3](#configmanagerv3)
   - [DataCacheV3](#datacachev3)
   - [CevahirDataset](#cevahirdataset)
   - [BucketBatchSampler](#bucketbatchsampler)
   - [DynamicPaddingCollator](#dynamicpaddingcollator)
   - [DataLoader Factory](#dataloader-factory)
6. [train.py — TRAIN_CONFIG](#trainpy--train_config)
7. [55+ Parameter Reference](#55-parameter-reference)
8. [GPU Optimizations](#gpu-optimizations)
9. [Cache System](#cache-system)
10. [Integration](#integration)

---

## Overview

Training System V3 is a full rewrite of V2. It introduces three main changes:

| Change | V2 | V3 |
|---|---|---|
| Cache handling | Optional fallback (raw data can be processed) | **Strict mode** — no cache → `CacheNotFoundError` |
| GPU batching | Static pad (global `max_seq_length`) | **BucketSampler + DynamicPad** |
| Config passing | ~20 parameters | **55+ parameters** |
| Train/val split | Simple random split | **Source-ID aware split** (no data leakage) |
| Cache validation | None | **SHA-256 checksum + JSON metadata** |

```
training_system/
├── train.py                       ← Entry point (TRAIN_CONFIG here)
├── prepare_cache.py               ← Cache preparation (step 2)
├── v2/                            ← Backward compatibility
│   └── core/training_service.py
└── v3/                            ← New pipeline
    ├── core/
    │   ├── training_service_v3.py ← Orchestrator
    │   └── config_manager_v3.py   ← 55+ parameter passing
    └── data/
        ├── cache_v3.py            ← Strict cache manager
        ├── dataset_v3.py          ← CevahirDataset (length index)
        ├── sampler_v3.py          ← BucketBatchSampler
        ├── collator_v3.py         ← DynamicPaddingCollator
        └── dataloader_v3.py       ← DataLoader factory
```

---

## Mandatory Training Flow

```
STEP 1 → python tokenizer_management/train_bpe.py
           BPE vocab/merges files are created

STEP 2 → python training_system/prepare_cache.py
           Training data is tokenized and written to cache
           (with SHA-256 checksum + JSON metadata)

STEP 3 → python training_system/train.py
           Model training starts
           No cache: CacheNotFoundError → training does not start
```

> **⚠️ IMPORTANT:** Step 3 cannot run without step 2 output. This isolation is intentional — it separates data preparation from model training (MLOps best practice).

---

## V3 Directory Structure

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

## V2 → V3 Critical Changes

### 1. Strict Cache Mode

```
V2: No cache → raw data is read and processed (implicit, slow)
V3: No cache → CacheNotFoundError (explicit, fast failure)
```

`CacheNotFoundError` includes:
- Requested `cache_key` and `data_hash`
- List of existing cache files (with size + metadata if any)
- Possible causes of key mismatch (4 items)
- Step-by-step resolution instructions

### 2. Source-ID Aware Train/Val Split

V2 used a simple random shuffle. Chunks from the same document (source) could end up in both train and val — **data leakage**.

In V3:
```
source_id extraction → unique sources split 80%/20% →
all chunks from the same source go to the same split
```

If no source ID: simple random split is used and a WARNING is logged.

### 3. GPU Batching Stack

```
V2: torch.stack() → pad all sequences to max_seq_length (static)
    ~70–90% of GPU time spent on PAD token computation

V3: BucketBatchSampler → similar-length sequences in same batch
    DynamicPaddingCollator → pad to batch-internal max length
    → padding waste drops to ~20–40% (Schwartz et al. 2020)
```

### 4. Config: 55+ Parameters

V2 ConfigManager passed ~20 parameters. V3 passes the full set to TrainingManager: entropy regularization, SWA, curriculum learning, loss spike detection, and more.

### 5. Cache Integrity

When each cache file is written:
- `cached_data_<key>_<hash>.pkl` → main data
- `cached_data_<key>_<hash>.sha256` → SHA-256 checksum
- `cached_data_<key>_<hash>.meta.json` → human-readable metadata

Checksum is verified on load. Mismatch → `CacheIntegrityError`.

---

## Components

### TrainingServiceV3

**File:** `v3/core/training_service_v3.py`
**Pattern:** Facade — orchestrates the full V3 pipeline.

```python
from training_system.v3 import TrainingServiceV3

service = TrainingServiceV3(config=TRAIN_CONFIG)
train_loss, val_loss = service.train()
```

**`__init__` steps (in order):**

1. BPE file paths → directory creation
2. Device selection (GPU/CPU)
3. `data_dir` existence check
4. `BPEValidator` — vocab/merges format validation
5. `TokenizerCore` initialization
6. Vocab size from TokenizerCore → config override
7. `DataCacheV3` start in strict mode
8. `ModelManager.initialize(optimizer=True, criterion=False, scheduler=True)`
9. `CriterionManager` → loss with `entropy_coeff` support
10. `ConfigManagerV3` start

**`train()` pipeline (5 steps):**

```
1. Model init (load checkpoint — last.pth → best.pth → newest .pth)
2. load_data_from_cache() → may raise CacheNotFoundError
3. create_dataloaders_v3() (BucketSampler + DynamicPad)
4. ConfigManagerV3.prepare_training_config() → 55+ parameters
5. TrainingManager.train(epoch_callback=...) → (train_loss, val_loss)
```

**End-of-epoch test:** At the end of each epoch the model is put in `.eval()` mode, inference is run on test prompts with top-k=80 sampling, and results are logged.

**Checkpoint search order:**
```
resume_from_path → load_checkpoint_path → last.pth → best.pth → checkpoint_*.pth (newest)
```

---

### ConfigManagerV3

**File:** `v3/core/config_manager_v3.py`
**Pattern:** Adapter — `TRAIN_CONFIG` → TrainingManager config dict

```python
config = config_manager.prepare_training_config(
    base_config=TRAIN_CONFIG,
    tokenizer_core=tok,
    device="cuda"
)
# → dict with 55+ parameters
```

**11 Parameter Groups:**

| Group | Contents |
|---|---|
| 1. Basic | `epochs`, `batch_size`, `max_grad_norm`, `grad_accum_steps`, `use_amp` |
| 2. Loss | `label_smoothing`, `entropy_coeff`, `use_focal_loss`, `focal_gamma`, `aux_loss_weight` |
| 3. Optimizer | SAM, Lookahead, AGC, Gradient Noise |
| 4. EMA / SWA | `use_ema`, `ema_decay`, `use_swa`, `swa_start_epoch`, `swa_lr` |
| 5. LR Schedule | LLRD, Cosine Restarts |
| 6. Scheduled Sampling | `use_scheduled_sampling`, `ss_start_epoch`, `ss_decay_rate`, `min_teacher_forcing` |
| 7. Curriculum | `use_curriculum`, `curriculum_strategy`, `curriculum_max_len_start` |
| 8. Safety | NaN tolerance, NaN LR reduction, spike detection |
| 9. Monitoring | `inference_probe_interval`, `log_gradient_health`, `log_token_dist` |
| 10. GPU Batching | `use_bucket_batching`, `num_buckets`, `use_dynamic_padding`, workers |
| 11. Cache | `cache_dir`, `cache_strict_mode`, `cache_verify_integrity` |

**Validation (runs after config is produced):**

```python
# Checks that raise on error:
label_smoothing  ∈ [0, 0.5]
entropy_coeff    ∈ [0, 1.0]
ema_decay        ∈ (0, 1)
batch_size       > 0
epochs           > 0
```

---

### DataCacheV3

**File:** `v3/data/cache_v3.py`
**Pattern:** Cache + Fail-Fast

```python
cache = DataCacheV3(
    data_dir="education/",
    cache_dir=".cache/preprocessed_data",
    strict_mode=True,          # V3: error if no cache
    verify_integrity=True,     # V3: SHA-256 checksum
)
```

**Cache Key Components:**

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

Typical causes of cache key mismatch:
- `max_seq_length` changed
- Vocab file updated (vocab_hash changed)
- `alignment_format` changed
- Training data changed (data_hash changed)

**Cache File Layout:**

```
.cache/preprocessed_data/
├── cached_data_<key16>_<hash8>.pkl        ← pickle data
├── cached_data_<key16>_<hash8>.sha256    ← SHA-256 checksum
└── cached_data_<key16>_<hash8>.meta.json ← human-readable metadata
```

**Metadata contents:**
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

| Method | Description |
|---|---|
| `load_for_training(tokenizer_core, ...)` | Strict load — raises on error |
| `save(cache_key, data_hash, data, ...)` | Atomic write + checksum + metadata |
| `clear()` | Delete all cache files |
| `list_caches()` | List existing caches with metadata |

---

### CevahirDataset

**File:** `v3/data/dataset_v3.py`
**Base:** `torch.utils.data.Dataset`

```python
dataset = CevahirDataset(
    data=train_data,           # List[(inp_tensor, tgt_tensor)]
    pad_id=0,
    precompute_lengths=True,   # Precompute lengths for BucketBatchSampler
)
```

**Features:**

- **Length index:** True length of each sequence (excluding PADs) is provided to `BucketBatchSampler`
- **Lazy tensor:** If already a tensor, no conversion
- **Source ID handling:** 3-tuple `(inp, tgt, source_id)` → `__getitem__` drops source_id
- **Stats report:** `get_length_stats()` → min/max/mean/median/p25/p75/p90/p99

```python
stats = dataset.get_length_stats()
# → {"count": 560000, "min": 4, "max": 768, "mean": 312.5, ...}
```

---

### BucketBatchSampler

**File:** `v3/data/sampler_v3.py`
**Base:** `torch.utils.data.Sampler[List[int]]`

**Algorithm:**

```
1. Sort all sequences by length
2. Split into num_buckets groups (buckets)
3. Shuffle within buckets at epoch start
4. Form batches from each bucket
5. Shuffle all batches (so bucket order is not visible)
```

```python
sampler = BucketBatchSampler(
    lengths=dataset.lengths,    # True length of each sequence
    batch_size=64,
    num_buckets=32,             # More → better grouping, less diversity
    shuffle_buckets=True,       # Epoch-based randomness
    shuffle_within_bucket=True,
    drop_last=False,
    seed=42,
)
sampler.set_epoch(epoch)        # Epoch-based seed change
```

**Padding Savings (Schwartz et al. 2020):**

```
Static padding: All batches → max_seq_length=768
  → 70–90% of GPU time on PAD tokens

BucketSampler + DynamicPad: Similar lengths → pad to batch-internal max
  → padding waste drops to 20–40%
  → GPU throughput increases
```

---

### DynamicPaddingCollator

**File:** `v3/data/collator_v3.py`

Pads sequences to the maximum length within the batch.

```python
collator = DynamicPaddingCollator(
    pad_id=0,
    max_seq_length=768,   # Absolute upper limit (safety)
    non_blocking=True,    # Async GPU transfer
)
```

**Comparison:**

```
V2 (static):   Batch {len=10, len=12, len=15} → pad to 768
V3 (dynamic):  Batch {len=10, len=12, len=15} → pad to 15
```

For short-sequence batches, GPU computes ~97% fewer PAD tokens.

**Reference:** Ott et al. 2019, fairseq — ~2–3x throughput with dynamic padding.

For backward compatibility, `create_static_collate()` factory is also available (V2 interface).

---

### DataLoader Factory

**File:** `v3/data/dataloader_v3.py`

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
    num_workers=4,             # Linux/Colab; 0 recommended on Windows
    pin_memory=True,           # CUDA: pinned RAM → DMA transfer
    prefetch_factor=2,
    persistent_workers=True,
)
```

**GPU Optimization Layers:**

| Optimization | Parameter | Effect |
|---|---|---|
| `pin_memory=True` | Pinned RAM | Async DMA transfer over PCIe |
| `num_workers=4` | Parallel prefetch | CPU/GPU overlap |
| `prefetch_factor=2` | 2 batches per worker | Better bandwidth use |
| `persistent_workers=True` | Workers stay alive | No startup overhead between epochs |
| `BucketBatchSampler` | Length grouping | Padding waste ↓ |
| `DynamicPaddingCollator` | Batch-aware pad | GPU memory ↓ |

**Train vs Val:**

```
Train: shuffle=True, BucketBatchSampler, DynamicPad
Val:   shuffle=False, SequentialSampler, DynamicPad (no buckets)
```

---

## train.py — TRAIN_CONFIG

`TRAIN_CONFIG` in `training_system/train.py` holds all training parameters.

**`main()` flow:**

```
1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True → reduce fragmentation
2. set_seed(42)
3. log_env_info() → GPU name, CC, VRAM info
4. ensure_dirs() → create directories
5. normalize_config(TRAIN_CONFIG) → load BPE/tokenizer settings from config
6. continuation_lr check → checkpoint present → LR override
7. If V3 available, start TrainingServiceV3; else V2 fallback
8. service.train() → training starts
```

**Config normalization (`normalize_config`):**

- `gradient_clip` → `max_grad_norm` alias
- BPE/tokenizer settings loaded from `tokenizer_management/config.py` (no hardcoded values)
- Scheduler kwargs built (`lr_decay_factor`, `lr_decay_patience`, `lr_threshold`, `lr_min`)
- TensorBoard defaults set
- `torch.set_float32_matmul_precision("high")` (PyTorch 2.x)

---

## 55+ Parameter Reference

### Basic Training

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 100 | Number of epochs |
| `batch_size` | 64 | Mini-batch size |
| `learning_rate` | 0.0002 | Learning rate |
| `grad_accum_steps` | 8 | Gradient accumulation steps (effective batch = 64×8=512) |
| `max_grad_norm` | 1.0 | Gradient clipping upper bound |
| `use_amp` | True | Mixed precision (AMP) |
| `early_stopping_patience` | 10 | Early stopping patience (epochs) |
| `dropout` | 0.2 | Dropout rate |
| `weight_decay` | 0.01 | L2 regularization |

### Optimizer

| Parameter | Default | Description |
|---|---|---|
| `optimizer` | `"adamw8bit"` | `adamw` / `adamw8bit` / `adam` / `radam` / `sgd` |
| `use_sam` | `False` | Sharpness-Aware Minimization (Foret et al. 2021) |
| `sam_rho` | 0.05 | SAM perturbation size |
| `use_lookahead` | `False` | Lookahead (Zhang et al. 2019) |
| `lookahead_k` | 5 | Slow-weights update frequency |
| `lookahead_alpha` | 0.5 | Slow-weights interpolation factor |
| `use_agc` | `False` | Adaptive Gradient Clipping (Brock et al. 2021) |
| `use_gradient_noise` | `False` | Gradient Noise (Neelakantan et al. 2015) |

### Loss

| Parameter | Default | Description |
|---|---|---|
| `label_smoothing` | 0.1 | Label Smoothing (Szegedy et al. 2016) |
| `eos_token_weight` | 1.0 | EOS token weight (1.0 = standard) |
| `entropy_coeff` | 0.01 | Entropy regularization (Pereyra et al. 2017) — overconfidence penalty |
| `use_focal_loss` | `False` | Focal Loss (Lin et al. 2017) |
| `focal_gamma` | 2.0 | Focal Loss gamma |
| `aux_loss_weight` | 0.01 | MoE auxiliary loss weight |

### Scheduler & LR

| Parameter | Default | Description |
|---|---|---|
| `scheduler_type` | `"reduce_on_plateau"` | `plateau` / `cosine` / `cawr` / `step` / `onecycle` |
| `lr_decay_factor` | 0.75 | Plateau factor |
| `lr_decay_patience` | 15 | Plateau patience (epochs) |
| `lr_min` | 1e-6 | Minimum LR |
| `warmup_steps` | 1500 | Warmup steps (dynamically computed) |
| `warmup_start_factor` | 0.1 | Warmup initial LR multiplier |
| `use_llrd` | `False` | Layer-wise LR Decay |
| `llrd_decay_factor` | 0.9 | Per-layer LR multiplier |
| `use_cosine_restarts` | `False` | SGDR (Loshchilov & Hutter 2016) |

### EMA & SWA

| Parameter | Default | Description |
|---|---|---|
| `use_ema` | `True` | Exponential Moving Average |
| `ema_decay` | 0.999 | EMA decay factor |
| `ema_update_after_step` | 100 | Start EMA updates after first N steps |
| `ema_update_every` | 10 | Update every N steps |
| `use_swa` | `False` | Stochastic Weight Averaging (Izmailov et al. 2018) |
| `swa_start_epoch` | 80 | SWA start epoch |
| `swa_lr` | 1e-5 | SWA fixed LR |
| `swa_anneal_epochs` | 10 | Annealing epochs |

### Exposure Bias & Curriculum

| Parameter | Default | Description |
|---|---|---|
| `use_scheduled_sampling` | `True` | Scheduled Sampling (Bengio et al. 2015) |
| `ss_start_epoch` | 10 | Scheduled sampling start epoch |
| `ss_decay_rate` | 0.05 | Teacher forcing decay per epoch |
| `min_teacher_forcing` | 0.3 | Minimum teacher forcing |
| `use_curriculum` | `False` | Curriculum Learning (Bengio et al. 2009) |
| `curriculum_strategy` | `"length_based"` | `length_based` / `loss_based` |
| `curriculum_max_len_start` | 64 | Initial max sequence length |
| `curriculum_warmup_epochs` | 20 | Epochs until full data is seen |

### Safety

| Parameter | Default | Description |
|---|---|---|
| `nan_tolerance` | 3 | Consecutive NaN count (revert to checkpoint) |
| `nan_lr_reduction` | 0.5 | LR multiplier after NaN |
| `spike_n_sigma` | 3.0 | Loss spike threshold (N-sigma) |
| `spike_window_size` | 20 | Reference window size (batches) |
| `spike_lr_reduction` | 0.8 | LR multiplier after spike |

### Model (V5 Architecture)

| Parameter | Default | Description |
|---|---|---|
| `embed_dim` | 512 | Embedding dimension |
| `num_heads` | 8 | Attention head count |
| `num_kv_heads` | 2 | GQA KV heads (~75% KV cache reduction) |
| `num_layers` | 8 | Transformer layer count |
| `ffn_dim` | `None` | `None` → auto `embed_dim × 4` |
| `pe_mode` | `"rope"` | Positional encoding type |
| `rope_scaling_type` | `"yarn"` | `"none"` / `"yarn"` / `"linear"` |
| `rope_scaling_factor` | 2.0 | YaRN context extension factor |
| `sliding_window` | 512 | Sliding window attention size |
| `use_rmsnorm` | `True` | RMSNorm |
| `use_swiglu` | `True` | SwiGLU activation |
| `use_kv_cache` | `True` | KV Cache |
| `use_moe` | `False` | Mixture of Experts |
| `num_experts` | 8 | Expert count (when MoE on) |
| `moe_top_k` | 2 | Experts selected per token |

### GPU Batching & Cache

| Parameter | Default | Description |
|---|---|---|
| `use_bucket_batching` | `True` | BucketBatchSampler |
| `num_buckets` | 32 | Number of buckets |
| `use_dynamic_padding` | `True` | DynamicPaddingCollator |
| `data_loader_num_workers` | 4 (Linux) / 0 (Win) | DataLoader worker count |
| `data_loader_pin_memory` | `True` | Pinned RAM |
| `prefetch_factor` | 2 | Prefetch batches per worker |
| `persistent_workers` | `True` | Workers stay alive between epochs |
| `cache_strict_mode` | `True` | Cache required (raise on missing) |
| `cache_verify_integrity` | `True` | SHA-256 checksum verification |

---

## GPU Optimizations

### OOM Protection

```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

When a large contiguous block is not found, the allocator uses fragmented segments → 30–60% less fragmentation.

### AMP (Mixed Precision)

```python
"use_amp": True   # FP16/BF16 forward, FP32 gradients
```

~2x speedup on A100. Matmul precision tuned with `torch.set_float32_matmul_precision("high")`.

### Gradient Accumulation

```python
"batch_size": 64,
"grad_accum_steps": 8,
# Effective batch = 64 × 8 = 512
```

Simulates a large effective batch with lower GPU memory.

---

## Cache System

### Why Cache Is Mandatory

Tokenization is much slower than model training. Tokenizing from raw data on every run would:
- Delay training start by 10–30 minutes
- Make tokenization non-reproducible (vocab may change)
- Prevent data integrity verification

V3 strict mode removes these issues.

### Cache Invalidation

Cache is considered invalid if any component of the cache key changes:

```
1. max_seq_length changed → different seq length
2. vocab_hash changed → BPE vocab/merges updated
3. alignment_format changed → autoregressive format changed
4. data_hash changed → training data changed (file name/size)
5. encode_mode, include_whole_words, include_syllables, include_sep changed
```

Fix:
```bash
python training_system/prepare_cache.py
```

---

## Integration

### V3/V2 Auto-Selection in train.py

```python
try:
    from training_system.v3 import TrainingServiceV3
    _TRAINING_SYSTEM_V3_AVAILABLE = True
except ImportError:
    TrainingServiceV3 = None
    _TRAINING_SYSTEM_V3_AVAILABLE = False

# In main():
if _TRAINING_SYSTEM_V3_AVAILABLE:
    service = TrainingServiceV3(config=effective_cfg)
else:
    service = TrainingService(config=effective_cfg)   # V2 fallback
```

### TrainingManager Selection (inside V3 Service)

```python
try:
    from training_management.v3 import TrainingManager as V3TrainingManager
    _has_v3 = True
except ImportError:
    _has_v3 = False

# On error, V2 TrainingManager is used
from training_management.v2.core.training_manager import TrainingManager as V2TrainingManager
```

### Backward Compatibility with V2

V3 still uses these V2 components:
- `training_management.v2.utils.checkpoint_manager.CheckpointManager`
- `training_management.v2.monitoring.tensorboard_manager.TensorBoardManager`
- `training_management.v2.utils.training_logger.TrainingLogger`
- `training_management.v2.utils.training_scheduler.TrainingScheduler`
- `training_system.v2.core.bpe_validator.BPEValidator`
- `training_system.v2.core.criterion_manager.CriterionManager`
- `training_system.v2.utils.warmup_calculator.calculate_warmup_steps`

---

*Author: Muhammed Yasin Yılmaz | Copyright © 2024 Muhammed Yasin Yılmaz. All Rights Reserved.*
