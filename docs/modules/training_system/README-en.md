# 🎯 Training System — Comprehensive Documentation

**Version:** V-3 (Current)
**Last Updated:** 2026-03-16
**Status:** ✅ Production-Ready | V2 + V3 Dual Stack

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Mandatory Workflow](#mandatory-workflow)
4. [Root Tools](#root-tools)
   - [train.py](#trainpy)
   - [prepare_cache.py](#prepare_cachepy)
   - [config_validator.py](#config_validatorpy)
   - [health_check.py](#health_checkpy)
   - [data_cache.py](#data_cachepy)
5. [V3 System](#v3-system)
   - [TrainingServiceV3](#trainingservicev3)
   - [ConfigManagerV3](#configmanagerv3)
   - [DataCacheV3](#datacachev3)
   - [CevahirDataset](#cevahirdataset)
   - [BucketBatchSampler](#bucketbatchsampler)
   - [DynamicPaddingCollator](#dynamicpaddingcollator)
   - [DataLoader V3](#dataloader-v3)
6. [V2 System](#v2-system)
   - [TrainingService (V2)](#trainingservice-v2)
   - [ConfigManager (V2)](#configmanager-v2)
   - [CriterionManager](#criterionmanager)
   - [BPEValidator](#bpevalidator)
   - [DataPreparator (Deprecated)](#datapreparator-deprecated)
   - [DataLoaderWrapper (V2)](#dataloaderwrapper-v2)
   - [WarmupCalculator](#warmupcalculator)
7. [V2 → V3 Automatic Selection](#v2--v3-automatic-selection)
8. [Configuration Parameters](#configuration-parameters)
9. [GPU Optimizations](#gpu-optimizations)
10. [Cache System Comparison](#cache-system-comparison)

---

## Overview

The Training System is the end-to-end system that runs Cevahir-AI model training. It provides two parallel stacks:

- **V2 Stack:** Stable, production-ready base training system
- **V3 Stack:** GPU-optimized advanced training — Strict Cache, BucketBatchSampler, DynamicPadding

```
train_bpe.py → prepare_cache.py → train.py
     ↓               ↓               ↓
  BPE vocab      V3 Cache        V2 or V3
  creation       preparation     auto selection
```

> ⚠️ **This order is mandatory.** Do not start `train.py` without running `prepare_cache.py` first.

---

## Directory Structure

```
training_system/
├── train.py                    # Main entry point (80+ param TRAIN_CONFIG)
├── train_bpe.py                # BPE tokenizer training
├── prepare_cache.py            # Data preprocessing and cache preparation
├── config_validator.py         # 5-stage config validation
├── health_check.py             # Post-training model quality check
├── data_cache.py               # V2 DataCache (graceful fallback)
│
├── v2/
│   ├── core/
│   │   ├── training_service.py     # V2 TrainingService (936 lines)
│   │   ├── config_manager.py       # V2 ConfigManager (21 params)
│   │   ├── criterion_manager.py   # CriterionManager + EntropyRegCriterion
│   │   ├── bpe_validator.py       # BPE file existence check
│   │   └── data_preparator.py    # DEPRECATED stub
│   ├── data/
│   │   └── data_loader_wrapper.py # SimpleDataset + create_dataloaders()
│   └── utils/
│       └── warmup_calculator.py   # Dynamic warmup step calculation
│
└── v3/
    ├── core/
    │   ├── training_service_v3.py  # V3 TrainingService (725 lines)
    │   └── config_manager_v3.py    # V3 ConfigManager (55+ params, 11 groups)
    └── data/
        ├── cache_v3.py             # DataCacheV3 (strict mode, SHA-256)
        ├── dataset_v3.py           # CevahirDataset (length index)
        ├── sampler_v3.py           # BucketBatchSampler
        ├── collator_v3.py          # DynamicPaddingCollator
        └── dataloader_v3.py        # create_dataloaders_v3() factory
```

---

## Mandatory Workflow

### Step 1 — BPE Training
```bash
python training_system/train_bpe.py
```
Output: `bpe_vocab.json`, `bpe_merges.txt`

### Step 2 — Cache Preparation
```bash
python training_system/prepare_cache.py \
  --data-dir ./data/raw \
  --cache-dir ./training_system/cache \
  --max-seq-length 512
```
Output: `cache/*.pkl` + `cache/*.sha256` + `cache/*.meta.json` (V3 triple layout)

### Step 3 — Training
```bash
python training_system/train.py
```
V3 or V2 is selected automatically based on `use_v3_training_system: True/False` in `TRAIN_CONFIG`.

### Step 4 — Health Check (Optional)
```bash
python training_system/health_check.py \
  --model-path saved_models/cevahir_model.pth
```

---

## Root Tools

### train.py

Main entry point. Holds 80+ parameters in the `TRAIN_CONFIG` dict.

**Key functions:**

| Function | Description |
|---|---|
| `normalize_config()` | `gradient_clip → max_grad_norm`, BPE paths, scheduler_kwargs, TensorBoard defaults |
| `log_env_info()` | PyTorch version, CUDA availability, GPU name |
| `ensure_dirs()` | Creates `saved_models/`, `logs/`, `cache/` |
| `main()` | OOM fix → seed → normalize_config → V3/V2 choice → training |

**OOM fix (critical):**
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```
Reduces GPU memory fragmentation; set before training starts.

**V5 architecture (TRAIN_CONFIG):**
```python
"num_kv_heads": 2,                # GQA — ~75% KV cache reduction
"rope_scaling_type": "yarn",      # YaRN — long context extension
"rope_scaling_factor": 2.0,
"sliding_window": 512,            # Sliding Window Attention
```

---

### prepare_cache.py

Converts training data to autoregressive format and saves it in the V3 cache layout.

**4-step process:**

```
1. Clear cache (skippable with --no-clear-cache)
2. Load TokenizerCore (BPE vocab + merges)
3. Encode data → format_data_func()
4. DataCacheV3.save() → pkl + sha256 + meta.json
```

**`format_data_func()` — Autoregressive format:**
```
inp: [BOS] + encoded_input   (truncate from right, BOS kept)
tgt: encoded_target + [EOS]  (truncate from right, EOS kept)
```

**Deduplication:** Duplicate detection via `hash(inp_tuple) + source_id`; source separation preserved.

**Overlap analysis:** Pre-format and post-format hash collision rate is logged (data leakage check).

**CLI parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--data-dir` | — | Raw data directory |
| `--cache-dir` | `./cache` | Cache output directory |
| `--max-seq-length` | 512 | Maximum token length |
| `--include-whole-words` | False | Whole-word encoding |
| `--include-syllables` | False | Syllable-based encoding |
| `--include-sep` | False | Add SEP token |
| `--no-clear-cache` | False | Do not clear existing cache |

---

### config_validator.py

Validates the `TRAIN_CONFIG` dict before training. 5-stage validation.

**`ValidationResult` dataclass:**
```python
@dataclass
class ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

    def print_report() -> None  # Colored terminal output
```

**5 validation stages:**

| Stage | Check |
|---|---|
| 1. Required fields | Presence of 11 required fields (`vocab_size`, `embed_dim`, `num_heads`, `num_layers`, `batch_size`, `epochs`, `lr`, `device`, `data_dir`, `bpe_vocab_path`, `bpe_merges_path`) |
| 2. Types | 50+ field type checks (`int`, `float`, `bool`, `str`, `list`, `Optional`) |
| 3. Ranges | 30+ value ranges (`lr ∈ [1e-7, 1.0]`, `dropout ∈ [0,1]`, `label_smoothing ∈ [0,0.5]`) |
| 4. Consistency | 8 cross-field rules (see below) |
| 5. Best practice | 6 suggestions (e.g. label_smoothing=0.0 warning) |

**Consistency rules (stage 4):**
- `swa_start_epoch < epochs`
- LLRD + AdamW compatibility
- `tie_weights=True` → `embed_dim == seq_proj_dim`
- `grad_accum_steps > batch_size` → WARNING (unusual setup)
- `use_amp=True + device="cpu"` → WARNING
- `moe_top_k < num_experts`
- `embed_dim % num_heads == 0`
- SAM + Lookahead together → warning

**Usage:**
```python
from training_system.config_validator import ConfigValidator

result = ConfigValidator.validate(config)
result.print_report()

# Raises ValueError on error, logs warnings
ConfigValidator.validate_and_raise(config)
```

---

### health_check.py

Post-training model quality check. Inference test with 8 fixed Turkish/English prompts.

**Metrics:**

| Metric | Production criterion | Description |
|---|---|---|
| `entropy` | > 2.0 | Shannon entropy — low = repetitive output |
| `eos_ratio` | < 0.3 | Early EOS ratio — high = insufficient generation |
| `avg_len` | > 5 | Average response length (tokens) |
| `ttr` | > 0.3 | Type-Token Ratio — lexical diversity |

**Output:** JSON report + terminal summary

**CLI:**
```bash
python health_check.py \
  --model-path saved_models/cevahir_model.pth \
  --config-path training_system/train.py \
  --verbose
```

---

### data_cache.py

V2 DataCache — cache handling with graceful fallback.

| Method | Description |
|---|---|
| `get_cached_data()` | Load from cache; returns `None` if not found (unlike V3!) |
| `save_cached_data()` | Save to cache with atomic write |
| `get_or_process()` | Load if cached, else process and save |
| `get_or_process_corpus()` | BPE training corpus cache |

**Cache key:** `MD5(data_dir + encode_mode + vocab_path + merges_path + max_seq + ...)`

**Difference from V3:**
- V2: `get_cached_data()` → returns `None` (fallback supported)
- V3: `load_strict()` → raises `CacheNotFoundError` (strict mode)

---

## V3 System

### TrainingServiceV3

**File:** `v3/core/training_service_v3.py` (725 lines)

V3 training orchestrator. Provides Strict Cache Mode, source-ID aware split, and advanced GPU batching.

**`__init__()` — 10 steps:**

```
1.  Get BPE paths (from config or default)
2.  Device setup (CUDA/CPU)
3.  data_dir validation
4.  BPEValidator.validate_files()
5.  Load TokenizerCore
6.  Get vocab_size
7.  Create DataCacheV3 (strict_mode=True)
8.  Create ModelManager
9.  Create CriterionManager
10. Create ConfigManagerV3
```

**`train()` — 5 steps:**

```
1. Model init (_initialize_model)
2. Load data from cache (load_data_from_cache — STRICT)
3. Source-ID aware split (_source_id_aware_split)
4. Create V3 DataLoaders (create_dataloaders_v3)
5. Prepare V3 config + start TrainingManager
```

**`_source_id_aware_split()` — No data leakage:**

Ensures all records with the same `source_id` go to the same split:
```
source_id=1 → all examples to train
source_id=2 → all examples to val
```
Fallback: if no source_id, simple random split + WARNING.

**`_initialize_model()` — Checkpoint fallback chain:**
```
last.pth → best.pth → checkpoint_*.pth (newest) → from scratch
```

**`_test_model_inline()` — End-of-epoch test:**
- Generation with `top_k=80`
- Min 5 tokens before EOS
- Logging only (does not affect training)

---

### ConfigManagerV3

**File:** `v3/core/config_manager_v3.py`

Passes 55+ parameters in 11 groups to the V3 TrainingManager.

**11 parameter groups:**

| Group | Main parameters |
|---|---|
| **Basic** | `vocab_size`, `epochs`, `batch_size`, `device`, `seed` |
| **Optimizer** | `lr`, `weight_decay`, `optimizer_type`, `use_adagrad`, `use_adamw8bit` |
| **Scheduler** | `scheduler_type`, `scheduler_kwargs`, `warmup_steps` |
| **Regularization** | `dropout`, `label_smoothing`, `entropy_coeff`, `focal_loss_gamma` |
| **Gradient** | `max_grad_norm`, `grad_accum_steps`, `use_amp`, `agc_clip_val` |
| **Optimization** | `use_sam`, `sam_rho`, `use_lookahead`, `lookahead_k`, `lookahead_alpha` |
| **EMA/SWA** | `use_ema`, `ema_decay`, `use_swa`, `swa_start_epoch`, `swa_lr` |
| **LLRD** | `use_llrd`, `llrd_decay` |
| **Curriculum** | `use_curriculum`, `curriculum_strategy`, `scheduled_sampling_start` |
| **Safety** | `nan_recovery_enabled`, `spike_detection_enabled`, `spike_threshold` |
| **Token** | `pad_id`, `bos_id`, `eos_id`, `unk_id` |

**Internal validation:**
```python
assert 0.0 <= label_smoothing <= 0.5
assert 0.0 <= entropy_coeff <= 1.0
assert 0.0 < ema_decay < 1.0
assert batch_size > 0
assert epochs > 0
```

---

### DataCacheV3

**File:** `v3/data/cache_v3.py`

**Strict cache mode** — If cache is missing, training stops; no graceful fallback.

**Custom exceptions:**
```python
class CacheNotFoundError(Exception): ...  # Cache not found
class CacheIntegrityError(Exception): ... # SHA-256 mismatch
```

**Triple file layout:**
```
cache/
├── {cache_key}.pkl       # Actual data (pickle)
├── {cache_key}.sha256    # SHA-256 checksum
└── {cache_key}.meta.json # Metadata (date, params, size)
```

**Cache key components (MD5, 9 components):**
```
data_dir + encode_mode + vocab_hash + max_seq + include_whole_words
+ include_syllables + include_sep + bpe_vocab_path + bpe_merges_path
```

**`load_strict()` flow:**
```
1. Find matching cache_key
2. No match → CacheNotFoundError (detailed message: list existing caches)
3. Match → verify SHA-256 checksum
4. Bad checksum → CacheIntegrityError
5. Success → return data
```

**`save()` — Atomic write:**
```
1. Write to tmp file
2. Compute SHA-256
3. Rename tmp → final file (atomic)
4. Write .sha256 and .meta.json
```

**`load_for_training()` — High-level API:**
```python
cache.load_for_training(
    tokenizer=tokenizer,
    data_dir=config["data_dir"],
    config=config
)
# Compute vocab_hash + data_hash → cache_key → load_strict()
```

---

### CevahirDataset

**File:** `v3/data/dataset_v3.py`

Subclass of `torch.utils.data.Dataset`. Keeps a length index for BucketBatchSampler.

**Features:**
- **Length index:** True length per sequence (excluding PAD) for BucketBatchSampler
- **Lazy tensor:** If data is a list, convert to tensor in `__getitem__`
- **Source-ID drop:** 3-tuple `(inp, tgt, source_id)` → `(inp, tgt)`
- **`get_length_stats()`:** `min / max / mean / median / p25 / p75 / p90 / p99`

**Length computation (`_compute_lengths`):**
```python
# Count PADs from the end; find last non-PAD position
for i in range(len(inp_list) - 1, -1, -1):
    if inp_list[i] != pad_id:
        real_len = i + 1; break
```

---

### BucketBatchSampler

**File:** `v3/data/sampler_v3.py`

Groups by sequence length to minimize padding waste.

**Reference:** Schwartz et al. 2020, "Right Tool for the Job"

**Algorithm:**
```
1. Sort sequences by length (sorted_indices)
2. Split into num_buckets equal-sized buckets
3. Each epoch: shuffle within buckets → form batches → shuffle batches
4. Epoch-based seed: rng = Random(seed + epoch)
```

**Padding waste comparison:**
```
Static padding:  70–90% of GPU time on PAD token computation
Bucket batching: Padding waste drops to 20–40%
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `lengths` | — | List of true length per example |
| `batch_size` | — | Examples per batch |
| `num_buckets` | 32 | Number of buckets (↑ = less padding, ↓ randomness) |
| `shuffle_buckets` | True | Shuffle batches at epoch start |
| `shuffle_within_bucket` | True | Shuffle within bucket |
| `drop_last` | False | Drop last incomplete batch |
| `seed` | 42 | Base seed for reproducibility |

**Epoch update:**
```python
sampler.set_epoch(epoch)  # Must be called at start of each epoch
```

---

### DynamicPaddingCollator

**File:** `v3/data/collator_v3.py`

Pads each batch to the maximum sequence length within that batch.

**Reference:** Ott et al. 2019, "Scaling Neural Machine Translation" (fairseq)

**V2 comparison:**
```
V2 (custom_collate): torch.stack() → pad to global max_seq_length
V3 (DynamicPaddingCollator): pad to batch-internal max length
```

**GPU effect (example):**
```
Batch {len=10, len=12, len=15}:
  V2: pad to 512 tokens → 97% PAD computation
  V3: pad to 15 tokens  → no redundant PAD computation
```

**`__call__()` flow:**
```python
max_len = max(item[0].size(-1) for item in batch)
if max_seq_length: max_len = min(max_len, max_seq_length)

inputs  = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
targets = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

for i, (inp, tgt) in enumerate(batch):
    inp_len = min(inp.size(-1), max_len)
    inputs[i, :inp_len] = inp[:inp_len]
    # ... targets same
```

**Backward compatibility:**
```python
# V2 static collate (for pre-padded data):
collate_fn = create_static_collate(pad_id=0)
```

---

### DataLoader V3

**File:** `v3/data/dataloader_v3.py`

**6 GPU optimization layers:**

| # | Optimization | Effect |
|---|---|---|
| 1 | `pin_memory=True` | Pinned RAM → fast PCIe DMA transfer |
| 2 | `non_blocking=True` | Async GPU transfer (overlaps with compute) |
| 3 | `prefetch_factor=2` | Preload N batches |
| 4 | `persistent_workers=True` | Worker processes stay alive across epochs |
| 5 | `BucketBatchSampler` | Minimize padding waste |
| 6 | `DynamicPaddingCollator` | Per-batch dynamic pad length |

**`create_dataloaders_v3()` — Train vs Val:**

| | Train | Val |
|---|---|---|
| Sampler | `BucketBatchSampler` (shuffle) | `SequentialSampler` |
| Bucket | `use_bucket_batching=True` | `False` |
| Shuffle | `True` | `False` |

**Windows note:**
```python
if num_workers > 0 and os.name == "nt":
    logger.warning("num_workers>0 on Windows may be problematic...")
```

---

## V2 System

### TrainingService (V2)

**File:** `v2/core/training_service.py` (936 lines)

**`__init__()` steps:**
- BPEValidator → TokenizerCore → vocab_size → DataCache (optional) → CriterionManager → ConfigManager → DataPreparator

**`train()` main flow:**
```
1. Model init (_initialize_model)
2. Load data from cache (graceful fallback: returns None)
3. Random split (ignores source_id)
4. V2 DataLoader (create_dataloaders)
5. Prepare V2 config + TrainingManager
```

**`prepare_from_cache()` — Next-token alignment check:**
```python
# inp[i+1] == tgt[i] check — autoregressive format validation
_validate_alignment(data)
```

**`_test_model_after_epoch()` — End-of-epoch test:**
- Does not use Cevahir.generate() (standalone test)
- Model called directly

---

### ConfigManager (V2)

**File:** `v2/core/config_manager.py` (109 lines)

Converts `TRAIN_CONFIG` to V2 TrainingManager format.

**`prepare_training_config()` — 21 parameters:**

Special token ID extraction:
```python
pad_id = tokenizer.special_tokens["<PAD>"]
bos_id = tokenizer.special_tokens["<BOS>"]
eos_id = tokenizer.special_tokens["<EOS>"]
unk_id = tokenizer.special_tokens["<UNK>"]
```

**Difference from V3 ConfigManagerV3:**

| | V2 ConfigManager | V3 ConfigManagerV3 |
|---|---|---|
| Parameter count | ~21 | 55+ |
| Groups | None (flat dict) | 11 groups |
| Entropy coeff | ✗ | ✓ |
| SAM/Lookahead | ✗ | ✓ |
| EMA/SWA | ✗ | ✓ |
| LLRD | ✗ | ✓ |
| Curriculum | ✗ | ✓ |
| NaN recovery | ✗ | ✓ |

---

### CriterionManager

**File:** `v2/core/criterion_manager.py`

Creates and configures the loss function.

**`EntropyRegCriterion(nn.Module)` — Entropy regularization:**

> Reference: Pereyra et al. 2017, "Regularizing Neural Networks by Penalizing Confident Output Distributions"

```python
loss = CE_loss + entropy_coeff * (-mean_entropy)
# High confidence → negative entropy → penalize
# Encourages less overconfident outputs
```

**Memory-safe chunk computation:**
```python
_CHUNK = 512  # Max 512 examples per forward
# Avoids OOM risk with large vocab_size
```

**`LossComputation` compatibility properties:**
```python
@property
def weight(self) -> Optional[Tensor]: ...
@property
def label_smoothing(self) -> float: ...
@property
def ignore_index(self) -> int: ...
```

**`CriterionManager.create_criterion()` flow:**
```
1. Build vocab weight tensor (EOS special weight)
2. CrossEntropyLoss(label_smoothing, ignore_index=pad_id)
3. If entropy_coeff > 0 → wrap with EntropyRegCriterion
4. Return
```

---

### BPEValidator

**File:** `v2/core/bpe_validator.py` (99 lines)

Checks existence and content of BPE vocab and merges files.

**Strategy:** Fixed vocab (vocab is only created via `train_bpe.py`, no auto-creation)

```python
validator = BPEValidator(
    vocab_path="bpe_vocab.json",
    merges_path="bpe_merges.txt"
)
validator.validate_files()
# Missing or empty → RuntimeError (run train_bpe.py message)
```

---

### DataPreparator (Deprecated)

**File:** `v2/core/data_preparator.py` (62 lines)

> ⛔ **DEPRECATED** — No longer used.

```python
# Warning when imported:
DeprecationWarning: "DataPreparator is deprecated.
Use TrainingService.prepare_from_cache()."
```

Functionality moved to `training_service.py` and `prepare_cache.py`.

---

### DataLoaderWrapper (V2)

**File:** `v2/data/data_loader_wrapper.py` (130 lines)

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
# All sequences assumed same length (pre-padded)
```

**`create_dataloaders()`:**
```python
DataLoader(
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=(prefetch_factor if num_workers > 0 else None),
)
```

**Difference from V3:**

| | V2 DataLoader | V3 DataLoader |
|---|---|---|
| Sampler | RandomSampler / SequentialSampler | BucketBatchSampler |
| Collate | `torch.stack()` (static) | DynamicPaddingCollator (dynamic) |
| Padding | Global max_seq_length | Batch-internal max length |
| Padding waste | 70–90% | 20–40% |

---

### WarmupCalculator

**File:** `v2/utils/warmup_calculator.py` (68 lines)

```python
def calculate_warmup_steps(
    batches_per_epoch: int,
    grad_accum_steps: int = 1,
    warmup_epochs: float = 1.0,
) -> int:
    steps = (batches_per_epoch // grad_accum_steps) * warmup_epochs
    return max(1, int(steps))  # Minimum 1 warmup step
```

Called inside V2 TrainingService in the `train()` flow.

---

## V2 → V3 Automatic Selection

In `train.py → main()`:

```python
if config.get("use_v3_training_system", False):
    from training_system.v3.core.training_service_v3 import TrainingServiceV3
    service = TrainingServiceV3(config)
else:
    from training_system.v2.core.training_service import TrainingService
    service = TrainingService(config)
```

**TRAIN_CONFIG for V3:**
```python
"use_v3_training_system": True,
"use_bucket_batching": True,
"num_buckets": 32,
"use_dynamic_padding": True,
"prefetch_factor": 2,
"persistent_workers": True,
```

**When to use V3?**

| Situation | Recommendation |
|---|---|
| GPU training | V3 (pin_memory + async transfer) |
| Variable-length sequences | V3 (BucketBatchSampler) |
| Large dataset (>100K examples) | V3 (DynamicPadding) |
| Cache integrity critical | V3 (SHA-256 strict mode) |
| Source-based data split | V3 (source_id_aware_split) |
| Fast CPU prototype | V2 (simpler) |

---

## Configuration Parameters

### Critical V5 Architecture Parameters

```python
TRAIN_CONFIG = {
    # Basic architecture
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 8,
    "ffn_dim": 2048,

    # GQA (Grouped Query Attention)
    "num_kv_heads": 2,           # 4 groups → ~75% KV cache reduction

    # RoPE + YaRN
    "rope_scaling_type": "yarn",
    "rope_scaling_factor": 2.0,

    # Sliding Window Attention
    "sliding_window": 512,

    # Optimizer
    "use_adamw8bit": True,       # bitsandbytes — ~8 GB VRAM saving

    # V3 Training System
    "use_v3_training_system": True,
    "use_bucket_batching": True,
    "num_buckets": 32,
}
```

### Advanced Training Parameters (V3 ConfigManagerV3)

```python
# Entropy Regularization (Pereyra et al. 2017)
"entropy_coeff": 0.01,          # 0 = disabled

# Focal Loss (Lin et al. 2017)
"focal_loss_gamma": 2.0,        # 0 = standard CrossEntropy

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

## GPU Optimizations

### Memory Management

```python
# At start of train.py — reduce OOM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 8-bit AdamW with bitsandbytes
"use_adamw8bit": True  # ~8 GB VRAM saving
```

### DataLoader GPU Optimizations

```python
DataLoader(
    pin_memory=True,          # RAM → GPU fast DMA transfer
    num_workers=4,            # Parallel data loading
    prefetch_factor=2,        # Preload 2 batches
    persistent_workers=True, # Zero process overhead
)
```

### Gradient Optimizations

```python
"use_amp": True,              # Mixed precision (FP16/BF16)
"grad_accum_steps": 4,       # Effective batch_size = batch * 4
"max_grad_norm": 1.0,         # Gradient clipping
"agc_clip_val": 0.01,         # Adaptive Gradient Clipping
```

---

## Cache System Comparison

| Feature | V2 DataCache | V3 DataCacheV3 |
|---|---|---|
| On cache miss | Returns `None` | Raises `CacheNotFoundError` |
| Integrity check | None | SHA-256 checksum |
| Metadata | None | `.meta.json` (date, params, size) |
| Atomic write | No (direct write) | tmp → rename |
| Cache key components | ~5 components | 9 components |
| Fallback allowed | Yes | No (strict) |
| Use case | Prototype / dev | Production training |

---

*Author: Muhammed Yasin Yılmaz — Cevahir-AI Project*
*© 2024 Muhammed Yasin Yılmaz. All Rights Reserved.*
