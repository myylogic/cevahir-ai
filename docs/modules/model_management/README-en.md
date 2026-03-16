# Model Management Module — Documentation

**Version:** 4.1.0
**Last Updated:** 2026-03-16
**Status:** Production-Ready
**Root Directory:** `model_management/`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture and File Structure](#architecture-and-file-structure)
3. [ModelManager](#modelmanager)
4. [ModelInitializer](#modelinitializer)
5. [ModelSaver](#modelsaver)
6. [ModelLoader](#modelloader)
7. [ModelUpdater](#modelupdater)
8. [ModelProfiler](#modelprofiler)
9. [ModelHealthMonitor](#modelhealthmonitor)
10. [Config Schema](#config-schema)
11. [Exception Hierarchy](#exception-hierarchy)
12. [Usage Examples](#usage-examples)
13. [Training Integration](#training-integration)

---

## Overview

The `model_management` module manages the Cevahir-AI model lifecycle: creation, training component setup, save/load, profiling, and health monitoring. It applies SOLID principles; each file has a single, focused responsibility.

```
train.py / cevahir.py
        |
        v
   ModelManager          <- High-level API (facade)
   |-- ModelInitializer  <- Model + Optimizer + Scheduler creation
   |-- ModelSaver        <- Atomic checkpoint save + SHA-256
   |-- ModelLoader       <- Safe checkpoint load + version check
   |-- ModelUpdater      <- Freeze/Unfreeze, LR update
   |-- ModelProfiler     <- Parameter count, memory, FLOP, timing
   +-- ModelHealthMonitor<- Gradient, weight, attention health
```

---

## Architecture and File Structure

```
model_management/
|-- __init__.py              # Public API (all classes re-exported)
|-- model_manager.py         # ModelManager — central facade
|-- model_initializer.py     # ModelInitializer — model/opt/sched creation
|-- model_saver.py           # ModelSaver — checkpoint saving
|-- model_loader.py          # ModelLoader — checkpoint loading
|-- model_updater.py         # ModelUpdater — parameter updates
|-- profiler.py              # ModelProfiler — profiling tools
|-- health_monitor.py        # ModelHealthMonitor — health monitoring
|-- config_schema.py         # Typed config dataclasses
|-- exceptions.py            # Exception hierarchy
+-- test/
    |-- test_model_manager.py
    +-- test_model_manager_comprehensive.py
```

### Layer Architecture (inner to outer)

| Layer | File | Responsibility |
|-------|------|-----------------|
| 1 | `exceptions.py` | Error hierarchy |
| 2 | `config_schema.py` | Type-safe configuration |
| 3 | `profiler.py` | Model profiling tools |
| 4 | `health_monitor.py` | Health monitoring |
| 5 | `model_initializer.py` | Model/opt/sched creation |
| 6 | `model_saver.py` | Checkpoint saving |
| 7 | `model_loader.py` | Checkpoint loading |
| 8 | `model_updater.py` | Parameter updates |
| 9 | `model_manager.py` | High-level facade |

---

## ModelManager

**File:** `model_management/model_manager.py`

Central management class that combines all subcomponents. Entry point for training and inference.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Dict[str, Any]` | — | Model and training configuration |
| `model_class` | `Type[nn.Module]` | `CevahirNeuralNetwork` | Model class |
| `device` | `str\|torch.device\|None` | from config | Device |
| `initializer` | class | `ModelInitializer` | DI: builder |
| `saver` | class | `ModelSaver` | DI: saver |
| `updater` | class | `ModelUpdater` | DI: updater |
| `tokenizer` | Any | `None` | Multimodal: tokenizer |
| `audio_processor` | Any | `None` | Multimodal: audio |
| `vision_processor` | Any | `None` | Multimodal: vision |

### Main Methods

| Method | Description |
|--------|-------------|
| `build_model()` | Creates model, moves to device, logs profile report |
| `build_optimizer()` | Creates optimizer |
| `build_criterion()` | Creates loss function |
| `build_scheduler()` | Creates LR scheduler |
| `initialize(...)` | Initializes all components in one call |
| `forward(input_ids, ...)` | Model forward pass; includes OOM recovery |
| `generate(input_ids, ...)` | Autoregressive token generation |
| `predict(input_ids, ...)` | Top-k prediction, logit/softmax option |
| `save(epoch, ...)` | Saves checkpoint |
| `load(path, ...)` | Loads checkpoint |
| `train_mode()` | `model.train()` + dropout on |
| `eval_mode()` | `model.eval()` + dropout off |
| `health_check()` | Returns `HealthReport` |
| `profile()` | Returns profile report |
| `setup_tensorboard(log_dir)` | Starts TensorBoard writer |

### Device Selection (Priority Order)

```
1. Explicitly set via __init__(device=...)
2. config["device"] == "cuda" and CUDA available -> cuda
3. config["device"] == "mps" and MPS available -> mps
4. Auto-detect CUDA -> cuda (if present)
5. Fallback -> cpu
```

### TensorBoard Integration

```python
mm = ModelManager(config)
mm.setup_tensorboard(log_dir="runs/cevahir_v6")

# Metrics written automatically on each model forward:
# - train/loss, train/perplexity
# - grad_norm, lr
# - attn_entropy (if model stores this)
```

### Automatic Profile Report

After `build_model()` is called, `ModelProfiler.full_report()` runs automatically:

```
[Profiler] == Model Report ==
  Parameters : ParamStats(total=85.23M, trainable=85.23M, frozen=0, trainable_mem=325.2 MB)
  Model size : 325.2 MB
  Memory     : MemorySnapshot(cuda) alloc=1.32 GB / total=8.00 GB (16.5%)
  FLOP (T=512) : FlopEstimate(total=42.3 GFLOPs, attn=18.1, ffn=24.1, seq=512, batch=1)
```

---

## ModelInitializer

**File:** `model_management/model_initializer.py`

Static methods to create model instance, optimizer, loss function, and LR scheduler. All methods are `@staticmethod`; no instance needed.

### `build_model()`

```python
model = ModelInitializer.build_model(
    model_class=CevahirNeuralNetwork,
    config=config_dict,
    device=torch.device("cuda"),
    compile_model=True,    # torch.compile (PyTorch 2.0+)
)
```

**Steps in order:**
1. `_apply_seed(config)` — deterministic init (optional)
2. `_resolve_device(config)` — device selection
3. `_filter_kwargs_for_ctor(model_class, config)` — filter config to model signature
4. `model_class(**ctor_kwargs).to(device)` — create model
5. `torch.compile(model, ...)` — compile (optional, `torch_compile=True`)
6. `gradient_checkpointing_enable()` — (optional)
7. `_apply_quantization(...)` — INT8/INT4 (optional)
8. `_wrap_distributed(...)` — DDP/FSDP (optional)

**Safe signature filtering:** Unknown keys in config are filtered automatically; model `__init__` signature is read via `inspect.signature`.

### `initialize_optimizer()`

**Supported optimizers:**

| Name | Description |
|------|-------------|
| `adamw` | PyTorch AdamW (optional fused) |
| `adamw8bit` / `adamw_8bit` | bitsandbytes 8-bit AdamW — stores optimizer m/v states in uint8, ~75% memory reduction |
| `adam` | Standard Adam |
| `radam` | Rectified Adam |
| `rmsprop` | RMSProp |
| `sgd` | Stochastic Gradient Descent |

**Parameter groups (3 groups):**

```
Group 1 — Embedding: lr = base_lr x embedding_lr_scale (default 1.0)
Group 2 — Decay:     lr = base_lr, weight_decay = config value
Group 3 — No-decay:  lr = base_lr, weight_decay = 0.0
         (bias, norm, layernorm, bn, etc.)
```

> **Note:** Default `embedding_lr_scale` is 1.0 (same as base_lr). The old value 0.1 weakened EOS/rare token learning.

**AdamW8bit (bitsandbytes):**

```python
# Dettmers et al. 2022 -- 8-bit optimizer
# Install: pip install bitsandbytes
# Falls back to standard AdamW if not available
optimizer: str = "adamw8bit"
```

### `initialize_criterion()`

| Name | Class |
|------|-------|
| `cross_entropy` / `ce` | `nn.CrossEntropyLoss` (label_smoothing, ignore_index supported) |
| `bce_with_logits` / `bce` | `nn.BCEWithLogitsLoss` |
| `mse` | `nn.MSELoss` |
| `smooth_l1` / `huber` | `nn.SmoothL1Loss` |

```python
# To exclude PAD tokens from loss:
ignore_index: 0   # 0=PAD ignore, -100=count all (default -100)
```

### `initialize_scheduler()`

| Type | Config key | Description |
|------|------------|-------------|
| `reduce_on_plateau` | `scheduler_type: "rop"` | Reduce LR when validation loss plateaus |
| `cosine` | `scheduler_type: "cosine"` | Cosine decay |
| `cosine_warm_restarts` | `scheduler_type: "cawr"` | Cosine with warm restarts |
| `step` | `scheduler_type: "step"` | LR multiplier at fixed steps |
| `exponential` | `scheduler_type: "explr"` | Exponential decay |
| `onecycle` | `scheduler_type: "onecycle"` | OneCycleLR (steps_per_epoch and epochs required) |
| `none` | `scheduler_type: "none"` | No scheduler |

### `build_training_components()` — Single Call

```python
optimizer, criterion, scheduler = ModelInitializer.build_training_components(model, config)
```

### Quantization Support

| Type | Description | Requirement |
|------|-------------|-------------|
| `int8` | LLM.int8() — threshold=6.0, ~50% VRAM reduction | `bitsandbytes` |
| `int4` | NF4 + double quantization, ~75% VRAM reduction | `bitsandbytes` |

### Distributed Training

| Strategy | Description |
|----------|-------------|
| `ddp` | DistributedDataParallel — synchronizes gradients |
| `fsdp` | FullyShardedDataParallel — shards model parameters across GPUs |

```python
config = {
    "distributed_strategy": "ddp",
    "distributed_backend": "nccl",
    "local_rank": 0,
}
# Started with torchrun
```

---

## ModelSaver

**File:** `model_management/model_saver.py`
**Checkpoint Version:** `4.1` / Format: `2`

Safe checkpoint handling with atomic write (tmp → `os.replace`) and SHA-256 integrity verification.

### `save_checkpoint()` — Main API

```python
path = ModelSaver.save_checkpoint(
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    config=config,
    metadata={"val_loss": 2.34, "train_loss": 1.87},
    save_dir="saved_models/checkpoints",
    filename_template="checkpoint_ep{epoch:04d}.pth",
    create_latest_marker=True,  # creates latest.txt
    keep_last_n=5,              # delete all but last 5 checkpoints
    prefix_for_prune="checkpoint_",
)
```

**Saved checkpoint structure:**

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch": epoch,
    "config": config,
    "metadata": {
        # User metadata (val_loss, train_loss, etc.)
        "cevahir_version": "4.1",
        "checkpoint_format": 2,
        "saved_at": "2026-03-16T10:30:00Z",
        "total_params": 85234567,
        "trainable_params": 85234567,
        "sha256": "abc123..."
    }
}
```

**SHA-256 sidecar file:** `checkpoint_ep0010.pth.sha256` — for fast external integrity verification.

### Smart Checkpoint Pruning

```
prefer_best_val_loss=True (default):
  1. val_loss in each checkpoint metadata is checked
  2. Checkpoint with best val_loss is ALWAYS kept
  3. Remaining slots filled with newest checkpoints
```

### Atomic Write

```
1. BytesIO -> serialize
2. tempfile.mkstemp() -> write to temp file
3. os.replace(tmp, target) -> atomic move
```

On CUDA error, tensors are moved to CPU and retried.

### Other Methods

| Method | Description |
|--------|-------------|
| `save_weights_only(model, ...)` | Saves only state_dict |
| `save_full_model(model, ...)` | Full model (pickle) — generally not recommended |
| `save_additional_info(info, ...)` | Saves extra info as JSON |
| `save_model(...)` | Legacy API (backward compatibility) |

---

## ModelLoader

**File:** `model_management/model_loader.py`

### `load_model()` — Single Model Load

```python
model = ModelLoader.load_model(
    model_class=CevahirNeuralNetwork,
    model_path="saved_models/checkpoints/checkpoint_ep0010.pth",
    device="cuda",
    config=config,
    strict=True,
    weights_only=None,
)
```

**Load steps:**
1. `_verify_sha256(path)` — SHA-256 sidecar check (if present)
2. `_check_version_compatibility(ckpt)` — format version compatibility
3. `_extract_state_dicts(ckpt)` — separate model/opt/sch state dicts
4. `model_class(**ctor_kwargs).to(device)` — create model instance
5. Vocab size check (`embedding.weight` shape)
6. `model.load_state_dict(model_sd, strict=strict)` — load weights
7. Missing/unexpected key warnings

### `load_all()` — Everything in One Call

```python
model, opt_sd, sch_sd, meta = ModelLoader.load_all(
    model_class=CevahirNeuralNetwork,
    ckpt_path="checkpoint_ep0010.pth",
    device="cuda",
    config=config,
)
# meta: {"epoch": 10, "config": {...}}
```

### Checkpoint Format Support

| Format | Description |
|--------|-------------|
| `{"model_state_dict": ..., "optimizer_state_dict": ..., ...}` | Full checkpoint (recommended) |
| `{"state_dict": ...}` | Framework-compatible |
| `{str: Tensor, ...}` | Flat state_dict |

### Version Compatibility

```
Format 1 -> Format 2: Backward compatible
Format 3+: Raises CheckpointVersionError
```

---

## ModelUpdater

**File:** `model_management/model_updater.py`

Static methods to update model parameters and training components at runtime.

### Basic Methods

| Method | Description |
|--------|-------------|
| `freeze_layers(model, layers)` | Freezes specified layers (`requires_grad=False`) |
| `unfreeze_layers(model, layers)` | Unfreezes frozen layers |
| `freeze_all_except(model, patterns)` | Freezes all layers except given patterns |
| `update_learning_rate(optimizer, lr)` | Updates LR in all param groups |
| `update_weight_decay(optimizer, wd)` | Updates weight decay |
| `step_scheduler(scheduler, metric)` | Steps scheduler (metric required for plateau) |
| `apply_weight_noise(model, std)` | Adds Gaussian noise to weights |
| `reset_parameters(model, layers)` | Resets weights of selected layers |

### Pattern-Based Freezing

```python
# Glob/regex pattern support
ModelUpdater.freeze_all_except(model, patterns=["layers.7.*", "output_layer"])
# Only last layer and output projection remain trainable
```

---

## ModelProfiler

**File:** `model_management/profiler.py`

Static tools for model size, parameter count, FLOP estimate, and timing. No instance required.

### Data Classes

| Class | Fields |
|-------|--------|
| `ParamStats` | `total`, `trainable`, `frozen`, `trainable_mb`, `by_layer` |
| `MemorySnapshot` | `allocated_mb`, `reserved_mb`, `free_mb`, `total_mb`, `device` |
| `FlopEstimate` | `total_flops`, `attention_flops`, `ffn_flops`, `embedding_flops`, `gflops` |
| `TimingResult` | `mean_ms`, `std_ms`, `min_ms`, `max_ms`, `tokens_per_second` |

### `count_parameters()`

```python
stats = ModelProfiler.count_parameters(model)
print(stats)
# ParamStats(total=85.23M, trainable=85.23M, frozen=0, trainable_mem=325.2 MB)

# Per-layer breakdown (top 5):
#   embedding:    30.72M params
#   layers:       54.01M params
#   output_layer:  0.50M params
```

### `memory_snapshot()`

```python
mem = ModelProfiler.memory_snapshot("cuda")
print(mem)
# MemorySnapshot(cuda) alloc=3.72 GB / total=8.00 GB (46.5%)

print(f"Usage: {mem.utilization_pct:.1f}%")
print(f"Free: {mem.free_mb:.0f} MB")
```

### `estimate_flops()`

Theoretical FLOP estimate per Kaplan et al. 2020 / PaLM paper:

```
Attention: 4 x B x L x T x D^2   (QKV + Output projection)
           2 x B x L x H x T^2   (attention score)
FFN:       2 x B x L x T x D x F x 2
B=batch, L=num_layers, T=seq_len, D=embed_dim, H=num_heads, F=ffn_dim
```

```python
flops = ModelProfiler.estimate_flops(model, seq_len=512, batch_size=4)
print(flops)
# FlopEstimate(total=42.3 GFLOPs, attn=18.1, ffn=24.1, seq=512, batch=4)
```

### `benchmark_forward()`

```python
sample = torch.randint(0, 32000, (1, 512)).cuda()
timing = ModelProfiler.benchmark_forward(model, sample, n_warmup=3, n_runs=20)
print(timing)
# TimingResult(mean=23.45ms +-0.87, tok/s=21834, runs=20)
```

Uses `torch.cuda.Event` for precise GPU timing.

### `profile_context()` — torch.profiler Integration

```python
with ModelProfiler.profile_context(model, output_path="./profiler_trace") as prof:
    logits, _ = model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# Saved as Chrome trace (viewable in TensorBoard)
```

### `full_report()` — Summary Report

```python
report = ModelProfiler.full_report(model, seq_len=512, run_timing=True)
# Returns: {"params": ParamStats, "memory": MemorySnapshot,
#           "flops": FlopEstimate, "timing": TimingResult, "size_mb": float}
```

---

## ModelHealthMonitor

**File:** `model_management/health_monitor.py`

Detects gradient flow, weight distribution, and attention entropy pathologies.

### Detected Pathologies

| Pathology | Threshold | Level |
|-----------|-----------|-------|
| NaN gradient | Any | CRITICAL |
| Inf gradient | Any | CRITICAL |
| Gradient vanishing | `grad_norm < 1e-8` | INFO |
| Gradient exploding | `grad_norm > 1e4` | WARNING |
| NaN weight | Any | CRITICAL |
| Dead weight | `std < 1e-9` | INFO |
| Weight explosion | `abs_max > 1e3` | WARNING |
| Attention collapse | `entropy < 0.05` | WARNING |
| Attention uniform | `entropy > 0.99` | INFO |

### Data Classes

| Class | Description |
|-------|-------------|
| `GradientHealth` | Gradient NaN/Inf/vanish/explode info |
| `WeightHealth` | Weight distribution, dead/exploding layers |
| `AttentionHealth` | Attention entropy, collapse/uniform layers |
| `HealthReport` | Combined report of all three |

### Severity Levels

```
OK       -> All normal
INFO     -> Vanishing gradient or dead layer (monitor)
WARNING  -> Exploding gradient or attention collapse (intervene)
CRITICAL -> NaN/Inf (training corrupted, stop)
```

### `full_health_check()` — Combined Check

```python
report = ModelHealthMonitor.full_health_check(
    model,
    sample_input=sample,
    check_gradients=True,
    check_weights=True,
    check_attention=True,
    raise_on_critical=True,     # Raises HealthCheckError on CRITICAL
)

if not report.is_healthy:
    print(report.summary())
```

**Example output:**

```
============================================================
  ⚠ MODEL HEALTH REPORT -- WARNING
============================================================
GradientHealth [WARNING]
  ⚠  Exploding gradients: ['layers.3.ffn.fc1.weight']
  norm: max=1.23e+05, min=1.23e-06, mean=3.45e+00

WeightHealth [OK]
  global: mean=0.0012, std=0.0345, max_abs=0.8921

AttentionHealth [OK]
  entropy: mean=0.423, min=0.312, max=0.687
============================================================
```

### `quick_gradient_check()` — Fast Check Per Batch

```python
# After backward(), every batch:
is_safe, msg = ModelHealthMonitor.quick_gradient_check(model)
if not is_safe:
    logger.warning(f"NaN/Inf gradient -- skipping batch: {msg}")
    optimizer.zero_grad()
    continue
```

### `log_gradient_norms()` — TensorBoard Integration

```python
norms = ModelHealthMonitor.log_gradient_norms(
    model, step=global_step, tb_writer=writer, top_n=10
)
# Writes top 10 gradient norms to TensorBoard
```

### Attention Entropy Monitoring

If the model holds the `_last_attn_entropy` attribute (CevahirNeuralNetwork does), it is read automatically after each forward:

```python
# Computed and stored during model forward():
self._last_attn_entropy = normalized_entropy   # [0, 1]

# HealthMonitor reads it:
health = ModelHealthMonitor.check_attention_entropy(model)
# Collapse (< 0.05), Normal, Uniform (> 0.99)
```

---

## Config Schema

**File:** `model_management/config_schema.py`

Type-safe, validatable configuration schemas instead of plain `Dict[str, Any]`.

### Classes

#### `ModelArchConfig`

```python
arch = ModelArchConfig(
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    vocab_size=32000,
    ffn_dim=None,
    use_swiglu=True,
    use_rmsnorm=True,
    num_kv_heads=2,
    pe_mode="rope",
    rope_scaling_type="yarn",
    rope_scaling_factor=4.0,
    use_moe=False,
    quantization_type="none",
    tie_weights=True,
)
arch.validate()

print(arch.head_dim)                  # 64
print(arch.effective_ffn_dim)         # 2048
print(arch.parameter_count_estimate)  # ~85M
```

**Validated rules:**
- `embed_dim % num_heads == 0`
- `num_heads % num_kv_heads == 0` (GQA compatibility)
- `rope_scaling_factor >= 1.0`
- `moe_top_k <= num_experts`
- If `tie_weights=True` then `seq_proj_dim == embed_dim`

#### `TrainingConfig`

```python
training = TrainingConfig(
    learning_rate=2e-4,
    batch_size=72,
    grad_accum_steps=4,
    optimizer="adamw8bit",
    scheduler_type="reduce_on_plateau",
    use_amp=True,
    use_gradient_checkpointing=True,
    use_ema=True,
    ema_decay=0.999,
)
print(training.effective_batch_size)  # 288 (72 x 4)
```

#### `CheckpointConfig`

```python
ckpt_cfg = CheckpointConfig(
    save_dir="saved_models/checkpoints",
    keep_last_n=5,
    save_every_n_epochs=10,
    enable_sha256=True,
)
```

#### `DistributedConfig`

```python
dist_cfg = DistributedConfig(
    enabled=True,
    backend="nccl",
    strategy="ddp",
    world_size=4,
)
```

#### `QuantConfig`

```python
quant_cfg = QuantConfig(
    quant_type="int4",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)
```

#### `CevahirConfig` — Unified Configuration

```python
# From flat dict in train.py:
cfg = CevahirConfig.from_flat_dict(TRAIN_CONFIG)
cfg.validate_all()

print(cfg.arch.embed_dim)
print(cfg.training.effective_batch_size)

d = cfg.to_dict()
```

---

## Exception Hierarchy

**File:** `model_management/exceptions.py`

```
CevahirModelError
|-- ModelNotInitializedError     -> Use before initialize() called
|-- ModelBuildError              -> Model/optimizer/scheduler creation error
|    +-- QuantizationError       -> INT8/INT4 quantization failure
|-- CheckpointError              -> Checkpoint I/O base error
|    |-- CheckpointNotFoundError -> File not found
|    |-- CheckpointCorruptError  -> SHA-256 mismatch / corrupt format
|    +-- CheckpointVersionError -> Version incompatibility
|-- ForwardError                 -> Model forward pass error
|    +-- OOMRecoveryError        -> CUDA OOM -> recovery failed
|-- DeviceError                  -> Device selection / transfer error
|    +-- DeviceMismatchError     -> Tensor device mismatch
|-- ShapeError                   -> Tensor shape mismatch
|    +-- VocabSizeMismatchError  -> vocab_size checkpoint vs model
|-- DistributedSetupError       -> DDP/FSDP setup error
+-- HealthCheckError             -> Model health check failed
```

### Usage

```python
from model_management import (
    CevahirModelError,
    CheckpointNotFoundError,
    OOMRecoveryError,
    VocabSizeMismatchError,
)

try:
    model = ModelLoader.load_model(CevahirNeuralNetwork, path, config=config)
except CheckpointNotFoundError as e:
    print(f"Checkpoint not found: {e.path}")
except VocabSizeMismatchError as e:
    print(f"Vocab mismatch: model={e.model_vocab}, ckpt={e.checkpoint_vocab}")
except CevahirModelError as e:
    print(f"General model error: {e.message} | {e.context}")
```

---

## Usage Examples

### Quick Start

```python
from model_management import ModelManager

config = {
    "vocab_size": 32000,
    "embed_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 2,
    "num_layers": 8,
    "ffn_dim": None,
    "use_swiglu": True,
    "use_pytorch_sdpa": True,
    "logit_soft_cap": 30.0,
    "dropout": 0.1,
    "max_seq_length": 2048,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "optimizer": "adamw8bit",
    "scheduler_type": "reduce_on_plateau",
    "criterion": "cross_entropy",
    "ignore_index": 0,
    "device": "cuda",
}

mm = ModelManager(config)
mm.initialize(
    build_model=True,
    build_optimizer=True,
    build_criterion=True,
    build_scheduler=True,
)
```

### Training Loop Integration

```python
mm.train_mode()

for batch in dataloader:
    input_ids = batch["input_ids"].to(mm.device)
    labels = batch["labels"].to(mm.device)

    logits, _ = mm.forward(input_ids)
    loss = mm.criterion(
        logits[:, :-1].reshape(-1, vocab_size),
        labels[:, 1:].reshape(-1)
    )
    (loss / grad_accum_steps).backward()

    # NaN check
    is_safe, msg = ModelHealthMonitor.quick_gradient_check(mm.model)
    if not is_safe:
        mm.optimizer.zero_grad()
        continue

    torch.nn.utils.clip_grad_norm_(mm.model.parameters(), 1.0)
    mm.optimizer.step()
    mm.optimizer.zero_grad()

# End-of-epoch checkpoint
mm.save(
    epoch=epoch,
    metadata={"val_loss": val_loss, "train_loss": avg_loss},
    keep_last_n=5,
)
```

### Save / Load Checkpoint

```python
# Save
path = ModelSaver.save_checkpoint(
    model, optimizer=optimizer, epoch=10,
    config=config, metadata={"val_loss": 2.34},
    save_dir="saved_models", keep_last_n=5,
)

# Load
model, opt_sd, sch_sd, meta = ModelLoader.load_all(
    CevahirNeuralNetwork, path, device="cuda", config=config,
)
optimizer.load_state_dict(opt_sd)
print(f"Loaded epoch {meta['epoch']}")
```

### Profile + Health Check

```python
from model_management import ModelProfiler, ModelHealthMonitor

# Profile
stats = ModelProfiler.count_parameters(model)
mem = ModelProfiler.memory_snapshot("cuda")
flops = ModelProfiler.estimate_flops(model, seq_len=512)

# Health check (end of epoch)
report = ModelHealthMonitor.full_health_check(
    model, sample_input=sample, raise_on_critical=True
)
if not report.is_healthy:
    print(report.summary())
```

---

## Training Integration

### Typical Usage in train.py

```python
mm = ModelManager(TRAIN_CONFIG)
mm.initialize(build_model=True, build_optimizer=True,
              build_criterion=True, build_scheduler=True)
mm.setup_tensorboard("runs/cevahir_v6")

if resume_path:
    mm.load(resume_path)

for epoch in range(start_epoch, total_epochs):
    mm.train_mode()
    for batch in train_loader:
        # training step...
        pass

    mm.eval_mode()
    val_loss = validate(mm, val_loader)

    if mm.scheduler:
        mm.scheduler.step(val_loss)

    mm.save(epoch=epoch, metadata={"val_loss": val_loss})

    if epoch % 10 == 0:
        report = mm.health_check()
        if not report.is_healthy:
            logger.warning(report.summary())
```

---

## Dependencies

| Package | Version | Required | Purpose |
|---------|---------|----------|---------|
| `torch` | >= 2.0 | Yes | PyTorch core |
| `bitsandbytes` | >= 0.41 | No | AdamW8bit + INT8/INT4 quantization |
| `torch.distributed` | Ships with PyTorch | No | DDP/FSDP |

---

*Author: Muhammed Yasin Yılmaz — Cevahir-AI Project*
*Copyright: © 2024-2026 Muhammed Yasin Yılmaz. All Rights Reserved.*
