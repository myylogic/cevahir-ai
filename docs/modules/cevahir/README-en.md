# Cevahir API — Module Documentation

**Version:** V-4 (Unified Inference API)
**File:** `model/cevahir.py`
**Line Count:** ~2084
**Purpose:** Facade layer that combines the TokenizerCore + ModelManager + CognitiveManager components into a single inference API.

---

## Table of Contents

1. [General Overview](#general-overview)
2. [Architecture](#architecture)
3. [Exception Hierarchy](#exception-hierarchy)
4. [Helper Classes](#helper-classes)
5. [CevahirConfig](#cevahirconfig)
6. [CevahirModelAPI](#cevahirmodelapi)
7. [Cevahir Main Class](#cevahir-main-class)
8. [API Reference](#api-reference)
9. [Usage Examples](#usage-examples)
10. [Design Patterns](#design-patterns)
11. [Difference from the Training System](#difference-from-the-training-system)

---

## General Overview

`cevahir.py` is the **single entry point (Unified Inference API)** of the Cevahir-AI system. It is designed exclusively for inference (generation). This file is not used during training.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cevahir (Facade)                         │
│              ⚠️  INFERENCE ONLY — NOT FOR TRAINING               │
├─────────────────────┬───────────────────┬───────────────────────┤
│    TokenizerCore    │   ModelManager    │   CognitiveManager    │
│  encode / decode    │ forward / predict │  process / memory     │
│  BPE tokenization   │ save / load       │  tools / critic       │
│  GPU batch encode   │ KV Cache          │  Self-Refine          │
└─────────────────────┴───────────────────┴───────────────────────┘
```

### Training vs. Inference Separation

| Task | File to Use |
|---|---|
| BPE vocab/merges training | `tokenizer_management/train_bpe.py` |
| Model training entry point | `training_system/train.py` |
| Training service | `training_system/training_service.py` |
| Neural network training | `model_management/model_manager.py` |
| **Inference / Chat / Production** | **`model/cevahir.py`** ← here |

---

## Architecture

```
cevahir.py
├── Exception Hierarchy
│   ├── CevahirError (root)
│   ├── CevahirInitializationError
│   ├── CevahirConfigurationError
│   └── CevahirProcessingError (context + suggestion)
│
├── Helper Classes (internal)
│   ├── _ErrorContextBuilder     → Error message + suggestion generation
│   └── _InputValidator          → str/list/Tensor → [B, T] Tensor
│
├── Decorators
│   ├── @requires_tokenizer
│   ├── @requires_model_manager
│   ├── @requires_cognitive_manager
│   └── @requires_model_api
│
├── CevahirConfig (dataclass)    → All configuration
│
├── CognitiveModelAPI (Protocol) → Interface expected by CognitiveManager
│
├── CevahirModelAPI              → Adapter Pattern
│   ├── generate()               → Autoregressive generation (KV Cache)
│   ├── _autoregressive_generate() → Core generation loop
│   ├── _generate_with_beam_search() → Beam search generation
│   ├── score()                  → Log-probability scoring
│   ├── entropy_estimate()       → Shannon entropy measurement
│   └── process_audio/image/multimodal()
│
└── Cevahir (main class)         → Public API
    ├── Tokenization API         → encode / decode / train_tokenizer
    ├── Model API                → forward / generate / predict / freeze
    ├── Cognitive API            → process / generate_batch / memory / tools
    ├── Monitoring               → get_metrics / get_health_status
    └── Properties               → tokenizer / model / device / cognitive
```

---

## Exception Hierarchy

```
CevahirError (Exception)
├── CevahirInitializationError   → Component initialization error
├── CevahirConfigurationError    → Configuration error
└── CevahirProcessingError       → Processing error
     ├── .context: Dict          → Additional context information
     └── .suggestion: str        → Contains a suggestion for the user
```

### CevahirProcessingError

```python
class CevahirProcessingError(CevahirError):
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ): ...
```

The `suggestion` field explains to the user what to do after an error. For example, on TokenizerCore errors a suggestion such as `"Ensure tokenizer vocab and merges files exist and are valid"` is delivered.

---

## Helper Classes

### `_ErrorContextBuilder`

Catches exceptions from components and generates structured error messages and suggestions.

```python
msg, suggestion = _ErrorContextBuilder.build_error_message(
    operation="Forward pass",
    error=exc,
    component="ModelManager"
)
# msg      → "Forward pass failed (component: ModelManager): ..."
# suggestion → "Verify model is initialized and model weights are loaded correctly"
```

Component-based suggestions:

| Component | Suggestion |
|---|---|
| `TokenizerCore` | Check vocab/merges files |
| `ModelManager` | Is the model initialized, are the weights loaded |
| `CognitiveManager` | Check configuration and dependencies |

---

### `_InputValidator`

Normalizes inputs passed to methods such as `forward()`.

```python
tensor = _InputValidator.validate_and_convert_input(
    inputs=inputs,       # str | List[int] | torch.Tensor
    device="cuda",
    tokenizer_core=tok,  # for str conversion
    vocab_size=50000     # for clamping
)
```

**Conversion rules:**

| Input Type | Operation |
|---|---|
| `str` | Encode with TokenizerCore → `[1, T]` Tensor |
| `List[int]` | `torch.tensor([ids])` → `[1, T]` Tensor |
| `torch.Tensor` | `.long()`, if 1D then `unsqueeze(0)` → `[B, T]` Tensor |
| Empty input | `[1, 1]` Tensor with PAD token |

Token IDs outside the vocab range are clamped to `vocab_size - 1`.

---

## CevahirConfig

The main `dataclass` that configures the entire system.

```python
@dataclass
class CevahirConfig:
    device: str = "cuda"
    seed: Optional[int] = 42
    log_level: str = "INFO"

    # Tokenizer configuration (dict)
    tokenizer: Dict[str, Any] = field(default_factory=dict)

    # Model configuration — V-4 defaults:
    model: Dict[str, Any] = field(default_factory=lambda: {
        "vocab_size": 50000,
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
        "ffn_dim": 2048,
        "pe_mode": "rope",
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_kv_cache": True,
        "use_moe": False,
        "dropout": 0.1,
        "max_seq_len": 2048,
    })

    # CognitiveManager configuration
    cognitive: Optional[CognitiveManagerConfig] = None

    # Checkpoint loading
    load_model_path: Optional[str] = None  # None → auto-detect

    # Config file (for hot reload)
    config_path: Optional[str] = None
    enable_hot_reload: bool = False
```

### `validate()`

Checks configuration validity; raises `CevahirConfigurationError` if an invalid value is found.

### CevahirConfig from a dict

```python
config = CevahirConfig(**my_dict)
```

---

## CevahirModelAPI

An **Adapter class** that adapts `ModelManager` to the `CognitiveModelAPI` Protocol expected by `CognitiveManager`.

```
CevahirModelAPI
├── Implements: CognitiveModelAPI Protocol
├── Wraps: ModelManager + TokenizerCore
└── Methods:
    ├── generate(prompt, decoding_cfg) → str
    ├── score(prompt, candidate)       → float
    └── entropy_estimate(text)         → float [0, 1]
```

### `generate(prompt, decoding_cfg)` → `str`

Autoregressive generation — KV Cache based.

```
Flow:
1. Prompt → encode → [1, T] Tensor
2. Clear KV Cache (prevents contamination from the previous turn)
3. First forward: full prompt → initialize cache
4. Subsequent forwards: only the last token (with cache_position)
5. Repetition penalty → Temperature → Top-k → Top-p → multinomial
6. Stop when EOS arrives (conditional on generating at least 5 tokens)
7. Decode new tokens (prompt tokens excluded)
```

**Parameter bounds:**

| Parameter | Range |
|---|---|
| `max_new_tokens` | `[0, 2048]` |
| `top_p` | `[0.01, 1.0]` |
| `top_k` | `>= 0` (0 = disabled) |
| `repetition_penalty` | `> 1.0` active, applied over the last 256 tokens |

**EOS behavior:** EOS is permitted only after a minimum of 5 tokens have been generated — a safeguard against early collapse.

**NaN protection:** If all probabilities are NaN/zero, a uniform fallback is used; as a last resort the first token is selected.

---

### `_generate_with_beam_search(prompt, max_new_tokens, beam_width)`

```
Flow:
1. Initialize beam_width beams (all with the same prompt)
2. For each beam: forward → top (beam_width × 2) candidates
3. Sort candidates by cumulative log-probability
4. Select the best beam_width candidates
5. Stop when all beams reach EOS or max_new_tokens is exhausted
6. The highest-scoring beam is decoded
7. On error, fall back to standard generation
```

---

### `score(prompt, candidate)` → `float [0, 1]`

Measures the probability of the candidate text given the prompt.

```
avg_log_prob = mean(log P(candidate_token_i | prompt + candidate[:i]))
score = clamp((avg_log_prob + 10) / 10, 0.0, 1.0)
```

A length-based heuristic fallback is available.

---

### `entropy_estimate(text)` → `float [0, 1]`

Shannon entropy measurement over the model's next-token distribution.

```python
H = -sum(p_i * log(p_i))            # Raw Shannon entropy
normalized = H / log(vocab_size)     # [0, 1]
# 0 → model is confident (single dominant token)
# 1 → model is uncertain (uniform distribution)
```

**Academic reference:** Kuhn et al. 2023 (active learning, LLM calibration)

**Heuristic fallback:** If the model forward pass fails, Type-Token Ratio (TTR) is used.

---

## Cevahir Main Class

```python
class Cevahir:
    """Unified API: TokenizerCore + ModelManager + CognitiveManager"""
```

### `__init__`

```python
Cevahir(
    config: Union[CevahirConfig, Dict[str, Any]],
    *,
    tokenizer_core: Optional[TokenizerCore] = None,  # Dependency injection
    model_manager: Optional[ModelManager] = None,    # Dependency injection
    cognitive_manager: Optional[CognitiveManager] = None,  # Dependency injection
)
```

Initialization sequence:

```
1. Config parse + validate
2. Set seed (for reproducibility)
3. Set log level
4. TokenizerCore init (or inject)
5. ModelManager init + model auto-load (or inject)
6. Create CevahirModelAPI adapter
7. CognitiveManager init (or inject)
```

**Model auto-load:** If `load_model_path` is None, `saved_models/cevahir_model.pth` is checked. If not found, initialization continues with random weights (a warning is issued).

**ModelManager initialization mode:** Inference only — `build_optimizer=False, build_criterion=False, build_scheduler=False`

---

## API Reference

### Decorator Guards

| Decorator | Condition | Error |
|---|---|---|
| `@requires_tokenizer` | `_tokenizer_core` is present | `CevahirProcessingError` |
| `@requires_model_manager` | `_model_manager` is present | `CevahirProcessingError` |
| `@requires_cognitive_manager` | `_cognitive_manager` is present | `CevahirProcessingError` |
| `@requires_model_api` | `_model_api` is present | `CevahirProcessingError` |

---

### Tokenization API

#### `encode(text, mode="inference") → Tuple[List[str], List[int]]`

```python
tokens, ids = cevahir.encode("Merhaba dünya")
# tokens → ["Mer", "haba", " dü", "nya"]
# ids    → [142, 883, 2041, 567]
```

#### `decode(token_ids) → str`

```python
text = cevahir.decode([142, 883, 2041, 567])
# → "Merhaba dünya"
```

#### `train_tokenizer(corpus)`

Trains the tokenizer on the given corpus. (BPE training)

---

### Model API

#### `forward(inputs, **kwargs) → torch.Tensor`

```python
logits = cevahir.forward("Merhaba")       # str input
logits = cevahir.forward([142, 883])      # list[int] input
logits = cevahir.forward(tensor)          # Tensor input
# → [B, T, vocab_size]
```

Type conversion is performed automatically via `_InputValidator`.

---

#### `generate(prompt, ...) → str`

```python
text = cevahir.generate(
    prompt="Türkiye'nin başkenti",
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    use_cognitive_pipeline=True,   # Default: True
)
```

**Routing logic:**

```
use_cognitive_pipeline=True + CognitiveManager is available
  → CognitiveManager.handle(state, input, decoding=...)
     (Self-Refine + Memory + Critic active)
  → On error: fallback to direct generation

use_cognitive_pipeline=False OR CognitiveManager not available
  → CevahirModelAPI.generate(prompt, decoding_cfg)
     (raw autoregressive generation)
```

---

#### `predict(inputs, topk=1, apply_softmax=True, return_logits=False) → Dict`

```python
result = cevahir.predict("Merhaba", topk=5)
# → {"predictions": [...], "probabilities": [...]}
```

---

#### `freeze(patterns) → Dict`

```python
report = cevahir.freeze(["embedding.*", "layers.0"])
# → {"frozen": [...], "skipped": [...]}
```

#### `unfreeze(patterns) → Dict`

```python
report = cevahir.unfreeze("layers.*")
```

---

#### `save_model(path) / load_model(path)`

```python
cevahir.save_model("checkpoints/v6_best.pt")
cevahir.load_model("checkpoints/v6_best.pt")
```

---

#### `train_mode() / eval_mode()`

```python
cevahir.eval_mode()   # For inference (dropout disabled)
cevahir.train_mode()  # For fine-tuning (dropout enabled)
```

---

#### `configure_tensorboard(...)`

```python
cevahir.configure_tensorboard(
    log_dir="runs/v6",
    log_every_n=100,
    log_histograms=True,
    log_attention_image=True,
    enable=True
)
writer = cevahir.get_tb_writer()
```

---

#### `update_model(update_params, dry_run=False) → Dict`

Performs a parameter update through ModelManager.

---

### Cognitive API

#### `process(text, state=None) → CognitiveOutput`

```python
output = cevahir.process("Bana yapay zeka hakkında bilgi ver")
print(output.text)
```

Passes through the full CognitiveManager pipeline (Critic, Memory, Deliberation).

---

#### `generate_batch(prompts, ...) → List[str]`

```python
results = cevahir.generate_batch(
    ["Question 1", "Question 2", "Question 3"],
    max_new_tokens=64,
    temperature=0.7
)
```

Sequential processing (currently). If one prompt fails the others continue; `""` is returned for the failed prompt.

---

#### `process_batch(texts, states=None) → List[CognitiveOutput]`

```python
outputs = cevahir.process_batch(["Text 1", "Text 2"])
```

An independent CognitiveState is used for each text.

---

#### `add_memory(note)` / `search_memory(query, limit=5) → List[Dict]`

```python
cevahir.add_memory("User prefers Python")
results = cevahir.search_memory("programming language", limit=3)
```

---

#### `register_tool(name, func, description, parameters)` / `list_tools()`

```python
def calculate(expr: str) -> float:
    return eval(expr)

cevahir.register_tool(
    name="calculate",
    func=calculate,
    description="Performs a mathematical operation",
    parameters={"expr": {"type": "string"}}
)
tools = cevahir.list_tools()
```

---

### Training API (DEPRECATED)

```python
cevahir.train(...)
# → DeprecationWarning + CevahirProcessingError
# For training use: python training_system/train.py
```

`Cevahir.train()` always raises an error. Use `training_system/train.py` for actual training.

---

### Monitoring

```python
metrics = cevahir.get_metrics()
# → Metrics dictionary from CognitiveManager

health = cevahir.get_health_status()
# → {"status": "healthy", ...}
```

---

### Properties

| Property | Return | Description |
|---|---|---|
| `cevahir.tokenizer` | `TokenizerCore` | Access to TokenizerCore |
| `cevahir.model` | `ModelManager` | Access to ModelManager |
| `cevahir.cognitive` | `CognitiveManager` | Access to CognitiveManager |
| `cevahir.device` | `torch.device` | Device on which the model resides |
| `cevahir.is_initialized` | `bool` | Whether the model is initialized |

---

## Usage Examples

### Basic Inference

```python
from model.cevahir import Cevahir, CevahirConfig

config = CevahirConfig(
    device="cuda",
    model={
        "vocab_size": 50000,
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
        "pe_mode": "rope",
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_kv_cache": True,
        "dropout": 0.1,
    },
    tokenizer={
        "vocab_path": "tokenizer_management/data/vocab.json",
        "merges_path": "tokenizer_management/data/merges.json",
    },
    load_model_path="saved_models/cevahir_v6_best.pt",
)

cevahir = Cevahir(config)
```

---

### Text Generation

```python
# With cognitive pipeline (default)
response = cevahir.generate(
    "Türkiye'nin başkenti nedir?",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    use_cognitive_pipeline=True,
)

# Raw model generation (cognitive bypass)
response = cevahir.generate(
    "Türkiye'nin başkenti nedir?",
    use_cognitive_pipeline=False,
)
```

---

### Encode / Decode

```python
tokens, ids = cevahir.encode("Merhaba dünya")
print(f"Tokens: {tokens}")  # ['Mer', 'haba', ' dün', 'ya']
print(f"IDs: {ids}")        # [142, 883, 2041, 567]

text = cevahir.decode(ids)
print(text)  # "Merhaba dünya"
```

---

### Uncertainty Measurement

```python
entropy = cevahir.model_api.entropy_estimate("Bu sorunun cevabı")
# 0.0 → model is confident
# 1.0 → model is uncertain
print(f"Entropy: {entropy:.3f}")
```

---

### Dependency Injection

```python
from model_management.model_manager import ModelManager
from tokenizer_management.core.tokenizer_core import TokenizerCore

# With pre-built components
tok = TokenizerCore(tok_config)
mm = ModelManager(model_config)
mm.initialize(build_optimizer=False)

cevahir = Cevahir(
    config,
    tokenizer_core=tok,
    model_manager=mm,
)
```

---

### Batch Processing

```python
prompts = ["Question 1", "Question 2", "Question 3"]
results = cevahir.generate_batch(prompts, max_new_tokens=50)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}")
    print()
```

---

### TensorBoard

```python
cevahir.configure_tensorboard(
    log_dir="runs/cevahir_v6",
    log_every_n=50,
    log_histograms=True,
    enable=True
)
writer = cevahir.get_tb_writer()
# Manual logging can be done with writer
```

---

## Design Patterns

### Facade Pattern

`Cevahir` hides three major components (TokenizerCore, ModelManager, CognitiveManager) under a single coherent API. The user can work without knowing the internal details of each component.

### Adapter Pattern

`CevahirModelAPI` converts the `ModelManager` interface to the `CognitiveModelAPI` Protocol expected by `CognitiveManager`.

```
ModelManager  →→  CevahirModelAPI  →→  CognitiveManager
(complex API)     (protocol adapter)   (Protocol: generate, score, entropy_estimate)
```

### Decorator Pattern

Manages component dependencies centrally. Instead of writing `if not self._xxx: raise` at the start of every method:

```python
@requires_model_manager
def forward(self, inputs, **kwargs):
    # Directly to business logic — the check has already been done
    ...
```

### Dependency Injection

```python
Cevahir(config, tokenizer_core=..., model_manager=..., cognitive_manager=...)
```

Components can be injected from outside. This makes it easy to use mocks or custom components in testing, profiling, and special scenarios.

---

## Difference from the Training System

```
cevahir.py (INFERENCE)          training_system/ (TRAINING)
─────────────────────            ─────────────────────────────
ModelManager.initialize(         ModelManager.initialize(
  build_optimizer=False,           build_optimizer=True,
  build_criterion=False,           build_criterion=True,
  build_scheduler=False            build_scheduler=True
)                                )

No optimizer                     AdamW / AdamW8bit
No loss computation              CrossEntropy / BCE / MSE
No gradients                     Gradient accumulation
Cevahir.train() → ERROR          training_service.train_epoch()
```

The `Cevahir` class has been deliberately stripped of training capabilities. Incorrect use is caught immediately:

```python
cevahir.train(data)
# DeprecationWarning: Cevahir.train() DEPRECATED!
# CevahirProcessingError: Use training_system/train.py for training
```

---

## Factory Function

```python
from model.cevahir import create_cevahir, CevahirConfig

config = CevahirConfig(device="cuda", ...)
cevahir = create_cevahir(config)
```

`create_cevahir()` is equivalent to calling `Cevahir(config)` directly; it also supports dependency injection parameters.

---

*Author: Muhammed Yasin Yılmaz | Copyright © 2024 Muhammed Yasin Yılmaz. All Rights Reserved.*
