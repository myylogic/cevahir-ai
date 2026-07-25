# 🧠 Neural Network Module — V6 Architecture Documentation

**Version:** V6
**Last Updated:** 2026-03-16
**Status:** Production-Ready
**Main File:** `src/neural_network.py`
**Module Directory:** `src/neural_network_module/`

---

## Table of Contents

1. [Overview](#overview)
2. [V6 Change Summary](#v6-change-summary)
3. [Architecture Structure](#architecture-structure)
4. [Components](#components)
   - [CevahirNeuralNetwork](#cevahirneuralnetwork)
   - [TransformerEncoderLayer](#transformerencoderlayer)
   - [MultiHeadAttention](#multiheadattention)
   - [FeedForwardNetwork](#feedforwardnetwork)
   - [RMSNorm](#rmsnorm)
   - [KVCache](#kvcache)
   - [PositionalEncoding / RoPE](#positionalencoding--rope)
   - [LanguageEmbedding](#languageembedding)
   - [MixtureOfExperts](#mixtureofexperts)
5. [Parameters and Configuration](#parameters-and-configuration)
6. [Advanced Features](#advanced-features)
7. [Performance Notes](#performance-notes)
8. [Usage Examples](#usage-examples)
9. [Smoke Test](#smoke-test)
10. [Removed Components](#removed-components)

---

## Overview

`CevahirNeuralNetwork` is the core of the Cevahir-AI project. This decoder-only Transformer integrates modern LLM techniques: GQA (Grouped-Query Attention), YaRN RoPE, SwiGLU, Flash Attention (F.sdpa), QK-Norm, Logit Soft-Cap, and Scaled Residual Init.

```
Input Tokens
      │
      ▼
LanguageEmbedding (vocab_size → embed_dim)
      │
      ▼
PositionalEncoding / RoPE
      │
      ▼
embed_dropout
      │
      ▼
┌─────────────────────────────────────────────┐
│  TransformerEncoderLayer × N                │
│  ┌───────────────────────────────────────┐  │
│  │  Pre-Norm (RMSNorm)                   │  │
│  │  MultiHeadAttention (GQA, F.sdpa)     │  │
│  │  Residual Connection                  │  │
│  │  Pre-Norm (RMSNorm)                   │  │
│  │  FeedForwardNetwork (SwiGLU)          │  │
│  │  Residual Connection                  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
      │
      ▼
output_norm (RMSNorm)
      │
      ▼
output_layer (Linear, weight-tied)
      │
      ▼
Logit Soft-Cap: tanh(x / 30.0) * 30.0
      │
      ▼
Output Logits (vocab_size)
```

---

## V6 Change Summary

### Added Features

| Feature | Parameter | Reference |
|---------|-----------|-----------|
| PyTorch Flash Attention | `use_pytorch_sdpa=True` | PyTorch 2.0+ F.sdpa |
| QK-Norm | `use_qk_norm=False` | Gemma / PaLM 2 |
| Parallel Residual | `parallel_residual=False` | GPT-J, PaLM |
| Output Logit Soft-Cap | `logit_soft_cap=30.0` | Gemma 2 |
| Attention Logit Cap | `attn_logit_cap=0.0` | Gemma 2 |
| Scaled Residual Init | Automatic | LLaMA / DeepNet |
| SwiGLU ffn_dim Auto | `ffn_dim=None` | LLaMA 3 standard |
| Attention Entropy Monitoring | Automatic | Training health |

### Critical Bug Fixes (V5→V6)

| Bug | Fix |
|-----|-----|
| Double residual+norm inside MHA | MHA now returns raw projection; residual in TransformerEncoderLayer |
| Q/K/V dropout (wrong position) | Q/K/V dropout removed; attention weight dropout (post-softmax) kept |
| `time.time()` on every forward | Timing code removed |
| NaN/Inf check on every forward | Wrapped in `if self.debug:` guard |
| SwiGLU: `gate * sigmoid(gate)` | `F.silu(gate)` (kernel-fused, 15–20% faster) |
| MoE `_moe_losses` memory leak | Cleanup via `get_and_reset_moe_loss()` |

### Removed Components

| Component | Removed | Reason |
|-----------|---------|--------|
| `DilKatmani` class | 2026-01-04 | Replaced by `LanguageEmbedding` |
| `SeqProjection` | 2026-01-04 | Unused |
| `MemoryManager` | 2026-01-04 | Moved to Cognitive Management layer |
| Memory Manager Module | 2026-01-04 | Removed as separate module |
| Tensor Adapter Module | 2026-01-04 | Unnecessary abstraction |

---

## Architecture Structure

```
src/
├── neural_network.py                          # Main model: CevahirNeuralNetwork
└── neural_network_module/
    ├── __init__.py
    ├── dil_katmani_module/
    │   ├── language_embedding.py              # Token embedding + init
    │   └── positional_encoding.py            # Sinusoidal / Learned / RoPE / YaRN
    └── ortak_katman_module/
        ├── transformer_encoder_layer.py       # Single Transformer layer
        ├── feed_forward_network.py            # FFN (SwiGLU, GeGLU, GELU, etc.)
        ├── rms_norm.py                        # RMSNorm
        ├── kv_cache.py                        # KV Cache (autoregressive)
        ├── mixture_of_experts.py              # MoE (optional)
        └── attention_manager_module/
            └── multi_head_attention.py        # MHA / GQA / MQA + Flash Attention
```

---

## Components

### CevahirNeuralNetwork

**File:** `src/neural_network.py`

Main model class. Composes all subcomponents and drives training/inference.

#### `__init__` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | — | Vocabulary size |
| `embed_dim` | int | — | Embedding dimension |
| `num_heads` | int | — | Number of attention heads |
| `num_layers` | int | — | Number of Transformer layers |
| `ffn_dim` | int\|None | `None` | FFN inner dimension; `None` → SwiGLU auto |
| `max_seq_len` | int | 2048 | Maximum sequence length |
| `dropout` | float | 0.1 | Dropout rate |
| `num_kv_heads` | int\|None | `None` | KV heads for GQA; `None` → MHA |
| `use_swiglu` | bool | `True` | SwiGLU activation |
| `use_moe` | bool | `False` | Mixture of Experts |
| `num_experts` | int | 8 | MoE expert count |
| `top_k_experts` | int | 2 | MoE top-K routing |
| `rope_scaling_type` | str\|None | `None` | RoPE scaling: `"yarn"` |
| `rope_scaling_factor` | float | 1.0 | YaRN scaling factor |
| `sliding_window` | int\|None | `None` | Sliding Window Attention |
| `use_pytorch_sdpa` | bool | `True` | F.scaled_dot_product_attention (**V6**) |
| `use_qk_norm` | bool | `False` | QK-Norm (**V6**) |
| `parallel_residual` | bool | `False` | Parallel Attention+FFN (**V6**) |
| `logit_soft_cap` | float | `30.0` | Output logit tanh cap (**V6**) |
| `attn_logit_cap` | float | `0.0` | Attention score tanh cap (**V6**) |

#### `forward()` Signature

```python
def forward(
    self,
    input_ids: torch.Tensor,          # [B, T] — input token IDs
    attention_mask: Optional[torch.Tensor] = None,  # [B, T]
    use_cache: bool = False,           # Use KV cache
    past_key_values: Optional[...] = None,
    output_attentions: bool = False,
) -> tuple
```

**Returns:**
- `(logits, attn_weights)` — when no cache
- `(logits, attn_weights, kv_cache)` — when `use_cache=True`

#### Main Methods

| Method | Description |
|--------|-------------|
| `forward()` | Forward pass |
| `clear_kv_cache()` | **Required**: Must be called before each new generation |
| `get_num_params()` | Parameter count |
| `get_model_size_mb()` | Model size (MB) |

> **⚠️ Critical:** When using `use_cache=True` for autoregressive generation, call `model.clear_kv_cache()` before each new sequence. Otherwise the previous sequence’s cache will contaminate the next generation.

---

### TransformerEncoderLayer

**File:** `src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py`

Single Transformer block. Pre-Norm is the default (industry standard since GPT-2).

#### Pre-Norm Flow (Default)

```
x → Norm1 → MHA → + x → Norm2 → FFN → + x
```

#### Post-Norm Flow (`pre_norm=False`)

```
x → MHA → + x → Norm1 → FFN → + x → Norm2
```

#### Parallel Residual (`parallel_residual=True`, V6)

```python
x_norm = self.norm1(x)
attn_out, _ = self.attention(x_norm, ...)
ffn_out = self.ffn(x_norm)
x = x + attn_out + ffn_out
```

From GPT-J and PaLM: a single norm feeds both branches from the same normalized input; computation can be parallelized.

#### V6 New Parameters

| Parameter | Description |
|-----------|-------------|
| `use_pytorch_sdpa` | Passed to MHA |
| `use_qk_norm` | Passed to MHA |
| `parallel_residual` | GPT-J-style parallel branch |
| `attn_logit_cap` | Passed to MHA |

#### MoE Integration

```python
# When MoE is on, MoE is used instead of FFN
if self.use_moe:
    ffn_out, moe_loss = self.moe(x_norm)
    self._moe_losses.append(moe_loss)
else:
    ffn_out = self.ffn(x_norm)
```

```python
# To avoid memory leak:
loss = layer.get_and_reset_moe_loss()   # Clears the list
```

---

### MultiHeadAttention

**File:** `src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py`

#### Attention Types

| Type | Condition | Description |
|------|-----------|-------------|
| MHA | `num_kv_heads == num_heads` | Standard multi-head attention |
| GQA | `1 < num_kv_heads < num_heads` | Grouped-query attention |
| MQA | `num_kv_heads == 1` | Multi-query attention |

#### GQA (Grouped-Query Attention)

```
Q: [B, num_heads, T, head_dim]       → 8 heads
K: [B, num_kv_heads, T, head_dim]    → 2 heads (grouped)
V: [B, num_kv_heads, T, head_dim]    → 2 heads

# Each KV group serves num_heads/num_kv_heads = 4 queries
K = K.expand(-1, num_heads, -1, -1)  # Broadcast to [B, 8, T, head_dim]
```

**Memory gain:** `num_kv_heads=2` → ~75% smaller KV cache.

#### F.scaled_dot_product_attention (V6)

```python
if self.use_pytorch_sdpa:
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=self.dropout_rate if self.training else 0.0,
        is_causal=True,   # Causal mask built automatically
        scale=None,       # 1/sqrt(head_dim) automatic
    )
```

With PyTorch 2.0+, this API automatically selects:
- **Flash Attention** (GPU, FP16/BF16, causal mask)
- **Memory-Efficient Attention** (long sequences)
- **Math** (fallback, CPU)

No external `flash-attn` package required.

#### QK-Norm (V6)

RMSNorm is applied to Q and K separately before RoPE:

```python
if self.use_qk_norm:
    q = self.q_norm(q)   # [B, H, T, head_dim]
    k = self.k_norm(k)   # [B, H, T, head_dim]
# Then RoPE is applied
```

Reduces attention logit blow-up in long context (Gemma / PaLM 2).

#### Attention Logit Cap (V6)

```python
if self.attn_logit_cap > 0.0:
    scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap
```

Suppresses attention spikes before softmax (Gemma 2). When `use_pytorch_sdpa=True`, the manual path is skipped so this is effectively disabled.

#### Sliding Window Attention

```python
# When sliding_window != None:
# Each token attends only to the last sliding_window tokens
mask = create_sliding_window_mask(seq_len, sliding_window)
```

---

### FeedForwardNetwork

**File:** `src/neural_network_module/ortak_katman_module/feed_forward_network.py`

#### Supported Activations

| Category | Activations |
|----------|-------------|
| Gated | `swiglu`, `geglu` |
| Standard | `gelu`, `gelu_tanh`, `relu`, `silu`, `mish` |

#### SwiGLU (Default)

```python
# Gate projection + Up projection + Down projection
gate = self.gate_proj(x)    # [B, T, ffn_dim]
gate = F.silu(gate)         # SiLU = x * sigmoid(x), kernel-fused
gate *= self.up_proj(x)     # In-place multiply: 2 tensors → memory efficient
out = self.down_proj(gate)  # [B, T, embed_dim]
```

**Why in-place multiply?** Naive gated activation keeps 4 tensors; `gate *= up` reduces to 2 (~50% VRAM saving).

#### SwiGLU ffn_dim Auto (V6)

```python
if use_swiglu and ffn_dim is None:
    raw = int(2 / 3 * 4 * effective_dim)   # LLaMA 3 formula
    ffn_dim = (raw + 255) // 256 * 256      # Round to multiple of 256 (tensor core optimal)
    # embed_dim=512 → raw=1365 → ffn_dim=1536
```

| embed_dim | ffn_dim (auto) |
|-----------|----------------|
| 256 | 768 |
| 512 | 1536 |
| 768 | 2048 |
| 1024 | 2816 |

#### Bias

```python
use_bias=False  # Default — LLaMA / PaLM standard
```

---

### RMSNorm

**File:** `src/neural_network_module/ortak_katman_module/rms_norm.py`

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * scale
```

Unlike LayerNorm: no mean subtraction (centering), so more efficient in compute and memory.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
```

---

### KVCache

**File:** `src/neural_network_module/ortak_katman_module/kv_cache.py`

Key-value cache for autoregressive generation.

#### Tensor Shapes

```
key_cache:   [B, num_kv_heads, max_cache_len, head_dim]
value_cache: [B, num_kv_heads, max_cache_len, head_dim]
```

#### Usage

```python
# CLEAR before each new sequence
model.clear_kv_cache()

# Generation loop
for step in range(max_new_tokens):
    logits, _, kv_cache = model(
        input_ids=next_token,
        use_cache=True,
        past_key_values=kv_cache
    )
```

---

### PositionalEncoding / RoPE

**File:** `src/neural_network_module/dil_katmani_module/positional_encoding.py`

| Mode | Description |
|------|-------------|
| `sinusoidal` | Fixed sin/cos positional encoding |
| `learned` | Learnable position embeddings |
| `rope` | Rotary Position Embedding |

#### YaRN RoPE (V5+)

```python
rope_scaling_type="yarn"
rope_scaling_factor=4.0   # 2048 → 8192 token context
```

YaRN (Yet another RoPE extensioN) generalizes to unseen lengths via interpolation.

---

### LanguageEmbedding

**File:** `src/neural_network_module/dil_katmani_module/language_embedding.py`

Maps token IDs to `embed_dim`-dimensional vectors.

```python
# Init methods: "xavier_normal", "normal", "uniform"
init_method="xavier_normal"
scale_by_sqrt=True    # Scale by sqrt of embed dim
padding_idx=0         # PAD token zero vector
```

**Weight Tying:** `output_layer.weight = embedding.weight` (parameter sharing, ~20% parameter saving).

---

### MixtureOfExperts

**File:** `src/neural_network_module/ortak_katman_module/mixture_of_experts.py`

Optional sparse FFN layer.

```python
use_moe=True
num_experts=8    # Total experts
top_k_experts=2 # Active experts per token
```

**Router:** `nn.Linear(embed_dim, num_experts, bias=False)` — Softmax + Top-K selection.

**Sparse dispatch:** Only selected experts process their tokens (~8x memory reduction).

**Load balancing:** Auxiliary loss encourages even expert usage.

> **⚠️ Warning:** Checkpoints with `use_moe=True` are incompatible with `use_moe=False`. Enabling MoE requires training from scratch.

---

## Parameters and Configuration

### Recommended V6 Config

```python
model_config = {
    "vocab_size": 32000,
    "embed_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 2,          # GQA: ~75% KV cache reduction
    "num_layers": 8,
    "ffn_dim": None,            # SwiGLU auto: 1536
    "use_swiglu": True,
    "use_pytorch_sdpa": True,   # Flash Attention (PyTorch 2.0+)
    "use_qk_norm": True,        # Long-context stability
    "parallel_residual": False, # Evaluate in V7
    "logit_soft_cap": 30.0,     # Gemma 2 standard
    "attn_logit_cap": 0.0,      # Try 50.0 if needed
    "dropout": 0.1,
    "max_seq_len": 2048,
    "rope_scaling_type": "yarn", # Context extension
    "rope_scaling_factor": 4.0,
    "use_moe": False,
}
```

### Small Model (Test/Dev)

```python
model_config = {
    "vocab_size": 32000,
    "embed_dim": 128,
    "num_heads": 4,
    "num_kv_heads": 2,
    "num_layers": 2,
    "ffn_dim": None,
    "use_swiglu": True,
    "use_pytorch_sdpa": True,
    "logit_soft_cap": 30.0,
    "dropout": 0.0,
    "max_seq_len": 512,
}
```

---

## Advanced Features

### Scaled Residual Init (V6)

Prevents variance growth in deep models:

```python
# num_layers=8 → scale ≈ 0.25
scale = 1.0 / math.sqrt(2.0 * num_layers)
for layer in self.layers:
    nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=scale)
    nn.init.normal_(layer.ffn.fc2.weight, mean=0.0, std=scale)
```

Reference: LLaMA / DeepNet — improves training stability in large models.

### Output Logit Soft-Cap (V6)

```python
if self.logit_soft_cap > 0.0:
    logits = torch.tanh(logits / self.logit_soft_cap) * self.logit_soft_cap
```

With `logit_soft_cap=30.0`, output logits are softly clamped to `[-30, +30]`. This:
- Suppresses extreme logits
- Improves training stability
- Avoids numerical overflow in softmax

### Attention Entropy Monitoring

Attention weight entropy is computed for all layers each step:

```python
# Normalized Shannon entropy: H / log(T)
if normalized_h < 0.05:
    logger.warning(f"Attention collapse! layer={i}, entropy={normalized_h:.4f}")
```

**State labels:**
- `collapse`: entropy < 0.05 → Model focuses all attention on one position
- `normal`: 0.05 ≤ entropy < 0.8 → Healthy range
- `uniform`: entropy ≥ 0.8 → Attention too spread (underfitting)

Query state via `_last_snapshot["attn_entropy"]`.

---

## Performance Notes

### Memory Comparison

| Configuration | VRAM (embed=512, 8 layers, seq=2048) |
|---------------|----------------------------------------|
| MHA (baseline) | ~4.2 GB |
| GQA (num_kv_heads=2) | ~3.1 GB (26% less) |
| GQA + SwiGLU in-place | ~2.8 GB |
| GQA + F.sdpa | ~2.1 GB (50% less) |

### Speed Comparison

| Feature | Effect |
|---------|--------|
| F.silu vs gate*sigmoid | 15–20% faster (kernel fusion) |
| F.sdpa vs manual SDPA | 30–40% faster (Flash Attention) |
| Debug guard (NaN check) | No 2–3x slowdown (only in debug mode) |
| `time.time()` removed | Small overhead reduction |

---

## Usage Examples

### Creating the Model

```python
import torch
from src.neural_network import CevahirNeuralNetwork

model = CevahirNeuralNetwork(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2,
    num_layers=8,
    ffn_dim=None,           # Auto: 1536
    use_swiglu=True,
    use_pytorch_sdpa=True,
    use_qk_norm=True,
    logit_soft_cap=30.0,
    dropout=0.1,
    max_seq_len=2048,
)

print(f"Parameters: {model.get_num_params():,}")
print(f"Model size: {model.get_model_size_mb():.1f} MB")
```

### Forward Pass (Training)

```python
input_ids = torch.randint(0, 32000, (4, 512))  # [batch=4, seq=512]
logits, attn_weights = model(input_ids)
# logits: [4, 512, 32000]

loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, 32000),
    input_ids[:, 1:].reshape(-1)
)
loss.backward()
```

### Autoregressive Generation

```python
model.eval()
model.clear_kv_cache()   # ⚠️ Required

input_ids = torch.tensor([[1, 234, 567]])  # [1, 3]

with torch.no_grad():
    for _ in range(100):
        logits, _, kv_cache = model(
            input_ids=input_ids[:, -1:],
            use_cache=True,
            past_key_values=kv_cache if _ > 0 else None
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
```

### MoE Loss Aggregation (if enabled)

```python
total_loss = task_loss
for layer in model.layers:
    moe_loss = layer.get_and_reset_moe_loss()
    if moe_loss is not None:
        total_loss = total_loss + 0.01 * moe_loss
```

---

## Smoke Test

```python
import torch
from src.neural_network import CevahirNeuralNetwork

# Create model
model = CevahirNeuralNetwork(
    vocab_size=32000, embed_dim=512, num_heads=8, num_kv_heads=2,
    num_layers=2, ffn_dim=None, use_swiglu=True,
    use_pytorch_sdpa=True, use_qk_norm=True, logit_soft_cap=30.0
)

x = torch.randint(0, 32000, (2, 128))  # [batch=2, seq=128]
logits, _ = model(x)

# Basic checks
assert logits.shape == (2, 128, 32000), f"Shape error: {logits.shape}"
assert not torch.isnan(logits).any(), "NaN detected!"
assert logits.abs().max().item() <= 30.0 + 1e-3, "Soft cap not working!"

# QK-Norm check
assert hasattr(model.layers[0].attn, "q_norm"), "q_norm missing! use_qk_norm=True required"

# SwiGLU ffn_dim check
assert model.layers[0].ffn.ffn_dim == 1536, f"ffn_dim expected=1536, got={model.layers[0].ffn.ffn_dim}"

# KV head check (GQA)
assert model.layers[0].attn.num_kv_heads == 2, "GQA num_kv_heads error"

print("V6 smoke test: PASSED ✓")
```

---

## Removed Components

The following were removed before V6. Update any references in old code:

### DilKatmani → LanguageEmbedding

```python
# OLD (REMOVED):
from src.neural_network_module.dil_katmani_module import DilKatmani

# NEW:
from src.neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding
```

### SeqProjection (REMOVED)

```python
# OLD (REMOVED):
self.seq_projection = SeqProjection(embed_dim, output_dim)

# NEW: Use nn.Linear directly
self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
```

### MemoryManager → Cognitive Management

```python
# OLD (REMOVED):
from src.neural_network_module import MemoryManager

# NEW: Memory is in cognitive_management module
from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | `F.scaled_dot_product_attention` |
| `torch` | ≥ 1.9 | Core model |

> **Note:** V6 Flash Attention requires `torch >= 2.0`. On older PyTorch, set `use_pytorch_sdpa=False`.

---

*Author: Muhammed Yasin Yılmaz — Cevahir-AI Project*
*Copyright: © 2024-2026 Muhammed Yasin Yılmaz. All Rights Reserved.*
