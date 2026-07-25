# Cevahir AI & Engine

**Languages:** English · [Türkçe](README-TR.md)

An **end-to-end language model engine** implemented from scratch in a single
repository. Cevahir covers the full stack of building a language model — from
training a BPE tokenizer, through a decoder-only Transformer implemented in
PyTorch, to a training pipeline, an inference engine, and a cognitive reasoning
layer with memory, tools and self-critique. Every layer is source-visible and
built as an explicit, inspectable component rather than a wrapper around an
external model.

The engine was first tuned for Turkish (its tokenizer ships with Turkish
morphology and syllabification rules), but the architecture is
**language-agnostic**: the vocabulary, merges and model can be retrained for any
language and dataset.

> **Architecture documentation:** the authoritative, code-derived description of
> the system lives in [`docs/architecture/`](docs/architecture/). Start with
> [`master-architecture.md`](docs/architecture/master-architecture.md) and the
> [search index](docs/architecture/architecture-search-index.md); per-unit
> internals are under [`docs/architecture/code-reality/`](docs/architecture/code-reality/).

<p align="center">
  <img src="image/87E09A64-4E1F-41D5-84AF-7D7C56F6C229.png" style="max-width:100%;">
</p>

---

## What it is, technically

Cevahir is a **decoder-only (causal) language model engine**. It is not a
fine-tuning script and not a client for a hosted model — it is the model and its
surrounding machinery, implemented directly:

- A **from-scratch BPE tokenizer** with a Turkish-aware pre-tokenizer,
  syllabifier and morphological analyzer, plus CPU and optional GPU code paths.
- A **Transformer decoder** written in PyTorch (`torch.nn`) with RoPE/YaRN
  positional encoding, multi-head causal attention (Flash / PyTorch-SDPA /
  manual backends), SwiGLU feed-forward, RMSNorm, an optional Mixture-of-Experts
  layer, a KV cache with sliding-window eviction, weight tying, quantization and
  activation checkpointing.
- A **unified inference engine** (`Cevahir`) that composes the tokenizer, model
  and cognitive layer behind one API, with autoregressive and beam-search
  decoding.
- A **two-layer training stack**: a run/service layer (data cache, split, model
  init) driving a training-engine layer (loop, gradient/loss management,
  stability safeguards, curriculum, EMA/SAM/Lookahead, TensorBoard).
- A **cognitive layer** that turns a single `generate` call into a reasoning
  pipeline: feature extraction, entropy-based policy routing (direct / think /
  debate / tree-of-thoughts / self-consistency), RAG memory over a vector store,
  constitutional + fact-checking critique with self-refinement, and tool use.
- A **serving surface**: a session/chat manager and a Flask REST API (JWT auth,
  versioned routes), plus a repository-based persistence layer.

The internal model architecture is tagged `V-4` in the codebase; this README
describes it by its actual components rather than a version label, since the
source mixes several internal version tags.

---

## Canonical units

The system decomposes into ten canonical units. Each is documented in depth,
directly from the code, under [`docs/architecture/code-reality/`](docs/architecture/code-reality/).

| Unit | Source | Responsibility |
|------|--------|----------------|
| **Tokenizer** | `tokenizer_management/` | BPE train + encode/decode; Turkish morphology/syllabification; OOV char fallback |
| **Neural Network** | `src/` | Decoder-only Transformer: embedding, RoPE/YaRN, attention, SwiGLU, RMSNorm, MoE, KV cache, quantization |
| **Model / Engine** | `model/`, `model_management/` | `Cevahir` facade + model lifecycle (build/init/save/load/update), decoding, health, profiling |
| **Training Management** | `training_management/` | Training engine: loop, gradient/loss/batch, safety detectors, curriculum, optimizers, monitoring |
| **Training System** | `training_system/` | Run/service layer: data cache, source-aware split, model init, epoch QA |
| **Data** | `data_loader_management/`, `data_processing/` | Train-time data loading (smart split) + offline collection (Wikipedia/subtitles) |
| **Cognitive** | `cognitive_management/` | Reasoning pipeline, policy routing, deliberation, RAG memory, critic, tools, AIOps monitoring |
| **Chatting** | `chatting_management/` | Session/conversation/user management, context building |
| **API** | `api/` | Flask REST surface (v3 routes, services, JWT, middleware) |
| **Database** | `database/` | Repository + Unit-of-Work persistence, typed models |

---

## Architecture

Layered view (dependency direction top → bottom). See
[`master-architecture.md`](docs/architecture/master-architecture.md) for the full
description and flows.

```
        ┌──────────────────────────────────────────────┐
        │  Serving: api/ (Flask, JWT)  ·  chatting_mgmt  │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Cognitive: cognitive_management/ (v2)         │
        │  policy routing · deliberation · RAG · critic  │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Engine (facade): model/cevahir.py :: Cevahir  │
        │  CevahirModelAPI adapter (model ↔ cognitive)   │
        └──────────────┬──────────────────┬─────────────┘
                       ▼                  ▼
        ┌──────────────────────┐ ┌────────────────────────┐
        │ Model lifecycle       │ │ Neural network (src/)  │
        │ model_management/     │ │ CevahirNeuralNetwork   │
        └──────────────┬────────┘ └────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────┐
        │  Training: training_system/ + training_mgmt/   │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────┐ ┌────────────────────────┐
        │ Tokenizer             │ │ Data + Database        │
        │ tokenizer_management/ │ │ loaders · repositories │
        └──────────────────────┘ └────────────────────────┘
```

**Composition:** `Cevahir.__init__` (in `model/cevahir.py`) is the composition
root for inference — it builds the tokenizer, the model (via `ModelManager`), an
adapter (`CevahirModelAPI`) and the cognitive manager, wiring them together.
Training does **not** go through this facade; it has its own entry point
(`training_system/train.py`) that shares `ModelManager`.

---

## Component detail

### Tokenizer (`tokenizer_management/`)
`TokenizerCore` wraps `BPEManager`, which orchestrates a `Pretokenizer`
(Unicode normalization, Turkish İ/ı handling, punctuation/whitespace splitting),
an optional `Syllabifier` and `Morphology` analyzer, and the BPE
`Encoder`/`Decoder`/`Trainer`. Out-of-vocabulary tokens fall back to
character-level ids, then `[UNK]`. GPU batch paths exist alongside CPU paths.

### Neural network (`src/`)
`CevahirNeuralNetwork` assembles: `LanguageEmbedding` (with √d scaling and
optional weight tying) → `PositionalEncoding` (sinusoidal / learned / RoPE /
YaRN) → N × decoder blocks (pre-norm: RMSNorm → causal multi-head attention →
residual → RMSNorm → SwiGLU FFN **or** MoE → residual) → final RMSNorm → output
projection. Attention selects a Flash, PyTorch-SDPA or manual backend at
runtime. A sliding-window `KVCache` accelerates autoregressive decoding.
The block class is named `TransformerEncoderLayer` for historical reasons but is
a **causal decoder** block.

### Engine (`model/`, `model_management/`)
`Cevahir` exposes `encode/decode`, `forward`, `generate` (autoregressive and
beam search), `process` (cognitive), batch variants, and memory/tool APIs.
`ModelManager` owns the model lifecycle; `CevahirModelAPI` adapts the model to
the cognitive layer's `ModelAPI` protocol so the cognitive layer never depends
on the concrete network.

### Training (`training_system/`, `training_management/`)
`TrainingService` (service layer) prepares device, data (from cache), a
source-id-aware train/val split and model initialization, then delegates to
`TrainingManager` (engine layer) which runs the epoch loop with gradient/loss
management, NaN/loss-spike/divergence safeguards, curriculum, optimizer
strategies (SAM, Lookahead, EMA) and TensorBoard monitoring.

### Cognitive (`cognitive_management/`)
`CognitiveManager` (facade) → `CognitiveOrchestrator` → an 8-stage
Chain-of-Responsibility pipeline: `FeatureExtraction → PolicyRouting →
Deliberation → ContextBuilding → Generation → SelfConsistency → Critic →
MemoryUpdate`. Policy routing picks a reasoning mode from entropy/length;
deliberation runs CoT/debate/ToT/react; memory combines a RAM session with an
episodic ChromaDB vector store; the critic applies constitutional principles and
optional fact-checking with self-refinement. Middleware, an event bus, a DI
container and AIOps monitoring surround the pipeline.

### Serving (`chatting_management/`, `api/`, `database/`)
`ChattingManager` handles sessions, conversations and context building over
`Cevahir.process`. The Flask API (`api/app_factory.py` is the composition root)
exposes v3 chat/session/user/health routes behind JWT auth and middleware.
`database/` provides Repository + Unit-of-Work persistence with typed models.

---

## Installation

Clone the repository and run it in your own Python environment. The project
targets Python with PyTorch (CUDA optional but recommended for training).
Dependencies are not pinned at the repository root; install PyTorch and the
libraries imported by the modules you use (e.g. ChromaDB and
sentence-transformers for the cognitive memory layer, Flask for the API,
`python-docx` for docx data loading). Setup specifics depend on your platform
and CUDA/PyTorch versions.

---

## Quick start (inference)

```python
from model.cevahir import Cevahir, CevahirConfig

# 1. Define the architecture (must match the trained checkpoint)
config = CevahirConfig(
    device="cuda",  # or "cpu"
    model={
        "vocab_size": 60000,   # comes from the trained tokenizer
        "embed_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
    },
)

# 2. Build the engine (loads saved_models/cevahir_model.pth if present)
cevahir = Cevahir(config)

# 3. Cognitive response — returns a CognitiveOutput
output = cevahir.process("Merhaba, nasılsın?")
print(output.text)          # NOTE: the field is `text`

# 4. Plain text generation (bypasses the cognitive layer)
text = cevahir.generate("Türkiye'nin başkenti", max_new_tokens=50, temperature=0.8)
print(text)
```

### Terminal chat

```bash
python model_management/chat_pipeline.py
```

Uses the `Cevahir` + `ChattingManager` pipeline; requires a checkpoint or saved
model.

---

## Training from scratch

Run the steps **in order**. (Commands reflect the actual file locations.)

**1 — Train the tokenizer** → produces the vocabulary and merges:
```bash
python tokenizer_management/train_bpe.py
```
Output: `vocab.json`, `merges.txt` (or the paths defined in config).

**2 — Build the training data cache** → converts raw data into autoregressive
training format (BOS/EOS/PAD/SEP + input/target sequences, chunked to a fixed
token length with padding):
```bash
python training_system/prepare_cache.py
```
Supported inputs: `docx`, `txt` (raw text), `json` (question–answer). To change
chunk length or padding behavior, edit `training_system/prepare_cache.py`.

**3 — Train the model** with the prepared cache:
```bash
python training_system/train.py
```
The cache is loaded automatically. A GPU is recommended.

### Changing model parameters

Model size and hyperparameters (`embed_dim`, `num_layers`, `num_heads`, `lr`,
`dropout`, …) currently must be kept consistent in **two places**:

- `model/cevahir.py` — `CevahirConfig` / model defaults (inference + pipeline).
- `training_system/train.py` — training config and model parameters.

If they diverge, loading the trained checkpoint will fail with a shape or
behavior mismatch. (Unifying this into a single source of truth is a tracked
item in the [development roadmap](docs/architecture/development-roadmap.md).)

---

## Sample outputs during training

Inference samples captured during training / epoch-end tests (prompt, generated
response, token count, EOS info are visible in the training log):

<p align="center"><img src="image/1.jpeg" style="max-width:100%;"></p>
<p align="center"><img src="image/2.jpeg" style="max-width:100%;"></p>
<p align="center"><img src="image/3.jpeg" style="max-width:100%;"></p>
<p align="center"><img src="image/4.jpeg" style="max-width:100%;"></p>
<p align="center"><img src="image/5.jpeg" style="max-width:100%;"></p>
<p align="center"><img src="image/6.jpeg" style="max-width:100%;"></p>

---

## Training data

The dataset used to train the reference model contains ~680k examples
(docx, txt, question–answer json), convertible to the training format via
`training_system/prepare_cache.py`.

- **[Training data (Google Drive)](https://drive.google.com/drive/folders/19G5uGS5YM3rf42OefjM3KsXRyn0ZEshW?usp=sharing)**

---

## Repository structure

```
cevahir-ai/
├── model/                 # Unified inference engine (cevahir.py)
├── model_management/      # Model lifecycle (build, save/load, forward, health)
├── src/                   # Neural network (CevahirNeuralNetwork + modules)
├── tokenizer_management/  # BPE tokenizer (core, bpe, tokenization)
├── training_system/       # Training run/service layer (train.py, cache, v2/v3)
├── training_management/   # Training engine (loop, safety, curriculum, v2/v3)
├── cognitive_management/  # Cognitive layer (pipeline, memory, critic, tools)
├── chatting_management/   # Sessions, conversations, context
├── api/                   # Flask REST API (v3 routes, services, auth)
├── database/              # Persistence (repositories, models, unit of work)
├── data_loader_management/# Train-time data loading
├── data_processing/       # Offline data collection (Wikipedia, subtitles)
├── scripts/ · tests/      # Utilities and top-level tests
└── docs/architecture/     # Code-derived architecture documentation
```

---

## Documentation

- **System architecture:** [`docs/architecture/master-architecture.md`](docs/architecture/master-architecture.md)
- **Navigation / concept index:** [`docs/architecture/architecture-search-index.md`](docs/architecture/architecture-search-index.md)
- **Per-unit internals (code reality):** [`docs/architecture/code-reality/`](docs/architecture/code-reality/)
- **Development roadmap:** [`docs/architecture/development-roadmap.md`](docs/architecture/development-roadmap.md)

The `docs/_archive/` folder contains earlier documentation that no longer
reflects the codebase and should not be used as a reference.

---

## Status

Open source, under active development. The architecture is currently being
documented and prepared for a structured refactor/enhancement pass; see the
[development roadmap](docs/architecture/development-roadmap.md).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Contributions are welcome via
fork, feature branch and pull request.

## Author

Muhammed Yasin Yılmaz — [@myylogic](https://github.com/myylogic)
