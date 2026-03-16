# Cognitive Management Module

**Version:** V2 (Enterprise)
**Last Updated:** 2026-03-16
**Status:** Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [What It Does](#what-it-does)
4. [Core Data Types](#core-data-types)
5. [Cognitive Modes](#cognitive-modes)
6. [Quick Start](#quick-start)
7. [Dependencies](#dependencies)

---

## Overview

**Cognitive Management** is the meta-cognitive layer of the Cevahir Neural System. Beyond raw text generation, it:

- Classifies what kind of query the user sent
- Selects the best reasoning strategy (direct reply, chain-of-thought, tree-of-thoughts, etc.)
- Uses tools when needed (calculator, search, file)
- Critically evaluates the generated response and revises it if necessary
- Stores session history in vector-backed memory and automatically retrieves relevant past context

The system combines components inspired by academic LLM research (CoT, ToT, Self-Consistency, Self-Refine, Constitutional AI) with industrial enterprise patterns (Dependency Injection, Chain of Responsibility, Event Bus, Middleware).

### Design Philosophy

```
User Message
       │
       ▼
┌──────────────────────────────────────┐
│          CognitiveManager            │  ← Entry point (facade)
│  ┌────────────────────────────────┐  │
│  │      CognitiveOrchestrator     │  │  ← Internal coordinator
│  │                                │  │
│  │  Middleware → Pipeline         │  │  ← Processing chain
│  │     ↓                          │  │
│  │  PolicyRouter (select strategy)│  │
│  │     ↓                          │  │
│  │  DeliberationEngine (reason)   │  │
│  │     ↓                          │  │
│  │  MemoryService (manage memory) │  │
│  │     ↓                          │  │
│  │  CriticV2 (evaluate/revise)    │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
       │
       ▼
  CognitiveOutput
```

---

## Directory Structure

```
cognitive_management/
│
├── cognitive_manager.py          # Main entry — CognitiveManager class
├── cognitive_types.py            # All data types (dataclass + Literal)
├── config.py                     # CognitiveManagerConfig and sub-configs
├── exceptions.py                 # Exception hierarchy
│
├── utils/
│   ├── logging.py                # Logging helpers
│   └── timers.py                 # Timing helpers
│
└── v2/                           # Enterprise V2 architecture
    │
    ├── adapters/
    │   └── backend_adapter.py    # Adapter abstracting model backend
    │
    ├── components/               # Core cognitive components
    │   ├── critic_v2.py          # CriticV2 — Self-Refine loop
    │   ├── constitutional_critic.py  # Constitutional AI control layer
    │   ├── deliberation_engine_v2.py # CoT / ToT / debate generation
    │   ├── embedding_adapter.py  # Embedding provider adapter
    │   ├── memory_service_v2.py  # Session memory + vector RAG
    │   ├── policy_router_v2.py   # Strategy selection engine
    │   ├── rag_enhancer.py       # RAG context enricher
    │   ├── tool_executor_v2.py   # Tool executor
    │   ├── tool_policy_v2.py     # Tool usage policy
    │   ├── tree_of_thoughts.py   # ToT search algorithm
    │   │
    │   ├── fact_checkers/        # External fact verification
    │   │   ├── base.py           # FactChecker interface
    │   │   └── wikipedia_checker.py
    │   │
    │   └── vector_store/         # Vector store layer
    │       ├── base.py           # VectorStore interface
    │       ├── chroma_vector_store.py   # ChromaDB implementation
    │       └── memory_vector_store.py  # In-memory implementation
    │
    ├── config/
    │   ├── config_manager.py     # Runtime config manager
    │   └── constitutional_principles.py  # Default constitutional principles
    │
    ├── container/
    │   └── dependency_container.py  # DI Container — dependency injection
    │
    ├── core/
    │   └── orchestrator.py       # CognitiveOrchestrator — central coordinator
    │
    ├── events/
    │   ├── event_bus.py          # EventBus — async/sync event handling
    │   └── event_handlers.py     # Built-in event listeners
    │
    ├── interfaces/
    │   ├── backend_protocols.py  # Model backend Protocol definitions
    │   └── component_protocols.py  # Component Protocol definitions
    │
    ├── middleware/
    │   ├── metrics.py            # Metrics collection middleware
    │   ├── tracing.py            # Distributed tracing middleware
    │   └── validation.py         # Input validation middleware
    │
    ├── monitoring/               # AIOps monitoring subsystem
    │   ├── alerting.py           # Alert manager
    │   ├── anomaly_detector.py   # Anomaly detection
    │   ├── health_check.py       # Health check API
    │   ├── performance_monitor.py # Performance monitor
    │   ├── predictive_analytics.py # Predictive analytics
    │   └── trend_analyzer.py     # Trend analysis
    │
    ├── processing/
    │   ├── handlers.py           # Chain of Responsibility handlers
    │   ├── pipeline.py           # Synchronous processing pipeline
    │   ├── async_handlers.py     # Async handler variants
    │   └── async_pipeline.py     # Async pipeline
    │
    └── utils/
        ├── cache.py              # In-memory LRU cache
        ├── cache_warming.py      # Cache warming
        ├── claim_extraction.py   # Claim extraction (for fact-checking)
        ├── connection_pool.py    # Connection pool
        ├── context_pruning.py    # Context pruning
        ├── heuristics.py         # build_features — query feature extraction
        ├── performance_profiler.py # Performance profiler
        ├── request_batcher.py    # Batch request handler
        ├── selectors.py          # Self-Consistency candidate selector
        ├── semantic_cache.py     # Semantic similarity cache
        └── tracing.py            # Tracing helpers
```

---

## What It Does

### 1. Query Classification

Each user message is analyzed by `FeatureExtractionHandler` before processing:

- **QueryType**: `factual`, `reasoning`, `creative`, `conversational`, `math`, `code`, `unknown`
- **DomainType**: `math`, `science`, `law`, `medical`, `technology`, `history`, `creative`, `general`
- **Complexity score** (0.0–1.0): Based on message length, lexical diversity, question-mark density
- **Entropy estimate**: Model logit uncertainty — determines which reasoning mode to use

### 2. Strategy Selection (PolicyRouter)

Mode is chosen by entropy and length thresholds:

| Entropy | Length | Selected Mode | Academic Reference |
|---------|--------|---------------|---------------------|
| < 1.5 | — | `direct` | — |
| 1.5–2.5 | — | `think1` (CoT) | Wei et al. 2022 |
| 2.5–3.0 | > 200 tok. | `debate2` / `self_consistency` | Wang et al. 2022 |
| > 3.0 | > 300 tok. | `tot` | Yao et al. 2023 |

Temperature is also auto-tuned by domain:
- `math` / `code` → temperature ≈ 0.45–0.50
- `creative` → temperature ≈ 0.85

### 3. Deliberation (Reasoning)

`DeliberationEngineV2` produces inner reasoning steps according to the selected mode:
- **think1**: Single CoT step — think first, then answer
- **debate2**: Two parallel candidate generations, winner selected
- **tot**: Tree search — expand → evaluate → prune
- **self_consistency**: N samples (default 3), majority/hybrid selection

### 4. Memory Management (MemoryService)

Two-tier memory:
- **Session history** (short-term): Last N messages, pruned by token limit
- **Episodic vector memory** (long-term): Persisted to disk via ChromaDB

Every 6 turns a session summary is produced and stored in the vector store. New queries automatically retrieve relevant past context by cosine similarity (RAG).

### 5. Critique and Revision (CriticV2)

Generated responses go through:

1. **Constitutional AI check** — Compliance with constitutional principles
2. **Safety/risk check** — Sensitive area detection
3. **Task match check** — Does the answer address the question?
4. **Claim density** — If numerical/statistical claims exist, external verification is triggered
5. **External fact-checking** — Wikipedia (and optional Google/Wolfram)
6. **Self-Refine revision** — If revision is needed, the model regenerates

### 6. Tool Usage

`ToolPolicyV2` decides on tools via heuristic rules:

| Trigger | Tool |
|---------|------|
| "today", "current", "latest news" | `search` |
| "calculate", "sum", "ratio", "%" | `calculator` |
| "file", "read", "save" | `file` |

---

## Core Data Types

### Input / Output Types

```python
@dataclass
class CognitiveInput:
    user_message: str          # User message (required)
    system_prompt: str | None  # System behavior directive (optional)
    metadata: dict             # Extra signals, risk flags

@dataclass
class CognitiveOutput:
    text: str                        # Generated response
    used_mode: Mode                  # Cognitive mode used
    tool_used: str | None            # Tool name used
    revised_by_critic: bool          # Was critic revision applied?
    reasoning_chain: list[ReasoningTrace]  # Reasoning steps
    critic_passes: int               # Number of Self-Refine rounds
    critic_feedback: list[CriticFeedback] | None  # Structured feedback
    memory_hits: int                 # Number of memory items from RAG
    latency_ms: float                # Total processing time (ms)
    query_type: QueryType | None     # Detected query type
    domain: DomainType | None        # Detected domain
    self_consistency_result: SelfConsistencyResult | None
    context_sources: list[str]       # "vector", "history", etc.
    metadata: dict                   # Tracing data
```

### Session State

```python
@dataclass
class CognitiveState:
    history: list[dict]            # [{"role": ..., "content": ...}] list
    step: int                      # Current cognitive turn
    last_entropy: float | None     # Last uncertainty estimate
    last_mode: Mode | None         # Last mode used
    session_id: str                # 8-char UUID-based session ID
    turn_count: int                # Total user-assistant turn count
    query_type: QueryType | None
    domain: DomainType | None
    reasoning_traces: list[ReasoningTrace]
    metadata: dict
```

### Reasoning Trace

```python
@dataclass
class ReasoningTrace:
    step: int       # Step number
    content: str    # Content of this step
    score: float   # Quality score (0.0–1.0)
    source: str    # "cot" | "tot" | "debate" | "direct" | "react"

@dataclass
class CriticFeedback:
    aspect: str           # "coherence", "relevance", "safety", etc.
    score: float          # 0.0–1.0
    message: str          # Human-readable message
    needs_revision: bool
    constitutional: bool  # Constitutional AI violation?

@dataclass
class SelfConsistencyResult:
    candidates: list[str]   # N candidate responses
    selected: str           # Best selected response
    agreement_score: float  # Agreement between candidates (0.0–1.0)
    method: str             # "majority" | "score" | "hybrid"

@dataclass
class ThoughtCandidate:
    text: str          # Thought text
    score: float       # Evaluation score
    depth: int         # ToT tree depth
    path: list[str]    # Chain of thoughts leading to this point
```

---

## Cognitive Modes

| Mode | Description | Academic Reference |
|------|-------------|---------------------|
| `direct` | Single-pass fast generation — no reasoning step | — |
| `think1` | Chain-of-Thought — inner reasoning first, then answer | Wei et al. 2022 |
| `debate2` | Two candidate generations, winner selected | Wang et al. 2022 |
| `tot` | Tree of Thoughts — tree-based search | Yao et al. 2023 |
| `react` | Reason+Act — interleaved thought and action | Yao et al. 2022 |
| `self_consistency` | N samples, majority/hybrid voting | Wang et al. 2022 |

---

## Quick Start

### Basic Usage

```python
from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.config import CognitiveManagerConfig

# Start with default config
config = CognitiveManagerConfig()
manager = CognitiveManager(model_manager=model, config=config)

# Get response
output = manager.handle(
    user_message="What is Turkey's largest city?",
    system_prompt="You are Cevahir, a Turkish-speaking assistant.",
)

print(output.text)          # → "Istanbul is Turkey's largest city."
print(output.used_mode)     # → "direct"
print(output.latency_ms)    # → 124.5
```

### Custom Config

```python
from cognitive_management.config import (
    CognitiveManagerConfig, PolicyConfig, MemoryConfig
)

config = CognitiveManagerConfig()

# Enable Tree of Thoughts earlier
config.policy.entropy_gate_tot = 2.0
config.policy.tot_max_depth = 4

# Persist vector memory to disk
config.memory.vector_store_path = "./my_episodic_memory"
config.memory.enable_rag = True

config.validate()
manager = CognitiveManager(model_manager=model, config=config)
```

### Loading from Environment

```bash
export CM_CRITIC_ENABLED=true
export CM_TOOLS_ENABLE=true
export CM_DEBATE_ENABLED=false
export CM_MAX_NEW_TOKENS_LO=32
export CM_MAX_NEW_TOKENS_HI=512
```

```python
config = CognitiveManagerConfig.from_env()
```

### Accessing Session State

```python
# CognitiveManager holds state internally
output = manager.handle("Hello!")
output2 = manager.handle("What about Izmir?")  # Remembers previous context

state = manager.state
print(state.turn_count)   # → 2
print(state.session_id)   # → "a3f8b21c"
```

---

## Dependencies

### Required

| Package | Use |
|---------|-----|
| `torch` | Model inference backend |
| `dataclasses` | Type definitions |

### Optional

| Package | Use | Enabling Config |
|---------|-----|-----------------|
| `chromadb` | Episodic vector memory | `memory.vector_store_provider = "chroma"` |
| `sentence-transformers` | Embedding generation | `memory.embedding_provider = "sentence-transformers"` |
| `wikipedia` | External fact-checking | `critic.enable_wikipedia = True` |
| `openai` | OpenAI embedding/fact-check | `memory.embedding_provider = "openai"` |

### Installation

```bash
# Core
pip install chromadb sentence-transformers

# Wikipedia fact-checking
pip install wikipedia-api

# All optional features
pip install chromadb sentence-transformers wikipedia-api
```

---

For more information:
- [Architecture Details →](architecture/README-en.md)
- [API Reference →](api/README-en.md)
- [Usage Guides →](guides/README-en.md)
