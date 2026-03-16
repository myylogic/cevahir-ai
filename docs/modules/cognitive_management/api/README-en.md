# Cognitive Management — API Reference

**Module:** `cognitive_management`
**Last Updated:** 2026-03-16

---

## Table of Contents

1. [CognitiveManagerConfig](#cognitivemanagerconfig)
2. [Sub-Configuration Classes](#sub-configuration-classes)
3. [Data Types](#data-types)
4. [CognitiveManager API](#cognitivemanager-api)
5. [Exception Hierarchy](#exception-hierarchy)
6. [Environment Variables](#environment-variables)

---

## CognitiveManagerConfig

Main configuration class. Holds settings for all sub-modules.

```python
from cognitive_management.config import CognitiveManagerConfig

config = CognitiveManagerConfig()
config.validate()   # Validation check
d = config.to_dict()  # Serialization
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `critic` | `CriticConfig` | Critique and fact-checking settings |
| `memory` | `MemoryConfig` | Memory and RAG settings |
| `policy` | `PolicyConfig` | Strategy selection thresholds |
| `tools` | `ToolsConfig` | Tool usage policy |
| `decoding_bounds` | `DecodingBounds` | Generation parameter bounds |
| `safety` | `SafetyConfig` | Safety and risk settings |
| `features` | `FeatureRules` | Heuristic rule dictionaries |
| `runtime` | `RuntimeToggles` | Runtime on/off toggles |
| `default_decoding` | `DecodingConfig` | Default generation parameters |
| `default_system_prompt` | `str` | Default system behavior text |

### Methods

```python
# Update fields from external dict (recursive merge)
config.override_with({
    "policy": {"entropy_gate_tot": 2.5},
    "critic": {"strictness": 0.8},
})

# Serialize
d = config.to_dict()

# Validate (raises ValueError on invalid values)
config.validate()

# Load from environment variables
config = CognitiveManagerConfig.from_env()
config = CognitiveManagerConfig.from_env(env={"CM_CRITIC_ENABLED": "true"})
```

---

## Sub-Configuration Classes

### CriticConfig

```python
@dataclass
class CriticConfig:
    enabled: bool = True
    checks: tuple[str, ...] = ("task_match", "safety", "claim_density")
    strictness: float = 0.5          # 0.0-1.0 (higher = stricter revision)
    max_passes: int = 1              # Maximum Self-Refine rounds

    # External fact-checking
    enable_external_fact_checking: bool = True
    fact_checking_providers: tuple = ("wikipedia",)
    enable_wikipedia: bool = True
    wikipedia_max_results: int = 3

    # Google Fact Check (optional)
    enable_google_fact_check: bool = False
    google_fact_check_api_key: str | None = None

    # Wolfram Alpha (optional)
    enable_wolfram: bool = False
    wolfram_api_key: str | None = None

    # LLM-based verification
    enable_llm_fact_verification: bool = True

    # Constitutional AI
    enable_constitutional_ai: bool = True
    constitutional_strictness: float = 0.7   # 0.0-1.0
    custom_principles: tuple[str, ...] = ()  # Additional constitutional principles

    # Thresholds
    fact_checking_score_threshold: float = 0.7
    claim_extraction_min_confidence: float = 0.5
```

**Valid values for `checks`:**

| Value | Description |
|-------|-------------|
| `"task_match"` | Checks query–response alignment |
| `"safety"` | Safety keyword check |
| `"claim_density"` | Triggers external verification when claim density is high |

---

### MemoryConfig

```python
@dataclass
class MemoryConfig:
    session_summary_every: int = 6       # Summarize every N turns
    salient_topk: int = 8                 # Number of salient messages in summary
    max_history_tokens: int = 3072       # Session history token limit
    enable_session_summary: bool = True
    enable_salient_pruning: bool = True

    # Vector memory and RAG
    enable_vector_memory: bool = True
    embedding_provider: str = "sentence-transformers"  # "sentence-transformers" | "openai" | "none"
    embedding_model: str | None = None   # None = all-MiniLM-L6-v2
    openai_api_key: str | None = None

    vector_store_provider: str = "chroma"  # "chroma" | "pinecone" | "weaviate" | "qdrant" | "milvus" | "memory"
    vector_store_path: str | None = "./memory/episodic_store"
    vector_store_collection_name: str = "cognitive_memory"

    # Pinecone (optional)
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str | None = None

    # RAG
    enable_rag: bool = True
    rag_top_k: int = 3                   # Number of memory items to retrieve
    rag_score_threshold: float = 0.7     # Minimum cosine similarity score

    # Vector search
    vector_search_top_k: int = 5
    hybrid_search_alpha: float = 0.7     # 0.0 = keyword only, 1.0 = vector only
```

**Special values for `vector_store_path`:**

| Value | Behavior |
|-------|----------|
| `"./memory/episodic_store"` | Persisted to disk (remembers across sessions) |
| `None` | In-memory (lost when session ends) |

---

### PolicyConfig

```python
@dataclass
class PolicyConfig:
    debate_enabled: bool = True

    # Entropy thresholds (0–3 normalized scale)
    entropy_gate_think: float = 1.5   # Above this → CoT
    entropy_gate_debate: float = 2.5  # Above this → debate/self_consistency
    entropy_gate_tot: float = 3.0     # Above this → ToT

    # Length thresholds (tokens)
    length_gate_debate: int = 200
    length_gate_tot: int = 300

    allow_inner_steps: bool = True

    # Tree of Thoughts
    tot_enabled: bool = True
    tot_max_depth: int = 3
    tot_branching_factor: int = 3
    tot_top_k: int = 5

    # Self-Consistency (Wang et al. 2022)
    self_consistency_enabled: bool = True
    self_consistency_n: int = 3           # Number of samples (3–5 recommended)
    self_consistency_method: str = "hybrid"  # "majority" | "score" | "hybrid"
```

**Options for `self_consistency_method`:**

| Value | Description |
|-------|-------------|
| `"majority"` | Majority vote by bigram overlap |
| `"score"` | Selection by model quality score |
| `"hybrid"` | Overlap + score combination (recommended) |

---

### ToolsConfig

```python
@dataclass
class ToolsConfig:
    allow: tuple[str, ...] = ("calculator", "search", "file")
    policy: str = "heuristic"   # "heuristic" | future: "learned"
    enable_tools: bool = True
```

---

### DecodingBounds

```python
@dataclass
class DecodingBounds:
    max_new_tokens_bounds: tuple[int, int] = (32, 512)
    temperature_bounds: tuple[float, float] = (0.40, 0.90)
    top_p_default: float = 0.90
    repetition_penalty_default: float = 1.15

    # Domain-based temperature
    math_temperature: float = 0.45     # Deterministic output
    creative_temperature: float = 0.85  # Diversity
    code_temperature: float = 0.50     # Consistent code
```

---

### SafetyConfig

```python
@dataclass
class SafetyConfig:
    risk_keywords_sensitive: tuple[str, ...] = ("politics", "medical", "legal")
    claim_markers: tuple[str, ...] = ("statistic", "ratio", "evidence", "%")
    raise_on_high_risk: bool = False  # If True, critic required on high risk
    high_risk_threshold: float = 0.6
```

---

### RuntimeToggles

```python
@dataclass
class RuntimeToggles:
    enable_logging: bool = True
    enable_telemetry: bool = True
    enable_internal_thought_logging: bool = False  # Privacy: do not log inner voice
    fail_fast: bool = False   # If True, raise on error quickly

    # AIOps
    aiops_sensitivity: str = "medium"     # "low" | "medium" | "high"
    enable_auto_anomaly_check: bool = True

    # Performance tuning
    enable_profiling: bool = False
    enable_semantic_cache: bool = False
    semantic_cache_threshold: float = 0.85
    semantic_cache_max_size: int = 1000
    enable_cache_warming: bool = False

    # Advanced
    enable_connection_pool: bool = False
    connection_pool_size: int = 10
    enable_batch_processing: bool = False
    batch_size: int = 10
    batch_timeout: float = 1.0
```

---

### DecodingConfig

```python
@dataclass
class DecodingConfig:
    max_new_tokens: int = 256
    min_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = 0       # 0 = disabled
    repetition_penalty: float = 1.1
```

---

## Data Types

### Literal Types

```python
# Cognitive mode
Mode = Literal["direct", "think1", "debate2", "tot", "react", "self_consistency"]

# Tool usage decision
ToolDecision = Literal["none", "maybe", "must"]

# Query type
QueryType = Literal[
    "factual",        # "What is the capital of Turkey?"
    "reasoning",      # "Why...?", "How does... affect...?"
    "creative",       # Story, poem, script
    "conversational", # Greeting, chit-chat
    "math",           # Numeric computation, equation
    "code",           # Code writing, debug
    "unknown",
]

# Domain type
DomainType = Literal[
    "math", "science", "law", "medical",
    "technology", "history", "creative", "general",
]
```

### PolicyOutput

```python
@dataclass
class PolicyOutput:
    mode: Mode                    # Selected cognitive mode
    tool: ToolDecision            # Tool usage recommendation
    decoding: DecodingConfig      # Generation parameters
    inner_steps: int = 0          # Number of reasoning steps
    query_type: QueryType | None = None
    domain: DomainType | None = None
    complexity: float = 0.0       # 0.0-1.0 complexity score
```

### ThoughtCandidate

```python
@dataclass
class ThoughtCandidate:
    text: str          # Thought text
    score: float       # Evaluation score (higher = better)
    depth: int = 0     # ToT tree depth (root = 0)
    path: list[str]    # Previous thoughts leading to this point
```

---

## CognitiveManager API

`CognitiveManager` is the main external interface of the module.

### Initialization

```python
from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.config import CognitiveManagerConfig

manager = CognitiveManager(
    model_manager=model,    # ModelManager / CevahirModel instance
    config=CognitiveManagerConfig(),
)
```

### handle()

```python
def handle(
    self,
    user_message: str,
    system_prompt: str | None = None,
    metadata: dict | None = None,
    decoding: DecodingConfig | None = None,
) -> CognitiveOutput:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_message` | `str` | User message (required) |
| `system_prompt` | `str \| None` | System behavior directive (None = config.default_system_prompt) |
| `metadata` | `dict \| None` | Extra signals, precomputed query_type, etc. |
| `decoding` | `DecodingConfig \| None` | Override generation parameters manually |

**Returns:** `CognitiveOutput`

```python
@dataclass
class CognitiveOutput:
    text: str                              # Generated response
    used_mode: Mode                        # Cognitive mode used
    tool_used: str | None                  # Tool used (None = no tool)
    revised_by_critic: bool                # Was critic revision applied?
    reasoning_chain: list[ReasoningTrace]  # Reasoning steps
    critic_passes: int                     # Self-Refine round count
    critic_feedback: list[CriticFeedback] | None
    memory_hits: int                       # Number of items from RAG
    latency_ms: float                      # Total time (ms)
    query_type: QueryType | None
    domain: DomainType | None
    self_consistency_result: SelfConsistencyResult | None
    context_sources: list[str]             # "vector", "history", "tool", etc.
    metadata: dict                         # Tracing data
```

### Session State

```python
# Access active session state
state: CognitiveState = manager.state

# Reset session
manager.reset_session()

# Session ID
print(state.session_id)   # "a3f8b21c"
print(state.turn_count)   # 5
```

### reset_session()

```python
def reset_session(self) -> None:
    """Reset session history and state. Vector memory is preserved."""
```

---

## Exception Hierarchy

```
CognitiveError (base)
    |
    +-- CognitiveConfigError         # Invalid config value
    |
    +-- CognitiveInputError          # Invalid or empty user input
    |
    +-- CognitiveBackendError        # Model backend access error
    |       |
    |       +-- CognitiveTimeoutError   # Backend timeout
    |
    +-- CognitiveMemoryError         # Vector store access error
    |
    +-- CognitiveToolError           # Tool execution error
    |
    +-- CognitiveCriticError        # Critic evaluation error
    |
    +-- CognitivePipelineError      # Pipeline processing error
```

### Usage

```python
from cognitive_management.exceptions import (
    CognitiveError,
    CognitiveBackendError,
    CognitiveInputError,
)

try:
    output = manager.handle(user_message)
except CognitiveInputError as e:
    print(f"Invalid input: {e}")
except CognitiveBackendError as e:
    print(f"Model error: {e}")
except CognitiveError as e:
    print(f"General cognitive error: {e}")
```

---

## Environment Variables

`CognitiveManagerConfig.from_env()` reads the following environment variables:

| Variable | Type | Maps to | Example |
|----------|------|---------|---------|
| `CM_CRITIC_ENABLED` | bool | `critic.enabled` | `true` |
| `CM_TOOLS_ENABLE` | bool | `tools.enable_tools` | `false` |
| `CM_DEBATE_ENABLED` | bool | `policy.debate_enabled` | `true` |
| `CM_MAX_NEW_TOKENS_LO` | int | `decoding_bounds.max_new_tokens_bounds[0]` | `32` |
| `CM_MAX_NEW_TOKENS_HI` | int | `decoding_bounds.max_new_tokens_bounds[1]` | `512` |
| `CM_TEMP_LO` | float | `decoding_bounds.temperature_bounds[0]` | `0.4` |
| `CM_TEMP_HI` | float | `decoding_bounds.temperature_bounds[1]` | `0.9` |

**Valid bool values:** `1`, `true`, `yes`, `on` → `True`; `0`, `false`, `no`, `off` → `False`

---

## DEFAULT_RULES Dictionary

Ready-made rule set for tool policy and mode selection:

```python
from cognitive_management.config import DEFAULT_RULES

# Structure:
DEFAULT_RULES = {
    "tool_policy": {
        "priority": ["search", "calculator", "file"],
        "force_search_if_recent": True,
        "force_calc_if_numeric": True,
    },
    "mode_selection": {
        "prefer_direct_for_short_inputs": True,
        "debate_on_long_or_uncertain": True,
    },
    "critic_policy": {
        "require_on_high_risk": True,
        "light_on_low_risk": True,
        "revise_once_max": True,
    },
}
```

---

## FeatureRules (Heuristic Rule Dictionaries)

```python
@dataclass
class FeatureRules:
    tool_rules: dict = {
        "needs_recent_info_triggers": ("today", "current", "latest news", "newest"),
        "needs_calc_or_parse_triggers": ("calculate", "sum", "count", "ratio"),
        "default_decision": "none",  # "none" | "maybe" | "must"
    }
    risk_rules: dict = {
        "base_risk": 0.0,
        "claim_bonus": 0.5,      # Risk increase when claim detected
        "sensitive_bonus": 0.5,  # Risk increase when sensitive area detected
        "max_risk": 1.0,
    }
```
