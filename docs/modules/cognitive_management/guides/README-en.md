# Cognitive Management — Usage Guides

**Module:** `cognitive_management`
**Last Updated:** 2026-03-16

---

## Table of Contents

1. [Academic Foundations](#academic-foundations)
2. [Typical Usage Scenarios](#typical-usage-scenarios)
3. [Config Guide](#config-guide)
4. [Vector Memory and RAG Setup](#vector-memory-and-rag-setup)
5. [Writing Custom Components](#writing-custom-components)
6. [AIOps Monitoring Integration](#aiops-monitoring-integration)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Version History](#version-history)

---

## Academic Foundations

The cognitive strategies implemented in Cognitive Management are inspired directly by academic work:

### Chain-of-Thought (CoT) — `think1` mode

> Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.

The model is given a chance to think step by step before answering. An inner monologue starting with a phrase like "Let's think step by step: ..." is followed by the final response.

**When to use:**
- Medium query complexity (entropy 1.5–2.5)
- Multi-step logic questions
- "Why?", "How?", analytical questions

```python
# Force manually:
config.policy.entropy_gate_think = 0.0   # CoT on every query
config.policy.entropy_gate_debate = 999  # Never use debate
```

---

### Self-Consistency — `self_consistency` mode

> Wang, X. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.

N different reasoning paths are generated for the same question and the most reliable answer is chosen by majority/hybrid voting. Especially improves accuracy on math and logic questions.

**Parameters:**
- `self_consistency_n = 3` → 3 different reasoning paths
- `self_consistency_method = "hybrid"` → overlap + score combination

**Trade-off:**
- Accuracy increases but 3x model calls = 3x latency
- `n = 5` gives better results but 5x latency

---

### Tree of Thoughts (ToT) — `tot` mode

> Yao, S. et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS.

Response generation is modeled as tree search. At each node multiple thought candidates are generated, scored, and the best are selected to continue.

**Parameters:**
- `tot_max_depth = 3` → 3 levels of depth
- `tot_branching_factor = 3` → 3 branches per level
- `tot_top_k = 5` → Keep best 5 branches

**When to use:**
- Questions requiring multi-step planning
- Creative content generation
- High uncertainty (entropy ≥ 3.0)

---

### Self-Refine — inside CriticHandler

> Madaan, A. et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS.

The model first generates a response, then criticizes its own response and revises if needed. The number of rounds is limited by the `max_passes` parameter.

```python
config.critic.max_passes = 2     # At most 2 revision rounds
config.critic.strictness = 0.7   # Higher = more frequent revision
```

---

### Constitutional AI — ConstitutionalCritic

> Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic.

Model outputs are compared against a predefined set of "constitutional principles." Violations are flagged for revision.

Default principles (`constitutional_principles.py`):
- No content causing physical harm
- No leaking personal information
- No deceptive or misleading content
- No hate speech or discrimination
- No encouragement of illegal activity
- ... (16 principles total)

Adding custom principles:

```python
config.critic.custom_principles = (
    "Cevahir never gives definitive medical diagnosis.",
    "Cevahir never gives stock advice.",
)
```

---

### ReAct — `react` mode

> Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.

Thought (Reason) and Action (Act) steps are interleaved:
```
Thought: To answer this I need a web search.
Action: search("current weather Istanbul")
Observation: Today Istanbul 18°C, partly cloudy.
Thought: Now I can answer.
Answer: In Istanbul today 18°C and partly cloudy weather is expected.
```

---

## Typical Usage Scenarios

### Scenario 1: Chat Bot (Latency Priority)

Minimize latency, use `direct` mode most of the time:

```python
config = CognitiveManagerConfig()

# Raise thresholds — almost always direct
config.policy.entropy_gate_think = 2.8
config.policy.entropy_gate_debate = 99.0
config.policy.tot_enabled = False
config.policy.self_consistency_enabled = False

# Disable critic (source of latency)
config.critic.enabled = False

# Simple session memory, no disk write
config.memory.enable_vector_memory = False
config.memory.enable_rag = False

manager = CognitiveManager(model_manager=model, config=config)
```

---

### Scenario 2: Research Assistant (Accuracy Priority)

```python
config = CognitiveManagerConfig()

# CoT with low threshold, always on
config.policy.entropy_gate_think = 0.5
config.policy.entropy_gate_debate = 1.5
config.policy.self_consistency_enabled = True
config.policy.self_consistency_n = 5    # 5 samples

# Strict critic
config.critic.strictness = 0.9
config.critic.max_passes = 2
config.critic.enable_external_fact_checking = True
config.critic.enable_wikipedia = True

# Episodic memory on
config.memory.enable_vector_memory = True
config.memory.enable_rag = True
config.memory.rag_top_k = 5

config.validate()
manager = CognitiveManager(model_manager=model, config=config)
```

---

### Scenario 3: Math / Code Solver

```python
config = CognitiveManagerConfig()

# CoT required
config.policy.entropy_gate_think = 0.0
# Low temperature — consistent like a calculator
config.decoding_bounds.math_temperature = 0.3
config.decoding_bounds.code_temperature = 0.3

# Tool: calculator required (every query with numbers)
config.features.tool_rules["force_calc_if_numeric"] = True

# Memory: session only (no vector memory)
config.memory.enable_vector_memory = False

manager = CognitiveManager(model_manager=model, config=config)
output = manager.handle("153 × 47 = ?")
# expected: mode="think1", tool_used="calculator"
```

---

### Scenario 4: Creative Content Generation

```python
config = CognitiveManagerConfig()

# High temperature — diversity
config.decoding_bounds.creative_temperature = 0.92
config.default_decoding.temperature = 0.88

# ToT — search creative branches
config.policy.entropy_gate_tot = 1.5
config.policy.tot_branching_factor = 4
config.policy.tot_top_k = 8

# Light critic (fact-check not meaningful for creative content)
config.critic.checks = ("task_match",)
config.critic.enable_external_fact_checking = False

manager = CognitiveManager(model_manager=model, config=config)
output = manager.handle("Write a short story set on a winter night.")
# expected: mode="tot", used_mode="tot"
```

---

## Config Guide

### Entropy Gate Tuning

Model logit entropy is on a normalized 0–3 scale:

| Entropy Value | Meaning | Recommended Response |
|---------------|---------|----------------------|
| 0.0 – 1.0 | Very low uncertainty (clear query) | `direct` |
| 1.0 – 1.5 | Low uncertainty | `direct` or `think1` |
| 1.5 – 2.5 | Medium uncertainty | `think1` |
| 2.5 – 3.0 | High uncertainty | `debate2` / `self_consistency` |
| 3.0+ | Very high uncertainty | `tot` |

To see typical entropy for your system:

```python
output = manager.handle("test message")
print(manager.state.last_entropy)
```

### Token Generation Limits

```python
# Short responses (chat)
config.decoding_bounds.max_new_tokens_bounds = (16, 128)
config.default_decoding.max_new_tokens = 64

# Long responses (article, report)
config.decoding_bounds.max_new_tokens_bounds = (64, 1024)
config.default_decoding.max_new_tokens = 512
```

### Repetition Penalty

`repetition_penalty > 1.0` suppresses repeating phrases:

```python
config.default_decoding.repetition_penalty = 1.2   # Stronger suppression
config.default_decoding.repetition_penalty = 1.05  # Light (for creative content)
```

---

## Vector Memory and RAG Setup

### ChromaDB (Built-in, Recommended)

```bash
pip install chromadb sentence-transformers
```

```python
config.memory.enable_vector_memory = True
config.memory.embedding_provider = "sentence-transformers"
config.memory.embedding_model = None           # None = all-MiniLM-L6-v2 (default)
config.memory.vector_store_provider = "chroma"
config.memory.vector_store_path = "./memory/episodic_store"  # Persist to disk
config.memory.enable_rag = True
config.memory.rag_top_k = 3
config.memory.rag_score_threshold = 0.7
```

### In-Memory (Development / Testing)

```python
config.memory.vector_store_provider = "memory"
config.memory.vector_store_path = None
# All memory lost when session ends
```

### How RAG Works

```
User: "What about the topics we discussed in the previous session?"
                |
                v
EmbeddingAdapter.embed(user_message)
                |
                v
VectorStore.similarity_search(embed, top_k=3, threshold=0.7)
                |
                v
Similar past session chunks:
    - "Last week we talked about Cevahir training..."
    - "About tokenizer configuration..."
                |
                v
ContextBuildingHandler --> add to [MEMORY] section
                |
                v
Model gives more consistent answer with past context
```

### Memory Consolidation (Every 6 Turns)

```python
# session_summary_every = 6 (default)
# Runs automatically every 6 turns:

def _consolidate_memory():
    summary = model.generate(
        f"Summarize this session: {history[-6:]}"
    )
    embedding = embed_adapter.embed(summary)
    vector_store.add(
        text=summary,
        embedding=embedding,
        metadata={"session_id": state.session_id, "turn": state.turn_count}
    )
    # Old history is shortened — summary is in vector store
```

---

## Writing Custom Components

### Custom PolicyRouter

```python
from cognitive_management.v2.interfaces.component_protocols import PolicyRouter
from cognitive_management.cognitive_types import PolicyOutput, DecodingConfig, CognitiveState

class MyDomainPolicyRouter:
    """Custom router that always uses ToT for medical queries."""

    def route(self, features: dict, state: CognitiveState) -> PolicyOutput:
        domain = features.get("domain", "general")

        if domain == "medical":
            return PolicyOutput(
                mode="tot",
                tool="none",
                decoding=DecodingConfig(temperature=0.3, max_new_tokens=512),
                inner_steps=9,
                query_type=features.get("query_type"),
                domain=domain,
            )
        # Default
        return PolicyOutput(
            mode="direct",
            tool="none",
            decoding=DecodingConfig(),
        )
```

### Custom FactChecker

```python
from cognitive_management.v2.components.fact_checkers.base import BaseFactChecker

class MyCustomChecker(BaseFactChecker):
    def verify(self, claim: str) -> tuple[bool, float]:
        """
        Returns:
            (verified: bool, confidence: float 0.0-1.0)
        """
        # Your verification logic
        result = my_api.check(claim)
        return result.is_true, result.confidence
```

### Custom VectorStore

```python
from cognitive_management.v2.components.vector_store.base import BaseVectorStore

class MyVectorStore(BaseVectorStore):
    def add(self, text: str, embedding: list[float], metadata: dict) -> str:
        # Return ID
        ...

    def search(self, embedding: list[float], top_k: int, threshold: float) -> list[dict]:
        # Return [{"text": ..., "score": ..., "metadata": ...}]
        ...

    def delete(self, doc_id: str) -> bool:
        ...
```

---

## AIOps Monitoring Integration

### Basic Monitoring

```python
from cognitive_management.v2.monitoring.performance_monitor import PerformanceMonitor
from cognitive_management.v2.monitoring.health_check import HealthCheck

perf_mon = PerformanceMonitor()

# Attach to orchestrator
orchestrator = CognitiveOrchestrator(
    ...,
    performance_monitor=perf_mon,
)

# Read metrics
metrics = perf_mon.get_metrics()
print(f"Avg latency: {metrics['avg_latency_ms']:.1f} ms")
print(f"Requests/min: {metrics['requests_per_minute']:.1f}")
print(f"p95 latency: {metrics['p95_latency_ms']:.1f} ms")
```

### Anomaly Detection

```python
from cognitive_management.v2.monitoring.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(sensitivity="high")

# Check after each request (automatic when enable_auto_anomaly_check=True)
anomalies = detector.check(metrics)
if anomalies:
    for anomaly in anomalies:
        print(f"Anomaly: {anomaly.type} — {anomaly.description}")
```

### Health Check

```python
from cognitive_management.v2.monitoring.health_check import HealthCheck

health = HealthCheck(backend=backend, memory_service=memory_service)
status = health.check()

print(status.overall)         # "healthy" | "degraded" | "unhealthy"
print(status.backend_ok)      # True/False
print(status.vector_store_ok) # True/False
print(status.latency_ok)      # True/False
```

### Alerting

```python
from cognitive_management.v2.monitoring.alerting import AlertingManager

alerting = AlertingManager()
alerting.set_threshold("avg_latency_ms", max_value=2000)  # Alert above 2 sec
alerting.set_threshold("error_rate", max_value=0.05)      # Alert at 5% error rate

# Register callback
alerting.on_alert(lambda alert: print(f"ALERT: {alert.message}"))
```

---

## Performance Tuning

### Semantic Cache

Serve semantically similar queries (cosine ≥ 0.85) from cache:

```python
config.runtime.enable_semantic_cache = True
config.runtime.semantic_cache_threshold = 0.85  # High similarity required
config.runtime.semantic_cache_max_size = 1000   # Max cache entries

# "What is Istanbul's population?" and "How many people in Istanbul?"
# → Similarity 0.88 → cache hit → direct response, no model call
```

### Cache Warming

Pre-fill cache with frequent questions at startup:

```python
config.runtime.enable_cache_warming = True

# Warmup list (defined in cache_warming.py):
warmup_queries = [
    "Hello, how can I help?",
    "What is Cevahir?",
    "Can you help in Turkish?",
]
```

### Request Batching

Batch request processing under heavy load:

```python
config.runtime.enable_batch_processing = True
config.runtime.batch_size = 10
config.runtime.batch_timeout = 1.0  # Seconds; group processed on timeout
```

### Profiling

See which handler takes the most time:

```python
config.runtime.enable_profiling = True

output = manager.handle("test")
profile = output.metadata.get("handler_times", {})
for handler, ms in sorted(profile.items(), key=lambda x: -x[1]):
    print(f"  {handler}: {ms:.1f} ms")
# Example output:
#   GenerationHandler: 1240.3 ms
#   DeliberationHandler: 312.1 ms
#   CriticHandler: 89.5 ms
#   MemoryRetrievalHandler: 45.2 ms
```

---

## Troubleshooting

### Issue: Responses Too Slow

**Checklist:**
1. Check `used_mode` — may be `tot` or `self_consistency`
2. What is `critic.max_passes`? 2+ increases latency
3. Is `enable_external_fact_checking = True`? Wikipedia call ~500ms
4. What is `rag_top_k`? Higher increases embedding cost

**Quick fix:**
```python
config.policy.entropy_gate_think = 2.0     # CoT less often
config.policy.tot_enabled = False          # Disable ToT
config.critic.max_passes = 1               # Single revision round
config.critic.enable_external_fact_checking = False
config.memory.rag_top_k = 2
```

---

### Issue: Too Many Self-Refine Revisions

If critic often returns `needs_revision=True`:

```python
config.critic.strictness = 0.3        # More tolerant
config.critic.checks = ("task_match",) # Only check task match
config.critic.enable_constitutional_ai = False  # Turn off
config.safety.high_risk_threshold = 0.9  # Raise risk threshold
```

---

### Issue: Vector Memory Not Working

```python
# Dependencies installed?
try:
    import chromadb
    import sentence_transformers
    print("Dependencies OK")
except ImportError as e:
    print(f"Missing dependency: {e}")
    # Fix: pip install chromadb sentence-transformers
```

```python
# Is path writable?
import os
path = "./memory/episodic_store"
os.makedirs(path, exist_ok=True)
assert os.access(path, os.W_OK), "No write permission!"
```

---

### Issue: Tool Not Used

```python
# Ensure tool is in allow list
config.tools.allow = ("calculator", "search", "file")
config.tools.enable_tools = True

# Check trigger keywords
features = config.features.tool_rules
print(features["needs_calc_or_parse_triggers"])
# ("calculate", "sum", "count", "ratio")
# User message should contain these
```

---

### Issue: Session Memory Overflow

```python
# Increase token limit
config.memory.max_history_tokens = 4096

# Summarize more often
config.memory.session_summary_every = 4   # Default 6, do more often

# Salient pruning on?
config.memory.enable_salient_pruning = True
config.memory.salient_topk = 6   # Messages to keep in summary
```

---

## Version History

### V2 (Current — 2025)

**New features:**
- CognitiveOrchestrator: Full enterprise integration
- ProcessingPipeline: Chain of Responsibility architecture
- EventBus: Pub/Sub event handling
- Middleware chain: Validation, Metrics, Tracing
- DependencyContainer: Dependency Injection
- MemoryServiceV2: Two-tier memory (session + vector)
- CriticV2 + ConstitutionalCritic: Self-Refine + Constitutional AI
- SelfConsistencyHandler: Wang et al. 2022
- TreeOfThoughts: Yao et al. 2023
- AIOps monitoring: PerformanceMonitor, AnomalyDetector, TrendAnalyzer
- Semantic cache: Similarity-based cache
- Request batching: Batch request processing
- Async pipeline: AsyncProcessingPipeline

**Bug fixes:**
- Session memory overflow issues
- Vector memory consolidation inconsistency
- Critic re-entry loop

### V1 (Archive — 2024)

- Basic CognitiveManager structure
- PolicyRouter (entropy-based)
- MemoryService (session history only)
- Simple Critic (heuristic checks)
- Tool executor (calculator, search, file)
- CoT / debate2 support
