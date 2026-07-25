# Cognitive Management — Architecture

**Module:** `cognitive_management`
**Last Updated:** 2026-03-16

---

## Table of Contents

1. [General Architecture Approach](#general-architecture-approach)
2. [Processing Pipeline](#processing-pipeline)
3. [Chain of Responsibility Handlers](#chain-of-responsibility-handlers)
4. [CognitiveOrchestrator](#cognitiveorchestrator)
5. [Middleware Layer](#middleware-layer)
6. [Event Bus](#event-bus)
7. [Dependency Injection Container](#dependency-injection-container)
8. [Interfaces / Protocol Definitions](#interfaces--protocol-definitions)
9. [Async Pipeline](#async-pipeline)
10. [Component Details](#component-details)

---

## General Architecture Approach

Cognitive Management V2 applies multiple design patterns in a layered manner:

```
+-----------------------------------------------------------------+
|                     CognitiveManager                           |
|                  (Facade Pattern - Entry Point)                |
+--------------------------+--------------------------------------+
                           |
                           v
+-----------------------------------------------------------------+
|                  CognitiveOrchestrator                         |
|           (Orchestrator Pattern - Internal Coordination)       |
|                                                                 |
|  +----------------+  +-----------------+  +-----------------+  |
|  |  Middleware    |  |  EventBus       |  |  PerformanceMon |  |
|  |  Chain         |  |  (Pub/Sub)      |  |  itor           |  |
|  +-------+--------+  +-----------------+  +-----------------+  |
|          |                                                      |
|          v                                                      |
|  +-------------------------------------------------------+      |
|  |              ProcessingPipeline                       |      |
|  |         (Chain of Responsibility Pattern)             |      |
|  |                                                       |      |
|  |  Handler 1 --> Handler 2 --> ... --> Handler N        |      |
|  +-------------------------------------------------------+      |
+-----------------------------------------------------------------+
```

### Applied Design Patterns

| Pattern | Class | Purpose |
|---------|-------|---------|
| Facade | `CognitiveManager` | Single entry point, hides internal complexity |
| Orchestrator | `CognitiveOrchestrator` | Coordinates components |
| Chain of Responsibility | `ProcessingPipeline` | Each handler has a single responsibility |
| Strategy | `PolicyRouterV2` | Runtime algorithm selection |
| Observer / Pub-Sub | `EventBus` | Loosely coupled event management |
| Dependency Injection | `DependencyContainer` | Injects dependencies from outside |
| Adapter | `BackendAdapter` | Abstracts the model backend |
| Repository | `VectorStore` | Abstracts memory storage/retrieval |

### SOLID Principles

- **S** (Single Responsibility): Each handler performs exactly one task
- **O** (Open/Closed): New handlers can be added without modifying existing ones
- **L** (Liskov Substitution): Conforming to Protocol definitions such as `IMemoryService`, `ICritic` is sufficient
- **I** (Interface Segregation): Separate Protocols such as `FullModelBackend`, `MinimalBackend`
- **D** (Dependency Inversion): `CognitiveOrchestrator` depends on Protocols, not concrete classes

---

## Processing Pipeline

### Full Flow Diagram

```
User Message (user_message: str)
          |
          v
  [ValidationMW]  <- Input validation middleware
          |
  [MetricsMW]     <- Start metric counters
          |
  [TracingMW]     <- Start distributed tracing span
          |
          v
  =========================================
         ProcessingPipeline
  =========================================

  1. FeatureExtractionHandler
     - Classify QueryType
     - Determine DomainType
     - Compute complexity score
     - Model entropy estimate (0-3 normalized)

  2. MemoryRetrievalHandler
     - Retrieve session history (pruned to token limit)
     - RAG: Query VectorStore for relevant memory chunks

  3. ToolHandler
     - Is a tool needed? (heuristic rule matching)
     - If yes, execute tool (calculator / search / file)

  4. PolicyRoutingHandler
     - Entropy + length --> mode selection
     - Domain --> adjust temperature
     - Build DecodingConfig

  5. SelfConsistencyHandler
     - Active only when mode == "self_consistency"
     - Generate N samples (default N=3)
     - Select by majority / hybrid voting

  6. ContextBuildingHandler
     - Build structured prompt:
       [SYSTEM | MEMORY | COT | USER]

  7. DeliberationHandler
     - Generate CoT / ToT / debate / react steps
     - Build ThoughtCandidate list

  8. GenerationHandler
     - Final model call (backend.generate)
     - Record ReasoningTrace

  9. CriticHandler
     - Constitutional AI check
     - Risk / safety check
     - Fact verification (Wikipedia etc.)
     - Self-Refine revision

  10. MemoryUpdateHandler
      - Write to history
      - Every 6 turns: summary + vector store write
  =========================================
          |
          v
  CognitiveOutput (text, mode, metadata, ...)
```

### ProcessingContext

Each handler receives and mutates a `ProcessingContext` object:

```python
@dataclass
class ProcessingContext:
    cognitive_input: CognitiveInput    # Original user input
    cognitive_state: CognitiveState    # Session state
    policy_output: PolicyOutput | None # Policy decision
    context_messages: list[dict]       # Message list for prompt
    raw_response: str                  # Raw model output
    final_response: str                # Processed final response
    tool_result: str | None            # Tool output
    metadata: dict                     # Metadata shared between handlers
```

---

## Chain of Responsibility Handlers

Each handler inherits from `BaseProcessingHandler` and implements the `handle(ctx)` method:

```python
class BaseProcessingHandler:
    def __init__(self):
        self._next: BaseProcessingHandler | None = None

    def set_next(self, handler) -> BaseProcessingHandler:
        self._next = handler
        return handler

    def handle(self, ctx: ProcessingContext) -> ProcessingContext:
        # Subclass implements; calls self._next.handle(ctx) at the end
        raise NotImplementedError
```

### Handler Descriptions

#### 1. FeatureExtractionHandler
- Extracts `QueryType`, `DomainType`, and complexity score from the user message
- Computes logit entropy from the model backend
- Populates `ctx.metadata["features"]`
- Initialises `ReasoningTrace(source="direct")`

#### 2. MemoryRetrievalHandler
- Retrieves session history from `MemoryServiceV2` (pruned to token limit)
- If `enable_rag=True`: embed the user message → vector similarity query → retrieve relevant memory chunks
- Populates `ctx.metadata["memory_context"]` and `ctx.metadata["memory_hits"]`

#### 3. ToolHandler
- Tool-use decision via `ToolPolicyV2`: `none` | `maybe` | `must`
- If `must`, execute the tool via `ToolExecutorV2`
- Result is written to `ctx.tool_result`

#### 4. PolicyRoutingHandler
- Calls `PolicyRouterV2.route()`
- Input: entropy, length, query_type, domain
- Output: `PolicyOutput(mode, decoding, inner_steps)`
- Updates `ctx.policy_output`

#### 5. SelfConsistencyHandler
- Active only when `mode == "self_consistency"`
- Calls `backend.generate()` N times (default N=3)
- Measures candidate similarity with `_bigram_overlap()`
- `method = "majority"` → by word overlap; `"score"` → by quality score; `"hybrid"` → both combined
- Populates `ctx.metadata["self_consistency_result"]`

#### 6. ContextBuildingHandler
- Builds the final prompt message list:
  - `[SYSTEM]` → system_prompt
  - `[MEMORY]` → vector memory + session summary
  - `[COT]` → internal reasoning steps (for think1/debate2/tot)
  - `[USER]` → user message
- Populates `ctx.context_messages`

#### 7. DeliberationHandler
- Generates reasoning steps via `DeliberationEngineV2`
- `think1`: Single CoT step — "Let's think step by step: ..."
- `debate2`: Generate responses from two different perspectives, select the best
- `tot`: BFS/beam search via the `TreeOfThoughts` component
- `react`: Thought → Action → Observation loop
- Populates `ctx.metadata["reasoning_steps"]`

#### 8. GenerationHandler
- Generates the final response via `backend.generate(messages, decoding)`
- Appends the final step to the `ReasoningTrace` list
- On error, falls back to direct generation

#### 9. CriticHandler

```
Incoming response text
    |
    v
ConstitutionalCritic.check()
    - Default 16 principles (violence, harmful content, bias, etc.)
    - If violation found --> CriticFeedback(constitutional=True)
    |
    v
Risk / Safety check
    - risk_keywords_sensitive present in message? -> risk_score increases
    - claim_markers present in response? -> fact-check triggered
    |
    v
Task-match score
    - Word overlap between query and response
    - Low overlap --> needs_revision = True
    |
    v
FactChecker.verify()  (if enable_external_fact_checking=True)
    - WikipediaChecker: Is the claim valid on Wikipedia?
    - LLM fact verifier: Model queries its own output
    |
    v
Revision needed?
    Yes --> Build revision prompt --> backend.generate()
             critic_passes += 1 (max_passes=1 default)
    No  --> Response approved
```

#### 10. MemoryUpdateHandler
- Writes to history via `MemoryServiceV2.add_turn(role, content)`
- Every 6 turns: `_generate_session_summary()` → summary text → write to vector store
- Updates `CognitiveState.query_type` and `CognitiveState.domain`

---

## CognitiveOrchestrator

`CognitiveOrchestrator` is the central coordinator of all components. It constructs the pipeline, chains the middleware, and publishes events.

### Configuration

```python
orchestrator = CognitiveOrchestrator(
    backend=backend_adapter,           # FullModelBackend implementation
    policy_router=policy_router,       # IPolicyRouter implementation
    memory_service=memory_service,     # IMemoryService implementation
    critic=critic,                     # ICritic implementation
    deliberation_engine=deliberation,  # IDeliberationEngine (optional)
    event_bus=event_bus,               # EventBus (optional)
    middleware=[                       # Middleware chain (optional)
        ValidationMiddleware(),
        MetricsMiddleware(),
        TracingMiddleware(),
    ],
    performance_monitor=perf_mon,      # PerformanceMonitor (optional)
    tool_executor=tool_executor,       # ToolExecutor (optional)
)
```

### Synchronous Processing

```python
output = orchestrator.process(cognitive_input, cognitive_state)
```

### Asynchronous Processing

```python
output = await orchestrator.process_async(cognitive_input, cognitive_state)
```

---

## Middleware Layer

Middleware is used for cross-cutting concerns in the request/response cycle. Each middleware implements `handle_request()` and `handle_response()` methods.

### Built-in Middleware

**ValidationMiddleware**
- Validates the incoming `CognitiveInput` object
- Raises an error if `user_message` is empty
- Security keyword filtering (optional)

**MetricsMiddleware**
- Request counter, error counter, latency histogram
- Prometheus-compatible metric format
- Records `ctx.metadata["request_start_time"]`

**TracingMiddleware**
- Creates spans for distributed tracing
- Generates `trace_id`, `span_id`
- Per-handler duration measurement

### Writing Custom Middleware

```python
from cognitive_management.v2.middleware.base import BaseMiddleware

class MyMiddleware(BaseMiddleware):
    def handle_request(self, ctx: ProcessingContext) -> ProcessingContext:
        ctx.metadata["my_flag"] = True
        return ctx

    def handle_response(self, ctx: ProcessingContext) -> ProcessingContext:
        print(f"Processing complete: {ctx.final_response[:50]}")
        return ctx
```

---

## Event Bus

`EventBus` provides loosely coupled communication between components. Any component can publish events without holding a direct reference to other components.

### CognitiveEvent Types

| Event | When Triggered |
|-------|----------------|
| `REQUEST_STARTED` | When a request enters the pipeline |
| `POLICY_SELECTED` | When PolicyRouter selects a mode |
| `DELIBERATION_COMPLETE` | When reasoning steps finish |
| `GENERATION_COMPLETE` | When the model produces output |
| `CRITIC_TRIGGERED` | When Critic starts evaluation |
| `REVISION_APPLIED` | When Self-Refine revision is applied |
| `MEMORY_UPDATED` | When memory is updated |
| `REQUEST_COMPLETE` | When the entire pipeline finishes |
| `ERROR_OCCURRED` | When any error occurs |

### Usage

```python
# Subscribe to an event
def on_policy_selected(event: CognitiveEvent):
    print(f"Selected mode: {event.data['mode']}")

event_bus.subscribe("POLICY_SELECTED", on_policy_selected)

# Register async handler
async def on_generation(event: CognitiveEvent):
    await log_to_external_service(event.data)

event_bus.subscribe_async("GENERATION_COMPLETE", on_generation)
```

---

## Dependency Injection Container

`DependencyContainer` manages the lifecycle and dependencies of components.

```python
from cognitive_management.v2.container.dependency_container import DependencyContainer

container = DependencyContainer()

# Singleton registration
container.register_singleton("event_bus", EventBus())
container.register_singleton("memory_service", MemoryServiceV2(config.memory))

# Factory registration (new instance on every resolve)
container.register_factory("critic", lambda: CriticV2(backend, config.critic))

# Resolution
event_bus = container.resolve("event_bus")
```

---

## Interfaces / Protocol Definitions

### Backend Protocols (`backend_protocols.py`)

```python
class MinimalBackend(Protocol):
    """Minimal backend requiring only generate()."""
    def generate(self, messages: list[dict], config: DecodingConfig) -> str: ...

class FullModelBackend(Protocol):
    """Full-featured backend - embed + generate + forward."""
    def generate(self, messages: list[dict], config: DecodingConfig) -> str: ...
    def embed(self, text: str) -> list[float]: ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def entropy_estimate(self, text: str) -> float: ...
```

### Component Protocols (`component_protocols.py`)

```python
class PolicyRouter(Protocol):
    def route(self, features: dict, state: CognitiveState) -> PolicyOutput: ...

class MemoryService(Protocol):
    def get_history(self, max_tokens: int) -> list[dict]: ...
    def add_turn(self, role: str, content: str) -> None: ...
    def retrieve_relevant(self, query: str, top_k: int) -> list[dict]: ...

class DeliberationEngine(Protocol):
    def deliberate(self, mode: Mode, context: str, config: DecodingConfig) -> list[ThoughtCandidate]: ...

class Critic(Protocol):
    def review(self, text: str, query: str) -> tuple[bool, list[CriticFeedback]]: ...

class ToolExecutor(Protocol):
    def execute(self, tool_name: str, args: dict) -> str: ...
```

---

## Async Pipeline

`AsyncProcessingPipeline` and `AsyncProcessingHandler` are the fully asynchronous counterparts of the synchronous pipeline.

### Async Handler Structure

```python
class BaseAsyncProcessingHandler:
    async def handle(self, ctx: ProcessingContext) -> ProcessingContext:
        # Perform the operation
        result = await some_async_operation()
        ctx.metadata["result"] = result
        # Continue the chain
        if self._next:
            return await self._next.handle(ctx)
        return ctx
```

### Creating an Async Pipeline

```python
pipeline = AsyncProcessingPipeline()
pipeline.add_handler(AsyncFeatureExtractionHandler(backend))
pipeline.add_handler(AsyncMemoryRetrievalHandler(memory_service))
pipeline.add_handler(AsyncGenerationHandler(backend))

output = await pipeline.execute(ctx)
```

---

## Component Details

### PolicyRouterV2 — Decision Logic

```
entropy < entropy_gate_think (1.5)
    --> mode = "direct"

entropy_gate_think <= entropy < entropy_gate_debate (2.5)
    --> mode = "think1"  (Chain-of-Thought)
    --> inner_steps = 1

entropy_gate_debate <= entropy < entropy_gate_tot (3.0)
    --> mode = "debate2" or "self_consistency"
    --> inner_steps = self_consistency_n (3)

entropy >= entropy_gate_tot (3.0) AND length > length_gate_tot (300)
    --> mode = "tot"
    --> inner_steps = tot_max_depth x tot_branching_factor
```

Domain-based temperature adjustment:

```python
temperature_map = {
    "math":     0.45,   # config.decoding_bounds.math_temperature
    "code":     0.50,   # config.decoding_bounds.code_temperature
    "creative": 0.85,   # config.decoding_bounds.creative_temperature
    # Other domains: default_decoding.temperature (0.65)
}
```

### TreeOfThoughts Algorithm

```
Root node: initial contexts
    |
    +-- Expand (branching_factor = 3 candidates)
    |       |
    |       +-- Evaluate (scorer: word/semantic alignment score)
    |       |
    |       +-- Prune (keep top_k = 5 best)
    |
    +-- Increase depth (up to max_depth = 3)
    |
    +-- Highest-scoring leaf --> final response
```

### MemoryServiceV2 — Memory Layers

```
Session Memory (RAM - short-term)
    history: list[dict]
    max_history_tokens: 3072
    Pruning: delete oldest messages when token limit is exceeded

Episodic Vector Memory (ChromaDB - long-term)
    Collection: "cognitive_memory"
    Path: "./memory/episodic_store"
    Embedding: sentence-transformers/all-MiniLM-L6-v2
    top_k: 5, score_threshold: 0.7

Every 6 turns:
    1. Generate session summary
    2. Embed the summary (EmbeddingAdapter)
    3. Write to ChromaDB (metadata + vector)
    4. history.clear() (old messages are now in the vector store)
```

### AIOps Monitoring System

```
PerformanceMonitor
    - Request latency (average, p95, p99)
    - Token/second throughput
    - Per-handler duration profile

AnomalyDetector
    - Latency spike detection (EWMA-based)
    - Error rate increase
    - Memory usage anomaly

TrendAnalyzer
    - Time series trend analysis
    - Seasonality detection

PredictiveAnalytics
    - Prediction of upcoming load increases
    - Proactive cache warming suggestions

AlertingManager
    - Threshold-based alert generation
    - Webhook / callback integration

HealthCheck
    - System status summary
    - Backend connectivity health
    - VectorStore connectivity health
```
