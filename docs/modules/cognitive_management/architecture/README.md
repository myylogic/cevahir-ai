# Cognitive Management V2 - Architecture Documentation

**Versiyon:** 2.0  
**Son Güncelleme:** 2025-01-27

---

## 📋 İÇİNDEKİLER

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Diagrams](#component-diagrams)
4. [Flow Diagrams](#flow-diagrams)
5. [Sequence Diagrams](#sequence-diagrams)

---

## 🎯 System Overview

V2 Cognitive Management, büyük dil modelleri (LLM) için **endüstri standartlarında, akademik doğrulukta** bir bilişsel yönetim katmanıdır.

### Mimari Katmanlar

```
┌─────────────────────────────────────────────────┐
│         CognitiveManager (Public API)            │
│  - handle() / handle_async()                    │
│  - Enterprise Features API                      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         CognitiveOrchestrator (Core)            │
│  - Request orchestration                        │
│  - Middleware chain execution                   │
│  - Pipeline processing                          │
│  - Performance monitoring                      │
└─────────────────────────────────────────────────┘
         ↓                    ↓
┌─────────────────┐  ┌──────────────────────────┐
│  Middleware     │  │   Processing Pipeline     │
│  Chain          │  │   (Chain of Responsibility)│
│  - Validation   │  │   - Feature Extraction    │
│  - Tracing      │  │   - Policy Routing        │
│  - Cache        │  │   - Deliberation          │
│  - Error        │  │   - Context Building      │
│  - Metrics      │  │   - Generation            │
└─────────────────┘  │   - Critic                │
                     │   - Memory Update         │
                     └──────────────────────────┘
```

---

## 🏗️ Architecture Diagrams

### High-Level Architecture

```mermaid
graph TB
    A[User Request] --> B[CognitiveManager]
    B --> C[CognitiveOrchestrator]
    C --> D[Middleware Chain]
    D --> E[Processing Pipeline]
    E --> F[PolicyRouter]
    E --> G[DeliberationEngine]
    E --> H[MemoryService]
    E --> I[Critic]
    E --> J[ToolExecutor]
    E --> K[Model Backend]
    K --> L[Response]
    L --> M[User]
```

### Component Architecture

```mermaid
graph LR
    A[CognitiveManager] --> B[Orchestrator]
    B --> C[Container]
    C --> D[PolicyRouterV2]
    C --> E[MemoryServiceV2]
    C --> F[CriticV2]
    C --> G[DeliberationEngineV2]
    C --> H[ToolExecutorV2]
    C --> I[EventBus]
    C --> J[PerformanceMonitor]
```

### Middleware Chain

```mermaid
graph LR
    A[Request] --> B[ValidationMiddleware]
    B --> C[TracingMiddleware]
    C --> D[CacheMiddleware]
    D --> E[ErrorHandlingMiddleware]
    E --> F[MetricsMiddleware]
    F --> G[Pipeline]
    G --> H[Response]
```

### Processing Pipeline

```mermaid
graph TD
    A[FeatureExtractionHandler] --> B[PolicyRoutingHandler]
    B --> C[DeliberationHandler]
    C --> D[ContextBuildingHandler]
    D --> E[GenerationHandler]
    E --> F[CriticHandler]
    F --> G[MemoryUpdateHandler]
    G --> H[Output]
```

---

## 🔧 Component Diagrams

### PolicyRouterV2

```mermaid
classDiagram
    class PolicyRouterV2 {
        +route(features, state) PolicyOutput
        -_select_mode() Mode
        -_decoding_from_features() DecodingConfig
    }
    class CognitiveManagerConfig {
        +policy PolicyConfig
    }
    PolicyRouterV2 --> CognitiveManagerConfig
```

### MemoryServiceV2

```mermaid
classDiagram
    class MemoryServiceV2 {
        +add_turn() List
        +retrieve_context() List
        +build_context() str
        +summarize() str
    }
    class EpisodicMemory {
        +episodes List
    }
    class SemanticMemory {
        +vectors Dict
    }
    MemoryServiceV2 --> EpisodicMemory
    MemoryServiceV2 --> SemanticMemory
```

### CriticV2

```mermaid
classDiagram
    class CriticV2 {
        +review() Tuple[str, bool]
        -_evaluate_aspects() Dict
        -_should_revise() bool
        -_revise_text() str
    }
    class ModelAPI {
        +generate() str
        +score() float
    }
    CriticV2 --> ModelAPI
```

### DeliberationEngineV2

```mermaid
classDiagram
    class DeliberationEngineV2 {
        +generate_thoughts() List[ThoughtCandidate]
        -_build_inner_prompt() str
        -_build_debate_prompt() str
        -_heuristic_score() float
    }
    class ThoughtCandidate {
        +text str
        +score float
    }
    DeliberationEngineV2 --> ThoughtCandidate
```

---

## 🔄 Flow Diagrams

### Request Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant CM as CognitiveManager
    participant Orch as Orchestrator
    participant MW as Middleware
    participant Pipe as Pipeline
    participant Backend as ModelBackend

    User->>CM: handle(state, request)
    CM->>Orch: handle(state, request)
    Orch->>MW: before(state, request)
    MW-->>Orch: state, request
    Orch->>Pipe: process(state, request)
    Pipe->>Backend: generate(prompt)
    Backend-->>Pipe: text
    Pipe-->>Orch: response
    Orch->>MW: after(state, request, response)
    MW-->>Orch: response
    Orch-->>CM: response
    CM-->>User: response
```

### Tool Execution Flow

```mermaid
sequenceDiagram
    participant Handler as ContextBuildingHandler
    participant Policy as ToolPolicyV2
    participant Executor as ToolExecutorV2
    participant Tool

    Handler->>Policy: choose_tool(features)
    Policy->>Policy: _select_tool_from_features()
    Policy-->>Handler: tool_name
    Handler->>Executor: execute(tool_name, params)
    Executor->>Tool: tool_func(**params)
    Tool-->>Executor: result
    Executor-->>Handler: result
```

### Config Hot Reload Flow

```mermaid
sequenceDiagram
    participant File as Config File
    participant Watcher as FileWatcher
    participant Manager as ConfigManager
    participant Listener as ConfigListener
    participant CM as CognitiveManager

    File->>Watcher: File modified
    Watcher->>Manager: _reload_from_file()
    Manager->>Manager: _load_config()
    Manager->>Manager: _update_cache()
    Manager->>Listener: notify(event)
    Listener->>CM: on_config_change()
    CM->>CM: update components
```

---

## 📊 Sequence Diagrams

### Think Mode (CoT) Flow

```mermaid
sequenceDiagram
    participant Router as PolicyRouter
    participant Deliberation as DeliberationEngine
    participant Backend as ModelBackend
    participant Selector as ThoughtSelector

    Router->>Router: route() -> mode="think1"
    Router-->>Deliberation: generate_thoughts(num=1)
    Deliberation->>Backend: generate(CoT prompt)
    Backend-->>Deliberation: thought text
    Deliberation->>Backend: score(prompt, thought)
    Backend-->>Deliberation: score
    Deliberation-->>Selector: ThoughtCandidate
    Selector-->>Router: selected_thought
```

### Debate Mode (Self-Consistency) Flow

```mermaid
sequenceDiagram
    participant Router as PolicyRouter
    participant Deliberation as DeliberationEngine
    participant Backend as ModelBackend
    participant Selector as ThoughtSelector

    Router->>Router: route() -> mode="debate2"
    Router-->>Deliberation: generate_thoughts(num=2)
    loop For each thought
        Deliberation->>Backend: generate(debate prompt)
        Backend-->>Deliberation: thought text
        Deliberation->>Backend: score(prompt, thought)
        Backend-->>Deliberation: score
    end
    Deliberation-->>Selector: [ThoughtCandidate, ThoughtCandidate]
    Selector->>Selector: select_best()
    Selector-->>Router: selected_thought
```

### Critic Review Flow

```mermaid
sequenceDiagram
    participant Critic as CriticV2
    participant Model as ModelAPI
    participant Evaluator

    Critic->>Evaluator: _evaluate_aspects()
    Evaluator->>Evaluator: check_length()
    Evaluator->>Evaluator: check_coherence()
    Evaluator->>Evaluator: check_safety()
    Evaluator->>Evaluator: check_facts()
    Evaluator->>Evaluator: check_style()
    Evaluator->>Evaluator: check_relevance()
    Evaluator-->>Critic: scores
    Critic->>Critic: _should_revise(scores)
    alt Needs revision
        Critic->>Model: generate(revise prompt)
        Model-->>Critic: revised_text
        Critic-->>Critic: (revised_text, True)
    else No revision
        Critic-->>Critic: (draft_text, False)
    end
```

---

## 🔗 İlgili Dokümantasyon

- [API Reference](../api/README.md)
- [Usage Guides](../guides/README.md)
- [Development Guide](../development/README.md)
- [Main Documentation](../README.md)

---

**Hazırlayan:** AI Assistant (Auto)  
**Versiyon:** 2.0

