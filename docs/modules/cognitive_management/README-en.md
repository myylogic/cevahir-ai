# Cognitive Management

[Türkçe](README.md) · [Architecture contract](../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md) · [Development roadmap](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md)

Cognitive Management is Cevahir's response-processing layer around the language model.
It selects a generation strategy and combines conversation memory, registered tools,
candidate generation and response revision in one processing flow.
Response quality depends on the connected model's training and the context supplied to it.

The active entry point is `CognitiveManager`, coordinated by `v2/core/CognitiveOrchestrator`.
V2, V3 and Phase labels in these files describe this module's development history;
they are separate from the neural core's V7/V8 labels.
This guide reflects the code reviewed on 20 September 2026.

## Processing flow

```text
Cevahir.process / CognitiveManager.handle
  → request scope and middleware (validation, cache, tracing)
  → feature extraction and PolicyRouterV2
  → optional deliberation / Tree of Thoughts
  → context from history, RAG and tool results
  → response generation through the model
  → optional self-consistency and critic revision
  → memory, session state and CognitiveOutput
```

`handle_async` is also available; the implementation runs model-related steps in threads.
Its ToT wiring differs from the synchronous path, as documented in the open work below.
The orchestrator's request batcher is inactive; the async interface does not imply batched model generation.

| Component | Implemented behavior |
|---|---|
| PolicyRouterV2 | Selects modes and decoding settings using query types, keyword signals, complexity and thresholds. |
| DeliberationEngineV2 | Generates candidates through model `generate`, ranks with `score`, and can use heuristic scores on failure. |
| TreeOfThoughts | Expands and scores candidate paths through model calls. |
| SelfConsistencyHandler | Selects among multiple generations using text similarity or model scores. |
| CriticV2 | Evaluates task match, relevance, coherence and content signals; requests model revisions when needed. |
| MemoryServiceV2 | Manages conversation history and scoped episodic memory, adding relevant records to context. |

`direct`, `think1`, `debate2` and `tot` are generation strategies.
`debate2` generates candidates from two perspectives and selects one.
Self-consistency is a separately enabled selection step.
Most policy and critic scores use rules; there is no separate trained judge model.
Basic factuality checking looks for claim and uncertainty markers and does not establish factual correctness.

## Application integration

[CognitiveManager](../../../cognitive_management/cognitive_manager.py) accepts a ModelAPI:
`generate(prompt, decoding_cfg)` and `score(prompt, candidate)` are required.
The Cevahir facade supplies this connection through `CevahirModelAPI`.
The function below accepts an initialized model adapter; it neither trains a model nor downloads weights.

```python
from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.cognitive_types import CognitiveInput, CognitiveState
from cognitive_management.config import CognitiveManagerConfig

def create_conversation(model_api):
    cfg = CognitiveManagerConfig()
    cfg.memory.enable_vector_memory = False
    cfg.memory.enable_rag = False
    cfg.policy.allow_inner_steps = False
    cfg.policy.self_consistency_enabled = False
    cfg.critic.enabled = False
    cfg.tools.enable_tools = False
    return CognitiveManager(model_manager=model_api, cfg=cfg), CognitiveState()

# model_api: the application's initialized generate/score adapter
# manager, state = create_conversation(model_api)
# output = manager.handle(state, CognitiveInput(user_message="Hello"))
# print(output.text)
```

These settings start integration with a single generation path.
Enable strategies, tools, RAG and critic options through [configuration](../../../cognitive_management/config.py) as needed.
The defaults enable more components and may introduce additional model calls and dependencies.
`CognitiveOutput` carries the response, selected mode, successful tool use and evaluation information.

## Memory, tools and caching

- **Memory scope:** Uses the session ID and application-authenticated `state.metadata["user_id"]`.
  Keep state for the same conversation; do not share one state across different users.
  Notes, summaries and episodic retrieval follow this scope.
- **Vector memory:** Implemented stores are `memory` and `chroma`.
  Embedding adapters are optional; initialization failure falls back to keyword search.
  Pinecone, Weaviate, Qdrant and Milvus options do not have implementations yet.
- **Tools:** The built-in tool is a restricted AST calculator supporting `+`, `-`, `*`, `/`.
  Applications must supply search and file tools through `register_tool` and include them in the allow list.
  A tool is reported as used only after execution succeeds.
- **Caching:** Exact response keys include identity, history, decoding settings, configuration and memory revision.
  A semantic-cache class exists, but that path is disabled for normal scoped orchestrator requests.
  Response caching stores reusable text outputs and is separate from the model's KV cache.

## Next development boundaries

Code inspection and small checks with controlled dependencies identified five open behaviors:

1. **Entropy:** `CevahirModelAPI.entropy_estimate` uses string tokens instead of IDs;
   it can fall back to token diversity before reaching the model forward call.
2. **System instructions:** Request-specific and default system instructions are omitted from final generation context.
3. **Calculator input:** Automatic extraction can truncate `2+3*4` to `2+3`;
   the full-expression executor and natural-language parameter extraction need a shared contract.
4. **Async ToT:** The async handler receives no ToT object, so a `tot` selection falls back to one candidate.
5. **Critic concurrency:** Shared `_last_feedback` and `_last_passes` fields can mix data between requests.

Priorities and acceptance criteria are tracked in the [development roadmap](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md).
These issues were not fixed by this documentation update.

## Code and further documentation

- [Orchestrator](../../../cognitive_management/v2/core/orchestrator.py) and [processing steps](../../../cognitive_management/v2/processing/handlers.py)
- [Model adapter](../../../model/cevahir.py), [critic](../../../cognitive_management/v2/components/critic_v2.py) and [tool policy](../../../cognitive_management/v2/components/tool_policy_v2.py)
- [Memory](../../../cognitive_management/v2/components/memory_service_v2.py), [store factory](../../../cognitive_management/v2/components/vector_store/__init__.py) and [cache](../../../cognitive_management/v2/middleware/cache.py)

Older documents under this folder's `architecture`, `guides`, `api` and `development`
directories contain design history and examples; they are not necessarily current verified contracts.
Resolve discrepancies against this guide, the architecture contract and the active code together.
