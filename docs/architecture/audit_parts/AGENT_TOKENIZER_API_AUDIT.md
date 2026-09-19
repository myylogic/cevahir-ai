# Tokenizer, cognitive runtime, chat, API and database forensic audit

Audit date: 2026-09-14. This is the **pre-change source baseline**. Ownership: `tokenizer_management`, `cognitive_management`, `chatting_management`, `api`, `database`. No application code was modified or tests executed for this report. Referenced line numbers describe the source at audit time. IMPLEMENTED below means an actual executable implementation and call path were inspected; it does not mean measured quality or a passing runtime regression. The parent audit records runtime verification separately.

## Inventory and runtime boundaries

The five owned directories contained 224 files: tokenizer 42, cognitive 108, chatting 14, API 34, database 26. Their complete file inventory is appended below. Test and documentation files were inventoried, with representative tests inspected for realism and stale expectations. Runtime entry points and capability-bearing implementations were inspected directly, rather than inferred from module names or README claims.

- Tokenization path: `TokenizerCore` -> `BPEManager` -> pretokenizer/optional syllabifier/optional morphology -> `BPEEncoder`; decoding uses `BPEDecoder`. `BPETokenizer` is a separate encoder/decoder facade, not the full text preprocessing pipeline.
- Cognitive path: `CognitiveManager` constructs V2 implementations (`cognitive_manager.py:200`) and delegates to `CognitiveOrchestrator`; sync and async processing use separate pipeline builders. The presence of V3 text in handlers does not define another model architecture.
- Chat path: `ChattingManager.send_message` validates session ownership, builds `CognitiveState`, calls a shared Cevahir instance, and stores user/assistant messages (`chatting_manager.py:149`, `:168`, `:180`). Database user memory and cognitive vector/episodic memory are separate systems with separate scope rules.
- API path: Flask factory -> Cevahir + ChattingManager -> services -> V3 routes. Legacy V2 routes remain. Database uses SQLAlchemy declarative models, repositories, and UnitOfWork with PostgreSQL/SQLite/MySQL configuration.

## Capability verdicts

| Capability | Status | Executable evidence and limits |
|---|---|---|
| BPE merge encoding | IMPLEMENTED | `tokenizer_management/bpe/bpe_encoder.py:280`, `:371`, `:405`: ranked adjacent merge application and vocabulary ID mapping. Actual trained artifacts exist at `data/vocab_lib/vocab.json` and `data/merges_lib/merges.txt`. Throughput and quality remain unmeasured here. |
| Turkish morphology logic | PARTIAL | `bpe/tokenization/morphology.py:130` calls rule/exception-based `split_morpheme_impl`, performs suffix-chain and root normalization. In manager `bpe_manager.py:391-403`, morphology is nested under syllable emission and runs on syllables, not the full word. Both `include_syllables` and `include_morphology` default false (`config.py:759`, `:767`). This is not a learned morphology model. |
| Turkish case/Unicode preservation | PARTIAL | Manager normalization uses NFC and Turkish-specific I/İ lowercasing when requested (`bpe_manager.py:304-314`). Morphology separately uses generic `token.lower()` (`morphology.py:149`, `:244`) and suffix `casefold()` (`:83`). Exact surface preservation requires benchmark coverage, especially decomposed Unicode. |
| Unknown/fallback behavior | IMPLEMENTED | `bpe_encoder.py:529-618`: Unicode character lookup -> legacy character-with-word-boundary lookup -> `<UNK_CHAR>` or `<UNK>`. This is **character fallback, not byte fallback**; arbitrary Unicode cannot be losslessly represented if absent from vocabulary. Manager has optional syllable retry (`bpe_manager.py:595-668`). |
| Exact arbitrary-string round trip | BROKEN | `bpe_manager.py:313` collapses and strips whitespace; `:368` removes role tags; `bpe_decoder.py:153` exposes cleanup options. Original newlines/repeated spaces/tags cannot be reconstructed. A normalized-text round-trip metric must be reported separately from exact round trip. |
| SentencePiece backend | CONFIGURED_BUT_UNUSED | `tokenizer_management/config.py:128-133` defines settings; no executable SentencePiece backend is in the owned tokenizer inventory. |
| RAG | PARTIAL | Real retrieval in `memory_service_v2.py:206`, vector search at `:240`, weighted vector/keyword combination at `:290`; `processing/handlers.py:455-460` calls `RAGEnhancer`. Optional SentenceTransformers/OpenAI embeddings and Chroma/in-memory stores exist. Missing dependency/model initializes keyword fallback (`memory_service_v2.py:123`). No labeled retrieval evaluation establishes benefit. |
| Memory | PARTIAL | `memory_service_v2.py:130` adds conversation turns, `:219` retrieves; database `MemoryStorage` provides user-keyed persistence (`chatting_management/storage/memory_storage.py:58`). Cognitive global-memory isolation and persisted ID uniqueness are broken; see P0/P1 findings. |
| Tools as standalone registry/executor | PARTIAL | `tool_executor_v2.py:111`, `:139` register and invoke callables; allow-list checked at execution. Schema dictionaries are only stored, not type-validated (`:183-188`). Search/file defaults return placeholder strings (`:292-321`). Calculator evaluates a restricted-character expression without operation/time bounds (`:290`). |
| Tools in agent runtime | BROKEN | `handlers.py:433-480` chooses a name and appends an `[ARAÇ İSTEĞİ]` string. No runtime call to `tool_executor.execute` or `infer_tool_parameters` is present. `processing/pipeline.py:295` and `async_pipeline.py:223` report chosen `tool_name` as `tool_used`, even though it was never invoked. |
| Critic/evaluator | IMPLEMENTED | `critic_v2.py:223-298`: conditional constitutional review, feedback evaluation, revision passes; lexical/heuristic relevance/task/coherence/factuality checks and model rewrite. The algorithm exists; factual accuracy and task success are unmeasured. |
| Direct generation | IMPLEMENTED | `policy_router_v2.py:145` and `processing/handlers.py:490-533`: routing and backend generation; requires a functioning model backend. |
| Think / Debate | IMPLEMENTED | `deliberation_engine_v2.py:77-142`: one or two differently prompted candidate generations, backend scoring with heuristic fallback; selected thought enters final prompt (`handlers.py:283`). Debate here is candidate generation/selection, not autonomous inter-agent debate. |
| Tree of Thoughts | PARTIAL | Real BFS expansion, evaluation, pruning and path ranking in `tree_of_thoughts.py:149`, `:200`; sync orchestrator creates it (`orchestrator.py:187-209`). Async builder (`:507-598`) does not provide ToT to its deliberation adapter, so feature parity is not established. |
| Monitoring | IMPLEMENTED | Actual operation timing/count aggregation with locking at `monitoring/performance_monitor.py:182-237`, called by orchestrator (`orchestrator.py:465-485`). Health, alerts, anomaly/trend modules exist. These counters do not establish model quality or agent success. |
| Tracing | PARTIAL | `utils/tracing.py:266` stores traces with locking; events are published via `events/event_bus.py:131`. `middleware/tracing.py:97-98` keeps shared mutable current trace/span state, so concurrent requests can cross-contaminate. Context storage uses thread-local rather than async task-local scope (`utils/tracing.py:320-336`). |
| Deterministic replay | NOT_IMPLEMENTED | No record-and-replay executor/replay API in the owned runtime. Trace JSON exports are not deterministic execution replay. |
| Retry/timeout policy | BROKEN | `middleware/error_handler.py:161-175` enters a retry loop but returns an error response on the first iteration without reinvoking work. `self.timeout` is assigned (`:114`) without actual deadline enforcement. Circuit breaker counters do exist. |
| Context budgeting | PARTIAL | Chat history budget uses `len(chars)/4` (`components/context_builder.py:221-235`), then appends current message and memory after history pruning (`:105-124`). `enable_semantic_search` is configured (`chatting_management/config.py:57`) but memory context uses high-priority rows, ignores query (`context_builder.py:201-215`). |
| Agent batching | PARTIAL | Orchestrator exposes `handle_batch`, request batcher exists; this is request orchestration, not evidence of tensor-level batched autoregressive decoding. |
| Native multimodal backbone | NOT_IMPLEMENTED | `cognitive_manager.py:598` obtains text from `mm.process_multimodal` and routes it through text handling. `v2/adapters/backend_adapter.py:150` forwards processor methods. No vision encoder -> learned projector -> language hidden-state fusion exists in these modules. Model-owned multimedia analysis is supplied by the model audit. |
| Model beam search/top-k/top-p | NOT_IMPLEMENTED in owned runtime | Cognitive `DecodingConfig`/bounds configure decoding and forward it to the backend; no token sampler or neural beam search resides in these five directories. Model generation capabilities must be classified by the separate model audit. ToT top-k is candidate-path ranking, not token top-k sampling. |
| API / SQL persistence integration | BROKEN | Source contains actual routes/repository operations but factory import/blueprint ordering and declarative-model defects block integration. See findings below. |
| Typed configuration consistency | PARTIAL | Cognitive, chatting and database dataclasses exist independently. Tokenizer uses dicts, API copies hardcoded model dimensions (`app_factory.py:95-105`). No shared architecture version/migration exists in these modules. |

## Prioritized correctness and configuration findings

1. **P0 — API cannot start from this checkout through documented factory imports.** `api/api_config.py:23` and `api/app_factory.py:29` import `config.parameters`, but no root `config/parameters.py` is present in the initial inventory. Independent of this, factory registers the blueprint at `app_factory.py:325` before functions at `:328-330` add decorated routes. These setup calls must happen before registration, and per-app factories must avoid accumulating duplicate routes on the module-global blueprint. Parent environment check additionally reports Flask/SQLAlchemy missing; those dependencies must be installed before runtime verification.
2. **P0 — SQLAlchemy model reserved name.** `database/models.py:138`, `:182`, `:230`, `:281` declare `metadata = Column(...)` on declarative models. SQLAlchemy reserves class `metadata`; this is an import-time blocker expected from source, not a runtime result in this audit. Preserve the SQL column name while migrating Python attributes/callers. Do not silently discard persisted metadata.
3. **P0 — tools falsely reported as executed.** Runtime only selects tools and writes their names into model prompts; executor is uncalled. Fix must add a typed invocation/result record, invoke exactly once after policy checks, feed verified output into context, and derive `tool_used` from successful execution. Placeholder tools must never count as real retrieval/file access.
4. **P0 — cross-session cognitive memory scope.** A shared `CognitiveManager` owns one `MemoryServiceV2` (`cognitive_manager.py:200`). Added memory metadata contains only role/timestamp (`memory_service_v2.py:178-188`), and retrieval carries no user/session filter (`:240-244`). Chat database history has ownership checks, but cognitive retrieval can consume another user's prior turns when the shared API instance serves multiple users. Scope must be explicit through request, memory IDs, vector filters and cache keys before multi-user deployment.
5. **P0 — response-cache identity is incomplete.** Exact key uses only last three history entries, step, last mode, user message and system prompt (`middleware/cache.py:191-228`), excluding decoding config, model/config version and memory/user scope. Semantic cache queries only user text (`:130-131`). Identical latest messages with different earlier context can reuse an incorrect answer. Scope and full effective context/version must enter cache identity.
6. **P1 — episodic IDs collide after pruning/restart.** `memory_service_v2.py:167` uses `len(_episodic_memory)` as timestamp, `:184` constructs `turn_{timestamp}`, and `:199-202` shrinks that same list to ten entries. Future additions reuse existing IDs; process restart also starts at zero while Chroma is configured persistent (`config.py:116-120`). Use persisted unique identifiers; do not reinterpret previously stored IDs.
7. **P1 — config overrides are not coherent.** `BPEManager.__new__` keys solely on paths and GPU (`bpe_manager.py:110-128`), so a second instance with different config/vocab silently inherits the first. Manager config is not passed into encoder/decoder/trainer/morphology/syllabifier (`:276-287`), and `Morphology` applies global BPE defaults after caller overrides (`morphology.py:60-67`). `CognitiveManager.set_config_value/reload_config/update_config` replaces manager `cfg` (`cognitive_manager.py:1444`, `:1462`, `:1502`), whereas previously built components still hold the old object; there is no whole-runtime reconstruction/migration.
8. **P1 — trace current-span state is shared.** Two threads can interleave middleware `_before` and `_after`, overwriting `self._current_trace/span`. Trace-storage locking does not protect request identity. Use per-request or context-local active span state with proper cleanup, including unsampled/error/cache-hit paths.
9. **P1 — tokenizer constructor has side effects.** Explicit `vocab` writes the provided file at `bpe_manager.py:143-146`, even before `read_only` is evaluated. Missing merges may be created before initialization (`:123`). Reproducible benchmarks should use existing read-only artifacts or dedicated temporary copies, never pass an experimental vocab against production artifact paths.
10. **P1 — tokenizer chunk progression can stall.** `TokenizerCore._split_token_ids_by_length` advances to `end_idx-overlap` without validating `overlap < max_length` (`core/tokenizer_core.py:1165`, `:1201`). If overlap equals/exceeds max length, a long input never advances. Add boundary validation and regression coverage.

## Benchmark and regression guidance

The least costly faithful tokenizer baseline is `BPEManager(vocab_file=<absolute existing vocab>, merges_file=<absolute existing merges>, use_gpu=False, config={"read_only": True})`. Call `tokens, ids = manager.encode(text, mode="inference", add_special_tokens=False)` and `manager.decode(ids)`. There is **no** `return_tokens` argument: manager always returns a tuple. Use inverse vocabulary IDs to obtain actual emitted subword surfaces; the returned preprocessing token list is not guaranteed to be one-to-one with merged/fallback IDs. Avoid `TokenizerCore` data-loading initialization or tokenizer training for the baseline. Keep unmodified artifacts, compute artifact hashes, isolate any config experiment because the manager singleton ignores config.

Measure Turkish and English separately: word fertility, Unicode characters/UTF-8 bytes per emitted token, exact and normalized round-trip, separately annotated suffix fragmentation and morpheme-boundary recall, encode/decode throughput, vocabulary utilization, explicit unknown counts (both `<UNK>` and `<UNK_CHAR>`), and empty/whitespace/emoji/OOV/composed-decomposed Unicode samples. Hand-annotated suffix examples are diagnostic, not a statistically representative linguistic benchmark. Preserve failing examples in JSON. Neither normalized round-trip nor absence of `<UNK>` implies semantic or morphology quality.

Representative existing tests reveal drift rather than a clean baseline. `tokenizer_management/tests/test_bpe_manager.py:46-61` expects EOS during inference and removal of specials during train, while current encode defaults differ. API test `api/service/tests/test_chat_service.py:11` imports absent `api.service.chat_service` (real implementation is `api.services.chat_service`), constructs it without its required manager, and asserts a legacy response shape. Cognitive fixtures use mock generation/scoring (`cognitive_management/tests/conftest.py:29-69`) and even use `estimate_entropy` where the legacy adapter expects `entropy_estimate`; such tests do not establish trained model behavior. Smoke tests allow `tool_used=None` even for a search request (`tests/test_cognitive_manager_smoke.py:117`), so cannot catch the missing invocation.

Useful next P0 tests: real tokenizer round-trip/unknown diagnostics without changing artifacts; tool callable invocation count/result propagation/denial/schema failure; two-user memory isolation; cache separation by decoding/config/history/user scope; concurrent trace identity; memory add beyond pruning threshold and reopen; import-only SQLAlchemy models and two independent Flask app creations. Agent success-rate benchmarks should use small deterministic task fixtures, forced strategy selection, actual call counts and latency, and explicitly distinguish stubbed backend plumbing from trained-model quality.

## Claim hygiene

Unverified model analogies are scattered through tokenizer comments (`config.py:761`, `:774`, `bpe_encoder.py:531`) and cognitive headers. Replace them with local algorithm facts after baseline capture. A heuristic critic, a morphology helper, a strategy name, or a monitoring class is not evidence of frontier-model parity. No speed, quality or memory-efficiency improvement is claimed by this report.

## Complete owned file inventory

- chatting_management/__init__.py
- tokenizer_management/__init__.py
- tokenizer_management/train_bpe.py
- tokenizer_management/test_vocab_size_control.py
- tokenizer_management/test_oov_fallback_debug.py
- tokenizer_management/test_comprehensive_integration.py
- chatting_management/storage/__init__.py
- chatting_management/storage/user_storage.py
- chatting_management/storage/session_storage.py
- chatting_management/storage/memory_storage.py
- chatting_management/storage/conversation_storage.py
- chatting_management/exceptions.py
- chatting_management/config.py
- cognitive_management/__init__.py
- tokenizer_management/tests/__init__.py
- tokenizer_management/tests/test_turkish_processor.py
- tokenizer_management/tests/test_tokenizer_core_comprehensive.py
- tokenizer_management/tests/test_tokenizer_core.py
- tokenizer_management/tests/test_syllabifier.py
- tokenizer_management/tests/test_processors.py
- tokenizer_management/tests/test_pretokenizer.py
- tokenizer_management/tests/test_postprocessor.py
- tokenizer_management/tests/test_morphology.py
- tokenizer_management/tests/test_bpe_trainer.py
- tokenizer_management/tests/test_bpe_tokenizer.py
- tokenizer_management/tests/test_bpe_manager.py
- tokenizer_management/tests/test_bpe_encoder.py
- tokenizer_management/tests/test_bpe_decoder.py
- tokenizer_management/tests/test_ai_morphology.py
- tokenizer_management/prepare_bpe_cache.py
- tokenizer_management/evaluate_bpe_training.py
- tokenizer_management/debug_cache.py
- chatting_management/components/__init__.py
- chatting_management/components/user_manager.py
- chatting_management/components/session_manager.py
- chatting_management/components/conversation_manager.py
- chatting_management/components/context_builder.py
- chatting_management/chatting_manager.py
- cognitive_management/config.py
- cognitive_management/cognitive_types.py
- cognitive_management/cognitive_manager.py
- tokenizer_management/core/__init__.py
- tokenizer_management/core/tokenizer_core.py
- tokenizer_management/config.py
- tokenizer_management/check_vocab.py
- cognitive_management/v2/__init__.py
- cognitive_management/tests/__init__.py
- cognitive_management/v2/utils/__init__.py
- cognitive_management/v2/utils/tracing.py
- cognitive_management/v2/utils/semantic_cache.py
- cognitive_management/v2/utils/selectors.py
- cognitive_management/v2/utils/request_batcher.py
- cognitive_management/v2/utils/performance_profiler.py
- cognitive_management/v2/utils/heuristics.py
- cognitive_management/v2/utils/context_pruning.py
- cognitive_management/v2/utils/connection_pool.py
- cognitive_management/v2/utils/claim_extraction.py
- cognitive_management/v2/utils/cache_warming.py
- cognitive_management/v2/utils/cache.py
- tokenizer_management/bpe/__init__.py
- cognitive_management/v2/processing/__init__.py
- cognitive_management/v2/processing/pipeline.py
- cognitive_management/v2/processing/handlers.py
- cognitive_management/v2/processing/async_pipeline.py
- cognitive_management/v2/processing/async_handlers.py
- cognitive_management/tests/v2/__init__.py
- cognitive_management/tests/v2/test_policy_router_v2.py
- cognitive_management/tests/v2/test_orchestrator.py
- cognitive_management/tests/v2/test_memory_service_v2.py
- cognitive_management/tests/v2/test_integration.py
- cognitive_management/tests/v2/test_deliberation_engine_v2.py
- cognitive_management/tests/v2/test_critic_v2.py
- cognitive_management/tests/v2/README.md
- cognitive_management/tests/v2/PHASE5_TESTING_COMPLETE.md
- cognitive_management/tests/v2/conftest.py
- cognitive_management/tests/TEST_SUITE_INDEX.md
- cognitive_management/tests/TEST_SUITE_COMPLETE_REPORT.md
- cognitive_management/tests/test_cognitive_manager_tracing.py
- cognitive_management/tests/test_cognitive_manager_tools_extended.py
- cognitive_management/tests/test_cognitive_manager_tools.py
- cognitive_management/tests/test_cognitive_manager_smoke.py
- cognitive_management/tests/test_cognitive_manager_performance.py
- cognitive_management/tests/test_cognitive_manager_monitoring_extended.py
- cognitive_management/tests/test_cognitive_manager_monitoring.py
- cognitive_management/tests/test_cognitive_manager_memory_extended.py
- cognitive_management/tests/test_cognitive_manager_memory.py
- cognitive_management/tests/test_cognitive_manager_integration.py
- cognitive_management/tests/test_cognitive_manager_events_extended.py
- cognitive_management/tests/test_cognitive_manager_events.py
- cognitive_management/tests/test_cognitive_manager_core_extended.py
- cognitive_management/tests/test_cognitive_manager_core.py
- cognitive_management/tests/test_cognitive_manager_connection_pool.py
- cognitive_management/tests/test_cognitive_manager_config.py
- cognitive_management/tests/test_cognitive_manager_cache.py
- cognitive_management/tests/test_cognitive_manager_aiops.py
- cognitive_management/tests/EXTENDED_TESTS_PROGRESS.md
- cognitive_management/tests/EXTENDED_TESTS_PLAN.md
- cognitive_management/tests/conftest.py
- cognitive_management/exceptions.py
- tokenizer_management/bpe/tokenization/__init__.py
- tokenizer_management/bpe/tokenization/_syllabifier_utils.py
- tokenizer_management/bpe/tokenization/_morphology_utils.py
- tokenizer_management/bpe/tokenization/syllabifier.py
- tokenizer_management/bpe/tokenization/pretokenizer.py
- tokenizer_management/bpe/tokenization/postprocessor.py
- tokenizer_management/bpe/tokenization/morphology.py
- tokenizer_management/bpe/bpe_trainer.py
- tokenizer_management/bpe/bpe_tokenizer.py
- tokenizer_management/bpe/bpe_manager_utils.py
- tokenizer_management/bpe/bpe_manager.py
- tokenizer_management/bpe/bpe_encoder.py
- tokenizer_management/bpe/bpe_decoder.py
- tokenizer_management/base_tokenizer_manager.py
- cognitive_management/v2/container/__init__.py
- cognitive_management/v2/container/dependency_container.py
- cognitive_management/v2/monitoring/__init__.py
- cognitive_management/v2/monitoring/trend_analyzer.py
- cognitive_management/v2/monitoring/predictive_analytics.py
- cognitive_management/v2/monitoring/performance_monitor.py
- cognitive_management/v2/monitoring/health_check.py
- cognitive_management/v2/monitoring/anomaly_detector.py
- cognitive_management/v2/monitoring/alerting.py
- cognitive_management/v2/config/__init__.py
- cognitive_management/v2/config/constitutional_principles.py
- cognitive_management/v2/config/config_manager.py
- cognitive_management/v2/middleware/__init__.py
- cognitive_management/v2/middleware/validation.py
- cognitive_management/v2/middleware/tracing.py
- cognitive_management/v2/middleware/metrics.py
- cognitive_management/v2/middleware/error_handler.py
- cognitive_management/v2/middleware/cache.py
- cognitive_management/v2/middleware/base.py
- cognitive_management/v2/middleware/async_middleware.py
- cognitive_management/v2/interfaces/__init__.py
- cognitive_management/v2/interfaces/component_protocols.py
- cognitive_management/v2/interfaces/backend_protocols.py
- cognitive_management/v2/events/__init__.py
- cognitive_management/v2/events/event_handlers.py
- cognitive_management/v2/events/event_bus.py
- cognitive_management/v2/components/__init__.py
- cognitive_management/v2/core/__init__.py
- cognitive_management/v2/core/orchestrator.py
- cognitive_management/v2/components/vector_store/__init__.py
- cognitive_management/v2/components/vector_store/memory_vector_store.py
- cognitive_management/v2/components/vector_store/chroma_vector_store.py
- cognitive_management/v2/components/vector_store/base.py
- cognitive_management/v2/components/tree_of_thoughts.py
- cognitive_management/v2/components/tool_policy_v2.py
- cognitive_management/v2/components/tool_executor_v2.py
- cognitive_management/v2/components/rag_enhancer.py
- cognitive_management/v2/components/policy_router_v2.py
- cognitive_management/v2/components/memory_service_v2.py
- cognitive_management/v2/components/constitutional_critic.py
- cognitive_management/v2/adapters/__init__.py
- cognitive_management/v2/adapters/backend_adapter.py
- cognitive_management/v2/components/fact_checkers/__init__.py
- cognitive_management/v2/components/fact_checkers/wikipedia_checker.py
- cognitive_management/v2/components/fact_checkers/base.py
- cognitive_management/v2/components/embedding_adapter.py
- cognitive_management/v2/components/deliberation_engine_v2.py
- cognitive_management/v2/components/critic_v2.py
- cognitive_management/utils/__init__.py
- cognitive_management/utils/timers.py
- cognitive_management/utils/logging.py
- api/__init__.py
- api/app.py
- api/api_config.py
- api/app_factory.py
- api/monitoring/__init__.py
- api/monitoring/metrics.py
- api/monitoring/health.py
- api/security/__init__.py
- api/security/password.py
- api/security/jwt.py
- api/security/headers.py
- api/middleware/__init__.py
- api/middleware/validator.py
- api/middleware/security.py
- api/middleware/request_id.py
- api/middleware/error_handler.py
- api/middleware/auth.py
- api/service/__init__.py
- api/service/tests/test_chat_service.py
- api/services/__init__.py
- api/services/user_service.py
- api/services/session_service.py
- api/services/chat_service.py
- api/routes/__init__.py
- api/routes/v3/__init__.py
- api/routes/v3/users.py
- api/routes/v3/sessions.py
- api/routes/v3/health.py
- api/routes/v3/chat.py
- api/routes/chat_routes_v2.py
- api/utils/__init__.py
- api/utils/response.py
- api/utils/logging.py
- api/utils/exceptions.py
- database/__init__.py
- database/README.md
- database/models.py
- database/connection.py
- database/config.py
- database/exceptions.py
- database/interfaces/__init__.py
- database/interfaces/unit_of_work.py
- database/interfaces/repository.py
- database/tests/__init__.py
- database/schemas/schema_sqlite.sql
- database/tests/test_connection.py
- database/tests/test_config.py
- database/tests/conftest.py
- database/unit_of_work.py
- database/requirements.txt
- database/repositories/__init__.py
- database/repositories/user_repository.py
- database/repositories/user_memory_repository.py
- database/repositories/session_repository.py
- database/repositories/message_repository.py
- database/repositories/base_repository.py
- database/schemas/schema_postgresql.sql
- database/utils/helpers.py
- database/utils/__init__.py
- database/utils/migrations.py

