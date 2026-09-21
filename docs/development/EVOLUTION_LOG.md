# Evolution log

## 2026-09-14 — Phase 0: discovery and baseline

1. Findings: 608 original files / 513 Python modules inventoried and parsed. Core, training and tokenizer/agent/API call paths inspected; `docs/architecture/CURRENT_ARCHITECTURE_AUDIT.md` records initial classifications and evidence.
2. Changes: documentation and inventory only; original source archived before application edits. No rewrite or checkpoint conversion.
3. Tests: selected existing architecture tests: 101 passed, 6 skipped. Additional reproduction checks found six cache equivalence failures, one overlapping cache eviction exception, and invalid zero-head validation.
4. Benchmark delta: not applicable; original measurements in `docs/architecture/BASELINE_INVESTIGATION.json`. No speed or quality improvement claim.
5. Technical debt: architecture/config drift, broken cached decoding, unused MoE auxiliary loss, incompatible V3 training helpers, API startup and tokenizer/cognitive integration gaps. Full legacy suite not yet clean.
6. Next phase: P0 correctness and capability-based configuration consolidation, then a lightweight baseline/experiment/delta benchmark. User clarified that heavy training and new research architectures wait for subsequent reevaluation.

Environment: Windows, Python 3.14.3, PyTorch 2.10.0+cpu. CUDA/VRAM/distributed GPU measurements unavailable; absence is recorded as null or skipped with a reason, never zero performance.

## 2026-09-14 — Existing-core reconciliation and CPU verification

### Changes and reasons

- Repaired cached attention positions, causal/window masks, bounded eviction and sink retention. External Flash failure retains masking. Soft-cap no longer silently disappears in the default attention path. Unused attention-local normalization parameters are frozen while checkpoint keys remain; parallel residual similarly freezes its unused second norm.
- Integrated MoE auxiliary loss with the active training objective and checkpoint recomputation. Fixed token-weighted accumulation/tail updates, scheduler step ownership, precision resolution, empty epochs and epoch-boundary resume state. The V3 service now explicitly uses the supported V2 training manager; unsupported experimental flags fail early.
- Consolidated effective model configuration, aliases, nested sections and version metadata. Retained explicit legacy dimensions and direct-manager legacy layer defaults. Quantization aliases now agree across typed/flat paths; unsupported bitsandbytes loading flags fail explicitly. Cevahir construction does not accidentally invoke a competing bitsandbytes implementation. Dynamic INT8 remains an explicit post-training conversion.
- Fixed zero-temperature/zero-length generation, EOS limits and independent beam prefixes. Serialized generation on a shared API model so concurrent requests cannot interleave its mutable KV state. Added observable compile fallback without state_dict prefixes. Verified raw, versioned and explicitly trusted full-module loading.
- Forwarded tokenizer configuration through both TokenizerCore and BPEManager, isolated singleton variants, retained read-only assets and corrected punctuation spacing. Preserved measured token IDs and shipped asset hashes.
- Executed real selected tools, removed placeholder search/file defaults, isolated memory and response cache by caller/session/context, replaced reused turn IDs, and corrected sync/async cache-hit lifecycle. Async middleware no longer traverses each sync chain multiple times; ContextVars remain in the request task.
- Repaired API imports, per-app blueprint construction, SQLAlchemy metadata mapping and SQLite DDL compatibility. Tests create independent injected applications, not a live deployment.
- Shared SHA-256 content fingerprints between cache preparation and V3 consumption, including merges and BPE settings. Disabled automatic mismatched-cache reuse. Strict checksum verification rejects missing/corrupt checksums. Preparation only attests its current output. Existing prepared-data caches require regeneration; none were deleted or rebuilt during this phase.
- Replaced unsupported closed-model attribution and unmeasured speed percentages in core comments/config descriptions. Added the canonical architecture specification and README verification notice; historical documentation remains historical evidence.

### Verification

Final selected command is recorded in `benchmarks/README.md`: **232 passed, 6 skipped, 7 warnings**. This includes **101 evolution regression cases**, existing architecture/entropy/forward tests and V2 training-core tests. Machine-readable results: `benchmarks/results/verification.xml` and `verification_summary.json`.

The real V3-service integration probe creates a tiny Cevahir model, criterion, optimizer and scheduler; one synthetic training/validation batch changes actual weights and writes `last.pth`. Only the tokenizer asset boundary is substituted for this probe. A separate dropout resume test matches uninterrupted training weights and history exactly. No heavy training or trained-language evaluation was performed.

Additional targeted manager gradient-flow and service tests passed. Broad legacy probes were intentionally not concealed: one earlier manager run reported 194 passed / 2 skipped / 8 failed before subsequent gradient-flow repair; another cognitive/tokenizer probe reported 26 passed / 6 failed; an attention-heavy broad probe stopped at 103 passed / 10 failed. These expose removed field expectations, legacy unsafe pickle defaults, old mode lists, decoder API drift and a then-unresolved attention-weights result. Follow-up inspection confirmed that these tests already requested `return_attention_weights=True`; the SDPA routing ignored the request. This was a code defect, repaired in the second phase below. They are diagnostic history, not the final selected suite. The repository-wide suite is **not** claimed clean.

Final syntax scan parsed 501 Python files under the recorded application/test/benchmark roots with no errors. No original inventoried file is missing. The changed-original-file manifest includes before/after hashes; the original source archive remains available. Asset SHA-256 values still match the tokenizer baseline.

### Baseline / experiment / delta

Both final core runs were sequential, CPU one thread, seed 42, 2 warmups and 5 measurements. The same tiny configuration and synthetic data were used; JSON includes source and data hashes.

| Metric | Original | Repaired | Interpretation |
|---|---:|---:|---|
| Total parameters | 24,928 | 24,928 | Existing architecture retained |
| Trainable parameters | 24,928 | 24,800 | Unused normalization weights excluded |
| Cached/full-prefix max absolute logit error | 0.4678871 | 1.4901161e-7 | Correctness repair |
| KV stored bytes/token | 512 | 512 | Same benchmark geometry |
| Decode tokens/s | 300.60 | 97.44 | Slower in this small CPU observation; optimization remains |
| Prefill tokens/s | 5,575.88 | 2,341.24 | Slower in this observation |
| Training tokens/s | 1,881.66 | 2,796.01 | Noisy tiny-workload observation; no general speed claim |
| Validation loss | 4.8804183 | 4.8812313 | Synthetic, no language-quality interpretation |
| NaN/Inf occurrences | 0 | 0 | Measured runs only |

Tokenizer diagnostics (8 Turkish / 6 English texts): fertility unchanged at 4.7843 / 6.2857 tokens per word; exact roundtrip 0.125 → 0.5 / 0 → 0.6667; normalized roundtrip 0.375 → 0.75 / 0.1667 → 0.8333. All measured token IDs and vocabulary/merges hashes remain unchanged. Improvement is decoder spacing, not a newly trained tokenizer. Unsupported-character removal before encoding remains explicitly visible in fallback probes. Warm cached throughput is not a robust general benchmark, and repeat counts differ in the recorded tokenizer comparison.

### Remaining debt and next evaluation

The completed phase establishes tested correctness contracts, not complete production readiness. Remaining work includes legacy test migration, cached inference cost, lossless tokenizer handling, real multiuser/API integration and retrieval/critic quality evaluation. Source-aware splitting does not detect duplicated content across different source IDs. Epoch-midpoint resume, distributed GPU training, actual Flash/compile kernels and long-context quality remain unverified. Vector retrieval filters scope after ranking, so scoped recall needs evaluation. Slow custom sync middleware should use native async hooks.

No new research architecture was added. MLA/MTP/Muon/Gated DeltaNet/new MoE/native multimodality remain deferred for the user's next evaluation. `CEVAHIR_ARCHITECTURE_SPEC.md` is the handoff specification; the initial audit and research ledger remain separate.


## 2026-09-14 — Technical debt follow-up

1. **Attention API correctness:** explicit attention-weight requests now route through the existing manual attention implementation; normal generation retains SDPA. Added output equivalence, causal and padding assertions. The low-level attention method preserves its `(output, weights)` default; callers needing the accelerated weights-free path opt out. Unambiguous `(B,S)` padding is accepted; for `B == T`, callers must use `(B,1,1,S)` to distinguish it from a sequence mask.
2. **Cache cost:** no-grad appends into spare capacity avoid full-history membership tests, sorting and K/V reconstruction. Explicit replacement, overflow, oversized chunks and gradient-enabled paths retain general handling. A regression forbids sorting/membership calls during append and then verifies replacement still works. Existing MHA/GQA/MQA/window/eviction tests remain active.
3. **Scoped retrieval:** the memory service passes `filter_metadata` before top-k selection. A real in-memory vector-store regression verifies that a higher scoring other-session item cannot crowd out the user's valid result. No external embedding model was downloaded.
4. **Data split:** duplicate examples connect their source groups transitively, including common trailing PAD normalization. Groups are shuffled deterministically and retained intact. Missing source IDs still get exact-content grouping; a single connected group fails rather than inventing independent validation data. Near duplicates and partial overlap remain outside this guarantee.
5. **Decoder:** empty vocabulary raises ValueError; custom postprocessor is actually called on final text; invalid vocabulary updates preserve the prior vocabulary/reverse map. Existing decoder tests now assert stable IDs, explicit unknown/collision errors, word-end spacing and punctuation preservation instead of silently adding/reassigning IDs or dropping content. Shipped token-ID and asset-hash regressions remain unchanged.
6. **Legacy tests:** ModelManager module-pickle test explicitly trusts its own locally created file. Think-mode test uses entropy between configured gates and asserts exactly think1; validators include the implemented ToT mode. Low-temperature attention is checked against reference softmax rather than assuming every nearly tied random score exceeds .99. A hardware-independent 100 ms assertion proved flaky (153.7 ms observed); duration remains recorded with an optional explicit machine budget, and forward correctness remains asserted.

Before the final timing-test correction, the expanded run had 352 passed, 2 skipped and one timing-budget failure. Final rerun: **353 passed, 2 skipped, 8 warnings in 63.05 seconds**. The result is recorded in `benchmarks/results/debt_verification.xml` and `debt_verification_summary.json`; this is still not the whole repository suite.

CPU measurement against the preceding repaired artifact: decode 97.44 → 366.94 tokens/s; prefill 2341.24 → 5024.49 tokens/s; cache logit error 1.49e-7 → 1.79e-7; total/trainable parameters 24928/24800 unchanged; synthetic validation loss unchanged at 4.8812313. Both measured runs have zero non-finite occurrences. These separate five-iteration measurements do not establish universal speed gains. Original and first-phase artifacts remain intact.

Remaining: wider removed-module test migration; lossless Unicode tokenizer migration; real authenticated multiuser API testing; semantic retrieval/critic/ToT quality; near-duplicate split protection; GPU/distributed/long-context/actual compile validation. Heavy training and new architecture research remain deferred.


## 2026-09-14 — Third debt pass: model contracts, text loss and JWT

- Initial whole `model_management/test` run: 195 passed / 2 skipped / 7 failed. Failures referred to removed model containers, split SwiGLU projections and an all-parameters-trainable assumption. Migrated tests to real embedding/RoPE identity, merged projection dimensions, actual KV write/reset and the precise frozen-normalization parameter count. Removed no-op `hasattr` guards from two tests. No compatibility dummy modules were introduced.
- Tokenizer retains vocabulary and normal encoding IDs but detects character removal before UNK accounting. Default `text_loss_policy=warn` reports only codepoints/counts; `error` rejects lossy input; `ignore` is explicit legacy silence. It does not claim lossless Unicode encoding. Added emoji/CJK/unassigned-character and private-text logging regressions. Fixed explicit special-token arguments being overwritten by config.
- JWT used environment defaults even inside differently configured apps, mixed UTC/local timestamp expiration checks and allowed invalid Authorization to fall back to debug user identity. Shared signing settings now honor app configuration, reject known default keys, require token identity/lifetime/type fields and protect reserved claims; invalid tokens cannot downgrade to header identity. Added real Flask session-route JWT tests with two identities/two app keys, expiration, wrong token type and spoofed headers. Lower session storage remains injected; full DB authorization is still pending.
- Intermediate model/evolution verification: 315 passed / 2 skipped. Focused JWT/tokenizer verification: 28 passed. Final expanded run: **499 passed, 2 skipped, 15 warnings in 32.56 seconds**. Evidence is `benchmarks/results/debt3_verification.xml` and `debt3_verification_summary.json`; modified original Python sources parse successfully and vocabulary/merges hashes still match the original baseline. Heavy training, external deployment and vocabulary changes were not performed.

Migration notes: configure an explicit non-default JWT_SECRET_KEY; app-specific token lifetime settings now take effect. Explicit add_special_tokens arguments now win over configuration. Text loss warnings are enabled by default and BPE config participates in cache identity, so affected prepared caches require refresh. Lossless tokenizer migration, wider source-module test cleanup, production DB ownership and real retrieval/multimodal/GPU evaluation remain open.


## 2026-09-14 — Database ownership and persistence debt

- Added a temporary file-backed SQLite integration test using actual ORM models, repositories, UnitOfWork, storage, ChattingManager, Flask routes and JWT. Only language-model inference is replaced; unauthorized requests must never invoke it. Global production database configuration is not changed.
- Existing ownership checks correctly deny another user, but API catch-all blocks returned 500. Chat history/message endpoints now return 403 for denied ownership, 404 for absent sessions and 400 for invalid messages. Empty explicit user IDs no longer bypass manager ownership validation.
- Session metadata, user preferences and memory metadata updates now copy JSON mappings before merging. Reusing a loaded dict could leave SQLAlchemy unaware of changes. Two consecutive updates and fresh transactions verify actual persistence.
- Context selection previously filled its token budget from the oldest portion of retrieved history, dropping the newest messages. It now takes the newest suffix and restores chronological order. A database-backed example verifies the two latest messages are retained.
- Tests verify owner history succeeds, other-user history/write fails, other-user session list is empty, missing sessions return 404 and rejected writes leave stored history unchanged. No deployment or heavy training performed.

Final evolution regression run: **116 passed, 18 warnings in 17.12 seconds**; all seven changed Python files parse. Evidence: `benchmarks/results/database_debt_verification.xml`. SQLite validates these paths; PostgreSQL, concurrent ownership changes and a complete production security review remain unverified. Internal storage delete/update methods are trusted service primitives, not independently authenticated HTTP endpoints.


## Lower-core focus after user correction

User explicitly redirected work to AI infrastructure, ahead of upper application layers. This pass changes only core FFN/MoE/layer contracts, their schema, tests and documentation. Shared FFN dimension resolution preserves existing weights while fixing schema disagreement. Parameter estimates now account for GQA KV width, gated FFN projections and MoE router weights. Layer bias settings reach MoE experts. The norm2 freeze is restricted to the actual parallel pre-norm path, preserving post-norm gradients. See `docs/architecture/LOWER_CORE_CONTRACTS.md` for resolved and still-open lower-layer boundaries. No claim that V4–V8 architectural debt is fully closed.


## Lower-core attention routing

Removed unconditional attention-weight requests from all three active residual paths. Added explicit model/layer forward diagnostics flag, propagated through standard and advanced non-reentrant checkpointing. Normal output tuple retains its shape with weights=None; diagnostics explicitly request manual attention. Tests observe actual fused SDPA invocation and compare outputs and parameter gradients across pre/post/parallel norm paths and checkpoint modes, plus cached token decoding. Existing weight-specific architecture tests now request diagnostics rather than silently skipping. Benchmark diagnostics support the explicit flag and legacy baseline signatures. Validation and migration details: docs/architecture/LOWER_CORE_CONTRACTS.md.


## Lower-core checkpoint policy consolidation

Unified model and standalone layer policy creation through the existing factory. Unknown strategies no longer silently fall back; invalid strides/counts/indices fail early and explicit overrides are honored. Standalone selective checkpointing now selects its only layer. Documented adaptive as the existing static heuristic, not memory adaptation. Moved attention divisibility validation from obsolete seq_proj_dim to actual embed_dim while preserving legacy untied checkpoint behavior. Added selection, override, validation and standalone backward regressions.


## Cross-subsystem lifecycle consolidation

Unified checkpoint envelope interpretation and saved construction identity across saver, two loaders, manager and active training restore. Replaced config mutation after restore with pre-load compatibility checks, cleared KV state, removed ignored optimizer/scheduler failures and facade random-model fallback. Centralized V2/V3 source-and-exact-content split, removed mismatched cache selection, isolated cache temporary files, fixed manager no_grad/wrapper argument propagation, made expensive core diagnostics explicit, scoped cognitive notes and summaries, and streamed atomic model saves without full RAM serialization buffers. Detailed migration, evidence and remaining architectural boundaries: docs/architecture/LIFECYCLE_CONSOLIDATION.md. Broad verification encountered actual disk exhaustion; failed evidence is retained rather than reported as passing.


## Checkpoint/tokenizer identity completion

Connected actual tokenizer fingerprints to manager saves, active V2/V3 training configuration and checkpoint envelopes. Loaders and training resume reject equal-vocabulary-size/different-token-ID checkpoints before weight mutation. Added tokenizer arguments to standalone load APIs, preserved identity across direct resave, rejected rebranding an identified model with another tokenizer, and removed V2 training's swallowed restore error. Legacy identity-free checkpoints remain explicit unverified compatibility with a warning. No vocabulary migration or heavy training. Evidence: benchmarks/results/tokenizer_checkpoint_verification.xml.

## 2026-09-20 — Documentation aligned with the engine and the next development round

Rewrote both root READMEs around the engine's actual tokenizer, neural core, training, generation and cognitive flow. Preserved the developer's six real training-output images, project identity and training-data link. Documented V4–V8 source history without confusing it with the independent TrainingService V3, active TrainingManager V2, cognitive V2 or configuration-schema versions. Corrected preparation/launch paths and provided a small CPU example and checkpoint-loading guidance.

Updated the Turkish and English entry guides for neural_network, training_system, training_management and cognitive_management. Training management now describes the actual learning loop rather than duplicating the training service guide. Added a module index distinguishing refreshed entry guides from older documents. Reorganized the current architecture specification; corrected lifecycle notes that still listed completed tokenizer identity and checkpoint-policy work as open.

The next development roadmap records five reproduced cognitive faults: missing final system instructions, token-text/ID confusion in entropy, truncated calculator expressions, absent async ToT wiring and shared critic feedback. It also records model/cache ownership, training checkpoint RAM buffering, partial resume failure, whole-dataset alignment and quality/installation work. These are open findings, not fixes performed in this documentation pass. In particular, ModelSaver's streamed write does not yet extend to the V2 training checkpoint writer.

Validation: the README ModelManager example ran on CPU and returned finite logits shaped `[1, 4, 128]`. Local links, referenced headings, retained image inventory, Markdown fences and Python snippet syntax were checked; module guides were cross-read against their source paths. No new training, checkpoint migration, commit or push was performed.

## 2026-09-20 — Language representation correction and system synthesis

Corrected an inaccurate README claim that working in another language requires a tokenizer prepared specifically for that language. One tokenizer can represent multiple languages; preprocessing coverage and token efficiency are distinct from the language capabilities learned through model training. Cevahir is now described as having tokenizer infrastructure developed specifically for Turkish, with configurable syllabification/morphology and explicit current character-coverage boundaries.

Reworked both README introductions around the architectural contribution of the project: control over representation, a configurable trainable model, cognitive use of the learned model and conversational application integration. Followed the main data, training, model, cognitive, chat and application connections in source, including the separate preparation tools, profiling and health interfaces. Added SYSTEM_OVERVIEW.md to explain these relationships and refreshed both tokenizer entry guides. Preserved the author's identity and the real training-output images; moved detailed recent repair discussion out of the main project narrative into the existing architecture/development references. No production logic or trained assets changed.


## 2026-09-20 — Research and educational direction; source-linked book

Added the modular Turkish book **Cevahir AI — Bir Yapay Zeka Motorunun Anatomisi**, with twelve chapters and an English reading guide, linked prominently from both READMEs. The project now explicitly continues as an open-source research and educational legacy. Existing bilingual guides, author text, training images, source code and research records remain in place. The book connects mathematical concepts to current configuration, callers, actual Python symbols, input/output contracts and evidence. Historical guide/API differences are identified rather than copied as current behavior.

The standard-library book checker validates local inline links, registered Python symbols and reviewed-source fingerprints. Text fingerprints normalize checkout line endings. Its CI workflow only checks and never silently refreshes review records. Semantic review remains necessary. Maintenance instructions require relevant chapters, module references and research history to move with meaningful changes. This local work did not dispatch the GitHub workflow, commit or push.

Research continued separately with the correction-state audit. Two histories with identical current version spaces require different outcomes after the same old event is corrected. For a fixed finite hypothesis class and trusted old event contents, mismatch counts exactly reproduce full replay; a stronger six-bin histogram matches them and may use fewer counters. All 1,507,050 finite class/edit cases and 18,378 sequential intermediate states were checked, alongside a fresh-process continuation. ID-only corrections and enlarged hypothesis classes expose distinct insufficiencies. This is a bounded result with established truth-maintenance/incremental-maintenance connections, not a solution of the general living-learning problem. Prior preservation chains verified 88 existing research files unchanged.

Validation for this book turn: 82 selected attention/core/MoE tests plus 67 training/cache/checkpoint/tokenizer/runtime tests passed on Python 3.14.3 and PyTorch 2.10.0+cpu, with six deprecation-related warnings. The chapter-one real tiny-model example produced finite logits shaped `[1,4,128]`; eleven Python fences parsed. Checker boundary probes exercise missing targets, invalid lines/headings, path escape, renamed symbols and CRLF/LF equivalence. The whole repository suite, trained-language quality and GPU performance were not evaluated. Application code, token assets and trained checkpoints were not modified.


## 2026-09-21 — Research publication series and future-query state audit

Prepared nine research-report editions under Muhammed Yasin Yılmaz's name, each with Turkish and English abstracts, explicit scope and the full source report. Original research records remain unchanged: 94 files are checked byte for byte against commit 58cf9a2; the older witness benchmark is checked with its existing Git line-ending normalization. The publication catalog, CFF citation, Zenodo metadata and release procedure connect the research archive to a versioned GitHub release. External DOI registration is recorded separately after the archive service confirms publication. AI assistance and the absence of external peer review are disclosed.

Research continued by comparing exact state classes under three future-query contracts. A signed-label summary suffices for all Boolean loss queries on a fixed finite input domain; identity-only label editing requires different information. Enumeration of 61,575 histories, 712,632 label writes and 358,020 appends found no count-formula or replay mismatches. The previous 1,507,050-case correction experiment was also reproduced without changing its stored result, and 15 witness tests passed. This is a selected reproducibility audit, not a rerun of every historical experiment or the whole model test suite.

The book's open-questions chapter explains this refinement without equating it to a general solution of living-learning. Publication checks validate complete editions, preserved evidence, local navigation and consistent author/version metadata; CI also runs these checks. No model, training runtime or production integration was changed in this publication step.

## 2026-09-21 — Verified archive publication and executable tokenizer chapter

Published release research-2026.09.21-v1 at commit 5ae1e2f876ef53f14a170215e8c32414ccd2f3a2 through the authorized GitHub–Zenodo integration. Zenodo registered version DOI 10.5281/zenodo.22864661 and concept DOI 10.5281/zenodo.22864660 under Muhammed Yasin Yılmaz's name. The version DOI resolves to the expected record. All 829 archived file paths and contents match the tagged Git archive byte for byte; evidence records the archive checksum and comparison method. This verifies deposited metadata and source correspondence, not personal identity, scientific correctness or external peer review. Nine reports and the book share one software archive DOI. Zenodo's September 20 UTC publication date corresponds to September 21 in Europe/Istanbul.

Expanded book chapter 2 after the archive release with a topic-to-file/method map and six reproducible examples using the actual 60,000-entry vocabulary and 24,896 loaded merges. The chapter follows normalization, intermediate pieces, ID expansion, decoding, embedding row selection, configuration precedence, training-mode defaults, preprocessing loss and skipped batch items. The read-only walkthrough reproduces its recorded outputs and verifies that vocabulary and merge bytes do not change. All 24 existing tokenizer contract tests passed on CPU. No tokenizer implementation or trained assets changed. This later chapter expansion is available on the main branch; it is not retroactively part of the immutable v1 DOI snapshot.
