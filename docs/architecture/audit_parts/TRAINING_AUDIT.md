# Training and data pipeline audit

Audit date: 2026-09-14. Scope: `training_system`, `training_management`, `data_loader_management`, `data_processing`, and the model initializer interfaces these modules call. This report records the pre-remediation source state. No training, test, scraping, cache preparation, or application-code mutation was performed by this audit worker. Findings marked broken are established by incompatible signatures or control flow; GPU performance and numerical claims are not experimentally verified. The parent reports Python 3.14 / PyTorch 2.10 CPU and an initial root test selection of 101 passed, 6 skipped; those results are not evidence that the V3 entry point works.

## Actual runtime ownership

1. `training_system/train.py:667` seeds the process, normalizes `TRAIN_CONFIG` with tokenizer settings, and selects `TrainingServiceV3` whenever it imports (`:735-758`). Import failure selects the V2 service (`:762-781`). Runtime initialization failures do not select a compatible fallback.
2. V3 service validates BPE assets and creates `TokenizerCore`, a V3 cache object, and `ModelManager` (`training_system/v3/core/training_service_v3.py:103-150`). It builds optimizer and scheduler before verifying that usable cached training data exists.
3. **The V3 service currently cannot complete initialization**: it passes `entropy_coeff` to the V2 `CriterionManager.create_criterion` (`:158-166`), whose signature ends at `eos_weight` (`training_system/v2/core/criterion_manager.py:50-58`). Python raises an unexpected-keyword `TypeError` after model allocation.
4. Once that blocker is fixed, V3 loads a checkpoint, strict cache, source split, V3 dataloaders, and translated configuration (`training_service_v3.py:426-486`). Its manager-selection code imports V3, but unconditionally instantiates **V2TrainingManager** (`:499-504`, `:506-535`, `:560-574`). Checkpoint, logging, TensorBoard and scheduler collaborators are also V2.
5. The active training implementation therefore remains `training_management/v2/core/training_loop.py`: model forward, tuple element zero as logits, masked cross entropy, accumulation, gradient clipping, optional CUDA AMP, optimizer step, metrics. V3's rich loop and manager are separate implementations with additional incompatible collaborator contracts.
6. Data preparation is a separate command: `tokenizer_management/train_bpe.py` → `training_system/prepare_cache.py` → `training_system/train.py`. Preparation still uses the legacy `DataCache`; V3 checksum/metadata are added afterward (`prepare_cache.py:45-48`, `:545-575`).

## Capability inventory

Statuses describe the normal training entry point unless explicitly labeled standalone.

| Capability | Status | Source evidence and effective behavior |
|---|---|---|
| V3 service startup | BROKEN | Unexpected `entropy_coeff` argument, service `:158-166` versus criterion `:50-58`. |
| V3 training backend selection | BROKEN | V3 import flag is unused; service `:563` always creates V2 manager. |
| V2/V3 training stacks | DUPLICATED | Two managers, loops, schedulers, checkpoint schemas, safety and monitoring stacks. V3 service combines V3 data with V2 training. |
| Strict cache existence | IMPLEMENTED | `training_system/v3/data/cache_v3.py:259-307` requires an exact file match and raises on absence. |
| `cache_strict_mode=False` | CONFIGURED_BUT_UNUSED | Stored in cache construction, but `load_for_training` always calls `load_strict` (`cache_v3.py:459-495`). |
| Cache corruption check | PARTIAL | SHA-256 verified when present, but missing checksum is accepted even with integrity enabled (`cache_v3.py:202-212`). |
| Dataset/tokenizer cache identity | PARTIAL | Data hash is relative filename plus byte size, not content (`cache_v3.py:103-122`); vocabulary hash includes token IDs (`:124-134`), not BPE merge content or preprocessing version. Same-size content edits need not invalidate cache. |
| Legacy cache validity | BROKEN | Legacy `DataCache.get_cached_data` defaults both key and data-hash mismatch allowances to True (`training_system/data_cache.py:144-151`). Existing unrelated data can be reused by fallback. |
| Source-group train/validation split | PARTIAL | V3 groups all chunks with matching source IDs (`training_service_v3.py:315-365`), but tests only first row for source-ID availability (`:303-313`). Missing IDs trigger sample-random split. One source gives zero training sources with 0.8 ratio (`:342`). |
| Content deduplication | PARTIAL | Preparation deduplicates `(content_hash, source_id)` (`prepare_cache.py:322-340`); identical content from different files is retained and can span splits (`:433-437`). |
| Next-token target verification | PARTIAL | Preparation constructs BOS/input and target/EOS (`prepare_cache.py:232-274`); V3 validation only rejects equality of the first input and target, not an incorrect shift elsewhere (`training_service_v3.py:270-285`). |
| Bucket batching | IMPLEMENTED | Sorted length buckets and per-bucket shuffling in `sampler_v3.py:95-138`; service creates this loader. Small buckets can generate many undersized microbatches. |
| Epoch shuffling | BROKEN | Sampler RNG is `seed + _epoch` (`sampler_v3.py:114-115`), but `set_epoch` is called only at loader construction (`dataloader_v3.py:132`), never from V2 or V3 training loops. Repeated epochs use identical bucket order. |
| Dynamic padding on actual prepared cache | BROKEN | Preparation pads every sequence to global maximum (`prepare_cache.py:263-274`); collator uses stored tensor length (`collator_v3.py:79`) and dataset returns original padded tensors (`dataset_v3.py:71-87`). Computed non-PAD lengths influence buckets but do not trim tensors. |
| Multiprocess loading / pinned memory | PARTIAL | DataLoader workers, pinning, prefetch, persistence wired (`dataloader_v3.py:134-165`). Active V2 transfer uses blocking `.to(device)` (`training_loop.py:277`); collator `non_blocking` field is unused. No dedicated device prefetch stream. |
| Cross entropy / EOS weighting / label smoothing | IMPLEMENTED | V2 loss reads weights and smoothing from criterion and masks PAD (`v2/core/loss_computation.py:128-163`). It does not call the criterion's forward method. |
| Entropy / focal / auxiliary loss | CONFIGURED_BUT_UNUSED | ConfigV3 forwards settings (`config_manager_v3.py:125-134`) but active V2 loss computes only cross entropy. V2 loop discards tuple auxiliary outputs (`v2/core/training_loop.py:281-305`). Standalone V3 composite loss exists (`v3/core/loss_manager.py:589-695`). |
| Scheduled sampling | CONFIGURED_BUT_UNUSED | Enabled in `train.py:576`, implemented only in V3 loop, not instantiated by entry point. Config names `ss_start_epoch`, `ss_decay_rate`, `min_teacher_forcing` do not match V3 loop dataclass names (`training_loop.py:111-116`). |
| EMA | CONFIGURED_BUT_UNUSED | `train.py:582` enables it, active V2 manager never constructs or updates EMA. Standalone V3 manager calls `ema.update(model=...)` (`v3/core/training_manager.py:526-527`), while provided EMA accepts `update(self)` (`v3/utils/ema.py:121`). |
| SAM / Lookahead | CONFIGURED_BUT_UNUSED | Wrapper implementations exist in `v3/optimizers`; normal service uses ModelManager optimizer directly (`training_service_v3.py:478-486`). V3 loop calls `optimizer.step()` without a closure (`v3/core/training_loop.py:729-734`); SAM requires a closure for two-pass operation (`optimizers/sam.py:248-290`). |
| SWA | CONFIGURED_BUT_UNUSED | Flags forwarded (`config_manager_v3.py:173-178`); checkpoint helper can save an externally supplied SWA state, but no normal training path creates or updates AveragedModel/SWALR. |
| LLRD | CONFIGURED_BUT_UNUSED | Standalone builder `v3/utils/training_scheduler.py:512` exists; service takes prebuilt ModelManager optimizer and does not call it. |
| AGC / gradient noise | CONFIGURED_BUT_UNUSED | V3-only loop implementations exist; active V2 ignores flags. Adapter names `use_agc`, `agc_clip_factor`, `gradient_noise_eta` disagree with V3 loop's `use_adaptive_clip`, `agc_lambda`, `noise_eta`. |
| Curriculum | CONFIGURED_BUT_UNUSED | V3 only; even explicit injection calls missing `adjust_data` (`v3/core/training_manager.py:478-483`), while provided collaborator exposes `filter_dataloader` (`curriculum_manager.py:311`). |
| Gradient accumulation | PARTIAL | V2 divides each microbatch loss by configured accumulation count (`training_loop.py:366`); remaining partial group is stepped without correcting denominator (`:460-476`). Different valid token counts receive equal microbatch weight. |
| Gradient clipping | IMPLEMENTED | V2 unscales AMP gradients then clips before step (`training_loop.py:377-411`). Non-AMP nonfinite gradient rejection is incomplete. |
| CUDA fp16 AMP | IMPLEMENTED | V2 CUDA GradScaler (`training_loop.py:127-128`) and default CUDA autocast (`:192-196`). GPU execution not tested here. |
| bf16/precision policy | PARTIAL | V3 loop accepts `amp_dtype`, but adapter does not forward it and active V2 has no configurable dtype. FSDP wrapper separately hardcodes bf16. No centralized capability-based precision selection or CPU bf16 policy. |
| Scheduler configuration | BROKEN | Service passes nested `scheduler_kwargs` (`training_service_v3.py:527-535`); V2 scheduler expects flattened kwargs (`v2/utils/training_scheduler.py:198-207`, `:265-327`). Nondefault values vanish; OneCycle construction fails without visible total_steps. |
| Scheduler update semantics | PARTIAL | V2 steps per optimizer update during warmup and per epoch after warmup (`v2/core/training_loop.py:198-224`, `v2/core/training_manager.py:458-472`). Partial accumulation update omits warmup step; OneCycle should continue stepping per update. |
| Checkpoint atomic write / aliases / rotation | IMPLEMENTED | V2 helper `utils/checkpoint_manager.py:123-208`, `:330-446`. These are real writes and rotation, not only documentation. |
| Last / periodic checkpoint policy | CONFIGURED_BUT_UNUSED | Active manager saves only new best validation epochs (`v2/core/training_manager.py:525-565`). Helper's last alias consequently means last best save, not last epoch. V3 config `save_every_n_epochs` is ignored by active manager. |
| Exact resume | BROKEN | Service requests model load but passes no restored start epoch to new manager (`training_service_v3.py:602-608`, `:563-574`); manager defaults start_epoch=1. Active saved payload lacks scheduler, scaler, RNG, sampler position (`v2/utils/checkpoint_manager.py:171-185`). Resume is not trajectory-equivalent. |
| Standalone V3 checkpoint integration | BROKEN | V3 manager calls `save_best`, `save_last`, `save_periodic` (`v3/core/training_manager.py:907-921`), whereas provided V3 helper exposes `save` (`v3/utils/checkpoint_manager.py:171`). Helper payload names `model_state`, `optimizer_state` differ from manager resume names `model_state_dict`, `optimizer_state_dict`. |
| Standalone V3 scheduler integration | BROKEN | Manager calls `scheduler.step` (`v3/core/training_manager.py:539-539` onward); provided V3 scheduler exposes `step_batch` and `step_epoch` (`utils/training_scheduler.py:350`, `:379`). |
| V3 configuration translation | BROKEN | `from_config` silently filters keys by dataclass names (`v3/core/training_manager.py:979-1005`). Adapter provides `epochs`, manager expects `total_epochs`; multiple feature names similarly mismatch. |
| Error propagation | PARTIAL | V2 training batch exceptions raise (`v2/core/training_loop.py:452-458`), but invalid logits/NaN loss are skipped. V3 `_safe_call` suppresses collaborator errors (`v3/core/training_manager.py:391-417`) and failed train epochs continue (`:487-501`). Missing save methods may silently prevent checkpoints. |
| Token-weighted evaluation | PARTIAL | Active V2 averages per-batch mean loss/accuracy equally (`training_loop.py:437-439`, `:486-490`); variable-length/undersized batches distort corpus token metrics. |
| Deterministic seeding | PARTIAL | Python/NumPy/Torch/CUDA seeds and cuDNN flags (`train.py:130-146`). No RNG checkpointing; service does not forward loader seed; source-set iteration can alter ordering for string IDs. Inference probes consume global sampling RNG. |
| `torch.compile` | PARTIAL | Real optional initializer call and eager fallback (`model_management/model_initializer.py:230-242`). No backend selection/benchmark gate; lazy first-forward compile failures are outside construction try block. |
| DDP / FSDP | PARTIAL | Initializer has real wrappers (`model_initializer.py:659-746`), but returns unwrapped model if process group is absent (`:687-692`). Training code does not initialize group, set local rank from environment, shard data, reduce metrics, use no_sync, or restrict saves to rank zero. End-to-end distributed training is not implemented. |
| Activation checkpointing | PARTIAL | Model initializer forwards flag and checks model API (`model_initializer.py:244-252`); exact model-side implementation belongs to neural-network audit. No measured recompute/memory evidence here. |
| Streaming / mmap / resumable datastream | NOT_IMPLEMENTED | Cache unpickles whole corpus to list (`cache_v3.py:309-318`), split tensorizes all records (`training_service_v3.py:383-402`). Loader stores list in memory. |
| Measured GPU speedup claims | DOCUMENTED_ONLY | `train.py` comments promise fp16/FlashAttention/prefetch gains; this audit found no benchmark output proving those claims for current end-to-end runtime. |

## Data ingestion and processing

`data_loader_management/data_loader_manager.py` is an actual multi-format loader with QA, text-input and raw-text modes. It splits long text, handles recognized QA/instruction/sentiment records, and can attach file indices (`:283`, `:326-364`, `:420-488`, `:588-689`). File IDs are incrementing counters, so adding or reordering input files can change IDs and train/validation membership. The QA source-ID branch iterates configured QA extensions but processes only `.json` (`:436-481`); DOCX support in the ordinary branch does not imply that source-ID QA mode reads DOCX.

Preprocessing is not a validated immutable dataset build: there is no corpus manifest with source content hash, tokenizer/merge hash, preprocessing version, retained/excluded count, split assignment, license/provenance and quality score attached to each training record. Existing statistics and source IDs are useful foundations, but do not supply that contract.

`data_processing/wikipedia_api_scraper.py:63-77` collects title/text/URL/character length. TXT output discards source metadata; JSON retains URL/title (`:128-143`). Its `total_tokens` metric increments character length (`:147`), not tokenizer counts. `get_random_pages` explicitly returns an empty list (`:100-104`): DOCUMENTED_ONLY/unimplemented behavior behind a method name. Topic-based scraping writes plain TXT (`topic_based_scraper.py:171-183`) with no content manifest. These tools use short inter-request delays, but the review did not find dataset-wide provenance or quality filtering.

`subtitle_processor.py:31-92` parses SRT/VTT, removes HTML tags and normalizes whitespace. `save_as_json` treats adjacent subtitles as QA pairs without checking speakers, timing, question structure or whether the second line is a continuation (`:143-159`): the conversion is IMPLEMENTED, conversational-label quality control is NOT_IMPLEMENTED. UTF-8 decoding uses `errors='ignore'` (`:113`), so encoding damage may be silent. Conversion and scraper tools are offline preparation utilities, not part of the core training hot path.

## Priority and verification plan

P0 correctness blockers before any expensive run:

1. Repair the service/criterion signature mismatch and prove V3 startup with stub tokenizer/cache/model, without real assets or GPU allocation.
2. Choose an explicit training backend and wire its real collaborators. Do not merely switch the V3 constructor: EMA, checkpoint, scheduler, curriculum and configuration contracts above must be reconciled first.
3. Fail explicitly when enabled features are unsupported, checkpoint writes fail, or all training/validation batches are invalid. A successful log must not represent zero optimization steps or missing checkpoints.

High-value CPU tests, in order:

- Service startup and selected backend identity; assert requested loss coefficients alter the loss and gradients.
- Tiny model fixed batch overfit; independent expected token-normalized loss and finite gradients; auxiliary-loss gradient reaches router when enabled.
- Full batch versus equivalent microbatch updates, including incomplete final accumulation group, unequal non-PAD token counts, NaN/invalid batch and scaler-skipped-step scheduling cases.
- Nondefault scheduler kwargs reach the real scheduler; warmup and OneCycle advance per optimizer step exactly once.
- Prepared padded records become compact dynamic batches while preserving shifted labels/EOS; every record is visited once; bucket order changes by epoch reproducibly; empty/single-source splits fail with a useful error.
- Cross-source duplicate content cannot cross a strict split; mixed/missing source IDs fail or use a declared policy; same-size corpus edits and merge-file changes invalidate cache; missing/incorrect integrity sidecars follow explicit policy.
- Save last/best/periodic with real collaborator; round-trip model, optimizer, scheduler, scaler, loop/early-stopping state, RNG and sampler position. Compare uninterrupted versus resumed next update.
- V3 component contract tests for EMA, curriculum and checkpoint; unsupported SAM/SWA/distributed modes fail explicitly until verified integration exists.
- Data fixture tests for QA JSON, instruction/sentiment JSON, TXT, DOCX with file IDs, source metadata retention, Turkish encoding, SRT/VTT malformed and continuation lines.

Existing candidates: `training_management/v2/test/core/`, `training_management/v2/test/integration/`, `training_system/v2/test/test_training_service_comprehensive.py`, `tests/test_prepare_cache_pipeline.py`, `tests/test_training_loop_gradient_flow.py`, and checkpoint tests in `tests/`. Much of this coverage targets V2 or external assets; some cache tests skip when preparation fails (`tests/test_prepare_cache_pipeline.py:308-313`), so pass counts alone do not establish entry-point health. No dedicated V3 test directory was present at audit time.

GPU-only later gates: fp16/bf16 finite-loss parity, peak allocated/reserved memory, steady-state tokens/sec and compilation warmup overhead, activation checkpoint equivalence, asynchronous loader transfer measurement, and multi-process data/metric/checkpoint correctness. CPU tests cannot certify these performance claims.
