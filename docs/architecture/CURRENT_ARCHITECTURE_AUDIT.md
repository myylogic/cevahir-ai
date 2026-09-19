# Cevahir current architecture audit

Date: 2026-09-14. This is the **pre-change** inventory, not a claim of production readiness. Later corrections and their evidence belong in `docs/development/EVOLUTION_LOG.md`.

## Scope and evidence

The original tree contains 608 files, including 513 Python files. Every Python file was parsed successfully; syntax validity does not establish runtime correctness. `SOURCE_INVENTORY.json` records paths, SHA-256 hashes, imports, top-level classes and line counts. Module audits in `audit_parts/` follow construction, call sites and consumers instead of relying on version labels or comments. The source snapshot is `.evolution/snapshots/original-source-2026-09-14.zip`; this supplied directory has no Git metadata.

Runtime: Python 3.14.3, PyTorch 2.10.0+cpu, no CUDA device. Initial selected architecture suite: **101 passed, 6 skipped**. `BASELINE_INVESTIGATION.json` contains independent reproductions: all six MHA/GQA/MQA manual/SDPA cache comparisons fail (maximum logit errors 0.15488–0.47634); sliding cache eviction raises an overlapping-memory error; invalid zero heads raises ZeroDivisionError instead of a validation error. These expose gaps in the existing tests.

Status is scoped to the actual application path. IMPLEMENTED means an executable implementation and a reachable consumer were found; it does not certify every configuration, hardware backend or model quality. PARTIAL means working pieces with missing integration/verification; BROKEN means a concrete incorrect path; DUPLICATED means multiple competing implementations; CONFIGURED_BUT_UNUSED means accepted settings lack an effective consumer; DOCUMENTED_ONLY means claims lack implementation; NOT_IMPLEMENTED means no implementation found in the inventoried code.

## Capability inventory

| Capability | Initial status | Evidence and practical limit |
|---|---|---|
| MHA | IMPLEMENTED | `src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py`: Q/K/V projections and output projection execute in Transformer layers; full-sequence tests pass. Cached path is separately broken. |
| MQA / GQA | IMPLEMENTED | Same class constructs smaller K/V projections with `num_kv_heads`, expands after compact caching. Cached correctness is not established by this existence. |
| PyTorch SDPA | IMPLEMENTED | `scaled_dot_product_attention` calls `F.scaled_dot_product_attention`; CPU full-sequence tests execute it. |
| External Flash Attention | PARTIAL | Optional `flash_attn_func` path exists; no installed CUDA backend to validate. Fallback can lose the causal mask. SDPA does not guarantee a Flash kernel. |
| RoPE | BROKEN | `PositionalEncoding.apply_rotary_pos_emb` supports positions, but MHA does not pass cache positions. Full-sequence path exists; incremental positions restart at zero. |
| YaRN / linear scaling | PARTIAL | Frequency scaling in `positional_encoding.py`; configuration pass-through and long-context quality remain unverified. |
| Sliding window attention | BROKEN | MHA materializes a window mask; uses pre-cache key length and zero-based query rows during incremental decoding. This is a dense mask, not a sparse kernel. |
| QK-Norm | IMPLEMENTED | MHA calls Q/K RMS normalization before rotary encoding. |
| Attention logit soft-cap | CONFIGURED_BUT_UNUSED | Manual branch applies tanh cap; default SDPA branch bypasses it. |
| RMSNorm | IMPLEMENTED | `rms_norm.py`, layer input norms and output norm; numerical tests execute the path. |
| SwiGLU | IMPLEMENTED | `feed_forward_network.py`: gated projection, SiLU and output projection; used by layers and experts. |
| Parallel residual | IMPLEMENTED | `transformer_encoder_layer.py` selects parallel attention/FFN residual contributions. Defaults differ across entry points. |
| Stochastic depth / LayerDrop | PARTIAL | Per-example residual DropPath is implemented and layer rates are assigned; it is not compute-skipping LayerDrop. |
| KV cache | BROKEN | Cache storage/update exists; cached logits differ materially from full-prefix logits. See baseline reproductions. |
| StreamingLLM eviction / attention sinks | BROKEN | `kv_cache.py` retains a prefix and sliding tail; overlapping copy fails on overflow; explicit absolute positions conflict with bounded storage. Root constructor does not forward all cache options. |
| Mixture of experts | IMPLEMENTED | `mixture_of_experts.py` sparse expert loop invokes only assigned experts and aggregates weighted outputs. |
| Top-k routing / router jitter | IMPLEMENTED | Router computes logits, training jitter, top-k indices and weights, consumed by dispatch. |
| MoE load-balancing loss | BROKEN | Scalar differentiable auxiliary objective exists; layer accumulates it, but no trainer calls `get_and_reset_moe_loss`. It is not part of the active training objective. |
| Gradient checkpointing | PARTIAL | Standard non-reentrant and advanced wrappers execute; MoE mutable loss accumulation during recomputation needs correction. |
| Quantization | BROKEN | Core dynamic quantization discards the returned replacement model; configuration advertises unsupported int4. Other optional paths require hardware/dependencies. |
| Turkish tokenizer morphology | PARTIAL | Rule-based morphology/syllable fallback and BPE implementation exist. Manager drops nested config, singleton ignores configuration identity; exact whitespace is normalized. |
| RAG / memory | PARTIAL | Retrieval/vector stores and prompt enrichment exist; retrieval quality unmeasured, persistent turn IDs can collide after truncation/restart. |
| Tools | BROKEN | Executor/registry exist but runtime handler only mentions selected tool in the prompt; pipeline reports it as used without execution. Default search/file tools are placeholders. |
| Critic | PARTIAL | Heuristic/evaluator components and pipeline checks exist; effectiveness not measured. |
| Direct / Think / Debate / ToT | PARTIAL | Strategy router and generation orchestration exist in cognitive V2; names are not evidence of task success. |
| Monitoring | PARTIAL | Metrics, health, gradient/token monitors exist; some paths use missing or mismatched helper APIs. |
| Tracing | PARTIAL | Middleware/event logging exists; shared mutable trace/span state can mix concurrent requests. |
| Inference generation | BROKEN | `model/cevahir.py` calls forward repeatedly, but temperature=0 and max_new_tokens=0 are replaced by defaults; cache is forced, and early EOS is suppressed. |
| Beam search | BROKEN | Private method has no normal dispatch, shares cache across competing beams, normalizes truncated logits, and its exception path references missing `self._model_api`. |
| Top-k / top-p sampling | IMPLEMENTED | Active autoregressive path filters logits; greedy/zero argument semantics need repair. |
| Batch / streaming generation | PARTIAL | `generate_batch` is a sequential loop with swallowed per-item errors. No genuine continuous batching or token streaming implementation found. |
| Precision / compile / distributed | PARTIAL | Existing AMP, compile and DDP/FSDP code is fragmented; lazy compilation failures and complete resume need verification. See training audit. |
| Native multimodality | DOCUMENTED_ONLY | Processor adapters delegate to external objects; no vision encoder/projector/token-level fusion in the language backbone. |

## Integration and configuration findings

`training_system/train.py` prefers V3 service, but V3 construction passes an unsupported `entropy_coeff` to the V2 criterion factory. Its actual training path constructs V2 manager regardless of the V3 import flag. V3 manager itself has incompatible checkpoint/scheduler/EMA/curriculum helper calls hidden behind exception swallowing. Switching to it without fixing contracts would add failures.

There are two different `CevahirConfig` classes (inference facade and model management), explicit training presets, low-level constructor defaults, API presets and V2/V3 normalizers. The typed schema omits implemented capabilities and silently drops unknown fields. Dimensions and defaults disagree between paths. Legacy explicit dimensions must be preserved for checkpoint compatibility; migration must distinguish missing defaults from explicit overrides.

Flask entry points import absent `config.parameters`. The factory registers a blueprint before decorating its routes. Database declarative models use the reserved attribute `metadata`. Flask/SQLAlchemy are absent initially, so these findings require isolated dependency installation and runtime tests before being recorded as repaired.

Many legacy tests reference modules removed from the supplied tree, require an external trained checkpoint, or catch errors and return False without failing pytest. Passing a subset must never be represented as a clean repository-wide suite. No training corpus content was uploaded or used for long training runs.

## Priority and scope after user clarification

P0: preserve baseline; fix configuration contracts, cached causality/positions, invalid argument handling, actual training loss composition, startup/import failures, and deterministic small CPU regressions.

P1: consolidate existing runtime precision/compile/checkpoint handling; make measurements and tokenizer/agent behavior truthful; clean unsupported comparison claims; verify existing capabilities together.

The user subsequently asked to equalize and repair the existing system on limited hardware before reevaluating further development. New MLA, MTP, MoE-vNext, Muon, Gated DeltaNet, gated residual, n-gram and native multimodal experiments are therefore **deferred by scope**, not marked implemented or accepted. No heavy training or GPU performance claim is authorized by these CPU measurements.
