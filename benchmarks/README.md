# Lightweight verification and comparison

Run from the repository root with Python, PyTorch, pytest, numpy, regex and the project's tokenizer dependencies installed. The recorded environment is Windows, Python 3.14.3 and PyTorch 2.10.0+cpu. No CUDA run or language-model training corpus is required.

## Regression verification

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest tests/evolution tests/architecture_standards/test_architecture_standards.py tests/architecture_standards/test_entropy_metrics.py tests/architecture_standards/test_neural_network_forward_comprehensive.py training_management/v2/test/core src/neural_network_module/test/test_multi_head_attention.py model_management/test/test_model_manager.py tokenizer_management/tests/test_bpe_decoder.py cognitive_management/tests/v2/test_memory_service_v2.py cognitive_management/tests/v2/test_policy_router_v2.py cognitive_management/tests/v2/test_orchestrator.py -q --junitxml=benchmarks/results/debt_verification.xml
```

This is a selected verification suite, not the entire legacy repository suite. API tests need Flask, Flask-Cors, Flask-Limiter, SQLAlchemy and PyJWT. On this workstation those packages were installed under `.evolution/deps`; the API test imports that directory if present and otherwise uses installed dependencies or skips with a reason. There was no global environment replacement. The real service training probe substitutes only the tokenizer asset boundary and uses one tiny synthetic batch.

## Original / repaired / delta

The original source snapshot is `.evolution/snapshots/original-source-2026-09-14.zip`, extracted at `.evolution/baseline-source`. The benchmark executes the same harness against each source root. Run sequentially, without simultaneous tests:

```powershell
python benchmarks/core.py --source-root .evolution/baseline-source --label original --output benchmarks/results/core_original.json
python benchmarks/core.py --label repaired --output benchmarks/results/core_repaired.json
python benchmarks/core.py --compare benchmarks/results/core_original.json benchmarks/results/core_repaired.json --output benchmarks/results/core_delta.json
python benchmarks/tokenizer_benchmark.py --baseline benchmarks/results/tokenizer_baseline.json --output benchmarks/results/tokenizer_repaired.json --repeats 5
```

Both core runs use seed 42, one CPU thread, 2 warmups, 5 measured iterations, vocabulary 128, embedding 32, two layers and synthetic tokens. JSON records source/data hashes and effective configuration. `--moe`, `--kv-heads 1` and `--manual` are optional diagnostic variants, not automatically accepted experiments. Preserve the original tokenizer baseline when rerunning the repaired tokenizer.

Core metrics cover parameter counts, linear/attention FLOPs estimates, KV tensor memory, prefill/decode/training throughput, forward/backward times, loss/perplexity, gradient norm, non-finite values, checkpoint size and cached/full-prefix equivalence. Router metrics are null for dense FFN; CUDA memory is null on CPU. Compile is disabled; graph breaks, recompilation counts and GPU kernel behavior are not measured.

Timings are tiny-workload observations on a shared computer. The repaired decode measurement is slower: correctness was the acceptance criterion for this phase, not speed. Synthetic loss/perplexity cannot establish language quality. Attention entropy is a separate manual diagnostic; FLOPs exclude nonlinearities, normalization, rotary and optimizer work.

Tokenizer results cover 8 Turkish and 6 English texts, Turkish character roundtrips, small morphology/fallback probes and warm cached throughput. Unsupported text may be dropped before UNK accounting; inspect `fallback_probes`, not just the UNK count. Exact roundtrip improved through punctuation spacing, with unchanged measured token IDs and unchanged shipped vocabulary/merges hashes. This corpus is too small to establish general linguistic superiority.


## Debt follow-up measurements

The initial repaired results above are preserved. The append optimization was measured separately in `core_cache_append.json`, with `core_cache_append_delta.json` comparing it against `core_repaired.json`. Reproduce a new measurement without overwriting that evidence:

```powershell
python benchmarks/core.py --label cache_append_current --output benchmarks/results/core_cache_append_current.json
```

Observed decode throughput was 97.44 → 366.94 tokens/s, with maximum cache-equivalence error 1.79e-7. These are small, separate single-machine observations, not a statistical speed guarantee. The attention unit test records warmed forward duration in JUnit properties; a hardware-specific budget can be requested with `CEVAHIR_ATTENTION_TEST_MAX_SECONDS`. There is no universal 100 ms requirement on unknown hardware.


## Repository synchronization verification — 20 September 2026

The selected `tests/evolution` suite passed: 158 tests, no failures, errors or skips, with 18 dependency/deprecation warnings. The summary is in [github_sync_verification_summary.json](results/github_sync_verification_summary.json). This is not a claim that the entire historical suite passes.

Published historical evidence replaces the local checkout prefix with `.` and the home-directory prefix with `$USERPROFILE`; JUnit hostnames are recorded as `local`. Test outcomes, failures and measurement values are retained. Unmodified originals remain in the local, ignored `.evolution/private-verification-originals/` directory.
