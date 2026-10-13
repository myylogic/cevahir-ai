# Living-learning campaign, round 23

Local-only research started 2026-09-24. Preserve previous work; do not commit,
push, publish, or integrate into the application as part of this campaign.

The 200-entry registry has 48 executable conditional comparisons in four
engines and 152 queued propositions. Entries overlap and are not 200 independent
mechanisms. Seed repetitions do not increase the hypothesis count.

- `registry.json`: full registry, including status and falsifiers.
- `results/freeze.json`: source hashes, environment, discovery/confirmation seeds.
- `results/preflight.json`: independent references, ablations, saved-state continuation.
- `results/discovery/*.jsonl`: seeds 0–23, all outcomes.
- `results/confirmation/*.jsonl`: seeds 1000–1047, all outcomes.
- `results/summary.json`: all 48 analyses, 96-direction Holm correction.
- `results/verification.json`: full coverage, deterministic reruns, preservation.
- `results/preservation_baseline.json`: pre-existing file hashes and Git HEAD.
- Turkish catalog/protocol/results/report: `docs/research/living_learning_campaign_2026_09_24/`.

Python 3.14 and NumPy 2.4 were used; exact versions are in the freeze. No model
download, GPU, external dataset or network access is needed to reproduce runs.

From the repository root:

```powershell
python research/living_learning_campaign_2026_09_24/campaign.py run
python research/living_learning_campaign_2026_09_24/campaign.py analyze
python research/living_learning_campaign_2026_09_24/campaign.py verify --rerun all
```

`run` resumes an existing validated prefix, refusing changed frozen source.
`verify --rerun all` recomputes every engine lifetime without replacing raw data.
`prepare` was run once, before the main runs; it refuses to overwrite the freeze.
Future engines/thresholds require a separate version/wave and untouched seeds.
Do not alter the frozen sources to retrofit a preferred result.

Preflight assertions are software checks, not empirical hypothesis tests. A
deterministic rerun is reproducibility, not an independent scientific replication.
Virtual operation counters differ across engines; error reduction alone is not
resource superiority. The broad living-learning problem remains open.
