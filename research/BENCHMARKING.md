# Benchmarking Cevahir Zihin

How to put Cevahir Zihin on the same board as Transformer models — reproducibly, on your own machine, without
waiting on a cloud session.

## What we compare, and why

The claim Cevahir makes is **not** "bigger than a large model". It is: a mind that learns **by living** — no
gradient, no weight matrices, seconds of compute, a fraction of the data — generalizes **exactly** to commands it
was never shown, and can say **why** for every word it answers (`!neden`). The honest, strong comparison is
therefore against a model trained in the **same setting**:

- **Same dialogues, from scratch.** Both learners see exactly the same training dialogues. Cevahir watches each
  once an epoch (no gradient); a small Transformer (3 layers, 4 heads, width 64) is trained from scratch on them.
  This is the fair control, and it is what `benchmark.py` runs.
- **Published baselines.** For the standard splits, seq2seq/Transformer numbers are already published (Lake &
  Baroni 2018; Loula et al. 2018). Citing them needs no GPU and no training.

A comparison against a **pretrained** large model (Llama, GPT-4, …) is a *different* question: it tests pretraining
plus in-context learning, not from-scratch data efficiency. Modern large models already solve SCAN add-jump
in-context, so that comparison flatters neither side and is not the strong claim. Worth running as a side note —
but see the caveats below.

## The scoreboard (reproducible, no download)

```
python research/living_mind_2026_11_15/benchmark.py            # quick smoke run (small, fast)
python research/living_mind_2026_11_15/benchmark.py --full     # the sizes used in the research reports
python research/living_mind_2026_11_15/benchmark.py --only add-jump,around-right --full
python research/living_mind_2026_11_15/benchmark.py --full --no-transformer   # the mind only, fast
```

It reuses `t_scan` / `t_scan_splits` / `t_dans`, adds no world, and downloads nothing. Both learners see the same
dialogues. The splits: **add-jump** (jump seen only alone), **length** (answers longer than any seen),
**around-right** and **jump-around-right** (a template never seen in training), and **t-dans** (a Turkish world F43
was not developed on).

## Running on your own machine (no GPU needed)

Cevahir is pure Python + CPU PyTorch — **no GPU, no HF account, no download**:

```
git clone https://github.com/myylogic/cevahir-ai && cd cevahir-ai
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt                     # or: pip install numpy
python research/living_mind_2026_11_15/benchmark.py --full
```

- The per-answer work is single-thread; the test harness forks `--workers` processes, so **more cores = faster**
  answering (near-linear). On an 8- or 16-core desktop the `--full` board finishes much faster than in a 4-core
  cloud container.
- The training ("life") is sequential per seed; parallelize across seeds by running seeds in separate processes.
- The real per-answer cost is the mind's own decisions (member votes at each moment), not I/O. A bit-identical
  speed pass on that hot path is the open performance task (`research/.../NOTES.md`).

## Comparing against Hugging Face models

- **Account:** a **free** Hugging Face account is enough to download most open models. Some are **gated** (Llama,
  some Gemma): you accept the license on the model page once, then use an access token
  (`huggingface-cli login`, or set `HF_TOKEN`). No paid membership is required for this.
- **Libraries:** `pip install transformers datasets huggingface_hub`.
- **What runs on CPU:** small models — GPT-2, T5-small/base, ByT5-small, Pythia-160m/410m — run on CPU in seconds
  to minutes. **Large models (7B+)** need a GPU or a lot of RAM; on CPU they are possible but slow, and best run
  quantized (GGUF via `llama.cpp`) for inference only.
- **The data:** SCAN is on the Hub (`datasets.load_dataset("scan", "addprim_jump")`, also `"length"`,
  `"template_around_right"`, …), and the **official** split files are at `github.com/brendenlake/SCAN` (a few MB,
  public). Using either is a **download to approve first** (per the project rule on datasets).
- **Caveat for a fair write-up:** if you compare against a big pretrained model, say so plainly — it is pretraining
  + in-context learning, a different setting from Cevahir's from-scratch, no-gradient learning. The result that
  stands up to scrutiny is the same-data, same-compute control above, plus the published seq2seq numbers.

## Beyond SCAN

The same method applies to the other standard compositional-generalization benchmarks; each needs a small adapter
(a world that turns its commands into watched dialogues) and your OK to download its data:

- **COGS** (semantic parsing, systematic gaps), **CFQ** (with MCD splits — maximum compound divergence),
  **PCFG / productivity & systematicity**, **gSCAN** (grounded SCAN).
- Published Transformer/seq2seq baselines for all of these live on **Papers with Code** and in the source papers,
  so a credible table can cite them without training anything large.

## Open item

To verify the generated splits byte-for-byte against the originals, the official SCAN split files
(`github.com/brendenlake/SCAN`, a few MB, public) can be downloaded — **with your approval**. Say the word and the
generated splits are checked against the published train/test files, and the exact published test sizes are used.
