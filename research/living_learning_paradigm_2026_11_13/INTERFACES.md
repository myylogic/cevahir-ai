# CEV-11 "Paradigma" — module interfaces (implementation contract)

Study folder: `research/living_learning_paradigm_2026_11_13/` (called `STUDY` below). Package: `STUDY/cev11/`.
Scripts live in `STUDY/` and start with:

```python
import sys, pathlib
STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.path[:0] = [str(STUDY), str(REPO)]
```

Rules for every module:
- Python 3.13, torch (CPU), numpy, stdlib. No new dependencies (matplotlib optional, guarded).
- Never modify any existing repository file. Use `tokenizer_management/byte_bpe.py` (`ByteLevelBPE`, `BOS_ID=2`,
  `EOS_ID=3`, `PAD_ID=0`, `pretokenize`) unmodified, and build models only through
  `model_management.config_schema.tiny_model_config` + `model_management.model_manager.ModelManager`.
- Deterministic: every random choice takes an explicit `random.Random(seed)` / `torch.Generator`; no global RNG.
- Write files only under `STUDY/results/` (or a caller-given directory). Tests use `tmp_path` only.
- Module docstrings in Turkish or English; keep the signatures below. If a signature must change, document the
  change at the top of the module and in your report.
- Fast tests go to `tests/evolution/test_cev11_<area>.py` (total ≤ 10 s per file, tmp_path only). They import the
  package by inserting `STUDY` into `sys.path`.

Index conventions: token positions are indices into `ids = [BOS_ID] + tok.encode(prefix_text)`. A *prefix* is the
frame text up to, not including, `" " + obj + "."`. The full training sentence is `prefix + " " + obj + "."` and is
trained as one sequence `[BOS] ids [EOS]`.

---------------------------------------------------------------------------------------------------------------
## cev11/morph_tr.py — Turkish proper-noun case morphology

```python
CASES: tuple[str, ...] = ("nom", "gen", "dat", "acc", "loc", "abl", "ins")
INFLECTED: tuple[str, ...] = CASES[1:]
def suffix(root: str, case: str) -> str          # allomorph, no apostrophe; "" for nom. 2/4-way harmony,
                                                 # buffer n (gen) / y (dat, acc, ins), d→t after fçşhpstk
def formal(root: str, case: str) -> str          # "Ayten" | "Ayten'den" (apostrophe U+0027)
def informal(root: str, case: str) -> str        # "Ayten" | "Aytenden"
def tr_upper(s: str) -> str; def tr_lower(s: str) -> str; def tr_capitalize(s: str) -> str   # i/İ, ı/I aware
```
`STUDY/cev11/gold_forms.tsv`: 120 hand-checked rows `root<TAB>case<TAB>formal<TAB>informal` (header line).

## cev11/world.py — TR-Paradigma world generator

```python
RELATIONS = ("A", "B", "C")
NP = {"A": "memleketi", "B": "mesleği", "C": "en sevdiği yemek"}
CONTEXTS = {"none": "", "P1": "Bilindiği gibi, ", "P2": "Dün öğrendik ki "}
CARRIERS = {   # {S} = inflected subject (formal or informal), {NP} = relation NP; object follows after a space
  "nom": "{S} hakkında bilinen: {NP}", "gen": "{S} {NP}", "dat": "{S} dair bilinen: {NP}",
  "acc": "{S} tanıyan herkes bilir: {NP}", "loc": "{S} duran kimlik kaydına göre {NP}",
  "abl": "{S} bahsedenler bilir: {NP}", "ins": "{S} ilgili bilgi: {NP}"}

@dataclass(frozen=True)
class World:
    seed: int
    names: tuple[str, ...]                 # 400 invented one-word names, sorted
    objects: dict[str, tuple[str, ...]]    # "A": 48 towns, "B": 30 occupations, "C": 30 dishes
    facts: dict[tuple[str, str], str]      # (name, rel) -> object, uniform assignment
    partition: dict[str, str]              # name -> "p3" (60) | "c0" (60) | "probe" (280)
    p3_held: frozenset[tuple[str, str]]    # (name, case): 2 random frames of relation A per p3 entity, never trained
    prefix_pairs: tuple[tuple[str, str], ...]   # 20 pairs (n1, n2), n2 = n1 + one syllable
    tok_frequent: frozenset[str]           # designated 25% subset (exploratory F_freq arm)
    def name_hash(self) -> str             # sha256 over the canonical JSON of the world

def make_world(seed: int, n_names: int = 400) -> World
def frame_prefix(name: str, rel: str, case: str, orth: str = "formal", ctx: str = "none") -> str
def sentence(name: str, rel: str, case: str, obj: str, orth: str = "formal", ctx: str = "none") -> str
def name_char_span(prefix: str, name: str, ctx: str) -> tuple[int, int]   # [start, end) of the bare name (root)
def word_char_end(prefix: str, name: str, ctx: str) -> int                # end of the whitespace word holding it
def pass_sentences(world: World, pass_idx: int, rng: random.Random) -> list[tuple[str, dict]]
    # one pass: every (entity, rel, case) cell once except p3_held; per cell ctx ∈ none/P1/P2 with
    # p = .50/.25/.25 and, for inflected cases, orth = informal with p = .25. dict = cell metadata
def load_prose(repo_root: pathlib.Path) -> tuple[list[str], list[str]]
    # train: docs/book/tr chapters 01–09 prose (fenced code removed) + education/ texts EXCEPT files whose
    #        name starts with "mektup" and the health-formulation file (document the exact exclusion list);
    #        .docx via python-docx (guarded), .txt/.json read as UTF-8. Split into paragraphs.
    # heldout: docs/book/tr chapters 10–12 prose. Never print corpus samples.
def prose_chunks(texts: list[str], tok, max_tokens: int = 60) -> list[tuple[str, bool]]
    # (chunk_text, is_end): chunks of <= max_tokens cut at sentence/word boundaries; is_end is True only for the
    # last chunk of a paragraph. The trainer appends EOS only when is_end is True (no false EOS mid-text).
class ExposureLedger: add(cell_meta), to_json() -> dict
def contamination(world: World, forbidden: list[tuple[str, str, str]], texts: list[str]) -> list[str]
    # forbidden (name, rel, obj) triples / p3 cells that must not occur in training text; returns violations
```

## cev11/tokenizers.py — tokenizer utilities (repo ByteLevelBPE unmodified)

```python
def train_tokenizer(prose_texts: list[str], vocab_size: int = 4096) -> ByteLevelBPE
def token_spans(tok, text: str) -> list[tuple[int, int, int]]   # (id, char_start, char_end) for tok.encode(text);
    # derive offsets from pretokenize() chunks + per-chunk BPE pieces; raise if a token's bytes cross a char boundary
def root_token_range(tok, prefix: str, name: str, ctx: str = "none") -> tuple[int, int]
    # [first, last] token indices (in [BOS]+ids) that overlap the bare-name span
def classify_informal(tok, name: str, case: str) -> str          # "IDENT" | "RESEG" | "FUSED" (see PROTOCOL §3),
    # computed on the subject-initial informal frame vs the formal frame of the same case
def last_root_token(tok, name: str) -> int                      # id of the name's last root token (nom, no ctx)
def segmentation_stats(tok, names) -> dict                      # tokens/name histogram, last-root-token clusters
class FusedFreqBPE(ByteLevelBPE)                                # exploratory: letters run through apostrophe;
    # own FORMAT_NAME/identity/save/load; never used by the primary arms
def gpt2_regex_stats(names) -> dict                             # exploratory segmentation with a GPT-2-style regex
```

## cev11/model.py — model build and checkpoints

```python
def build_model(vocab_size: int, layers: int = 6, d: int = 192, heads: int = 4, kv_heads: int = 4,
                max_len: int = 64, seed: int = 0) -> tuple[torch.nn.Module, dict]
    # tiny_model_config(vocab_size=..., embed_dim=d, seq_proj_dim=d, num_heads=heads, num_kv_heads=kv_heads,
    # num_layers=layers, ffn_dim=None, max_seq_length=max_len, max_cache_len=max_len, dropout=0.0) +
    # ModelManager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False); torch.manual_seed(seed)
def model_sha256(model) -> str                                  # canonical over sorted state_dict tensors (fp32 bytes)
def save_checkpoint(path, model, cfg: dict, tok, meta: dict) -> None   # tokenizer saved next to it
def load_checkpoint(path) -> tuple[torch.nn.Module, dict, ByteLevelBPE, dict]
    # rebuild FROM THE SAVED cfg (never from defaults), load_state_dict(strict=True), verify tokenizer identity
```

## cev11/train.py — deterministic pretraining and gates

```python
def pretrain(world, tok, prose_train: list[str], out_dir, passes: int, seed: int, log=print,
             layers: int = 6, d: int = 192, prose_share: float = 0.20) -> pathlib.Path
    # AdamW lr 2e-3, betas (0.9, 0.98), wd 0.01 (no decay on 1-D params; groups by NAME), 100 warmup steps,
    # cosine to 0.1x, batch 64 sequences, grad clip 1.0, threads 1. One sequence = [BOS] text [EOS] (prose
    # chunks get EOS only when is_end),
    # right-padded with PAD, loss ignores PAD. Data order from random.Random(seed + 1). Writes
    # out_dir/model.pt (+ tokenizer), out_dir/curve.jsonl, out_dir/exposure.json. Returns the checkpoint path.
def evaluate_gates(model, tok, world, heldout_prose, n_cells: int = 600, seed: int = 0) -> dict
    # G1 round-trip, G3 seen formal no-context, G4 p3-held, G5 informal (IDENT/NONIDENT split), G6 prefixed,
    # G7 editable set (probe entities correct on rel A in all 7 formal frames and >= 3 of 6 informal inflected
    # frames, no context). Uses decode.exact_check_batch. Returns values, thresholds, pass/fail, editable list.
```

## cev11/decode.py — decoding and the batched exact check

```python
def greedy_ids(model, ids: list[int], max_new: int = 8, use_cache: bool = True, stop_ids=(EOS_ID,)) -> list[int]
def greedy_text(model, tok, prefix: str, max_new: int = 8) -> str     # decoded continuation
def exact_check_batch(model, tok, prefixes: list[str], objects: list[str], batch: int = 64) -> list[bool]
    # True iff greedy decoding from [BOS]+encode(prefix) reproduces encode(" " + obj + ".") exactly:
    # one teacher-forced forward over prefix + continuation; argmax must equal the continuation token at
    # every continuation position (the "." included). Right-pad within a batch (causal model: padding after
    # the sequence cannot affect earlier positions); compare at the true positions only.
def clear_cache(model) -> None                                      # model.clear_kv_cache() if present
```

## cev11/keysites.py — key positions and averaged keys

```python
PROSE_PREFIXES: tuple[str, ...]        # 10 fixed short Turkish prose prefixes ending with ". " (no names)
def sites(tok, prefix: str, name: str, ctx: str = "none") -> dict[str, int]
    # "ROOT": last token wholly inside the bare name; "STD": last token overlapping the bare name;
    # "LAST": last token of the whitespace word containing the name; "PRED": last prefix token
def fc2_inputs(model, layer: int, ids_batch: list[list[int]]) -> list[torch.Tensor]   # per sequence [T, f]
def key_at(model, layer, tok, prefix, pos) -> torch.Tensor          # fc2 input at pos
def std_key(model, layer, tok, prefix, name, ctx="none") -> torch.Tensor
    # mean of the STD-site key over the lived prefix and its 10 PROSE_PREFIXES copies (ROME prefix averaging)
def para_orth_key(model, layer, tok, name, rel, exclude: tuple[str, str] | None) -> torch.Tensor
    # STD key averaged additionally over the 7 formal and 6 informal inflected subject-initial frames of `rel`,
    # leaving out `exclude = (case, orth)` (the tested variant)
```

## cev11/memory.py — in-network memory and ledger

```python
class FFNMemory:                      # forward hook on model.layers[layer].ffn.fc2: out + h @ M.T
    def __init__(self, model, layer: int): ...
    def set_M(self, M: torch.Tensor) -> None     # [d, f]; clears the model KV cache
    def zero(self) -> None; def remove(self) -> None
    def substitute(self, pos_by_row: list[int] | None, value: torch.Tensor | None)   # used by v* search
def rome_update(W: torch.Tensor, C0: torch.Tensor, k: torch.Tensor, v_star: torch.Tensor) -> torch.Tensor
    # M = (v* − W k)(C0⁻¹k)ᵀ / (kᵀC0⁻¹k)   (float64 internally)
class Ledger:                         # float64; W = W0 + B (λ C0 + A)^-1, A = Σ k kᵀ, B = Σ r kᵀ
    def __init__(self, W0, C0, lam: float, identity: dict): ...
    def insert(self, fact_id: str, entity: str, k, r) -> None
    def correct(self, fact_id: str, r_new) -> None
    def retract(self, fact_id: str) -> None
    def erase(self) -> None
    def merge(self, other: "Ledger") -> None
    def delta(self) -> torch.Tensor             # M = W − W0 (float32 for the hook; float64 kept)
    def export(self, path) -> None; @classmethod def load(cls, path) -> "Ledger"   # refuses identity mismatch
def key_covariance(model, layer, ids_batches) -> torch.Tensor   # float64 mean h hᵀ (+ 1e-4·tr/f·I added by caller)
```

## cev11/vstar.py — target value search

```python
def search_delta(model, layer: int, prompts: list[tuple[list[int], int]], target_ids: list[int],
                 kl_prompt_ids: list[int] | None, step_size: float, cap: float, max_steps: int = 120,
                 stop_nll: float = 0.05, kl_weight: float = 1.0) -> tuple[torch.Tensor, dict]
    # δ added to fc2 output at each prompt's key position (substitution v0+δ); Adam from δ=0;
    # loss = mean NLL(target | prompt) + kl_weight · KL(base || edited) at the last position of kl_prompt;
    # ‖δ‖ ≤ cap (projection); early stop when mean NLL < stop_nll; deterministic; returns δ and
    # {"nll", "steps", "delta_norm", "stopped"}. Optional content-hash memo dict argument.
```

## cev11/baselines.py — comparison arms and life state variants

```python
def ft_layer(model, layer, prompts, target_ids, lr, steps, eps) -> torch.Tensor   # ΔW on fc2 (ℓ∞ clamp), restores W
def ft_all(model, prompts, target_ids, lr, steps) -> dict                          # new state_dict; model restored
def normalize_informal(text: str, lexicon: set[str]) -> str     # restore apostrophe before a valid case allomorph
class IndependentState: ...   # per-fact ROME deltas vs W0; correct = replace; retract = subtract
class SequentialState: ...    # sequential ROME on current weights; retract = write back W0·k
```

## cev11/probes.py — probe families and τ

```python
def edit_items(world, editable: list[str], seed: int, per_case: int = 10) -> list[dict]
def probe_families(world, tok, item: dict, base_answers: dict) -> dict[str, list[dict]]   # WF PF6 PF5 IF5 CF14 RN RD SS RB RC PX
def tau(k_prime, k_bar, C0_inv) -> float     # k'ᵀC0⁻¹k̄ / k̄ᵀC0⁻¹k̄
```

## analyze.py — statistics (STUDY/analyze.py; helpers may live in cev11/stats.py)

```python
def signflip_pvalue(d: list[float], margin: float) -> float    # exact one-sided over all 2^n sign patterns of (d − m)
def holm(pvals: dict[str, float], alpha=0.05) -> dict[str, bool]
def decide(d: list[float], margin: float, rejected: bool) -> str   # "supported" | "refuted" | "inconclusive";
    # refuted iff upper bound of the two-sided 95% t-interval of mean(d) < margin
def hierarchical_bootstrap(per_world_edits: dict[int, list[float]], n=10000, seed=0) -> tuple[float, float]
```
