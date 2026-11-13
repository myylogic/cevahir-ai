# -*- coding: utf-8 -*-
"""Tokenizer utilities for CEV-11 "Paradigma" (PROTOCOL §3), built around the UNMODIFIED repo ByteLevelBPE.

Türkçe özet: Birincil tokenizer deponun ``tokenizer_management/byte_bpe.ByteLevelBPE`` sınıfıdır (aa307b2,
değiştirilmeden). Bu modül onu eğitir, kayıpsızlığını ve kimliğini doğrular (V11), her tokenin karakter aralığını
çıkarır, çıplak adı kapsayan tokenleri bulur ve kesme işaretsiz (gündelik) biçimleri IDENT / RESEG / FUSED diye
sınıflar (V12). Keşif amaçlı ``FusedFreqBPE`` (F_freq, §10.4) harf dizisinin kesme işaretinden geçmesine izin veren
kendi ön bölütleme deseni, kimliği, kaydı ve yüklemesiyle ayrı bir alt sınıftır; birincil kollarda asla kullanılmaz.

Contents
- V11: ``train_tokenizer``, ``roundtrip_failures``, ``assert_roundtrip``, ``assert_repo_tokenizer``.
- Offsets: ``token_byte_spans`` and ``token_spans`` give (id, start, end) for every id of ``tok.encode(text)``,
  derived from the tokenizer's pre-token chunks and the per-chunk BPE pieces (``tok._bpe(chunk)``; each piece's
  bytes come from ``tok.id_to_token`` and the byte decoder). The result is checked against ``tok.encode(text)``.
- V12: ``root_segmentation``, ``root_token_range``, ``classify_texts``, ``classify_informal``, ``informal_labels``,
  ``formal_fused``, ``formal_fused_cases``.
- Neighbours: ``last_root_token``, ``last_root_token_clusters``, ``segmentation_stats``.
- Exploratory: ``PatternBPE`` (study-local pre-token pattern), ``FusedFreqBPE``, ``train_ffreq_tokenizer``,
  ``ffreq_sentences``, ``bpe_merges_from_counts``; ``GPT2_PATTERN``, ``gpt2_pretokenize``, ``Gpt2RegexBPE``,
  ``train_gpt2_regex_tokenizer``, ``gpt2_regex_stats``.

Classification (PROTOCOL §3). For an inflected case, the subject-initial informal frame prefix (relation A, no
context) is compared with the formal prefix of the same case; offsets are computed in UTF-8 bytes, so the result
never depends on how multi-byte letters are split:
- FUSED: one token of the informal prefix spans the root-end byte offset;
- RESEG: a token boundary lies at the root end, but the tokens up to the root end differ from the formal ones;
- IDENT: a boundary lies at the root end and the tokens up to it are identical to the formal ones, so the formal
  key position exists in the informal query (causal masking).
The formal prefix must have a boundary at the root end (always true for the repo pre-tokenizer, which starts a
new chunk at the apostrophe); otherwise the classes are undefined and ``ValueError`` is raised.

Structural fact (verified by a property test in tests/evolution/test_cev11_tokenizers.py): under the repo BPE
RESEG cannot occur in these frames. The informal word is one chunk r+s and the formal root chunk is exactly r
(possibly with the same leading space). BPE repeatedly takes the lowest-rank pair present and merges all its
occurrences left to right. While no merge has joined a symbol of r to a symbol of s, a chosen pair that occurs in
r is also the lowest-rank pair inside r, and occurrences inside r are scanned before the cross-boundary one, so
the r-part passes through exactly the states of BPE(r) alone. Merges are never undone; hence a surviving boundary
at the root end implies root tokens equal to the formal root tokens. NONIDENT therefore equals FUSED for the
primary tokenizer (and for FusedFreqBPE); RESEG needs a non-BPE segmenter (e.g. unigram/Viterbi) and is exercised
in the tests with a table-driven double. The classifier still implements all three classes as preregistered.

Signature notes (compatible extensions of INTERFACES.md; positional use is unchanged):
- ``train_tokenizer(prose_texts, vocab_size=4096, *, min_frequency=2, require_full_vocab=True)``: raises if the
  corpus cannot fill the requested vocabulary (the repo trainer would silently stop early).
- ``token_spans(tok, text, lenient=False)`` returns ``TokenSpans`` (a ``list`` of 3-tuples with ``.flagged``).
- ``root_token_range(tok, prefix, name, ctx="none", *, span=None)``: ``span`` bypasses ``world.name_char_span``.
- ``classify_informal(tok, name, case, *, rel="A", ctx="none")``.
- ``FusedFreqBPE`` derives from ``PatternBPE``, itself a ``ByteLevelBPE`` subclass.
- ``gpt2_regex_stats(names, *, tok=None)``: an optional ``Gpt2RegexBPE`` adds BPE-level statistics.
Sibling ``cev11.world`` is imported lazily (inside functions); ``cev11.morph_tr`` is fixed and imported directly.
Nothing here prints corpus text; error messages give indices and byte offsets only.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tokenizer_management import byte_bpe
    from tokenizer_management.byte_bpe import (
        BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID, ByteLevelBPE, bytes_to_unicode,
    )
except ImportError as exc:  # pragma: no cover - depends on the caller's sys.path
    raise ImportError("cev11.tokenizers needs the repository root on sys.path "
                      "(INTERFACES.md: sys.path[:0] = [str(STUDY), str(REPO)])") from exc

from .morph_tr import INFLECTED
from .morph_tr import formal as formal_form
from .morph_tr import informal as informal_form

# ----------------------------------------------------------------------------------------------- constants
IDENT, RESEG, FUSED = "IDENT", "RESEG", "FUSED"
LABELS: Tuple[str, ...] = (IDENT, RESEG, FUSED)
NONIDENT = frozenset({RESEG, FUSED})

# V11 reference values: the repo tokenizer exactly as committed in aa307b2 (byte_bpe.py unchanged since then).
REPO_COMMIT = "aa307b23bc3ea7f3772fe9fed05b88c860f20a1f"
BYTE_BPE_SHA256 = "b5510b39a01da6e4e827c1c04d5a67317334be1210ee6fef5e9424224ed8ea21"
REPO_FORMAT_NAME = "cevahir-byte-bpe"
REPO_FORMAT_VERSION = 1
REPO_PATTERN = r" ?[^\W\d_]+|['’][^\W\d_]+| ?\d{1,3}| ?_+| ?[^\s\w]+|\s+(?!\S)|\s+"
REPO_SPECIAL_TOKENS: Tuple[str, ...] = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
REPO_SPECIAL_IDS: Tuple[int, int, int, int] = (0, 1, 2, 3)   # PAD, UNK, BOS, EOS

# Exploratory F_freq pattern: the repo pattern with ONE change — a letter run may continue through an apostrophe
# (' or ’) followed by letters, so "Ayten'den" is one pre-token. Every other alternative is the repo's.
FUSED_PATTERN = r" ?[^\W\d_]+(?:['’][^\W\d_]+)*|['’][^\W\d_]+| ?\d{1,3}| ?_+| ?[^\s\w]+|\s+(?!\S)|\s+"

# Exploratory GPT-2-style pre-tokenizer (Radford et al. 2019, encoder.py), as a Python `re` approximation:
# \p{L} -> [^\W\d_], \p{N} -> \d, [^\s\p{L}\p{N}] -> ([^\s\w]|_). English contractions are kept on purpose:
# they split Turkish suffixes such as 'den -> 'd + en and 'te -> 't + e.
GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
GPT2_CONTRACTIONS = frozenset({"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"})
_GPT2_RE = re.compile(GPT2_PATTERN)

_BYTE_ENCODER: Dict[int, str] = bytes_to_unicode()
_BYTE_DECODER: Dict[str, int] = {c: b for b, c in _BYTE_ENCODER.items()}
_N_SPECIAL = len(SPECIAL_TOKENS)


# ----------------------------------------------------------------------------------------------- helpers
def _world():
    """Lazy import of the sibling world module (only its fixed text-level functions are used here)."""
    from . import world
    return world


def _as_text_list(texts, what: str) -> List[str]:
    if isinstance(texts, (str, bytes)):
        raise TypeError(f"{what} must be a list of strings, not a single {type(texts).__name__}")
    out = list(texts)
    for i, t in enumerate(out):
        if not isinstance(t, str):
            raise TypeError(f"{what}[{i}] is {type(t).__name__}, expected str")
    return out


def _unique_names(names) -> List[str]:
    out = _as_text_list(names, "names")
    if not out:
        raise ValueError("names is empty")
    dup = [n for n, c in Counter(out).items() if c > 1]
    if dup:
        raise ValueError(f"duplicate names ({len(dup)}), e.g. {dup[:3]}")
    return out


def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _share(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def _tok_info(tok) -> dict:
    return {"class": type(tok).__name__, "identity": tok.identity(), "vocab_size": tok.vocab_size}


def _display(tok, tid: int) -> str:
    """Readable token text for reports (a piece may hold part of a multi-byte character)."""
    return tok.decode([tid], errors="replace")


# ----------------------------------------------------------------------------------------------- V11
def train_tokenizer(prose_texts: Sequence[str], vocab_size: int = 4096, *, min_frequency: int = 2,
                    require_full_vocab: bool = True) -> ByteLevelBPE:
    """Train the primary tokenizer: the repo ``ByteLevelBPE.train`` unchanged, on the prose training split only.

    ``require_full_vocab`` (default) raises ``ValueError`` when the corpus cannot fill ``vocab_size`` entries;
    the repo trainer would otherwise stop early without notice (protocol: 4,096 entries).
    """
    texts = _as_text_list(prose_texts, "prose_texts")
    if not texts:
        raise ValueError("prose_texts is empty")
    tok = ByteLevelBPE.train(texts, vocab_size=vocab_size, min_frequency=min_frequency)
    if require_full_vocab and tok.vocab_size != vocab_size:
        raise ValueError(f"trained vocabulary has {tok.vocab_size} entries, requested {vocab_size}: no pair "
                         f"reached min_frequency={min_frequency} (corpus too small); pass "
                         f"require_full_vocab=False to accept a smaller vocabulary")
    return tok


def roundtrip_failures(tok, texts: Sequence[str]) -> List[int]:
    """Indices of texts with ``tok.decode(tok.encode(t)) != t`` (texts themselves are never returned)."""
    return [i for i, t in enumerate(_as_text_list(texts, "texts")) if tok.decode(tok.encode(t)) != t]


def assert_roundtrip(tok, texts: Sequence[str], what: str = "texts") -> int:
    """Raise ``AssertionError`` unless every text round-trips exactly (G1/V11). Returns the number checked."""
    texts = _as_text_list(texts, what)
    bad = roundtrip_failures(tok, texts)
    if bad:
        raise AssertionError(f"tokenizer round-trip failed on {len(bad)}/{len(texts)} {what}; first indices "
                             f"{bad[:10]} (texts not printed)")
    return len(texts)


def _repo_identity(merges: Sequence[Tuple[str, str]]) -> str:
    """Independent recomputation of ByteLevelBPE.identity() from the aa307b2 reference constants."""
    payload = json.dumps({"format": REPO_FORMAT_NAME, "version": REPO_FORMAT_VERSION, "pattern": REPO_PATTERN,
                          "special": REPO_SPECIAL_TOKENS, "merges": [tuple(m) for m in merges]},
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def byte_bpe_source_sha256() -> str:
    """SHA-256 of the imported ``tokenizer_management/byte_bpe.py`` source file."""
    return hashlib.sha256(Path(byte_bpe.__file__).read_bytes()).hexdigest()


def assert_repo_tokenizer(tok, *, check_source: bool = True) -> dict:
    """V11: ``tok`` is the unmodified repo class and its identity is what that class computes.

    Checks (``AssertionError`` on failure): exact type ``ByteLevelBPE`` (no subclass); the module constants
    equal the aa307b2 values; identity equals that of a freshly constructed ``ByteLevelBPE(tok.merges)`` and an
    independent recomputation; the vocabulary equals the fresh one; with ``check_source`` the byte_bpe.py file
    hash equals ``BYTE_BPE_SHA256`` (the file at aa307b2). Returns a small record for logs and manifests.
    """
    if type(tok) is not ByteLevelBPE:
        raise AssertionError(f"expected the repo class ByteLevelBPE, got {type(tok).__module__}."
                             f"{type(tok).__qualname__}")
    consts = {"FORMAT_NAME": (byte_bpe.FORMAT_NAME, REPO_FORMAT_NAME),
              "FORMAT_VERSION": (byte_bpe.FORMAT_VERSION, REPO_FORMAT_VERSION),
              "PRETOKENIZE_PATTERN": (byte_bpe.PRETOKENIZE_PATTERN, REPO_PATTERN),
              "SPECIAL_TOKENS": (tuple(byte_bpe.SPECIAL_TOKENS), REPO_SPECIAL_TOKENS),
              "special ids": ((PAD_ID, UNK_ID, BOS_ID, EOS_ID), REPO_SPECIAL_IDS)}
    for key, (got, want) in consts.items():
        if got != want:
            raise AssertionError(f"byte_bpe.{key} differs from aa307b2: {got!r} != {want!r}")
    fresh = ByteLevelBPE(tok.merges)
    ident = tok.identity()
    if fresh.identity() != ident:
        raise AssertionError("identity differs from a freshly constructed ByteLevelBPE with the same merges")
    if _repo_identity(tok.merges) != ident:
        raise AssertionError("identity differs from the independent aa307b2 recomputation")
    if fresh.id_to_token != tok.id_to_token:
        raise AssertionError("vocabulary differs from a freshly constructed ByteLevelBPE with the same merges")
    src = byte_bpe_source_sha256()
    if check_source and src != BYTE_BPE_SHA256:
        raise AssertionError(f"tokenizer_management/byte_bpe.py was modified: sha256 {src} != {BYTE_BPE_SHA256} "
                             f"(commit {REPO_COMMIT[:7]})")
    return {"class": "tokenizer_management.byte_bpe.ByteLevelBPE", "identity": ident, "vocab_size": tok.vocab_size,
            "n_merges": len(tok.merges), "byte_bpe_sha256": src, "repo_commit": REPO_COMMIT}


# ----------------------------------------------------------------------------------------------- offsets
def _pretokenizer_for(tok) -> Callable[[str], List[str]]:
    """The pre-tokenizer ``tok.encode`` uses: a study-local pattern class's own, else the repo function."""
    if isinstance(tok, PatternBPE):
        return tok.pretokenize
    if isinstance(tok, ByteLevelBPE):
        return byte_bpe.pretokenize
    raise TypeError(f"unsupported tokenizer type {type(tok).__name__}")


def _piece_bytes(tok, tid: int) -> bytes:
    if tid < _N_SPECIAL:
        raise ValueError(f"special token id {tid} inside a BPE chunk")
    return bytes(_BYTE_DECODER[c] for c in tok.id_to_token[tid])


def token_byte_spans(tok, text: str) -> List[Tuple[int, int, int]]:
    """(id, byte_start, byte_end) in UTF-8 bytes of ``text`` for every id of ``tok.encode(text)``.

    Built from the tokenizer's pre-token chunks and its per-chunk BPE pieces; raises ``ValueError`` if the
    pieces do not reproduce the text bytes or the ids differ from ``tok.encode(text)``.
    """
    chunks = _pretokenizer_for(tok)(text)
    if "".join(chunks) != text:
        raise ValueError("pre-tokenizer is not lossless on this text")
    spans: List[Tuple[int, int, int]] = []
    ids: List[int] = []
    pos = 0
    for chunk in chunks:
        data = chunk.encode("utf-8")
        off = 0
        for tid in tok._bpe(chunk):
            piece = _piece_bytes(tok, tid)
            if data[off:off + len(piece)] != piece:
                raise ValueError(f"BPE piece (id {tid}) does not match the chunk bytes at text byte {pos + off}")
            spans.append((tid, pos + off, pos + off + len(piece)))
            ids.append(tid)
            off += len(piece)
        if off != len(data):
            raise ValueError(f"BPE pieces cover {off} of {len(data)} bytes of a chunk at text byte {pos}")
        pos += len(data)
    if ids != list(tok.encode(text)):
        raise ValueError("chunk-wise BPE ids differ from tok.encode(text) (tokenizer/pre-tokenizer mismatch)")
    return spans


def _byte_char_maps(text: str) -> Tuple[List[int], List[bool]]:
    """owner[b] = index of the character holding byte b (len(text) at the end); is_start[b] = b starts a char."""
    widths = [1 if o < 0x80 else 2 if o < 0x800 else 3 if o < 0x10000 else 4 for o in map(ord, text)]
    n = sum(widths)
    owner = [0] * (n + 1)
    is_start = [False] * (n + 1)
    b = 0
    for ci, w in enumerate(widths):
        is_start[b] = True
        for k in range(w):
            owner[b + k] = ci
        b += w
    owner[n], is_start[n] = len(text), True
    return owner, is_start


class TokenSpans(list):
    """``list`` of (token_id, char_start, char_end). ``flagged`` holds the indices of tokens with a boundary
    inside a multi-byte character; such boundaries were attached to that character's start (lenient mode)."""

    def __init__(self, items: Iterable[Tuple[int, int, int]] = (), flagged: Iterable[int] = ()) -> None:
        super().__init__(items)
        self.flagged: Tuple[int, ...] = tuple(flagged)


def token_spans(tok, text: str, lenient: bool = False) -> TokenSpans:
    """(id, char_start, char_end) for each id of ``tok.encode(text)``; spans are contiguous and cover the text.

    A token boundary inside a multi-byte character raises ``ValueError``, unless ``lenient=True``: then the
    boundary is attached to the start of that character (a piece holding only a character's leading bytes gets
    an empty span; the piece completing it owns the character) and the token index is listed in ``.flagged``.
    """
    spans = token_byte_spans(tok, text)
    owner, is_start = _byte_char_maps(text)
    out: List[Tuple[int, int, int]] = []
    flagged: List[int] = []
    for k, (tid, b0, b1) in enumerate(spans):
        if not (is_start[b0] and is_start[b1]):
            if not lenient:
                where = b0 if not is_start[b0] else b1
                raise ValueError(f"token {k} (id {tid}) has a boundary inside a multi-byte character at text "
                                 f"byte {where}; use lenient=True to attach it to the character start")
            flagged.append(k)
        out.append((tid, owner[b0], owner[b1]))
    return TokenSpans(out, flagged)


# ----------------------------------------------------------------------------------------------- root tokens
@dataclass(frozen=True)
class RootSegmentation:
    """How the tokens of a text cover a root (bare name) span; token indices are 0-based in ``tok.encode``."""
    ids: Tuple[int, ...]             # tok.encode(text)
    root_bytes: Tuple[int, int]      # [start, end) of the root in UTF-8 bytes
    first: int                       # first token overlapping the root
    last: int                        # last token overlapping the root
    boundary_at_end: bool            # a token boundary lies exactly at the root end
    head_ids: Tuple[int, ...]        # tokens lying wholly before the root end (text start .. root end)
    root_ids: Tuple[int, ...]        # tokens overlapping the root (incl. a fused token, if any)
    char_split: bool                 # some token edge within first..last falls inside a multi-byte character

    @property
    def fused(self) -> bool:
        return not self.boundary_at_end


def root_segmentation(tok, text: str, char_span: Tuple[int, int]) -> RootSegmentation:
    """Token coverage of ``text[char_span[0]:char_span[1]]``, computed exactly in byte offsets."""
    s, e = int(char_span[0]), int(char_span[1])
    if not 0 <= s < e <= len(text):
        raise ValueError(f"invalid root span {char_span!r} for a text of {len(text)} characters")
    bs = _utf8_len(text[:s])
    be = bs + _utf8_len(text[s:e])
    spans = token_byte_spans(tok, text)
    over = [k for k, (_, b0, b1) in enumerate(spans) if b0 < be and b1 > bs]
    if not over:
        raise ValueError("no token overlaps the root span")
    first, last = over[0], over[-1]
    boundary = any(b1 == be for _, _, b1 in spans)
    _, is_start = _byte_char_maps(text)
    split = any(not (is_start[spans[k][1]] and is_start[spans[k][2]]) for k in over)
    return RootSegmentation(ids=tuple(t for t, _, _ in spans), root_bytes=(bs, be), first=first, last=last,
                            boundary_at_end=boundary,
                            head_ids=tuple(t for t, _, b1 in spans if b1 <= be),
                            root_ids=tuple(spans[k][0] for k in over), char_split=split)


def root_token_range(tok, prefix: str, name: str, ctx: str = "none", *,
                     span: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """[first, last] indices in ``[BOS] + tok.encode(prefix)`` of the tokens overlapping the bare name.

    The name's character span comes from ``cev11.world.name_char_span(prefix, name, ctx)`` unless ``span`` is
    given (it must then hold exactly ``name``).
    """
    if span is None:
        span = _world().name_char_span(prefix, name, ctx)
    s, e = span
    if prefix[s:e] != name:
        raise ValueError(f"span {span!r} does not hold the name {name!r}")
    seg = root_segmentation(tok, prefix, (s, e))
    return seg.first + 1, seg.last + 1


def _label(formal_seg: RootSegmentation, informal_seg: RootSegmentation) -> str:
    if not formal_seg.boundary_at_end:
        raise ValueError("the formal form has no token boundary at the root end, so IDENT/RESEG/FUSED (defined "
                         "relative to the formal root tokens) is undefined for this tokenizer")
    if not informal_seg.boundary_at_end:
        return FUSED
    return IDENT if informal_seg.head_ids == formal_seg.head_ids else RESEG


def classify_texts(tok, formal_text: str, informal_text: str, root_span: Tuple[int, int],
                   informal_root_span: Optional[Tuple[int, int]] = None) -> str:
    """IDENT | RESEG | FUSED for an informal text against its formal counterpart (PROTOCOL §3).

    ``root_span`` is the bare name's character span in ``formal_text``; ``informal_root_span`` defaults to it.
    Both texts must hold the same characters before and inside the root (same context, same name).
    """
    ispan = root_span if informal_root_span is None else informal_root_span
    (fs, fe), (is_, ie) = root_span, ispan
    if formal_text[:fe] != informal_text[:ie] or fe - fs != ie - is_:
        raise ValueError("formal and informal texts differ before or inside the root")
    return _label(root_segmentation(tok, formal_text, (fs, fe)), root_segmentation(tok, informal_text, (is_, ie)))


def _frame_pair(name: str, case: str, rel: str, ctx: str):
    w = _world()
    fp = w.frame_prefix(name, rel, case, "formal", ctx)
    ip = w.frame_prefix(name, rel, case, "informal", ctx)
    return fp, w.name_char_span(fp, name, ctx), ip, w.name_char_span(ip, name, ctx)


def classify_informal(tok, name: str, case: str, *, rel: str = "A", ctx: str = "none") -> str:
    """IDENT | RESEG | FUSED of (name, case) written without the apostrophe (PROTOCOL §3, V12).

    Compares the subject-initial informal frame prefix with the formal one of the same case (relation A, no
    context by default). NOM has no informal variant: its informal string equals the formal one, hence IDENT.
    """
    fp, fspan, ip, ispan = _frame_pair(name, case, rel, ctx)
    return classify_texts(tok, fp, ip, fspan, ispan)


def informal_labels(tok, names: Sequence[str], cases: Sequence[str] = INFLECTED) -> Dict[Tuple[str, str], str]:
    """{(name, case): label} for every name and case (e.g. for G5 and the IF5 tags)."""
    return {(n, c): classify_informal(tok, n, c) for n in _unique_names(names) for c in cases}


def formal_fused(tok, name: str, case: str, *, rel: str = "A", ctx: str = "none") -> bool:
    """True if a token spans the root end in the formal frame (never for the repo tokenizer; F_freq can)."""
    fp, fspan, _, _ = _frame_pair(name, case, rel, ctx)
    return root_segmentation(tok, fp, fspan).fused


def formal_fused_cases(tok, name: str, *, rel: str = "A") -> Tuple[str, ...]:
    """Inflected cases whose formal subject-initial form is fused (F_freq: 'tokenizer-frequent fused')."""
    return tuple(c for c in INFLECTED if formal_fused(tok, name, c, rel=rel))


def _nom_segmentation(tok, name: str) -> RootSegmentation:
    w = _world()
    prefix = w.frame_prefix(name, "A", "nom", "formal", "none")
    seg = root_segmentation(tok, prefix, w.name_char_span(prefix, name, "none"))
    if not seg.boundary_at_end:
        raise ValueError(f"no token boundary at the end of the name in its NOM frame ({name!r})")
    return seg


def last_root_token(tok, name: str) -> int:
    """Id of the name's last root token in its subject-initial NOM frame (no context)."""
    return _nom_segmentation(tok, name).root_ids[-1]


def last_root_token_clusters(tok, names: Sequence[str]) -> Dict[int, List[str]]:
    """{last root token id: sorted names sharing it} (RN neighbour sets are drawn from these clusters)."""
    groups: Dict[int, List[str]] = defaultdict(list)
    for n in sorted(_unique_names(names)):
        groups[last_root_token(tok, n)].append(n)
    return dict(sorted(groups.items()))


def segmentation_stats(tok, names: Sequence[str]) -> dict:
    """Descriptive tokenization statistics for a world's names (PROTOCOL §3 Phase-0 quantities).

    - tokens per name (root tokens in the NOM frame) and the number of names whose root tokens split a
      multi-byte character;
    - clusters of names sharing their last root token: size histogram, share of names in clusters of size >= 2;
    - per inflected case: IDENT / RESEG / FUSED counts and shares over informal forms (relation A, no context),
      the NONIDENT share, and formal forms that are themselves fused (those cells are left unclassified).
    """
    names = _unique_names(names)
    w = _world()
    n_tokens: Counter = Counter()
    last: Dict[str, int] = {}
    char_split = 0
    for n in names:
        seg = _nom_segmentation(tok, n)
        n_tokens[len(seg.root_ids)] += 1
        last[n] = seg.root_ids[-1]
        char_split += seg.char_split
    groups: Dict[int, List[str]] = defaultdict(list)
    for n in names:
        groups[last[n]].append(n)
    sizes = Counter(len(v) for v in groups.values())
    in_ge2 = sum(len(v) for v in groups.values() if len(v) >= 2)
    top = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]

    per_case: Dict[str, dict] = {}
    total: Counter = Counter()
    for case in INFLECTED:
        c: Counter = Counter()
        for n in names:
            fp = w.frame_prefix(n, "A", case, "formal", "none")
            ip = w.frame_prefix(n, "A", case, "informal", "none")
            f = root_segmentation(tok, fp, w.name_char_span(fp, n, "none"))
            if f.fused:
                c["FORMAL_FUSED"] += 1
                continue
            c[_label(f, root_segmentation(tok, ip, w.name_char_span(ip, n, "none")))] += 1
        per_case[case] = _label_summary(c)
        total.update(c)
    overall = _label_summary(total)
    return {
        "tokenizer": _tok_info(tok),
        # informal NONIDENT share over all names x inflected cases (the per-edit manipulation check of PROTOCOL
        # §3 uses the target forms only; this is the world-level description)
        "nonident_share": overall["nonident_share"],
        "n_names": len(names),
        "tokens_per_name": dict(sorted(n_tokens.items())),
        "tokens_per_name_mean": sum(k * v for k, v in n_tokens.items()) / len(names),
        "single_token_names": n_tokens.get(1, 0),
        "names_with_char_split_root": char_split,
        "last_root_token": {
            "n_clusters": len(groups),
            "cluster_size_hist": dict(sorted(sizes.items())),
            "largest_cluster": max(sizes),
            "names_in_clusters_ge2": in_ge2,
            "share_in_clusters_ge2": in_ge2 / len(names),
            "top_clusters": [{"token_id": t, "token": _display(tok, t), "size": len(v)} for t, v in top],
        },
        "informal": {"per_case": per_case, "overall": overall},
        "formal_fused_share": total["FORMAL_FUSED"] / (len(names) * len(INFLECTED)),
    }


def _label_summary(c: Counter) -> dict:
    n_cls = sum(c[lab] for lab in LABELS)
    return {"n_classified": n_cls, "formal_fused": c["FORMAL_FUSED"],
            "counts": {lab: c[lab] for lab in LABELS},
            "shares": {lab: _share(c[lab], n_cls) for lab in LABELS},
            "nonident_share": _share(c[RESEG] + c[FUSED], n_cls)}


# ----------------------------------------------------------------------------------------------- pattern BPEs
def bpe_merges_from_counts(counts: Dict[str, int], vocab_size: int,
                           min_frequency: int = 2) -> List[Tuple[str, str]]:
    """BPE merges from pre-token counts: a line-by-line re-implementation of the merge loop in
    ``ByteLevelBPE.train`` (aa307b2), with the same ordering and tie-breaking.

    The repo trainer calls the module-level ``pretokenize`` itself, so a subclass with another pattern cannot
    reuse it without patching module state; this function takes counts instead. A test checks that it
    reproduces ``ByteLevelBPE.train`` exactly when fed the repo pre-token counts.
    """
    base = _N_SPECIAL + 256
    if vocab_size < base:
        raise ValueError(f"vocab_size must be >= {base}")
    words: List[List[str]] = []
    freqs: List[int] = []
    for chunk, freq in sorted(counts.items()):
        if not isinstance(freq, int) or freq <= 0:
            raise ValueError(f"pre-token counts must be positive integers (got {freq!r})")
        words.append([_BYTE_ENCODER[b] for b in chunk.encode("utf-8")])
        freqs.append(freq)

    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    where: Dict[Tuple[str, str], set] = defaultdict(set)
    for wi, (sym, f) in enumerate(zip(words, freqs)):
        for pair in zip(sym, sym[1:]):
            pair_counts[pair] += f
            where[pair].add(wi)
    heap = [(-c, pair) for pair, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: List[Tuple[str, str]] = []
    target = vocab_size - base
    while len(merges) < target and heap:
        neg, pair = heapq.heappop(heap)
        current = pair_counts.get(pair, 0)
        if current <= 0 or -neg != current:
            if current > 0:  # lazy deletion: push again with the current count
                heapq.heappush(heap, (-current, pair))
            continue
        if current < min_frequency:
            break
        merges.append(pair)
        a, b = pair
        merged = a + b
        touched: Dict[Tuple[str, str], int] = defaultdict(int)
        for wi in sorted(where.pop(pair, ())):
            sym, f = words[wi], freqs[wi]
            if len(sym) < 2:
                continue
            for p in zip(sym, sym[1:]):  # remove the old pairs
                pair_counts[p] -= f
                touched[p] += 0
            out: List[str] = []
            i = 0
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            words[wi] = out
            for p in zip(out, out[1:]):  # add the new pairs
                pair_counts[p] += f
                where[p].add(wi)
                touched[p] += 0
        pair_counts.pop(pair, None)
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                pair_counts.pop(p, None)
    return merges


class PatternBPE(ByteLevelBPE):
    """Byte-level BPE with a study-local pre-token pattern (abstract; exploratory arms only).

    Subclasses set ``FORMAT_NAME``, ``FORMAT_VERSION`` and ``PRETOKENIZE_PATTERN``. Everything that depends on
    the pattern is overridden: ``pretokenize``, ``train`` (via ``bpe_merges_from_counts``), ``encode``,
    ``identity``, ``save`` and ``load``. ``_bpe``, ``decode`` and ``tokenize`` are inherited unchanged.
    """
    FORMAT_NAME: str = ""
    FORMAT_VERSION: int = 1
    PRETOKENIZE_PATTERN: str = ""
    _PRETOKENIZE: Optional[re.Pattern] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.PRETOKENIZE_PATTERN:
            cls._PRETOKENIZE = re.compile(cls.PRETOKENIZE_PATTERN)

    def __init__(self, merges: Sequence[Tuple[str, str]], cache_size: int = 100_000) -> None:
        if not type(self).FORMAT_NAME or type(self)._PRETOKENIZE is None:
            raise TypeError("PatternBPE is abstract: subclass it with FORMAT_NAME and PRETOKENIZE_PATTERN")
        super().__init__(merges, cache_size=cache_size)

    @classmethod
    def pretokenize(cls, text: str) -> List[str]:
        """Lossless split by this class's pattern."""
        if cls._PRETOKENIZE is None:
            raise TypeError("PatternBPE is abstract: no pre-token pattern")
        return cls._PRETOKENIZE.findall(text)

    @classmethod
    def chunk_counts(cls, texts: Iterable[str]) -> Counter:
        counts: Counter = Counter()
        for t in _as_text_list(texts, "texts"):
            counts.update(cls.pretokenize(t))
        return counts

    @classmethod
    def from_counts(cls, counts: Dict[str, int], vocab_size: int, min_frequency: int = 2,
                    cache_size: int = 100_000) -> "PatternBPE":
        return cls(bpe_merges_from_counts(counts, vocab_size, min_frequency), cache_size=cache_size)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, min_frequency: int = 2,
              cache_size: int = 100_000) -> "PatternBPE":
        return cls.from_counts(cls.chunk_counts(texts), vocab_size, min_frequency, cache_size)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = [BOS_ID] if add_bos else []
        for chunk in self.pretokenize(text):
            ids.extend(self._bpe(chunk))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def identity(self) -> str:
        payload = json.dumps({"format": self.FORMAT_NAME, "version": self.FORMAT_VERSION,
                              "pattern": self.PRETOKENIZE_PATTERN, "special": SPECIAL_TOKENS,
                              "merges": self.merges}, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path) -> None:
        data = {"format": self.FORMAT_NAME, "version": self.FORMAT_VERSION, "pattern": self.PRETOKENIZE_PATTERN,
                "special_tokens": list(SPECIAL_TOKENS), "vocab_size": self.vocab_size,
                "identity": self.identity(), "merges": [list(m) for m in self.merges]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, cache_size: int = 100_000) -> "PatternBPE":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != cls.FORMAT_NAME or data.get("version") != cls.FORMAT_VERSION:
            raise ValueError(f"not a {cls.FORMAT_NAME} v{cls.FORMAT_VERSION} file: {path}")
        if data.get("pattern") != cls.PRETOKENIZE_PATTERN or tuple(data.get("special_tokens", ())) != SPECIAL_TOKENS:
            raise ValueError("pre-tokenizer pattern or special tokens differ from this implementation")
        tok = cls([tuple(m) for m in data["merges"]], cache_size=cache_size)
        if data.get("identity") != tok.identity():
            raise ValueError("tokenizer identity mismatch (file was modified or identity missing)")
        return tok


class FusedFreqBPE(PatternBPE):
    """Exploratory F_freq tokenizer (PROTOCOL §3, §10.4): a letter run continues through an apostrophe, so
    frequent formal forms such as ``Ayten'den`` can become whole-word tokens that fuse root and suffix.
    Own format name and identity; never used by the primary arms."""
    FORMAT_NAME = "cev11-fused-freq-bpe"
    FORMAT_VERSION = 1
    PRETOKENIZE_PATTERN = FUSED_PATTERN


class Gpt2RegexBPE(PatternBPE):
    """Exploratory byte-level BPE with the GPT-2-style pre-tokenizer (descriptive comparison only)."""
    FORMAT_NAME = "cev11-gpt2-regex-bpe"
    FORMAT_VERSION = 1
    PRETOKENIZE_PATTERN = GPT2_PATTERN


def _check_vocab(tok, vocab_size: int, require_full_vocab: bool, min_frequency: int):
    if require_full_vocab and tok.vocab_size != vocab_size:
        raise ValueError(f"trained vocabulary has {tok.vocab_size} entries, requested {vocab_size} "
                         f"(min_frequency={min_frequency}); pass require_full_vocab=False to accept it")
    return tok


def train_ffreq_tokenizer(prose_texts: Sequence[str], frequent_sentences: Sequence[str], vocab_size: int = 4096,
                          repeat: int = 30, *, min_frequency: int = 2,
                          require_full_vocab: bool = True) -> FusedFreqBPE:
    """F_freq: ``FusedFreqBPE`` trained on prose plus the tokenizer-frequent entities' formal sentences repeated
    ``repeat`` times. Identical to ``FusedFreqBPE.train(prose + frequent * repeat)`` (counts are scaled instead
    of copying texts)."""
    prose = _as_text_list(prose_texts, "prose_texts")
    freq = _as_text_list(frequent_sentences, "frequent_sentences")
    if not isinstance(repeat, int) or repeat < 1:
        raise ValueError(f"repeat must be a positive integer, got {repeat!r}")
    counts = FusedFreqBPE.chunk_counts(prose)
    for chunk, c in FusedFreqBPE.chunk_counts(freq).items():
        counts[chunk] += repeat * c
    tok = FusedFreqBPE.from_counts(counts, vocab_size, min_frequency)
    return _check_vocab(tok, vocab_size, require_full_vocab, min_frequency)


def ffreq_sentences(world) -> List[str]:
    """Formal, context-free sentences of ``world.tok_frequent`` (all relations and cases), skipping P3-held
    cells so that the tokenizer corpus never contains a cell the model must not see."""
    w = _world()
    from .morph_tr import CASES
    out = []
    for name in sorted(world.tok_frequent):
        for rel in w.RELATIONS:
            for case in CASES:
                if rel == "A" and (name, case) in world.p3_held:
                    continue
                out.append(w.sentence(name, rel, case, world.facts[(name, rel)], "formal", "none"))
    return out


def train_gpt2_regex_tokenizer(prose_texts: Sequence[str], vocab_size: int = 4096, *, min_frequency: int = 2,
                               require_full_vocab: bool = True) -> Gpt2RegexBPE:
    """Exploratory: byte-level BPE with the GPT-2-style pre-tokenizer on the same prose."""
    tok = Gpt2RegexBPE.train(_as_text_list(prose_texts, "prose_texts"), vocab_size, min_frequency)
    return _check_vocab(tok, vocab_size, require_full_vocab, min_frequency)


# ----------------------------------------------------------------------------------------------- GPT-2 regex
def gpt2_pretokenize(text: str) -> List[str]:
    """Lossless split by the GPT-2-style pattern."""
    return _GPT2_RE.findall(text)


def _chunk_ends(chunks: Sequence[str]) -> set:
    ends, pos = set(), 0
    for ch in chunks:
        pos += len(ch)
        ends.add(pos)
    return ends


def gpt2_regex_stats(names: Sequence[str], *, tok=None) -> dict:
    """Exploratory, descriptive: pre-token segmentation of each name's inflected subject forms (formal and
    informal) under the GPT-2-style regex, next to the repo pre-tokenizer.

    Per case and orthography: share of forms with a pre-token boundary at the root end, share identical to the
    repo pre-tokens, pre-token count histograms; for formal forms also the chunks after the root (e.g.
    "'d|en": the English contraction rule splits the suffix) and the share where a contraction fired. With
    ``tok`` (a ``Gpt2RegexBPE`` trained by the caller) the BPE-level ``segmentation_stats`` are added.
    """
    names = _unique_names(names)
    per_case: Dict[str, dict] = {}
    tot: Dict[str, Counter] = {"formal": Counter(), "informal": Counter()}
    for case in INFLECTED:
        rec = {}
        for orth, make in (("formal", formal_form), ("informal", informal_form)):
            c: Counter = Counter()
            n_g: Counter = Counter()
            n_r: Counter = Counter()
            after_root: Counter = Counter()
            for name in names:
                form = make(name, case)
                g, r = gpt2_pretokenize(form), byte_bpe.pretokenize(form)
                if "".join(g) != form or "".join(r) != form:
                    raise ValueError("a pre-tokenizer is not lossless on a generated form")
                eg, er = _chunk_ends(g), _chunk_ends(r)
                c["root_boundary_gpt2"] += len(name) in eg
                c["root_boundary_repo"] += len(name) in er
                c["same_as_repo"] += g == r
                n_g[len(g)] += 1
                n_r[len(r)] += 1
                if orth == "formal":
                    c["contraction"] += any(ch in GPT2_CONTRACTIONS for ch in g)
                    c["apostrophe_alone"] += "'" in g
                    if len(name) in eg:
                        k = sorted(eg).index(len(name)) + 1
                        after_root["|".join(g[k:])] += 1
            n = len(names)
            rec[orth] = {"root_boundary_gpt2": c["root_boundary_gpt2"] / n,
                         "root_boundary_repo": c["root_boundary_repo"] / n,
                         "same_as_repo": c["same_as_repo"] / n,
                         "n_pretokens_gpt2": dict(sorted(n_g.items())),
                         "n_pretokens_repo": dict(sorted(n_r.items()))}
            if orth == "formal":
                rec[orth].update({"contraction_share": c["contraction"] / n,
                                  "apostrophe_alone_share": c["apostrophe_alone"] / n,
                                  "chunks_after_root": dict(after_root.most_common())})
            tot[orth].update(c)
        per_case[case] = rec
    n_all = len(names) * len(INFLECTED)
    overall = {orth: {"root_boundary_gpt2": tot[orth]["root_boundary_gpt2"] / n_all,
                      "root_boundary_repo": tot[orth]["root_boundary_repo"] / n_all,
                      "same_as_repo": tot[orth]["same_as_repo"] / n_all} for orth in tot}
    overall["formal"]["contraction_share"] = tot["formal"]["contraction"] / n_all
    ex = names[0]
    out = {"pattern": GPT2_PATTERN, "repo_pattern": byte_bpe.PRETOKENIZE_PATTERN, "n_names": len(names),
           "per_case": per_case, "overall": overall,
           "example": {f"{case}/{orth}": {"gpt2": gpt2_pretokenize(make(ex, case)),
                                          "repo": byte_bpe.pretokenize(make(ex, case))}
                       for case in INFLECTED for orth, make in (("formal", formal_form),
                                                                ("informal", informal_form))}}
    if tok is not None:
        if not isinstance(tok, Gpt2RegexBPE):
            raise TypeError("tok must be a Gpt2RegexBPE (train_gpt2_regex_tokenizer)")
        out["bpe"] = segmentation_stats(tok, names)
    return out


__all__ = [
    "IDENT", "RESEG", "FUSED", "LABELS", "NONIDENT", "REPO_COMMIT", "BYTE_BPE_SHA256", "REPO_PATTERN",
    "FUSED_PATTERN", "GPT2_PATTERN", "train_tokenizer", "roundtrip_failures", "assert_roundtrip",
    "byte_bpe_source_sha256", "assert_repo_tokenizer", "token_byte_spans", "TokenSpans", "token_spans",
    "RootSegmentation", "root_segmentation", "root_token_range", "classify_texts", "classify_informal",
    "informal_labels", "formal_fused", "formal_fused_cases", "last_root_token", "last_root_token_clusters",
    "segmentation_stats", "bpe_merges_from_counts", "PatternBPE", "FusedFreqBPE", "Gpt2RegexBPE",
    "train_ffreq_tokenizer", "ffreq_sentences", "train_gpt2_regex_tokenizer", "gpt2_pretokenize",
    "gpt2_regex_stats",
]
