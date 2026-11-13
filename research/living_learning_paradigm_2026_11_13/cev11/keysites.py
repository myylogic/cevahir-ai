# -*- coding: utf-8 -*-
"""Anahtar konumları ve ortalamalı anahtarlar (CEV-11 PROTOCOL §8, §10.3 V2/V3, §10.5).

Positions index ``ids = [BOS] + tok.encode(prefix)`` (INTERFACES.md). For the bare name occupying characters
``[s, e)`` of the prefix and the whitespace word ``[s, w)`` that contains it:

* ``ROOT`` — the last token lying wholly inside the name. The single space that the pre-tokenizer glues to the
  front of a word (``" Ayten"`` after a context or a prose prefix) does not disqualify a token; the token must
  contain at least one name character and must not reach past ``e``. Undefined (``SiteUndefined``) when every
  name token also covers suffix characters (possible only for fused informal forms).
* ``STD`` — the last token overlapping the name (ROME/MEMIT "last subject token" on the lemma span). In formal
  text this is the root-final token; in a FUSED informal form it is the fused root+suffix token.
* ``LAST`` — the last token of the whitespace word containing the name (the suffix token in formal text).
* ``PRED`` — the last prefix token.

Keys are fc2 inputs (``h`` of ``layers[l].ffn.fc2``). ``key_at``, ``std_key`` and ``para_orth_key`` run each
sequence alone and truncated to ``ids[:pos + 1]``: a key depends only on the tokens up to its position (causal
model), and this makes keys of prefixes that share those tokens bitwise identical (V2) and independent of batch
composition. They return float64 tensors (exact copies of float32 activations, or float64 means).
``fc2_inputs`` returns raw float32 activations of whole sequences, each run alone (no padding) by default.

Token character offsets come from ``cev11.tokenizers.token_spans`` when that module exists; otherwise an
equivalent local implementation is used, which re-derives the offsets from the tokenizer's own pre-tokenizer and
BPE pieces and raises unless its ids equal ``tok.encode(text)``.

Additions to INTERFACES.md (backward compatible): ``sites(..., strict=True)`` (``strict=False`` maps undefined
sites to None), ``SiteUndefined``, ``site_prompts`` (prompt list for the v* search), ``prose_subset`` (seeded
choice of the prose copies used by v*, PROTOCOL §12 seed w+4), ``variant_std_keys`` and ``para_orth_from_variants``
(compute the 13 variant keys once and derive every leave-one-out PARA-ORTH key from them).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tokenizer_management import byte_bpe
from tokenizer_management.byte_bpe import BOS_ID

from .memory import _fc2
from .morph_tr import CASES, INFLECTED

# Ten fixed short Turkish prose prefixes: no proper nouns, no relation words (memleket, meslek, yemek),
# each ending with ". " so that the name follows a space, as in ROME's prefix averaging.
PROSE_PREFIXES: Tuple[str, ...] = (
    "Hava bugün çok güzeldi. ",
    "Akşam olmak üzereydi. ",
    "Yol uzun ve tozluydu. ",
    "Herkes erkenden kalktı. ",
    "Kitap masanın üstündeydi. ",
    "Saat gece yarısını geçmişti. ",
    "Kapı yavaşça açıldı. ",
    "Sokakta kimse yoktu. ",
    "Toplantı geç saatte bitti. ",
    "Bahçedeki ağaçlar çiçek açmıştı. ",
)
SITE_NAMES: Tuple[str, ...] = ("ROOT", "STD", "LAST", "PRED")


class SiteUndefined(ValueError):
    """A key site does not exist for this prefix (e.g. ROOT when every name token is fused with the suffix)."""


# ---------------------------------------------------------------------------------------------------- token spans
def _local_token_spans(tok, text: str) -> List[Tuple[int, int, int]]:
    """(id, char_start, char_end) for ``tok.encode(text)`` from pre-tokenizer chunks and per-chunk BPE pieces."""
    pre = getattr(tok, "pretokenize", None)
    chunks = pre(text) if callable(pre) else byte_bpe.pretokenize(text)
    if "".join(chunks) != text:
        raise ValueError("pre-tokenizer chunks do not reassemble the text")
    spans: List[Tuple[int, int, int]] = []
    c0 = 0
    for chunk in chunks:
        byte_to_char = {0: 0}
        nb = 0
        for ci, ch in enumerate(chunk):
            nb += len(ch.encode("utf-8"))
            byte_to_char[nb] = ci + 1
        b = 0
        for tid in tok._bpe(chunk):
            width = len(tok.id_to_token[tid])          # one byte symbol per byte
            if b not in byte_to_char or b + width not in byte_to_char:
                raise ValueError(f"token {tid} crosses a character boundary in {chunk!r}")
            spans.append((int(tid), c0 + byte_to_char[b], c0 + byte_to_char[b + width]))
            b += width
        if b != nb:
            raise ValueError(f"BPE pieces do not cover chunk {chunk!r}")
        c0 += len(chunk)
    if [s[0] for s in spans] != list(tok.encode(text)):
        raise ValueError("local token spans disagree with tok.encode(text)")
    return spans


def token_spans(tok, text: str) -> List[Tuple[int, int, int]]:
    """``cev11.tokenizers.token_spans`` if that module exists, else the equivalent local implementation."""
    try:
        from .tokenizers import token_spans as sibling
    except ImportError as exc:   # only a missing sibling module/function falls back; any other error propagates
        if exc.name != f"{__package__}.tokenizers":
            raise
        sibling = _local_token_spans
    return sibling(tok, text)


# ---------------------------------------------------------------------------------------------------- sites
def _sites_from_spans(spans: Sequence[Tuple[int, int, int]], text: str, s: int, e: int, w: int,
                      strict: bool = True) -> Dict[str, Optional[int]]:
    """Sites for the name ``[s, e)`` inside the word ``[s, w)``; indices are shifted by one for BOS."""
    if text[s:e] == "" or not s < e <= w <= len(text):
        raise ValueError(f"bad spans s={s} e={e} w={w} for a text of length {len(text)}")
    s_glued = s - 1 if s > 0 and text[s - 1] == " " else s
    root = std = last = None
    for i, (_, cs, ce) in enumerate(spans):
        if cs < e and ce > s:
            std = i
            if cs >= s_glued and ce <= e:
                root = i
        if cs < w and ce > s:
            last = i
    if std is None or last is None:
        raise ValueError("no token overlaps the name")
    if spans[last][2] != w:
        raise ValueError(f"a token crosses the end of the word at char {w}")
    out: Dict[str, Optional[int]] = {"ROOT": None if root is None else root + 1, "STD": std + 1,
                                     "LAST": last + 1, "PRED": len(spans)}
    if strict and out["ROOT"] is None:
        raise SiteUndefined(f"no token lies wholly inside the name {text[s:e]!r}")
    return out


def _name_bounds(prefix: str, name: str, ctx: str) -> Tuple[int, int, int]:
    from .world import name_char_span, word_char_end   # lazy: fixed text-level functions of the world module
    s, e = name_char_span(prefix, name, ctx)
    return s, e, word_char_end(prefix, name, ctx)


def sites(tok, prefix: str, name: str, ctx: str = "none", strict: bool = True) -> Dict[str, Optional[int]]:
    """ROOT / STD / LAST / PRED positions in ``[BOS] + tok.encode(prefix)`` (see the module docstring)."""
    s, e, w = _name_bounds(prefix, name, ctx)
    return _sites_from_spans(token_spans(tok, prefix), prefix, s, e, w, strict=strict)


def _copy_sites(tok, prefix: str, name: str, ctx: str, copy: Optional[int], strict: bool = True):
    """(text, sites) for the lived prefix (copy None) or its prose-prefixed copy ``PROSE_PREFIXES[copy]``."""
    s, e, w = _name_bounds(prefix, name, ctx)
    pp = "" if copy is None else PROSE_PREFIXES[copy]
    text = pp + prefix
    o = len(pp)
    return text, _sites_from_spans(token_spans(tok, text), text, s + o, e + o, w + o, strict=strict)


def site_prompts(tok, prefix: str, name: str, ctx: str = "none", site: str = "STD",
                 copies: Sequence[int] = ()) -> List[Tuple[List[int], int]]:
    """v* prompts ``[(ids, key_pos)]``: the lived prefix first, then the chosen prose-prefixed copies."""
    if site not in SITE_NAMES:
        raise ValueError(f"unknown site {site!r}")
    out = []
    for c in [None, *copies]:
        text, st = _copy_sites(tok, prefix, name, ctx, c, strict=(site == "ROOT"))
        out.append(([BOS_ID] + list(tok.encode(text)), int(st[site])))   # type: ignore[arg-type]
    return out


def prose_subset(seed: int, k: int = 4) -> Tuple[int, ...]:
    """Sorted indices of ``k`` of the 10 prose copies, drawn with ``random.Random(seed)`` (v* uses 4)."""
    if not 0 <= k <= len(PROSE_PREFIXES):
        raise ValueError(f"k must be in 0..{len(PROSE_PREFIXES)}")
    return tuple(sorted(random.Random(seed).sample(range(len(PROSE_PREFIXES)), k)))


# ---------------------------------------------------------------------------------------------------- activations
@torch.no_grad()
def fc2_inputs(model, layer: int, ids_batch: List[List[int]], batch_size: int = 1) -> List[torch.Tensor]:
    """fc2 inputs per sequence, ``[T_i, f]`` float32. With ``batch_size=1`` (default) every sequence runs alone,
    so its activations never depend on the other sequences; larger values right-pad within chunks (faster,
    float rounding may then differ at the 1e-8 level)."""
    fc2 = _fc2(model, layer)
    captured: List[torch.Tensor] = []
    handle = fc2.register_forward_hook(lambda mod, inp, out: captured.append(inp[0].detach()))
    was_training = model.training
    model.eval()
    seqs = [list(map(int, s)) for s in ids_batch]
    if any(len(s) == 0 for s in seqs):
        raise ValueError("empty sequence")
    out: List[torch.Tensor] = []
    try:
        for i in range(0, len(seqs), max(1, int(batch_size))):
            chunk = seqs[i:i + max(1, int(batch_size))]
            T = max(len(s) for s in chunk)
            x = torch.full((len(chunk), T), byte_bpe.PAD_ID, dtype=torch.long)
            for j, s in enumerate(chunk):
                x[j, :len(s)] = torch.tensor(s, dtype=torch.long)
            captured.clear()
            model(x)
            if len(captured) != 1:
                raise RuntimeError(f"expected one fc2 call per forward, got {len(captured)}")
            for j, s in enumerate(chunk):
                out.append(captured[0][j, :len(s)].clone())
    finally:
        handle.remove()
        model.train(was_training)
    return out


def _key_from_ids(model, layer: int, ids: Sequence[int], pos: int) -> torch.Tensor:
    if not 0 <= pos < len(ids):
        raise IndexError(f"position {pos} outside a sequence of length {len(ids)}")
    return fc2_inputs(model, layer, [list(ids[:pos + 1])])[0][pos].double()


def key_at(model, layer: int, tok, prefix: str, pos: int) -> torch.Tensor:
    """fc2 input at ``pos`` of ``[BOS] + tok.encode(prefix)`` (float64 copy; computed on the truncated prefix)."""
    return _key_from_ids(model, layer, [BOS_ID] + list(tok.encode(prefix)), pos)


def std_key(model, layer: int, tok, prefix: str, name: str, ctx: str = "none") -> torch.Tensor:
    """Mean STD-site key over the lived prefix and its 10 ``PROSE_PREFIXES`` copies (ROME prefix averaging)."""
    keys = []
    for c in [None, *range(len(PROSE_PREFIXES))]:
        text, st = _copy_sites(tok, prefix, name, ctx, c, strict=False)
        keys.append(_key_from_ids(model, layer, [BOS_ID] + list(tok.encode(text)), int(st["STD"])))  # type: ignore
    return torch.stack(keys).mean(0)


VARIANTS: Tuple[Tuple[str, str], ...] = tuple(
    [(c, "formal") for c in CASES] + [(c, "informal") for c in INFLECTED])


def variant_std_keys(model, layer: int, tok, name: str, rel: str) -> Dict[Tuple[str, str], torch.Tensor]:
    """STD key (prefix-averaged) of each of the 13 subject-initial variants: 7 formal + 6 informal inflected."""
    from .world import frame_prefix   # lazy: fixed text-level function of the world module
    return {(c, o): std_key(model, layer, tok, frame_prefix(name, rel, c, o, "none"), name, "none")
            for c, o in VARIANTS}


def para_orth_from_variants(keys: Dict[Tuple[str, str], torch.Tensor],
                            exclude: Optional[Tuple[str, str]]) -> torch.Tensor:
    """Equal-weight mean of the variant keys, leaving out ``exclude = (case, orth)`` (the tested variant)."""
    if set(keys) != set(VARIANTS):
        raise ValueError("need exactly the 13 variant keys")
    if exclude is not None and tuple(exclude) not in keys:
        raise ValueError(f"exclude must be one of {VARIANTS}, got {exclude!r}")
    chosen = [keys[v] for v in VARIANTS if v != (None if exclude is None else tuple(exclude))]
    return torch.stack(chosen).mean(0)


def para_orth_key(model, layer: int, tok, name: str, rel: str, exclude: Optional[Tuple[str, str]]) -> torch.Tensor:
    """STD key additionally averaged over the 7 formal + 6 informal inflected frames of ``rel``, leaving out
    ``exclude = (case, orth)``, so the tested variant's own key never enters the write tested on it."""
    if exclude is not None and tuple(exclude) not in VARIANTS:
        raise ValueError(f"exclude must be one of {VARIANTS}, got {exclude!r}")
    return para_orth_from_variants(variant_std_keys(model, layer, tok, name, rel), exclude)


__all__ = ["PROSE_PREFIXES", "SITE_NAMES", "SiteUndefined", "token_spans", "sites", "site_prompts", "prose_subset",
           "fc2_inputs", "key_at", "std_key", "VARIANTS", "variant_std_keys", "para_orth_from_variants",
           "para_orth_key"]
