# -*- coding: utf-8 -*-
"""CEV-11 tokenizer utilities (research/living_learning_paradigm_2026_11_13/cev11/tokenizers.py).

V11: the primary tokenizer is the unmodified repo ByteLevelBPE (identity, source hash) and round-trips 100%.
V12: the IDENT / RESEG / FUSED classifier reproduces hand-labelled cases. Each class is shown to occur: IDENT and
FUSED with real ByteLevelBPE merge tables, RESEG with a table-driven re-segmenting double (the repo BPE cannot
produce RESEG in these frames; a property test checks that). Also: token offsets, root token ranges,
last-root-token clusters, the exploratory FusedFreqBPE (save/load/identity) and the GPT-2-style regex statistics.
Fast (tiny tokenizers trained here on a few hundred synthetic Turkish sentences); files only under tmp_path.
"""
import hashlib
import json
import pathlib
import random
import sys
from collections import Counter

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
STUDY = REPO / "research" / "living_learning_paradigm_2026_11_13"
for _p in (str(REPO), str(STUDY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokenizer_management import byte_bpe  # noqa: E402
from tokenizer_management.byte_bpe import ByteLevelBPE  # noqa: E402

from cev11 import tokenizers as T  # noqa: E402
from cev11.morph_tr import CASES, INFLECTED, formal, informal, tr_capitalize  # noqa: E402

# --------------------------------------------------------------------------------------------- fixtures
SUBJECTS = ["Ankara", "İzmir", "Bursa", "Ayşe", "Mehmet", "Zeynep", "Ömer", "Şule", "Çağrı", "Ünal", "Kemal",
            "Deniz"]
NOUNS = ["evinden", "okulundan", "kitabından", "şehrinden", "köyünden", "denizden", "dağdan", "yolundan",
         "bahçeden", "penceresinden", "kapısından", "annesinden", "gününden", "önünden", "kedinin", "annenin",
         "evine", "okuluna", "köyüne", "kediye", "masaya", "sokakta", "ormanda", "nehirde", "evde"]
VERBS = ["geldi", "gitti", "okudu", "yazdı", "gördü", "sevdi", "buldu", "aldı", "verdi", "bekledi", "döndü"]
ADJS = ["güzel", "büyük", "küçük", "eski", "yeni", "uzun", "kısa", "sıcak", "soğuk", "ılık"]
# Invented names (not in the corpus), as the study's entities are unseen by the tokenizer.
NAMES = ["Ayten", "Gülnur", "Şenay", "Kerem", "Selin", "Tülay", "Berkay", "Nurten", "Gülten", "Ozan", "Ebru",
         "Cemre", "Doruk", "Ilgaz", "İpek", "Öykü", "Umut", "Melis", "Arda", "Sinan"]
HARD = ["", " ", "\n", "  baştaki boşluklar", "sondaki boşluklar  ", "tab\tlı\tmetin", "Ayten'den Ayten’den",
        "Aytenden bahsedenler bilir: memleketi Kayayurt.", "İ̇ (birleşik nokta), ß, 中文, кириллица, ελληνικά",
        "snake_case ve __dunder__", "13.11.2026 saat 14:30'da %80", "Tırnak 'tek' \"çift\" ‘eğik’ “eğik”",
        "\x00 kontrol \x1f", "emoji 😀👍🏽 son", "IŞIK İzmir'de ışık izmir'de", "Çekoslovakyalılaştıramadıklarımızdan"]
VOCAB = 500   # reachable on the synthetic corpus below (it saturates at ~546)


def synthetic_corpus(n: int = 300, seed: int = 7) -> list:
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        s, c = rnd.choice(SUBJECTS), rnd.choice(INFLECTED)
        kind = rnd.randrange(3)
        if kind == 0:
            out.append(f"{formal(s, c)} {rnd.choice(ADJS)} {rnd.choice(NOUNS)} {rnd.choice(VERBS)}.")
        elif kind == 1:
            out.append(f"Bugün {rnd.choice(NOUNS)} {rnd.choice(ADJS)} bir haber {rnd.choice(VERBS)}; "
                       f"{s} {rnd.choice(VERBS)}.")
        else:
            out.append(f"{rnd.choice(ADJS).capitalize()} {rnd.choice(NOUNS)} {rnd.choice(NOUNS)} "
                       f"{rnd.choice(VERBS)}, {formal(s, c)} de {rnd.choice(VERBS)}.")
    return out


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus()


@pytest.fixture(scope="module")
def tok(corpus):
    return T.train_tokenizer(corpus, VOCAB)


# Literal copies of the fixed relation-A carriers after the subject (cev11/world.py CARRIERS, NP["A"]).
CARRIER_TAILS = {"nom": " hakkında bilinen: memleketi", "gen": " memleketi", "dat": " dair bilinen: memleketi",
                 "acc": " tanıyan herkes bilir: memleketi", "loc": " duran kimlik kaydına göre memleketi",
                 "abl": " bahsedenler bilir: memleketi", "ins": " ilgili bilgi: memleketi"}
BYTE_SYMBOL = byte_bpe.bytes_to_unicode()


def subject_frames(name: str, case: str):
    """Literal subject-initial formal/informal prefixes of relation A (no dependency on the world module)."""
    tail = CARRIER_TAILS[case]
    return formal(name, case) + tail, informal(name, case) + tail, (0, len(name))


# --------------------------------------------------------------------------------------------- V11
def test_v11_primary_tokenizer_is_the_unmodified_repo_class(tok, tmp_path):
    assert type(tok) is ByteLevelBPE and tok.vocab_size == VOCAB
    rec = T.assert_repo_tokenizer(tok)
    assert rec["identity"] == tok.identity() == ByteLevelBPE(tok.merges).identity()
    assert rec["byte_bpe_sha256"] == T.BYTE_BPE_SHA256 == T.byte_bpe_source_sha256()
    assert byte_bpe.PRETOKENIZE_PATTERN == T.REPO_PATTERN
    path = tmp_path / "tok.json"
    tok.save(path)
    again = ByteLevelBPE.load(path)
    assert again.identity() == tok.identity() and T.assert_repo_tokenizer(again)["identity"] == tok.identity()


def test_v11_identity_check_rejects_other_classes(tok):
    class Sub(ByteLevelBPE):
        pass

    for other in (Sub(tok.merges), T.FusedFreqBPE(tok.merges)):
        with pytest.raises(AssertionError):
            T.assert_repo_tokenizer(other)


def test_v11_roundtrip_is_exact_on_varied_strings(tok, corpus):
    world_like = [f"{p}{formal(n, c) if o == 'f' else informal(n, c)} ilgili bilgi: memleketi Kayayurt."
                  for n in NAMES for c in CASES for o in "fi" for p in ("", "Bilindiği gibi, ", "Dün öğrendik ki ")]
    assert T.assert_roundtrip(tok, HARD + corpus + world_like) == len(HARD) + len(corpus) + len(world_like)
    rnd = random.Random(3)
    alphabet = "abcçğıöşüİIâîû019 \t\n.,;:!?'’\"()-_/€😀ßж中̇\x00"
    texts = ["".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 30))) for _ in range(500)]
    assert T.roundtrip_failures(tok, texts) == []


def test_v11_roundtrip_helper_raises_without_printing_texts(tok):
    class Lossy(ByteLevelBPE):
        def decode(self, ids, skip_special=True, errors="strict"):
            return super().decode(ids, skip_special, errors).replace("ş", "s")

    lossy = Lossy(tok.merges)
    with pytest.raises(AssertionError) as err:
        T.assert_roundtrip(lossy, ["tamam", "gizli şifre", "yine tamam"])
    assert "[1]" in str(err.value) and "şifre" not in str(err.value)


def test_train_tokenizer_refuses_silent_shortfall_and_single_strings(corpus):
    with pytest.raises(ValueError, match="requested 4096"):
        T.train_tokenizer(corpus[:20], 4096)
    small = T.train_tokenizer(corpus[:20], 4096, require_full_vocab=False)
    assert 260 < small.vocab_size < 4096
    with pytest.raises(TypeError):
        T.train_tokenizer("tek bir metin", 300)


# --------------------------------------------------------------------------------------------- offsets
def test_token_spans_reconstruct_the_text(tok, corpus):
    for text in corpus[:50] + HARD + [subject_frames(n, c)[i] for n in NAMES for c in INFLECTED for i in (0, 1)]:
        spans = T.token_spans(tok, text, lenient=True)
        assert [t for t, _, _ in spans] == tok.encode(text)
        assert "".join(text[s:e] for _, s, e in spans) == text
        pos = 0
        for _, s, e in spans:
            assert s == pos and e >= s
            pos = e
        assert pos == len(text)
        for k, (t, s, e) in enumerate(spans):   # each edge in place, not only a contiguous tiling of the text
            if k not in spans.flagged:            # edges at character starts: the token holds exactly text[s:e]
                assert tok.decode([t]) == text[s:e]
        if not spans.flagged:   # strict mode agrees whenever no boundary falls inside a character
            assert list(T.token_spans(tok, text)) == list(spans)


def test_token_spans_strict_raises_inside_multibyte_chars_and_lenient_flags():
    raw = ByteLevelBPE([])                      # bytes only: every multi-byte character is split
    with pytest.raises(ValueError, match="multi-byte"):
        T.token_spans(raw, "ğ")
    spans = T.token_spans(raw, "ağ😀", lenient=True)
    assert list(spans) == [(raw.encode("a")[0], 0, 1)] + [(i, 1, 1 + (k == 1)) for k, i in
                                                         enumerate(raw.encode("ğ"))] + \
        [(i, 2, 2 + (k == 3)) for k, i in enumerate(raw.encode("😀"))]
    assert spans.flagged == (1, 2, 3, 4, 5, 6)
    # byte offsets are exact regardless
    assert T.token_byte_spans(raw, "ağ") == [(raw.encode("a")[0], 0, 1)] + \
        [(i, 1 + k, 2 + k) for k, i in enumerate(raw.encode("ğ"))]


def test_root_token_range_with_given_span_and_with_world(tok):
    name, ctx = "Ayten", "P1"
    prefix = "Bilindiği gibi, Ayten'den bahsedenler bilir: memleketi"
    first, last = T.root_token_range(tok, prefix, name, ctx, span=(16, 21))
    ids = [byte_bpe.BOS_ID] + tok.encode(prefix)
    spans = T.token_spans(tok, prefix)
    for pos in range(1, len(ids)):
        _, s, e = spans[pos - 1]
        assert (s < 21 and e > 16) == (first <= pos <= last)
    assert tok.decode(ids[first:last + 1]).lstrip(" ") == "Ayten"   # the root chunk may carry the leading space
    assert T.root_token_range(tok, prefix, name, ctx) == (first, last)   # via world.name_char_span
    with pytest.raises(ValueError):
        T.root_token_range(tok, prefix, name, ctx, span=(15, 20))


# --------------------------------------------------------------------------------------------- V12
# Hand-labelled cases: (merges, name, case, label). The label follows by hand from the merge ranks on the
# informal chunk (ASCII letters are their own byte symbols):
HAND = [
    # Aytenden: A y t e n d e n -> Ay -> te -> ten: Ay|ten|d|e|n, root tokens = formal Ay|ten
    ([("A", "y"), ("t", "e"), ("te", "n")], "Ayten", "abl", T.IDENT),
    # Aytenden: Ay, then (n,d) before (t,e): Ay|te|nd|e|n; 'nd' spans the root end
    ([("A", "y"), ("n", "d"), ("t", "e")], "Ayten", "abl", T.FUSED),
    # Gülnurun: (r,u) joins the root-final r with the suffix u
    ([("r", "u")], "Gülnur", "gen", T.FUSED),
    # Gülnurun: (u,r) stays inside the root: ...|n|ur|u|n
    ([("u", "r")], "Gülnur", "gen", T.IDENT),
    # Şenaya: (y,a) crosses the root end
    ([("y", "a")], "Şenay", "dat", T.FUSED),
    # Şenaya: (a,y) inside the root: ...|n|ay|a
    ([("a", "y")], "Şenay", "dat", T.IDENT),
    # Kereme: (e,m) then (em,e): K|e|r|eme -> 'eme' spans the root end
    ([("e", "m"), ("em", "e")], "Kerem", "dat", T.FUSED),
    # nominative has no informal variant: identical strings
    ([("n", "d")], "Ayten", "nom", T.IDENT),
]


class TableSegmenter(ByteLevelBPE):
    """Test double for a non-BPE segmenter (unigram/Viterbi-like) that re-segments a root when a suffix
    follows. Chunks listed in ``table`` are segmented as given, all others by the inherited BPE."""

    def __init__(self, merges, table):
        super().__init__(merges)
        self.table = table

    def _bpe(self, chunk):
        if chunk in self.table:
            return tuple(self.token_to_id[p] for p in self.table[chunk])
        return super()._bpe(chunk)


RESEG_DOUBLE = TableSegmenter([("A", "y"), ("t", "e"), ("te", "n"), ("y", "ten"), ("d", "e"), ("de", "n")],
                              {"Ayten": ["Ay", "ten"], "Aytenden": ["A", "yten", "den"]})


@pytest.mark.parametrize("merges,name,case,label", HAND)
def test_v12_classify_reproduces_hand_labels(merges, name, case, label):
    t = ByteLevelBPE(merges)
    fp, ip, span = subject_frames(name, case)
    assert T.classify_texts(t, fp, ip, span) == label                       # literal strings
    assert T.classify_informal(t, name, case) == label                      # real frames via cev11.world
    from cev11.world import frame_prefix                                     # the literal copies are faithful
    assert (frame_prefix(name, "A", case), frame_prefix(name, "A", case, "informal")) == (fp, ip)
    assert T.formal_fused(t, name, case) is False


def test_v12_reseg_is_recognised_with_a_resegmenting_double():
    assert RESEG_DOUBLE.tokenize("Ayten'den") == ["Ay", "ten", "'", "den"]
    assert RESEG_DOUBLE.tokenize("Aytenden") == ["A", "yten", "den"]
    assert T.classify_informal(RESEG_DOUBLE, "Ayten", "abl") == T.RESEG
    fp, ip, span = subject_frames("Ayten", "abl")
    assert T.classify_texts(RESEG_DOUBLE, fp, ip, span) == T.RESEG
    assert T.RESEG in T.NONIDENT and T.FUSED in T.NONIDENT and T.IDENT not in T.NONIDENT


def test_v12_all_three_classes_occur():
    seen = {T.classify_informal(ByteLevelBPE(m), n, c) for m, n, c, _ in HAND}
    seen.add(T.classify_informal(RESEG_DOUBLE, "Ayten", "abl"))
    assert seen == set(T.LABELS)


def test_v12_formal_fused_tokenizer_cannot_be_classified():
    t = T.FusedFreqBPE([("n", "'")])           # Ayten'den is one chunk; 'n' + apostrophe merge spans the root end
    assert T.formal_fused(t, "Ayten", "abl") is True
    assert T.formal_fused_cases(t, "Ayten") == tuple(INFLECTED)
    with pytest.raises(ValueError, match="formal form"):
        T.classify_informal(t, "Ayten", "abl")
    with pytest.raises(ValueError, match="differ"):
        T.classify_texts(ByteLevelBPE([]), "Ayten'den x", "Aytanden x", (0, 5))


# Merges that fire only after a space (the byte symbol of " " is "Ġ"): after a left context the root chunk is
# " Ayten..." and its Ġ-led token swallows the d of -de/-den, while sentence-initial "Ayten..." stays as bytes.
SPACE = BYTE_SYMBOL[ord(" ")]
CTX_MERGES = [(SPACE, "A"), (SPACE + "A", "y"), (SPACE + "Ay", "t"), (SPACE + "Ayt", "e"), (SPACE + "Ayte", "n"),
              (SPACE + "Ayten", "d")]


def test_classification_and_neighbours_use_the_context_free_frame():
    """PROTOCOL §3 classifies the subject-initial form without context, and INTERFACES defines the last root token
    on the NOM frame without context. A left context can change both (the root chunk gains a leading space)."""
    t = ByteLevelBPE(CTX_MERGES)
    assert t.tokenize("Aytenden") == list("Aytenden") and t.tokenize(" Aytenden") == [" Aytend", "e", "n"]
    for case in INFLECTED:
        assert T.classify_informal(t, "Ayten", case) == T.IDENT
        after_ctx = T.FUSED if case in ("loc", "abl") else T.IDENT      # Aytende / Aytenden: suffix starts with d
        for ctx in ("P1", "P2"):
            assert T.classify_informal(t, "Ayten", case, ctx=ctx) == after_ctx
    assert T.informal_labels(t, ["Ayten"]) == {("Ayten", c): T.IDENT for c in INFLECTED}
    assert T.last_root_token(t, "Ayten") == t.token_to_id["n"]           # not the space-led " Ayten" token
    st = T.segmentation_stats(t, ["Ayten"])
    assert st["tokens_per_name"] == {5: 1} and st["nonident_share"] == 0.0
    assert all(st["informal"]["per_case"][c]["counts"] == {T.IDENT: 1, T.RESEG: 0, T.FUSED: 0} for c in INFLECTED)
    prefix = "Bilindiği gibi, Ayten'den bahsedenler bilir: memleketi"   # after P1 the name is one " Ayten" token
    first, last = T.root_token_range(t, prefix, "Ayten", "P1")
    assert first == last and t.decode([([byte_bpe.BOS_ID] + t.encode(prefix))[first]]) == " Ayten"


def _random_name(rnd: random.Random) -> str:
    cons, vows = "bcçdgklmnprsştyz", "aeıioöuü"
    syl = lambda: rnd.choice(cons) + rnd.choice(vows) + rnd.choice(["", "", rnd.choice("lnrmy")])  # noqa: E731
    return tr_capitalize(syl() + syl())


def test_v12_reseg_cannot_occur_under_the_repo_bpe():
    """Property: with the repo BPE, a surviving boundary at the root end implies identical root tokens."""
    rnd = random.Random(11)
    labels = Counter()
    for _ in range(150):
        names = [_random_name(rnd) for _ in range(2)]
        syms = sorted({BYTE_SYMBOL[b] for n in names for c in INFLECTED for b in informal(n, c).encode("utf-8")})
        merges = []
        for _ in range(rnd.randint(1, 30)):
            pair = (rnd.choice(syms), rnd.choice(syms))
            if pair not in merges:
                merges.append(pair)
                syms.append(pair[0] + pair[1])
        t = ByteLevelBPE(merges)
        for n in names:
            for c in INFLECTED:
                fp, ip, span = subject_frames(n, c)
                labels[T.classify_texts(t, fp, ip, span)] += 1
                ctx = "Dün öğrendik ki "             # space-led root chunk, as after a left context
                cspan = (len(ctx), len(ctx) + len(n))
                labels[T.classify_texts(t, ctx + fp, ctx + ip, cspan)] += 1
    assert labels[T.RESEG] == 0 and labels[T.IDENT] > 0 and labels[T.FUSED] > 0


def test_trained_tokenizer_classes_and_stats_agree(tok):
    labels = T.informal_labels(tok, NAMES)
    counts = Counter(labels.values())
    assert counts[T.IDENT] > 0 and counts[T.FUSED] > 0 and counts[T.RESEG] == 0
    st = T.segmentation_stats(tok, NAMES)
    assert st["tokenizer"]["identity"] == tok.identity() and st["n_names"] == len(NAMES)
    assert sum(st["tokens_per_name"].values()) == len(NAMES)
    assert st["formal_fused_share"] == 0.0
    for case in INFLECTED:
        pc = st["informal"]["per_case"][case]
        assert pc["counts"] == {lab: sum(1 for n in NAMES if labels[(n, case)] == lab) for lab in T.LABELS}
        assert pc["n_classified"] == len(NAMES) and pc["formal_fused"] == 0
    ov = st["informal"]["overall"]
    assert ov["counts"] == {lab: counts[lab] for lab in T.LABELS}
    assert ov["nonident_share"] == st["nonident_share"] == pytest.approx(counts[T.FUSED] / len(labels))
    json.dumps(st, ensure_ascii=False)          # JSON-serialisable


# --------------------------------------------------------------------------------------------- neighbours
def test_last_root_token_and_clusters():
    t = ByteLevelBPE([("t", "e"), ("te", "n"), ("A", "y")])
    names = ["Ayten", "Gülten", "Nurten", "Ozan", "Sinan", "Berk"]
    ten, n_ = t.token_to_id["ten"], t.token_to_id["n"]
    assert [T.last_root_token(t, n) for n in names[:3]] == [ten] * 3
    assert T.last_root_token(t, "Ozan") == t.encode("Ozan")[-1] == n_
    cl = T.last_root_token_clusters(t, names)
    assert cl[ten] == ["Ayten", "Gülten", "Nurten"] and cl[n_] == ["Ozan", "Sinan"]
    st = T.segmentation_stats(t, names)["last_root_token"]
    assert st["share_in_clusters_ge2"] == pytest.approx(5 / 6) and st["largest_cluster"] == 3
    assert st["cluster_size_hist"] == {1: 1, 2: 1, 3: 1} and st["top_clusters"][0]["token"] == "ten"
    with pytest.raises(ValueError, match="duplicate"):
        T.segmentation_stats(t, ["Ayten", "Ayten"])


def test_segmentation_stats_counts_fused_formal_cells_separately():
    t = T.FusedFreqBPE([("n", "'")])            # every formal form of Ayten is fused, none of Tülay
    st = T.segmentation_stats(t, ["Ayten", "Tülay"])
    assert st["formal_fused_share"] == 0.5
    for case in INFLECTED:
        pc = st["informal"]["per_case"][case]
        assert pc["formal_fused"] == 1 and pc["n_classified"] == 1 and pc["counts"][T.IDENT] == 1
    assert st["names_with_char_split_root"] == 1    # Tülay: no merges, so 'ü' is split into two byte tokens


def test_token_byte_spans_detects_a_tokenizer_pretokenizer_mismatch():
    class Mismatch(ByteLevelBPE):               # encodes with another pattern but is not a PatternBPE
        def encode(self, text, add_bos=False, add_eos=False):
            return [i for ch in T.FusedFreqBPE.pretokenize(text) for i in self._bpe(ch)]

    with pytest.raises(ValueError, match="differ"):
        T.token_byte_spans(Mismatch([("n", "'")]), "Ayten'den")


def test_last_root_token_matches_root_encoding(tok):
    for n in NAMES:
        assert T.last_root_token(tok, n) == tok.encode(n)[-1]


# --------------------------------------------------------------------------------------------- F_freq
def test_fused_pattern_extends_only_the_letter_run():
    assert T.FUSED_PATTERN == T.REPO_PATTERN.replace(r" ?[^\W\d_]+|", r" ?[^\W\d_]+(?:['’][^\W\d_]+)*|", 1)
    pre = T.FusedFreqBPE.pretokenize
    assert pre("Ayten'den bahsedenler") == ["Ayten'den", " bahsedenler"]
    assert pre(" Türkiye’nin 2026'da 'tek'") == [" Türkiye’nin", " 202", "6", "'da", " '", "tek", "'"]
    assert byte_bpe.pretokenize("Ayten'den bahsedenler") == ["Ayten", "'den", " bahsedenler"]
    rnd = random.Random(5)
    alphabet = "abcçğıöşüİI019 \t\n.,;:!?'’\"()-_/😀ßж̇\x00"
    for _ in range(2000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 30)))
        assert "".join(pre(text)) == text


def test_merge_loop_reimplementation_matches_the_repo_trainer(corpus):
    counts = Counter()
    for text in corpus:
        counts.update(byte_bpe.pretokenize(text))
    for vocab in (300, VOCAB, 4096):
        assert T.bpe_merges_from_counts(counts, vocab) == ByteLevelBPE.train(corpus, vocab).merges


@pytest.fixture(scope="module")
def ffreq(corpus):
    frequent = [f"{formal('Ayten', c)} {tail} Kayayurt." for c in CASES for tail in
                ("hakkında bilinen: memleketi", "ilgili bilgi: memleketi")]
    return T.train_ffreq_tokenizer(corpus, frequent, VOCAB, repeat=30), frequent


def test_ffreq_training_equals_repeated_corpus_and_fuses_frequent_entities(corpus, ffreq):
    tok_f, frequent = ffreq
    assert isinstance(tok_f, ByteLevelBPE) and type(tok_f) is T.FusedFreqBPE and tok_f.vocab_size == VOCAB
    assert tok_f.merges == T.FusedFreqBPE.train(corpus + frequent * 30, VOCAB).merges
    assert len(T.formal_fused_cases(tok_f, "Ayten")) >= 4          # tokenizer-frequent: root+suffix fused
    assert T.formal_fused_cases(tok_f, "Berkay") == ()              # unseen name: boundary kept
    assert T.assert_roundtrip(tok_f, HARD + corpus) == len(HARD) + len(corpus)
    spans = T.token_spans(tok_f, "Ayten'den bahsedenler bilir: memleketi")
    assert [t for t, _, _ in spans] == tok_f.encode("Ayten'den bahsedenler bilir: memleketi")
    with pytest.raises(ValueError, match="requested 4096"):
        T.train_ffreq_tokenizer(corpus[:20], frequent, 4096)


def test_ffreq_sentences_are_formal_and_skip_p3_held_cells():
    from types import SimpleNamespace
    fake = SimpleNamespace(tok_frequent=frozenset({"Ayten", "Tülay"}), p3_held=frozenset({("Ayten", "abl")}),
                           facts={(n, r): o for n in ("Ayten", "Tülay", "Kerem")
                                  for r, o in (("A", "Kayayurt"), ("B", "terzi"), ("C", "mantı"))})
    sents = T.ffreq_sentences(fake)
    assert len(sents) == 2 * 3 * 7 - 1
    assert "Ayten'den bahsedenler bilir: memleketi Kayayurt." not in sents
    assert "Ayten'den bahsedenler bilir: mesleği terzi." in sents
    assert "Tülay'la ilgili bilgi: en sevdiği yemek mantı." in sents
    assert not any("Kerem" in s for s in sents) and all("'" in s for s in sents if "hakkında" not in s)


def test_ffreq_identity_save_and_load(ffreq, tmp_path):
    tok_f, _ = ffreq
    ident = tok_f.identity()
    assert ident == T.FusedFreqBPE(tok_f.merges).identity()
    assert ident != ByteLevelBPE(tok_f.merges).identity()            # own format name and pattern
    payload = json.dumps({"format": "cev11-fused-freq-bpe", "version": 1, "pattern": T.FUSED_PATTERN,
                          "special": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"], "merges": [list(m) for m in tok_f.merges]},
                         ensure_ascii=False, separators=(",", ":"))
    assert ident == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path = tmp_path / "ffreq.json"
    tok_f.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "cev11-fused-freq-bpe" and data["pattern"] == T.FUSED_PATTERN
    again = T.FusedFreqBPE.load(path)
    assert type(again) is T.FusedFreqBPE and again.identity() == ident
    probe = "Ayten'in memleketi; Aytenin memleketi; Berkay'dan bahsedenler."
    assert again.encode(probe) == tok_f.encode(probe)
    with pytest.raises(ValueError):
        ByteLevelBPE.load(path)                                       # the repo class refuses the F_freq file
    repo_path = tmp_path / "repo.json"
    ByteLevelBPE(tok_f.merges).save(repo_path)
    with pytest.raises(ValueError):
        T.FusedFreqBPE.load(repo_path)                                # and vice versa
    data["merges"] = data["merges"][:-1]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        T.FusedFreqBPE.load(path)                                     # tampered merges: identity mismatch
    with pytest.raises(TypeError):
        T.PatternBPE([])                                              # abstract base


# --------------------------------------------------------------------------------------------- GPT-2 regex
def test_gpt2_regex_pretokenizer():
    g = T.gpt2_pretokenize
    assert g("Ayten'den") == ["Ayten", "'d", "en"]
    assert g("Ayten'in") == ["Ayten", "'", "in"]
    assert g("Mehmet'te") == ["Mehmet", "'t", "e"]
    assert g("Aytenden bahsedenler") == ["Aytenden", " bahsedenler"]
    rnd = random.Random(9)
    alphabet = "abcçğıöşüİI019 \t\n.,;:!?'’\"()-_/😀ßж̇\x00²"
    for _ in range(2000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 30)))
        assert "".join(g(text)) == text


def test_gpt2_regex_stats(corpus):
    st = T.gpt2_regex_stats(NAMES)
    ov = st["overall"]
    assert ov["formal"]["root_boundary_gpt2"] == 1.0 and ov["formal"]["root_boundary_repo"] == 1.0
    assert ov["informal"]["root_boundary_gpt2"] == 0.0 and ov["informal"]["same_as_repo"] == 1.0
    assert ov["formal"]["same_as_repo"] == 0.0
    for case in INFLECTED:   # the English contraction rule fires exactly on the d/t-initial suffixes
        expect = 1.0 if case in ("loc", "abl") else 0.0
        assert st["per_case"][case]["formal"]["contraction_share"] == expect
    assert st["example"]["abl/formal"] == {"gpt2": ["Ayten", "'d", "en"], "repo": ["Ayten", "'den"]}
    tok_g = T.train_gpt2_regex_tokenizer(corpus, VOCAB)
    with_bpe = T.gpt2_regex_stats(NAMES, tok=tok_g)
    assert with_bpe["bpe"]["formal_fused_share"] == 0.0 and with_bpe["bpe"]["informal"]["overall"]["counts"]["RESEG"] == 0
    json.dumps(with_bpe, ensure_ascii=False)
