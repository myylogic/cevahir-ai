# -*- coding: utf-8 -*-
"""CEV-11 TR-Paradigma world (research/living_learning_paradigm_2026_11_13/cev11/world.py) and the gold table.

V9: morph_tr agrees with cev11/gold_forms.tsv (120 hand-checked rows) and with an independent, table-driven
re-implementation written in this file, on the gold rows and on every form of every generated name.
V10: contamination() reports planted violations (an edit target stated in a training text, formal and informal
p3-held frames, canonically equivalent spellings such as NFD or a typographic apostrophe) and nothing on clean pass
sentences.
Also: make_world determinism (in-process and across processes), sizes and invariants, world_problems on crafted
violations, the partition (p3-held frames drawn from all seven cases), p3-held cells never in pass_sentences, frame
strings, prose_chunks (never above max_tokens, is_end only at paragraph ends, sentence cuts before word cuts), the
exposure ledger, the world JSON round trip, load_prose on a synthetic repository (exact cleaning output, exclusions,
refusal guards), and load_prose on the repository (about 2 s; set CEV11_SKIP_PROSE=1 to skip it). Files are written
only under tmp_path; no corpus text is printed, also not in assertion messages.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import unicodedata

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
STUDY = REPO / "research" / "living_learning_paradigm_2026_11_13"
for _p in (str(REPO), str(STUDY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cev11 import morph_tr  # noqa: E402
from cev11 import world as W  # noqa: E402

GOLD_TSV = STUDY / "cev11" / "gold_forms.tsv"

# --------------------------------------------------------------------------------------------------------------------
# Independent reference morphology (V9). Written separately from morph_tr: lower-casing by str.translate over the
# 29-letter alphabet, harmony from front/rounded features instead of a vowel map.
# --------------------------------------------------------------------------------------------------------------------
_LOWER = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ", "abcçdefgğhıijklmnoöprsştuüvyz")
_VOWELS = frozenset("aeıioöuü")
_FRONT = frozenset("eiöü")
_ROUNDED = frozenset("ouöü")
_HARD = frozenset("çfhkpsşt")          # "fıstıkçı şahap": after these -DA/-DAn start with t


def low(s: str) -> str:
    return s.translate(_LOWER)


def ref_suffix(root: str, case: str) -> str:
    r = low(root)
    last = [ch for ch in r if ch in _VOWELS][-1]
    front, rounded = last in _FRONT, last in _ROUNDED
    a = "e" if front else "a"
    i = {(False, False): "ı", (False, True): "u", (True, False): "i", (True, True): "ü"}[(front, rounded)]
    after_vowel = r[-1] in _VOWELS
    d = "t" if r[-1] in _HARD else "d"
    buffer_y = "y" if after_vowel else ""
    return {"nom": "", "gen": ("n" if after_vowel else "") + i + "n", "dat": buffer_y + a, "acc": buffer_y + i,
            "loc": d + a, "abl": d + a + "n", "ins": buffer_y + "l" + a}[case]


def ref_formal(root: str, case: str) -> str:
    s = ref_suffix(root, case)
    return root + ("'" + s if s else "")


def ref_informal(root: str, case: str) -> str:
    return root + ref_suffix(root, case)


def load_gold() -> list[tuple[str, str, str, str]]:
    lines = GOLD_TSV.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == ["root", "case", "formal", "informal"]
    rows = [tuple(line.split("\t")) for line in lines[1:]]
    assert all(len(r) == 4 for r in rows)
    return rows  # type: ignore[return-value]


@pytest.fixture(scope="module")
def world0() -> W.World:
    return W.make_world(0)


# --------------------------------------------------------------------------------------------------------- V9
def test_gold_table_shape_and_coverage():
    rows = load_gold()
    assert len(rows) == 120 and len({(r[0], r[1]) for r in rows}) == 120
    roots = sorted({r[0] for r in rows})
    assert len(roots) == 20
    assert set(collections.Counter(r[0] for r in rows).values()) == {6}
    assert {r[1] for r in rows} == set(morph_tr.INFLECTED)

    def last_vowel(root: str) -> str:
        return [ch for ch in low(root) if ch in _VOWELS][-1]

    # front/back x rounded/unrounded, and every one of the eight vowels as the deciding (last) vowel
    assert {(last_vowel(r) in _FRONT, last_vowel(r) in _ROUNDED) for r in roots} == \
        {(False, False), (False, True), (True, False), (True, True)}
    assert {last_vowel(r) for r in roots} == set(_VOWELS)
    finals = collections.Counter(
        "vowel" if low(r)[-1] in _VOWELS else ("voiceless" if low(r)[-1] in _HARD else "voiced") for r in roots)
    assert min(finals["vowel"], finals["voiceless"], finals["voiced"]) >= 6
    assert {low(r)[-1] for r in roots} >= _HARD          # every voiceless consonant ends some root (d -> t)
    assert any(r.startswith("İ") for r in roots) and any(r.startswith("I") for r in roots)
    for root, case, formal_form, informal_form in rows:
        assert informal_form == formal_form.replace("'", "")
        assert formal_form.startswith(root + "'")


def test_v9_morph_tr_matches_gold_table():
    for root, case, formal_form, informal_form in load_gold():
        assert morph_tr.formal(root, case) == formal_form, (root, case)
        assert morph_tr.informal(root, case) == informal_form, (root, case)
        assert morph_tr.suffix(root, case) == informal_form[len(root):], (root, case)


def test_v9_independent_reimplementation_matches_gold_table():
    for root, case, formal_form, informal_form in load_gold():
        assert ref_formal(root, case) == formal_form, (root, case)
        assert ref_informal(root, case) == informal_form, (root, case)


def test_v9_morph_tr_matches_independent_on_every_generated_form(world0):
    for name in world0.names:
        for case in morph_tr.CASES:
            assert morph_tr.formal(name, case) == ref_formal(name, case) == W.subject_form(name, case, "formal")
            assert morph_tr.informal(name, case) == ref_informal(name, case) == \
                W.subject_form(name, case, "informal")


# --------------------------------------------------------------------------------------------- make_world
def test_make_world_is_deterministic_and_seed_dependent(world0):
    again = W.make_world(0)
    assert again == world0 and again.name_hash() == world0.name_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", world0.name_hash())
    other = W.make_world(1)
    assert other.name_hash() != world0.name_hash()
    assert len(set(other.names) & set(world0.names)) < 40


def test_make_world_is_identical_across_processes(world0):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from cev11.world import make_world; "
            "print(make_world(0).name_hash())")
    env = dict(os.environ, PYTHONHASHSEED="12345", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", code, str(STUDY)], capture_output=True, text=True, env=env,
                         check=True, timeout=60)
    assert out.stdout.strip() == world0.name_hash()


def test_world_sizes_and_invariants(world0):
    assert len(world0.names) == 400 and len(set(world0.names)) == 400
    assert list(world0.names) == sorted(world0.names)
    assert {rel: len(objs) for rel, objs in world0.objects.items()} == {"A": 48, "B": 30, "C": 30}
    assert len(world0.facts) == 1200
    assert len(world0.prefix_pairs) == 20
    assert len(world0.p3_held) == 120
    assert len(world0.tok_frequent) == 100
    assert W.world_problems(world0) == []
    small = W.make_world(7, n_names=40)
    assert W.world_problems(small) == []
    assert collections.Counter(small.partition.values()) == {"p3": 6, "c0": 6, "probe": 28}
    assert len(small.prefix_pairs) == 2
    with pytest.raises(ValueError):
        W.make_world(0, n_names=10)


def test_names_turkish_phonotactics_capitalisation_and_shared_endings(world0):
    for name in world0.names:
        n = low(name)
        assert morph_tr.tr_capitalize(n) == name and name[1:] == n[1:]
        assert " " not in name and "'" not in name
        assert 2 <= sum(ch in _VOWELS for ch in n) <= 3                       # 2-3 syllables
        assert not any(a in _VOWELS and b in _VOWELS for a, b in zip(n, n[1:]))  # no hiatus, no doubled vowel
        assert not any(a == b for a, b in zip(n, n[1:]))                      # no doubled letter
        assert n[-1] not in "bcdg"                                            # final devoicing
        assert not any(ch in "oö" for ch in n[[i for i, ch in enumerate(n) if ch in _VOWELS][1]:])
    # Turkish-aware capitals occur: dotted İ for i, dotless I for ı
    assert any(n.startswith("İ") for n in world0.names) and any(n.startswith("I") for n in world0.names)
    shared = [n for n in world0.names if any(low(n).endswith(e) for e in W.SHARED_ENDINGS)]
    assert len(shared) == 200
    assert {e for n in shared for e in W.SHARED_ENDINGS if low(n).endswith(e)} == set(W.SHARED_ENDINGS)


def test_surface_forms_unique_and_prefix_relations_only_in_designated_pairs(world0):
    owner: dict[str, str] = {}
    for name in world0.names:
        for case in morph_tr.CASES:
            form = low(ref_informal(name, case))
            assert owner.setdefault(form, name) == name, "an informal form is shared by two names"
    by_low = {low(n): n for n in world0.names}
    pairs = set(world0.prefix_pairs)
    for name in world0.names:
        n = low(name)
        for k in range(1, len(n)):
            if n[:k] in by_low:
                assert (by_low[n[:k]], name) in pairs
    for n1, n2 in world0.prefix_pairs:
        a, b = low(n1), low(n2)
        ext = b[len(a):]
        assert b.startswith(a) and sum(ch in _VOWELS for ch in a) == 2
        assert ext[0] not in _VOWELS and sum(ch in _VOWELS for ch in ext) == 1 and len(ext) <= 3
        assert world0.partition[n1] == world0.partition[n2] == "probe"


def test_objects(world0):
    towns = world0.objects["A"]
    assert len(set(towns)) == 48
    endings = collections.Counter(next(e for e in W.TOWN_ENDINGS if low(t).endswith(e)) for t in towns)
    assert endings == {e: 4 for e in W.TOWN_ENDINGS}
    for town in towns:
        t = low(town)
        assert morph_tr.tr_capitalize(t) == town and " " not in town
        assert not any(a in _VOWELS and b in _VOWELS for a, b in zip(t, t[1:]))
    assert sorted(world0.objects["B"]) == sorted(W.OCCUPATIONS) and len(set(W.OCCUPATIONS)) == 30
    assert sorted(world0.objects["C"]) == sorted(W.DISHES) and len(set(W.DISHES)) == 30
    all_objects = {low(o) for objs in world0.objects.values() for o in objs}
    assert len(all_objects) == 108 and all(" " not in o for o in all_objects)
    assert not ({low(n) for n in world0.names} & all_objects)
    assert not ({low(n) for n in world0.names} & W.FRAME_WORDS)


def test_facts_uniform_and_balanced(world0):
    for rel in W.RELATIONS:
        counts = collections.Counter(world0.facts[(n, rel)] for n in world0.names)
        assert set(counts) == set(world0.objects[rel])
        assert max(counts.values()) - min(counts.values()) <= 1


def test_partition_p3_held_and_tok_frequent(world0):
    assert collections.Counter(world0.partition.values()) == {"p3": 60, "c0": 60, "probe": 280}
    assert len(world0.members("p3")) == 60 and len(world0.members("probe")) == 280
    per_entity = collections.Counter(name for name, _ in world0.p3_held)
    assert set(per_entity) == set(world0.members("p3")) and set(per_entity.values()) == {2}
    assert {case for _, case in world0.p3_held} <= set(morph_tr.CASES)
    assert collections.Counter(world0.partition[n] for n in world0.tok_frequent) == {"p3": 15, "c0": 15,
                                                                                     "probe": 70}


def test_p3_held_frames_are_drawn_from_all_seven_cases(world0):
    # PROTOCOL §4: "2 random frames of relation A" out of the seven frames, NOM included (not only inflected ones).
    # 60 entities x 2 draws: a case that can be drawn is missing with probability < 1e-8.
    assert {case for _, case in world0.p3_held} == set(morph_tr.CASES)


def test_forbidden_town_is_avoided(world0):
    town = world0.objects["A"][0]
    w = W.make_world(0, forbidden_words={town})
    assert town not in w.objects["A"] and len(set(w.objects["A"])) == 48
    assert W.world_problems(w, forbidden_words={town}) == []
    assert f"town {town} is a forbidden (prose or frame) word" in W.world_problems(world0, forbidden_words={town})


def test_world_problems_reports_crafted_violations(world0):
    """world_problems is make_world's safety net: each crafted violation must be reported."""
    def problems(**changes):
        return W.world_problems(dataclasses.replace(world0, **changes))

    paired = {x for pair in world0.prefix_pairs for x in pair}
    victim = next(n for n in world0.members("probe") if n not in paired)
    # a town equal to an informal case form of a name
    clash = morph_tr.tr_capitalize(morph_tr.informal(low(victim), "dat"))
    towns = tuple(sorted((clash,) + world0.objects["A"][1:]))
    assert f"{victim}: surface form {low(clash)!r} equals an object" in problems(objects={**world0.objects, "A": towns})
    # a designated pair dropped from the list: its prefix relation becomes illegal
    a, b = world0.prefix_pairs[0]
    assert f"{a} is a prefix of {b} but not a designated pair" in problems(prefix_pairs=world0.prefix_pairs[1:])
    # a p3 entity with one held frame
    p3_first = world0.members("p3")[0]
    held = set(world0.p3_held)
    held.discard(next(cell for cell in sorted(held) if cell[0] == p3_first))
    assert f"p3 entity {p3_first} has 1 held frames, expected 2" in problems(p3_held=frozenset(held))
    # one fact moved from a least-used town to a most-used town: counts 8/9 become 7/10
    facts = dict(world0.facts)
    counts = collections.Counter(facts[(n, "A")] for n in world0.names)
    least = min(sorted(counts), key=counts.get)
    most = max(sorted(counts), key=counts.get)
    facts[(next(n for n in world0.names if facts[(n, "A")] == least), "A")] = most
    assert any(p.startswith("relation A: object counts unbalanced") for p in problems(facts=facts))
    # tok_frequent one short in the p3 group
    tf = set(world0.tok_frequent)
    tf.discard(sorted(tf & set(world0.members("p3")))[0])
    assert "tok_frequent has 14 p3 entities, expected 15" in problems(tok_frequent=frozenset(tf))
    # a forbidden inflected form
    form = morph_tr.informal(victim, "loc")
    assert f"{victim}: surface form {low(form)!r} is a forbidden (prose or frame) word" in \
        W.world_problems(world0, forbidden_words=[form])


def test_forbidden_words_are_respected_turkish_aware():
    base = W.make_world(0)
    victim, other = base.names[0], base.names[1]
    forbidden = {morph_tr.tr_upper(victim), morph_tr.informal(other, "abl")}   # a bare word and an inflected form
    w = W.make_world(0, forbidden_words=forbidden)
    assert victim not in w.names and other not in w.names
    assert W.world_problems(w, forbidden_words=forbidden) == []
    assert W.make_world(0, forbidden_words=forbidden) == w                     # still deterministic
    assert W.world_problems(base, forbidden_words=forbidden)                    # the old world would collide
    with pytest.raises(TypeError):
        W.make_world(0, forbidden_words="kelime")


def test_world_json_round_trip(world0, tmp_path):
    path = tmp_path / "world.json"
    path.write_text(json.dumps(world0.to_json(), ensure_ascii=False), encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert W.world_from_json(data, expected_hash=world0.name_hash()) == world0
    with pytest.raises(ValueError):
        W.world_from_json(data, expected_hash="0" * 64)


# --------------------------------------------------------------------------------------------- frames
def test_frame_strings_hand_written_expectations():
    assert W.frame_prefix("Ayten", "A", "nom") == "Ayten hakkında bilinen: memleketi"
    assert W.frame_prefix("Ayten", "A", "gen") == "Ayten'in memleketi"
    assert W.frame_prefix("Ayten", "B", "dat") == "Ayten'e dair bilinen: mesleği"
    assert W.frame_prefix("Ayten", "C", "acc", "informal") == "Ayteni tanıyan herkes bilir: en sevdiği yemek"
    assert W.frame_prefix("Zehra", "A", "loc") == "Zehra'da duran kimlik kaydına göre memleketi"
    assert W.frame_prefix("Köşk", "B", "abl", "formal", "P1") == \
        "Bilindiği gibi, Köşk'ten bahsedenler bilir: mesleği"
    assert W.frame_prefix("Aysu", "C", "ins", "informal", "P2") == \
        "Dün öğrendik ki Aysuyla ilgili bilgi: en sevdiği yemek"
    assert W.frame_prefix("İbrahim", "A", "nom", "informal") == W.frame_prefix("İbrahim", "A", "nom")
    assert W.sentence("Ayten", "A", "abl", "Kalköy") == "Ayten'den bahsedenler bilir: memleketi Kalköy."
    prefix = W.frame_prefix("Ayten", "A", "abl", "formal", "P1")
    assert W.name_char_span(prefix, "Ayten", "P1") == (16, 21)
    assert W.word_char_end(prefix, "Ayten", "P1") == 25
    assert W.word_char_end(W.frame_prefix("Ayten", "A", "gen", "informal"), "Ayten") == len("Aytenin")


# --------------------------------------------------------------------------------------------- passes
def test_pass_sentences_cover_every_cell_once_and_never_p3_held(world0):
    held = world0.p3_held
    expected = {(n, r, c) for n in world0.names for r in W.RELATIONS for c in morph_tr.CASES} - \
        {(n, "A", c) for n, c in held}
    rng = random.Random(1)
    ctx_counts, orth_counts = collections.Counter(), collections.Counter()
    passes = []
    for pass_idx in range(2):
        items = W.pass_sentences(world0, pass_idx, rng)
        passes.append(items)
        cells = [(m["name"], m["rel"], m["case"]) for _, m in items]
        assert len(cells) == len(set(cells)) == 400 * 3 * 7 - 120
        assert set(cells) == expected
        for text, m in items:
            assert set(m) == {"name", "rel", "case", "orth", "ctx", "obj"}
            assert not (m["rel"] == "A" and (m["name"], m["case"]) in held)
            assert m["obj"] == world0.facts[(m["name"], m["rel"])]
            assert text == W.sentence(m["name"], m["rel"], m["case"], m["obj"], m["orth"], m["ctx"])
            if m["case"] == "nom":
                assert m["orth"] == "formal"
            else:
                orth_counts[m["orth"]] += 1
            ctx_counts[m["ctx"]] += 1
    n_ctx, n_orth = sum(ctx_counts.values()), sum(orth_counts.values())
    assert abs(ctx_counts["none"] / n_ctx - 0.50) < 0.02
    assert abs(ctx_counts["P1"] / n_ctx - 0.25) < 0.02 and abs(ctx_counts["P2"] / n_ctx - 0.25) < 0.02
    assert abs(orth_counts["informal"] / n_orth - 0.25) < 0.02
    assert [m for _, m in passes[0]] != [m for _, m in passes[1]]            # each pass re-draws ctx / orth
    assert W.pass_sentences(world0, 0, random.Random(1)) == passes[0]          # deterministic given the stream
    # no held frame occurs in any orthography or context of any pass (string search, V10-style)
    assert W.contamination(world0, [], [t for items in passes for t, _ in items]) == []
    with pytest.raises(ValueError):
        W.pass_sentences(world0, -1, random.Random(0))


# --------------------------------------------------------------------------------------------- prose chunks
class PieceTok:
    """Fake tokenizer: one token per run of up to three letters, per digit, per punctuation mark, per space."""

    def encode(self, text: str) -> list[int]:
        return [len(p) for p in re.findall(r"[^\W\d_]{1,3}|\d|\s|[^\w\s]|_", text)]


PARAGRAPHS = [
    "Kısa bir paragraf. İkinci cümle de kısa.",
    "Bu paragrafta çok sayıda cümle var. Her biri birkaç sözcükten oluşuyor! Bölme yalnız cümle sonlarında "
    "olmalı mı? Evet, mümkünse öyle. Ama uzun bir cümle gelirse sözcük sınırında bölünür.",
    "   ",
    "Tek cümlelik ama oldukça uzun bir paragraf, virgüllerle uzayıp giden, hiçbir noktaya varmadan sürüp "
    "giden ve sınırı aşacak kadar çok sözcük içeren bir anlatım biçimiyle yazılmış metin parçası",
    "Çok uzun bir sözcük şurada: " + "a" * 200 + " ve sonra devam.",
    "Son paragraf.",
]


def _check_chunks(chunks, paragraphs, tok, max_tokens):
    non_empty = [p for p in paragraphs if p.strip()]
    assert sum(is_end for _, is_end in chunks) == len(non_empty)
    assert chunks[-1][1] is True
    for text, _ in chunks:
        assert text and text == text.strip()
        assert 1 <= len(tok.encode(text)) <= max_tokens
    groups, current = [], []
    for text, is_end in chunks:
        current.append(text)
        if is_end:
            groups.append(current)
            current = []
    assert not current and len(groups) == len(non_empty)
    return groups


def test_prose_chunks_with_fake_tokenizer():
    tok = PieceTok()
    for max_tokens in (8, 20, 60):
        chunks = W.prose_chunks(PARAGRAPHS, tok, max_tokens=max_tokens)
        groups = _check_chunks(chunks, PARAGRAPHS, tok, max_tokens)
        non_empty = [p for p in PARAGRAPHS if p.strip()]
        for paragraph, group in zip(non_empty, groups):
            if "a" * 200 in paragraph:          # the over-long word is cut at character boundaries
                assert "".join(group).replace(" ", "") == paragraph.replace(" ", "")
            else:                               # cut only at spaces: the chunks re-join to the paragraph
                assert " ".join(group) == " ".join(paragraph.split())
    # sentence packing: a short paragraph that fits stays one chunk
    assert W.prose_chunks(["Kısa bir paragraf. İkinci cümle de kısa."], tok, 60) == \
        [("Kısa bir paragraf. İkinci cümle de kısa.", True)]
    # with room for exactly one sentence, the cut falls at the sentence boundary
    one = len(tok.encode("İkinci cümle de kısa."))
    assert W.prose_chunks(["Kısa bir paragraf. İkinci cümle de kısa."], tok, one) == \
        [("Kısa bir paragraf.", False), ("İkinci cümle de kısa.", True)]
    assert W.prose_chunks(["", "  "], tok, 10) == []
    with pytest.raises(ValueError):
        W.prose_chunks(PARAGRAPHS, tok, 0)


def test_prose_chunks_cut_at_sentence_boundary_before_word_boundary():
    tok = PieceTok()
    paragraph = "Ab cd. Ef gh ij kl mn."
    assert len(tok.encode(paragraph)) == 15 and len(tok.encode("Ef gh ij kl mn.")) == 10
    # Greedy word packing would give "Ab cd. Ef gh ij" | "kl mn."; sentence units keep the second sentence whole.
    assert W.prose_chunks([paragraph], tok, 10) == [("Ab cd.", False), ("Ef gh ij kl mn.", True)]


def test_prose_chunks_with_real_byte_bpe():
    from tokenizer_management.byte_bpe import ByteLevelBPE
    tok = ByteLevelBPE.train([p for p in PARAGRAPHS if p.strip()] * 5, vocab_size=320)
    for max_tokens in (6, 12, 60):
        chunks = W.prose_chunks(PARAGRAPHS, tok, max_tokens=max_tokens)
        _check_chunks(chunks, PARAGRAPHS, tok, max_tokens)


# --------------------------------------------------------------------------------------------- ledger, V10
def test_exposure_ledger_counts_and_round_trip(world0, tmp_path):
    items = W.pass_sentences(world0, 0, random.Random(5))
    ledger = W.ExposureLedger()
    ledger.add_all(m for _, m in items)
    assert ledger.total == len(items)
    name = world0.names[3]
    assert ledger.count(name=name) == 3 * 7 - sum(1 for n, _ in world0.p3_held if n == name)
    assert ledger.count(orth="informal") == sum(m["orth"] == "informal" for _, m in items)
    assert ledger.count(rel="A", case="nom", ctx="P1") == sum(
        (m["rel"], m["case"], m["ctx"]) == ("A", "nom", "P1") for _, m in items)
    data = ledger.to_json()
    path = tmp_path / "exposure.json"
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    back = W.ExposureLedger.from_json(json.loads(path.read_text(encoding="utf-8")))
    assert back.to_json() == data and data["total"] == sum(c["n"] for c in data["cells"]) == len(items)
    bad = {"name": name, "rel": "A", "case": "nom", "orth": "informal", "ctx": "none", "obj": "X"}
    with pytest.raises(ValueError):
        ledger.add(bad)
    with pytest.raises(KeyError):
        ledger.add({"name": name, "rel": "A"})


def test_v10_contamination_detects_planted_violations(world0):
    items = W.pass_sentences(world0, 0, random.Random(3))
    texts = [t for t, _ in items] + ["Bugün hava güzel; rüzgâr serin esiyor ve deniz sakin görünüyor."]
    targets = []
    for name in world0.members("probe")[:6]:
        old = world0.facts[(name, "A")]
        targets.append((name, "A", next(t for t in world0.objects["A"] if t != old)))
    assert W.contamination(world0, targets, texts) == []

    name, _, new_town = targets[2]
    planted_fact = W.sentence(name, "A", "loc", new_town, "informal", "P2")   # informal, with a left context
    held_name, held_case = sorted(world0.p3_held)[0]
    planted_cell = W.sentence(held_name, "A", held_case, world0.facts[(held_name, "A")], "formal", "P1")
    dirty = texts + [planted_fact, planted_cell]
    found = W.contamination(world0, targets, dirty)
    assert len(found) == 2
    assert any(f"({name}, A, {new_town})" in v and v.endswith(f"#{len(texts)}") for v in found)
    assert any(f"({held_name}, A, {held_case})" in v and v.endswith(f"#{len(texts) + 1}") for v in found)
    assert not any(planted_fact in v or planted_cell in v for v in found)      # never quotes a text
    # explicit held-cell triples; the world's p3 cells can be switched off
    assert W.contamination(world0, [], dirty, include_p3_held=False) == []
    assert len(W.contamination(world0, [(held_name, "A", held_case)], dirty, include_p3_held=False)) == 1
    # a name glued inside a longer word is not a hit; an unknown entity is an error
    bare_cell = W.sentence(held_name, "A", held_case, world0.facts[(held_name, "A")], "formal", "none")
    assert len(W.contamination(world0, [], [bare_cell])) == 1
    assert W.contamination(world0, [], ["Xy" + bare_cell]) == []
    with pytest.raises(ValueError):
        W.contamination(world0, [("Yokadı", "A", new_town)], texts)


def test_v10_contamination_informal_held_cell_and_equivalent_spellings(world0):
    held_name, held_case = next(cell for cell in sorted(world0.p3_held) if cell[1] != "nom")
    obj = world0.facts[(held_name, "A")]
    # an informal held frame behind a left context: the bare name is not a word of this text
    informal_cell = W.sentence(held_name, "A", held_case, obj, "informal", "P2")
    found = W.contamination(world0, [], [informal_cell])
    assert len(found) == 1 and f"({held_name}, A, {held_case}): informal frame occurs in text #0" in found[0]
    # canonically equivalent spellings of a formal held frame: typographic apostrophe, double space, line break
    formal_cell = W.sentence(held_name, "A", held_case, obj, "formal", "none")
    for variant in (formal_cell.replace("'", "’"), formal_cell.replace(" ", "  ", 1),
                    formal_cell.replace(" ", "\n", 1)):
        assert variant != formal_cell and len(W.contamination(world0, [], [variant])) == 1
    # NFD input is still found: a relation-B held frame ("mesleği" decomposes) and a fact sentence whose town has a
    # decomposable letter (e.g. ö in -köy, ü in -bük)
    b_cell = W.sentence(held_name, "B", held_case, world0.facts[(held_name, "B")])
    nfd = unicodedata.normalize("NFD", b_cell)
    assert nfd != b_cell
    assert len(W.contamination(world0, [(held_name, "B", held_case)], [nfd], include_p3_held=False)) == 1
    name = world0.members("probe")[0]
    town = next(t for t in world0.objects["A"]
                if t != world0.facts[(name, "A")] and unicodedata.normalize("NFD", t) != t)
    nfd_fact = unicodedata.normalize("NFD", W.sentence(name, "A", "abl", town))
    assert len(W.contamination(world0, [(name, "A", town)], [nfd_fact], include_p3_held=False)) == 1


# --------------------------------------------------------------------------------------------- prose corpus
SYNTHETIC_CHAPTER_01 = "\n".join([
    "# Başlık satırı burada atlanır",
    "",
    "Birinci paragraf düz yazıdır ve beş sözcükten uzundur.",
    "İkinci satır aynı paragrafa aittir.",
    "",
    "```python",
    "kod satırı asla eğitim metnine girmemeli",
    "```",
    "",
    "~~~",
    "```",
    "tilde çiti içindeki satır da atlanır burada",
    "~~~",
    "",
    "$$",
    "x = y + z formül bloğu da atlanır tamamen",
    "$$",
    "",
    "Satır içi $$a+b$$ gösterimi atlanır ama bu cümle kalır.",
    "",
    "| tablo | satırı | atlanır |",
    "|---|---|---|",
    "",
    "<div>HTML satırı atlanır tamamen burada</div>",
    "",
    "- Birinci liste öğesi kendi paragrafıdır tamam mı",
    "- İkinci liste öğesi de ayrı bir paragraftır",
    "  ve devam satırı ona eklenir",
    "",
    "> Alıntı satırı işaretinden arındırılır ve kalır.",
    "",
    "Bir [bağlantı metni](http://example.com) ile ![resim](x.png) ve `satır içi kod` burada **kalın** yazılır.",
    "",
    "Kısa satır.",
    "",
    "---",
    "",
])
SYNTHETIC_CHAPTER_01_PARAGRAPHS = [   # expected output, written by hand from the cleaning rules
    "Birinci paragraf düz yazıdır ve beş sözcükten uzundur. İkinci satır aynı paragrafa aittir.",
    "Satır içi gösterimi atlanır ama bu cümle kalır.",
    "Birinci liste öğesi kendi paragrafıdır tamam mı",
    "İkinci liste öğesi de ayrı bir paragraftır ve devam satırı ona eklenir",
    "Alıntı satırı işaretinden arındırılır ve kalır.",
    "Bir bağlantı metni ile ve satır içi kod burada kalın yazılır.",
]


def _synthetic_repo(root: pathlib.Path) -> pathlib.Path:
    """A miniature repository with the layout load_prose expects; synthetic text only."""
    book = root / "docs" / "book" / "tr"
    book.mkdir(parents=True)
    (book / "01-giris.md").write_text(SYNTHETIC_CHAPTER_01, encoding="utf-8")
    for n in range(2, 10):
        (book / f"{n:02d}-bolum.md").write_text(f"# Başlık\n\nBölüm {n} için yeterince uzun bir düz yazı "
                                                f"paragrafı.\n", encoding="utf-8")
    for n in (10, 11, 12):
        (book / f"{n}-ayrilmis.md").write_text(f"Ayrılmış bölüm {n} metni de en az beş sözcük içerir.\n",
                                              encoding="utf-8")
    edu = root / "education"
    (edu / "ozel").mkdir(parents=True)
    (edu / "notlar.txt").write_bytes("Notlar dosyasının birinci paragrafı yeterince uzun bir metindir.\r\n\r\n"
                                     "İkinci paragraf da\r\nbir satır kırılmasıyla devam eder burada.\r\n\r\n"
                                     "Kısa.\r\n".encode("utf-8"))
    (edu / "veri.json").write_text(json.dumps({
        "soru": "Kısa soru?", "cevap": "Bu cevap metni beş sözcükten daha uzun bir yanıttır.",
        "liste": ["Liste içindeki uzun bir dize de alınır elbette.", 3, True]}, ensure_ascii=False),
        encoding="utf-8")
    # Excluded files are recognised by NAME and never opened, so placeholders suffice (python-docx not needed).
    (edu / "MEKTUPLAR-3.docx").write_bytes(b"placeholder")
    (edu / "ozel" / "biobot-3.0-sonhali-01.06.2025.docx").write_bytes(b"placeholder")
    return root


def test_load_prose_cleaning_exclusions_and_guards_on_a_synthetic_repo(tmp_path):
    repo = _synthetic_repo(tmp_path / "repo")
    book_paragraphs = SYNTHETIC_CHAPTER_01_PARAGRAPHS + [
        f"Bölüm {n} için yeterince uzun bir düz yazı paragrafı." for n in range(2, 10)]
    education_paragraphs = [
        "Notlar dosyasının birinci paragrafı yeterince uzun bir metindir.",
        "İkinci paragraf da bir satır kırılmasıyla devam eder burada.",
        "Bu cevap metni beş sözcükten daha uzun bir yanıttır.",
        "Liste içindeki uzun bir dize de alınır elbette.",
    ]
    heldout_paragraphs = [f"Ayrılmış bölüm {n} metni de en az beş sözcük içerir." for n in (10, 11, 12)]
    train, heldout = W.load_prose(repo)
    assert train == book_paragraphs + education_paragraphs
    assert heldout == heldout_paragraphs
    assert W.load_prose(repo, include_education=False) == (book_paragraphs, heldout_paragraphs)
    sources = W.prose_sources(repo)
    assert [e["path"] for e in sources["train"]][9:] == ["education/notlar.txt", "education/veri.json"]
    assert {pathlib.Path(e["path"]).name for e in sources["excluded"]} == {
        "MEKTUPLAR-3.docx", "biobot-3.0-sonhali-01.06.2025.docx"}          # "mektup" prefix is case-insensitive
    assert all(re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) for part in ("train", "heldout", "excluded")
               for e in sources[part])

    edu = repo / "education"
    (edu / "resim.png").write_bytes(b"\x89PNG")                            # unsupported type: refuse, never skip
    with pytest.raises(ValueError):
        W.load_prose(repo)
    (edu / "resim.png").unlink()
    biobot = edu / "ozel" / "biobot-3.0-sonhali-01.06.2025.docx"
    biobot.unlink()                                                        # renamed/missing excluded file: refuse
    with pytest.raises(FileNotFoundError):
        W.load_prose(repo)
    assert W.load_prose(repo, include_education=False)[0] == book_paragraphs
    biobot.write_bytes(b"placeholder")

    book = repo / "docs" / "book" / "tr"
    (book / "05-bolum.md").write_text("Açık kalan bir kod çiti burada başlıyor.\n\n```\nkod\n", encoding="utf-8")
    with pytest.raises(ValueError):                                        # unclosed fence: refuse
        W.load_prose(repo)
    (book / "05-bolum.md").write_text("Bölüm 5 için yeterince uzun bir düz yazı paragrafı.\n", encoding="utf-8")
    (book / "05-kopya.md").write_text("Aynı numaralı ikinci bir bölüm dosyası burada.\n", encoding="utf-8")
    with pytest.raises(ValueError):                                        # two files for one chapter
        W.load_prose(repo)
    (book / "05-kopya.md").unlink()
    (book / "12-ayrilmis.md").unlink()
    with pytest.raises(FileNotFoundError):                                 # a held-out chapter is missing
        W.load_prose(repo)


@pytest.fixture(scope="module")
def prose():
    if os.environ.get("CEV11_SKIP_PROSE") == "1":
        pytest.skip("CEV11_SKIP_PROSE=1")
    pytest.importorskip("docx")
    return W.load_prose(REPO)


def test_load_prose_sources_and_cleaning(prose):
    train, heldout = prose
    assert len(train) > 1000 and len(heldout) > 50
    everything = train + heldout
    # structural residue, counted (a failure shows numbers, never text)
    residue = {
        "fence": sum("```" in p or "~~~" in p for p in everything),
        "display math": sum("$$" in p for p in everything),
        "markdown link": sum(bool(re.search(r"\]\(", p)) for p in everything),
        "table row": sum(p.startswith("|") for p in everything),
        "heading": sum(p.startswith("#") for p in everything),
        "newline": sum("\n" in p for p in everything),
        "short": sum(len(p.split()) < W.MIN_PARAGRAPH_WORDS for p in everything),
    }
    assert residue == dict.fromkeys(residue, 0)
    sources = W.prose_sources(REPO)
    excluded = {pathlib.Path(e["path"]).name for e in sources["excluded"]}
    assert excluded == {"mektuplar.docx", "mektuplar-2.docx", "biobot-3.0-sonhali-01.06.2025.docx"}
    used = [pathlib.Path(e["path"]).name for e in sources["train"]]
    assert used[:9] == sorted(used[:9]) and [u[:2] for u in used[:9]] == [f"{c:02d}" for c in range(1, 10)]
    assert not any(u.lower().startswith("mektup") or u.startswith("biobot") for u in used)
    assert [pathlib.Path(e["path"]).name[:2] for e in sources["heldout"]] == ["10", "11", "12"]
    # no paragraph of an excluded file reaches the training split
    excluded_paragraphs = set()
    for e in sources["excluded"]:
        excluded_paragraphs.update(W._education_paragraphs(REPO / e["path"]))
    assert len(excluded_paragraphs) > 0
    assert sum(p in excluded_paragraphs for p in train) == 0
    book_train, book_heldout = W.load_prose(REPO, include_education=False)
    assert book_heldout == heldout and 0 < len(book_train) < len(train) and book_train == train[:len(book_train)]


def test_study_world_collides_with_no_prose_word(prose):
    train, heldout = prose
    vocab = W.prose_vocabulary(train + heldout)
    assert "memleketi" in vocab or "bir" in vocab
    world = W.make_world(0, forbidden_words=vocab)
    assert W.world_problems(world, forbidden_words=vocab) == []
    forms = {low(ref_informal(n, c)) for n in world.names for c in morph_tr.CASES}
    assert not (forms & vocab)
    assert not ({low(t) for t in world.objects["A"]} & vocab)
    # the canonical one-call entry point builds exactly this world (pipelines must use it or a saved world JSON)
    assert W.make_study_world(0, REPO) == world


STUDY_SEEDS = (0, 1) + tuple(range(1000, 1010))     # discovery, confirmatory, reserves (PROTOCOL §12)


def test_v9_on_every_form_of_every_planned_study_world(prose):
    train, heldout = prose
    vocab = W.prose_vocabulary(train + heldout)
    hashes = set()
    for seed in STUDY_SEEDS:
        world = W.make_world(seed, forbidden_words=vocab)   # self-checks world_problems(world, vocab); raises
        hashes.add(world.name_hash())
        for name in world.names:
            for case in morph_tr.CASES:
                assert morph_tr.formal(name, case) == ref_formal(name, case), (seed, name, case)
                assert morph_tr.informal(name, case) == ref_informal(name, case), (seed, name, case)
    assert len(hashes) == len(STUDY_SEEDS)
