# -*- coding: utf-8 -*-
"""TR-Paradigma: sentetik, lisans açısından temiz Türkçe hâl paradigması dünyası (CEV-11).

Kişiler uydurma tek sözcüklü adlardır; olgular rastgele atanır, gerçek kişi ve yerlerle ilgisi yoktur. Her olgu,
öznenin yedi hâl biçiminden birinde cümle başında geçtiği kalıplarda yazılır. İlişki ad öbeği (NP) nesneden önce
gelir ve nesne cümlenin son sözcüğüdür; böylece nesneyi tahmin eden model ilişki ipucunu görmüş olur.

Bu dosyanın metin düzeyindeki işlevleri (kalıplar, önekler, ad aralıkları) sabittir ve diğer modüller onlara
dayanır. Dünya üretimi (``make_world``), geçiş derlemi, düzyazı yükleme, maruz kalma defteri ve sızıntı denetimi
aynı dosyada tamamlanır (bkz. ../INTERFACES.md).

Contract notes (additions to INTERFACES.md; every documented signature is kept and still callable as written)
-----------------------------------------------------------------------------------------------------------------
* ``make_world(seed, n_names=400, *, forbidden_words=None)`` — the optional keyword ``forbidden_words`` is the set
  of word forms that names (and each of their six informal case forms) and towns must avoid. For study runs it is
  obtained as::

      train, heldout = load_prose(REPO)
      forbidden = prose_vocabulary(train + heldout)     # every maximal letter run, Turkish-lower-cased
      world = make_world(seed, forbidden_words=forbidden)  # == make_study_world(seed, REPO)

  Frame words (carriers, relation NPs, left contexts) and all object strings are always forbidden, also when
  ``forbidden_words`` is None. Different forbidden sets can give different worlds for the same seed; the content
  hash ``World.name_hash()`` tells them apart, so a pipeline should store the world JSON (``World.to_json``) and
  reload it with ``world_from_json`` instead of regenerating it with a possibly different forbidden set.
* ``load_prose(repo_root, *, include_education=True)`` — ``include_education=False`` is the protocol's fallback
  "book prose only" (owner veto of education/). The exact source list is in ``prose_sources``.
* Extra public helpers: ``make_study_world``, ``World.to_json``, ``World.members``, ``world_from_json``,
  ``world_problems``, ``prose_vocabulary``, ``prose_sources``, ``trained_cells``, ``ends_with_shared_ending``,
  ``ExposureLedger.count/total/add_all/from_json``.
* ``contamination(world, forbidden, texts, *, include_p3_held=True)`` — a forbidden item ``(name, rel, x)`` is an
  object triple unless ``x`` is a case name, in which case it is a held frame cell; the world's p3-held cells are
  always checked unless ``include_p3_held=False``. Messages name the item and the text index, never the text.

Generator decisions the protocol leaves open (frozen with GENERATOR_VERSION; review before the freeze)
-----------------------------------------------------------------------------------------------------------------
* Names: syllables V/VC/CV/CVC, 2 syllables or (p = .30) 3; o/ö only in the first syllable; no ğ, no j; no
  word-final b/c/d/g; no doubled letter (hence no doubled vowel) and no vowel hiatus; 4-9 letters. Exactly n // 2
  names end in a shared ending (string suffix), the others in none. The 7 surface forms of a name (bare + 6
  informal) are unique across names and outside the forbidden words; no name is a prefix of another except in the
  designated pairs. Pairs: n // 20; n1 has 2 syllables, n2 = n1 + one consonant-initial syllable; both in the
  probe pool (PX is a locality probe).
* Objects: 4 towns per ending, distinct, not names; B/C fixed real-word lists; all object tuples sorted.
* Facts: uniform AND balanced per relation (every object used floor(n/k) or ceil(n/k) times).
* Partition: p3 = c0 = round(15 % of n) (60 / 60 / 280 at 400); p3_held = 2 of the 7 cases (NOM included) per p3
  entity; tok_frequent = 25 % of every partition group (15 / 15 / 70).
* Determinism: one stream per component, random.Random(f"{GENERATOR_VERSION}|seed={seed}|{component}").

No function here prints anything; corpus text never appears in return values other than the paragraphs and chunks
themselves, nor in error messages.
"""
from __future__ import annotations

from .morph_tr import CASES, INFLECTED, formal, informal

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .morph_tr import tr_capitalize

RELATIONS: tuple[str, ...] = ("A", "B", "C")
NP: dict[str, str] = {"A": "memleketi", "B": "mesleği", "C": "en sevdiği yemek"}
CONTEXTS: dict[str, str] = {"none": "", "P1": "Bilindiği gibi, ", "P2": "Dün öğrendik ki "}

# {S} = çekimli özne (biçimsel ya da gündelik yazım), {NP} = ilişki ad öbeği; nesne bir boşluktan sonra gelir.
CARRIERS: dict[str, str] = {
    "nom": "{S} hakkında bilinen: {NP}",
    "gen": "{S} {NP}",
    "dat": "{S} dair bilinen: {NP}",
    "acc": "{S} tanıyan herkes bilir: {NP}",
    "loc": "{S} duran kimlik kaydına göre {NP}",
    "abl": "{S} bahsedenler bilir: {NP}",
    "ins": "{S} ilgili bilgi: {NP}",
}
ORTHOGRAPHIES: tuple[str, ...] = ("formal", "informal")


def subject_form(name: str, case: str, orth: str = "formal") -> str:
    if orth == "formal":
        return formal(name, case)
    if orth == "informal":
        return informal(name, case)
    raise ValueError(f"unknown orthography {orth!r}")


def frame_prefix(name: str, rel: str, case: str, orth: str = "formal", ctx: str = "none") -> str:
    """Nesneden önceki metin: ``Bilindiği gibi, Ayten'den bahsedenler bilir: memleketi``."""
    if rel not in NP:
        raise ValueError(f"unknown relation {rel!r}")
    if case not in CARRIERS:
        raise ValueError(f"unknown case {case!r}")
    return CONTEXTS[ctx] + CARRIERS[case].format(S=subject_form(name, case, orth), NP=NP[rel])


def sentence(name: str, rel: str, case: str, obj: str, orth: str = "formal", ctx: str = "none") -> str:
    """Tam eğitim cümlesi: önek + " " + nesne + "."."""
    return f"{frame_prefix(name, rel, case, orth, ctx)} {obj}."


def name_char_span(prefix: str, name: str, ctx: str = "none") -> tuple[int, int]:
    """Önekte çıplak adın (kökün) [başlangıç, bitiş) karakter aralığı."""
    start = len(CONTEXTS[ctx])
    if prefix[start:start + len(name)] != name:
        raise ValueError(f"name {name!r} not found at offset {start} of {prefix!r}")
    return start, start + len(name)


def word_char_end(prefix: str, name: str, ctx: str = "none") -> int:
    """Adı içeren boşlukla ayrılmış sözcüğün bitişi (biçimsel yazımda ekin sonu)."""
    start, _ = name_char_span(prefix, name, ctx)
    end = prefix.find(" ", start)
    return len(prefix) if end < 0 else end


# =====================================================================================================================
# Shared text helpers
# =====================================================================================================================
_TR_LOWER_TABLE = {ord("İ"): "i", ord("I"): "ı"}
_LETTER_RUN = re.compile(r"[^\W\d_]+")        # same letter class as the repo pre-tokenizer


def _lower(s: str) -> str:
    """Turkish-aware lower-casing at C speed (İ→i, I→ı, then str.lower); equals morph_tr.tr_lower on Turkish."""
    return s.translate(_TR_LOWER_TABLE).lower()


def _words_of(text: str) -> list[str]:
    return _LETTER_RUN.findall(text)


# Every word of the carriers, relation NPs and left contexts; names may never equal one of them.
FRAME_WORDS: frozenset[str] = frozenset(
    _lower(w)
    for text in (*CARRIERS.values(), *NP.values(), *CONTEXTS.values())
    for w in _words_of(text.replace("{S}", " ").replace("{NP}", " ")))

# =====================================================================================================================
# World generator constants (part of the frozen generator configuration; change => bump GENERATOR_VERSION)
# =====================================================================================================================
GENERATOR_VERSION = "cev11-world-1"

# Shared final components of names (PROTOCOL §4). Exactly n_names // 2 names end in one of them (string suffix);
# the other half end in none of them.
SHARED_ENDINGS: tuple[str, ...] = ("ten", "gül", "nur", "han", "can", "tay", "su", "er", "ay", "un", "ün", "ra",
                                   "na", "li", "ce", "se")
TOWN_ENDINGS: tuple[str, ...] = ("köy", "pınar", "dere", "kent", "ova", "yurt", "tepe", "su", "hisar", "bük",
                                 "alan", "yazı")
TOWNS_PER_ENDING = 4                       # 12 endings x 4 = 48 towns (relation A)
OCCUPATIONS: tuple[str, ...] = (           # relation B: 30 real one-word occupations
    "öğretmen", "doktor", "mühendis", "aşçı", "çiftçi", "terzi", "avukat", "mimar", "pilot", "berber",
    "fırıncı", "eczacı", "hemşire", "marangoz", "kuyumcu", "balıkçı", "şoför", "garson", "ressam", "yazar",
    "gazeteci", "polis", "kasap", "bahçıvan", "tamirci", "çoban", "postacı", "veteriner", "müzisyen", "dişçi")
DISHES: tuple[str, ...] = (                # relation C: 30 real one-word Turkish dishes (no drinks, no desserts)
    "mantı", "kebap", "pilav", "dolma", "lahmacun", "börek", "köfte", "çorba", "sarma", "kumpir",
    "menemen", "musakka", "kısır", "cacık", "güveç", "tarhana", "pide", "gözleme", "kokoreç", "tantuni",
    "hamsi", "karnıyarık", "döner", "türlü", "kavurma", "mücver", "keşkek", "piyaz", "kuymak", "çılbır")

# Name phonotactics. Syllables are V, VC, CV or CVC; only the first syllable may lack an onset, so there is never a
# vowel hiatus (a vowel-initial shared ending such as -er needs a consonant-final stem). Native-Turkish regularities:
# o/ö only in the first syllable; no ğ (never syllable-initial) and no j (loan letter); no word-final b/c/d/g (final
# devoicing); no doubled letters (which also excludes doubled vowels). Syllable count = vowel count, 2 or 3.
VOWELS_LOWER = "aeıioöuü"
_FIRST_VOWELS = VOWELS_LOWER
_LATER_VOWELS = "aeıiuü"
_ONSETS = "bcçdfghklmnprsştvyz"
_CODAS = "çfhklmnprsştyz"
_NAME_LETTERS = frozenset(_ONSETS + _CODAS + VOWELS_LOWER)
P_EMPTY_ONSET_FIRST = 0.15
P_EMPTY_CODA = 0.40
P_THREE_SYLLABLES = 0.30
NAME_MIN_LEN, NAME_MAX_LEN = 4, 9
_LATER_SYLLABLE = re.compile(f"[{_ONSETS}][{_LATER_VOWELS}][{_CODAS}]?")
_MAX_ATTEMPTS_PER_ITEM = 5000


def partition_sizes(n_names: int) -> dict[str, int]:
    """60 p3 / 60 c0 / 280 probe at 400 names; p3 and c0 are round(15 %) of n_names in general."""
    n_p3 = (3 * n_names + 10) // 20
    return {"p3": n_p3, "c0": n_p3, "probe": n_names - 2 * n_p3}


def n_prefix_pairs(n_names: int) -> int:
    """20 prefix-extension pairs at 400 names (one pair per 20 names)."""
    return n_names // 20


def n_tok_frequent(group_size: int) -> int:
    """25 % of a partition group, rounded half up (15 / 15 / 70 = 100 at 400 names)."""
    return (group_size + 2) // 4


def ends_with_shared_ending(name: str) -> bool:
    low = _lower(name)
    return any(low.endswith(e) for e in SHARED_ENDINGS)


def _surface_forms(low: str) -> tuple[str, ...]:
    """Bare name plus its six informal (apostrophe-less) inflected forms, lower case. Formal forms contain the bare
    name followed by an apostrophe, so they are covered by the bare form."""
    return (low,) + tuple(informal(low, c) for c in INFLECTED)


def _name_shape_problems(low: str) -> list[str]:
    problems = []
    if not NAME_MIN_LEN <= len(low) <= NAME_MAX_LEN:
        problems.append(f"length {len(low)} outside [{NAME_MIN_LEN}, {NAME_MAX_LEN}]")
    if any(ch not in _NAME_LETTERS for ch in low):
        problems.append("letter outside the name alphabet")
    vowel_pos = [i for i, ch in enumerate(low) if ch in VOWELS_LOWER]
    if not 2 <= len(vowel_pos) <= 3:
        problems.append(f"{len(vowel_pos)} syllables")
    if any(a in VOWELS_LOWER and b in VOWELS_LOWER for a, b in zip(low, low[1:])):
        problems.append("vowel hiatus")
    if any(a == b for a, b in zip(low, low[1:])):
        problems.append("doubled letter")
    if low and low[-1] in "bcdgğ":
        problems.append("word-final voiced stop")
    if any(low[i] in "oö" for i in vowel_pos[1:]):
        problems.append("o/ö outside the first syllable")
    return problems


def _syllable(rng: random.Random, first: bool) -> str:
    # Draw order (part of the generator definition): [onset-empty draw, first syllable only], onset, vowel, coda.
    onset = "" if (first and rng.random() < P_EMPTY_ONSET_FIRST) else rng.choice(_ONSETS)
    vowel = rng.choice(_FIRST_VOWELS if first else _LATER_VOWELS)
    coda = "" if rng.random() < P_EMPTY_CODA else rng.choice(_CODAS)
    return onset + vowel + coda


def _syllables(rng: random.Random, k: int) -> str:
    return "".join(_syllable(rng, first=(i == 0)) for i in range(k))


def _name_candidate(rng: random.Random, kind: str) -> str:
    n_syllables = 3 if rng.random() < P_THREE_SYLLABLES else 2
    if kind == "shared":
        ending = rng.choice(SHARED_ENDINGS)
        stem = _syllables(rng, n_syllables - 1)
        if ending[0] in VOWELS_LOWER and stem[-1] in VOWELS_LOWER:
            # A vowel-initial ending (-er, -ay, -un, -ün) needs a consonant-final stem (no hiatus). Adding a coda
            # here, instead of rejecting, keeps the 16 endings equally likely.
            stem += rng.choice(_CODAS)
        return stem + ending
    return _syllables(rng, n_syllables)


def _stream(seed: int, label: str) -> random.Random:
    """Independent, platform-stable random stream per generator component (str seeds hash with SHA-512)."""
    return random.Random(f"{GENERATOR_VERSION}|seed={seed}|{label}")


def _normalise_words(words: Iterable[str] | None) -> frozenset[str]:
    if words is None:
        return frozenset()
    if isinstance(words, str):
        raise TypeError("forbidden_words must be an iterable of words, not a single string")
    out: set[str] = set()
    for w in words:
        if not isinstance(w, str):
            raise TypeError(f"forbidden word must be str, got {type(w).__name__}")
        w = unicodedata.normalize("NFC", w)
        out.add(_lower(w))
    return frozenset(out)


# =====================================================================================================================
# World
# =====================================================================================================================
@dataclass(frozen=True)
class World:
    seed: int
    names: tuple[str, ...]                       # n invented one-word names, sorted (Python code-point order)
    objects: dict[str, tuple[str, ...]]          # "A": 48 towns, "B": 30 occupations, "C": 30 dishes (sorted)
    facts: dict[tuple[str, str], str]            # (name, rel) -> object; uniform and balanced per relation
    partition: dict[str, str]                    # name -> "p3" | "c0" | "probe"
    p3_held: frozenset[tuple[str, str]]          # (name, case): 2 frames of relation A per p3 entity, never trained
    prefix_pairs: tuple[tuple[str, str], ...]    # (n1, n2), n2 = n1 + one syllable; both in the probe pool
    tok_frequent: frozenset[str]                 # 25 % of every partition group (exploratory F_freq arm)

    def to_json(self) -> dict:
        """Canonical JSON-able form (everything sorted); the input of ``name_hash``."""
        return {
            "schema": GENERATOR_VERSION,
            "seed": self.seed,
            "names": list(self.names),
            "objects": {rel: list(objs) for rel, objs in sorted(self.objects.items())},
            "facts": [[name, rel, obj] for (name, rel), obj in sorted(self.facts.items())],
            "partition": {name: part for name, part in sorted(self.partition.items())},
            "p3_held": [[name, case] for name, case in sorted(self.p3_held)],
            "prefix_pairs": [list(pair) for pair in sorted(self.prefix_pairs)],
            "tok_frequent": sorted(self.tok_frequent),
        }

    def name_hash(self) -> str:
        """SHA-256 over the canonical JSON of the world (sorted keys, no whitespace, UTF-8)."""
        payload = json.dumps(self.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def members(self, part: str) -> tuple[str, ...]:
        """Sorted names of one partition group ("p3", "c0" or "probe")."""
        if part not in ("p3", "c0", "probe"):
            raise ValueError(f"unknown partition group {part!r}")
        return tuple(name for name in self.names if self.partition[name] == part)


def world_from_json(data: Mapping, expected_hash: str | None = None) -> World:
    """Inverse of ``World.to_json``; verifies the schema and, if given, the content hash."""
    if data.get("schema") != GENERATOR_VERSION:
        raise ValueError(f"world schema {data.get('schema')!r} != {GENERATOR_VERSION!r}")
    world = World(
        seed=int(data["seed"]),
        names=tuple(data["names"]),
        objects={rel: tuple(objs) for rel, objs in data["objects"].items()},
        facts={(name, rel): obj for name, rel, obj in data["facts"]},
        partition=dict(data["partition"]),
        p3_held=frozenset((name, case) for name, case in data["p3_held"]),
        prefix_pairs=tuple((a, b) for a, b in data["prefix_pairs"]),
        tok_frequent=frozenset(data["tok_frequent"]),
    )
    if expected_hash is not None and world.name_hash() != expected_hash:
        raise ValueError("world content hash mismatch")
    return world


def _make_towns(rng: random.Random, blocked: frozenset[str]) -> list[str]:
    """48 invented towns: one first-type syllable + a town ending, 4 per ending, distinct, outside ``blocked``."""
    towns: list[str] = []
    seen: set[str] = set()
    for ending in TOWN_ENDINGS:
        made = attempts = 0
        while made < TOWNS_PER_ENDING:
            attempts += 1
            if attempts > _MAX_ATTEMPTS_PER_ITEM:
                raise RuntimeError(f"could not generate towns ending in {ending!r}")
            town = _syllable(rng, first=True) + ending
            if _town_shape_problems(town) or town in seen or town in blocked:
                continue
            seen.add(town)
            towns.append(tr_capitalize(town))
            made += 1
    return towns


def _town_shape_problems(low: str) -> list[str]:
    problems = []
    if len(low) < 4:
        problems.append("too short")
    if sum(low.endswith(e) for e in TOWN_ENDINGS) < 1:
        problems.append("no town ending")
    if any(a in VOWELS_LOWER and b in VOWELS_LOWER for a, b in zip(low, low[1:])):
        problems.append("vowel hiatus")
    if any(a == b for a, b in zip(low, low[1:])):
        problems.append("doubled letter")
    if low[:1] == "ğ":
        problems.append("initial ğ")
    return problems


def _make_names(rng: random.Random, n_names: int, blocked: frozenset[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Lower-case names and prefix pairs. Invariants enforced on every accepted name:
    shape (``_name_shape_problems``), none of its 7 surface forms blocked or used by another name, and no prefix
    relation to another name except the designated pairs."""
    accepted: list[str] = []
    name_set: set[str] = set()
    proper_prefixes: set[str] = set()       # all proper prefixes of accepted names
    forms_used: set[str] = set()

    def fits(low: str) -> bool:
        if _name_shape_problems(low):
            return False
        forms = _surface_forms(low)
        if any(f in blocked or f in forms_used for f in forms):
            return False
        if low in name_set or low in proper_prefixes:
            return False
        return not any(low[:k] in name_set for k in range(1, len(low)))

    def accept(low: str) -> None:
        accepted.append(low)
        name_set.add(low)
        forms_used.update(_surface_forms(low))
        proper_prefixes.update(low[:k] for k in range(1, len(low)))

    # 1) prefix-extension pairs: n1 has two syllables, n2 = n1 + one consonant-initial syllable.
    pairs: list[tuple[str, str]] = []
    attempts = 0
    while len(pairs) < n_prefix_pairs(n_names):
        attempts += 1
        if attempts > _MAX_ATTEMPTS_PER_ITEM * max(1, n_prefix_pairs(n_names)):
            raise RuntimeError("could not generate the prefix-extension pairs")
        n1 = _syllables(rng, 2)
        n2 = n1 + _syllable(rng, first=False)
        if not (fits(n1) and fits(n2)) or set(_surface_forms(n1)) & set(_surface_forms(n2)):
            continue
        accept(n1)
        accept(n2)
        pairs.append((n1, n2))

    # 2) the other names, with quotas so that exactly n_names // 2 names end in a shared ending.
    shared_in_pairs = sum(ends_with_shared_ending(w) for pair in pairs for w in pair)
    need_shared = n_names // 2 - shared_in_pairs
    need_plain = n_names - len(accepted) - need_shared
    if need_shared < 0 or need_plain < 0:
        raise RuntimeError("shared-ending quota cannot be met")
    slots = ["shared"] * need_shared + ["plain"] * need_plain
    rng.shuffle(slots)
    for kind in slots:
        for _ in range(_MAX_ATTEMPTS_PER_ITEM):
            low = _name_candidate(rng, kind)
            if kind == "plain" and ends_with_shared_ending(low):
                continue
            if fits(low):
                accept(low)
                break
        else:
            raise RuntimeError(f"could not generate a {kind} name (blocked set too large?)")
    return accepted, pairs


def _assign_facts(rng: random.Random, names: Sequence[str],
                  objects: Mapping[str, tuple[str, ...]]) -> dict[tuple[str, str], str]:
    """Uniform and balanced: per relation every object is used floor(n/k) or ceil(n/k) times, in random order."""
    facts: dict[tuple[str, str], str] = {}
    for rel in RELATIONS:
        objs = list(objects[rel])
        pool = objs * (len(names) // len(objs)) + rng.sample(objs, len(names) % len(objs))
        rng.shuffle(pool)
        for name, obj in zip(names, pool):
            facts[(name, rel)] = obj
    return facts


def _partition(rng: random.Random, names: Sequence[str],
               pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Pair members go to the probe pool (PX is a locality probe); the rest is shuffled into p3, c0, probe."""
    sizes = partition_sizes(len(names))
    paired = {w for pair in pairs for w in pair}
    free = [w for w in names if w not in paired]
    if len(free) < sizes["p3"] + sizes["c0"]:
        raise RuntimeError("too few unpaired names for the p3 and c0 pools")
    rng.shuffle(free)
    partition = {w: "probe" for w in names}
    for w in free[:sizes["p3"]]:
        partition[w] = "p3"
    for w in free[sizes["p3"]:sizes["p3"] + sizes["c0"]]:
        partition[w] = "c0"
    return partition


def _p3_held(rng: random.Random, names: Sequence[str], partition: Mapping[str, str]) -> frozenset[tuple[str, str]]:
    held: set[tuple[str, str]] = set()
    for name in names:                       # sorted order
        if partition[name] == "p3":
            for case in rng.sample(CASES, 2):
                held.add((name, case))
    return frozenset(held)


def _tok_frequent(rng: random.Random, names: Sequence[str], partition: Mapping[str, str]) -> frozenset[str]:
    chosen: set[str] = set()
    for part in ("p3", "c0", "probe"):
        members = [w for w in names if partition[w] == part]
        chosen.update(rng.sample(members, n_tok_frequent(len(members))))
    return frozenset(chosen)


def make_world(seed: int, n_names: int = 400, *, forbidden_words: Iterable[str] | None = None) -> World:
    """Generate the TR-Paradigma world for one world seed (PROTOCOL §4). Deterministic in (seed, n_names,
    forbidden_words). See the module docstring for how ``forbidden_words`` is obtained.

    Components draw from independent streams (towns, names, facts, partition, p3_held, tok_frequent), so e.g. the
    partition does not depend on how many name candidates were rejected. The result is self-checked with
    ``world_problems`` and a RuntimeError is raised if any invariant fails.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(n_names, bool) or not isinstance(n_names, int) or n_names < 20:
        raise ValueError("n_names must be an int >= 20")
    forbidden = _normalise_words(forbidden_words)
    real_objects = frozenset(_lower(o) for o in OCCUPATIONS + DISHES)
    towns = _make_towns(_stream(seed, "towns"), forbidden | FRAME_WORDS | real_objects)
    objects = {"A": tuple(sorted(towns)), "B": tuple(sorted(OCCUPATIONS)), "C": tuple(sorted(DISHES))}
    blocked = forbidden | FRAME_WORDS | real_objects | frozenset(_lower(t) for t in towns)
    names_low, pairs_low = _make_names(_stream(seed, "names"), n_names, blocked)
    names = tuple(sorted(tr_capitalize(w) for w in names_low))
    prefix_pairs = tuple(sorted((tr_capitalize(a), tr_capitalize(b)) for a, b in pairs_low))
    facts = _assign_facts(_stream(seed, "facts"), names, objects)
    partition = _partition(_stream(seed, "partition"), names, prefix_pairs)
    world = World(
        seed=seed,
        names=names,
        objects=objects,
        facts=facts,
        partition=partition,
        p3_held=_p3_held(_stream(seed, "p3_held"), names, partition),
        prefix_pairs=prefix_pairs,
        tok_frequent=_tok_frequent(_stream(seed, "tok_frequent"), names, partition),
    )
    problems = world_problems(world, forbidden_words=forbidden)
    if problems:
        raise RuntimeError(f"generated world violates {len(problems)} invariant(s): {problems[:5]}")
    return world


def world_problems(world: World, forbidden_words: Iterable[str] | None = None) -> list[str]:
    """All violated world invariants (empty list = valid). Messages contain only world strings."""
    problems: list[str] = []
    names = world.names
    n = len(names)
    name_set = set(names)
    if list(names) != sorted(names):
        problems.append("names are not sorted")
    if len(name_set) != n:
        problems.append("duplicate names")
    lows = [_lower(x) for x in names]
    if len(set(lows)) != n:
        problems.append("names collide after lower-casing")

    # --- names: shape, capitalisation, shared endings, unique surface forms, prefix relations
    for name, low in zip(names, lows):
        if tr_capitalize(low) != name:
            problems.append(f"{name}: not Turkish-capitalised")
        problems.extend(f"{name}: {why}" for why in _name_shape_problems(low))
    n_shared = sum(ends_with_shared_ending(low) for low in lows)
    if n_shared != n // 2:
        problems.append(f"{n_shared} names end in a shared ending, expected {n // 2}")
    owner: dict[str, str] = {}
    for name, low in zip(names, lows):
        for form in _surface_forms(low):
            if owner.setdefault(form, name) != name:
                problems.append(f"surface form {form!r} shared by {owner[form]} and {name}")
    pairs = set(world.prefix_pairs)
    by_low = dict(zip(lows, names))
    for name, low in zip(names, lows):
        for k in range(1, len(low)):
            other = by_low.get(low[:k])
            if other is not None and (other, name) not in pairs:
                problems.append(f"{other} is a prefix of {name} but not a designated pair")
    if len(world.prefix_pairs) != n_prefix_pairs(n):
        problems.append(f"{len(world.prefix_pairs)} prefix pairs, expected {n_prefix_pairs(n)}")
    for a, b in world.prefix_pairs:
        la, lb = _lower(a), _lower(b)
        if a not in name_set or b not in name_set:
            problems.append(f"prefix pair ({a}, {b}) not in the world")
        if not (lb.startswith(la) and _LATER_SYLLABLE.fullmatch(lb[len(la):])):
            problems.append(f"prefix pair ({a}, {b}) is not n1 + one syllable")
        if sum(ch in VOWELS_LOWER for ch in la) != 2:
            problems.append(f"prefix pair ({a}, {b}): n1 does not have two syllables")
        if world.partition.get(a) != "probe" or world.partition.get(b) != "probe":
            problems.append(f"prefix pair ({a}, {b}) not in the probe pool")

    # --- objects
    if set(world.objects) != set(RELATIONS):
        problems.append(f"object relations {sorted(world.objects)} != {list(RELATIONS)}")
    else:
        towns = world.objects["A"]
        if len(towns) != len(TOWN_ENDINGS) * TOWNS_PER_ENDING or len(set(towns)) != len(towns):
            problems.append("relation A needs 48 distinct towns")
        for ending in TOWN_ENDINGS:
            k = sum(_lower(t).endswith(ending) for t in towns)
            if k != TOWNS_PER_ENDING:
                problems.append(f"{k} towns end in {ending!r}, expected {TOWNS_PER_ENDING}")
        for town in towns:
            if tr_capitalize(_lower(town)) != town or " " in town:
                problems.append(f"town {town}: not one capitalised word")
            problems.extend(f"town {town}: {why}" for why in _town_shape_problems(_lower(town)))
        if tuple(world.objects["B"]) != tuple(sorted(OCCUPATIONS)):
            problems.append("relation B objects differ from OCCUPATIONS")
        if tuple(world.objects["C"]) != tuple(sorted(DISHES)):
            problems.append("relation C objects differ from DISHES")
        all_objects = [_lower(o) for rel in RELATIONS for o in world.objects[rel]]
        if len(set(all_objects)) != len(all_objects):
            problems.append("object strings are not distinct across relations")
        object_set = set(all_objects)
        for form, name in owner.items():
            if form in object_set:
                problems.append(f"{name}: surface form {form!r} equals an object")

        # --- facts: complete, valid, balanced
        expected_keys = {(name, rel) for name in names for rel in RELATIONS}
        if set(world.facts) != expected_keys:
            problems.append("facts do not cover exactly every (name, relation)")
        for rel in RELATIONS:
            objs = world.objects[rel]
            counts = Counter(o for (name, r), o in world.facts.items() if r == rel)
            if any(o not in objs for o in counts):
                problems.append(f"relation {rel}: fact object outside the relation's objects")
            used = [counts.get(o, 0) for o in objs]
            if used and max(used) - min(used) > 1:
                problems.append(f"relation {rel}: object counts unbalanced ({min(used)}..{max(used)})")

    # --- partition, p3_held, tok_frequent
    if set(world.partition) != name_set:
        problems.append("partition does not cover exactly the names")
    sizes = partition_sizes(n)
    got = Counter(world.partition.values())
    if dict(got) != {k: v for k, v in sizes.items() if v}:
        problems.append(f"partition sizes {dict(got)} != {sizes}")
    held_per: Counter = Counter()
    for name, case in world.p3_held:
        if world.partition.get(name) != "p3":
            problems.append(f"p3_held cell for non-p3 entity {name}")
        if case not in CASES:
            problems.append(f"p3_held cell with unknown case {case!r}")
        held_per[name] += 1
    for name in names:
        if world.partition.get(name) == "p3" and held_per[name] != 2:
            problems.append(f"p3 entity {name} has {held_per[name]} held frames, expected 2")
    if not world.tok_frequent <= name_set:
        problems.append("tok_frequent contains unknown names")
    for part in ("p3", "c0", "probe"):
        members = [w for w in names if world.partition.get(w) == part]
        k = sum(w in world.tok_frequent for w in members)
        if k != n_tok_frequent(len(members)):
            problems.append(f"tok_frequent has {k} {part} entities, expected {n_tok_frequent(len(members))}")

    # --- collisions with prose words and frame words
    blocked = _normalise_words(forbidden_words) | FRAME_WORDS
    for form, name in owner.items():
        if form in blocked:
            problems.append(f"{name}: surface form {form!r} is a forbidden (prose or frame) word")
    for town in world.objects.get("A", ()):
        if _lower(town) in blocked:
            problems.append(f"town {town} is a forbidden (prose or frame) word")
    return problems


# =====================================================================================================================
# One pretraining pass (PROTOCOL §5)
# =====================================================================================================================
CTX_PROBS: tuple[tuple[str, float], ...] = (("none", 0.50), ("P1", 0.25), ("P2", 0.25))
P_INFORMAL = 0.25


def _draw_context(u: float) -> str:
    """Map one uniform draw u in [0, 1) to a context by the cumulative CTX_PROBS (none < .50 <= P1 < .75 <= P2)."""
    acc = 0.0
    for ctx, p in CTX_PROBS:
        acc += p
        if u < acc:
            return ctx
    return CTX_PROBS[-1][0]


def trained_cells(world: World) -> list[tuple[str, str, str]]:
    """Every (name, rel, case) cell except the p3-held cells of relation A, in canonical order
    (sorted names x RELATIONS x CASES)."""
    return [(name, rel, case) for name in world.names for rel in RELATIONS for case in CASES
            if not (rel == "A" and (name, case) in world.p3_held)]


def pass_sentences(world: World, pass_idx: int, rng: random.Random) -> list[tuple[str, dict]]:
    """One pass: every trained cell exactly once, in canonical order (``trained_cells``; NOT shuffled — the trainer
    mixes in prose chunks and shuffles with its data-order stream).

    Per cell, in this order: one ``rng.random()`` draw for the context (none < .50 <= P1 < .75 <= P2), then, for
    inflected cases only, one ``rng.random()`` draw for the orthography (informal iff < .25). NOM is always formal
    (its informal form is identical). All randomness comes from ``rng`` (the caller's data-order stream);
    ``pass_idx`` is validated but does not alter the draws. meta = {name, rel, case, orth, ctx, obj}.
    """
    if isinstance(pass_idx, bool) or not isinstance(pass_idx, int) or pass_idx < 0:
        raise ValueError("pass_idx must be a non-negative int")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be a random.Random instance")
    out: list[tuple[str, dict]] = []
    for name, rel, case in trained_cells(world):
        ctx = _draw_context(rng.random())
        orth = "formal"
        if case != "nom" and rng.random() < P_INFORMAL:
            orth = "informal"
        obj = world.facts[(name, rel)]
        meta = {"name": name, "rel": rel, "case": case, "orth": orth, "ctx": ctx, "obj": obj}
        out.append((sentence(name, rel, case, obj, orth, ctx), meta))
    return out


# =====================================================================================================================
# Prose corpus (PROTOCOL §5): never printed, never sampled into logs
# =====================================================================================================================
BOOK_DIR = Path("docs") / "book" / "tr"
EDUCATION_DIR = Path("education")
TRAIN_CHAPTERS: tuple[int, ...] = tuple(range(1, 10))      # docs/book/tr/01-*.md .. 09-*.md
HELDOUT_CHAPTERS: tuple[int, ...] = (10, 11, 12)           # bits-per-byte text, never trained
# Exclusions from education/, identified by file NAME only (PROTOCOL §5, data-rights decision):
EDUCATION_EXCLUDE_PREFIXES: dict[str, str] = {
    "mektup": "personal letters (mektuplar.docx, mektuplar-2.docx; any file whose name starts with 'mektup')",
}
EDUCATION_EXCLUDE_FILES: dict[str, str] = {
    "biobot-3.0-sonhali-01.06.2025.docx": "health-supplement formulation (unreviewed health content)",
}
EDUCATION_SUFFIXES: tuple[str, ...] = (".docx", ".txt", ".json")
MIN_PARAGRAPH_WORDS = 5          # shorter paragraphs (headings, captions, one-line questions) are dropped

_CHAPTER_FILE = re.compile(r"^(\d{2})-.+\.md$")
_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _book_chapters(repo_root: Path) -> dict[int, Path]:
    book = repo_root / BOOK_DIR
    if not book.is_dir():
        raise FileNotFoundError(f"book directory missing: {book}")
    chapters: dict[int, Path] = {}
    for path in sorted(book.iterdir()):
        m = _CHAPTER_FILE.match(path.name)
        if m:
            number = int(m.group(1))
            if number in chapters:
                raise ValueError(f"two files for book chapter {number:02d}")
            chapters[number] = path
    missing = [c for c in TRAIN_CHAPTERS + HELDOUT_CHAPTERS if c not in chapters]
    if missing:
        raise FileNotFoundError(f"book chapters missing: {missing}")
    return chapters


def _education_files(repo_root: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """(included, excluded-with-reason), sorted by relative path. Raises on an unsupported file type or when a
    file named in EDUCATION_EXCLUDE_FILES is absent (a renamed file must never slip into training)."""
    edu = repo_root / EDUCATION_DIR
    if not edu.is_dir():
        raise FileNotFoundError(f"education directory missing: {edu}")
    included: list[Path] = []
    excluded: list[tuple[Path, str]] = []
    seen_excluded_names: set[str] = set()
    for path in sorted((p for p in edu.rglob("*") if p.is_file()), key=lambda p: p.relative_to(repo_root).as_posix()):
        low_name = _lower(path.name)
        prefix_reason = next((r for pre, r in EDUCATION_EXCLUDE_PREFIXES.items() if low_name.startswith(pre)), None)
        if prefix_reason is not None:
            excluded.append((path, prefix_reason))
        elif path.name in EDUCATION_EXCLUDE_FILES:
            excluded.append((path, EDUCATION_EXCLUDE_FILES[path.name]))
            seen_excluded_names.add(path.name)
        elif path.suffix.lower() in EDUCATION_SUFFIXES:
            included.append(path)
        else:
            raise ValueError(f"unsupported file type in education/: {path.relative_to(repo_root).as_posix()}")
    absent = sorted(set(EDUCATION_EXCLUDE_FILES) - seen_excluded_names)
    if absent:
        raise FileNotFoundError(f"excluded education file(s) not found (renamed?): {absent}; update "
                                f"EDUCATION_EXCLUDE_FILES before loading prose")
    return included, excluded


def prose_sources(repo_root: Path | str, *, include_education: bool = True) -> dict:
    """File-level manifest of the prose configuration (paths relative to the repo, sizes, SHA-256, reasons for
    exclusion). Contains no text; suitable for the protocol freeze manifest."""
    repo_root = Path(repo_root)
    chapters = _book_chapters(repo_root)

    def entry(path: Path, **extra) -> dict:
        return {"path": path.relative_to(repo_root).as_posix(), "bytes": path.stat().st_size,
                "sha256": _sha256_file(path), **extra}

    train = [entry(chapters[c]) for c in TRAIN_CHAPTERS]
    excluded = []
    if include_education:
        included, excl = _education_files(repo_root)
        train += [entry(p) for p in included]
        excluded = [entry(p, reason=r) for p, r in excl]
    return {"train": train, "heldout": [entry(chapters[c]) for c in HELDOUT_CHAPTERS], "excluded": excluded,
            "include_education": include_education, "min_paragraph_words": MIN_PARAGRAPH_WORDS}


def _finish_paragraph(raw: str) -> str | None:
    text = " ".join(unicodedata.normalize("NFC", raw).split())
    return text if len(text.split()) >= MIN_PARAGRAPH_WORDS else None


def _clean_inline_markdown(line: str) -> str:
    """Remove inline markdown syntax but keep the words: inline code keeps its content (backticks dropped), links
    keep their text, images and inline HTML tags are dropped, emphasis markers and $ math delimiters are dropped."""
    # Images and links first: their bracketed text may itself contain a code span ([`Name`](path)).
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
    parts = re.split(r"(`+[^`]*`+)", line)
    out = []
    for part in parts:
        if part.startswith("`"):
            out.append(part.strip("`"))
            continue
        part = re.sub(r"</?[A-Za-z][^>]*>", " ", part)
        part = re.sub(r"\*\*|__", "", part)
        part = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"\1", part)
        part = re.sub(r"(?<![\w$])\$(?!\$)([^$\n]+?)\$(?![\w$])", r"\1", part)
        out.append(part)
    return "".join(out)


def _markdown_paragraphs(text: str, source: str) -> list[str]:
    """Book chapter -> prose paragraphs. Dropped: fenced code blocks (``` and ~~~), display math ($$ blocks and
    $$..$$ spans), tables (| rows), HTML lines, headings (#), horizontal rules. Blockquote and list markers are
    stripped; each list item is its own paragraph."""
    kept: list[str] = []
    fence: str | None = None
    in_math = False
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if fence is not None:
            if re.fullmatch(re.escape(fence[0]) + "{%d,}\\s*" % len(fence), s):
                fence = None
            continue
        m = _FENCE_OPEN.match(s)
        if m:
            fence = m.group(1)
            continue
        if s == "$$":
            in_math = not in_math
            continue
        if in_math:
            continue
        kept.append(line)
    if fence is not None or in_math:
        raise ValueError(f"unclosed code fence or $$ block in {source}")
    body = re.sub(r"\$\$.*?\$\$", " ", "\n".join(kept), flags=re.S)
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", body):
        items: list[list[str]] = []
        for line in block.split("\n"):
            s = line.strip()
            if not s or s.startswith("|") or s.startswith("<") or s.startswith("#") or _RULE.match(s):
                continue
            while s.startswith(">"):
                s = s[1:].strip()
            if _LIST_ITEM.match(s):
                items.append([_LIST_ITEM.sub("", s, count=1)])
            elif items:
                items[-1].append(s)
            else:
                items.append([s])
        for item in items:
            par = _finish_paragraph(_clean_inline_markdown(" ".join(item)))
            if par is not None:
                paragraphs.append(par)
    return paragraphs


def _json_strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _json_strings(v)
    elif isinstance(value, dict):
        for k in value:                       # file order (json preserves it)
            yield from _json_strings(value[k])


def _education_paragraphs(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        try:
            import docx  # python-docx (requirements.txt)
        except ImportError as exc:
            raise ImportError("load_prose needs python-docx (>=1.1, listed in requirements.txt) to read "
                              "education/*.docx; install it or call load_prose(..., include_education=False) "
                              "for the protocol's book-only fallback") from exc
        raw = [p.text for p in docx.Document(str(path)).paragraphs]
    elif suffix == ".txt":
        raw = re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        raw = [block for s in _json_strings(json.loads(path.read_text(encoding="utf-8")))
               for block in re.split(r"\n\s*\n", s)]
    else:
        raise ValueError(f"unsupported education file type: {path.name}")
    return [p for p in (_finish_paragraph(r) for r in raw) if p is not None]


def load_prose(repo_root: Path | str, *, include_education: bool = True) -> tuple[list[str], list[str]]:
    """(train, heldout) prose paragraphs (PROTOCOL §5).

    train: docs/book/tr chapters 01-09 (``_markdown_paragraphs``: fenced code removed) followed by education/
    texts in sorted path order (.docx via python-docx, .txt and .json as UTF-8), except EDUCATION_EXCLUDE_PREFIXES
    and EDUCATION_EXCLUDE_FILES. heldout: book chapters 10-12. Paragraphs are NFC-normalised, whitespace-collapsed,
    and kept only with at least MIN_PARAGRAPH_WORDS words. Nothing is printed.
    """
    repo_root = Path(repo_root)
    chapters = _book_chapters(repo_root)
    train: list[str] = []
    for c in TRAIN_CHAPTERS:
        train.extend(_markdown_paragraphs(chapters[c].read_text(encoding="utf-8"), chapters[c].name))
    if include_education:
        included, _ = _education_files(repo_root)
        for path in included:
            train.extend(_education_paragraphs(path))
    heldout: list[str] = []
    for c in HELDOUT_CHAPTERS:
        heldout.extend(_markdown_paragraphs(chapters[c].read_text(encoding="utf-8"), chapters[c].name))
    if not train or not heldout:
        raise RuntimeError("prose loading produced an empty split")
    return train, heldout


def prose_vocabulary(texts: Iterable[str]) -> frozenset[str]:
    """Every maximal letter run in ``texts`` (NFC, Turkish-lower-cased): the ``forbidden_words`` of make_world."""
    vocab: set[str] = set()
    for text in texts:
        vocab.update(_lower(w) for w in _words_of(unicodedata.normalize("NFC", text)))
    return frozenset(vocab)


def make_study_world(seed: int, repo_root: Path | str, n_names: int = 400, *,
                     include_education: bool = True) -> World:
    """The canonical study world: ``make_world`` with forbidden_words = the vocabulary of the train AND held-out
    prose of the same prose configuration (so names collide with no prose word and towns with none either)."""
    train, heldout = load_prose(repo_root, include_education=include_education)
    return make_world(seed, n_names, forbidden_words=prose_vocabulary(train + heldout))


_SENTENCE_END = re.compile(r"([.!?…]+[\"'”’»)\]]*)\s+(?=\S)")


def _split_sentences(paragraph: str) -> list[str]:
    out, start = [], 0
    for m in _SENTENCE_END.finditer(paragraph):
        out.append(paragraph[start:m.end(1)])
        start = m.end()
    out.append(paragraph[start:])
    return [s.strip() for s in out if s.strip()]


def prose_chunks(texts: list[str], tok, max_tokens: int = 60) -> list[tuple[str, bool]]:
    """Paragraphs -> (chunk_text, is_end) with len(tok.encode(chunk_text)) <= max_tokens (BOS/EOS not counted).

    Whitespace inside a paragraph is normalised to single spaces. Sentences are packed greedily; a sentence longer
    than max_tokens is cut at spaces; a single word longer than max_tokens (rare: long paths/URLs) is cut at
    character boundaries and its pieces are never joined to neighbours with an added space. is_end is True only for
    the last chunk of each paragraph, so the trainer appends EOS only at true paragraph ends. Empty paragraphs give
    no chunk. Every chunk length is verified with ``tok.encode``.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive int")

    def ntok(s: str) -> int:
        return len(tok.encode(s))

    def split_word(word: str) -> list[str]:
        pieces, start = [], 0
        while start < len(word):
            end = start + 1
            if ntok(word[start:end]) > max_tokens:
                raise ValueError("max_tokens is smaller than the token count of a single character")
            while end < len(word) and ntok(word[start:end + 1]) <= max_tokens:
                end += 1
            pieces.append(word[start:end])
            start = end
        return pieces

    out: list[tuple[str, bool]] = []
    for text in texts:
        paragraph = " ".join(text.split())
        if not paragraph:
            continue
        units: list[tuple[str, bool]] = []          # (text, may be joined to neighbours with a space)
        for sent in _split_sentences(paragraph):
            if ntok(sent) <= max_tokens:
                units.append((sent, True))
                continue
            for word in sent.split(" "):
                if ntok(word) <= max_tokens:
                    units.append((word, True))
                else:
                    units.extend((piece, False) for piece in split_word(word))
        chunks: list[str] = []
        current: str | None = None
        for unit, joinable in units:
            if not joinable:
                if current is not None:
                    chunks.append(current)
                    current = None
                chunks.append(unit)
            elif current is None:
                current = unit
            elif ntok(current + " " + unit) <= max_tokens:
                current = current + " " + unit
            else:
                chunks.append(current)
                current = unit
        if current is not None:
            chunks.append(current)
        for j, chunk in enumerate(chunks):
            if ntok(chunk) > max_tokens:   # cannot happen by construction; never emit an over-long chunk silently
                raise RuntimeError("internal error: prose chunk exceeds max_tokens")
            out.append((chunk, j == len(chunks) - 1))
    return out


# =====================================================================================================================
# Exposure ledger and contamination test (PROTOCOL §5, V10)
# =====================================================================================================================
_META_KEYS: tuple[str, ...] = ("name", "rel", "obj", "case", "ctx", "orth")


class ExposureLedger:
    """Counts of trained cell exposures per (fact, frame, ctx, orth); fact = (name, rel, obj), frame = case."""

    SCHEMA = "cev11-exposure-1"

    def __init__(self) -> None:
        self._counts: Counter = Counter()

    @staticmethod
    def _key(meta: Mapping[str, str]) -> tuple[str, ...]:
        missing = [k for k in _META_KEYS if k not in meta]
        if missing:
            raise KeyError(f"cell metadata lacks {missing}")
        key = tuple(str(meta[k]) for k in _META_KEYS)
        _, rel, _, case, ctx, orth = key
        if rel not in RELATIONS or case not in CASES or ctx not in CONTEXTS or orth not in ORTHOGRAPHIES:
            raise ValueError(f"invalid cell metadata: rel={rel!r} case={case!r} ctx={ctx!r} orth={orth!r}")
        if case == "nom" and orth != "formal":
            raise ValueError("a NOM cell is always formal")
        return key

    def add(self, cell_meta: Mapping[str, str]) -> None:
        self._counts[self._key(cell_meta)] += 1

    def add_all(self, metas: Iterable[Mapping[str, str]]) -> None:
        for meta in metas:
            self.add(meta)

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def count(self, **filters: str) -> int:
        """Number of exposures matching all given fields, e.g. count(name="Ayten", rel="A", orth="informal")."""
        unknown = set(filters) - set(_META_KEYS)
        if unknown:
            raise KeyError(f"unknown ledger fields {sorted(unknown)}")
        idx = [(_META_KEYS.index(k), v) for k, v in filters.items()]
        return sum(c for key, c in self._counts.items() if all(key[i] == v for i, v in idx))

    def to_json(self) -> dict:
        cells = [dict(zip(_META_KEYS, key), n=c) for key, c in sorted(self._counts.items())]
        by_ctx = Counter()
        by_orth = Counter()
        for key, c in self._counts.items():
            by_ctx[key[4]] += c
            by_orth[key[5]] += c
        return {"schema": self.SCHEMA, "total": self.total, "n_cells": len(cells),
                "by_ctx": dict(sorted(by_ctx.items())), "by_orth": dict(sorted(by_orth.items())), "cells": cells}

    @classmethod
    def from_json(cls, data: Mapping) -> "ExposureLedger":
        if data.get("schema") != cls.SCHEMA:
            raise ValueError(f"exposure ledger schema {data.get('schema')!r} != {cls.SCHEMA!r}")
        ledger = cls()
        for cell in data["cells"]:
            ledger._counts[ledger._key(cell)] += int(cell["n"])
        if ledger.total != data["total"]:
            raise ValueError("exposure ledger total does not match its cells")
        return ledger


def _occurs_at_word_start(haystack: str, needle: str) -> bool:
    start = haystack.find(needle)
    while start >= 0:
        if start == 0 or not _LETTER_RUN.match(haystack[start - 1]):
            return True
        start = haystack.find(needle, start + 1)
    return False


# Typographic apostrophes that Turkish text uses in place of U+0027 (right single quote, modifier letter apostrophe).
_APOSTROPHE_VARIANTS = {ord("\u2019"): "'", ord("\u02bc"): "'"}


def _search_form(text: str) -> str:
    """Canonical search form for the V10 string search: NFC, typographic apostrophes -> U+0027, every whitespace run
    -> one space, Turkish-aware lower case. Canonically equivalent spellings of a name or a frame (NFD input, ’ for
    ', a line break or double space between frame words) then match instead of slipping through as false negatives.
    Letters are never folded (no i/ı merging), so distinct names stay distinct."""
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHE_VARIANTS)
    return _lower(" ".join(text.split()))


def contamination(world: World, forbidden: list[tuple[str, str, str]], texts: list[str], *,
                  include_p3_held: bool = True) -> list[str]:
    """String-search contamination test (V10). Returns violation messages (empty = clean); never quotes a text.

    * ``(name, rel, obj)`` with obj not a case name — e.g. an edit target (entity, "A", new town): a violation is
      any text in which a surface form of the name (bare, formal, informal; whole word, case-insensitive) and the
      object (whole word(s)) co-occur. Training sentences mention exactly one entity and one object, so this is
      a conservative superset of "the fact is stated".
    * ``(name, rel, case)`` — a held frame cell: a violation is any text containing the carrier of that cell for
      that name in either orthography (the frame prefix without context, which is contained in every context
      variant), starting at a word boundary, case-insensitive.
    * all ``world.p3_held`` cells (relation A) are added unless include_p3_held=False.
    * Texts and frames are compared in ``_search_form`` (NFC, typographic apostrophes as U+0027, whitespace runs as
      one space, Turkish-aware lower case), so canonically equivalent spellings are not missed.
    """
    names = set(world.names)
    fact_items: list[tuple[str, str, str]] = []
    cell_items: set[tuple[str, str, str]] = set()
    for item in forbidden:
        if len(item) != 3:
            raise ValueError(f"forbidden items are (name, rel, obj_or_case) triples, got {item!r}")
        name, rel, x = item
        if name not in names:
            raise ValueError(f"forbidden item names an entity outside the world: {name!r}")
        if rel not in RELATIONS:
            raise ValueError(f"unknown relation {rel!r}")
        if x in CASES:
            cell_items.add((name, rel, x))
        else:
            fact_items.append((name, rel, x))
    if include_p3_held:
        cell_items.update((name, "A", case) for name, case in world.p3_held)

    name_forms = {name: frozenset(_surface_forms(_lower(name)))
                  for name in {i[0] for i in fact_items} | {i[0] for i in cell_items}}
    obj_words: dict[str, tuple[str, ...]] = {}
    for _, _, obj in fact_items:
        words = tuple(_lower(w) for w in _words_of(obj))
        if not words:
            raise ValueError(f"object without letters: {obj!r}")
        obj_words[obj] = words
    watched = set().union(*name_forms.values(), *(set(w) for w in obj_words.values()))
    index: dict[str, set[int]] = defaultdict(set)
    search: dict[int, str] = {}               # _search_form of every text that contains a watched word
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"text #{i} is not a str")
        form = _search_form(text)
        for w in _words_of(form):
            if w in watched:
                index[w].add(i)
                search[i] = form

    violations: set[str] = set()
    for name, rel, obj in fact_items:
        with_name = set().union(*(index.get(f, set()) for f in name_forms[name]))
        with_obj = set.intersection(*(index.get(w, set()) for w in obj_words[obj]))
        for i in with_name & with_obj:
            violations.add(f"forbidden fact ({name}, {rel}, {obj}): entity and object co-occur in text #{i}")
    for name, rel, case in sorted(cell_items):
        frames: dict[str, list[str]] = defaultdict(list)
        for orth in ORTHOGRAPHIES:
            frames[_search_form(CARRIERS[case].format(S=subject_form(name, case, orth), NP=NP[rel]))].append(orth)
        candidates = index.get(_lower(name), set()) | index.get(_lower(informal(name, case)), set())
        for i in candidates:
            for frame, orths in frames.items():
                if _occurs_at_word_start(search[i], frame):
                    violations.add(f"held cell ({name}, {rel}, {case}): {'/'.join(orths)} frame occurs in "
                                   f"text #{i}")
    return sorted(violations)


__all__ = ["RELATIONS", "NP", "CONTEXTS", "CARRIERS", "ORTHOGRAPHIES", "CASES", "INFLECTED", "subject_form",
           "frame_prefix", "sentence", "name_char_span", "word_char_end",
           "FRAME_WORDS", "GENERATOR_VERSION", "SHARED_ENDINGS", "TOWN_ENDINGS", "TOWNS_PER_ENDING", "OCCUPATIONS",
           "DISHES", "World", "make_world", "world_from_json", "world_problems", "partition_sizes",
           "n_prefix_pairs", "n_tok_frequent", "ends_with_shared_ending", "CTX_PROBS", "P_INFORMAL",
           "trained_cells", "pass_sentences", "BOOK_DIR", "EDUCATION_DIR", "TRAIN_CHAPTERS", "HELDOUT_CHAPTERS",
           "EDUCATION_EXCLUDE_PREFIXES", "EDUCATION_EXCLUDE_FILES", "MIN_PARAGRAPH_WORDS", "prose_sources",
           "load_prose", "prose_vocabulary", "make_study_world", "prose_chunks", "ExposureLedger",
           "contamination"]
