# -*- coding: utf-8 -*-
"""Developmental test world for Zihin: random vocabulary per seed and per domain, a teacher, sensed events.

Nothing here is known to the core. The mind receives sequences of word symbols on one channel ("text") with a
provenance per event (told by the teacher, sensed from the world, observed in another learner); the truth
tables below are used only to generate the stream and to score the probes.

A world has entities, categories (every entity belongs to one), dimensions with values, and per category a
RULE for some dimensions (the value most members have). Members can be exceptions. Some entities are HELD OUT:
the teacher states their category and their free dimensions but never their ruled dimensions, so whatever the
mind expects there comes from the category.

Two domains with different kinds of content and different surface order:

* ``nitelik`` (attributes):  "<entity> <dimension> <value>", "<entity> <isa> <category>",
  generic rule "<category> <dimension> <value>". Values are words of their own.
* ``iliski`` (relations):    "<relation> <entity> <target>", "<isa> <entity> <category>",
  generic rule "<relation> <category> <target>". Targets are themselves entities of the world.

The vocabulary (every word, including the function word) is drawn at random per (seed, domain).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CONSONANTS, VOWELS = "bcdfghklmnprstvyz", "aeiou"

DOMAINS = {
    "nitelik": dict(order=("ent", "dim", "val"), n_ent=30, n_cat=3, n_dim=4, n_ruled=2, n_val=4,
                    relational=False),
    "iliski": dict(order=("dim", "ent", "val"), n_ent=24, n_cat=4, n_dim=3, n_ruled=2, n_val=5,
                   relational=True),
}
P_EXCEPTION = 0.15      # share of members that deviate from a category rule
HELD_OUT_PER_CAT = 2    # members whose ruled dimensions are never stated
N_LIES = 8              # facts first told wrongly; half corrected by the teacher, half by the world
N_EARLY = 10            # facts told once in the first fifth of the life and never again


def _words(rng: random.Random, n: int, used: set) -> List[str]:
    out = []
    while len(out) < n:
        syllables = rng.choice((2, 2, 3))
        word = "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(syllables))
        if rng.random() < 0.3:
            word += rng.choice(CONSONANTS)
        if word not in used:
            used.add(word)
            out.append(word)
    return out


@dataclass
class Experience:
    """One utterance or one look at the world: symbols, provenance, and what it is (for the scorer only)."""

    kind: str                       # fact | isa | rule
    symbols: List[str]
    prov: str                       # told | sensed | imitated
    ent: Optional[str] = None
    dim: Optional[str] = None
    val: Optional[str] = None
    cat: Optional[str] = None
    true: bool = True
    tag: str = ""                   # lie | correction | early | repeat | ""


@dataclass
class World:
    seed: int
    domain: str
    entities: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    dims: List[str] = field(default_factory=list)
    values: Dict[str, List[str]] = field(default_factory=dict)
    isa: str = ""
    cat_of: Dict[str, str] = field(default_factory=dict)
    rule: Dict[Tuple[str, str], str] = field(default_factory=dict)
    value: Dict[Tuple[str, str], str] = field(default_factory=dict)
    held_out: List[str] = field(default_factory=list)
    exceptions: List[Tuple[str, str]] = field(default_factory=list)
    told_rule_dim: str = ""         # the ruled dimension whose rule the teacher states; the other is never stated
    ruled: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def make(cls, seed: int, domain: str) -> "World":
        spec = DOMAINS[domain]
        rng = random.Random(f"{domain}/{seed}")
        used: set = set()
        w = cls(seed=seed, domain=domain)
        w.entities = _words(rng, spec["n_ent"], used)
        w.categories = _words(rng, spec["n_cat"], used)
        w.dims = _words(rng, spec["n_dim"], used)
        w.isa = _words(rng, 1, used)[0]
        for dim in w.dims:
            w.values[dim] = rng.sample(w.entities, spec["n_val"]) if spec["relational"] \
                else _words(rng, spec["n_val"], used)
        w.ruled = w.dims[:spec["n_ruled"]]
        w.told_rule_dim = w.ruled[0]
        members: Dict[str, List[str]] = {c: [] for c in w.categories}
        for i, ent in enumerate(w.entities):
            cat = w.categories[i % len(w.categories)]
            w.cat_of[ent] = cat
            members[cat].append(ent)
        for cat in w.categories:
            for dim in w.ruled:
                w.rule[(cat, dim)] = rng.choice(w.values[dim])
            w.held_out += rng.sample(members[cat], HELD_OUT_PER_CAT)
        for ent in w.entities:
            for dim in w.dims:
                if dim in w.ruled:
                    val = w.rule[(w.cat_of[ent], dim)]
                    if ent not in w.held_out and rng.random() < P_EXCEPTION:
                        val = rng.choice([v for v in w.values[dim] if v != val])
                        w.exceptions.append((ent, dim))
                else:
                    val = rng.choice(w.values[dim])
                w.value[(ent, dim)] = val
        return w

    # ------------------------------------------------------------------ surface forms
    def order(self) -> Tuple[str, str, str]:
        return DOMAINS[self.domain]["order"]

    def sentence(self, subject: str, middle: str, last: str) -> List[str]:
        """Surface order of (entity or category, dimension or isa, value or category)."""
        slots = {"ent": subject, "dim": middle, "val": last}
        return [slots[s] for s in self.order()]

    def prefix(self, subject: str, middle: str) -> List[str]:
        return self.sentence(subject, middle, "")[:2]

    def fact(self, ent: str, dim: str, val: Optional[str] = None, prov: str = "told", tag: str = "") -> Experience:
        truth = self.value[(ent, dim)]
        val = truth if val is None else val
        return Experience("fact", self.sentence(ent, dim, val), prov, ent=ent, dim=dim, val=val,
                          cat=self.cat_of[ent], true=(val == truth), tag=tag)

    def membership(self, ent: str, tag: str = "") -> Experience:
        cat = self.cat_of[ent]
        return Experience("isa", self.sentence(ent, self.isa, cat), "told", ent=ent, cat=cat, tag=tag)

    def generic(self, cat: str, dim: str) -> Experience:
        val = self.rule[(cat, dim)]
        return Experience("rule", self.sentence(cat, dim, val), "told", dim=dim, val=val, cat=cat)

    # ------------------------------------------------------------------ the life
    def schedule(self, n: int, seed_offset: int = 0) -> List[Experience]:
        """A life of n experiences in random order (see module docstring for what the teacher does)."""
        rng = random.Random(f"{self.domain}/{self.seed}/life/{seed_offset}")
        tellable = [(e, d) for e in self.entities for d in self.dims
                    if not (e in self.held_out and d in self.ruled)]
        plain = [f for f in tellable if f not in self.exceptions]
        special = rng.sample(plain, N_LIES + N_EARLY)
        lies, early = special[:N_LIES], special[N_LIES:]
        base: List[Experience] = [self.membership(e) for e in self.entities]
        base += [self.fact(e, d) for (e, d) in tellable if (e, d) not in special]
        base += [self.generic(c, self.told_rule_dim) for c in self.categories]
        rng.shuffle(base)
        repeatable = [x for x in base if x.kind != "rule"]
        while len(base) < n - 3 * N_LIES - N_EARLY:
            again = rng.choice(repeatable)
            if again.kind == "fact" and rng.random() < 0.4:      # the world shows what the teacher said
                again = self.fact(again.ent, again.dim, prov="sensed", tag="repeat")
            base.insert(rng.randrange(len(base) + 1), again)
        first_fifth = max(1, len(base) // 5)
        for ent, dim in early:                                    # told once, early, never again
            base.insert(rng.randrange(first_fifth), self.fact(ent, dim, tag="early"))
        for i, (ent, dim) in enumerate(lies):                     # a wrong statement, corrected later
            wrong = rng.choice([v for v in self.values[dim] if v != self.value[(ent, dim)]])
            at = rng.randrange(len(base) // 2)
            base.insert(at, self.fact(ent, dim, val=wrong, tag="lie"))
            prov = "told" if i % 2 == 0 else "sensed"
            for _ in range(2):
                later = rng.randrange(at + 1, len(base) + 1)
                base.insert(later, self.fact(ent, dim, prov=prov, tag="correction"))
        return base
