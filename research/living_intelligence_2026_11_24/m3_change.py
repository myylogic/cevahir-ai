# -*- coding: utf-8 -*-
"""M3: facts that change. Where someone is, told and told again, and asked after it changed (NOTES.md, 8.30).

    python m3_change.py [--seeds 3] [--variant speaking] [--set newest=1] [--workers 2] [--out ...]

A world of people and places. The people of one half are each told to be somewhere ("Veli evde .") and asked where
they are, the answer watched ("Veli nerede ?", "Veli evde ."), so that the question and its answer's form are
lived; they move between the rounds and are told and asked again. The people of the other half are only told,
never asked while Cevahir lives, in four kinds:

- ``unchanged``: told once to be somewhere, and nothing more (recall after one reading);
- ``changed once``: told once to be somewhere, then once to be elsewhere;
- ``changed after three``: told three times to be in one place, then once to be elsewhere;
- ``changed twice after three``: told three times to be in one place, then twice to be elsewhere.

The tellings of all of them are interleaved with the other half's life, in a seeded order, each person's tellings
in their own order. Then every person of the second half is asked "X nerede ?" by a process that is the mind as it
stands, in every way (zihin, harmony, the hippocampus's analogy, cevahir). Right: the person and the place told
last (the closing mark apart). Only counts and a few examples are written.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from living.cevahir import Cevahir  # noqa: E402

PEOPLE = ["Veli", "Ayşe", "Can", "Ece", "Deniz", "Selin", "Kerem", "Elif", "Mert", "Zeynep", "Arda", "Defne",
          "Emre", "İpek", "Onur", "Sena", "Tolga", "Yasemin", "Burak", "Ceren", "Doruk", "Esra", "Kaan", "Leyla"]
PLACES = ["evde", "okulda", "parkta", "bahçede", "çarşıda", "limanda"]
KINDS = ["unchanged", "changed once", "changed after three", "changed twice after three"]


def code_version() -> str:
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


CODE = code_version()


def world(seed: int):
    """The life (a list of ("tell", sentence) and ("watch", question, answer)) and the people asked with the kind
    of each and the place told last."""
    rng = random.Random(seed)
    people = list(PEOPLE)
    rng.shuffle(people)
    lived, asked = people[:12], people[12:]
    streams = []
    for name in lived:                               # three rounds: told, asked and answered, then elsewhere
        steps = []
        for _ in range(3):
            place = rng.choice(PLACES)
            steps += [("tell", "%s %s ." % (name, place)), ("watch", "%s nerede ?" % name, "%s %s ." % (name, place))]
        streams.append(steps)
    questions = []
    for i, name in enumerate(asked):
        kind = KINDS[i % len(KINDS)]
        old, new = rng.sample(PLACES, 2)
        tellings = {"unchanged": [old], "changed once": [old, new], "changed after three": [old] * 3 + [new],
                    "changed twice after three": [old] * 3 + [new] * 2}[kind]
        streams.append([("tell", "%s %s ." % (name, p)) for p in tellings])
        questions.append((name, kind, tellings[-1], old))
    life = []
    while any(streams):                              # each stream in its own order, the streams interleaved
        s = rng.choice([s for s in streams if s])
        life.append(s.pop(0))
    return life, questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variant", default="speaking")
    ap.add_argument("--set", default="", help="switches of Zihin, e.g. newest=1 (F76)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_change.jsonl"))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    for seed in range(a.seeds):
        life, questions = world(seed)
        switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, a.set.split(",")))}
        c = Cevahir(variant=a.variant, seed=seed, switches=switches)
        t0 = time.time()
        for item in life:
            if item[0] == "tell":
                c.tell(item[1])
            else:
                c.watch(item[1], item[2])
        c.session.close_turn()
        life_s = time.time() - t0
        said = c.answers(["%s nerede ?" % name for name, _, _, _ in questions], way="every", workers=a.workers)
        scores = {k: {w: Counter() for w in ("zihin", "harmony", "analogy", "cevahir")} for k in KINDS}
        examples = []
        for (name, kind, last, old), s in zip(questions, said):
            for way in ("zihin", "harmony", "analogy", "cevahir"):
                words = [w for w in c.session.words(s.get(way, "") or "") if w not in (".", "!", "?")]
                if not words or s.get(way) == "(sessiz)":
                    verdict = "silent"
                elif words == [c.session.lower(name), last]:
                    verdict = "right"
                elif words == [c.session.lower(name), old]:
                    verdict = "old"
                else:
                    verdict = "wrong"
                scores[kind][way][verdict] += 1
            if len(examples) < 8:
                examples.append({"q": "%s nerede ?" % name, "kind": kind, "last": last, "old": old,
                                 **{w: s.get(w) for w in ("zihin", "analogy", "cevahir")},
                                 "thought": s.get("thought")})
        row = {"test": "m3_change", "code": CODE, "seed": seed, "variant": a.variant, "switches": switches,
               "life_items": len(life),
               "life_s": round(life_s, 1), "za": [c.za_won, c.za_all],
               "scores": {k: {w: dict(v) for w, v in ways.items()} for k, ways in scores.items()},
               "examples": examples}
        print(json.dumps({k: row[k] for k in ("seed", "za", "scores")}, ensure_ascii=False), flush=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
