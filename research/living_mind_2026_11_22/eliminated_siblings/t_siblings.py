# -*- coding: utf-8 -*-
"""t_siblings.py: the measurement of an ELIMINATED technique (see ``siblings.py`` in this folder): copies of a
mind live different things at the same time and are taken into one. What does the one mind then know, against a mind
that lived the same things in turn, and how long did it take?

    python research/living_mind_2026_11_22/eliminated_siblings/t_siblings.py library [--things 120] [--siblings 4]
    python research/living_mind_2026_11_22/eliminated_siblings/t_siblings.py scan [--train 150] [--test 60] [--siblings 4]
    python research/living_mind_2026_11_22/eliminated_siblings/t_siblings.py docs --data FOLDER [--siblings 4] [--n 200] [--base 0.5]
    (every world: [--rounds R]: the siblings part and are absorbed R times instead of once)

Every world is lived twice from the same beginning: **in turn** (one mind, one thing after another) and **apart**
(the mind is forked into ``--siblings`` processes, each lives its share, and the mind absorbs what each learned,
``siblings.live_apart``). The same questions are then asked of both. Reported: the scores of both, how many answers
are the same, the seconds of each way (apart: the living in processes, and the absorbing), the ties of both minds.

library (generated here): a mind that read about twelve things and watched "X nedir ?" answered for eight (T-want's
    beginning) reads a library of ``--things`` new things, two sentences each ("X <adjective> bir <class> .",
    "X <place> bulunur ."). Asked: "X nedir ?" of every thing (exactly the first sentence), and the first two words
    of each sentence brought to mind (the rest of the sentence, exactly). Apart, the things are dealt to the siblings.
scan (SCAN's add-jump life of ``t_scan.py``, F43 and F45 on, window 13): the commands without a connective and
    "jump" are lived first by the one mind; the joined commands are then lived in turn, or dealt to the siblings.
    Asked: the held-out commands.
docs: the documents of a folder, sentence by sentence as ``!oku`` reads them, from a new mind; apart, the sentences
    are dealt to the siblings in blocks. Asked: recall with a two-word cue of ``--n`` sampled sentences, as
    ``recall_once.py`` scores it. Only counts are written (a private folder leaves nothing of itself).

Appends to ``results/siblings.jsonl``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "living_mind_2026_11_15"))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "siblings.jsonl")
ADJ = ["mor", "ince", "gürültülü", "yavaş", "sulu", "göçmen", "kızıl", "yumuşak", "parlak", "sert", "hafif", "eski"]
CLASS = ["çiçektir", "ağaçtır", "alettir", "taşıttır", "meyvedir", "kuştur", "madendir", "bitkidir"]
PLACE = ["vazoda", "yamaçta", "çantada", "iskelede", "tabakta", "bacada", "madende", "tarlada", "rafta", "kıyıda"]
SYL = ["ka", "to", "mi", "re", "su", "la", "ne", "ba", "di", "go", "pe", "zu", "ha", "ve", "ri", "yo"]


def new_mind(switches=None, window: int = 0):
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(0)
    mind = Mind(R.speaking_config(switches or {}))
    mind.add_channel(CHANNEL, "act")
    if window:
        mind.knob["F"] = window
    Session(mind)                                # (the session's channels exist before any copy is made)
    return mind


def read(mind, sentences: List[str]) -> None:
    """Live sentences as ``Session.read_path`` lives a document's."""
    from zihin import IMITATED
    from zihin.__main__ import Session
    s = Session(mind)
    s.company(None)
    for sentence in sentences:
        s.listen(sentence, IMITATED)
        s.end_of_turn(IMITATED, tone=s.tone(sentence))
        mind.step([])


def watch(mind, dialogues) -> None:
    from zihin.__main__ import Session
    s = Session(mind)
    for command, answer in dialogues:
        s.watch(command, answer)


def both_ways(begin, shares: List[list], work, siblings: int, rounds: int = 1):
    """(the mind that lived every share in turn, the mind that absorbed its siblings, seconds, reports). With
    ``rounds`` > 1 the siblings part and are absorbed that many times: each share is cut into ``rounds`` pieces, in
    every round each sibling lives its next piece from the mind as it then is (what the others learned in the rounds
    before is theirs too), and the mind absorbs them before the next round."""
    import siblings as kardes
    if rounds > 1:
        cut = [[share[len(share) * r // rounds:len(share) * (r + 1) // rounds] for share in shares]
               for r in range(rounds)]
    else:
        cut = [shares]
    turn = begin()
    t0 = time.perf_counter()
    for jobs in cut:
        for job in jobs:
            work(turn, job)
    in_turn = time.perf_counter() - t0
    one = begin()
    lived = absorbed = 0.0
    reports = []
    for jobs in cut:
        jobs = [job for job in jobs if job]
        t0 = time.perf_counter()
        whats = kardes_lives(one, jobs, work, siblings)
        lived += time.perf_counter() - t0
        t0 = time.perf_counter()
        reports += [kardes.absorb(one, what) for what in whats]
        absorbed += time.perf_counter() - t0
    return turn, one, {"in_turn_s": round(in_turn, 1), "apart_living_s": round(lived, 1),
                       "absorbing_s": round(absorbed, 2)}, reports


def kardes_lives(mind, shares, work, workers: int):
    """What each sibling learned (the living part of ``kardes.live_apart``, timed apart from the absorbing)."""
    import multiprocessing
    import siblings as kardes
    kardes._MIND = mind
    try:
        with multiprocessing.get_context("fork").Pool(min(workers, len(shares)), maxtasksperchild=1) as pool:
            return pool.map(kardes._child, [(share, work) for share in shares], chunksize=1)
    finally:
        kardes._MIND = None


def deal(items: list, k: int) -> List[list]:
    return [items[i::k] for i in range(k)]


# ------------------------------------------------------------------------------------------------ library
def library(things: int, seed: int = 0):
    rng = random.Random(seed)
    names = set()
    while len(names) < things:
        names.add("".join(rng.choice(SYL) for _ in range(rng.randint(2, 3))))
    return {name: ("%s %s bir %s ." % (name, rng.choice(ADJ), rng.choice(CLASS)),
                   "%s %s bulunur ." % (name, rng.choice(PLACE))) for name in sorted(names)}


def run_library(args) -> dict:
    import t_want as W
    from zihin.__main__ import Session
    from zihin.dusunce import think
    lib = library(args.things)

    def begin():
        mind = new_mind()
        for _ in range(3):
            read(mind, [W.fact(a) for a in W.KNOWN])
            watch(mind, [(W.ASK % a, W.fact(a)) for a in list(W.KNOWN)[:8]])
        return mind

    pages = [list(lib[name]) for name in lib]
    shares = [[s for page in share for s in page] for share in deal(pages, args.siblings)]
    turn, one, seconds, reports = both_ways(begin, shares, read, args.siblings, args.rounds)

    def score(mind):
        out, said = {"answer": 0, "recall_first": 0, "recall_second": 0}, []
        for name, (first, second) in lib.items():
            first, second = first.rstrip(" ."), second.rstrip(" .")
            answer = Session(mind.clone()).reply(W.ASK % name, limit=40)
            said.append(answer)
            out["answer"] += answer == first
            for key, sentence in (("recall_first", first), ("recall_second", second)):
                words = sentence.split()
                thought, _ = think(Session(mind.clone()), " ".join(words[:2]), limit=40)
                out[key] += [w for w in thought if w != "."] == words[2:]
        return out, said
    a, said_a = score(turn)
    b, said_b = score(one)
    return {"world": "library", "things": len(lib), "siblings": args.siblings, "asked": len(lib),
            "in_turn": a, "apart": b, "same_answers": sum(x == y for x, y in zip(said_a, said_b)),
            "seconds": seconds, "ties": (turn.S.n_ties, one.S.n_ties), "moments": (turn.t, one.t),
            "frames_induced_at_absorbing": sum(r["frames_induced"] for r in reports)}


# ------------------------------------------------------------------------------------------------ scan
def watch_scan(mind, dialogues) -> None:
    from zihin.__main__ import Session
    s = Session(mind)
    for command, answer in dialogues:
        s.watch(" ".join(command), " ".join(answer) + " .")


def run_scan(args) -> dict:
    import t_scan as S
    from zihin.__main__ import Session
    train, test = S.split(0, args.train, args.test)
    life = S.life(0, train, 2)
    joined = [d for d in life if "and" in d[0] or "after" in d[0]]
    first = [d for d in life if not ("and" in d[0] or "after" in d[0])]

    def begin():
        mind = new_mind({"wholes": True, "told_apart": True}, window=13)
        watch_scan(mind, first)
        return mind

    shares = deal(joined, args.siblings)
    turn, one, seconds, reports = both_ways(begin, shares, watch_scan, args.siblings, args.rounds)

    def score(mind):
        said = [Session(mind.clone()).reply(" ".join(c), limit=400) for c, _ in test]
        return sum(x == " ".join(a) for x, (_, a) in zip(said, test)), said
    a, said_a = score(turn)
    b, said_b = score(one)
    return {"world": "scan add-jump (wholes, told apart, window 13)", "watched_first": len(first),
            "joined_dealt": len(joined), "siblings": args.siblings, "asked": len(test), "in_turn": a, "apart": b,
            "same_answers": sum(x == y for x, y in zip(said_a, said_b)), "seconds": seconds,
            "ties": (turn.S.n_ties, one.S.n_ties), "moments": (turn.t, one.t),
            "frames_induced_at_absorbing": sum(r["frames_induced"] for r in reports)}


# ------------------------------------------------------------------------------------------------ docs
def run_docs(args) -> dict:
    import recall_once as R
    from zihin.__main__ import Session, read_document, sentences
    from zihin.dusunce import think
    folder = os.path.join(args.data, "belgeler") if os.path.isdir(os.path.join(args.data, "belgeler")) else args.data
    files = sorted(os.path.join(root, name) for root, _, names in os.walk(folder) for name in names)
    every = []
    for name in files:
        try:
            every += sentences(read_document(name), "tr")
        except Exception:
            continue
    first, every = every[:int(len(every) * args.base)], every[int(len(every) * args.base):]
    block = max(1, len(every) // (args.siblings * args.blocks))
    blocks = [every[i:i + block] for i in range(0, len(every), block)]
    shares = [[s for b in share for s in b] for share in deal(blocks, args.siblings)]

    def begin():                                 # --base: the first part of the documents is read by the one mind
        mind = new_mind()                        # before anything is dealt (the siblings then share a beginning)
        if first:
            read(mind, first)
        return mind
    turn, one, seconds, reports = both_ways(begin, shares, read, args.siblings, args.rounds)
    if args.keep:                                # (kept outside the repository: a mind of private documents)
        from zihin.persist import save
        save(turn, args.keep + "_turn")
        save(one, args.keep + "_apart")

    def score(mind):
        session = Session(mind)
        sample, all_of = R.documents(args.data, 2, session)
        sc, said = R.Score(sample, all_of, 2), []
        for body in random.Random(0).sample(sample, min(args.n, len(sample))):
            thought, _ = think(session, " ".join(body[:2]), limit=60)
            sc.add(body, thought)
            said.append(tuple(thought))
        row = sc.row()
        return {k: row[k] for k in ("exact", "valid", "silent", "n")}, said
    with contextlib.redirect_stdout(io.StringIO()):
        a, said_a = score(turn)
        b, said_b = score(one)
    return {"world": "documents read once (%s)%s" % (args.label or "unnamed", (
        "; the first %d%% read by the one mind before" % round(100 * args.base)) if args.base else ""),
            "base": args.base, "sentences_first": len(first), "sentences": len(every),
            "siblings": args.siblings, "asked": a["n"], "in_turn": a, "apart": b,
            "same_answers": sum(x == y for x, y in zip(said_a, said_b)), "seconds": seconds,
            "ties": (turn.S.n_ties, one.S.n_ties), "moments": (turn.t, one.t),
            "frames_induced_at_absorbing": sum(r["frames_induced"] for r in reports)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("world", choices=["library", "scan", "docs"])
    ap.add_argument("--siblings", type=int, default=4)
    ap.add_argument("--things", type=int, default=120)
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--data", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--blocks", type=int, default=8, help="docs: blocks of sentences dealt to each sibling")
    ap.add_argument("--base", type=float, default=0.0, help="docs: this share is read by the one mind first")
    ap.add_argument("--rounds", type=int, default=1, help="the siblings part and are absorbed this many times")
    ap.add_argument("--keep", default="", help="docs: save the two minds here (outside the repository)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    row = {"library": run_library, "scan": run_scan, "docs": run_docs}[args.world](args)
    s = row["seconds"]
    row["rounds"] = args.rounds
    lines = ["== siblings: %s; %d siblings, %d round%s ==" % (row["world"], row["siblings"], args.rounds,
                                                             "" if args.rounds == 1 else "s"),
             "  lived in turn:  %s of %d asked; %d moments, %d ties; %.1f s" % (
                 row["in_turn"], row["asked"], row["moments"][0], row["ties"][0], s["in_turn_s"]),
             "  lived apart:    %s of %d asked; %d moments, %d ties; %.1f s of living in %d processes + %.2f s of "
             "absorbing (%.1fx the speed of living in turn)" % (
                 row["apart"], row["asked"], row["moments"][1], row["ties"][1], s["apart_living_s"],
                 row["siblings"], s["absorbing_s"], s["in_turn_s"] / max(1e-9, s["apart_living_s"] + s["absorbing_s"])),
             "  the same answer from both minds: %d of %d; frames induced while absorbing: %d" % (
                 row["same_answers"], row["asked"], row["frames_induced_at_absorbing"])]
    for line in lines:
        print(line, flush=True)
    row["summary"] = lines
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
