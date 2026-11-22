# -*- coding: utf-8 -*-
"""T-want, left to itself: does the mind decide by itself when to look something up (F47)?

    python research/living_mind_2026_11_22/t_want_self.py --set open_claims=1,look=1 [--seeds 5] [--silences 20]
                                                          [--empty | --holds N] [--form nedir|nerede]

The world and the library are T-want's (t_want.py, not changed: this is a script of its own). Nobody tells the
mind to look anything up, and no command is given. The session is allowed to carry out the mind's own looks
(Session.may_look), with T-want's library as its source, and the mind is left in silence:
  before:  --silences silent moments before anyone asked anything (it should look nothing up);
  asked:   four things of the library are asked about ("X nedir ?"); it says the name and no more;
  alone:   --silences silent moments; what it looks up then, and in which moments, is its own decision;
  again:   the four questions are asked again, on copies;
  later:   --silences more silent moments (does it go on looking?).
--empty: the library holds nothing about what it wants (a look leads to nothing). --holds N: it holds only
the first N of the four things asked about. Without look=1 the mind decides nothing (the control).

Appends to results/want.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_want as W  # noqa: E402

OUT = os.path.join(HERE, "results", "want.jsonl")


def run_self(seed: int, switches: Dict[str, bool], days: int, silences: int, empty: bool, holds: int = -1) -> dict:
    """F47: nobody tells the mind to look anything up. After the questions it could not answer it is left in
    silence (``silences`` silent moments of its life); what it does then is its own decision. With ``empty`` the
    library holds nothing about what it wants (a look leads to nothing)."""
    import t_relations as R
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    from zihin.read import look_expectation, open_claim
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    known = list(W.KNOWN)
    rng.shuffle(known)
    watched, read_only = known[:8], known[8:]
    unknown = list(W.LIBRARY)
    rng.shuffle(unknown)
    asked, never = unknown[:4], unknown[4:]
    for _ in range(days):
        read = [W.fact(a) for a in known]
        rng.shuffle(read)
        dialogues = [(W.ASK % a, W.fact(a)) for a in watched]
        rng.shuffle(dialogues)
        every = max(1, len(read) // len(dialogues))
        for i, sentence in enumerate(read):
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(sentence))
            mind.step([])
            if i % every == every - 1 and dialogues:
                s.watch(*dialogues.pop())
        for d in dialogues:
            s.watch(*d)
    held = [] if empty else (asked[:holds] if holds >= 0 else list(W.LIBRARY))
    library = W.Library({a: W.fact(a) for a in held})
    s.may_look, s.look_source, s.look_gap = True, library, 0.0
    quiet = lambda line: None   # noqa: E731

    def right(said: List[str], things) -> int:
        return sum(x == W.fact(a).rstrip(" .").strip() for x, a in zip(said, things))

    for _ in range(silences):                                         # before anyone asked: nothing to look for
        s.silent_moment(say=quiet)
    before_asked = len(library.asked)
    gap = [s.reply(W.ASK % a, limit=40) for a in asked]                 # asked; it says the name and no more
    s.close_turn()
    looked_at = []
    for i in range(silences):                                         # left to itself
        n = len(library.asked)
        s.silent_moment(say=quiet)
        if len(library.asked) > n:
            looked_at.append(i)
    looks = library.asked[before_asked:]
    after = [Session(mind.clone()).reply(W.ASK % a, limit=40) for a in asked]
    n_after = len(library.asked)
    for _ in range(silences):                                         # and after: does it go on looking?
        s.silent_moment(say=quiet)
    return {"seed": seed, "asked": asked, "never": never, "looks_before_asked": before_asked, "looks": looks,
            "looked_in_silence": looked_at, "looks_of_asked": sum(x in asked for x in looks),
            "looks_of_other": sum(x not in asked for x in looks), "gap_right": right(gap, asked),
            "after_said": after, "after_right": right(after, asked),
            "met": sum(bool(r.get("met")) for r in s.looks), "looks_later": len(library.asked) - n_after,
            "still_open": sum(open_claim(mind, a) > 0.5 for a in asked),
            "look_expectation": round(look_expectation(mind, "\0 never looked up"), 3),
            "moments": mind.t}


def main_self(args, switches) -> None:
    rows = [run_self(seed, switches, args.days, args.silences, args.empty, args.holds) for seed in range(args.seeds)]
    n = 4 * len(rows)
    total = lambda key: sum(r[key] for r in rows)   # noqa: E731
    lines = ["== T-want, left to itself (%s; %s): %d seeds, %d silent moments, switches %s ==" % (
        W.ASK % "X", "the library holds nothing it wants" if args.empty else (
            "the library holds every thing" if args.holds < 0 else
            "the library holds the first %d of the 4 things asked about" % args.holds),
        len(rows), args.silences, switches or "-"),
        "  before anyone asked: %d looks" % total("looks_before_asked"),
        "  asked about things it never met: %d / %d answered" % (total("gap_right"), n),
        "  left in silence, nobody telling it to: %d looks, %d of them of the %d things it could not answer, "
        "%d of anything else" % (sum(len(r["looks"]) for r in rows), total("looks_of_asked"), n,
                                 total("looks_of_other")),
        "  of those looks, the claim was met by what it read: %d" % total("met"),
        "  asked again: %d / %d answered exactly" % (total("after_right"), n),
        "  claims still open after that: %d / %d; looks in the silence that followed: %d" % (
            total("still_open"), n, total("looks_later")),
        "  what looking up is expected to lead to, for an item never looked up (1/2 at birth): %s" % (
            ", ".join("%.3f" % r["look_expectation"] for r in rows))]
    for r in rows[:2]:
        lines.append("  seed %d: asked %s; looked up, in order %s (at silent moments %s); after %s" % (
            r["seed"], r["asked"], r["looks"], r["looked_in_silence"], r["after_said"]))
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "t_want_self", "form": W.ASK % "X", "empty_library": args.empty,
                            "library_holds": args.holds,
                            "silences": args.silences, "switches": switches, "days": args.days, "rows": rows,
                            "summary": lines}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--set", default="open_claims=1,look=1")
    ap.add_argument("--form", default="nedir", choices=["nedir", "nerede"])
    ap.add_argument("--silences", type=int, default=20, help="silent moments before, after and later")
    ap.add_argument("--empty", action="store_true", help="the library holds nothing about what it wants")
    ap.add_argument("--holds", type=int, default=-1, help="the library holds only the first N things asked about")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    W.FORM = args.form
    if args.form == "nerede":
        W.ASK = "%s nerede bulunur ?"
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    main_self(args, switches)


if __name__ == "__main__":
    main()
