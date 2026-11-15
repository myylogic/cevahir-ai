# -*- coding: utf-8 -*-
"""F25 on held-out teacher questions: a fresh speaking mind watches the teacher's question/answer pairs, then answers
held-out questions, with the setting ``born_failures`` off and on.

    python research/living_mind_2026_11_15/f25_teacher.py --data PRIVATE_DIR --settings off a --questions 200

PRIVATE_DIR holds ``ogretmen.tsv`` (watched) and ``ogretmen.tsv.heldout`` (asked), tab separated, private and never
copied into the repository. Only aggregates and per-question scores are printed and written
(``results/f25_teacher.jsonl``); no question, answer or word is.

Protocol, the same for every setting: the speaking configuration of ``zihin.__main__`` (VARIANTS["speaking"] without
its ledger file; the ledger is not kept, which changes nothing the mind lives or says), no word pieces; the mind
watches every pair of ``ogretmen.tsv`` (``Session.read_path``: the question by someone, the answer in the role the
mind takes, imitated), then the lessons about itself once (``lessons_about_itself``, so that saying it does not know
is something it has lived), then answers the held-out questions in a fixed random order (seed 0, the first N) with
``Session.reply`` as in conversation (thinking first, curiosity on). It keeps living while it answers, so each setting
meets the same questions in the same order. Scores per answer: exact (its words equal the reference's words, letters
only), token F1 (sets of words), starts with "bilmiyorum", silent, asks back (ends in "?" without saying it does not
know).

``--documents DIR`` also reads the documents the questions were made from (private) after the lessons, as the morning
mind did. ``turn_end`` is a white box at the end of each question's turn, read only: the best act expectation, how
often one or two acts stand above the line at which an act would be said alone (1/2 + theta / 2T), the number of act
candidates, how often not knowing was kept (L7), and how often the mind spoke at once.

    python research/living_mind_2026_11_15/f25_teacher.py --compare
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402

from zihin import Config, Mind  # noqa: E402
from zihin.__main__ import CHANNEL, Session, words  # noqa: E402

SPEAKING = dict(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14, backoff=True,
                resemblance=True, race=True, situations=True, explore=True)
SETTINGS = {"off": {}, "a": {"born_failures": True}, "b": {"rivals": True}, "ab": {"born_failures": True, "rivals": True}}


def score(out: str, reference: str) -> dict:
    ref = [w for w in words(reference) if w.isalpha()]
    got = [w for w in out.split() if w.isalpha()]
    common = len(set(ref) & set(got))
    f1 = 0.0
    if ref and got and common:
        p, r = common / len(set(got)), common / len(set(ref))
        f1 = 2 * p * r / (p + r)
    unknown = got[:1] == ["bilmiyorum"]
    return {"exact": float(" ".join(got) == " ".join(ref)), "f1": f1, "bilmiyorum": float(unknown),
            "silent": float(out == "(sessiz)"), "asked": float(out.endswith("?") and not unknown),
            "length": len(out.split()) if out != "(sessiz)" else 0}


def run(data: str, setting: str, questions: int, documents: str = None) -> dict:
    torch.set_num_threads(1)
    mind = Mind(Config.changed(keep_log=False, **dict(SPEAKING, **SETTINGS[setting])))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    t0 = time.time()
    session.read_path(os.path.join(data, "ogretmen.tsv"))
    session.lessons_about_itself(1)
    if documents:                                      # the documents the questions were made from (read, imitated)
        session.read_path(documents)
    watched_moments, watched_s = mind.t, time.time() - t0
    with open(os.path.join(data, "ogretmen.tsv.heldout"), encoding="utf-8") as f:
        pairs = [line.rstrip("\n").split("\t", 1) for line in f if "\t" in line]
    random.Random(0).shuffle(pairs)
    pairs = pairs[:questions]
    turn_ends = []                                     # white box at the end of each question's turn (reads only)
    end_of_turn = session.end_of_turn

    def watched_end_of_turn(prov, listening=True, tone=None):
        acts = end_of_turn(prov, listening, tone)
        if not listening:
            act_e = [e for u, e in mind.predicted.items()
                     if int(mind.S.u_channel[u]) >= 0 and mind.ch.kind(int(mind.S.u_channel[u])) == "act"]
            alone = 0.5 + mind.knob["theta"] / (2.0 * mind.cfg.T)   # said at once were it alone (L7, F19)
            turn_ends.append((max(act_e) if act_e else 0.5, sum(e >= alone for e in act_e), len(act_e),
                              float(mind.unknown) > 0.0, bool(acts)))
        return acts

    session.end_of_turn = watched_end_of_turn
    per = []
    t1 = time.time()
    for question, answer in pairs:
        per.append(score(session.reply(question), answer))
    n = len(per)
    out = {"setting": setting, "documents": bool(documents), "questions": n, "watched_moments": watched_moments,
           "watch_s": round(watched_s, 1),
           "answer_s": round(time.time() - t1, 1), "moments": mind.t, "ties": mind.S.n_ties,
           "trust": [round(x, 4) for x in mind.trust.tolist()]}
    for key in ("exact", "f1", "bilmiyorum", "silent", "asked", "length"):
        out[key] = sum(p[key] for p in per) / n
    k = max(1, len(turn_ends))
    out["turn_end"] = {"best_e": sum(t[0] for t in turn_ends) / k, "one_above_alone": sum(t[1] >= 1 for t in turn_ends) / k,
                       "two_above_alone": sum(t[1] >= 2 for t in turn_ends) / k,
                       "candidates": sum(t[2] for t in turn_ends) / k, "not_knowing_kept": sum(t[3] for t in turn_ends) / k,
                       "said_at_once": sum(t[4] for t in turn_ends) / k}
    out["per_question"] = [[p["exact"], round(p["f1"], 4), p["bilmiyorum"], p["silent"], p["asked"], p["length"]]
                           for p in per]
    return out


def compare(path: str) -> None:
    rows = {(r["setting"], r.get("documents", False)): r for r in (json.loads(line) for line in open(path,
                                                                                                    encoding="utf-8"))}
    names = ("exact", "f1", "bilmiyorum", "silent", "asked", "length")
    for docs in (False, True):
        group = {s: r for (s, d), r in rows.items() if d == docs}
        if not group:
            continue
        print("teacher pairs and lessons%s:" % (" and the documents" if docs else ""))
        for setting, r in group.items():
            other = sum(1 for p in r["per_question"] if not (p[2] or p[3] or p[4]))
            print("  %-3s n=%d exact %.3f F1 %.3f bilmiyorum %.3f silent %.3f asked %.3f other %d length %.1f  "
                  "trust %s  (%d moments lived before, %.0f s; answering %.0f s)" % (
                      setting, r["questions"], r["exact"], r["f1"], r["bilmiyorum"], r["silent"], r["asked"], other,
                      r["length"], r["trust"], r["watched_moments"], r["watch_s"], r["answer_s"]))
            if "turn_end" in r:
                print("      at the end of the turn: %s" % ", ".join("%s %.3f" % kv for kv in r["turn_end"].items()))
        if "off" not in group:
            continue
        for setting, r in group.items():
            if setting == "off":
                continue
            for i, name in enumerate(names):
                d = [a[i] - b[i] for a, b in zip(r["per_question"], group["off"]["per_question"])]
                mu = sum(d) / len(d)
                se = (sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
                print("    %s - off, %-10s %+.3f +- %.3f  (questions higher / equal / lower: %d / %d / %d)" % (
                    setting, name, mu, se, sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, default="private_data")
    ap.add_argument("--settings", nargs="+", default=["off", "a"])
    ap.add_argument("--questions", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "f25_teacher.jsonl"))
    ap.add_argument("--documents", default=None, help="a folder of documents read after the lessons (private)")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(args.out)
        return
    for setting in args.settings:
        row = run(args.data, setting, args.questions, args.documents)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print({k: v for k, v in row.items() if k != "per_question"}, flush=True)


if __name__ == "__main__":
    main()
