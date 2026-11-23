# -*- coding: utf-8 -*-
"""t_documents.py: one configuration raised from nothing on the first sentences of a folder of documents, then
asked to go on from the first two words of sentences it read (section 32).

    python research/living_mind_2026_11_22/t_documents.py --data FOLDER [--sentences 2000] [--n 100]
                                                          [--variant full|speaking] [--off silent_claims,relied_on]

``FOLDER``: the folder whose ``belgeler`` holds the documents (``recall_once.documents`` reads the same files).
The first ``--sentences`` sentences are listened to as the program reads a file (``Session.read_path``'s steps),
by a new mind of ``--variant`` with the switches of ``--off`` turned off. Then ``--n`` of those sentences (seed 0,
at least five words) are cued by their first two words and the mind's thought (``dusunce.think``) is scored as
``recall_once.py`` scores it: exact, valid (some sentence read that begins so goes on so), silent, the share of
the words in place, word F1.

Printed and appended to ``results/documents.jsonl``: moments, ties, the processor seconds of reading and of
answering, milliseconds a moment in each fifth of the reading, in what share of the moments the full reading was
made (``read.members_map``) and what one cost, and the scores. Nothing of the documents and nothing the mind said
is printed or kept.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "documents.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sentences", type=int, default=2000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--off", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    import recall_once as RO
    import zihin.__main__ as chat
    from zihin import IMITATED, Mind
    from zihin import read as R
    from zihin.__main__ import CHANNEL, Session, read_document, sentences
    from zihin.dusunce import think
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS[args.variant]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    off = [name for name in args.off.split(",") if name]
    for name in off:
        if not hasattr(cfg, name):
            raise SystemExit("no such switch: %s" % name)
        setattr(cfg, name, False)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    folder = os.path.join(args.data, "belgeler")
    lines = []
    for path in sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs):
        try:
            lines += list(sentences(read_document(path), session.dil))
        except Exception:                                  # noqa: BLE001 (not a text file)
            continue
        if len(lines) >= args.sentences:
            break
    lines = lines[:args.sentences]
    readings = [0, 0.0]
    whole = R.members_map

    def counted(*a, **kw):
        readings[0] += 1
        t = time.process_time()
        try:
            return whole(*a, **kw)
        finally:
            readings[1] += time.process_time() - t
    R.members_map = counted
    session.company(None)
    fifths, t0, c0, m0 = [], time.process_time(), time.process_time(), 0
    for i, sentence in enumerate(lines):
        session.listen(sentence, IMITATED)
        session.end_of_turn(IMITATED, tone=session.tone(sentence))
        mind.step([])
        if (i + 1) % max(1, len(lines) // 5) == 0 and len(fifths) < 5:
            moments = mind.t - m0
            fifths.append({"moments": mind.t, "ms_a_moment": round(1000.0 * (time.process_time() - c0) / moments, 1),
                           "full_reading_share": round(readings[0] / moments, 2),
                           "ms_a_full_reading": round(1000.0 * readings[1] / max(1, readings[0]), 1)})
            print("  read %5d sentences, %6d moments: %s" % (i + 1, mind.t, fifths[-1]), flush=True)
            c0, m0, readings[0], readings[1] = time.process_time(), mind.t, 0, 0.0
    read_s = time.process_time() - t0
    R.members_map = whole
    bodies = [session.body(sentence) for sentence in lines]
    sample = [b for b in bodies if len(b) >= 5]
    every = [b for b in bodies if len(b) > 2]
    score = RO.Score(sample, every, 2)
    t0 = time.process_time()
    for body in random.Random(0).sample(sample, min(args.n, len(sample))):
        thought, _ = think(session, " ".join(body[:2]), limit=60)
        score.add(body, thought)
    row = {"test": "documents_part", "variant": args.variant, "off": off, "sentences": len(lines),
           "moments": fifths[-1]["moments"] if fifths else mind.t, "ties": mind.S.n_ties,
           "read_cpu_s": round(read_s, 1), "answer_cpu_s": round(time.process_time() - t0, 1), "fifths": fifths,
           "note": args.note}
    row.update(score.row())
    print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
