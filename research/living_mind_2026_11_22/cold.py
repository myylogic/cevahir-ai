# -*- coding: utf-8 -*-
"""cold.py: how much of a grown mind is still being written?

    python research/living_mind_2026_11_22/cold.py --scan 1200 [--window 13]
    python research/living_mind_2026_11_22/cold.py --load PATH --label NAME

The substance keeps, for every tie, the moment it was born and the last moment its evidence was written. This reads
those two columns of a mind at the end of a life (a SCAN add-jump life lived here, or a saved mind) and prints, of
its living ties: how many were written only in the moment they were born, how many were last written in the first
half and in the first nine tenths of the life, and the same by the form of the tie. Only counts are written (for a
saved mind: nothing of what it lived).

What this is evidence for: a tie that is no longer written still has to be read when its situation comes back, so it
cannot be thrown away; but it need not be in working memory. The share of such ties is the share of the substance
that a long-term store on disk could hold, if reading it there is cheap enough. What it is not: a measurement of how
often those ties are read (the substance does not record reads).
Appends to ``results/cold.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "cold.jsonl")
FORMS = {0: "EXPECT", 1: "PART", 2: "ABSTRACT"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=0)
    ap.add_argument("--load", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.load:
        from zihin.persist import load
        mind = load(args.load, ledger=False)
        what = "a saved mind (%s)" % (args.label or "unnamed")
    else:
        import t_relations as R
        import t_scan as S
        from zihin import Mind
        from zihin.__main__ import CHANNEL, Session
        torch.manual_seed(0)
        mind = Mind(R.speaking_config({}))
        mind.add_channel(CHANNEL, "act")
        mind.knob["F"] = args.window
        s = Session(mind)
        train, _ = S.split(0, args.scan, 1)
        for command, answer in S.life(0, train, 2):
            s.watch(" ".join(command), " ".join(answer) + " .")
        what = "SCAN add-jump, %d training dialogues watched twice" % len(train)
    v, n, now = mind.S.arrays(), mind.S.n_ties, max(1, mind.t)
    alive = v.alive[:n]
    birth, last, form = v.birth[:n][alive], v.last[:n][alive], v.form[:n][alive]
    last = np.maximum(last, birth)                         # a tie never written after its birth has last = 0
    total = int(alive.sum())

    def shares(mask) -> dict:
        m = int(mask.sum())
        if not m:
            return {"ties": 0}
        return {"ties": m,
                "written_only_at_birth": round(float((last[mask] == birth[mask]).mean()), 4),
                "last_written_in_first_half": round(float((last[mask] <= now // 2).mean()), 4),
                "last_written_in_first_nine_tenths": round(float((last[mask] <= now * 9 // 10).mean()), 4)}

    row = {"what": what, "moments": mind.t, "ties": total, "all": shares(np.ones(total, dtype=bool)),
           "by_form": {name: shares(form == f) for f, name in FORMS.items()}}
    lines = ["== cold ties: %s; %d moments, %d living ties ==" % (what, mind.t, total)]
    for name, s_ in [("all", row["all"])] + list(row["by_form"].items()):
        if s_["ties"]:
            lines.append("  %-9s %9d ties: written only at birth %.1f%%, last written in the first half of the life "
                         "%.1f%%, in the first nine tenths %.1f%%" % (
                             name, s_["ties"], 100 * s_["written_only_at_birth"],
                             100 * s_["last_written_in_first_half"], 100 * s_["last_written_in_first_nine_tenths"]))
    for line in lines:
        print(line)
    row["summary"] = lines
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
