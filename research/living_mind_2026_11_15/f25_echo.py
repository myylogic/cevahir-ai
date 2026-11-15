# -*- coding: utf-8 -*-
"""F25, the echo toy: X followed by Y three times, then "X Z" heard once. What does the mind expect after X, and what
does its inner voice (``zihin.dusunce.think``) say after X, with the setting ``born_failures`` off and on (and the
reading ``rivals``, design (b))?

    python research/living_mind_2026_11_15/f25_echo.py

Two toys, each in a fresh mind with the speaking configuration of ``zihin.__main__`` (no ledger file) and with the
"members" configuration of the first lives:

* plain: "kedi uyur" told three times, then "kedi koşar" told once;
* question: "kedi süt içer ." told three times, then the question "kedi ne yer ?" told once (the echo seen in
  conversation: a question heard once puts its own word after the topic).

Printed and written to ``results/f25_echo.jsonl``: the trust-weighted evidence (W+, W-) and expectation of X -> Y
and X -> Z read from X's own ties, and the inner voice after X (its words and its reality score).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402

from zihin import IMITATED, TOLD, Config, Mind  # noqa: E402
from zihin.__main__ import CHANNEL, Session  # noqa: E402
from zihin.dusunce import think  # noqa: E402
from zihin.evidence import expectation  # noqa: E402
from zihin.read import own_mass  # noqa: E402

SPEAKING = dict(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14, backoff=True,
                resemblance=True, race=True, situations=True, explore=True)
TOYS = {"plain": (["kedi uyur"] * 3, "kedi koşar", "kedi", "uyur", "koşar"),
        "question": (["kedi süt içer ."] * 3, "kedi ne yer ?", "kedi", "süt", "ne"),
        "read": (["kedi uyur"] * 3, "kedi koşar", "kedi", "uyur", "koşar")}   # read (imitated), not told
SETTINGS = {"off": {}, "a": {"born_failures": True}, "b": {"rivals": True}, "ab": {"born_failures": True, "rivals": True}}


def run(toy: str, config: str, setting: str) -> dict:
    lived, once, x, y, z = TOYS[toy]
    over = dict(SETTINGS[setting])
    cfg = Config.changed(**dict(SPEAKING, **over)) if config == "speaking" else Config.changed(support="members",
                                                                                                **over)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for text in lived + [once]:
        if toy == "read":                                  # as a document is read: watched, then a silent moment
            session.listen(text, IMITATED)
            mind.step([])
        else:
            session.live(text, TOLD)
    unit = mind.unit_of(CHANNEL, x)
    mass = own_mass(mind, unit)
    out = {"toy": toy, "config": config, "setting": setting}
    for name, word in (("y", y), ("z", z)):
        p, n = mass.get(mind.unit_of(CHANNEL, word), (0.0, 0.0))
        out[name] = word
        out[name + "_pos"], out[name + "_neg"], out[name + "_e"] = round(p, 4), round(n, 4), round(expectation(p, n), 4)
    mind.step([])
    thought, reality = think(session, x)
    out["thought"] = " ".join(thought)
    out["reality"] = round(reality, 3)
    return out


def main() -> None:
    torch.set_num_threads(1)
    rows = []
    for toy in TOYS:
        for config in ("speaking", "members"):
            for setting in SETTINGS:
                row = run(toy, config, setting)
                rows.append(row)
                print("%-8s %-8s %-3s  e(%s->%s) %.3f [W+ %.2f W- %.2f]  e(%s->%s) %.3f [W+ %.2f W- %.2f]  thinks: "
                      "%s ... %s (reality %.2f)" % (
                          toy, config, setting, TOYS[toy][2], row["y"], row["y_e"], row["y_pos"], row["y_neg"],
                          TOYS[toy][2], row["z"], row["z_e"], row["z_pos"], row["z_neg"], TOYS[toy][2],
                          row["thought"] or "(nothing)", row["reality"]), flush=True)
    path = os.path.join(HERE, "results", "f25_echo.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
