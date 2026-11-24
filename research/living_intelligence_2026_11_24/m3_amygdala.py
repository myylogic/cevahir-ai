# -*- coding: utf-8 -*-
"""M3: does an outcome change what Cevahir says? The amygdala's record, measured (NOTES.md, 8).

    python m3_amygdala.py [--seeds 3]

A world built for the question. Cevahir watches dialogues about birds only ("serçe ne yapar ?" -> "serçe uçar .")
and reads facts about the other classes without any dialogue ("hamsi bir balıktır .", "balıklar yüzer ."): over the
birds, the hippocampus's analogy foresees every answer, and its record of disagreements with Zihin is what it is.
Asked of a fish, the analogy substitutes into a bird's dialogue and says "hamsi uçar"; Zihin deduces "hamsi yüzer"
(F37) if it does. Which of the two thoughts Cevahir says is decided by the record (MODEL.md 12.5).

Then a study: half of the other classes' members are asked in conversation, in three conditions: "judged" (each
answer judged with ``feedback``; a wrong one forgotten and the right one shown, as in m3_etut.py), "judged, kept"
(judged and shown, the wrong attempt not forgotten), "shown" (nothing
judged; the attempt lived as it was and the right answer shown after it) and "none" (nothing judged, nothing shown).
The other half is asked before and after the study, each by a process that is the mind as it stands. "shown" against
"judged" separates what the judgement (the amygdala's record) adds from what the answers shown teach. Counted: right
answers on the held-out half, before and after, and the records.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from living.cevahir import Cevahir  # noqa: E402

BIRDS = ["serçe", "güvercin", "karga", "kartal", "baykuş", "martı", "leylek", "turna"]
OTHER = {  # class: (copula, plural, property, members)
    "balık": ("balıktır", "balıklar", "yüzer", ["hamsi", "sazan", "palamut", "levrek", "alabalık", "istavrit"]),
    "böcek": ("böcektir", "böcekler", "vızıldar", ["arı", "sinek", "eşekarısı", "sivrisinek", "yabanarısı", "bombus"]),
    "yılan": ("yılandır", "yılanlar", "sürünür", ["engerek", "kobra", "piton", "boa", "mamba", "çıngıraklı"]),
}


def code_version() -> str:
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def life(c: Cevahir, rng: random.Random) -> None:
    facts = ["%s bir kuştur ." % b for b in BIRDS] + ["kuşlar uçar ."] * 2
    for name, (copula, plural, prop, members) in OTHER.items():
        facts += ["%s bir %s ." % (m, copula) for m in members] + ["%s %s ." % (plural, prop)] * 2
    dialogues = [("%s ne yapar ?" % b, "%s uçar ." % b) for b in BIRDS]
    for _ in range(3):
        rng.shuffle(facts)
        rng.shuffle(dialogues)
        for f in facts:
            c.tell(f)
        for q, a in dialogues:
            c.watch(q, a)
    c.session.close_turn()


def run(seed: int, judged: str) -> dict:
    """``judged``: "judged" (each answer judged; a wrong one forgotten and the right one shown), "judged, kept"
    (judged and shown, the wrong attempt not forgotten), "shown" (nothing judged, the right answer shown after every
    attempt) or "none" (nothing judged, nothing shown)."""
    import torch
    torch.set_num_threads(1)
    rng = random.Random(seed)
    c = Cevahir(variant="speaking", switches={"inherit": True}, ledger=True, seed=seed)
    life(c, rng)
    study, held = [], []
    for name, (copula, plural, prop, members) in OTHER.items():
        order = list(members)
        rng.shuffle(order)
        study += [(m, prop) for m in order[:3]]
        held += [(m, prop) for m in order[3:]]

    def ask(pairs):
        said = c.answers(["%s ne yapar ?" % m for m, _ in pairs], way="every", workers=2)
        return said, sum(c.session.body(s["cevahir"]) == [m, p] for s, (m, p) in zip(said, pairs))

    before, right_before = ask(held)
    record_before = (c.az_won, c.az_all)
    for m, p in study:
        for _ in range(2):
            c.mind.step([])
        said = c.converse("%s ne yapar ?" % m, limit=20)
        ok = c.session.body(said) == [m, p]
        if judged == "judged":
            c.feedback(ok)
            if not ok:
                c.forget(len(c.entries) - 1)
                c.session.open_turn, c.session.answered = False, None
                c.watch("%s ne yapar ?" % m, "%s %s ." % (m, p))
        elif judged == "judged, kept":                        # judged, the wrong attempt not forgotten
            c.feedback(ok)
            if not ok:
                c.watch("%s ne yapar ?" % m, "%s %s ." % (m, p))
        elif judged == "shown":
            c.flush()
            c.watch("%s ne yapar ?" % m, "%s %s ." % (m, p))
        else:
            c.flush()
    after, right_after = ask(held)
    return {"seed": seed, "judged": judged, "held_out": len(held), "right_before": right_before,
            "right_after": right_after, "record_before": record_before, "record_after": [c.az_won, c.az_all],
            "thought_before": [s["thought"] for s in before], "thought_after": [s["thought"] for s in after],
            "examples_after": [{"q": "%s ne yapar ?" % m, "cevahir": s["cevahir"], "zihin": s["zihin"],
                                "analogy": s["analogy"]} for s, (m, _) in zip(after, held)][:4]}


CODE = code_version()                                   # the code that runs, read when it starts

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_amygdala.jsonl"))
    a = ap.parse_args()
    for seed in range(a.seeds):
        for judged in ("judged", "judged, kept", "shown", "none"):
            row = run(seed, judged)
            row["test"], row["code"] = "m3_amygdala", CODE
            print(json.dumps({k: row[k] for k in ("seed", "judged", "held_out", "right_before", "right_after",
                                                  "record_before", "record_after")}), flush=True)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
