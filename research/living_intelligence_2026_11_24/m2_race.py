# -*- coding: utf-8 -*-
"""M2: the recall race of the earlier study, run by the harmony (NOTES.md, 6.8).

    python m2_race.py --data FOLDER [--out results/m2_race.jsonl]

The race of research/living_mind_2026_11_22/t_race.py (section 40 of its notes), unchanged in what is asked:

- the same sentences: the program's own split of the documents in ``FOLDER/belgeler``, each as the mind's words
  (``Session.body``);
- the same 200 questions: sentences of at least five words drawn with ``random.Random(0)``, each cued by its first
  two words;
- the same scoring (``recall_once.Score``): exact (the rest of that sentence), valid (the rest of some sentence
  that begins with the same two words), silent, the share of the words in place, word F1.

The harmony reads the sentences once, in order, one sentence a line (its words joined by spaces). For each cue it
gets a new line beginning with the cue's two words and goes on, without learning, up to the end of the line or 60
words. Three ways:

- free: always its most probable symbol;
- under the speaking line: silent from the first symbol below e*;
- recall: the goal is to recall, so the mind locks onto the lived line that began as the cue begins and says it to
  its end (MODEL.md 6, the lock; 8, completion); with no such line it goes on freely.

t_race.py's ``transformer`` (a word-level GPT trained on the same sentences, asked at 2 to 60 minutes) and ``mind``
(Zihin) run on the same folder. Only counts and times are written: nothing of the documents and nothing said.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_22"))
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_15"))
sys.path.insert(0, HERE)
import m2_harmony  # noqa: E402
import t_race  # noqa: E402
from living.evidence import SPEAKING_LINE  # noqa: E402
from living.fast import FastCortex  # noqa: E402

MAX_NEW = t_race.MAX_NEW
CUE = t_race.CUE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="", help="changes to m2_harmony.DEFAULT, as dict(...)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m2_race.jsonl"))
    a = ap.parse_args()
    config = dict(m2_harmony.DEFAULT)
    if a.config:
        config.update(eval(a.config, {"dict": dict}))
    bodies, _ = t_race.corpus(a.data)
    chosen, _ = t_race.questions(bodies)
    stream = "".join(" ".join(b) + "\n" for b in bodies)
    mind = FastCortex(**config)
    w0, c0 = time.time(), time.process_time()
    mind.stream("\n" + stream)
    read_wall, read_cpu = time.time() - w0, time.process_time() - c0
    for way, line in (("free", 0.0), ("speaking line", SPEAKING_LINE), ("recall", 0.0)):
        _, score = t_race.questions(bodies)
        w0, c0 = time.time(), time.process_time()
        locked = 0
        for body in chosen:
            history = stream[-256:] + " ".join(body[:CUE]) + " "
            if way == "recall":
                text, lock = mind.recall(history, most=600)
                locked += lock
            else:
                text, _ = mind.say(history, most=600, line=line)
            said = text.split("\n")[0].split()[:MAX_NEW]
            score.add(body, said)
        row = {"test": "race", "who": "living intelligence (M2 harmony)", "way": way, "config": config,
               "sentences": len(bodies),
               "characters": len(stream), "learn_cpu_s": round(read_cpu, 1), "learn_wall_s": round(read_wall, 1),
               "passes": 1, "answer_cpu_s": round(time.process_time() - c0, 1),
               "answer_wall_s": round(time.time() - w0, 1), "threads": 1,
               "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}
        row.update(score.row())
        if way == "recall":
            row["locked_on_a_lived_line"] = locked
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
