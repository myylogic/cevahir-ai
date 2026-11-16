# -*- coding: utf-8 -*-
"""T-cause (exploration): does the mind expect rain from the clouds it saw, or from the umbrella words it heard?

    python research/living_mind_2026_11_15/t_cause.py [--seeds 2] [--set key=0|1,...]

The world (seeded, public, generated here). The sky is seen with the first utterance of an episode (the chat's !gor:
a sensed item). Clouds bring rain; people say "şemsiye al ." mostly when it is cloudy, so the advice and the rain have
a common cause and the advice is always followed by rain. Episodes of one person going out:
  cloudy, "şemsiye al .", "yağmur yağdı ."        x 4 a day
  cloudy, "iyi eğlenceler .", "yağmur yağdı ."    x 1 a day
  sunny,  "iyi eğlenceler .", "hava güzeldi ."    x 5 a day
for four days, shuffled. The probes (on fresh copies): the first word the mind expects of the next utterance after
[sky] "dışarı çıkıyorum ." ⏎ [advice] ⏎, read as the mind's expectations (e of "yağmur" and of "hava"). The probe of
interest is sunny with the umbrella advice, never lived: a mind led by the words expects rain, one led by what made a
difference (the sky: cloudy and sunny days with the same advice went on differently; the two advices on cloudy days
went on alike) expects the fine weather. Then the mind intervenes: it is the one who says "şemsiye al ." on four sunny
days (its own act), and no rain comes; the probes are asked again.

Appends to ``results/cause.jsonl``. An exploration: it measures what the mind does, no mechanism is tuned to it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "cause.jsonl")
DAY = [("bulutlu", "şemsiye al .", "yağmur yağdı .")] * 4 + [("bulutlu", "iyi eğlenceler .", "yağmur yağdı .")] + \
      [("güneşli", "iyi eğlenceler .", "hava güzeldi .")] * 5
PROBES = [("bulutlu", "şemsiye al ."), ("güneşli", "iyi eğlenceler ."), ("bulutlu", "iyi eğlenceler ."),
          ("güneşli", "şemsiye al .")]


def run(seed: int, switches: Dict[str, bool], days: int = 4) -> dict:
    import t_relations as R
    from zihin import ACTED, IMITATED, TOLD, Mind
    from zihin.__main__ import CHANNEL, Session
    from zihin.events import Event

    def turn(s, text, prov, speaker=None, sky=None):
        if sky is not None:
            s.view = (sky, [])
        s.listen(text, prov, speaker=speaker)
        s.end_of_turn(prov, tone=s.tone(text))

    def expect(s, sky, advice):
        twin = Session(s.mind.clone())
        twin.company(None)
        turn(twin, "dışarı çıkıyorum .", TOLD, speaker="ayşe", sky=sky)
        turn(twin, advice, TOLD)
        e = twin.mind.expected(CHANNEL)
        return {w: round(float(e.get(w, 0.5)), 3) for w in ("yağmur", "hava")}

    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    episodes = DAY * days
    rng.shuffle(episodes)
    for sky, advice, outcome in episodes:
        s.company(None)
        turn(s, "dışarı çıkıyorum .", IMITATED, speaker="ayşe", sky=sky)
        turn(s, advice, IMITATED)
        turn(s, outcome, IMITATED, speaker="ayşe")
        mind.step([])
    observed = {"%s / %s" % p: expect(s, *p) for p in PROBES}
    for _ in range(4):                                   # the mind's own umbrella advice on sunny days
        s.company(None)
        turn(s, "dışarı çıkıyorum .", TOLD, speaker="ayşe", sky="güneşli")
        for w in s.body("şemsiye al ."):
            mind.step([Event(CHANNEL, w, ACTED)])
        s.end_of_turn(ACTED, tone=".")
        turn(s, "hava güzeldi .", TOLD, speaker="ayşe")
        mind.step([])
    intervened = {"%s / %s" % p: expect(s, *p) for p in PROBES}
    return {"seed": seed, "observed": observed, "intervened": intervened}


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    rows = [run(seed, switches) for seed in range(args.seeds)]
    lines = ["== rain from the clouds or from the umbrella: %d seeds, switches %s ==" % (args.seeds, switches or "-"),
             "%-30s %-22s %s" % ("probe (sky / advice)", "observed only", "after 4 own umbrellas on sunny days")]
    for p in PROBES:
        name = "%s / %s" % p
        lines.append("%-30s %-22s %s" % (name, " ".join("%s %.3f" % kv for kv in rows[0]["observed"][name].items()),
                                          " ".join("%s %.3f" % kv for kv in rows[0]["intervened"][name].items())))
    for line in lines:
        print(line)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "switches": switches, "summary": lines, "rows": rows},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
