# -*- coding: utf-8 -*-
"""Sentences of the mind's own: the frozen set of inherit_make.py, lived and answered by the package.

    python inherit_run.py [--show]

The way measured (cevher/telling.py, "what is told of what it is"): a sentence that names a thing holds other
words; of some of them told sentences begin; the thing is put in the place of their first word. Which word of a
sentence about a thing says what the thing is, is not given: for every form of request the substance keeps a
record of each place a word may stand in (the word before it, the word after), judged by the dialogues watched.
The sentences it says were never told.

Fixed before the first run (28 November 2026). Killed if any fails:
  1. of the kinds no member of which was ever asked, at least 80 % of the ``must`` sentences are said;
  2. wrong sentences (in no key: a thing put where it does not belong) are at most 5 % of the sentences said;
  3. of the things nothing of the sort was told of, nothing is said.
Reported, not a criterion: other members of the kinds that were asked; a kind whose bare name was never told;
the member told to do otherwise than its kind (what is said of it against what it was told is counted).
"""
import argparse, hashlib, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from cevher import Cevahir  # noqa: E402

SHA = "adc240b720740eafae260391b124ab5f451e66f56d74e0cd60f19879c8726be8"
CLOSE = re.compile(r"(?<=[.!?…])\s+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    blob = open(os.path.join(HERE, "inherit.json"), "rb").read()
    assert hashlib.sha256(blob).hexdigest() == SHA, "inherit.json is not the frozen set"
    Y = json.loads(blob.decode("utf-8"))
    c = Cevahir(ledger=True)
    t0 = time.time()
    for s in Y["lessons"] + Y["world"]:
        c.tell(s)
    for d in Y["shown"]:
        c.watch(d["q"], " ".join(d["must"]))
    life_s = time.time() - t0
    rows = []
    for d in Y["asked"]:
        said = [x.strip() for x in CLOSE.split(c.reply(d["q"]).strip()) if x.strip()]
        must, may, no = set(d["must"]), set(d["may"]), set(d["not"])
        r = {"thing": d["thing"], "form": d["form"], "said": len(said), "right": sum(x in must for x in said),
             "beside": sum(x in may and x not in must for x in said), "against": sum(x in no for x in said),
             "wrong": sum(x not in must and x not in may and x not in no for x in said), "must": len(must),
             "answer": said}
        rows.append(r)
        print("%-11s %-34s said %d: right %d of %d, beside %d, against what it was told %d, wrong %d" % (
            r["thing"], r["form"], r["said"], r["right"], r["must"], r["beside"], r["against"], r["wrong"]))
        if a.show:
            for x in said:
                print("      %s %s" % ("+" if x in must else ("~" if x in may else ("x" if x in no else "!")), x))
    by = {}
    for r in rows:
        t = by.setdefault(r["form"], {"asked": 0, "said": 0, "right": 0, "must": 0, "beside": 0, "against": 0, "wrong": 0})
        t["asked"] += 1
        for k in ("said", "right", "must", "beside", "against", "wrong"):
            t[k] += r[k]
    said, wrong = sum(r["said"] for r in rows), sum(r["wrong"] for r in rows)
    strict, none = by["of a kind never asked"], by["nothing of the sort was told"]
    code = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"code": code, "what": "sentences of the mind's own, by the package", "life_s": round(life_s, 1), "said": said,
           "wrong": wrong, "1_kinds_never_asked": [strict["right"], strict["must"], strict["right"] >= 0.8 * strict["must"]],
           "2_wrong_at_most_5_percent": wrong <= 0.05 * said, "3_silent_where_nothing_was_told": none["said"] == 0,
           "by_form": by, "rows": rows}
    out["survives"] = bool(out["1_kinds_never_asked"][2] and out["2_wrong_at_most_5_percent"] and out["3_silent_where_nothing_was_told"])
    with open(os.path.join(HERE, "results", "inherit.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
