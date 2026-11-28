# -*- coding: utf-8 -*-
"""The frozen sets of speaking at length, lived and answered by the package (cevher.Cevahir).

    python speak_run.py [--set 1|2|3] [--show] [--ledger] [--note TEXT]

The life: the lessons told a sentence at a time, each text read as one text, the dialogues watched (a request and
its answer, the answer's sentences one after another). Then every request asked is answered by ``reply`` and the
answer is graded a sentence at a time against its key, as speak_kill.py grades. Rows: results/speak.jsonl.

Two changes to the telling grain (how a word ends as a mark of what a form of request asks for; a request in no
known form read through its own words) are measured on the third set, which was frozen before them. The code
before them (f05c10c) says there 60 sentences, none wrong: kinds 24 of 34 with 8 beside the question, care 0 of 4,
requests in no shown form 0 of 14. Fixed before the changed code is run on any set:

 1. no wrong sentence on any of the three sets;
 2. care, third set: at least 2 of the 4;
 3. kinds as a whole, third set: not fewer right than 24, and no more beside for each one right than 8 for 24;
 4. requests in no shown form, third set: something right is said of both, at least 4 of the 14 in all, and of what
    is said for them no more than a third is beside the question;
 5. what stood stays: one sentence and the whole text on every set, the lessons of the first set not under 74 of
    83, the row of the sentences of its own and the rows of the older batteries equal.

A change that fails is not taken, and its row is kept.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from cevher import Cevahir  # noqa: E402
from cevher.units import body, tokens  # noqa: E402

SHA = {1: ("speak.json", "3ba5745dd3db795f6bdc9528957b6b2e02d3c6d7d2747fbd7fd922e85f8c03b1"),
       2: ("speak2.json", "73d450f32294685974a4927891ba95d61f9b2b7b32536f05812b9a338069179f"),
       3: ("speak3.json", "85e7ca8f15ae78a6fcce319c4a6aa48e5f9b4a7a1fc52c41631d091c517db691")}
CLOSE = re.compile(r"(?<=[.!?…])\s+")


def live(c, Y):
    for s in Y["lessons"]:
        c.tell(s)
    for name in Y["order"]:
        c.read("\n".join(Y["texts"][name]) + "\n")
    for d in Y["shown"]:
        c.watch(d["q"], " ".join(d["must"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    name, sha = SHA[a.set]
    blob = open(os.path.join(HERE, name), "rb").read()
    assert hashlib.sha256(blob).hexdigest() == sha, "%s is not the frozen set" % name
    Y = json.loads(blob.decode("utf-8"))
    c = Cevahir(ledger=a.ledger)
    t0 = time.time()
    live(c, Y)
    life_s = time.time() - t0
    rows, seconds = [], []
    for d in Y["asked"]:
        t1 = time.perf_counter()
        said = c.reply(d["q"])
        seconds.append(time.perf_counter() - t1)
        said = [x.strip() for x in CLOSE.split(said.strip()) if x.strip()]
        must, may = set(d["must"]), set(d["may"])
        r = {"thing": d["thing"], "form": d["form"], "q": d["q"], "said": len(said),
             "right": sum(x in must for x in said), "beside": sum(x in may for x in said),
             "wrong": sum(x not in must and x not in may for x in said), "must": len(must),
             "way": c.telling.answer(body(tokens(d["q"]))[0])[0], "answer": said}
        rows.append(r)
        print("%-9s %-20s %-20s said %2d: right %2d of %2d, beside %2d, wrong %2d" % (
            r["thing"], r["form"], r["way"] or "-", r["said"], r["right"], r["must"], r["beside"], r["wrong"]))
        if a.show:
            for x in said:
                print("      %s %s" % ("+" if x in must else ("~" if x in may else "!"), x))
    said, wrong = sum(r["said"] for r in rows), sum(r["wrong"] for r in rows)
    one = [r for r in rows if r["form"] == "one sentence"]
    les = [r for r in rows if r["form"] == "what is known of it"]
    kinds = [r for r in rows if r["form"] in ("parts", "works", "made", "care", "two kinds")]
    code = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"code": code, "what": "speaking at length, by the package", "set": a.set, "ledger": a.ledger, "note": a.note,
           "life_s": round(life_s, 1), "answer_s_mean": round(sum(seconds) / len(seconds), 4),
           "answer_s_max": round(max(seconds), 4), "said": said, "wrong": wrong,
           "1_wrong_at_most_5_percent": wrong <= 0.05 * said,
           "2_one_sentence": all(r["said"] == 1 and r["right"] == 1 for r in one),
           "3_all_of_it": all(r["right"] >= 0.9 * r["must"] for r in rows if r["form"] == "all"),
           "4_lessons": [sum(r["right"] for r in les), sum(r["must"] for r in les)],
           "kinds_right_of_must": [sum(r["right"] for r in kinds), sum(r["must"] for r in kinds)],
           "kinds_beside": sum(r["beside"] for r in kinds),
           "by_kind": {k: [sum(r["right"] for r in kinds if r["form"] == k), sum(r["must"] for r in kinds if r["form"] == k),
                           sum(r["beside"] for r in kinds if r["form"] == k)] for k in ("parts", "works", "made", "care", "two kinds")},
           "no_shown_form": [(r["right"], r["must"], r["wrong"]) for r in rows if r["form"] == "in no shown form"],
           "size": c.size(), "rows": rows}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "speak.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "size")}, ensure_ascii=False))
    print("library:", out["size"]["library"], "| lived table:", out["size"]["lived"])


if __name__ == "__main__":
    main()
