# -*- coding: utf-8 -*-
"""The conversations of talk.json held with the package (cevher.Cevahir.converse), graded a turn at a time.

    python talk_run.py [--show] [--note TEXT]

Every conversation is held with a mind that has lived the lesson and nothing else. Rows: results/talk.jsonl.
"""
import argparse, hashlib, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from cevher import Cevahir  # noqa: E402


def norm(s: str) -> str:
    return " ".join(s.lower().replace("?", " ").replace(".", " ").split())


def grade(said: str, expect: dict, dont: str) -> str:
    said = norm(said)
    if expect["type"] == "silent":
        return "kept silent" if not said else "wrong"
    if not said:
        return "silent"
    if expect["type"] == "unknown":
        return "right" if said == norm(dont) else "wrong"
    return "right" if said in {norm(a) for a in expect["answers"]} else "wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    blob = open(os.path.join(HERE, "talk.json"), "rb").read()
    assert hashlib.sha256(blob).hexdigest() == open(os.path.join(HERE, "talk.sha256")).read().split()[0]
    Y = json.loads(blob.decode("utf-8"))
    code = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", ":(glob)cevher/**/*.py"]).returncode:
        code += " and changes not committed"
    kinds, rows, t0 = {}, [], time.time()
    for talk in Y["talks"]:
        c = Cevahir(ledger=False)
        for item in Y["lesson"]:
            if item[0] == "tell":
                c.tell(item[1])
            else:
                c.watch(item[1], item[2])
        for turn in talk["turns"]:
            said = c.converse(turn["said"])
            g = grade(said, turn["expect"], Y["dont_know"])
            k = kinds.setdefault(talk["kind"], {"asked": [0, 0, 0], "told": [0, 0], "never told": [0, 0, 0]})
            if turn["expect"]["type"] == "silent":
                k["told"][g == "wrong"] += 1                  # [kept silent, said something]
            else:
                k["asked" if turn["expect"]["type"] == "say" else "never told"][("right", "silent", "wrong").index(g)] += 1
            rows.append({"kind": talk["kind"], "said": turn["said"], "expect": turn["expect"], "answer": said, "grade": g})
            if a.show and g in ("wrong", "silent"):
                print("  %-34s %-7s %-52s -> %s" % (talk["kind"], g, turn["said"][:52], said[:70] or "(silent)"))
    total = {"asked": [0, 0, 0], "told": [0, 0], "never told": [0, 0, 0]}
    for k in kinds.values():
        for name, v in k.items():
            total[name] = [x + y for x, y in zip(total[name], v)]
    wrong = total["asked"][2] + total["told"][1] + total["never told"][2]
    out = {"code": code, "what": "free conversation, by the package", "note": a.note, "turns": len(rows),
           "asked_right_silent_wrong": total["asked"], "told_kept_silent_or_said_something": total["told"],
           "never_told_right_silent_wrong": total["never told"], "wrong_in_all": wrong,
           "seconds": round(time.time() - t0, 1), "by_kind": kinds, "rows": rows}
    for kind, k in kinds.items():
        print("%-36s asked %s  told %s  never told %s" % (kind, k["asked"], k["told"], k["never told"]))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "by_kind")}, ensure_ascii=False))
    if "not committed" not in code or a.note:
        with open(os.path.join(HERE, "results", "talk.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
