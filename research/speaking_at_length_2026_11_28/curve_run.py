# -*- coding: utf-8 -*-
"""The lives of curve.json lived and asked by the package (criteria: curve_make.py). Rows: results/curve.jsonl.

    python curve_run.py [--show]
"""
import argparse, hashlib, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from cevher import Cevahir  # noqa: E402
from cevher.telling import R_FREE  # noqa: E402
from cevher.units import body, tokens  # noqa: E402

CLOSE = re.compile(r"(?<=[.!?…])\s+")
KINDS = ("parts", "works", "made", "care", "two kinds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    blob = open(os.path.join(HERE, "curve.json"), "rb").read()
    assert hashlib.sha256(blob).hexdigest() == open(os.path.join(HERE, "curve.sha256")).read().split()[0]
    Y = json.loads(blob.decode("utf-8"))
    code = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", ":(glob)cevher/**/*.py"]).returncode:
        code += " and changes not committed"
    lives = []
    for shown in ("2", "4", "6"):
        c = Cevahir(ledger=False)
        for s in Y["lessons"]:
            c.tell(s)
        for name in Y["order"]:
            c.read("\n".join(Y["texts"][name]) + "\n")
        for d in Y["shown"][shown]:
            c.watch(d["q"], " ".join(d["must"]))
        rows = []
        for d in Y["asked"]:
            said = [x.strip() for x in CLOSE.split(c.reply(d["q"]).strip()) if x.strip()]
            must, may = set(d["must"]), set(d["may"])
            rows.append({"thing": d["thing"], "form": d["form"], "said": len(said), "must": len(must),
                         "right": sum(x in must for x in said), "beside": sum(x in may for x in said),
                         "wrong": sum(x not in must and x not in may for x in said),
                         "way": c.telling.answer(body(tokens(d["q"]))[0])[0], "answer": said})
            if a.show and d["form"] in KINDS + ("in no shown form",):
                print(shown, d["thing"], d["form"], "|", rows[-1]["way"])
                for x in said:
                    print("      %s %s" % ("+" if x in must else ("~" if x in may else "!"), x))
        kinds = [r for r in rows if r["form"] in KINDS]
        free = [r for r in rows if r["form"] == "in no shown form"]
        life = {"shown": int(shown), "said": sum(r["said"] for r in rows), "wrong": sum(r["wrong"] for r in rows),
                "kinds_right": sum(r["right"] for r in kinds), "kinds_must": sum(r["must"] for r in kinds),
                "kinds_beside": sum(r["beside"] for r in kinds),
                "by_kind": {k: [sum(r["right"] for r in kinds if r["form"] == k), sum(r["must"] for r in kinds if r["form"] == k),
                                sum(r["beside"] for r in kinds if r["form"] == k)] for k in KINDS},
                "no_shown_form": [[r["right"], r["must"], r["beside"], r["wrong"]] for r in free],
                "its_own_words_right_of_tried": list(c.telling._record(R_FREE)),
                "one_sentence": all(r["said"] == 1 and r["right"] == 1 for r in rows if r["form"] == "one sentence"),
                "all_of_it": all(r["right"] >= 0.9 * r["must"] for r in rows if r["form"] == "all"), "rows": rows}
        lives.append(life)
        print(json.dumps({k: v for k, v in life.items() if k != "rows"}, ensure_ascii=False))
    two, four, six = lives
    free = six["no_shown_form"]
    said = sum(r[0] + r[2] + r[3] for r in free)
    out = {"code": code, "what": "the same requests after two, four and six things were shown",
           "1_no_wrong_sentence": all(x["wrong"] == 0 for x in lives),
           "2_kinds_rise": two["kinds_right"] <= four["kinds_right"] <= six["kinds_right"] > two["kinds_right"]
           and six["kinds_beside"] * two["kinds_right"] <= two["kinds_beside"] * six["kinds_right"],
           "3_no_shown_form_at_six": all(r[0] > 0 for r in free) and sum(r[0] for r in free) >= 4
           and 3 * sum(r[2] for r in free) <= said,
           "4_one_sentence_and_all": all(x["one_sentence"] and x["all_of_it"] for x in lives), "lives": lives}
    out["survives"] = all(out[k] for k in out if k[:1] in "1234")
    print(json.dumps({k: v for k, v in out.items() if k != "lives"}, ensure_ascii=False))
    if "not committed" not in code:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        with open(os.path.join(HERE, "results", "curve.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
