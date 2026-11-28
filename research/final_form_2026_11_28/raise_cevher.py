# -*- coding: utf-8 -*-
"""Raise the open mind of the final form from nothing, on the repository's public material only.

    python raise_cevher.py [OUT_FOLDER]          (default: cevher/zihin, the mind ``python -m cevher`` first loads)

The 36 lessons of minds/raise_cevahir.py, in its order (the dialogue and text files of the first experiments, the
seven school subjects of t_school.py written out as lessons, and the 24 lessons of public knowledge), and after
them one lesson written for the final form, ``cevher/dersler/konusma.txt`` (conversation: something is said, then
asked for; what is said last holds). Each lesson is studied as m3_etut.py studies one: taught, its questions
answered by Cevahir in conversation, a right answer confirmed, a wrong one forgotten exactly and the answer shown,
three rounds at most. The mind is kept with ``Cevahir.save`` (one file of arrays and a manifest; no pickle).
"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import yardstick as Y  # noqa: E402


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "cevher", "zihin")
    paths = Y.lesson_files(tempfile.mkdtemp(prefix="raise-cevher-"))
    paths.append(os.path.join(REPO, "cevher", "dersler", "konusma.txt"))
    m = Y.New(ledger=True)
    t0 = time.time()
    rows = []
    for i, path in enumerate(paths):
        row = Y.etut(m, path, i)
        row.pop("converse_seconds")
        rows.append(row)
        print("  %s: rounds %s of %d, examination %d/%d" % (row["lesson"], row["rounds"], row["asked"], *row["exam"]),
              flush=True)
    m.flush()
    lessons = os.path.join(REPO, "cevher", "dersler")       # the lesson that shows telling: two texts, read each as
    for name in ("saat", "gitar"):                           # one text, and the requests shown about them
        with open(os.path.join(lessons, "metinler", name + ".txt"), encoding="utf-8") as f:
            m.c.read(f.read())
    with open(os.path.join(lessons, "anlatma.tsv"), encoding="utf-8") as f:
        told = [line.rstrip("\n").split("\t", 1) for line in f if "\t" in line and not line.startswith("#")]
    for q, a in told:
        m.c.watch(q, a)
    print("  anlatma: 2 texts read, %d dialogues watched" % len(told), flush=True)
    manifest = m.save(out)
    report = {"lessons": rows, "seconds": round(time.time() - t0, 1), "manifest": manifest}
    with open(os.path.join(out, "study.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"saved": out, "seconds": report["seconds"], "size": manifest["size"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
