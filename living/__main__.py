# -*- coding: utf-8 -*-
"""Living with Cevahir from the terminal (living/cevahir.py).

    python -m living [--load FOLDER] [--variant full|speaking] [--dil tr|diger]

Every line you type is said to Cevahir; it answers (or stays silent: "(sessiz)") and lives the turn. Commands:

    !oku PATH        read a text file, or every .txt/.md/.py file under a folder (each sentence; code line by line)
    !ogret PATH      watch the dialogues of a .tsv file ("what is said<TAB>the answer", one per line)
    !kaydet FOLDER   keep Cevahir in FOLDER (Zihin, the core, what Cevahir learned, a manifest of digests)
    !dogru / !yanlis how its last answer was received (the amygdala's record; a wrong answer is not lived as right)
    !gecmis          the entries of this session's history;  !unut N  forget entry N exactly
    !uyu             a night: the cortex forgets what did not recur, Zihin dreams
    !durum           what Cevahir holds
    !cik             leave (unsaved life is lost)
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default="")
    ap.add_argument("--variant", default="full", choices=["full", "speaking"])
    ap.add_argument("--dil", default="tr", choices=["tr", "diger"])
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    from living.cevahir import Cevahir
    c = Cevahir.load(a.load) if a.load else Cevahir(variant=a.variant, dil=a.dil, ledger=True)
    if c.entries is None:                                    # a loaded mind: the history begins now
        c.entries = []
    print("Cevahir %s. !cik ile çıkılır." % ("yüklendi: " + a.load if a.load else "doğdu"), flush=True)
    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("!cik"):
            break
        if line.startswith("!kaydet "):
            m = c.save(line[len("!kaydet "):].strip())
            print("  kaydedildi: %d dosya, %d an" % (len(m["files"]), m["zihin_moments"]), flush=True)
            continue
        if line.startswith("!dogru") or line.startswith("!yanlis"):
            c.feedback(line.startswith("!dogru"))
            print("  kaydedildi", flush=True)
            continue
        if line.startswith("!gecmis"):
            for i, e in enumerate(c.entries):
                print("  %d %s %s%s" % (i, e["kind"], e["text"][:60].replace("\n", " ").replace("\t", " | "),
                                        " (unutuldu)" if e["forgotten"] else ""), flush=True)
            continue
        if line.startswith("!unut "):
            got = c.forget(int(line.split()[1]))
            print("  unutuldu: %s" % got, flush=True)
            continue
        if line.startswith("!uyu"):
            print("  %s" % c.sleep(), flush=True)
            continue
        if line.startswith("!durum"):
            print("  Zihin %d an; korteks %s; ses ile yaşanan %d kelime" % (
                c.mind.t, c.cortex.size(), c.counts["voiced"]), flush=True)
            continue
        if line.startswith("!oku "):
            path = os.path.expanduser(line[len("!oku "):].strip())
            files = [path] if os.path.isfile(path) else sorted(
                os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.endswith((".txt", ".md", ".py")))
            for name in files:
                with open(name, encoding="utf-8", errors="replace") as f:
                    c.read(f.read() + "\n", code=name.endswith(".py"))
            print("  okundu: %d dosya" % len(files), flush=True)
            continue
        if line.startswith("!ogret "):
            n = 0
            with open(os.path.expanduser(line[len("!ogret "):].strip()), encoding="utf-8") as f:
                for row in f:
                    if "\t" in row:
                        q, ans = row.rstrip("\n").split("\t", 1)
                        c.watch(q, ans)
                        n += 1
            print("  izlendi: %d diyalog" % n, flush=True)
            continue
        print("  " + (c.converse(line) or "(sessiz)"), flush=True)


if __name__ == "__main__":
    main()
