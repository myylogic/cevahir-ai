# -*- coding: utf-8 -*-
"""The gates that are not batteries (docs/architecture/FINAL_FORM.md, section 11): G4 exactness, G5 the owner's
start, and the seconds an answer takes in conversation (G2).

    python gates.py exact                     # G4: live and forget; the digest of the substance before and after
    python gates.py start [--mind FOLDER]     # G5: python -m cevher as cevahir.ps1 runs it, timed from outside
    python gates.py answers --mind FOLDER     # G2: seconds an answer, on the raised mind

Rows are appended to results/gates.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "research", "living_intelligence_2026_11_24"))
import yardstick as Y  # noqa: E402


def put(args, row):
    row = {"code": Y.CODE, **row}
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False)[:1500], flush=True)


def exact(args):
    """Living a text and forgetting it; living dialogues and forgetting them; the same with tables that are full.
    Each case: the digest before, after living, after forgetting."""
    import corpora
    from cevher import Cevahir
    text = corpora.stream("tr-lessons")
    english = corpora.stream("en-pydoc")
    for name, budget, first, then in (
            ("a text, tables with room", 512, english[:200000], text),
            ("a text, tables full (2 MB for both)", 98, english[:400000], text),
            ("a longer text, tables full (2 MB for both)", 98, english[:400000], english[400000:500000])):
        c = Cevahir(budget_mb=budget, ledger=True)
        c.read(first)
        before = c.digest()
        size_before = c.size()
        c.read(then)
        lived = c.digest()
        size_lived = c.size()
        t0 = time.time()
        c.forget(len(c.entries) - 1)
        put(args, {"gate": "G4", "case": name, "chars_lived_before": len(first), "chars_lived_and_forgotten": len(then),
                   "digest_before": before, "digest_after_living": lived, "digest_after_forgetting": c.digest(),
                   "equal": c.digest() == before, "changed_by_living": lived != before,
                   "ties_given_way_while_living": [size_lived["ties"]["let_go"] - size_before["ties"]["let_go"],
                                                   size_lived["letters"]["let_go"] - size_before["letters"]["let_go"]],
                   "forget_s": round(time.time() - t0, 2), "ledger_bytes": c.ledger_bytes()})
    for budget in (512, 98):
        c = Cevahir(budget_mb=budget, ledger=True)
        if budget == 98:
            c.read(english[:400000])
        folder = tempfile.mkdtemp(prefix="gates-lessons-")
        paths = Y.lesson_files(folder)
        marks = []
        for path in paths[:12]:
            taught, _, _ = Y.lesson_lines(path)
            marks.append((c.digest(), len(c.entries)))
            for line in taught:
                if "\t" in line:
                    c.watch(*[x.strip() for x in line.split("\t", 1)])
                else:
                    c.tell(line.strip())
        ok = []
        for digest, n in reversed(marks):                    # the lessons forgotten again, the last first
            for k in range(len(c.entries) - 1, n - 1, -1):
                c.forget(k)
            ok.append(c.digest() == digest)
        put(args, {"gate": "G4", "case": "12 lessons lived one by one and forgotten again, the last first; tables "
                   + ("with room" if budget == 512 else "full (2 MB for both)"),
                   "entries": len(c.entries), "equal_after_each_lesson_forgotten": ok, "equal": all(ok)})
    # not in the order lived: an early entry forgotten after later ones were lived
    c = Cevahir(ledger=True)
    c.watch("gizli söz nedir ?", "mavi fil .")
    c.tell("Kedi süt içer ve uyur.")
    never = Cevahir(ledger=True)
    never.tell("Kedi süt içer ve uyur.")
    c.forget(0)
    put(args, {"gate": "G4", "case": "an earlier entry forgotten after a later one was lived (tables with room)",
               "words_equal_to_a_mind_that_never_lived_it": c.store.digest() == never.store.digest(),
               "letters_equal": c.letters.T.digest() == never.letters.T.digest(),
               "note": "the letters of the later entry were written after the forgotten entry's last bytes, and "
                       "keep those contexts: the two minds differ by them"})


def start(args):
    """python -m cevher, as cevahir.ps1 starts it, in a fresh process with a fresh home: the seconds until it has
    answered, its peak resident memory, and the exchange of gate G5."""
    said = ["merhaba", "benim adım yasin", "bugün hava çok güzel", "nasılsın", "benim adım ne ?", "adın ne",
            "python nedir ?", "!durum", "!cik"]
    home = tempfile.mkdtemp(prefix="gates-home-")
    env = dict(os.environ, CEVAHIR_HOME=home, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-m", "cevher"] + (["--load", args.mind] if args.mind else [])
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-v"] + cmd, input="\n".join(said) + "\n", capture_output=True, text=True,
                       cwd=REPO, env=env)
    seconds = time.time() - t0
    peak = [line for line in p.stderr.splitlines() if "Maximum resident" in line]
    lines = [x.strip() for x in p.stdout.splitlines()]
    put(args, {"gate": "G5", "command": " ".join(cmd[1:]), "whole_run_s": round(seconds, 2),
               "peak_resident_mb": round(int(peak[0].split()[-1]) / 1024, 1) if peak else None,
               "imports": sorted(m for m in ("torch", "numpy") if subprocess.run(
                   [sys.executable, "-c", "import sys, cevher; sys.exit(0 if %r in sys.modules else 1)" % m],
                   cwd=REPO).returncode == 0),
               "transcript": [[q, a] for q, a in zip(said, lines[1:])], "first_line": lines[0] if lines else ""})
    again = subprocess.run(cmd, input="benim adım ne ?\n!cik\n", capture_output=True, text=True, cwd=REPO, env=env)
    put(args, {"gate": "G5", "what": "started again from the home it saved itself to",
               "transcript": [x.strip() for x in again.stdout.splitlines()]})


def answers(args):
    """Seconds an answer in conversation, on the raised mind: the questions of the lessons, the examination lines
    never taught, and lines it was never told anything about."""
    from cevher import Cevahir
    c = Cevahir.load(args.mind)
    folder = tempfile.mkdtemp(prefix="gates-lessons-")
    qs, new = [], ["bugün hava nasıl ?", "ne düşünüyorsun ?", "bana bir şey anlat", "en sevdiğim renk ne ?",
                   "istanbul un nüfusu kaç ?", "dün akşam ne yedin ?", "kaç yaşındasın ?", "yarın ne yapalım ?"]
    for path in Y.lesson_files(folder):
        _, asked, exam = Y.lesson_lines(path)
        qs += [q for q, _ in asked + exam]
    times = []
    for q in qs + new * 5:
        t0 = time.perf_counter()
        c.reply(q)
        times.append(time.perf_counter() - t0)
    times.sort()
    put(args, {"gate": "G2", "what": "seconds an answer, the raised mind", "asked": len(times),
               "mean_s": round(sum(times) / len(times), 5), "median_s": round(times[len(times) // 2], 5),
               "p95_s": round(times[int(len(times) * 0.95)], 5), "max_s": round(times[-1], 5)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["exact", "start", "answers"])
    ap.add_argument("--mind", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "gates.jsonl"))
    a = ap.parse_args()
    {"exact": exact, "start": start, "answers": answers}[a.what](a)


if __name__ == "__main__":
    main()
