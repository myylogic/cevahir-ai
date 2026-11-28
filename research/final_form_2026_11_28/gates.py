# -*- coding: utf-8 -*-
"""The gates that are not batteries (docs/architecture/FINAL_FORM.md, section 11): G4 exactness, G5 the owner's
start, and the seconds an answer takes in conversation (G2).

    python gates.py exact                     # G4: live and forget; the digest of the substance before and after
    python gates.py start [--mind FOLDER]     # G5: python -m cevher as cevahir.ps1 runs it, timed from outside
    python gates.py answers --mind FOLDER     # G2: seconds an answer, on the raised mind
    python gates.py bound --file TEXT         # G1: the whole process, born empty, reads a long file; started again
    python gates.py lives_on --mind FOLDER    # does it go on learning, and keep what it learned, once it has read much

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


def measured(cmd, said, env):
    """A command run with lines on its standard input: what it printed, its seconds, and the most resident memory
    the process reached, megabytes (the kernel's own account of that child, read when it is waited for)."""
    with tempfile.TemporaryFile("w+", encoding="utf-8") as fin, tempfile.TemporaryFile("w+", encoding="utf-8") as fout:
        fin.write("\n".join(said) + "\n")
        fin.flush()
        fin.seek(0)
        t0 = time.time()
        p = subprocess.Popen(cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL, cwd=REPO, env=env)
        _, status, usage = os.wait4(p.pid, 0)
        seconds = time.time() - t0
        p.returncode = os.waitstatus_to_exitcode(status)
        fout.seek(0)
        printed = [x.strip() for x in fout.read().splitlines()]
    return printed, round(seconds, 2), round(usage.ru_maxrss / 1024.0, 1)


def exact(args):
    """Living a text and forgetting it; living dialogues and forgetting them; the same with tables that are full.
    Each case: the digest before, after living, after forgetting."""
    import corpora
    from cevher import Cevahir
    text = corpora.stream("tr-lessons")
    english = corpora.stream("en-pydoc")
    source = corpora.stream("code-pystdlib")[:150000]
    for name, budget, first, then, lines in (
            ("a text, tables with room", 512, english[:200000], text, False),
            ("a text, tables full (2 MB for both)", 98, english[:400000], text, False),
            ("a longer text, tables full (2 MB for both)", 98, english[:400000], english[400000:500000], False),
            ("source code read by lines, tables with room", 512, english[:200000], source, True),
            ("source code read by lines, tables full (2 MB for both)", 98, english[:400000], source, True)):
        c = Cevahir(budget_mb=budget, ledger=True)
        c.read(first)
        before, moments = c.digest(), c.moments()
        size_before = c.size()
        c.read(then, code=lines)
        lived = c.digest()
        size_lived = c.size()
        t0 = time.time()
        c.forget(len(c.entries) - 1)
        put(args, {"gate": "G4", "case": name, "chars_lived_before": len(first), "chars_lived_and_forgotten": len(then),
                   "digest_before": before, "digest_after_living": lived, "digest_after_forgetting": c.digest(),
                   "equal": c.digest() == before, "moments_equal": c.moments() == moments,
                   "changed_by_living": lived != before,
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
            marks.append((c.digest() + c.moments(), len(c.entries)))
            for line in taught:
                if "\t" in line:
                    c.watch(*[x.strip() for x in line.split("\t", 1)])
                else:
                    c.tell(line.strip())
        ok = []
        for digest, n in reversed(marks):                    # the lessons forgotten again, the last first
            for k in range(len(c.entries) - 1, n - 1, -1):
                c.forget(k)
            ok.append(c.digest() + c.moments() == digest)
        put(args, {"gate": "G4", "case": "12 lessons lived one by one and forgotten again, the last first; tables "
                   + ("with room" if budget == 512 else "full (2 MB for both)"),
                   "entries": len(c.entries), "equal_after_each_lesson_forgotten": ok, "equal": all(ok),
                   "note": "the digest of the counts and the digest of the moments, both"})
    # what was told last: said, said otherwise, said as at first; the last telling forgotten
    c = Cevahir(ledger=True)
    with open(os.path.join(REPO, "cevher", "dersler", "konusma.txt"), encoding="utf-8") as f:
        for row in f:
            row = row.rstrip("\n")
            if not row.strip() or row.startswith("#"):
                continue
            if "\t" in row:
                c.watch(*[x.strip() for x in row.split("\t", 1)])
            else:
                c.tell(row.strip())
    c.tell("halının rengi mavi .")
    c.tell("halının rengi sarı .")
    never, before, moments = c.reply("halının rengi ne ?"), c.digest(), c.moments()
    c.tell("halının rengi mavi .")
    lived = c.reply("halının rengi ne ?")
    c.forget(len(c.entries) - 1)
    put(args, {"gate": "G4", "case": "told one thing, then another, then the first again; the last telling forgotten",
               "answer_before_the_last_telling": never, "answer_after_it": lived,
               "answer_after_forgetting_it": c.reply("halının rengi ne ?"),
               "equal": c.digest() == before, "moments_equal": c.moments() == moments})
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
    lines, seconds, peak = measured(cmd, said, env)
    put(args, {"gate": "G5", "command": " ".join(cmd[1:]), "whole_run_s": seconds, "peak_resident_mb": peak,
               "imports": sorted(m for m in ("torch", "numpy") if subprocess.run(
                   [sys.executable, "-c", "import sys, cevher; sys.exit(0 if %r in sys.modules else 1)" % m],
                   cwd=REPO).returncode == 0),
               "transcript": [[q, a] for q, a in zip(said, lines[1:])], "first_line": lines[0] if lines else ""})
    again = subprocess.run(cmd, input="benim adım ne ?\n!cik\n", capture_output=True, text=True, cwd=REPO, env=env)
    put(args, {"gate": "G5", "what": "started again from the home it saved itself to",
               "transcript": [x.strip() for x in again.stdout.splitlines()]})


def bound(args):
    """Gate G1's bound on the whole process, as the owner would run it: ``python -m cevher`` born empty with the
    budget reads a long file and tells its state; then it is started again from what it saved. The most resident
    memory of each process is the kernel's account of it."""
    home = tempfile.mkdtemp(prefix="gates-bound-")
    env = dict(os.environ, CEVAHIR_HOME=home, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def run(cmd, said):
        printed, seconds, peak = measured(cmd, said, env)
        return {"command": " ".join(cmd[1:]), "said": said, "whole_run_s": seconds, "peak_resident_mb": peak,
                "printed": [x[:600] for x in printed]}

    row = run([sys.executable, "-m", "cevher", "--yeni", "--butce", str(args.budget)],
              ["!oku " + args.file, "!durum", "benim adım yasin", "benim adım ne ?"])
    kept = {f: os.path.getsize(os.path.join(home, f)) for f in sorted(os.listdir(home))}
    put(args, {"gate": "G1", "what": "the whole process: born empty, a long file read, kept on leaving",
               "budget_mb": args.budget, "file_bytes": os.path.getsize(args.file), **row, "kept_bytes": kept,
               "within_budget": row["peak_resident_mb"] <= args.budget})
    row = run([sys.executable, "-m", "cevher"], ["!durum", "benim adım ne ?"])
    put(args, {"gate": "G1", "what": "the whole process: started again from what it kept", "budget_mb": args.budget,
               **row, "within_budget": row["peak_resident_mb"] <= args.budget})


def lives_on(args):
    """Does the mind go on learning once it has read much, and does it keep what it had learned? The raised mind
    reads the first N characters of a text. Then: the dialogues of the lessons it was raised on are asked again
    (keeping); a dialogue is watched once and asked, another is watched three times (learning a new thing); five
    dialogues of a made-up world are watched and a combination never shown is asked (learning a new rule); and
    after 100,000 more characters of the text the three are asked once more. One row for each N."""
    from cevher import Cevahir
    folder = tempfile.mkdtemp(prefix="gates-lessons-")
    asked = []
    for path in Y.lesson_files(folder):
        asked += Y.lesson_lines(path)[1]
    text = Y.text_of(args.corpus, 0)

    def kept(c):
        return sum(1 for (_, a), said in zip(asked, c.answers([q for q, _ in asked])) if Y.judge(said, a)[0])

    def three(c):
        return {"watched once": c.reply("gizli söz nedir ?"), "watched three times": c.reply("saklı sayı kaç ?"),
                "a new rule (wif twice)": c.reply("wif twice")}

    for n in args.chars:
        c = Cevahir.load(args.mind)
        row = {"gate": "lives on", "corpus": os.path.basename(args.corpus), "chars_read": n,
               "lesson_dialogues": len(asked), "answered_before_reading": kept(c)}
        t0 = time.time()
        for i in range(0, n, 1000000):
            c.read(text[i:min(n, i + 1000000)])
        row["read_s"] = round(time.time() - t0, 1)
        row["answered_after_reading"] = kept(c)
        c.watch("gizli söz nedir ?", "mavi fil .")
        for _ in range(3):
            c.watch("saklı sayı kaç ?", "kırk iki .")
        for q, a in (("zap", "Z"), ("zap twice", "Z Z"), ("dax", "D"), ("dax twice", "D D"), ("wif", "W")):
            c.watch(q, a + " .")
        row["asked_at_once"] = three(c)
        c.read(text[n:n + 100000])
        row["asked_after_100000_more"] = three(c)
        row["answered_at_the_end"] = kept(c)
        size = c.size()
        row["ties"] = size["ties"]
        put(args, row)


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
    ap.add_argument("what", choices=["exact", "start", "answers", "bound", "lives_on"])
    ap.add_argument("--corpus", default="all", help="lives_on: a corpus of corpora.py, 'all', or a text file")
    ap.add_argument("--chars", type=int, nargs="*", default=[0, 100000, 1000000, 3000000, 10000000])
    ap.add_argument("--mind", default="")
    ap.add_argument("--file", default="", help="bound: the long text to read")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "gates.jsonl"))
    a = ap.parse_args()
    {"exact": exact, "start": start, "answers": answers, "bound": bound, "lives_on": lives_on}[a.what](a)


if __name__ == "__main__":
    main()
