# -*- coding: utf-8 -*-
"""The yardstick of the final-form study: one runner, two minds, the same batteries.

    python yardstick.py BATTERY --who old|new [options] [--out results/yardstick.jsonl]

``--who old`` is today's Cevahir (``living.cevahir.Cevahir``: Zihin, the cortex and the hippocampus); ``--who new``
is the final form (``cevher.Cevahir``). Both are driven through the same few calls (watch, tell, read, converse,
feedback, forget, answers), every life and every question is built by the battery's own code of the earlier
studies, and every answer is graded here by one rule that knows nothing of either mind: the words of what was said
(lower-cased by Zihin's Turkish rule, closing marks apart) against the words of the key. An answer is right,
silent or wrong; a wrong answer is worse than a silent one.

Batteries (the protocol of each is the earlier script's, which is named):

- ``scan``      m3_scan.py: the life of each split (every training command without a connective, 1,000 sampled with
                one, watched twice, the shorter first in the first epoch), ``--test`` seeded test commands asked.
                ``--scan FOLDER`` is SCAN in the published layout (scan_files.py writes it from the grammar).
- ``cogs``      m3_cogs.py: every line of train.tsv watched, 300 of test.tsv and 300 of gen.tsv asked. Needs
                ``--cogs FOLDER`` (the published data folder); without it a row says that it was not run.
- ``school``    m3_school.py: the seven subjects, three times, the examination by kind.
- ``change``    m3_change.py: facts that change; right is the place told last.
- ``dialogues`` minds/raise_cevahir.py with m3_etut.py: the 36 lessons studied in conversation (a right answer
                confirmed, a wrong one forgotten exactly and the answer shown, three rounds at most), then all 440
                dialogues asked again. ``--save FOLDER`` keeps the raised mind. The seconds of every answer given
                in conversation are kept.
- ``scale``     m3_scale.py memory: the first ``--chars`` characters of a corpus read in ten pieces; seconds,
                resident memory gained, and the bytes reached by the same walk of objects.
- ``heldout``   the set frozen in heldout.json before the design (gate G6), its sha256 checked; the transcript is
                printed in full.

Each row is appended to ``--out`` with the commit of the code that wrote it.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
M3 = os.path.join(REPO, "research", "living_intelligence_2026_11_24")
for p in (REPO, M3, os.path.join(REPO, "research", "living_mind_2026_11_22")):
    if p not in sys.path:
        sys.path.insert(0, p)

FINAL = (".", "?", "!", "…")
HELDOUT_SHA256 = "854506933efb9fa274bf9d5b40745b99d20b585d63c0057dfe452e1a0c5de45d"


# ------------------------------------------------------------------------------------------------- grading -------
def lower_tr(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def words(text: str) -> list:
    """Zihin's text adapter (zihin.__main__.words): words and marks, lower-cased by the Turkish rule."""
    return re.findall(r"\w+|[^\w\s]", lower_tr(text), flags=re.UNICODE)


def body(text: str) -> list:
    out = words(text)
    while out and out[-1] in FINAL:
        out = out[:-1]
    return out


def is_silent(said) -> bool:
    return not said or not said.strip() or said.strip() == "(sessiz)" or not body(said)


def verdict(said: str, key: str) -> str:
    if is_silent(said):
        return "silent"
    return "right" if body(said) == body(key) else "wrong"


def code_version() -> str:
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


CODE = code_version()
PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def resident_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE / 1e6


# ------------------------------------------------------------------------------------------------- the minds -----
class Old:
    """Today's Cevahir, as each battery of the earlier study configures it."""
    who = "old"

    def __init__(self, variant="full", switches=None, window=0, seed=0, ledger=False, loaded=None):
        import torch
        torch.set_num_threads(1)
        from living.cevahir import Cevahir
        self.c = loaded if loaded is not None else Cevahir(variant=variant, switches=switches, window=window,
                                                            seed=seed, ledger=ledger)

    def watch(self, first, answer):
        self.c.watch(first, answer)

    def tell(self, sentence, tone=None):
        self.c.tell(sentence, tone=tone)

    def read(self, text):
        return self.c.read(text)

    def converse(self, said, limit=80):
        return self.c.converse(said, limit=limit)

    def feedback(self, good):
        self.c.feedback(good)

    def entries(self):
        return len(self.c.entries)

    def forget(self, k):
        out = self.c.forget(k)
        self.c.session.open_turn, self.c.session.answered = False, None      # as m3_etut.py does after forgetting
        return out

    def pause(self):
        for _ in range(2):                                   # a silence between two questions (m3_etut.py)
            self.c.mind.step([])

    def close(self):
        self.c.session.close_turn()

    def flush(self):
        self.c.flush()

    def answers(self, questions, limit=400, workers=2):
        return self.c.answers(list(questions), way="cevahir", limit=limit, workers=workers)

    def save(self, folder):
        return self.c.save(folder)

    @classmethod
    def load(cls, folder):
        import torch
        torch.set_num_threads(1)
        from living.cevahir import Cevahir
        me = cls(loaded=Cevahir.load(folder))
        me.c.keep_ledger()                                   # as python -m living does for a loaded mind
        return me

    def stats(self):
        S = self.c.mind.S
        return {"zihin_moments": self.c.mind.t, "units": int(S.n_units), "ties": int(S.n_ties),
                "cortex": self.c.cortex.size()}

    def walk(self):
        import m3_scale
        return m3_scale.walk(self.c)


class New:
    """The final form (``cevher.Cevahir``): one configuration for every battery."""
    who = "new"

    def __init__(self, variant="full", switches=None, window=0, seed=0, ledger=False, loaded=None):
        from cevher import Cevahir
        self.c = loaded if loaded is not None else Cevahir(ledger=ledger)

    def watch(self, first, answer):
        self.c.watch(first, answer)

    def tell(self, sentence, tone=None):
        self.c.tell(sentence)

    def read(self, text):
        return self.c.read(text)

    def converse(self, said, limit=80):
        return self.c.converse(said, limit=limit)

    def feedback(self, good):
        self.c.feedback(good)

    def entries(self):
        return len(self.c.entries)

    def forget(self, k):
        return self.c.forget(k)

    def pause(self):
        pass

    def close(self):
        pass

    def flush(self):
        self.c.flush()

    def answers(self, questions, limit=400, workers=2):
        return self.c.answers(list(questions), limit=limit)

    def save(self, folder):
        return self.c.save(folder)

    @classmethod
    def load(cls, folder):
        from cevher import Cevahir
        return cls(loaded=Cevahir.load(folder))

    def stats(self):
        return self.c.size()

    def walk(self):
        import m3_scale
        parts, seen = {}, set()
        owner = self.c
        for k in sorted(set(getattr(type(owner), "__slots__", ())) | set(getattr(owner, "__dict__", {}))):
            if hasattr(owner, k):
                parts["cevher." + k] = m3_scale.deep_bytes(getattr(owner, k), seen)
        return parts


MINDS = {"old": Old, "new": New}


def put(args, row: dict) -> None:
    row = {"code": CODE, "who": args.who, **row}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shown = {k: v for k, v in row.items() if k not in ("not_right", "parts", "transcript", "lessons", "seconds_each")}
    print(json.dumps(shown, ensure_ascii=False)[:1800], flush=True)


# ---------------------------------------------------------------------------------------------------- scan -------
SCAN_FILES = {"add_jump": ("add_prim_split/tasks_train_addprim_jump.txt", "add_prim_split/tasks_test_addprim_jump.txt"),
              "length": ("length_split/tasks_train_length.txt", "length_split/tasks_test_length.txt"),
              "around_right": ("template_split/tasks_train_template_around_right.txt",
                               "template_split/tasks_test_template_around_right.txt"),
              "jump_around_right": ("template_split/tasks_train_template_jump_around_right.txt",
                                    "template_split/tasks_test_template_jump_around_right.txt"),
              "opposite_right": ("template_split/tasks_train_template_opposite_right.txt",
                                 "template_split/tasks_test_template_opposite_right.txt"),
              "right": ("template_split/tasks_train_template_right.txt",
                        "template_split/tasks_test_template_right.txt")}


def published(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                command, answer = line[len("IN: "):].split(" OUT: ")
                out.append((tuple(command.split()), tuple(answer.split())))
    return out


def scan_life(folder, split, seed, n_train, n_test):
    """m3_scan.life_of, unchanged."""
    train_file, test_file = SCAN_FILES[split]
    train = sorted(set(published(os.path.join(folder, train_file))))
    test = published(os.path.join(folder, test_file))
    rng = random.Random(seed)
    simple = [d for d in train if "and" not in d[0] and "after" not in d[0]]
    longer = [d for d in train if "and" in d[0] or "after" in d[0]]
    if n_train:
        longer = rng.sample(longer, min(n_train, len(longer)))
    if n_test:
        test = random.Random(seed + 7).sample(test, min(n_test, len(test)))
    lived = simple + longer
    rng = random.Random(seed + 1)
    life = []
    for epoch in range(2):
        block = list(lived)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0]))
        life += block
    return train, lived, life, test


def scan(args) -> None:
    for split in args.split or list(SCAN_FILES):
        train, lived, life, test = scan_life(args.scan, split, args.seed, args.train, args.test)
        m = MINDS[args.who](variant="speaking", switches={"wholes": True, "told_apart": True}, window=13,
                            seed=args.seed)
        t0 = time.time()
        for command, answer in life:
            m.watch(" ".join(command), " ".join(answer) + " .")
        m.close()
        life_s = time.time() - t0
        t0 = time.time()
        said = m.answers([" ".join(c) for c, _ in test], limit=400, workers=args.workers)
        answers_s = time.time() - t0
        got = Counter(verdict(s, " ".join(a)) for s, (_, a) in zip(said, test))
        chars = sum(len(" ".join(c)) + len(" ".join(a)) + 4 for c, a in life)
        put(args, {"battery": "scan", "split": split, "seed": args.seed, "train_watched": len(lived),
                   "train_published": len(train), "asked": len(test), "right": got["right"],
                   "silent": got["silent"], "wrong": got["wrong"], "life_s": round(life_s, 1),
                   "life_chars": chars, "answers_s": round(answers_s, 1), "workers": args.workers,
                   "stats": m.stats(),
                   "not_right": [{"command": " ".join(c), "said": s, "verdict": verdict(s, " ".join(a))}
                                 for s, (c, a) in zip(said, test) if verdict(s, " ".join(a)) != "right"][:8]})


# ---------------------------------------------------------------------------------------------------- cogs -------
def cogs(args) -> None:
    if not args.cogs or not os.path.exists(os.path.join(args.cogs, "train.tsv")):
        put(args, {"battery": "cogs", "run": False,
                   "why": "the published COGS files are not in the repository and this study downloads no dataset"})
        return

    def read(path):
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n").split("\t") for line in f if line.strip()]
    train = read(os.path.join(args.cogs, "train.tsv"))
    rng = random.Random(0)
    test = rng.sample(read(os.path.join(args.cogs, "test.tsv")), 300)
    gen = rng.sample(read(os.path.join(args.cogs, "gen.tsv")), 300)
    m = MINDS[args.who](variant="speaking", switches={"wholes": True, "told_apart": True}, window=13)
    t0 = time.time()
    for sentence, form, _ in train:
        m.watch(sentence, form)
    m.close()
    life_s = time.time() - t0
    t0 = time.time()
    said = m.answers([s for s, _, _ in test + gen], limit=120, workers=args.workers)
    row = {"battery": "cogs", "run": True, "train_lines": len(train), "life_s": round(life_s, 1),
           "answers_s": round(time.time() - t0, 1), "stats": m.stats()}
    for part, rows, got in (("test", test, said[:300]), ("gen", gen, said[300:])):
        c = Counter(verdict(s, form) for s, (_, form, _) in zip(got, rows))
        row[part] = {"asked": len(rows), "right": c["right"], "silent": c["silent"], "wrong": c["wrong"]}
    put(args, row)


# -------------------------------------------------------------------------------------------------- school -------
def school(args) -> None:
    import t_school as S
    for seed in range(args.seeds):
        m = MINDS[args.who](variant="full", window=S.WINDOW, seed=seed)
        t0 = time.time()
        course = S.course(seed)
        for lesson in course:
            if lesson[0] == "tell":
                m.tell(lesson[1], tone=".")
            else:
                m.watch(lesson[1], lesson[2])
        m.close()
        life_s = time.time() - t0
        jobs = [(name, kind, q, ans) for name, (_, exam) in S.subjects(seed).items() for kind, q, ans in exam]
        t0 = time.time()
        said = m.answers([q for _, _, q, _ in jobs], limit=20, workers=args.workers)
        answers_s = time.time() - t0
        kinds, subjects, bad = {}, {}, []
        for (name, kind, q, ans), got in zip(jobs, said):
            v = verdict(got, ans)
            kinds.setdefault(kind.split(":")[0], Counter())[v] += 1
            subjects.setdefault(name, {}).setdefault(kind, Counter())[v] += 1
            if v != "right" and len(bad) < 40:
                bad.append({"subject": name, "kind": kind, "q": q, "key": ans, "said": got, "verdict": v})
        chars = sum(len(" ".join(x[1:])) + 2 for x in course)
        put(args, {"battery": "school", "seed": seed, "lessons": len(course), "life_chars": chars,
                   "asked": len(jobs), "life_s": round(life_s, 1), "answers_s": round(answers_s, 1),
                   "workers": args.workers,
                   "kinds": {k: dict(v) for k, v in sorted(kinds.items())},
                   "subjects": {n: {k: dict(v) for k, v in ks.items()} for n, ks in subjects.items()},
                   "stats": m.stats(), "not_right": bad})


# -------------------------------------------------------------------------------------------------- change -------
def change(args) -> None:
    import m3_change as C
    for seed in range(args.seeds):
        life, questions = C.world(seed)
        m = MINDS[args.who](variant="speaking", seed=seed)
        t0 = time.time()
        for item in life:
            if item[0] == "tell":
                m.tell(item[1])
            else:
                m.watch(item[1], item[2])
        m.close()
        life_s = time.time() - t0
        said = m.answers(["%s nerede ?" % name for name, _, _, _ in questions], workers=args.workers)
        kinds, bad = {k: Counter() for k in C.KINDS}, []
        for (name, kind, last, old), s in zip(questions, said):
            got = [w for w in words(s or "") if w not in FINAL]
            if is_silent(s):
                v = "silent"
            elif got == [lower_tr(name), last]:
                v = "right"
            elif got == [lower_tr(name), old]:
                v = "old"
            else:
                v = "wrong"
            kinds[kind][v] += 1
            if v != "right":
                bad.append({"q": "%s nerede ?" % name, "kind": kind, "last": last, "old": old, "said": s})
        put(args, {"battery": "change", "seed": seed, "life_items": len(life), "life_s": round(life_s, 1),
                   "kinds": {k: dict(v) for k, v in kinds.items()}, "stats": m.stats(), "not_right": bad[:12]})


# ----------------------------------------------------------------------------------------------- dialogues -------
def lesson_files(folder: str) -> list:
    """The 36 lesson files of minds/raise_cevahir.py, in its order (the school's subjects written out first)."""
    import t_school
    os.makedirs(os.path.join(folder, "lessons"), exist_ok=True)
    for name, (lessons, _) in t_school.subjects(0).items():
        with open(os.path.join(folder, "lessons", "school_%s.txt" % name), "w", encoding="utf-8") as f:
            f.write("# %s (research/living_mind_2026_11_22/t_school.py, subjects(0))\n" % name)
            for lesson in lessons:
                f.write((lesson[1] + "\n") if lesson[0] == "tell" else (lesson[1] + "\t" + lesson[2] + "\n"))
    data = os.path.join(REPO, "research", "living_mind_2026_11_14", "data")
    out = [os.path.join(data, name) for name in ("first_dialogues.tsv", "feedback_dialogues.tsv", "school_text.txt",
                                                 "school_questions.tsv", "imagination_text.txt")]
    out += sorted(glob.glob(os.path.join(folder, "lessons", "school_*.txt")))
    out += sorted(glob.glob(os.path.join(REPO, "research", "living_mind_2026_11_22", "dersler", "*.txt")))
    return out


def lesson_lines(path: str):
    """research/living_mind_2026_11_22/etut.py ``lesson_lines``, unchanged."""
    taught, asked, exam = [], [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("?"):
                q, a = line[1:].split("\t", 1)
                exam.append((q.strip(), a.strip()))
                continue
            taught.append(line)
            if "\t" in line:
                q, a = line.split("\t", 1)
                asked.append((q.strip(), a.strip()))
    return taught, asked, exam


def judge(said: str, answer: str):
    """m3_etut.py's judgement: right when the answer begins with the whole key (closing marks apart); went on when
    more words follow."""
    want, got = body(answer), words(said or "")
    right = (not is_silent(said)) and got[:len(want)] == want
    return right, right and any(w not in FINAL for w in got[len(want):])


def dialogues(args) -> None:
    tmp = tempfile.mkdtemp(prefix="yardstick-lessons-")
    paths = lesson_files(tmp)
    m = MINDS[args.who](variant="full", ledger=True, seed=0)
    t_all = time.time()
    rows, seconds, first_right, asked_n, exam_right, exam_n, chars = [], [], 0, 0, 0, 0, 0
    for i, path in enumerate(paths):
        taught, asked, exam = lesson_lines(path)
        t0 = time.time()
        for line in taught:
            chars += len(line) + 1
            if "\t" in line:
                q, a = line.split("\t", 1)
                m.watch(q.strip(), a.strip())
            else:
                m.tell(line.strip())
        rng = random.Random(i)                               # m3_etut.py: seed + the lesson's number
        scores = []
        for r in range(3):
            order = list(asked)
            rng.shuffle(order)
            right = 0
            for q, a in order:
                m.pause()
                t1 = time.time()
                said = m.converse(q, limit=40)
                seconds.append(round(time.time() - t1, 4))
                ok, went = judge(said, a)
                if ok and not went:
                    right += 1
                    m.feedback(True)
                else:
                    right += 1 if ok else 0
                    m.feedback(False)
                    m.forget(m.entries() - 1)                # the attempt, taken back exactly
                    m.watch(q, a)                            # and the right answer shown
            scores.append(right)
            if right == len(asked):
                break
        passed = 0
        if exam:
            said = m.answers([q for q, _ in exam], limit=40, workers=args.workers)
            passed = sum(judge(x, a)[0] for x, (_, a) in zip(said, exam))
        first_right += scores[0] if scores else 0
        asked_n += len(asked)
        exam_right += passed
        exam_n += len(exam)
        rows.append({"lesson": os.path.splitext(os.path.basename(path))[0], "lines": len(taught),
                     "asked": len(asked), "rounds": scores, "exam": [passed, len(exam)],
                     "seconds": round(time.time() - t0, 1)})
        print("  %s: rounds %s of %d, examination %d/%d, %.0f s" % (
            rows[-1]["lesson"], scores, len(asked), passed, len(exam), rows[-1]["seconds"]), flush=True)
    study_s = time.time() - t_all
    review, kept, lost = {}, 0, []
    for path in paths:                                       # m3_etut.review: every dialogue asked again
        _, asked, _ = lesson_lines(path)
        if not asked:
            continue
        said = m.answers([q for q, _ in asked], limit=40, workers=args.workers)
        right = 0
        for x, (q, a) in zip(said, asked):
            ok = judge(x, a)[0]
            right += ok
            if not ok and len(lost) < 30:
                lost.append({"q": q, "key": a, "said": x})
        review[os.path.splitext(os.path.basename(path))[0]] = [right, len(asked)]
        kept += right
    saved = None
    if args.save:
        m.flush()
        saved = sorted(m.save(args.save)["files"]) if args.who == "old" else m.save(args.save)
    seconds.sort()
    put(args, {"battery": "dialogues", "lessons_n": len(paths), "asked": asked_n, "first_round_right": first_right,
               "after_rounds_right": sum(r["rounds"][-1] for r in rows if r["rounds"]),
               "exam": [exam_right, exam_n], "kept_after_all": [kept, asked_n], "study_s": round(study_s, 1),
               "life_chars": chars, "converse_n": len(seconds),
               "converse_s": {"mean": round(sum(seconds) / max(1, len(seconds)), 4),
                              "median": seconds[len(seconds) // 2] if seconds else None,
                              "p95": seconds[int(len(seconds) * 0.95)] if seconds else None,
                              "max": seconds[-1] if seconds else None},
               "review": review, "stats": m.stats(), "saved": saved, "lessons": rows, "not_right": lost})


# --------------------------------------------------------------------------------------------------- scale -------
def text_of(name: str, chars: int) -> str:
    """A corpus stream of corpora.py; ``all``: the public corpora joined (en-repo, en-pydoc, tr-lessons,
    code-pystdlib), about 12 million characters; a path: that UTF-8 file."""
    import corpora
    if os.path.exists(name):
        with open(name, encoding="utf-8", errors="replace") as f:
            text = f.read(chars) if chars else f.read()
    elif name == "all":
        text = corpora.SEP.join(corpora.stream(k) for k in ("en-repo", "en-pydoc", "tr-lessons", "code-pystdlib"))
    else:
        text = corpora.stream(name)
    if chars and len(text) < chars:
        raise SystemExit("%s holds %d characters, fewer than --chars %d" % (name, len(text), chars))
    return text[:chars] if chars else text


def scale(args) -> None:
    import m3_scale
    text = text_of(args.corpus, args.chars)
    gc.collect()
    base = resident_mb()
    m = MINDS[args.who](variant="full", ledger=(args.ledger != "none"))
    t0 = time.time()
    step = max(1, len(text) // 10)
    bits = 0.0
    for i in range(0, len(text), step):
        bits += m.read(text[i:i + step]) or 0.0
    seconds = time.time() - t0
    m.flush()
    gc.collect()
    rss = resident_mb() - base
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    rss_trimmed = resident_mb() - base
    parts = m.walk()
    n = len(text)
    walked = sum(parts.values())
    put(args, {"battery": "scale", "corpus": os.path.basename(args.corpus), "chars": n, "ledger": args.ledger,
               "seconds": round(seconds, 2), "chars_per_second": round(n / max(seconds, 1e-9), 1),
               "bits_per_char": round(bits / n, 4) if bits else None,
               "resident_mb": round(rss, 1), "resident_trimmed_mb": round(rss_trimmed, 1),
               "walked_mb": round(walked / 1e6, 2), "walked_bytes_per_char": round(walked / n, 2),
               "resident_bytes_per_char": round(rss_trimmed * 1e6 / n, 2), "stats": m.stats(),
               "parts": [[k, v, round(v / n, 2)] for k, v in sorted(parts.items(), key=lambda x: -x[1])
                         if v >= 10000][:16]})


# ------------------------------------------------------------------------------------------------- heldout -------
def grade(said: str, spec: dict) -> str:
    mode = spec["mode"]
    if mode == "none":
        return "no key"
    if is_silent(said):
        return "right" if mode == "silent" else "silent"
    if mode == "silent":
        return "wrong"
    got = body(said)
    if mode == "exact":
        return "right" if any(got == body(k) for k in spec["keys"]) else "wrong"
    low = words(said)                                        # holds: a key stands in it, and none of the others
    for k in spec["keys"]:
        want = body(k)
        if any(low[i:i + len(want)] == want for i in range(len(low) - len(want) + 1)) and \
                not any(w in low for no in spec["not"] for w in body(no)):
            return "right"
    return "wrong"


def heldout(args) -> None:
    path = os.path.join(HERE, "heldout.json")
    with open(path, "rb") as f:
        blob = f.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != HELDOUT_SHA256:
        raise SystemExit("heldout.json is not the frozen set: %s" % digest)
    worlds = json.loads(blob.decode("utf-8"))["worlds"]
    print("== held-out set %s, answered by %s ==" % (digest, args.who), flush=True)
    for w in worlds:
        if args.world and w["name"] not in args.world:
            continue
        t0 = time.time()
        if w["name"] == "text":
            with open(os.path.join(REPO, w["path"]), "rb") as f:
                raw = f.read()
            if hashlib.sha256(raw).hexdigest() != w["sha256"]:
                raise SystemExit("%s is not the text frozen with the set" % w["path"])
            text = raw.decode("utf-8")
            m = MINDS[args.who](variant="full")
            bits = m.read(text)
            m.flush()
            print("-- text: %s, %d characters, %.4f bits a character, %.1f s" % (
                w["path"], len(text), bits / len(text), time.time() - t0), flush=True)
            put(args, {"battery": "heldout", "world": "text", "chars": len(text),
                       "bits_per_char": round(bits / len(text), 4), "seconds": round(time.time() - t0, 1)})
            continue
        if w["mind"] == "raised":
            if not args.raised:
                raise SystemExit("--raised FOLDER: the mind kept by the dialogues battery")
            m = MINDS[args.who].load(args.raised)
        else:
            m = MINDS[args.who](variant=w.get("variant", "full"), switches=w.get("switches"),
                                window=w.get("window", 0), ledger=bool(w.get("ledger")))
        print("-- %s (%d items lived, %d asked)" % (w["name"], len(w["life"]),
              len(w["ask"]) + sum(1 for it in w["life"] if it[0] == "ask")), flush=True)
        transcript, entry_of, seconds = [], {}, []
        for k, item in enumerate(w["life"]):
            if item[0] == "tell":
                m.tell(item[1])
            elif item[0] == "watch":
                m.watch(item[1], item[2])
            elif item[0] == "converse":
                said = m.converse(item[1])
                print("   said to it: %-40s it said: %s" % (item[1], said or "(sessiz)"), flush=True)
            elif item[0] == "forget":
                m.forget(entry_of[item[1]])
                print("   forgotten: item %d of its life" % item[1], flush=True)
                continue
            elif item[0] == "ask":
                spec = item[1]
                t1 = time.time()
                said = m.converse(spec["q"])
                seconds.append(time.time() - t1)
                transcript.append((spec, said))
            if getattr(m.c, "entries", None) is not None:
                entry_of[k] = len(m.c.entries) - 1
        m.close()
        if w["ask"]:
            t1 = time.time()
            said = m.answers([s["q"] for s in w["ask"]], limit=400, workers=args.workers)
            seconds += [(time.time() - t1) / len(w["ask"])] * len(w["ask"])
            transcript += list(zip(w["ask"], said))
        tags, total = {}, Counter()
        for spec, said in transcript:
            v = grade(said, spec)
            tags.setdefault(spec["tag"], Counter())[v] += 1
            total[v] += 1
            print("   [%-6s] %-34s Q: %s | said: %s | key: %s" % (
                v, spec["tag"][:34], spec["q"], said if not is_silent(said) else "(sessiz)",
                " / ".join(spec["keys"]) if spec["keys"] else ("(silence)" if spec["mode"] == "silent" else "-")),
                flush=True)
        put(args, {"battery": "heldout", "world": w["name"], "asked": len(transcript), "right": total["right"],
                   "silent": total["silent"], "wrong": total["wrong"], "no_key": total["no key"],
                   "tags": {t: dict(c) for t, c in tags.items()}, "seconds": round(time.time() - t0, 1),
                   "answer_s_mean": round(sum(seconds) / max(1, len(seconds)), 3),
                   "transcript": [{"q": s["q"], "said": x, "keys": s["keys"], "mode": s["mode"], "tag": s["tag"],
                                   "verdict": grade(x, s)} for s, x in transcript]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("battery", choices=["scan", "cogs", "school", "change", "dialogues", "scale", "heldout"])
    ap.add_argument("--who", choices=["old", "new"], required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "yardstick.jsonl"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan", default="", help="scan: SCAN in the published layout")
    ap.add_argument("--split", nargs="*", default=[])
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--test", type=int, default=50)
    ap.add_argument("--cogs", default="", help="cogs: the published data folder")
    ap.add_argument("--save", default="", help="dialogues: keep the raised mind here")
    ap.add_argument("--corpus", default="en-repo", help="scale: a corpus of corpora.py, 'all', or a text file")
    ap.add_argument("--chars", type=int, default=100000)
    ap.add_argument("--ledger", default="disk", choices=["disk", "none"])
    ap.add_argument("--raised", default="", help="heldout: the raised mind's folder")
    ap.add_argument("--world", nargs="*", default=[])
    a = ap.parse_args()
    if a.battery == "scan" and not a.scan:
        ap.error("scan needs --scan FOLDER (python scan_files.py FOLDER)")
    {"scan": scan, "cogs": cogs, "school": school, "change": change, "dialogues": dialogues, "scale": scale,
     "heldout": heldout}[a.battery](a)


if __name__ == "__main__":
    main()
