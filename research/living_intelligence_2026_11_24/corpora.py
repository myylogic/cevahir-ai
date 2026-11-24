# -*- coding: utf-8 -*-
"""The texts the M1 measurements live on, built the same way every time.

- ``en-repo``: the repository's own English research text, read from commit 996cc2d (release v7 plus its DOI record), so
  later edits do not change it. Nine files, joined in a fixed order.
- ``en-pydoc``: the Python reference documentation bundled with Python 3.13 (``pydoc_data.topics``): 81 topics,
  sorted by name. English prose of a different author and subject.
- ``tr-lessons``: the public Turkish lessons the open mind lived (minds/sources.json): the first experiments' data,
  the seven school subjects of ``t_school.subjects(0)`` and the 24 lessons of ``dersler/``.
- ``tr-private``: the author's own Turkish documents (``education/ozel-dokumanlar`` in the author's checkout). They are
  read locally only, if present; nothing of them leaves the machine and only counts are reported.
- ``code-pystdlib``: the source of Python 3.13's standard library as installed with it (``/usr/lib/python3.13``,
  without its tests and third-party packages): 537 files, about 10 million characters, sorted by path. Code, the
  largest text on the machine.

Each corpus is a list of documents. The measurements join them with a blank line into one stream.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
COMMIT = "996cc2d"
EN_REPO_FILES = [
    "research/living_mind_2026_11_14/LAYERS.md",
    "research/living_mind_2026_11_14/NOTES.md",
    "research/living_mind_2026_11_14/RESULTS.md",
    "research/living_mind_2026_11_15/NOTES.md",
    "research/living_mind_2026_11_22/NOTES.md",
    "research/living_mind_2026_11_23/NOTES.md",
    "docs/ZIHIN_THEORY.md",
    "docs/research/living_evidence_mind_2026_11_24/PAPER.md",
    "README.md",
]
PRIVATE = os.environ.get("LIVING_PRIVATE_DOCS", "/workspace/cevahir-ai/education/ozel-dokumanlar")
SEP = "\n\n"


def en_repo() -> list[str]:
    return [subprocess.run(["git", "-C", REPO, "show", "%s:%s" % (COMMIT, path)], capture_output=True, check=True)
            .stdout.decode("utf-8") for path in EN_REPO_FILES]


def en_pydoc() -> list[str]:
    import pydoc_data.topics as topics
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("en-pydoc is defined on Python 3.13's pydoc_data")
    return [topics.topics[name] for name in sorted(topics.topics)]


def tr_lessons() -> list[str]:
    sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_22"))
    import t_school
    data = os.path.join(REPO, "research", "living_mind_2026_11_14", "data")
    docs = []
    for name in ("first_dialogues.tsv", "feedback_dialogues.tsv", "school_text.txt", "school_questions.tsv",
                 "imagination_text.txt"):
        with open(os.path.join(data, name), encoding="utf-8") as f:
            docs.append(f.read())
    for name, (lessons, _) in sorted(t_school.subjects(0).items()):        # as minds/raise_open.py writes them
        lines = ["# %s (research/living_mind_2026_11_22/t_school.py, subjects(0))" % name]
        for lesson in lessons:
            lines.append(lesson[1] if lesson[0] == "tell" else lesson[1] + "\t" + lesson[2])
        docs.append("\n".join(lines) + "\n")
    for path in sorted(glob.glob(os.path.join(REPO, "research", "living_mind_2026_11_22", "dersler", "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def tr_private() -> list[str]:
    """The author's documents, if this machine has them; read in memory only."""
    if not os.path.isdir(PRIVATE):
        return []
    docs = []
    for path in sorted(glob.glob(os.path.join(PRIVATE, "*"))):
        if path.endswith(".docx"):
            import docx
            text = "\n".join(p.text for p in docx.Document(path).paragraphs)
        elif path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                text = json.dumps(json.load(f), ensure_ascii=False, indent=0)
        elif path.endswith(".txt"):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            continue
        if text.strip():
            docs.append(text)
    return docs


def code_pystdlib() -> list[str]:
    root = "/usr/lib/python3.13"
    if sys.version_info[:2] != (3, 13) or not os.path.isdir(root):
        raise RuntimeError("code-pystdlib is defined on Python 3.13's installed standard library")
    paths = []
    for d, dirs, files in os.walk(root):
        parts = set(d[len(root):].split(os.sep))
        if parts & {"test", "tests", "site-packages", "dist-packages"}:
            continue
        paths += [os.path.join(d, f) for f in files if f.endswith(".py")]
    docs = []
    for path in sorted(paths):
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            docs.append(f.read())
    return docs


CORPORA = {"en-repo": en_repo, "en-pydoc": en_pydoc, "tr-lessons": tr_lessons, "tr-private": tr_private,
           "code-pystdlib": code_pystdlib}
PUBLIC = ("en-repo", "en-pydoc", "tr-lessons", "code-pystdlib")


def load(name: str) -> list[str]:
    return CORPORA[name]()


def stream(name: str) -> str:
    return SEP.join(load(name))


def say_positions(s: str, split: int, n: int = 500, before: int = 256, after: int = 40) -> list[int]:
    """Where the continuation test asks a mind to go on: n places spread evenly over the last tenth, each with
    ``before`` characters of the test text behind it and ``after`` ahead."""
    lo, hi = split + before, len(s) - after
    if hi <= lo:
        return []
    step = (hi - lo) / n
    return sorted({lo + int(i * step) for i in range(n)})


def describe(name: str) -> dict:
    docs = load(name)
    s = SEP.join(docs)
    out = {"corpus": name, "documents": len(docs), "chars": len(s), "utf8_bytes": len(s.encode("utf-8")),
           "distinct_symbols": len(set(s))}
    if name in PUBLIC:
        out["sha256"] = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return out


if __name__ == "__main__":
    for name in CORPORA:
        print(json.dumps(describe(name)))
