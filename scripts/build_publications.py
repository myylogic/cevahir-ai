"""Build or check publication editions without changing original research records."""
import argparse
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "docs/publications"


# A document withdrawn by its author (release v7) is replaced by this note, which names the release holding its full
# text and the English paper describing the work; check_publications.py requires each withdrawn preserved file to be
# exactly its note.
WITHDRAWN_PAPER = "docs/research/living_evidence_mind_2026_11_24/PAPER.md"
WITHDRAWN_ARCHIVE_TAG = "research-2026.11.16-v6"
WITHDRAWN_ARCHIVE_DOI = "10.5281/zenodo.24279940"
WITHDRAWN_NOTE = """# Withdrawn

This document was withdrawn from the repository on 20 November 2026 by its author. Its full text is archived in
release v6 (tag `{tag}`, the same path), DOI
[{doi}](https://doi.org/{doi}).

The work is described in English in [Cevahir Zihin: a Living Evidence Mind]({link}).
"""


def withdrawal_note(path):
    """The note for a repository path (POSIX, relative to the repository root)."""
    folder = os.path.dirname(path) or "."
    link = Path(os.path.relpath(WITHDRAWN_PAPER, folder)).as_posix()
    return WITHDRAWN_NOTE.format(tag=WITHDRAWN_ARCHIVE_TAG, doi=WITHDRAWN_ARCHIVE_DOI, link=link)


def relative(source, destination):
    return Path(os.path.relpath(source, destination)).as_posix()


def edition(paper, catalog):
    source = ROOT / paper["source"]
    parent = PUB / "papers"
    text = source.read_text(encoding="utf-8")
    def rebase(match):
        label, link = match.groups()
        if urlsplit(link).scheme or link.startswith(("#", "/")):
            return match.group(0)
        path, marker, anchor = link.partition("#")
        target = relative((source.parent / path).resolve(), parent)
        return f"[{label}]({target}{marker}{anchor})"
    text = re.sub(r"\[([^\]\n]+)\]\(([^\s)]+)\)", rebase, text)
    text = re.sub(r"\A# ([^\n]+)", r"## Research record: \1", text)
    return f'''# {paper["title"]}

**{catalog["author"]}**

{paper["id"]} · Version {catalog["version"]} · Published {catalog["date"]}

Research report / preprint. Not peer reviewed.

## Abstract

{paper["abstract"]}

## Interpretation of the findings

{paper["finding"]}

{paper["limit"]}

## Publication and evidence

This edition presents the research record below with its method, results, negative controls and source links, or
where that record was withdrawn, the note that says where its full text is archived. The original file is kept as
[{paper["source"]}]({relative(source, parent)}); links are rebased to this edition's location. Dates and earlier
publication statuses inside the record belong to the historical record; this edition's date is above.

This work is a separate text in the [publication series](../README.md). The DOI identifies the release archive that
holds every text and the code; it does not mean that each paper has a DOI of its own. Cite the work's identifier
together with the release DOI. The scope of the computational checks is described in the
[validation record](../evidence/validation.json) and the [publishing principles](../PUBLISHING.md).

The project and its research direction are Muhammed Yasin Yılmaz's. AI assistance was used for code, computational
experiments, literature review and text preparation. Neither this statement, the DOI record nor the automated tests
are external scientific peer review.

---

{text.rstrip()}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    catalog = json.loads((PUB / "catalog.json").read_text(encoding="utf-8"))
    for paper in catalog["papers"]:
        path = PUB / "papers" / paper["file"]
        expected = edition(paper, catalog)
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8", newline="\n")
        elif path.read_text(encoding="utf-8") != expected:
            # Not an assert: ``python -O`` removes asserts, and the check would then pass on anything.
            raise SystemExit(f"publication edition differs from its build: {path} (run with --write)")
    print(json.dumps({"publication_editions": len(catalog["papers"]), "mode": "write" if args.write else "check"}))


if __name__ == "__main__":
    main()
