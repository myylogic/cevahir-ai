"""Count the manuscript and build its detailed reading index.

No semantic completeness is inferred from counts. Default mode checks the stored
inventory and generated index; --write explicitly records the current manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from check_book import outside_fences

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs/book"
OUTPUT = BOOK / "evidence/manuscript_inventory.json"
INDEX = BOOK / "AYRINTILI_ICINDEKILER.md"


def inventory():
    chapters = []
    index = ["# Ayrıntılı içindekiler", "", "[Kitap girişi](README.md) · [Terimler](TERIMLER.md) · [Tamamlanma kapsamı](TAMAMLANMA.md)", "",
             "Bölüm içindeki ikinci ve üçüncü düzey başlıkların otomatik dizinidir. Başlık sayısı, bağımsız bilimsel kavram sayısı değildir. Alıştırma ve kaynakça başlıkları da sayılır.", ""]
    for path in sorted((BOOK / "tr").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        headings = []
        seen = {}
        for line_number, line in outside_fences(text):
            match = re.match(r"^(#{1,3})\s+(.+?)\s*#*\s*$", line)
            if not match:
                continue
            level, title = len(match[1]), match[2]
            slug = "".join(c for c in re.sub(r"<[^>]*>", "", title).lower() if c.isalnum() or c in " _-").replace(" ", "-")
            number = seen.get(slug, 0)
            seen[slug] = number + 1
            anchor = slug + (f"-{number}" if number else "")
            headings.append(dict(level=level, title=title, line=line_number, anchor=anchor))
        exercise_match = re.search(r"^## [^\n]*\balıştırma[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S | re.I)
        if not exercise_match:
            raise ValueError(f"Missing exercise section: {path}")
        exercises = exercise_match[1].split("### Yanıtlar")[0]
        solved_count = len(re.findall(r"^(?:### (?:Alıştırma )?\d+|\d+\.\s|\*\*\d+\.)", exercises, re.M))
        chapter = dict(file=path.relative_to(BOOK).as_posix(), title=headings[0]["title"],
                       whitespace_words_including_code_tables_math=len(text.split()),
                       topics_h2=sum(h["level"] == 2 for h in headings),
                       subtopics_h3=sum(h["level"] == 3 for h in headings), solved_exercises=solved_count,
                       sha256_utf8_lf=hashlib.sha256(text.encode()).hexdigest(), headings=headings)
        chapters.append(chapter)
        index += [f"## [{chapter['title']}]({chapter['file']})", ""]
        for h in headings[1:]:
            indent = "  " if h["level"] == 3 else ""
            index.append(f"{indent}- [{h['title']}]({chapter['file']}#{h['anchor']})")
        index.append("")
    glossary = (BOOK / "TERIMLER.md").read_text(encoding="utf-8")
    # Only the four term tables, not the mathematical-symbol table.
    glossary_body = glossary.split("## Matematiksel")[0]
    term_rows = [s for s in glossary_body.splitlines() if s.startswith("| ") and "](tr/" in s]
    totals = {key: sum(c[key] for c in chapters) for key in (
        "whitespace_words_including_code_tables_math", "topics_h2", "subtopics_h3", "solved_exercises")}
    data = {"counting_method": "12 Turkish chapter files; whitespace words include code, equations, tables and links. H2/H3 include exercise and bibliography headings. Solved exercises are numbered questions in the designated exercise section, counted once.",
            "chapters_count": len(chapters), "totals": totals, "glossary_term_rows": len(term_rows), "chapters": chapters}
    return data, "\n".join(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data, index = inventory()
    if args.write:
        OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        INDEX.write_text(index, encoding="utf-8", newline="\n")
    else:
        if data != json.loads(OUTPUT.read_text(encoding="utf-8")) or index != INDEX.read_text(encoding="utf-8"):
            raise SystemExit("Inventory or detailed index changed; review the manuscript before --write.")
    print(json.dumps({"status": "passed", "mode": "record" if args.write else "check",
                      "chapters": data["chapters_count"], **data["totals"], "glossary_terms": data["glossary_term_rows"]}))


if __name__ == "__main__":
    main()
