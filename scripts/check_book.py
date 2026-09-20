"""Check book links, source symbols, and explicitly reviewed file fingerprints.

Standard library only; no imports from application code and no network access.
Refreshing fingerprints is an explicit acknowledgement of a manual review,
not an automatic repair of documentation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs/book"
SNAPSHOT = BOOK / "evidence/reviewed_sources.json"
LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def read(path):
    return path.read_text(encoding="utf-8-sig")


def fingerprint(path):
    # Git may check out CRLF on Windows and LF on CI. Compare textual content,
    # not a platform's line-ending convention. Binary links keep byte identity.
    textual = path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".txt"}
    payload = read(path).encode("utf-8") if textual else path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(),
            "fingerprint_mode": "utf8-lf" if textual else "bytes"}


def outside_fences(source):
    marker = None
    for number, line in enumerate(source.splitlines(), 1):
        match = FENCE.match(line)
        if match:
            char = match[1][0]
            if marker is None:
                marker = char
            elif marker == char:
                marker = None
            continue
        if marker is None:
            yield number, line


def heading_anchors(source):
    counts, anchors = {}, set()
    for _, line in outside_fences(source):
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        heading = re.sub(r"<[^>]*>", "", match[1]).lower()
        slug = "".join(c for c in heading if c.isalnum() or c in " _-").replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(slug + (f"-{count}" if count else ""))
    return anchors


def python_symbols(path):
    found = {}
    tree = ast.parse(read(path), filename=str(path))

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                found[name] = child.lineno
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree)
    return found


def local_target(markdown, destination, root=ROOT):
    destination = destination.strip("<>")
    url = urlsplit(destination)
    if url.scheme or url.netloc:
        return None, None
    path = (markdown.parent / unquote(url.path)).resolve() if url.path else markdown.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"link leaves repository: {destination}")
    if not path.exists():
        raise ValueError(f"missing link target: {destination}")
    fragment = unquote(url.fragment)
    if fragment:
        line = re.fullmatch(r"L([1-9][0-9]*)(?:-L([1-9][0-9]*))?", fragment)
        if line:
            count = len(read(path).splitlines())
            start, end = int(line[1]), int(line[2] or line[1])
            if not 1 <= start <= end <= count:
                raise ValueError(f"invalid source line: {destination} ({count} lines)")
        elif path.suffix.lower() == ".md":
            if fragment not in heading_anchors(read(path)):
                raise ValueError(f"missing Markdown heading: {destination}")
        else:
            raise ValueError(f"unsupported local fragment: {destination}")
    return path, fragment


def inspect_book():
    errors, references, dependencies = [], [], {}
    markdowns = sorted(BOOK.rglob("*.md"))
    if not markdowns:
        errors.append("No book Markdown files found")

    def track(path, chapter):
        if path.is_file() and not path.is_relative_to(BOOK):
            dependencies.setdefault(path.relative_to(ROOT).as_posix(), set()).add(chapter)

    link_count = 0
    for markdown in markdowns:
        chapter = markdown.relative_to(BOOK).as_posix()
        for number, line in outside_fences(read(markdown)):
            for match in LINK.finditer(line):
                try:
                    target, _ = local_target(markdown, match[1])
                    if target is not None:
                        link_count += 1
                        track(target, chapter)
                except (ValueError, OSError) as error:
                    errors.append(f"{chapter}:{number}: {error}")

    for manifest in sorted((BOOK / "evidence").glob("*.json")):
        if manifest == SNAPSHOT:
            continue
        try:
            data = json.loads(read(manifest))
            if isinstance(data, dict):
                references.extend(data.get("references", []))
        except (ValueError, OSError) as error:
            errors.append(f"{manifest.name}: {error}")

    parsed = {}
    for ref in references:
        try:
            source = (ROOT / ref["path"]).resolve()
            chapter = (BOOK / ref["chapter"]).resolve()
            if not source.is_relative_to(ROOT) or not chapter.is_relative_to(BOOK):
                raise ValueError("reference leaves its allowed root")
            if not chapter.is_file():
                raise ValueError(f"missing chapter: {ref['chapter']}")
            if source not in parsed:
                parsed[source] = python_symbols(source)
            symbol = ref["symbol"]
            if symbol not in parsed[source]:
                raise ValueError(f"missing Python symbol: {symbol}")
            if "line" in ref and ref["line"] != parsed[source][symbol]:
                raise ValueError(f"{symbol} moved: recorded {ref['line']}, actual {parsed[source][symbol]}")
            track(source, ref["chapter"])
        except (ValueError, OSError, KeyError, SyntaxError) as error:
            errors.append(f"{ref}: {error}")

    registered = {ref.get("chapter") for ref in references}
    for chapter in sorted((BOOK / "tr").glob("*.md")):
        if chapter.relative_to(BOOK).as_posix() not in registered:
            errors.append(f"No symbol evidence for chapter {chapter.name}")
    for name in ("README.md", "README-TR.md"):
        if "docs/book/README" not in read(ROOT / name):
            errors.append(f"{name}: missing prominent book entry")

    current = {path: {**fingerprint(ROOT / path),
                      "chapters": sorted(chapters)}
               for path, chapters in sorted(dependencies.items())}
    summary = {"markdown_files": len(markdowns), "local_links": link_count,
               "symbol_references": len(references), "referenced_files": len(current)}
    return errors, current, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-reviewed-sources", action="store_true",
                        help="After manual review only: renew the reviewed-source snapshot")
    args = parser.parse_args()
    errors, current, summary = inspect_book()
    if not errors and args.record_reviewed_sources:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps({"format_version": 1, "files": current},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif not errors:
        if not SNAPSHOT.exists():
            errors.append("Reviewed-source snapshot missing; manually review before recording it")
        else:
            prior = json.loads(read(SNAPSHOT))["files"]
            for path in sorted(prior.keys() | current.keys()):
                if prior.get(path) != current.get(path):
                    affected = (current.get(path) or prior[path])["chapters"]
                    errors.append(f"Review required: {path} -> {', '.join(affected)}")
    if errors:
        print("BOOK CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print(json.dumps({"status": "passed", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
