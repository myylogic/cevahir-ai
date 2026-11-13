"""Read-only checks of publication editions, provenance, local links and metadata."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import build_publications

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "docs/publications"


def _require(condition, message):
    """Fail the check with a message. Unlike ``assert`` this is not removed by ``python -O``."""
    if not condition:
        raise SystemExit(f"publication check failed: {message}")


def main():
    catalog = json.loads((PUB / "catalog.json").read_text(encoding="utf-8"))
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    release = json.loads((PUB / "release.json").read_text(encoding="utf-8"))
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    _require(catalog["author"] == release["author"] == "Muhammed Yasin Yılmaz", "catalog/release author")
    _require(zenodo["creators"] == [{"name": "Yılmaz, Muhammed Yasin"}], ".zenodo.json creators")
    _require(catalog["version"] == zenodo["version"] == release["version"], "catalog/zenodo/release version")
    _require(f'version: "{catalog["version"]}"' in cff, "CITATION.cff version")
    _require('family-names: "Yılmaz"' in cff and 'given-names: "Muhammed Yasin"' in cff, "CITATION.cff author")
    _require(len({p["id"] for p in catalog["papers"]}) == len(catalog["papers"]), "duplicate paper id")
    for paper in catalog["papers"]:
        actual = (PUB / "papers" / paper["file"]).read_text(encoding="utf-8")
        _require(actual == build_publications.edition(paper, catalog), f"edition differs from its build: {paper['id']}")
    prior = json.loads((PUB / "evidence/prior_manifest.json").read_text(encoding="utf-8"))
    for path, sha in prior["files"].items():
        _require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, f"preserved file changed: {path}")
    for path, sha in prior.get("newline_normalized_files", {}).items():
        normalized = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        _require(hashlib.sha256(normalized).hexdigest() == sha, f"preserved file changed: {path}")
    result = json.loads((ROOT / "research/living_learning_query_state_2026_09_21/results/query_state.json").read_text())
    source = ROOT / "research/living_learning_query_state_2026_09_21/query_state.py"
    _require(result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest(), "query_state source digest")
    validation = json.loads((PUB / "evidence/validation.json").read_text(encoding="utf-8"))
    _require(validation["prior_files_changed"] == [], "validation.json lists changed prior files")
    _require(validation["query_state_exact_rerun_equal"] and validation["correction_exact_rerun_equal"],
             "validation.json exact reruns")
    links = 0
    for path in PUB.rglob("*.md"):
        fence = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                fence = not fence
                continue
            if fence:
                continue
            for target in re.findall(r"\]\(([^\s)]+)\)", line):
                split = urlsplit(target)
                if split.scheme or target.startswith("#"):
                    continue
                resolved = (path.parent / unquote(split.path)).resolve()
                _require(resolved.is_relative_to(ROOT) and resolved.exists(), f"broken local link in {path}: {target}")
                links += 1
    if release["doi"]:
        _require(release["status"] == "published", "release with a DOI must be published")
        _require(re.fullmatch(r"10\.5281/zenodo\.\d+", release["doi"]), "release DOI format")
        _require(f'doi: {release["doi"]}' in cff, "CITATION.cff DOI")
    print(json.dumps({"status": "passed", "papers": len(catalog["papers"]),
                      "preserved_prior_files": len(prior["files"]), "local_links": links,
                      "doi_status": release["status"]}))


if __name__ == "__main__":
    main()
