"""Read-only checks of publication editions, provenance, local links and metadata."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import build_publications

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "docs/publications"


def main():
    catalog = json.loads((PUB / "catalog.json").read_text(encoding="utf-8"))
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    release = json.loads((PUB / "release.json").read_text(encoding="utf-8"))
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert catalog["author"] == release["author"] == "Muhammed Yasin Yılmaz"
    assert zenodo["creators"] == [{"name": "Yılmaz, Muhammed Yasin"}]
    assert catalog["version"] == zenodo["version"] == release["version"]
    assert f'version: "{catalog["version"]}"' in cff
    assert 'family-names: "Yılmaz"' in cff and 'given-names: "Muhammed Yasin"' in cff
    assert len({p["id"] for p in catalog["papers"]}) == len(catalog["papers"])
    for paper in catalog["papers"]:
        actual = (PUB / "papers" / paper["file"]).read_text(encoding="utf-8")
        assert actual == build_publications.edition(paper, catalog), paper["id"]
    prior = json.loads((PUB / "evidence/prior_manifest.json").read_text(encoding="utf-8"))
    for path, sha in prior["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path
    for path, sha in prior.get("newline_normalized_files", {}).items():
        normalized = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(normalized).hexdigest() == sha, path
    result = json.loads((ROOT / "research/living_learning_query_state_2026_09_21/results/query_state.json").read_text())
    source = ROOT / "research/living_learning_query_state_2026_09_21/query_state.py"
    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    validation = json.loads((PUB / "evidence/validation.json").read_text(encoding="utf-8"))
    assert validation["prior_files_changed"] == []
    assert validation["query_state_exact_rerun_equal"] and validation["correction_exact_rerun_equal"]
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
                assert resolved.is_relative_to(ROOT) and resolved.exists(), (path, target)
                links += 1
    if release["doi"]:
        assert release["status"] == "published"
        assert re.fullmatch(r"10\.5281/zenodo\.\d+", release["doi"])
        assert f'doi: {release["doi"]}' in cff
    print(json.dumps({"status": "passed", "papers": len(catalog["papers"]),
                      "preserved_prior_files": len(prior["files"]), "local_links": links,
                      "doi_status": release["status"]}))


if __name__ == "__main__":
    main()
