"""Append-only research-round guard; do not rewrite older results."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent/'results'/'prior_manifest.json'
TARGETS = ['docs/research/living_learning_2026_09_20', 'research/living_learning',
           'docs/research/living_learning_reassessment_2026_09_20', 'research/living_learning_reassessment',
           'docs/research/living_learning_joint_2026_09_20', 'research/living_learning_joint_2026_09_20',
           'docs/research/living_learning_update_2026_09_20', 'research/living_learning_update_2026_09_20',
           'docs/research/RESEARCH_LEDGER.md', 'docs/research/FRONTIER_DISCOVERY_2026_09_20.md',
           'docs/research/REPRESENTATION_WITNESS_AUDIT.md', 'docs/research/EXPERIENCE_CONDITIONED_COMPUTE.md',
           'research/representation_discovery']


def fingerprint(p):
    data = p.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    if not OUT.exists():
        files = []
        for name in TARGETS:
            p = ROOT/name
            assert p.exists(), p
            files.extend(p.rglob('*') if p.is_dir() else [p])
        manifest = {p.relative_to(ROOT).as_posix(): fingerprint(p) for p in sorted(set(files))
                    if p.is_file() and '__pycache__' not in p.parts}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({'files': manifest}, indent=2)+'\n', encoding='utf-8')
    manifest = json.loads(OUT.read_text(encoding='utf-8'))['files']
    changed = [name for name, record in manifest.items() if not (ROOT/name).exists() or fingerprint(ROOT/name) != record]
    assert not changed, changed
    print(json.dumps({'preserved_files': len(manifest), 'changed': changed}))


if __name__ == '__main__':
    main()
