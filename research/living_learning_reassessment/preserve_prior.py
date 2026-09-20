"""Record old research bytes once; subsequent calls only verify them."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).parent / 'results' / 'prior_preservation_manifest.json'
TARGETS = (
    'docs/research/living_learning_2026_09_20',
    'research/living_learning',
    'docs/research/RESEARCH_LEDGER.md',
    'docs/research/FRONTIER_DISCOVERY_2026_09_20.md',
    'docs/research/REPRESENTATION_WITNESS_AUDIT.md',
    'docs/research/EXPERIENCE_CONDITIONED_COMPUTE.md',
    'research/representation_discovery',
)

def fingerprint(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def main():
    if not DEST.exists():
        paths = []
        for name in TARGETS:
            p = ROOT / name
            if not p.exists():
                raise FileNotFoundError(p)
            paths.extend(p.rglob('*') if p.is_dir() else [p])
        entries = {p.relative_to(ROOT).as_posix(): fingerprint(p)
                   for p in sorted(set(paths))
                   if p.is_file() and '__pycache__' not in p.parts}
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps({'purpose': 'Preserve pre-reassessment research without alteration',
                                    'files': entries}, indent=2) + '\n', encoding='utf-8')
    manifest = json.loads(DEST.read_text(encoding='utf-8'))
    changed = [name for name, record in manifest['files'].items()
               if not (ROOT / name).is_file() or fingerprint(ROOT / name) != record]
    print(json.dumps({'preserved_files': len(manifest['files']), 'changed': changed}))
    if changed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
