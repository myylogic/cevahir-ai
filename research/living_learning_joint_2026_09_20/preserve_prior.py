"""Preserve both preceding rounds of research before the joint-life round."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent/'results'/'prior_manifest.json'
TARGETS = ['docs/research/living_learning_2026_09_20', 'research/living_learning',
           'docs/research/living_learning_reassessment_2026_09_20',
           'research/living_learning_reassessment',
           'docs/research/RESEARCH_LEDGER.md', 'docs/research/FRONTIER_DISCOVERY_2026_09_20.md',
           'docs/research/REPRESENTATION_WITNESS_AUDIT.md',
           'docs/research/EXPERIENCE_CONDITIONED_COMPUTE.md', 'research/representation_discovery']

def fingerprint(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def main():
    if not OUT.exists():
        all_paths = []
        for name in TARGETS:
            path = ROOT/name
            if not path.exists():
                raise FileNotFoundError(path)
            all_paths.extend(path.rglob('*') if path.is_dir() else [path])
        records = {p.relative_to(ROOT).as_posix(): fingerprint(p) for p in sorted(set(all_paths))
                   if p.is_file() and '__pycache__' not in p.parts}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({'files': records}, indent=2)+'\n', encoding='utf-8')
    records = json.loads(OUT.read_text(encoding='utf-8'))['files']
    changed = [name for name, fp in records.items()
               if not (ROOT/name).is_file() or fingerprint(ROOT/name) != fp]
    result = {'preserved_files': len(records), 'changed': changed}
    print(json.dumps(result))
    assert not changed, changed

if __name__ == '__main__':
    main()
