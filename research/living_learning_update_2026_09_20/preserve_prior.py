"""Protect all preceding research rounds; only new round files may change."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent/'results'/'prior_manifest.json'
TARGETS=['docs/research/living_learning_2026_09_20','research/living_learning',
         'docs/research/living_learning_reassessment_2026_09_20','research/living_learning_reassessment',
         'docs/research/living_learning_joint_2026_09_20','research/living_learning_joint_2026_09_20',
         'docs/research/RESEARCH_LEDGER.md','docs/research/FRONTIER_DISCOVERY_2026_09_20.md',
         'docs/research/REPRESENTATION_WITNESS_AUDIT.md','docs/research/EXPERIENCE_CONDITIONED_COMPUTE.md',
         'research/representation_discovery']

def fp(path):
    b=path.read_bytes()
    return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def main():
    if not OUT.exists():
        files=[]
        for name in TARGETS:
            p=ROOT/name
            if not p.exists(): raise FileNotFoundError(p)
            files.extend(p.rglob('*') if p.is_dir() else [p])
        records={p.relative_to(ROOT).as_posix():fp(p) for p in sorted(set(files)) if p.is_file() and '__pycache__' not in p.parts}
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps({'files':records},indent=2)+'\n',encoding='utf-8')
    records=json.loads(OUT.read_text(encoding='utf-8'))['files']
    changed=[name for name,record in records.items() if not (ROOT/name).is_file() or fp(ROOT/name)!=record]
    print(json.dumps({'preserved_files':len(records),'changed':changed}))
    assert not changed

if __name__=='__main__': main()
