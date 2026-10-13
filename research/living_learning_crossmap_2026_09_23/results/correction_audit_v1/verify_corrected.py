from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tempfile
from fractions import Fraction
from pathlib import Path
import corrected as c
import analyze_corrected as a
import tree_audit


def preserves():
    baseline = json.loads((c.RESULTS / 'preservation_baseline.json').read_text())
    for name, expected in baseline['files'].items():
        assert c.sha(c.ROOT / name) == expected, name
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT).decode().strip() == baseline['base_head']
    return {'files': len(baseline['files']), 'all_unchanged': True}


def normalize_world(w):
    w = json.loads(json.dumps(w))
    for name in ('x_next', 'y_next', 'x_group', 'y_group', 'true_map'):
        w[name] = {int(k): int(v) for k, v in w[name].items()}
    return w


def inspect(data):
    answers = records = continuation = 0
    for run in data['runs']:
        w = normalize_world(run['world']); obs = c.observation(w)
        assert len(obs) == 24
        assert len(set(tuple(x) for x in w['bridges'])) == 6
        for method, track in run['tracks'].items():
            learner = c.Learner(method, obs, [w['true_map'][i] for i in range(6)] if method == 'oracle' else None)
            # Reconstruct packet by packet; targets are calculated only here.
            for row, packet in zip(track['records'], c.packets(w)):
                got = c.step(learner, packet)
                q = packet['query']; target = c.target(w, q)
                assert got['response']['counts'] == row['engine']['response']['counts']
                assert got['response']['prediction'] == row['engine']['response']['prediction']
                assert row['target'] == target
                assert row['correct'] == (row['engine']['response']['prediction'] == target)
                # No target value occurs in the learner state or packet.
                assert 'target' not in encode_forbidden(learner.__dict__)
                answers += 1
            assert learner.snapshot() == track['final']
            assert learner.local_probe() == track['final_local_probe']
            # Local source competence must be retained for all methods.
            assert learner.local_probe()['x_next'] == [w['x_next'][i] for i in range(6)]
            assert learner.local_probe()['y_next'] == [w['y_next'][i] for i in range(6)]
            records += len(track['records'])
        # Exact posterior/candidate and output equality across persistent/recompute/completion.
        def dist(counts):
            if counts is None: return None
            total = sum(counts)
            return tuple(Fraction(x, total) for x in counts)
        for left, right in [('persistent', 'recompute'), ('persistent', 'completion'), ('recompute', 'completion')]:
            for l, r in zip(run['tracks'][left]['records'], run['tracks'][right]['records']):
                assert dist(l['engine']['response']['counts']) == dist(r['engine']['response']['counts'])
        # By phase 0/1/2 uncertainty must remain; after five bridge observations maps are unique.
        candidates = [r['engine']['response']['candidate_count'] for r in run['tracks']['persistent']['records']]
        assert candidates[:6] == [24] * 6 and candidates[6:12] == [6] * 6
        assert candidates[12:18] == [2] * 6 and candidates[18:24] == [1] * 6
        continuation += 1
    return {'answers_replayed': answers, 'records_replayed': records, 'exact_distribution_pairs': continuation * 24 * 3,
            'target_accessed_only_in_verifier': True}


def encode_forbidden(obj):
    # Learner's state may contain source records, never evaluator target.
    return json.dumps(obj, sort_keys=True)


def resume_checks(data):
    checks=[]
    with tempfile.TemporaryDirectory(prefix='cevahir_crossmap_') as folder:
        folder=Path(folder)
        for run in data['runs'][:3]:
            w=normalize_world(run['world']); packets=c.packets(w)
            for method in c.METHODS:
                for cut in (6,12,18,24):
                    learner=c.Learner(method,c.observation(w),[w['true_map'][i] for i in range(6)] if method=='oracle' else None)
                    for packet in packets[:cut]: c.step(learner,packet)
                    src=folder/'in.json'; dst=folder/'out.json'
                    src.write_text(c.encode({'state':learner.snapshot(),'tail':packets[cut:]}))
                    expected=[c.step(learner,p) for p in packets[cut:]]
                    got=subprocess.run([sys.executable,str(c.HERE/'corrected.py'),'--resume',str(src),str(dst)],capture_output=True,text=True,check=True)
                    actual=json.loads(dst.read_text())
                    assert actual=={'records':expected,'final':learner.snapshot()}
                    checks.append({'method':method,'seed':run['seed'],'cut':cut,'exact':True})
    return checks


def main():
    p=argparse.ArgumentParser(); p.add_argument('--rerun',action='store_true'); args=p.parse_args()
    data=json.loads((c.RESULTS/'crossmap_corrected.json').read_text())
    assert data['source_sha256']==c.sha(c.HERE/'corrected.py')
    assert data['amendment_sha256']==c.sha(c.DOCS/'AMENDMENT.md')
    assert data['archived_generator_sha256']==c.sha(c.RESULTS/'invalid_initial_run/crossmap.py')
    assert data['methods']==c.METHODS and len(data['runs'])==24
    recorded=inspect(data); resumed=resume_checks(data); tree=tree_audit.audit()
    assert len(resumed)==3*len(c.METHODS)*4 and all(x['exact'] for x in resumed)
    result={'recorded':recorded,'fresh_process_resume':resumed,'tree_audit':tree,'preservation':preserves()}
    if args.rerun:
        proc=subprocess.run([sys.executable,str(c.HERE/'corrected.py')],capture_output=True,text=True)
        assert proc.returncode==0 and proc.stdout.strip()=='exact_rerun_passed'; result['full_rerun']=True
    (c.RESULTS/'verification_corrected.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
