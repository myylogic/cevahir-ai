"""Frozen, local, paired research campaign. No publication or model integration."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCS = ROOT / 'docs/research/living_learning_campaign_2026_09_24'
RESULTS = HERE / 'results'
MODULES = ('memory', 'representation', 'update', 'environment')
FAMILIES = ('MEM', 'CON', 'RES', 'REP', 'XFR', 'LAN', 'UPD', 'CRD', 'CTX', 'ENV')
SEEDS = {'discovery': list(range(24)), 'confirmation': list(range(1000, 1048))}
LABELS = {'supported': 'Desteklendi', 'contradicted': 'Ters sonuç', 'inconclusive': 'Belirsiz'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_registry():
    rows = []
    for name in MODULES:
        module = importlib.import_module(name)
        for raw in module.hypotheses():
            row = dict(raw, engine=name)
            required = ('id', 'family', 'title_tr', 'prediction_tr', 'intervention_tr',
                        'falsifier_tr', 'status', 'prior_art')
            assert all(k in row and row[k] for k in required), row
            assert row['status'] in ('wave1', 'queued')
            if row['status'] == 'wave1':
                assert row['direction'] == 'lower'
                assert set(row['contrast']) == {'candidate', 'control'}
                assert row['contrast']['candidate'] != row['contrast']['control']
                assert row['metric'] and row['regime']
                assert math.isfinite(row['practical_margin']) and row['practical_margin'] >= 0
            rows.append(row)
    assert len(rows) == len({r['id'] for r in rows}) == 200
    assert Counter(r['family'] for r in rows) == Counter({k: 20 for k in FAMILIES})
    assert Counter(r['engine'] for r in rows if r['status'] == 'wave1') == Counter({k: 12 for k in MODULES})
    return sorted(rows, key=lambda h: h['id'])


def sign_tail(wins, losses):
    """Exact one-sided P[Binomial(non-ties, .5) >= wins]."""
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def holm(pvalues):
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted, previous = {}, 0.0
    for rank, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = previous
    return adjusted


def stats_selfcheck():
    checks = 0
    for n in range(9):
        outcomes = list(itertools.product((0, 1), repeat=n))
        for wins in range(n + 1):
            reference = sum(sum(o) >= wins for o in outcomes) / len(outcomes)
            assert sign_tail(wins, n - wins) == reference
            checks += 1
    p = {'a': .01, 'b': .04, 'c': .03, 'd': .9}
    got = holm(p)
    expected = {'a': .04, 'b': .09, 'c': .09, 'd': .9}
    for k in p:
        assert abs(got[k] - expected[k]) < 1e-14
        checks += 1
    # Independent step-down rejection agrees with adjusted p threshold.
    for alpha in (.01, .04, .05, .09, .5, .95):
        rejection = set()
        for rank, k in enumerate(sorted(p, key=p.get)):
            if p[k] <= alpha / (len(p) - rank):
                rejection.add(k)
            else:
                break
        assert rejection == {k for k, v in got.items() if v <= alpha}
        checks += 1
    assert sign_tail(0, 0) == 1 and sign_tail(48, 0) == 2 ** -48
    return {'passed': True, 'checks': checks + 1,
            'details': 'Exhaustive binary sign outcomes n<=8; hand Holm values; independent step-down rejection.'}


def render_catalog(rows):
    lines = ['# Tur 23 — 200 araştırma önermesi', '',
             '48 madde ilk dalgada çalıştırılabilir; 152 madde bekliyor. Tohum tekrarları hipotez değildir.',
             'Aynı mekanizmanın farklı rejimlerdeki koşullu iddiaları ayrı satırdır; 200 bağımsız buluş iddiası yoktur.', '']
    for family in FAMILIES:
        lines += ['## ' + family, '']
        for h in rows:
            if h['family'] != family:
                continue
            lines += [f"### {h['id']} — {h['title_tr']}", '',
                      f"Durum: **{'İlk dalga' if h['status'] == 'wave1' else 'Henüz sınanmadı'}**.", '',
                      f"Öngörü: {h['prediction_tr']}", '',
                      f"Müdahale: {h['intervention_tr']}", '',
                      f"Yanlışlama: {h['falsifier_tr']}", '',
                      f"Bilinen aile / önceki çalışma: {h['prior_art']}", '']
            if h['status'] == 'wave1':
                lines += [f"Aday / kontrol: `{h['contrast']['candidate']}` / `{h['contrast']['control']}`. "
                          f"Rejim: `{h['regime']}`. Ölçüt: `{h['metric']}`. Pratik eşik: {h['practical_margin']}.", '']
    (DOCS / 'CATALOG_TR.md').write_text('\n'.join(lines), encoding='utf-8')


def prepare():
    assert not (RESULTS / 'freeze.json').exists(), 'Frozen campaign already exists; do not overwrite.'
    registry = collect_registry()
    write_json(HERE / 'registry.json', registry)
    render_catalog(registry)
    checks = {'statistics': stats_selfcheck()}
    for name in MODULES:
        checks[name] = importlib.import_module(name).selfcheck()
        assert checks[name]['passed'], (name, checks[name])
        print(name, 'selfcheck', checks[name]['checks'], flush=True)
    write_json(RESULTS / 'preflight.json', {'time': stamp(), 'checks': checks})
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.json')) + list(HERE.glob('*.md'))
    paths += [DOCS / 'PROTOCOL_TR.md', DOCS / 'CATALOG_TR.md']
    import numpy
    frozen = {'version': 1, 'time_utc': stamp(), 'seeds': SEEDS,
              'python': sys.version, 'numpy': numpy.__version__, 'platform': platform.platform(),
              'registry_count': len(registry), 'active_count': 48,
              'source_files': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(paths)}}
    write_json(RESULTS / 'freeze.json', frozen)
    print('Frozen', len(paths), 'files; 200 claims, 48 executable.', flush=True)


def frozen_check():
    path = RESULTS / 'freeze.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    assert frozen['seeds'] == SEEDS
    changed = [name for name, digest in frozen['source_files'].items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    assert not changed, ('Frozen source changed', changed)
    return sha(path)


def validate_run(name, result, registry):
    active = {h['id'] for h in registry if h['engine'] == name and h['status'] == 'wave1'}
    assert set(result['hypotheses']) == active
    assert result['checks'] and all(v is True for v in result['checks'].values()), result['checks']
    for hid, item in result['hypotheses'].items():
        h = next(h for h in registry if h['id'] == hid)
        for side in ('candidate', 'control'):
            assert isinstance(item[side], (int, float)) and math.isfinite(item[side]), hid
            assert math.isfinite(item['costs'][side]) and item['costs'][side] >= 0, hid
            bounds = h.get('metric_range')
            if isinstance(bounds, list):
                assert bounds[0] <= item[side] <= bounds[1], (hid, item[side], bounds)
        assert isinstance(item['diagnostics'], dict)
    encoded(result)  # strict recursive NaN/Inf rejection


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def execute(name):
    fingerprint = frozen_check()
    registry = collect_registry()
    module = importlib.import_module(name)
    for phase, seeds in SEEDS.items():
        path = RESULTS / phase / (name + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        old = read_records(path)
        assert [r['seed'] for r in old] == seeds[:len(old)], 'Only prefix resume is supported'
        for r in old:
            assert r['engine'] == name and r['phase'] == phase and r['freeze_sha256'] == fingerprint
            validate_run(name, r['result'], registry)
        with path.open('a', encoding='utf-8') as out:
            for i, seed in enumerate(seeds[len(old):], len(old)):
                start = time.perf_counter()
                result = module.run(seed)
                validate_run(name, result, registry)
                row = {'engine': name, 'phase': phase, 'seed': seed,
                       'freeze_sha256': fingerprint, 'time_utc': stamp(),
                       'elapsed_seconds': time.perf_counter() - start, 'result': result}
                out.write(encoded(row) + '\n')
                out.flush()
                if (i + 1) % 6 == 0:
                    print(name, phase, i + 1, '/', len(seeds), flush=True)
    frozen_check()


def load_all():
    fingerprint = frozen_check()
    registry = collect_registry()
    records = []
    for name in MODULES:
        for phase, seeds in SEEDS.items():
            part = read_records(RESULTS / phase / (name + '.jsonl'))
            assert [r['seed'] for r in part] == seeds, (name, phase, len(part))
            for r in part:
                assert r['freeze_sha256'] == fingerprint and r['engine'] == name and r['phase'] == phase
                validate_run(name, r['result'], registry)
            records += part
    return registry, records


def statistics_for(items):
    delta = [x['control'] - x['candidate'] for x in items]
    wins = sum(d > 1e-12 for d in delta)
    losses = sum(d < -1e-12 for d in delta)
    return {'n': len(delta), 'candidate_mean': statistics.mean(x['candidate'] for x in items),
            'control_mean': statistics.mean(x['control'] for x in items),
            'delta_mean': statistics.mean(delta), 'delta_median': statistics.median(delta),
            'delta_min': min(delta), 'delta_max': max(delta),
            'wins': wins, 'losses': losses, 'ties': len(delta) - wins - losses,
            'p_support': sign_tail(wins, losses), 'p_reverse': sign_tail(losses, wins),
            'candidate_cost_mean': statistics.mean(x['costs']['candidate'] for x in items),
            'control_cost_mean': statistics.mean(x['costs']['control'] for x in items)}


def analyze(write=True):
    registry, records = load_all()
    active = [h for h in registry if h['status'] == 'wave1']
    analyses, pvalues = [], {}
    for h in active:
        phases = {}
        for phase in SEEDS:
            items = [r['result']['hypotheses'][h['id']] for r in records
                     if r['engine'] == h['engine'] and r['phase'] == phase]
            phases[phase] = statistics_for(items)
        hid = h['id']
        pvalues[hid + ':support'] = phases['confirmation']['p_support']
        pvalues[hid + ':reverse'] = phases['confirmation']['p_reverse']
        analyses.append({'hypothesis': h, **phases})
    adjusted = holm(pvalues)
    for a in analyses:
        h, s = a['hypothesis'], a['confirmation']
        s['p_support_holm'] = adjusted[h['id'] + ':support']
        s['p_reverse_holm'] = adjusted[h['id'] + ':reverse']
        enough = math.ceil(2 * s['n'] / 3)
        support = s['p_support_holm'] <= .05 and s['delta_mean'] >= h['practical_margin'] and s['wins'] >= enough
        reverse = s['p_reverse_holm'] <= .05 and -s['delta_mean'] >= h['practical_margin'] and s['losses'] >= enough
        assert not (support and reverse)
        a['decision'] = 'supported' if support else 'contradicted' if reverse else 'inconclusive'
    totals = dict(Counter(a['decision'] for a in analyses))
    summary = {'catalog_count': len(registry), 'executed_claims': len(active),
               'queued_claims': len(registry) - len(active), 'engine_lifetimes': len(records),
               'paired_outcomes': sum(len(r['result']['hypotheses']) for r in records),
               'holm_test_count': len(pvalues), 'decisions': totals, 'analyses': analyses}
    if write:
        write_json(RESULTS / 'summary.json', summary)
        render_results(summary)
        print(encoded({k: v for k, v in summary.items() if k != 'analyses'}), flush=True)
    return summary


def render_results(summary):
    lines = ['# Tur 23 — ilk dalganın eksiksiz sonuçları', '',
             f"{summary['executed_claims']} koşullu iddia; {summary['paired_outcomes']} eşleştirilmiş sonuç; "
             f"{summary['engine_lifetimes']} motor-yaşamı. 24 keşif, 48 doğrulama tohumu.", '',
             'Karar doğrulama grubuna aittir. Pozitif Δ = kontrol − aday. Ölçütler aileler arasında aynı değildir.',
             'Sanal hesap sayacı CPU süresi değildir. Destek yalnız belirtilen yapay dünya, ölçüt ve rakibe ilişkindir.', '',
             '| Kimlik | Rejim | Aday / kontrol | Aday ölçütü | Kontrol ölçütü | Δ | + / = / − | Holm p (destek / ters) | Karar |',
             '|---|---|---|---:|---:|---:|---|---|---|']
    for a in summary['analyses']:
        h, s = a['hypothesis'], a['confirmation']
        lines.append(f"| {h['id']} | {h['regime']} | {h['contrast']['candidate']} / {h['contrast']['control']} | "
                     f"{s['candidate_mean']:.6g} | {s['control_mean']:.6g} | {s['delta_mean']:.6g} | "
                     f"{s['wins']} / {s['ties']} / {s['losses']} | {s['p_support_holm']:.3g} / {s['p_reverse_holm']:.3g} | {LABELS[a['decision']]} |")
    lines += ['', '## Ölçüt ve kaynak dökümü', '',
              '| Kimlik | Ölçüt | Pratik eşik | Aday sanal işlem | Kontrol sanal işlem | Keşif Δ |',
              '|---|---|---:|---:|---:|---:|']
    for a in summary['analyses']:
        h, s = a['hypothesis'], a['confirmation']
        lines.append(f"| {h['id']} | {h['metric']} | {h['practical_margin']} | {s['candidate_cost_mean']:.6g} | "
                     f"{s['control_cost_mean']:.6g} | {a['discovery']['delta_mean']:.6g} |")
    lines += ['', 'Bütün yaşamların bellek/edinim/evre ayrıntıları araştırma klasöründeki discovery ve confirmation JSONL kayıtlarındadır.',
              'Reddedilemeyen bir iddia doğru veya yanlış kabul edilmez; belirsizlik eşdeğerlik kanıtı değildir.', '']
    (DOCS / 'RESULTS_TR.md').write_text('\n'.join(lines), encoding='utf-8')


def verify(rerun):
    registry, records = load_all()
    current = analyze(write=False)
    saved = json.loads((RESULTS / 'summary.json').read_text(encoding='utf-8'))
    assert encoded(current) == encoded(saved), 'Summary mismatch'
    selected = records if rerun == 'all' else [r for r in records if r['seed'] in (0, 1000)]
    counts = Counter()
    for r in selected:
        result = importlib.import_module(r['engine']).run(r['seed'])
        assert encoded(result) == encoded(r['result']), (r['engine'], r['seed'], 'rerun differs')
        counts[r['engine']] += 1
        if sum(counts.values()) % 24 == 0:
            print('Verified deterministic reruns', sum(counts.values()), '/', len(selected), flush=True)
    baseline = json.loads((RESULTS / 'preservation_baseline.json').read_text(encoding='utf-8-sig'))
    files = baseline['files']
    if isinstance(files, list):
        files = {f['path']: f['sha256'] for f in files}
    changes = [name for name, digest in files.items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    assert not changes, ('Existing research modified', changes)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head == baseline['base_head'], 'Git HEAD changed'
    frozen_check()
    output = {'time_utc': stamp(), 'passed': True, 'catalog_entries': len(registry),
              'complete_engine_lifetimes': len(records), 'paired_outcomes': current['paired_outcomes'],
              'summary_recomputed': True, 'deterministic_reruns': dict(counts),
              'preserved_existing_files': len(files), 'head_unchanged': head,
              'frozen_sources_unchanged': True, 'statistics': stats_selfcheck(),
              'note': 'Deterministic replay is reproducibility, not an independent implementation or real-world validation.'}
    write_json(RESULTS / 'verification.json', output)
    print(encoded(output), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'analyze', 'verify'))
    parser.add_argument('--module', choices=MODULES)
    parser.add_argument('--rerun', choices=('sample', 'all'), default='sample')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run':
        if args.module: execute(args.module)
        else:
            for name in MODULES: execute(name)
    elif args.action == 'analyze': analyze()
    elif args.action == 'verify': verify(args.rerun)


if __name__ == '__main__':
    main()
