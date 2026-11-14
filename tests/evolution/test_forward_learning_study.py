# -*- coding: utf-8 -*-
"""Forward-learning study (research/forward_learning_2026_11_13): pipeline, records, selection, decisions.

* The per-world runner on a tiny random base (no trained weights): a first test runs most arms with the gate,
  capacity and retraction parts, a second one the remaining arms, so that every arm of the protocol's Part A and
  Part B runs at least once; records are schema-valid and complete, the base model stays bit-identical, a resumed
  run adds nothing (also after a kill in the middle of a record), and two runs give the same records.
* A base may only be used for its own world.
* Selection and decisions on synthetic records with known effects: labels, guards, the random-feedback control,
  Holm over the primary family, the intersection-union stream test, the capacity count rule, the retraction bound.
* The freeze gate: confirmatory worlds are refused while the protocol is not frozen; the selection is tied to the
  engine and configuration of the discovery records, confirmatory records to the selection; job lists are well formed.

No learning claim is made here: the tiny model is random and untrained.
"""
from __future__ import annotations

import json
import logging
import pathlib
import random
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
STUDY = REPO / "research" / "forward_learning_2026_11_13"
CEV11 = REPO / "research" / "living_learning_paradigm_2026_11_13"
for _p in (str(REPO), str(CEV11), str(STUDY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fwl import analysis as A  # noqa: E402
from fwl import common as C  # noqa: E402
from fwl import single as SG  # noqa: E402
from fwl import stream as ST  # noqa: E402
from fwl import world_run as W  # noqa: E402
from fwl import cli as CLI  # noqa: E402
from fwl import verification as V  # noqa: E402

# The package is imported; the study folder itself must not stay importable, because its wrapper scripts
# (analyze.py, verify.py, run_world.py, run_select.py) share their names with other study folders.
sys.path.remove(str(STUDY))

WIDE_GRID = (0.0,) + tuple(0.25 * 2.0 ** (i / 2.0) for i in range(44))
TINY_WORLD = 11                                # the world seed of the tiny base


@pytest.fixture(autouse=True)
def _single_thread_quiet():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(previous)
    torch.set_num_threads(threads)


def make_tiny_base():
    from cev11 import experiment as E
    from cev11.model import build_model, model_sha256
    from cev11.tokenizers import train_tokenizer
    from cev11.world import make_world, sentence
    world = make_world(TINY_WORLD, n_names=120)
    rnd = random.Random(0)
    texts = [sentence(n, r, c, world.facts[(n, r)], o, x)
             for n in world.names[:60] for r in ("A", "B", "C") for c in ("nom", "gen", "abl")
             for o in ("formal", "informal") for x in ("none", "P1")]
    rnd.shuffle(texts)
    letters = " ".join(ch * 3 for ch in "çğıöşüÇĞİÖŞÜâîû") + " "
    tok = train_tokenizer(texts + ["Hava bugün çok güzeldi.", "Kapı yavaşça açıldı.", letters] * 20,
                          vocab_size=900, require_full_vocab=False)
    model, cfg = build_model(tok.vocab_size, layers=2, d=32, heads=4, kv_heads=4, max_len=64, seed=0)
    model.eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():                      # a fresh model has near-zero residual branches
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    base = E.Base(model, cfg, tok, {}, world, texts[:200], texts[200:220], pathlib.Path("."), model_sha256(model))
    editable = sorted(n for n, p in world.partition.items() if p == "probe")
    return base, editable


@pytest.fixture(scope="module")
def tiny():
    torch.set_num_threads(1)
    return make_tiny_base()


def tiny_config() -> C.RunConfig:
    return C.RunConfig(
        label="test", per_case=1, edits_per_case=1, max_edits=1,
        single_arms=("FEC-TOP", "FEC-A", "RAND-0", "BP-1b", "BP-pred", "FT-L", "FT-ALL"),
        installs=("greedy", "nll"), single_grid=WIDE_GRID, bpb_edits=1, bpb_arms=("FEC-TOP",), bpb_texts=2,
        so_entities=2, streams=1, stream_n=6, checkpoints=(3, 6),
        stream_arms=("FEC-TOP+RLS", "FEC-A+RLS", "FEC-TOP+RLS+G", "BP-1+RLS", "FT-L-seq", "RLSEdit"),
        stream_install="greedy", stream_grid=WIDE_GRID, arm_limits={"RLSEdit": 3}, untouched=10, c0_tokens=600,
        gate_arms=("FEC-TOP+RLS",), thetas=(0.0, 0.5), gains=(1.0, 0.5), gate_bpb=False, hard_negatives=8,
        retract_indices=(1, 4), capacity_thetas=(0.5,), guard_rms_cap=1e9)


def _strip(records):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items() if k != "seconds"}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return [clean(r) for r in records]


# ---------------------------------------------------------------------------------------------- pipeline on a base
def test_world_runner_validates_resumes_and_is_deterministic(tiny, tmp_path, monkeypatch):
    from cev11.model import model_sha256
    base, editable = tiny
    cfg = tiny_config()
    # the comparators' frozen CEV-11 settings are far too long for a plumbing test on a random model
    monkeypatch.setitem(C.FT_L, "hp", {"eps": 0.02, "lr": 0.01, "steps": 3})
    monkeypatch.setitem(C.FT_ALL, "hp", {"lr": 0.001, "steps": 3})
    monkeypatch.setitem(C.ROOT_SITE, "hp", {"cap_mult": 8, "step": 2.0, "max_steps": 5})
    x = torch.randint(5, base.tok.vocab_size, (2, 12), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        before = base.model(x)[0].clone()
    info = W.run_world(TINY_WORLD, "smoke", cfg, tmp_path / "a", base=base, editable=editable, log=lambda m: None)
    path = C.results_path(tmp_path / "a", TINY_WORLD)
    records = C.read_records(path)
    assert info["n_records"] == len(records) and records[0]["key"] == "meta|run"
    # nothing is left installed and no parameter changed
    assert model_sha256(base.model) == base.base_sha
    with torch.no_grad():
        assert torch.equal(base.model(x)[0], before)
    assert all(len(layer.ffn.fc2._forward_hooks) == 0 for layer in base.model.layers)   # every hook is gone

    report = V.check_records(tmp_path / "a")
    assert report["problems"] == [] and report["worlds"][str(TINY_WORLD)]["complete"]
    kinds = {k: sum(r["kind"] == k for r in records) for k in C.KINDS}
    assert kinds["single"] == 5 * 2 + 2 and kinds["stream"] == 5 * 2 + 1
    assert kinds["gate"] == 3 and kinds["capacity"] == 1 and kinds["side"] == 1

    single = {r["key"]: r for r in records if r["kind"] == "single"}
    edits = sorted({r["edit_id"] for r in single.values()})
    assert len(edits) == 1
    for r in single.values():
        assert r["install"] in ("greedy", "nll", "ft") and isinstance(r["installed"], bool)
        assert not r["installed"] or r["write_installed"]
        if r["write_installed"] or r["install"] == "ft":           # measured under the write
            assert set(r["leak"]) >= {"RN", "RD", "RB", "RC"} and len(r["PF6"]) == 6
        else:                                                        # nothing was written: nothing to measure
            assert r["PF6"] == {} and r["targets"] == [] and r["leak"] == {}
    # the random-feedback control depends on the edit only; exact-gradient arms carry the same write machinery
    rand = [r for r in single.values() if r["arm"] == "RAND-0"]
    assert all(r["n_forward"] >= 1 for r in rand)
    assert SG.edit_seed("w1-e000") == SG.edit_seed("w1-e000") != SG.edit_seed("w1-e001")
    top = [r for r in single.values() if r["arm"] == "FEC-TOP" and r["install"] == "greedy"]
    assert any(r["write_installed"] for r in top)                 # the exact top-block rule installs on a random model
    assert all(set(r["block_share"]) == {"1"} for r in top if r["write_installed"] and r["mu"])
    assert sum("bpb" in r for r in single.values()) >= 1

    streams = [r for r in records if r["kind"] == "stream"]
    assert {r["arm"] for r in streams} == set(cfg.stream_arms)
    assert [r["n"] for r in streams if r["arm"] == "RLSEdit"] == [3]
    for r in streams:
        assert 0 <= r["installed"] <= r["n"] and 0.0 <= r["dmg_A"] <= 1.0
        assert abs(r["composite"] - ((r["retention"] + 0.5 * (r["PF6"] + r["IF"])) / 2 - r["dmg_A"])) < 1e-12
        assert ("dbpb" in r) == (r["n"] == max(x["n"] for x in streams if x["arm"] == r["arm"]))
    fec = [r for r in streams if r["arm"] == "FEC-TOP+RLS" and r["n"] == 6][0]
    assert fec["installed"] >= 4 and fec["ledger"] >= 4 and set(fec["rms_ratio"]) == {"1"}
    cap = next(r for r in records if r["kind"] == "capacity")
    assert len(cap["trace"]) == 6 and len(cap["predicted"]) == 6 and len(cap["predicted_v2"]) == 6
    assert cap["trace"][-1][0] == fec["n_keys"]["1"]              # stored keys of the top block
    assert all(0.0 <= m <= 1.0 and 0.0 <= p <= 1.0 for (_, m, _), p in zip(cap["trace"], cap["predicted"]))
    assert abs(sum(cap["shares"]) - 1.0) < 1e-12 and cap["knees"][0]["theta"] == 0.5
    # the crossing of the measured mean is on file next to the registered (median) knee, against the same law
    assert [k["theta"] for k in cap["knees_mean"]] == [0.5]
    assert cap["knees_mean"][0]["predicted_keys"] == cap["knees"][0]["predicted_keys"]
    assert cap["knees_mean"][0]["measured_keys"] == ST._crossing([t[0] for t in cap["trace"]],
                                                                 [t[1] for t in cap["trace"]], 0.5)
    gates = {r["setting"]: r for r in records if r["kind"] == "gate"}
    assert set(gates) == {"theta=0", "theta=0.5", "gain=0.5"}
    # the ungated read moves hard negatives of this random model; the gated read cannot leak more than it
    assert gates["theta=0"]["leak_hard"] > 0.0 and gates["theta=0"]["n_hard"] > 0
    assert gates["theta=0.5"]["leak_hard"] <= gates["theta=0"]["leak_hard"]
    retr = [r for r in records if r["kind"] == "retract"]
    side = next(r for r in records if r["kind"] == "side")
    assert sorted(r["index"] for r in retr) == side["retracted"] and len(retr) >= 1
    for r in retr:
        assert all(k >= -1e-9 for k in r["kl_replay_retract"]) and 0.0 <= r["others_retract"] <= 1.0

    # resume: nothing is added, the file is unchanged
    digest = C.sha256_file(path)
    again = W.run_world(TINY_WORLD, "smoke", cfg, tmp_path / "a", base=base, editable=editable,
                        log=lambda m: None)
    assert again["n_records"] == len(records) and C.sha256_file(path) == digest
    # a process killed in the middle of the last record: the fragment is cut off and the record is made again
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[-1] == "" and json.loads(lines[-2])["key"] == "stream|0|RLSEdit|3"
    path.write_text("\n".join(lines[:-2]) + "\n" + lines[-2][:len(lines[-2]) // 2], encoding="utf-8")
    with pytest.raises(ValueError):
        C.read_records(path)                                        # the strict reader does not hide the damage
    logged = []
    healed = W.run_world(TINY_WORLD, "smoke", cfg, tmp_path / "a", base=base, editable=editable, log=logged.append)
    assert any("unfinished last line dropped" in m for m in logged)
    assert healed["n_records"] == len(records)
    assert _strip(C.read_records(path)) == _strip(records)
    # another configuration in the same folder is refused
    other = C.RunConfig(**{**cfg.__dict__, "stream_n": 5, "checkpoints": (5,)})
    with pytest.raises(RuntimeError, match="another config_sha256"):
        W.run_world(TINY_WORLD, "smoke", other, tmp_path / "a", base=base, editable=editable, log=lambda m: None)
    # a base belongs to one world: the same base under another world number is refused before anything is written
    with pytest.raises(RuntimeError, match="pre-trained for world 11"):
        W.run_world(1000, "smoke", cfg, tmp_path / "c", base=base, editable=editable, log=lambda m: None)
    assert not (tmp_path / "c").exists()
    # determinism: a second run of the stream part gives the same records (wall-clock fields aside)
    W.run_world(TINY_WORLD, "smoke", cfg, tmp_path / "b", base=base, editable=editable, parts=("B",),
                log=lambda m: None)
    second = C.read_records(C.results_path(tmp_path / "b", TINY_WORLD))
    assert _strip(second) == _strip([r for r in records if r["kind"] != "single"])
    # ... and so do two writes of a one-step arm, the seeded random-feedback control included
    FL = C.engine()
    item = SG.protocol_items(base, editable, TINY_WORLD, cfg)[0]
    for arm in ("RAND-0", "FEC-A", "BP-pred"):
        prints = []
        for _ in range(2):
            learner = FL.ForwardLearner(base.model, SG.learner_config(arm, "greedy", cfg))
            learner.write(SG.make_reading(base, learner, item, arm), mu=1.0)
            prints.append(learner.fingerprint())
            learner.close()
        assert prints[0] == prints[1]
    # summaries do not fail on real records
    summary = W.summarise(records)
    assert "FEC-TOP|greedy" in summary["single"] and "FEC-TOP+RLS|6" in summary["stream"]
    assert model_sha256(base.model) == base.base_sha


def test_the_remaining_arms_run_so_that_every_protocol_arm_is_exercised(tiny, tmp_path):
    base, editable = tiny
    first = tiny_config()
    cfg = C.RunConfig(
        label="test-rest", per_case=1, edits_per_case=1, max_edits=1,
        single_arms=("FEC-0", "FEC-1", "HEBB-TOP", "HEBB-0", "BP-1"), installs=("greedy",), single_grid=WIDE_GRID,
        bpb_edits=0, bpb_texts=2, so_entities=2, streams=1, stream_n=3, checkpoints=(3,),
        stream_arms=("FEC-TOP", "HEBB-TOP+RLS", "HEBB-0+RLS"), stream_install="greedy", stream_grid=WIDE_GRID,
        untouched=10, c0_tokens=600, gate_arms=(), retract_indices=(), capacity_thetas=(0.5,), guard_rms_cap=1e9)
    # the two test configurations together are the protocol's arm lists
    assert set(first.single_arms) | set(cfg.single_arms) == set(C.SINGLE_ONE_STEP + C.SINGLE_FT)
    assert set(first.stream_arms) | set(cfg.stream_arms) == set(C.STREAM_ARMS)
    assert set(C.COMPARATORS) | set(C.FEC_STREAM) <= set(C.STREAM_ARMS)
    assert set(C.NORMALISED_STREAM_ARMS) == set(ST.LEARNER_ARMS)
    W.run_world(TINY_WORLD, "smoke", cfg, tmp_path, base=base, editable=editable, log=lambda m: None)
    records = C.read_records(C.results_path(tmp_path, TINY_WORLD))
    assert V.check_records(tmp_path)["problems"] == []
    single = {r["arm"]: r for r in records if r["kind"] == "single"}
    assert set(single) == set(cfg.single_arms)
    top = str(len(base.model.layers) - 1)
    for arm, r in single.items():
        if r["write_installed"] and r["mu"]:
            # the top-block rule writes one block, the others all of them
            assert (set(r["block_share"]) == {top}) == (arm == "HEBB-TOP"), arm
            assert abs(sum(r["block_share"].values()) - 1.0) < 1e-9
    assert any(r["write_installed"] for r in single.values())
    streams = {r["arm"]: r for r in records if r["kind"] == "stream"}
    assert set(streams) == set(cfg.stream_arms)
    assert all(r["n"] == 3 and 0 <= r["installed"] <= 3 for r in streams.values())
    assert set(streams["HEBB-TOP+RLS"].get("n_keys", {top: 0})) <= {top}       # top-block arm: top-block keys only
    assert not any(r["kind"] in ("gate", "capacity", "retract", "side") for r in records)
    from cev11.model import model_sha256
    assert model_sha256(base.model) == base.base_sha
    assert all(len(layer.ffn.fc2._forward_hooks) == 0 for layer in base.model.layers)


def test_a_base_is_bound_to_its_world(tiny):
    base, _ = tiny
    C.check_base_world(base, TINY_WORLD)
    for other in (0, 1, 1000):
        with pytest.raises(RuntimeError, match="may only use the base of its own world"):
            C.check_base_world(base, other)


def test_stream_probes_and_items_are_deterministic_and_disjoint(tiny):
    base, editable = tiny
    cfg = tiny_config()
    a, b = ST.build_stream(base, editable, 7, 0, cfg), ST.build_stream(base, editable, 7, 0, cfg)
    assert a.items == b.items and a.untouched == b.untouched
    other = ST.build_stream(base, editable, 7, 1, cfg)
    assert [it["name"] for it in other.items] != [it["name"] for it in a.items]
    names = {it["name"] for it in a.items}
    assert len(names) == cfg.stream_n and not names & set(a.untouched)
    assert all(it["new"] != it["old"] and it["rel"] == "A" for it in a.items)
    assert all(base.world.partition[u] != "p3" for u in a.untouched)
    items = SG.protocol_items(base, editable, 7, cfg)
    assert len(items) == 1 and items == SG.protocol_items(base, editable, 7, cfg)
    # the protocol list: CEV-11's item list, cut to the first k items of every lived case in list order
    from cev11 import probes as P
    full = P.edit_items(base.world, editable, seed=7 + 2, per_case=2)
    cut = SG.protocol_items(base, editable, 7, C.RunConfig(per_case=2, edits_per_case=1))
    assert len(full) == 14 and len(cut) == 7 and len({it["lived_case"] for it in cut}) == 7
    assert cut == [it for i, it in enumerate(full)
                   if it["lived_case"] not in {x["lived_case"] for x in full[:i]}]
    tok, pos = SG.fact_tokens(base, items[0])
    assert pos[-1] == len(tok) - 2 and len(pos) == len(base.tok.encode(" " + items[0]["new"] + "."))
    hard = ST.hard_negatives(base, a, 7, cfg)
    assert set(hard) == {"rn", "rel", "so"} and all(len(v) <= cfg.hard_negatives for v in hard.values())
    assert hard == ST.hard_negatives(base, a, 7, cfg)


def test_exactness_check_of_the_smoke_holds_on_the_tiny_base(tiny):
    base, editable = tiny
    out = W.exactness_check(base, SG.protocol_items(base, editable, 7, C.RunConfig(per_case=1, edits_per_case=1))[:3])
    assert out["n"] == 3 and all(v <= 1e-5 for v in out["max_relative_error"].values())
    cos = out["cos_fc2_gradient_with_backprop_per_block"]
    assert cos["FEC-A"][-1] == cos["FEC-0"][-1] == 1.0 and len(cos["FEC-A"]) == len(base.model.layers)


# ------------------------------------------------------------------------------------------------------- records
def test_record_validation_and_writer_resume(tmp_path):
    rec = C.make_record("stream", "smoke", 3, "stream|0|X|5", stream=0, arm="X", n=5, written=5, installed=4,
                        retention=0.8, PF6=0.5, IF=0.4, dmg_A=0.1, dmg_B=0.0, net_per_write=0.2, composite=0.4,
                        seconds=1.0)
    w = C.RecordWriter(tmp_path / "w3.jsonl")
    w.write(rec)
    with pytest.raises(ValueError, match="duplicate"):
        w.write(rec)
    assert C.RecordWriter(tmp_path / "w3.jsonl").has("stream|0|X|5")
    # an unfinished last line (a kill during a write): a complete record only lacks its newline and is kept,
    # a fragment is cut off; lines that end with a newline are never touched
    path = tmp_path / "w3.jsonl"
    clean = path.read_bytes()
    assert C.repair_tail(path) is None and C.repair_tail(tmp_path / "absent.jsonl") is None
    second = json.dumps(dict(rec, key="stream|0|X|6", n=6))
    path.write_bytes(clean + second.encode("utf-8"))
    assert C.repair_tail(path) == "completed" and path.read_bytes() == clean + second.encode("utf-8") + b"\n"
    assert [r["key"] for r in C.read_records(path)] == ["stream|0|X|5", "stream|0|X|6"]
    path.write_bytes(clean + second.encode("utf-8")[:40])
    resumed = C.RecordWriter(path)
    assert resumed.repaired == "dropped" and path.read_bytes() == clean and not resumed.has("stream|0|X|6")
    path.write_bytes(clean[:30] + b"\n" + clean)                  # damage in the middle is not repaired away
    with pytest.raises(ValueError):
        C.RecordWriter(path)
    for bad in ({**rec, "retention": 1.5}, {**rec, "kind": "nope"}, {**rec, "run": "x"}, {**rec, "world": "3"},
                {**rec, "dmg_A": float("nan")}, {k: v for k, v in rec.items() if k != "composite"},
                {**rec, "schema": "other"}):
        with pytest.raises(ValueError):
            C.validate_record(bad)
    with pytest.raises(ValueError):
        C.base_name(5)
    assert C.base_name(0) == "discovery_w0_K20" and C.base_name(1003) == "confirm_w1003_K20"
    assert C.sha256_json({"b": 1, "a": 2}) == C.sha256_json({"a": 2, "b": 1})
    assert C.median([3, 1, 2]) == 2 and C.median([4, 1, 2, 3]) == 2.5 and C.mean([]) is None


# ------------------------------------------------------------------------------------- synthetic selection, decisions
def _single(world, arm, install, good, n=10, leak=0.0, bpb=None, mu=1.0, installed=True, so=0.0):
    out = []
    for i in range(n):
        ok = installed and i < good
        rec = {"schema": C.RECORD_SCHEMA, "kind": "single", "run": "confirm", "world": world,
               "key": f"single|w{world}-e{i:03d}|{arm}|{install}", "edit_id": f"w{world}-e{i:03d}", "arm": arm,
               "install": install, "lived_case": "gen", "installed": bool(installed), "n_targets": 1,
               "write_installed": bool(installed), "mu": mu if installed else None, "flags": [], "n_forward": 5,
               "PF6": {c: int(ok) for c in "abcdef"} if installed else {},
               "targets": [{"case": "dat", "cls": "IDENT", "formal": int(ok), "informal": int(ok)}] if installed else [],
               "leak": ({f: {"changed": int(round(leak * 100)), "n": 100} for f in ("RN", "RD", "RB", "RC")}
                        | {"SO": {"changed": int(round(so * 100)), "n": 100}}) if installed else {},
               "seconds": 1.0}
        if bpb is not None and installed:
            rec["bpb"] = {"base": 2.0, "edited": 2.0 + bpb}
        out.append(rec)
    return out


def _meta(world, run, config: C.RunConfig, engine=None):
    """A header record as run_world writes it (the real engine hash unless another one is given)."""
    cfg = config.to_json()
    return {"schema": C.RECORD_SCHEMA, "kind": "meta", "run": run, "world": world, "key": "meta|run", "config": cfg,
            "config_sha256": C.sha256_json(cfg), "base_sha256": f"base-{world}",
            "engine_sha256": engine or C.sha256_file(C.ENGINE_FILE), "items_sha256": f"items-{world}",
            "n_items": 10}


def _stream(world, arm, composite, n=100, run="confirm", mu=2.0):
    return {"schema": C.RECORD_SCHEMA, "kind": "stream", "run": run, "world": world, "key": f"stream|0|{arm}|{n}",
            "stream": 0, "arm": arm, "n": n, "written": n, "installed": n, "retention": 0.9, "PF6": 0.6, "IF": 0.5,
            "dmg_A": 0.1, "dmg_B": 0.0, "known_A": 100, "known_B": 100, "net_per_write": composite - 0.5,
            "composite": composite, "mu_median": mu, "mu_max": None if mu is None else 2 * mu, "seconds": 1.0}


def _gate(world, arm, setting, theta, gain, recall, leak, run="confirm", dbpb=0.0):
    return {"schema": C.RECORD_SCHEMA, "kind": "gate", "run": run, "world": world, "key": f"gate|0|{arm}|{setting}",
            "stream": 0, "arm": arm, "setting": setting, "theta": theta, "gain": gain, "retention": 0.95,
            "recall_other": recall, "leak_hard": leak, "leak_generic": leak, "dmg_A": leak, "dbpb": dbpb}


def _capacity(world, ratio):
    return {"schema": C.RECORD_SCHEMA, "kind": "capacity", "run": "confirm", "world": world,
            "key": "capacity|0|FEC-TOP+RLS", "stream": 0, "arm": "FEC-TOP+RLS", "block": 5, "trace": [[10.0, 0.1, 0.1]],
            "predicted": [0.1], "knees": [{"theta": 0.5, "predicted_keys": 100.0, "measured_keys": 100.0 * ratio,
                                           "ratio": ratio}],
            "nu_mean_end": 0.4, "predicted_end": 0.4, "keys_end": 10.0}


def _retract(world, i, kl, p_before=0.9, others=(0.9, 0.9)):
    return {"schema": C.RECORD_SCHEMA, "kind": "retract", "run": "confirm", "world": world,
            "key": f"retract|0|FEC-TOP+RLS|{i}", "stream": 0, "arm": "FEC-TOP+RLS", "index": i, "write_id": i,
            "p_before": p_before, "p_retract": 0.01, "p_replay": 0.01, "kl_replay_retract": [kl, kl],
            "others_retract": others[0], "others_replay": others[1]}


def synthetic_world(world, *, fec_a=9, ft_l=8, ft_all=9, fec_0=7, fec_top=8, bp=9, rand_ok=False, s_fec=0.6,
                    s_cmp=0.6, s_hebb=0.2, gated=0.8, ungated=0.5, knee=1.0, kl=0.01, leak_fec=0.0, bpb=0.001,
                    gate_leak=0.01):
    recs = []
    recs += _single(world, "FEC-A", "nll", fec_a, leak=leak_fec, bpb=bpb)
    recs += _single(world, "FEC-A", "greedy", fec_a)
    recs += _single(world, "FEC-0", "greedy", fec_0) + _single(world, "FEC-0", "nll", fec_0)
    recs += _single(world, "FEC-TOP", "greedy", fec_top) + _single(world, "FEC-TOP", "nll", fec_top)
    recs += _single(world, "BP-pred", "greedy", bp) + _single(world, "BP-pred", "nll", bp, so=0.05)
    recs += _single(world, "BP-1", "greedy", bp - 2)
    recs += _single(world, "FT-L", "ft", ft_l) + _single(world, "FT-ALL", "ft", ft_all)
    recs += _single(world, "RAND-0", "greedy", 10, installed=rand_ok)
    recs += [_stream(world, "FEC-TOP+RLS", s_fec), _stream(world, "FEC-A+RLS", s_fec - 0.1),
             _stream(world, "RLSEdit", s_cmp), _stream(world, "FT-L-seq", s_cmp - 0.3, mu=None),
             _stream(world, "BP-1+RLS", s_cmp - 0.5), _stream(world, "HEBB-TOP+RLS", s_hebb),
             _stream(world, "HEBB-0+RLS", s_hebb - 0.1)]
    recs += [_gate(world, "FEC-TOP+RLS", "theta=0", 0.0, 1.0, 0.85, 0.4),
             _gate(world, "FEC-TOP+RLS", "theta=0.6", 0.6, 1.0, gated, gate_leak),
             _gate(world, "FEC-TOP+RLS", "theta=0.9", 0.9, 1.0, 0.3, 0.0),
             _gate(world, "FEC-TOP+RLS", "gain=0.5", 0.0, 0.5, ungated, 0.015),
             _gate(world, "FEC-TOP+RLS", "gain=0.25", 0.0, 0.25, 0.1, 0.0)]
    recs += [_capacity(world, knee), _retract(world, 5, kl), _retract(world, 20, kl),
             _retract(world, 35, 5.0, p_before=0.1)]                # not recalled before: excluded
    for r in recs:
        C.validate_record(r)
    return recs


def discovery_records(engine=None, config=None):
    out = {}
    for w in (0, 1):
        recs = [_meta(w, "discovery", config or C.protocol_config(), engine)]
        recs += [dict(r, run="discovery") for r in synthetic_world(w)]
        out[w] = recs
    return out


def test_selection_fixes_operating_points_from_discovery_only():
    sel = A.select(discovery_records())
    assert A.selection_hash_ok(sel) and sel["worlds"] == [0, 1]
    assert sel["bp_star"] == "BP-pred" and sel["fec_star"] == "FEC-TOP+RLS" and sel["cmp_star"] == "RLSEdit"
    assert sel["gate_arm"] == "FEC-TOP+RLS"
    theta, gain = sel["gate"]["FEC-TOP+RLS"]["theta"], sel["gate"]["FEC-TOP+RLS"]["gain"]
    assert theta["value"] == 0.6 and theta["feasible"] and theta["setting"] == "theta=0.6"   # best recall at leak <= 2%
    assert gain["value"] == 0.5 and gain["feasible"]
    assert sel["caps"]["single"]["FEC-A|nll"] == 4.0 and sel["caps"]["stream"]["FEC-TOP+RLS"] == 8.0
    assert sel["caps"]["single"]["RAND-0|greedy"] == sel["caps"]["single"]["FEC-0|greedy"]
    # the selection carries the engine and the configuration of the discovery records
    assert sel["engine_sha256"] == C.sha256_file(C.ENGINE_FILE)
    assert sel["discovery_config_sha256"] == C.sha256_json(C.protocol_config().to_json())
    # one common step cap for every unit-normalised stream arm, from the FEC arms: an arm that collapses without
    # a cap (large steps of its own) does not get a looser cap for it
    wild = discovery_records()
    for recs in wild.values():
        for r in recs:
            if r["kind"] == "stream" and r["arm"] == "BP-1+RLS":
                r["mu_median"], r["mu_max"] = 40.0, 128.0
            if r["kind"] == "stream" and r["arm"] == "RLSEdit":
                r["mu_median"] = 1.0
    caps = A.select(wild)["caps"]["stream"]
    assert {caps[a] for a in C.NORMALISED_STREAM_ARMS} == {8.0} and caps["BP-1+RLS"] == 8.0
    assert caps["RLSEdit"] == 4.0 and "FT-L-seq" not in caps       # its own scale; no step search at all
    # FEC* is chosen on net facts per write; the comparator ranking is descriptive (H3 tests every comparator)
    flipped = discovery_records()
    for recs in flipped.values():
        for r in recs:
            if r["kind"] == "stream" and r["arm"] == "FEC-A+RLS":
                r["net_per_write"] = 5.0                              # better net, worse composite
    assert A.select(flipped)["fec_star"] == "FEC-A+RLS"
    # deterministic, and any change of a discovery record changes the hash
    assert A.select(discovery_records())["sha256"] == sel["sha256"]
    changed = discovery_records()
    changed[0][0]["mu"] = 3.0
    assert A.select(changed)["sha256"] != sel["sha256"]
    tampered = dict(sel, bp_star="BP-1")
    assert not A.selection_hash_ok(tampered)
    # no feasible setting: the most conservative one is taken and flagged
    tables = [{"theta=0": _gate(0, "X", "theta=0", 0.0, 1.0, 0.9, 0.4), "theta=0.5": _gate(0, "X", "theta=0.5", 0.5,
                                                                                             1.0, 0.7, 0.1)}]
    point = A.choose_operating_point(tables, "theta")
    assert point["value"] == 0.5 and not point["feasible"]
    assert C.protocol_config(sel).caps["stream"]["FEC-TOP+RLS"] == 8.0


def confirm(selection=None, **overrides):
    """Synthetic confirmatory records; with ``selection`` every world also gets the header a real run writes."""
    per_world = {}
    for i, w in enumerate(C.CONFIRMATORY_WORLDS):
        kw = {k: (v[i] if isinstance(v, (list, tuple)) else v) for k, v in overrides.items()}
        head = [] if selection is None else [_meta(w, "confirm", C.protocol_config(selection))]
        per_world[w] = head + synthetic_world(w, **kw)
    return per_world


def test_decisions_labels_guards_and_controls():
    sel = A.select(discovery_records())
    jitter = [0, 1, 0, 1, 0, 0, 1, 0]
    base = A.compute_decisions(confirm(ft_all=[9 - j for j in jitter]), sel)
    d = base["decisions"]
    assert set(d) == {"H1", "H2a", "H2b", "H3", "H4a", "H4b", "H5"} and base["worlds"] == list(C.CONFIRMATORY_WORLDS)
    assert d["H1"]["label"] == A.SUPPORTED and d["H1"]["kind"] == "non-inferiority" and d["H1"]["n"] == 8
    assert all(g["ok"] for g in d["H1"]["guards"].values())
    assert d["H1"]["margin"] == 0.04                      # against FT-ALL, the stronger comparator
    assert d["H2a"]["label"] == A.SUPPORTED and abs(d["H2a"]["mean"] - 0.1) < 1e-9       # FEC-A 0.9 - FEC-TOP 0.8
    assert d["H2b"]["label"] == A.SUPPORTED and d["H2b"]["control_RAND-0"]["ok"]
    assert d["H3"]["label"] == A.SUPPORTED and d["H4a"]["label"] == A.SUPPORTED
    # H3: net facts per write of FEC* against every comparator; the binding one is the closest
    assert d["H3"]["metric"] == "net_per_write" and d["H3"]["arm"] == "FEC-TOP+RLS"
    assert set(d["H3"]["comparators"]) == set(C.COMPARATORS) and d["H3"]["against"] == "RLSEdit"
    assert d["H4a"]["operating_point"] == {"arm": "FEC-TOP+RLS", "theta": "theta=0.6", "gain": "gain=0.5",
                                           "feasible_on_discovery": [True, True]}
    assert d["H4b"]["label"] == A.SUPPORTED and d["H4b"]["inside_25"] == 8
    assert d["H5"]["label"] == A.SUPPORTED and d["H5"]["kl"]["n"] == 8      # the unrecalled fact is excluded
    assert all(v == pytest.approx(0.01) for v in d["H5"]["kl"]["worlds"])
    for h in A.PRIMARY:                                   # smallest exact p with 8 worlds, Holm over five tests
        assert d[h]["p"] == pytest.approx(1 / 256) and d[h]["p_holm"] == pytest.approx(5 / 256)
    assert base["secondary"]["H1_vs_FT-L"]["label"] == A.SUPPORTED
    assert base["secondary"]["routing_vs_FEC-0_greedy"]["label"] == A.SUPPORTED          # 0.9 - 0.7
    assert base["secondary"]["H3_superiority_vs_CMP"]["label"] != A.SUPPORTED   # equal nets: no superiority
    assert base["secondary"]["H3_composite_vs_CMP"]["label"] == A.SUPPORTED
    assert base["decisions"]["H4b"]["ratios_mean_crossing"] == []          # synthetic records carry no mean knee
    assert base["tables"]["1000"]["single"]["FEC-A|nll"]["G"] == pytest.approx(0.9)

    # H1: far below the comparator -> refuted; a leak guard that fails turns "supported" into "guard failed"
    d = A.compute_decisions(confirm(fec_a=[6, 7, 6, 7, 6, 6, 7, 6]), sel)["decisions"]
    assert d["H1"]["label"] == A.REFUTED and d["H2a"]["label"] == A.REFUTED
    d = A.compute_decisions(confirm(leak_fec=[0.06, 0.07, 0.06, 0.08, 0.06, 0.07, 0.06, 0.07]), sel)["decisions"]
    assert d["H1"]["label"] == A.GUARD_FAILED and not d["H1"]["guards"]["Leak_RD"]["ok"]
    d = A.compute_decisions(confirm(bpb=[0.02, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02, 0.03]), sel)["decisions"]
    assert d["H1"]["label"] == A.GUARD_FAILED and not d["H1"]["guards"]["dbpb"]["ok"]
    # mixed signs -> inconclusive
    d = A.compute_decisions(confirm(fec_a=[9, 7, 9, 8, 9, 7, 9, 8], ft_all=[8, 9, 8, 9, 8, 9, 8, 9]),
                            sel)["decisions"]
    assert d["H1"]["label"] == A.INCONCLUSIVE
    # H1 is decided against FT-ALL: beating the weaker FT-L does not help, and the FT-L test is only secondary
    full = A.compute_decisions(confirm(fec_a=8, ft_l=6, ft_all=10), sel)
    assert full["decisions"]["H1"]["label"] == A.REFUTED and full["secondary"]["H1_vs_FT-L"]["label"] == A.SUPPORTED
    # H2a is decided against FEC-TOP: an advantage over FEC-0 alone does not support routing
    full = A.compute_decisions(confirm(fec_a=9, fec_top=9, fec_0=6), sel)
    assert full["decisions"]["H2a"]["label"] == A.REFUTED
    assert full["secondary"]["routing_vs_FEC-0_greedy"]["label"] == A.SUPPORTED
    # the random-feedback control installs: a supported routing result is relabelled
    d = A.compute_decisions(confirm(rand_ok=True), sel)["decisions"]
    assert d["H2a"]["label"] == A.CONTROL_FAILED and d["H2b"]["label"] == A.CONTROL_FAILED
    assert d["H1"]["label"] == A.SUPPORTED
    # stream: the comparator clearly better -> refuted
    d = A.compute_decisions(confirm(s_fec=[0.3, 0.32, 0.3, 0.31, 0.3, 0.32, 0.3, 0.31]), sel)["decisions"]
    assert d["H3"]["label"] == A.REFUTED
    # intersection-union: level with the comparator the discovery worlds ranked first (RLSEdit) is not enough
    # when another comparator is clearly better on the confirmatory worlds
    d = A.compute_decisions(confirm(s_hebb=[0.9, 0.92, 0.9, 0.91, 0.9, 0.92, 0.9, 0.91]), sel)["decisions"]
    assert d["H3"]["against"] == "HEBB-TOP+RLS" and d["H3"]["label"] == A.REFUTED
    assert d["H3"]["comparators"]["RLSEdit"]["p"] == pytest.approx(1 / 256)
    assert d["H3"]["p"] == max(c["p"] for c in d["H3"]["comparators"].values()) == 1.0
    # ... and one noisy comparator makes it inconclusive, not supported
    d = A.compute_decisions(confirm(s_hebb=[0.7, 0.5, 0.7, 0.5, 0.7, 0.5, 0.7, 0.6]), sel)["decisions"]
    assert d["H3"]["label"] == A.INCONCLUSIVE and d["H3"]["against"] == "HEBB-TOP+RLS"
    # a comparator without records leaves H3 untested (it is never silently dropped from the union)
    missing = {w: [r for r in recs if r.get("arm") != "BP-1+RLS"] for w, recs in confirm().items()}
    d = A.compute_decisions(missing, sel)["decisions"]
    assert d["H3"]["label"].startswith(A.NOT_TESTABLE) and "BP-1+RLS" in d["H3"]["label"]
    assert d["H3"].get("p") is None and d["H1"]["p_holm"] == pytest.approx(5 / 256)   # still a family of five
    # gate: no gain over the ungated read -> refuted; leak above the bound -> guard failed
    d = A.compute_decisions(confirm(gated=[0.5, 0.52, 0.5, 0.51, 0.5, 0.52, 0.5, 0.51]), sel)["decisions"]
    assert d["H4a"]["label"] == A.REFUTED
    d = A.compute_decisions(confirm(gate_leak=[0.2, 0.22, 0.2, 0.21, 0.2, 0.22, 0.2, 0.21]), sel)["decisions"]
    assert d["H4a"]["label"] == A.GUARD_FAILED and d["H4a"]["guard_leak_hard"]["point_limit"] == 0.02
    # capacity: the count rule
    d = A.compute_decisions(confirm(knee=[1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.9, 1.0]), sel)["decisions"]
    assert d["H4b"]["label"] == A.INCONCLUSIVE and d["H4b"]["outside_50"] == 1
    d = A.compute_decisions(confirm(knee=[1.9, 2.0, 1.8, 2.1, 1.9, 1.0, 1.1, 2.0]), sel)["decisions"]
    assert d["H4b"]["label"] == A.REFUTED
    # retraction: a gap far above 0.05 nat -> refuted; around the bound -> inconclusive
    d = A.compute_decisions(confirm(kl=[0.5, 0.6, 0.5, 0.7, 0.5, 0.6, 0.5, 0.6]), sel)["decisions"]
    assert d["H5"]["label"] == A.REFUTED
    d = A.compute_decisions(confirm(kl=[0.01, 0.2, 0.01, 0.02, 0.01, 0.15, 0.01, 0.02]), sel)["decisions"]
    assert d["H5"]["label"] == A.INCONCLUSIVE
    # too few worlds: not testable, and an untested member still counts in the Holm family
    few = {w: recs for w, recs in list(confirm().items())[:4]}
    d = A.compute_decisions(few, sel)["decisions"]
    assert all(d[h]["label"].startswith(A.NOT_TESTABLE) for h in A.PRIMARY)
    assert d["H4b"]["label"] == A.NOT_TESTABLE and d["H5"]["label"] == A.NOT_TESTABLE


def test_statistics_helpers_match_the_cev11_package():
    from cev11 import stats as S
    d = [0.05, 0.02, 0.07, 0.01, 0.04, 0.06, 0.03, 0.05]
    ni = A.test_noninferiority(d, 0.05)
    assert ni["p"] == S.signflip_pvalue(d, -0.05) == pytest.approx(1 / 256)
    sup = A.test_superiority(d, 0.05)
    assert sup["p"] == S.signflip_pvalue(d, 0.0)
    assert A.label(dict(sup), 0.01) == A.INCONCLUSIVE           # significant, but the mean is below the SESOI
    assert A.label(dict(A.test_superiority([x + 0.05 for x in d], 0.05)), 0.01) == A.SUPPORTED
    assert A.label(dict(A.test_superiority([x - 0.05 for x in d], 0.05)), 1.0) == A.REFUTED
    assert A.label(dict(A.test_noninferiority([x - 0.2 for x in d], 0.05)), 1.0) == A.REFUTED
    g = A.guard([0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01], 0.02)
    assert g["ok"] and not A.guard([0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05], 0.02)["ok"]
    rows = _single(0, "X", "nll", 6, n=10) + _single(0, "X", "nll", 0, n=5, installed=False)
    for i, r in enumerate(rows):
        r["edit_id"] = r["key"] = f"e{i}"
    m = A.single_metrics(rows)
    assert m["n"] == 15 and m["WE"] == pytest.approx(10 / 15)
    assert m["PE6"] == pytest.approx(6 / 15) and m["IE5"] == pytest.approx(6 / 15) and m["G"] == pytest.approx(0.4)


# -------------------------------------------------------------------------------------------- freeze gate and jobs
def test_confirmatory_worlds_are_refused_while_the_protocol_is_not_frozen(tmp_path, capsys):
    # This test describes the study BEFORE its freeze. On the day PROTOCOL.md gets "Status: FROZEN <date>" it
    # fails on the next line by design: replace it then by a test of the frozen manifest.
    text = (STUDY / "PROTOCOL.md").read_text(encoding="utf-8")
    assert "NOT FROZEN" in text and V.protocol_frozen(text) is None
    assert V.protocol_frozen("# x\n\nStatus: FROZEN 2026-12-01\n") == "2026-12-01"
    assert V.protocol_frozen("Status: FROZEN soon\n") is None
    problems = V.check_preconditions(tmp_path / "selection.json", tmp_path / "manifest.sha256")
    assert any("not frozen" in p for p in problems) and any("selection" in p for p in problems)
    assert any("missing" in p for p in problems)
    assert V.freeze(tmp_path / "selection.json", tmp_path / "manifest.sha256") and not (tmp_path / "manifest.sha256").exists()
    with pytest.raises(SystemExit) as exc:
        CLI.world_main(["--run", "confirm", "--world", "1000", "--selection", str(tmp_path / "selection.json")])
    assert "refused" in str(exc.value)
    with pytest.raises(SystemExit):
        CLI.world_main(["--run", "discovery", "--world", "1000"])
    with pytest.raises(SystemExit):
        CLI.world_main(["--run", "confirm", "--world", "0"])
    with pytest.raises(SystemExit):
        CLI.world_main(["--run", "confirm", "--world", "1000", "--smoke"])
    # the selection and the analysis refuse incomplete inputs instead of writing a file
    assert CLI.select_main(["--results", str(tmp_path / "none"), "--out", str(tmp_path / "sel.json")]) == 1
    assert CLI.analyze_main(["--results", str(tmp_path / "none"), "--selection", str(tmp_path / "selection.json"),
                             "--out", str(tmp_path / "dec.json")]) == 1
    assert not (tmp_path / "sel.json").exists() and not (tmp_path / "dec.json").exists()
    # the manifest covers the protocol, the selection, the study code and the engine module
    names = set(V.frozen_files())
    assert {"PROTOCOL.md", "results/selection.json", "run_world.py", "analyze.py", "fwl/stream.py", "fwl/cli.py",
            "fwl/verification.py"} <= names
    assert any(n.startswith("engine:") and n.endswith("forward_learning.py") for n in names)
    # a valid selection file alone is not enough
    sel = A.select(discovery_records())
    (tmp_path / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    assert V.selection_problems(tmp_path / "selection.json") == []
    assert any("not frozen" in p for p in V.check_preconditions(tmp_path / "selection.json", tmp_path / "m"))
    capsys.readouterr()


def test_selection_is_tied_to_the_engine_and_configuration_of_the_discovery_records(tmp_path):
    def problems(discovery):
        path = tmp_path / "selection.json"
        path.write_text(json.dumps(A.select(discovery)), encoding="utf-8")
        return V.selection_problems(path)

    assert problems(discovery_records()) == []
    # discovery records made with another engine module than the one on disk: stale caps and stars
    assert any("engine module changed" in p for p in problems(discovery_records(engine="0" * 64)))
    # discovery records of a reduced (smoke-like) configuration cannot found a selection
    assert any("registered configuration" in p for p in problems(discovery_records(config=C.smoke_config())))
    # the two worlds disagree, or a world has no header
    mixed = discovery_records()
    mixed[1][0] = _meta(1, "discovery", C.protocol_config(), engine="1" * 64)
    assert any("do not share one engine" in p for p in problems(mixed))
    headless = discovery_records()
    headless[0] = headless[0][1:]
    assert any("do not share one engine" in p for p in problems(headless))
    # an old-schema selection is refused
    old = dict(A.select(discovery_records()), schema="fwl.selection.v1")
    old["sha256"] = C.sha256_json({k: v for k, v in old.items() if k != "sha256"})
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    assert any("schema" in p for p in V.selection_problems(tmp_path / "old.json"))


def test_freeze_writes_a_manifest_once_and_confirmatory_records_must_match_the_selection(tmp_path, monkeypatch):
    sel = A.select(discovery_records())
    sel_path, manifest = tmp_path / "selection.json", tmp_path / "freeze_manifest.sha256"
    sel_path.write_text(json.dumps(sel), encoding="utf-8")
    monkeypatch.setattr(V, "protocol_frozen", lambda text=None: "2026-12-01")       # as if the status line said so
    assert V.freeze(sel_path, manifest) == [] and manifest.exists()
    assert V.check_preconditions(sel_path, manifest) == []
    first = manifest.read_text(encoding="utf-8")
    assert V.freeze(sel_path, manifest) == [] and manifest.read_text(encoding="utf-8") == first   # same files: fine
    # anything changes after the freeze: the preconditions fail and a second freeze does not move the manifest
    changed = dict(sel, bp_star="BP-1")
    changed["sha256"] = C.sha256_json({k: v for k, v in changed.items() if k != "sha256"})
    sel_path.write_text(json.dumps(changed), encoding="utf-8")
    assert any("changed since the freeze: results/selection.json" in p
               for p in V.check_preconditions(sel_path, manifest))
    again = V.freeze(sel_path, manifest)
    assert again and "already frozen" in again[0] and manifest.read_text(encoding="utf-8") == first

    # confirmatory headers: the configuration (caps) of the frozen selection and the engine of the discovery records
    good = confirm(sel)
    assert V.run_header_problems(good, sel) == []
    uncapped = {w: [_meta(w, "confirm", C.protocol_config())] + recs[1:] for w, recs in good.items()}
    assert len(V.run_header_problems(uncapped, sel)) == 8
    assert all("configuration of the frozen selection" in p for p in V.run_header_problems(uncapped, sel))
    other = {w: [_meta(w, "confirm", C.protocol_config(sel), engine="2" * 64)] + recs[1:] for w, recs in good.items()}
    assert all("another engine" in p for p in V.run_header_problems(other, sel))
    assert all("no header" in p for p in V.run_header_problems(confirm(), sel))


def test_incomplete_or_foreign_records_are_reported(tmp_path):
    recs = synthetic_world(1000)
    path = C.results_path(tmp_path, 1000)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    report = V.check_records(tmp_path)
    assert any("no header" in p for p in report["problems"])
    path.write_text(json.dumps({"schema": "x"}) + "\n", encoding="utf-8")
    assert V.check_records(tmp_path)["problems"]
    assert V.check_records(tmp_path / "empty")["problems"]


def test_rerun_detects_changed_records_and_crosscheck_compares_with_cev11_records(tmp_path, monkeypatch):
    sel = A.select(discovery_records())
    sel_path, dec_path, results = tmp_path / "selection.json", tmp_path / "decisions.json", tmp_path / "confirm"
    sel_path.write_text(json.dumps(sel), encoding="utf-8")
    results.mkdir()
    per_world = confirm(sel)
    for w, recs in per_world.items():
        C.results_path(results, w).write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    assert V.rerun(results, sel_path, dec_path) and not dec_path.exists()         # no decisions file yet
    decisions = json.loads(json.dumps(A.compute_decisions(per_world, sel)))
    decisions["records_sha256"] = {str(w): C.sha256_file(C.results_path(results, w)) for w in per_world}
    decisions["freeze"] = {"ok": False, "problems": ["not frozen"]}                # written by analyze.py, not compared
    dec_path.write_text(json.dumps(decisions), encoding="utf-8")
    assert V.rerun(results, sel_path, dec_path) == []
    # a record that changed after the analysis is found twice: by its file hash and by the recomputation
    recs = per_world[1003]
    for r in recs:
        if r["kind"] == "single" and r["arm"] == "FEC-A" and r["install"] == "nll":
            r["PF6"] = {c: 0 for c in r["PF6"]}
            r["targets"][0]["informal"] = 0
    C.results_path(results, 1003).write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    problems = V.rerun(results, sel_path, dec_path)
    assert any("world 1003 changed" in p for p in problems) and any("differ" in p for p in problems)

    # comparator cross-check against (here synthetic) CEV-11 records of the same edits
    monkeypatch.setattr(C, "CEV11", tmp_path / "cev11")
    edits = tmp_path / "cev11" / "results" / "edits"
    edits.mkdir(parents=True)
    ours = [r for r in per_world[1000] if r["kind"] == "single" and r["arm"] == "FT-L"]
    theirs = [{"edit_id": r["edit_id"], "installed": r["installed"], "PF6": r["PF6"], "targets": r["targets"]}
              for r in ours]
    (edits / "confirm_1000_FT-L.jsonl").write_text("\n".join(json.dumps(r) for r in theirs) + "\n",
                                                   encoding="utf-8")
    report = V.crosscheck(results)
    assert report == {"compared": len(ours), "mismatches": []}
    theirs[2]["PF6"] = {c: 1 - v for c, v in theirs[2]["PF6"].items()}
    (edits / "confirm_1000_FT-L.jsonl").write_text("\n".join(json.dumps(r) for r in theirs) + "\n",
                                                   encoding="utf-8")
    assert V.crosscheck(results)["mismatches"] == [f"w1000 FT-L {ours[2]['edit_id']}"]


@pytest.mark.parametrize("name", ["jobs_discovery.json", "jobs_confirm.json"])
def test_job_lists_are_well_formed(name):
    jobs = json.loads((STUDY / name).read_text(encoding="utf-8"))
    assert isinstance(jobs, list) and jobs
    ids = [j["id"] for j in jobs]
    assert len(set(ids)) == len(ids)
    for j in jobs:
        assert set(j) == {"id", "cmd", "deps", "log"}
        assert all(d in ids for d in j["deps"]) and j["id"] not in j["deps"]
        scripts = [tok for tok in j["cmd"].split() if tok.endswith(".py")]
        assert scripts and all((REPO / s).is_file() for s in scripts), j["cmd"]
        assert j["log"].startswith("research/forward_learning_2026_11_13/") and j["log"].endswith(".log")
        assert "OMP_NUM_THREADS=1" in j["cmd"]
    text = " ".join(j["cmd"] for j in jobs)
    if name == "jobs_discovery.json":
        assert "--run confirm" not in text and "--world 0" in text and "--world 1" in text
        assert any("run_select.py" in j["cmd"] for j in jobs)
    else:
        pre = [j["id"] for j in jobs if "--preconditions" in j["cmd"]]
        assert len(pre) == 1
        worlds = [j for j in jobs if "run_world.py" in j["cmd"]]
        assert len(worlds) == 8 and all(pre[0] in j["deps"] and "--run confirm" in j["cmd"] for j in worlds)
        assert all(f"--world {w}" in text for w in C.CONFIRMATORY_WORLDS)
        assert any("analyze.py" in j["cmd"] for j in jobs) and any("--rerun" in j["cmd"] for j in jobs)
