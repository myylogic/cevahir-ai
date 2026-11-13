"""Weight keys, the weight-schema registry and migrations (torch part of the key system).

The weights key is a per-tensor table (name, dtype, shape, alias, sha256). The expected layout of
an architecture is read from the constructor on the meta device and pinned by golden snapshots.
"""
import copy
import json
import pathlib
import sys
import warnings

import pytest

torch = pytest.importorskip("torch")

from model_management.config_schema import tiny_model_config  # noqa: E402
from model_management.model_manager import ModelManager  # noqa: E402

from cevahir_ai.adapters import model_build  # noqa: E402
from cevahir_ai.errors import MigrationError, UnsupportedTensor  # noqa: E402
from cevahir_ai.keys import Key, schemas  # noqa: E402
from cevahir_ai.keys.torchkeys import migrate, optim, tensors, weights, wschema  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
STUDY = ROOT / "research" / "living_learning_paradigm_2026_11_13"


def build(seed=0, **overrides):
    torch.manual_seed(seed)
    manager = ModelManager(tiny_model_config(**overrides))
    manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    return manager.model.eval()


@pytest.fixture(scope="module")
def model():
    return build()


# ----------------------------------------------------------------------------------- weights key
def test_weights_key_names_what_changed(model):
    state = model.state_dict()
    key, table = weights.weights_key(state)
    assert key.scheme == "sha256-v1" and len(table) == 25 and weights.weights_key(model.state_dict())[0] == key
    changed = {name: tensor.clone() for name, tensor in state.items()}
    changed["output_layer.weight"] = changed["embedding.embedding.weight"]            # keep the tie
    changed["layers.1.ffn.fc2.weight"][0, 0] += 1.0
    other_key, other_table = weights.weights_key(changed)
    assert other_key != key
    delta = weights.diff_tables(table, other_table)
    assert delta == {"only_left": [], "only_right": [], "changed": [["layers.1.ffn.fc2.weight", "values"]]}


def test_weights_key_sensitivity_and_invariance(model):
    state = model.state_dict()
    key = weights.weights_key(state)[0]
    clone = {name: tensor.clone() for name, tensor in state.items()}
    clone["output_layer.weight"] = clone["embedding.embedding.weight"]
    assert weights.weights_key(clone)[0] == key                                       # another copy, same bytes
    # dtype is identity: an fp16 copy is another key
    half = {name: tensor.half() for name, tensor in clone.items()}
    half["output_layer.weight"] = half["embedding.embedding.weight"]
    assert weights.weights_key(half)[0] != key
    # a name or a shape is identity
    renamed = {("x." + name if name == "output_norm.scale" else name): tensor for name, tensor in clone.items()}
    reshaped = {**clone, "output_norm.scale": clone["output_norm.scale"].reshape(1, -1)}
    assert len({key, weights.weights_key(renamed)[0], weights.weights_key(reshaped)[0]}) == 3
    # memory layout is not: a non-contiguous view with the same logical content hashes the same
    matrix = clone["layers.0.attn.out_proj.weight"]
    strided = matrix.t().contiguous().t()
    assert not strided.is_contiguous() and torch.equal(strided, matrix)
    assert weights.weights_key({**clone, "layers.0.attn.out_proj.weight": strided})[0] == key
    assert tensors.tensor_sha256(torch.zeros(0)) == tensors.tensor_sha256(torch.zeros(0, 3))  # sha256(b"")


def test_tied_weights_are_one_alias_group_and_hashed_once(model):
    table = weights.weights_key(model.state_dict())[1]
    rows = {row[0]: row for row in table}
    assert rows["output_layer.weight"][3] == "embedding.embedding.weight" and rows["output_layer.weight"][4] is None
    assert rows["embedding.embedding.weight"][3] is None and len(rows["embedding.embedding.weight"][4]) == 64
    digests = tensors.tensor_digests(table)
    assert digests["output_layer.weight"] == digests["embedding.embedding.weight"]
    assert tensors.layout_of_table(table)["aliases"] == [["embedding.embedding.weight", "output_layer.weight"]]
    _, n_params = weights.dtype_summary(table)
    assert n_params == sum(parameter.numel() for parameter in model.parameters())      # the tie is counted once
    # The same values in separate storages are not an alias group: the tie is broken in such a file.
    untied = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    assert tensors.layout_of_table(weights.weights_key(untied)[1])["aliases"] == []
    expected = wschema.layout_of_module(model)
    assert wschema.broken_ties(untied, expected) == []                                  # equal values: not broken
    untied["output_layer.weight"] = untied["output_layer.weight"] + 1
    assert wschema.broken_ties(untied, expected) == [["embedding.embedding.weight", "output_layer.weight"]]


def test_unsupported_tensors_are_rejected_by_both_weight_schemes():
    cases = {"number": 3, "sparse": torch.eye(2).to_sparse(), "meta": torch.empty(2, device="meta")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # quantized tensor factories are deprecated in recent torch
        try:
            cases["quantized"] = torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)
        except (AttributeError, RuntimeError, NotImplementedError):
            pass
    for name, value in cases.items():
        for scheme in (weights.weights_key, weights.weights_cev11_fp32_key):
            with pytest.raises(UnsupportedTensor, match=name if name != "number" else "not a tensor"):
                scheme({name: value})


def test_cev11_weight_digest_is_reproduced_exactly(model):
    """weights:cev11-fp32-v1 is cev11.model.model_sha256: no CEV-11 checkpoint digest is invalidated."""
    for entry in (str(ROOT), str(STUDY)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from cev11 import model as cev11_model
    assert weights.weights_cev11_fp32_digest(model.state_dict()) == cev11_model.model_sha256(model)
    other = build(seed=1, num_layers=3)
    assert weights.weights_cev11_fp32_key(other.state_dict()) == Key("weights", "cev11-fp32-v1", cev11_model.model_sha256(other))
    # The legacy digest erases dtype (it casts to fp32); the canonical key does not.
    half = {name: tensor.half().float() for name, tensor in model.state_dict().items()}
    assert weights.weights_cev11_fp32_digest({n: t.half() for n, t in model.state_dict().items()}) == \
        weights.weights_cev11_fp32_digest(half)


def test_training_state_keys(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = optim.optim_content_key(optimizer.state_dict())
    assert first == optim.optim_content_key(copy.deepcopy(optimizer.state_dict()))
    model(torch.tensor([[1, 2, 3]]))[0].sum().backward()
    optimizer.step()
    assert optim.optim_content_key(optimizer.state_dict()) != first                    # two resume points differ
    rng = {"python": (3, (1, 2, 3), None), "torch": torch.get_rng_state(), "big": 2**70, "nan": float("nan")}
    assert optim.tstate_content_key(rng) == optim.tstate_content_key(copy.deepcopy(rng))
    assert optim.tstate_content_key({**rng, "python": (3, (1, 2, 4), None)}) != optim.tstate_content_key(rng)
    assert optim.sched_content_key({"last_epoch": 3}) != optim.sched_content_key({"last_epoch": 4})


# -------------------------------------------------------------------------- layout from the code
LAYOUT_CASES = {
    "tiny": {},
    "eight-heads": {"num_heads": 8},
    "kv-heads": {"num_kv_heads": 2},
    "untied": {"tie_weights": False},
    "moe": {"use_moe": True},
    "qk-norm": {"use_qk_norm": True},
    "relu-ffn-layernorm": {"use_swiglu": False, "use_rmsnorm": False},
    "sinusoidal": {"pe_mode": "sinusoidal"},
    "learned": {"pe_mode": "learned"},
    "yarn": {"rope_scaling_type": "yarn", "rope_scaling_factor": 2.0},
}


@pytest.mark.parametrize("name", sorted(LAYOUT_CASES))
def test_meta_layout_equals_the_real_build(name):
    """The cheap structural build (meta device) gives exactly the layout the real constructor gives."""
    config = tiny_model_config(**LAYOUT_CASES[name])
    before = torch.get_rng_state()
    expected = model_build.expected_layout(config)
    assert torch.equal(torch.get_rng_state(), before), "a layout readout shifted the global RNG"
    real = wschema.layout_of_module(build(**LAYOUT_CASES[name]))
    for part in ("tensors", "aliases", "non_persistent"):
        assert expected[part] == real[part], part
    assert expected["built_on"] == ("meta (RoPE scaling neutralised)" if name == "yarn" else "meta")
    assert wschema.layout_keys(expected) == wschema.layout_keys(real)


def test_meta_build_allocates_nothing_and_handles_the_full_size_model():
    meta_model, how = model_build.structure_model(schemas.preset_config("train-default"))
    assert how.startswith("meta") and {parameter.device.type for parameter in meta_model.parameters()} == {"meta"}
    assert sum(parameter.numel() for parameter in meta_model.parameters()) > 50_000_000


def test_layout_flags_that_change_names_and_flags_that_do_not():
    base = wschema.layout_keys(model_build.expected_layout(tiny_model_config()))[0]
    same_layout = ({"num_heads": 8}, {"parallel_residual": True}, {"sliding_window": 8}, {"causal_mask": False})
    for overrides in same_layout:       # the function changes, the layout does not: why the arch key exists
        assert wschema.layout_keys(model_build.expected_layout(tiny_model_config(**overrides)))[0] == base, overrides
        assert model_build.arch_of_config(tiny_model_config(**overrides))[0] != model_build.arch_of_config(tiny_model_config())[0]
    for overrides in ({"tie_weights": False}, {"use_qk_norm": True}, {"use_swiglu": False}, {"use_rmsnorm": False},
                      {"use_moe": True}, {"num_kv_heads": 2}, {"num_layers": 3}):
        assert wschema.layout_keys(model_build.expected_layout(tiny_model_config(**overrides)))[0] != base, overrides
    untied = model_build.expected_layout(tiny_model_config(tie_weights=False))
    assert untied["aliases"] == [] and "output_layer.bias" in {entry[0] for entry in untied["tensors"]}


# ------------------------------------------------------------------------------ golden snapshots
def test_golden_layout_snapshots_match_the_current_code():
    assert schemas.check() == []
    tiny = schemas.load_layout("tiny")
    assert len(tiny["layout"]["tensors"]) == 25 and tiny["layout"]["aliases"] == [["embedding.embedding.weight", "output_layer.weight"]]
    assert tiny["non_persistent"] == ["pos_encoding.pos_idx", "pos_encoding.rope_freqs"]
    assert tiny["layout_key"] == "cev:wschema:layout-v1:f96311fd51ccd68dbd49718923ed53e6d1df7bf2066d73ad0cce04fa00ef1968"
    assert tiny["arch"] == "cev:arch:cfg-v1:3fb0c6d99edf43409bc67d104b8f87a27823b5235479dd22474df4c5458ebf67"
    assert schemas.load_layout("cev11-primary")["arch_fields"]["num_heads"] == 4
    assert schemas.load_layout("train-default")["arch_fields"]["num_kv_heads"] == 2
    for preset in schemas.PRESETS:
        document = schemas.load_layout(preset)
        assert schemas.dump(document) == schemas.layout_path(preset).read_text(encoding="utf-8")
        assert schemas.preset_of_layout_key(document["layout_key"]) == preset
    assert schemas.preset_of_layout_key("cev:wschema:layout-v1:" + "0" * 64) is None


def test_a_layout_change_is_caught_by_the_check(monkeypatch):
    monkeypatch.setitem(schemas.PRESET_CONFIGS, "cev11-primary", {**schemas.PRESET_CONFIGS["cev11-primary"], "use_qk_norm": True})
    findings = schemas.check(("tiny", "cev11-primary"))
    assert len(findings) == 1 and findings[0].startswith("cev11-primary: the current code no longer builds the recorded layout")
    assert "tensor names changed" in findings[0]


def test_schema_cli_check(capsys):
    from cevahir_ai.keys.schemas.__main__ import main
    assert main(["--check"]) == 0
    assert "3 layout snapshots match" in capsys.readouterr().out


# ----------------------------------------------------------------------------------- migrations
def test_strip_wrapper_prefixes_migration_preserves_the_function(model):
    migration = schemas.load_migration("strip-wrapper-prefixes/1")
    assert migration["function_preserving"] is True and schemas.migration_ids() == ["strip-wrapper-prefixes/1"]
    tokens = torch.tensor([[1, 5, 9, 2]])
    with torch.no_grad():
        expected = model(tokens)[0]
    for prefix in ("module.", "_orig_mod.", "module._orig_mod."):
        wrapped = {prefix + name: tensor for name, tensor in model.state_dict().items()}
        before = dict(wrapped)
        migrated, applied = migrate.apply_migration(wrapped, migration)
        assert wrapped == before and list(wrapped) == list(before)                    # pure: the input is untouched
        assert sorted(migrated) == sorted(model.state_dict()) and applied
        assert weights.weights_key(migrated)[0] == weights.weights_key(model.state_dict())[0]
        fresh = build(seed=7)
        fresh.load_state_dict(migrated, strict=True)
        with torch.no_grad():
            assert torch.equal(fresh(tokens)[0], expected)                            # identical logits
    untouched, applied = migrate.apply_migration(model.state_dict(), migration)
    assert applied == [] and sorted(untouched) == sorted(model.state_dict())


def test_migration_operations():
    def doc(*ops):
        return {"format": "cevahir-wschema-migration", "version": 1, "id": "t/1", "ops": list(ops)}

    a, b = torch.arange(4.0), torch.arange(4.0)
    state = {"old.w": a, "tied.w": b, "drop.me": torch.zeros(1), "layers.0.fc1.weight": torch.ones(2)}
    out, applied = migrate.apply_migration(state, doc(
        {"op": "rename", "from": "old.w", "to": "new.w"},
        {"op": "rename_regex", "pattern": r"layers\.(\d+)\.fc1\.(.*)", "repl": r"layers.\1.gate_up_proj.\2"},
        {"op": "drop", "name": "drop.me", "reason": "derived buffer"},
        {"op": "tie", "canonical": "new.w", "member": "tied.w"}))
    assert sorted(out) == ["layers.0.gate_up_proj.weight", "new.w", "tied.w"] and len(applied) == 4
    assert out["tied.w"] is out["new.w"] and tensors.alias_groups(out) == {"tied.w": "new.w"}
    untied, _ = migrate.apply_migration(out, doc({"op": "untie", "name": "tied.w"}))
    assert tensors.alias_groups(untied) == {}
    half = {"w": torch.ones(2, dtype=torch.float16), "t": None, "i": torch.ones(2, dtype=torch.int64)}
    half["t"] = half["w"]
    up, _ = migrate.apply_migration(half, doc({"op": "cast", "dtype": "float32"}))
    assert up["w"].dtype == torch.float32 and up["i"].dtype == torch.int64 and tensors.alias_groups(up) == {"w": "t"}
    with pytest.raises(MigrationError, match="lose precision"):
        migrate.apply_migration({"w": torch.ones(2, dtype=torch.float64)}, doc({"op": "cast", "dtype": "float32"}))
    with pytest.raises(MigrationError, match="differ"):
        migrate.apply_migration({"a": torch.ones(2), "b": torch.zeros(2)}, doc({"op": "tie", "canonical": "a", "member": "b"}))
    with pytest.raises(MigrationError, match="two tensors"):
        migrate.apply_migration({"module.w": a, "w": b}, doc({"op": "strip_prefix", "prefix": "module."}))
    with pytest.raises(MigrationError, match="reason"):
        migrate.apply_migration(state, doc({"op": "drop", "name": "drop.me"}))
    with pytest.raises(MigrationError, match="unknown migration operation"):
        migrate.apply_migration(state, doc({"op": "transmute"}))
    with pytest.raises(MigrationError):
        migrate.apply_migration(state, {"format": "something-else", "version": 1, "ops": []})
    with pytest.raises(KeyError):
        schemas.load_migration("does-not-exist/1")
    for path in schemas.MIGRATION_DIR.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["format"] == "cevahir-wschema-migration"
