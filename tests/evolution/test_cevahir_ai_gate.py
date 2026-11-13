"""The compatibility gate on real files: every envelope shape, the governed load path and the CLI.

Checkpoints are written by the real writers (training CheckpointManager, ModelManager.save,
ModelSaver, the CEV-11 writer) into temporary directories, then inspected and loaded through the
gate. Regressions pinned here: a different head count with identical shapes, a bare state dict,
an fp16 copy, a tampered file, a stripped tokenizer identity, a pickled module.
"""
import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from model_management.config_schema import tiny_model_config  # noqa: E402
from model_management.model_manager import ModelManager  # noqa: E402
from model_management.model_saver import ModelSaver  # noqa: E402
from tokenizer_management.byte_bpe import ByteLevelBPE  # noqa: E402

from cevahir_ai.adapters import envelopes, tokenizers  # noqa: E402
from cevahir_ai.adapters.inspection import Target, inspect_checkpoint  # noqa: E402
from cevahir_ai.errors import EnvelopeError, IncompatibleCheckpoint  # noqa: E402
from cevahir_ai.keys import Key  # noqa: E402
from cevahir_ai.keys.cli import main as keys_cli  # noqa: E402
from cevahir_ai.keys.torchkeys import weights  # noqa: E402
from cevahir_ai.model.checkpoint import open_checkpoint, read_envelope  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
STUDY = ROOT / "research" / "living_learning_paradigm_2026_11_13"
BASES = STUDY / "results" / "bases"
TOKENS = torch.tensor([[1, 7, 9, 3]])
LEGACY_TOK = "ab" * 32
PURPOSES = ("inspect", "evaluate", "finetune", "resume", "serve", "publish", "export")


def manager(seed=0, optimizer=True, **overrides):
    torch.manual_seed(seed)
    result = ModelManager(tiny_model_config(**overrides))
    result.initialize(build_optimizer=optimizer, build_criterion=False, build_scheduler=optimizer)
    result.model.eval()
    return result


def tm3_envelope(source):
    """The envelope of training_management.v3's CheckpointManager (its ``CheckpointData`` schema)."""
    return {"epoch": 4, "model_state": source.model.state_dict(), "optimizer_state": source.optimizer.state_dict(),
            "scheduler_state": None, "ema_state": None, "swa_state": None, "metrics": {"loss": 1.0},
            "train_config": dict(tiny_model_config()), "saved_at": "2026-11-13T00:00:00+00:00"}


def cev11():
    for entry in (str(ROOT), str(STUDY)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from cev11 import model as cev11_model
    return cev11_model


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """One file per envelope shape, all holding the weights of the same seeded tiny model (4 heads)."""
    from training_management.v2.utils.checkpoint_manager import CheckpointManager
    directory = tmp_path_factory.mktemp("envelopes")
    source = manager()
    paths = {"source": source}
    trained = manager()
    trained.model._tokenizer_identity = LEGACY_TOK                 # what the training service stamps on the model
    extra = {"format_version": 1, "tokenizer_identity": LEGACY_TOK,
             "scheduler": {"scheduler_type": "cosine", "scheduler_state": trained.scheduler.state_dict()},
             "loop": {"global_step": 3}, "best_val_loss": 1.0, "rng": {"torch": torch.get_rng_state()}}
    paths["tm2"] = CheckpointManager(checkpoint_model_dir=str(directory / "tm2")).save(
        trained.model, trained.optimizer, 3, metric=1.5, extra_state=extra)
    paths["mm-save"] = source.save(str(directory / "mm.pth"), epoch=2)
    paths["saver-4.1"] = ModelSaver.save_checkpoint(source.model, optimizer=source.optimizer, scheduler=source.scheduler,
                                                    epoch=1, save_dir=str(directory / "saver"))
    paths["bare"] = ModelSaver.save_weights_only(source.model, save_dir=str(directory / "bare"))
    paths["module"] = ModelSaver.save_full_model(source.model, save_dir=str(directory / "full"))
    paths["tm3"] = str(directory / "tm3.pth")
    torch.save(tm3_envelope(source), paths["tm3"])
    study = cev11()
    tokenizer = ByteLevelBPE.train(["merhaba dünya merhaba", "dünya güzel bir yer"] * 20, vocab_size=290)
    cev_model, cfg = study.build_model(vocab_size=tokenizer.vocab_size, layers=2, d=32, heads=4, kv_heads=4, max_len=64, seed=0)
    paths["cev11-v1"] = str(directory / "cev11" / "model.pt")
    study.save_checkpoint(paths["cev11-v1"], cev_model, cfg, tokenizer, {"note": "test"})
    paths["cev11_model"], paths["cev11_tokenizer"] = cev_model, tokenizer
    paths["cev11-ledger-v1"] = str(directory / "ledger.pt")
    torch.save({"format": "cev11-ledger", "version": 1, "identity": {"base_sha256": "0" * 64, "layer": 3}, "W0": torch.zeros(2)},
               paths["cev11-ledger-v1"])
    return paths


ENVELOPES = ("tm2", "mm-save", "saver-4.1", "bare", "tm3", "cev11-v1")


# ------------------------------------------------------------------------------ envelope readers
def test_envelope_reader_recognises_every_shape(files):
    for name in ENVELOPES:
        view = read_envelope(files[name])
        assert view.envelope == name and len(view.state) == 25
    assert read_envelope(files["cev11-ledger-v1"]).envelope == "cev11-ledger-v1"
    tm2 = read_envelope(files["tm2"])
    assert tm2.tokenizer == ("legacy-v3", LEGACY_TOK) and tm2.epoch == 3 and tm2.metric == 1.5
    assert tm2.config_source == "model_config" and tm2.model_config["num_heads"] == 4
    assert tm2.optimizer_state["param_groups"] and tm2.scheduler_state["scheduler_type"] == "cosine"
    saver = read_envelope(files["saver-4.1"])
    assert saver.declared["cevahir_version"] == "4.1" and saver.declared["checkpoint_format"] == 2
    assert any("self-referential" in note for note in saver.notes)
    cev = read_envelope(files["cev11-v1"])
    assert cev.tokenizer == ("bytebpe-v1", files["cev11_tokenizer"].identity()) and cev.tokenizer_file == "model.tokenizer.json"
    assert cev.declared["weights:cev11-fp32-v1"] == cev11().model_sha256(files["cev11_model"])
    assert read_envelope(files["bare"]).model_config is None and read_envelope(files["tm3"]).config_source == "train_config"


def test_tm3_fixture_has_the_key_set_of_the_real_v3_writer(files):
    """The tm3 file in these tests is hand-built; its keys are pinned to the writer's own schema."""
    import ast
    source = (ROOT / "training_management/v3/utils/checkpoint_manager.py").read_text(encoding="utf-8-sig")
    schema = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef) and node.name == "CheckpointData")
    declared = {node.target.id for node in schema.body if isinstance(node, ast.AnnAssign)}
    assert declared == set(tm3_envelope(files["source"]))
    view = read_envelope(files["tm3"])
    assert view.epoch == 4 and view.optimizer_state["param_groups"] and view.tokenizer is None


def test_envelope_reader_agrees_with_the_legacy_unpacker_on_the_shapes_it_knows(files):
    from model_management.checkpoint_contract import unpack_checkpoint
    for name in ("tm2", "mm-save", "saver-4.1", "bare"):
        checkpoint = torch.load(files[name], map_location="cpu", weights_only=True)
        state, optimizer_state, scheduler_state, meta = unpack_checkpoint(checkpoint)
        view = read_envelope(files[name])
        assert sorted(view.state) == sorted(state) and all(torch.equal(view.state[k], state[k]) for k in state)
        assert (view.model_config or {}) == meta["config"] and view.epoch == meta["epoch"]
        assert (view.tokenizer[1] if view.tokenizer else None) == meta.get("tokenizer_identity")
        assert (view.optimizer_state is None) == (optimizer_state is None)
    # The shapes the legacy unpacker does not understand lose their identities there, not here.
    cev = torch.load(files["cev11-v1"], map_location="cpu", weights_only=True)
    assert unpack_checkpoint(cev)[3] == {"config": {}, "epoch": None, "tokenizer_identity": None}


def test_pickled_module_is_refused_without_being_unpickled(files):
    sniff = envelopes.sniff_pickle_globals(files["module"])
    assert sniff["container"] == "torch-zip" and sniff["unsafe"] == ["src.neural_network.CevahirNeuralNetwork"]
    with pytest.raises(EnvelopeError, match="code execution"):
        read_envelope(files["module"])
    for name in ENVELOPES:
        assert envelopes.sniff_pickle_globals(files[name])["unsafe"] == [], name
    report = inspect_checkpoint(files["module"], purpose="inspect")
    assert report.envelope == "module" and report.facts["C12"].status == "fail" and not report.verdict.ok


def test_safety_sniff_sees_an_arbitrary_global(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo should-never-run",))

    path = tmp_path / "evil.pth"
    torch.save({"model_state_dict": {"w": torch.zeros(1)}, "payload": Evil()}, path)
    sniff = envelopes.sniff_pickle_globals(path)
    assert any(name.endswith(".system") for name in sniff["unsafe"])
    report = inspect_checkpoint(path, purpose="inspect")
    assert report.facts["C12"].status == "fail" and report.verdict.verdict == "incompatible" and report.state is None
    text = tmp_path / "notes.pth"
    text.write_bytes(b"not a checkpoint")
    assert inspect_checkpoint(text, purpose="inspect").facts["C12"].status == "unsupported"


# ------------------------------------------------------------------- envelope x purpose matrix
# verdict per purpose in PURPOSES order: inspect, evaluate, finetune, resume, serve, publish, export
W, C, X = "compatible-with-warnings", "compatible", "incompatible"
MATRIX = {
    #             inspect evaluate finetune resume serve publish export
    "tm2":       (W,      W,       W,       W,     X,    X,      X),
    "mm-save":   (W,      W,       W,       W,     X,    X,      X),
    "saver-4.1": (C,      W,       W,       W,     X,    X,      C),
    "bare":      (W,      X,       X,       X,     X,    X,      X),
    "tm3":       (W,      W,       W,       W,     X,    X,      X),
    "cev11-v1":  (W,      W,       W,       X,     X,    X,      X),
    "module":    (X,      X,       X,       X,     X,    X,      X),
}


@pytest.mark.parametrize("name", sorted(MATRIX))
def test_envelope_purpose_matrix(files, name):
    """Golden verdicts with no target given: what each existing file is good for under the current code."""
    for purpose, expected in zip(PURPOSES, MATRIX[name]):
        report = inspect_checkpoint(files[name], purpose=purpose)
        assert report.verdict.verdict == expected, (name, purpose, report.verdict.summary_en)
    report = inspect_checkpoint(files[name], purpose="evaluate")
    if name == "bare":
        assert set(report.verdict.failing) == {"C4", "C5"} and "architecture unknown" in report.verdict.summary_en
    if name == "cev11-v1":
        assert report.facts["C3"].status == "pass" and report.facts["C7"].status == "pass" and report.intact == "yes"
        assert inspect_checkpoint(files[name], purpose="resume").verdict.failing == ("C10",)
    if name == "saver-4.1":
        assert report.facts["C1"].status == "pass" and report.intact == "yes"


def test_same_weights_have_one_key_in_every_container(files):
    expected = weights.weights_key(files["source"].model.state_dict())[0]
    reports = {name: inspect_checkpoint(files[name], purpose="inspect") for name in ("tm2", "mm-save", "saver-4.1", "bare", "tm3")}
    assert {report.keys["weights"] for report in reports.values()} == {expected}
    assert len({report.keys["file"] for report in reports.values()}) == 5               # only the file key differs
    assert len({report.keys["wschema"] for report in reports.values()}) == 1
    # A training checkpoint and a weights-only export share weights, not the checkpoint key.
    assert reports["tm2"].keys["ckpt"] != reports["bare"].keys["ckpt"]
    assert "optim" in reports["tm2"].keys and "optim" not in reports["bare"].keys
    assert reports["tm2"].keys["arch"] == reports["mm-save"].keys["arch"] == reports["tm3"].keys["arch"]
    assert Key("arch", "ctor-v0", reports["tm2"].aliases[0].digest) in reports["tm2"].aliases


# ------------------------------------------------------------------- regression: wrong head count
def test_checkpoint_with_another_head_count_but_identical_shapes_is_rejected(files):
    """4-head weights have exactly the shapes of an 8-head model; only the declared architecture differs."""
    eight = manager(seed=5, num_heads=8)
    four_state = files["source"].model.state_dict()
    assert {n: t.shape for n, t in four_state.items()} == {n: t.shape for n, t in eight.model.state_dict().items()}
    before = copy.deepcopy(eight.model.state_dict())
    for name in ("tm2", "mm-save", "saver-4.1", "tm3"):
        report = inspect_checkpoint(files[name], purpose="evaluate", target=Target(model=eight.model))
        assert report.facts["C4"].status == "pass"                                    # the shapes do match
        assert report.facts["C5"].status == "fail" and "num_heads: checkpoint=4, target=8" in report.facts["C5"].detail
        assert report.verdict.verdict == "incompatible" and "C5" in report.verdict.failing
        with pytest.raises(IncompatibleCheckpoint, match="num_heads: checkpoint=4, target=8") as refusal:
            open_checkpoint(files[name], model=eight.model)
        assert refusal.value.exit_code == 5 and refusal.value.verdict.verdict == "incompatible"
    for name, tensor in eight.model.state_dict().items():                             # refused before any mutation
        assert torch.equal(tensor, before[name])
    # The same weights computed by 4 and by 8 heads are different functions: that is what the gate prevents.
    eight.model.load_state_dict(four_state)
    with torch.no_grad():
        assert not torch.allclose(eight.model(TOKENS)[0], files["source"].model(TOKENS)[0])


def test_bare_state_dict_is_rejected_because_its_architecture_is_unknown(files):
    for target_heads in (4, 8):                                                       # even the right model: no proof
        target = manager(seed=5, num_heads=target_heads)
        report = inspect_checkpoint(files["bare"], purpose="evaluate", target=Target(model=target.model))
        assert report.facts["C4"].status == "pass" and report.facts["C5"].status == "unknown"
        assert report.verdict.verdict == "incompatible" and report.verdict.failing == ("C5",)
        assert "architecture unknown" in report.verdict.summary_en
        with pytest.raises(IncompatibleCheckpoint, match="architecture unknown"):
            open_checkpoint(files["bare"], model=target.model)
    # The legacy loader accepts this file into the 8-head model: the gap the gate closes.
    legacy = manager(seed=5, num_heads=8)
    legacy.load(files["bare"], weights_only=True)
    assert torch.equal(legacy.model.state_dict()["layers.0.attn.query_proj.weight"],
                       files["source"].model.state_dict()["layers.0.attn.query_proj.weight"])


def test_cev11_checkpoint_is_rejected_by_a_product_model_with_other_heads(files):
    target = manager(seed=5, num_heads=8, vocab_size=files["cev11_tokenizer"].vocab_size)
    report = inspect_checkpoint(files["cev11-v1"], purpose="evaluate", target=Target(model=target.model))
    assert report.facts["C5"].status == "fail" and "num_heads" in report.facts["C5"].detail
    assert report.verdict.verdict == "incompatible"


def test_other_architecture_differences_name_the_field(files):
    for overrides, field in (({"parallel_residual": True}, "parallel_residual"), ({"sliding_window": 8}, "sliding_window"),
                             ({"num_layers": 3}, "num_layers")):
        report = inspect_checkpoint(files["tm2"], purpose="evaluate", target=Target(config=tiny_model_config(**overrides)))
        assert report.facts["C5"].status == "fail" and field in report.facts["C5"].detail
    same = inspect_checkpoint(files["tm2"], purpose="evaluate", target=Target(config=tiny_model_config(dropout=0.4, device="cuda")))
    assert same.facts["C5"].status == "pass" and same.verdict.ok                       # training-only fields differ


# ----------------------------------------------------------------- regression: dtype, tampering
def test_fp16_weights_are_not_silently_cast(files, tmp_path):
    checkpoint = torch.load(files["mm-save"], map_location="cpu", weights_only=True)
    for field in ("state_dict", "model_state_dict"):
        checkpoint[field] = {name: tensor.half() for name, tensor in checkpoint[field].items()}
    path = tmp_path / "half.pth"
    torch.save(checkpoint, path)
    report = inspect_checkpoint(path, purpose="serve")
    assert report.facts["C9"].status == "fail" and "float16" in report.facts["C9"].detail and "C9" in report.verdict.failing
    assert report.keys["weights"] != inspect_checkpoint(files["mm-save"], purpose="inspect").keys["weights"]
    allowed = inspect_checkpoint(path, purpose="serve", target=Target(allow_cast=True))
    assert allowed.facts["C9"].status == "pass"
    down = inspect_checkpoint(files["mm-save"], purpose="serve", target=Target(dtype="float16", allow_cast=True))
    assert down.facts["C9"].status == "fail"                                          # never down-cast


def test_tampered_file_is_caught_by_sidecar_or_supplied_key(files, tmp_path):
    directory = tmp_path / "saver"
    shutil.copytree(pathlib.Path(files["saver-4.1"]).parent, directory)
    path = directory / pathlib.Path(files["saver-4.1"]).name
    genuine = inspect_checkpoint(path, purpose="inspect")
    assert genuine.facts["C1"].status == "pass" and genuine.verdict.verdict == "compatible"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["model_state_dict"]["output_norm.scale"] += 1.0
    torch.save(checkpoint, path)                                                      # weights changed, sidecar kept
    tampered = inspect_checkpoint(path, purpose="evaluate")
    assert tampered.facts["C1"].status == "fail" and tampered.intact == "no" and tampered.verdict.failing == ("C1",)
    pathlib.Path(str(path) + ".sha256").unlink()                                      # attacker deletes the sidecar
    assert inspect_checkpoint(path, purpose="evaluate").facts["C1"].status == "unknown"
    supplied = inspect_checkpoint(path, purpose="evaluate", expect_file=genuine.keys["file"])
    assert supplied.facts["C1"].status == "fail" and not supplied.verdict.ok
    pathlib.Path(str(path) + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())   # ... or rewrites it
    rewritten = inspect_checkpoint(path, purpose="evaluate", expect_weights=genuine.keys["weights"])
    assert rewritten.facts["C1"].status == "pass" and rewritten.facts["C3"].status == "fail" and not rewritten.verdict.ok
    assert inspect_checkpoint(files["saver-4.1"], purpose="serve", expect_weights=str(genuine.keys["weights"])).facts["C3"].status == "pass"


def test_cev11_weight_digest_and_tokenizer_file_are_verified(files, tmp_path):
    directory = tmp_path / "cev11"
    shutil.copytree(pathlib.Path(files["cev11-v1"]).parent, directory)
    path = directory / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"]["output_norm.scale"] += 1.0
    torch.save(checkpoint, path)
    report = inspect_checkpoint(path, purpose="evaluate")
    assert report.facts["C3"].status == "fail" and report.intact == "no" and "C3" in report.verdict.failing
    shutil.copyfile(files["cev11-v1"], path)
    other = ByteLevelBPE.train(["bambaşka bir metin"] * 20, vocab_size=290)
    other.save(directory / "model.tokenizer.json")
    assert inspect_checkpoint(path, purpose="evaluate").facts["C7"].status == "fail"
    (directory / "model.tokenizer.json").unlink()
    assert inspect_checkpoint(path, purpose="evaluate").facts["C7"].status == "unknown"
    live = inspect_checkpoint(path, purpose="evaluate", target=Target(tokenizer=files["cev11_tokenizer"]))
    assert live.facts["C7"].status == "pass"
    assert inspect_checkpoint(path, purpose="evaluate", target=Target(tokenizer=other)).facts["C7"].status == "fail"


def test_stripped_tokenizer_identity_warns_for_evaluation_and_denies_serving(files, tmp_path):
    checkpoint = torch.load(files["tm2"], map_location="cpu", weights_only=True)
    checkpoint.pop("tokenizer_identity")
    checkpoint["extra_state"].pop("tokenizer_identity")
    path = tmp_path / "stripped.pth"
    torch.save(checkpoint, path)
    evaluate = inspect_checkpoint(path, purpose="evaluate")
    assert evaluate.facts["C7"].status == "unknown" and evaluate.verdict.verdict == W and "C7" in evaluate.verdict.warnings
    serve = inspect_checkpoint(path, purpose="serve", expect_file=inspect_checkpoint(path, level="file").keys["file"])
    assert "C7" in serve.verdict.failing and serve.verdict.verdict == X
    assert "model" not in evaluate.keys and any("no model key" in note for note in evaluate.notes)


def test_layout_mismatch_and_migration(files, tmp_path):
    checkpoint = torch.load(files["tm2"], map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    wrapped = dict(checkpoint, model_state_dict={"module." + name: tensor for name, tensor in state.items()})
    path = tmp_path / "wrapped.pth"
    torch.save(wrapped, path)
    report = inspect_checkpoint(path, purpose="evaluate")
    assert report.facts["C4"].status == "fail" and "strip-wrapper-prefixes/1" in report.facts["C4"].detail
    allowed = inspect_checkpoint(path, purpose="evaluate", target=Target(allow_migrations=("strip-wrapper-prefixes/1",)))
    assert allowed.facts["C4"].status == "pass" and sorted(allowed.state) == sorted(state)
    opened = open_checkpoint(path, allow_migrations=("strip-wrapper-prefixes/1",))
    with torch.no_grad():
        assert torch.equal(opened.model(TOKENS)[0], files["source"].model(TOKENS)[0])
    missing = dict(checkpoint, model_state_dict={n: t for n, t in state.items() if n != "output_norm.scale"})
    torch.save(missing, tmp_path / "missing.pth")
    partial = inspect_checkpoint(tmp_path / "missing.pth", purpose="evaluate")
    assert partial.facts["C4"].status == "fail" and "missing tensors: output_norm.scale" in partial.facts["C4"].detail


def test_resume_signatures(files):
    source = manager()
    same = inspect_checkpoint(files["tm2"], purpose="resume", target=Target(model=source.model, optimizer=source.optimizer))
    assert same.facts["C10"].status == "pass" and same.verdict.ok
    sgd = torch.optim.SGD(source.model.parameters(), lr=0.1, momentum=0.9)
    other = inspect_checkpoint(files["tm2"], purpose="resume", target=Target(model=source.model, optimizer=sgd))
    assert other.facts["C10"].status == "fail" and "C10" in other.verdict.failing

    class Wrapper:                                                                   # the training scheduler's shape
        def __init__(self, kind, inner):
            self.kind, self.inner = kind, inner

        def state_dict(self):
            return {"scheduler_type": self.kind, "scheduler_state": self.inner.state_dict()}

    step = Wrapper("step", torch.optim.lr_scheduler.StepLR(torch.optim.SGD(source.model.parameters(), lr=0.1), step_size=2))
    wrong = inspect_checkpoint(files["tm2"], purpose="resume", target=Target(model=source.model, scheduler=step))
    assert wrong.facts["C10"].status == "fail" and "scheduler type differs" in wrong.facts["C10"].detail
    assert inspect_checkpoint(files["bare"], purpose="resume").facts["C10"].status == "unknown"
    assert inspect_checkpoint(files["tm2"], purpose="serve", target=Target(max_seq_length=64)).facts["C11"].status == "pass"
    assert inspect_checkpoint(files["tm2"], purpose="serve", target=Target(max_seq_length=128)).facts["C11"].status == "fail"


# ------------------------------------------------------------------------- governed load path
def test_open_checkpoint_loads_through_the_gate_and_attests(files):
    expected = files["source"].model(TOKENS)[0].detach()
    for name in ("tm2", "mm-save", "saver-4.1", "tm3"):
        opened = open_checkpoint(files[name])                                         # model built from the declared arch
        assert opened.verdict.ok and not opened.model.training
        assert torch.equal(opened.model(TOKENS)[0].detach(), expected)
        assert weights.weights_key(opened.model.state_dict())[0] == opened.keys["weights"]
    target = manager(seed=9)
    opened = open_checkpoint(files["tm2"], model=target.model, purpose="finetune")
    assert opened.model is target.model and torch.equal(target.model(TOKENS)[0].detach(), expected)
    resumed = manager(seed=9)
    open_checkpoint(files["tm2"], model=resumed.model, optimizer=resumed.optimizer, purpose="resume")
    assert resumed.optimizer.state_dict()["param_groups"][0]["lr"] == torch.load(
        files["tm2"], weights_only=True)["optimizer_state_dict"]["param_groups"][0]["lr"]
    cev = open_checkpoint(files["cev11-v1"])
    assert cev11().model_sha256(cev.model) == cev11().model_sha256(files["cev11_model"])
    assert str(cev.keys["tok"]) == f"cev:tok:bytebpe-v1:{files['cev11_tokenizer'].identity()}" and "model" in cev.keys
    with pytest.raises(IncompatibleCheckpoint):
        open_checkpoint(files["module"])
    with pytest.raises(IncompatibleCheckpoint):
        open_checkpoint(files["tm2"], purpose="serve")                               # integrity and digest unknown
    with pytest.raises(FileNotFoundError):
        open_checkpoint(str(pathlib.Path(files["tm2"]).with_name("absent.pth")))


def test_open_checkpoint_ignores_runtime_settings_of_the_run_that_wrote_the_file(files, tmp_path, monkeypatch):
    """A stored config may say TensorBoard on, a log directory, a CUDA device: rebuilding has no side effects."""
    checkpoint = torch.load(files["tm2"], map_location="cpu", weights_only=True)
    checkpoint["model_config"].update(use_tensorboard=True, tb_log_dir=str(tmp_path / "tb"), device="cuda", log_level=10)
    path = tmp_path / "runtime.pth"
    torch.save(checkpoint, path)
    monkeypatch.chdir(tmp_path)
    opened = open_checkpoint(path)
    with torch.no_grad():
        assert torch.equal(opened.model(TOKENS)[0], files["source"].model(TOKENS)[0])
    assert next(opened.model.parameters()).device.type == "cpu"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["runtime.pth"]      # no tb/, runs/ or logs/
    assert opened.keys["arch"] == inspect_checkpoint(files["tm2"], purpose="inspect").keys["arch"]


def test_post_load_attestation_detects_a_substitution(files, monkeypatch):
    from cevahir_ai.errors import AttestationError
    target = manager(seed=9)
    original = type(target.model).load_state_dict

    def corrupt(self, state, strict=True):
        result = original(self, state, strict=strict)
        with torch.no_grad():
            self.output_norm.scale.add_(1.0)
        return result

    monkeypatch.setattr(type(target.model), "load_state_dict", corrupt)
    with pytest.raises(AttestationError, match="output_norm.scale"):
        open_checkpoint(files["tm2"], model=target.model)


# -------------------------------------------------------------- legacy tokenizer identities
@pytest.fixture(scope="module")
def tracked_core(tmp_path_factory):
    from tokenizer_management.bpe.bpe_manager import BPEManager
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    directory = tmp_path_factory.mktemp("tokenizer")
    for source, name in (("data/vocab_lib/vocab.json", "vocab.json"), ("data/merges_lib/merges.txt", "merges.txt")):
        shutil.copyfile(ROOT / source, directory / name)
        os.utime(directory / name, ns=(10**18, 10**18))
    core = TokenizerCore({"vocab_path": str(directory / "vocab.json"), "merges_path": str(directory / "merges.txt"),
                          "read_only": True, "use_gpu": False})
    yield core, directory
    BPEManager._instances = {key: value for key, value in BPEManager._instances.items()
                             if not str(value.vocab_file).startswith(str(directory))}


def test_legacy_tokenizer_digest_is_reproduced_and_reconciled(tracked_core):
    from training_system.cache_identity import tokenizer_digest
    core, directory = tracked_core
    live = tokenizer_digest(core)
    assert tokenizers.tok_legacy_v3_key(core) == Key("tok", "legacy-v3", live)
    candidates = tokenizers.legacy_candidates(core)
    assert len(candidates) == 6 and candidates[live] == "read_only=True, merges=LF"
    assert tokenizers.reconcile(live, core)[0] == "exact"
    writable = next(digest for digest, how in candidates.items() if how == "read_only=absent, merges=LF")
    assert tokenizers.reconcile(writable, core)[0] == "drift"                         # the production training digest
    assert tokenizers.reconcile("0" * 64, core)[0] == "mismatch"
    key, explain = tokenizers.canonical_key(core)
    assert key.scheme == "bpe-sem-v1" and explain["vocab_size"] == 60000 and "read_only" not in explain["id_config"]
    # The tracked registry lists this tokenizer: canonical key, file keys and every legacy alias.
    registry = tokenizers.tracked_registry()["tokenizers"][0]
    assert registry["key"] == str(key)
    assert {alias["key"] for alias in registry["aliases"]} == {f"cev:tok:legacy-v3:{digest}" for digest in candidates}
    from cevahir_ai.keys.schemes.file import file_utf8lf_key
    for path, file_key in registry["files"].items():
        assert str(file_utf8lf_key(ROOT / path)) == file_key
    assert tokenizers.lookup_legacy(writable)["key"] == str(key) and tokenizers.lookup_legacy("0" * 64) is None
    assert sorted(os.listdir(directory)) == ["merges.txt", "vocab.json"]              # nothing written next to the files


def test_writable_core_digest_is_a_registered_alias_of_the_same_tokenizer(tmp_path):
    """A checkpoint stamped by a writable core (no read_only key) must not look like another tokenizer."""
    from tokenizer_management.bpe.bpe_manager import BPEManager
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    from training_system.cache_identity import tokenizer_digest
    for source, name in (("data/vocab_lib/vocab.json", "vocab.json"), ("data/merges_lib/merges.txt", "merges.txt")):
        shutil.copyfile(ROOT / source, tmp_path / name)
    try:
        writable = TokenizerCore({"vocab_path": str(tmp_path / "vocab.json"), "merges_path": str(tmp_path / "merges.txt"),
                                  "use_gpu": False})
        digest = tokenizer_digest(writable)
        tracked = tokenizers.lookup_legacy(digest)
        assert tracked is not None and tracked["how"] == "read_only=absent, merges=LF"
        assert tokenizers.canonical_key(writable)[0] == Key.parse(tracked["key"])
    finally:
        BPEManager._instances = {key: value for key, value in BPEManager._instances.items()
                                 if not str(value.vocab_file).startswith(str(tmp_path))}


def test_checkpoint_with_a_legacy_tokenizer_identity_resolves_to_the_canonical_key(files, tracked_core, tmp_path):
    core, _ = tracked_core
    candidates = tokenizers.legacy_candidates(core)
    writable = next(digest for digest, how in candidates.items() if how == "read_only=absent, merges=LF")
    checkpoint = torch.load(files["tm2"], map_location="cpu", weights_only=True)
    checkpoint["tokenizer_identity"] = writable
    checkpoint["extra_state"]["tokenizer_identity"] = writable
    path = tmp_path / "stamped.pth"
    torch.save(checkpoint, path)
    offline = inspect_checkpoint(path, purpose="evaluate")                             # no tokenizer supplied
    assert offline.facts["C7"].status == "pass" and "tracked tokenizer 'default'" in offline.facts["C7"].detail
    assert offline.keys["tok"] == Key("tok", "legacy-v3", writable) and offline.keys["tok_canonical"].scheme == "bpe-sem-v1"
    assert "model" in offline.keys
    live = inspect_checkpoint(path, purpose="evaluate", target=Target(tokenizer=core))
    assert live.facts["C7"].status == "pass" and "drift" in live.facts["C7"].detail
    assert live.keys["model"] == offline.keys["model"]
    unknown = inspect_checkpoint(files["tm2"], purpose="evaluate", target=Target(tokenizer=core))
    assert unknown.facts["C7"].status == "fail"                                        # "abab…" is not this tokenizer
    wrong_family = inspect_checkpoint(path, purpose="evaluate", target=Target(tokenizer=files["cev11_tokenizer"]))
    assert wrong_family.facts["C7"].status == "fail"


# ------------------------------------------------------------------------------------------ CLI
def test_cli_inspect_answers_the_four_questions(files, capsys):
    assert keys_cli(["inspect", files["tm2"]]) == 0
    text = capsys.readouterr().out
    for expected in ("intact      unknown", "envelope    tm2", f"tokenizer   cev:tok:legacy-v3:{LEGACY_TOK}",
                     "arch        cev:arch:cfg-v1:", "heads=4", "verdict     compatible-with-warnings", "uyarılarla uyumlu"):
        assert expected in text
    assert keys_cli(["inspect", files["tm2"], "--target", "cev11-primary"]) == 5
    text = capsys.readouterr().out
    assert "verdict     incompatible" in text and "why not:" in text and "num_layers: checkpoint=2, target=6" in text
    assert keys_cli(["inspect", files["bare"]]) == 5 and "architecture unknown" in capsys.readouterr().out
    assert keys_cli(["inspect", files["module"]]) == 5 and "code execution" in capsys.readouterr().out
    assert keys_cli(["inspect", files["cev11-v1"], "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["intact"] == "yes" and document["envelope"] == "cev11-v1" and document["verdict"]["verdict"] == W
    assert document["keys"]["tok"].startswith("cev:tok:bytebpe-v1:") and document["keys"]["model"].startswith("cev:model:triple-v1:")
    assert "tensors" not in document
    assert keys_cli(["inspect", files["cev11-v1"], "--json", "--tensors", "--byte-bpe",
                     str(pathlib.Path(files["cev11-v1"]).with_name("model.tokenizer.json"))]) == 0
    assert len(json.loads(capsys.readouterr().out)["tensors"]) == 25
    assert keys_cli(["inspect", str(pathlib.Path(files["tm2"]).with_name("absent.pth"))]) == 1


def test_cli_verify_explain_diff_and_schemes(files, tmp_path, capsys):
    report = inspect_checkpoint(files["tm2"], purpose="inspect")
    assert keys_cli(["verify", files["tm2"], str(report.keys["weights"])]) == 0
    assert keys_cli(["verify", files["mm-save"], str(report.keys["weights"])]) == 0     # same weights, other container
    assert keys_cli(["verify", files["mm-save"], str(report.keys["file"])]) == 5
    assert keys_cli(["verify", files["tm2"], "not-a-key"]) == 2
    capsys.readouterr()
    assert keys_cli(["explain", str(report.keys["tok"])]) == 0
    explained = capsys.readouterr().out
    assert "legacy" in explained and "not a tracked tokenizer" in explained
    tracked = tokenizers.tracked_registry()["tokenizers"][0]
    assert keys_cli(["explain", tracked["aliases"][0]["key"]]) == 0
    assert f"alias of {tracked['key']}" in capsys.readouterr().out                    # legacy value -> canonical key
    assert keys_cli(["explain", files["tm2"]]) == 0
    listing = capsys.readouterr().out
    assert "weights:sha256-v1" in listing and "arch:cfg-v1" in listing and "file:bytes-v1" in listing
    assert keys_cli(["explain", str(ROOT / "data/merges_lib/merges.txt")]) == 0
    assert "cev:file:utf8lf-v1:b5b0b9fcc04a73ea79ed1d3cee5f4f96860204cad6b020364ec9f20af4b080b1" in capsys.readouterr().out
    assert keys_cli(["diff", files["tm2"], files["mm-save"]]) == 5                    # weights equal, checkpoints differ
    assert "= weights" in capsys.readouterr().out
    checkpoint = torch.load(files["bare"], map_location="cpu", weights_only=True)
    checkpoint["layers.0.norm1.scale"] = checkpoint["layers.0.norm1.scale"] + 1
    torch.save(checkpoint, tmp_path / "changed.pth")
    assert keys_cli(["diff", files["bare"], str(tmp_path / "changed.pth")]) == 5
    assert "changed: layers.0.norm1.scale (values)" in capsys.readouterr().out
    assert keys_cli(["diff", files["bare"], files["bare"]]) == 0
    assert keys_cli(["schemes"]) == 0 and "tok:legacy-v3" in capsys.readouterr().out


def test_cli_file_level_is_torch_free(files):
    code = ("import json, sys\nfrom cevahir_ai.keys.cli import main\n"
            "code = main(['inspect', sys.argv[1], '--level', 'file', '--purpose', 'inspect', '--json'])\n"
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules, 'file level imported torch'\n"
            "sys.exit(code)\n")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for name, expected in (("saver-4.1", 0), ("module", 5)):
        result = subprocess.run([sys.executable, "-c", code, files[name]], cwd=ROOT, env=env, capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
        document = json.loads(result.stdout)
        assert document["level"] == "file" and document["keys"]["file"].startswith("cev:file:bytes-v1:")
    result = subprocess.run([sys.executable, "-m", "cevahir_ai.keys", "inspect", files["tm2"]], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0 and "verdict     compatible-with-warnings" in result.stdout and result.stderr == ""


# ------------------------------------------------------------------ real CEV-11 base checkpoints
def _real_bases():
    """CEV-11 base checkpoints are local artefacts, not in git: look in this checkout, in the main
    checkout when this is a git worktree, or where CEVAHIR_CEV11_BASES points. Read-only."""
    candidates = [os.environ.get("CEVAHIR_CEV11_BASES"), BASES]
    dot_git = ROOT / ".git"
    if dot_git.is_file():                                          # a worktree: "gitdir: <main>/.git/worktrees/<name>"
        gitdir = dot_git.read_text(encoding="utf-8").partition("gitdir:")[2].strip()
        if "/.git/" in gitdir.replace("\\", "/"):
            main = pathlib.Path(gitdir.replace("\\", "/").split("/.git/")[0])
            candidates.append(main / STUDY.relative_to(ROOT) / "results" / "bases")
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_dir():
            found = sorted(pathlib.Path(candidate).glob("*/model.pt"))
            if found:
                return found
    return []


REAL = _real_bases()


@pytest.mark.skipif(not REAL, reason="CEV-11 base checkpoints are local artefacts (not in git)")
@pytest.mark.parametrize("path", REAL, ids=[path.parent.name for path in REAL])
def test_real_cev11_base_checkpoints_are_recognised_and_still_loadable(path):
    """No existing CEV-11 checkpoint is invalidated: intact, known architecture, known tokenizer, compatible."""
    report = inspect_checkpoint(path, purpose="evaluate")
    assert report.envelope == "cev11-v1" and report.intact == "yes"
    assert report.facts["C3"].status == "pass" and report.facts["C4"].status == "pass"
    assert report.facts["C5"].status == "pass" and report.facts["C7"].status == "pass" and report.facts["C12"].status == "pass"
    assert report.verdict.verdict == W and report.verdict.warnings == ("C1",)          # only: no file digest on record
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert Key("weights", "cev11-fp32-v1", checkpoint["model_sha256"]) in report.aliases
    assert report.keys["tok"] == Key("tok", "bytebpe-v1", checkpoint["tokenizer"]["identity"])
    assert report.explain["expected_layout"]["preset"] == "cev11-primary"
    assert report.explain["arch"]["num_heads"] == 4 and report.explain["arch"]["num_layers"] == 6
    assert set(report.keys) >= {"file", "weights", "wschema", "arch", "tok", "ckpt", "model"}
    wrong = inspect_checkpoint(path, purpose="evaluate", target=Target(config={**checkpoint["cfg"]["config"], "num_heads": 8, "num_kv_heads": 8}))
    assert wrong.verdict.verdict == X and "num_heads: checkpoint=4, target=8" in wrong.facts["C5"].detail


@pytest.mark.skipif(not REAL, reason="CEV-11 base checkpoints are local artefacts (not in git)")
def test_real_cev11_checkpoint_reloads_bit_identically_through_both_paths():
    path = REAL[0]
    study = cev11()
    model, cfg, tokenizer, _ = study.load_checkpoint(path)                             # the study's own loader
    opened = open_checkpoint(path, tokenizer=tokenizer)
    assert study.model_sha256(opened.model) == study.model_sha256(model) == torch.load(path, weights_only=True)["model_sha256"]
    assert model_config_equal(cfg["config"])
    ids = torch.tensor([[2] + tokenizer.encode("Merhaba dünya")])
    with torch.no_grad():
        assert torch.equal(opened.model(ids)[0], model(ids)[0])
    from cevahir_ai.keys import schemas
    assert schemas.preset_config("cev11-primary") == {k: cfg["config"][k] for k in schemas.PRESET_CONFIGS["cev11-primary"]}


def model_config_equal(saved):
    """The exact-equality contract of cev11.load_checkpoint: rebuilding from the saved config gives it back."""
    rebuilt = ModelManager(dict(saved))
    rebuilt.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    return rebuilt.effective_config == saved
