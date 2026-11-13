"""Regressions for the holes an independent review found in the first version of the compatibility gate.

Each test reproduces the reviewer's evidence on real files and asserts the refusal or the repair:

  1. a gate pass followed by a failing strict load that left the target model half-overwritten;
  2. a head count the stored config does not state, filled with a default and reported as declared;
  3. a target model whose architecture cannot be read, treated as "no target";
  4. fp16 weights cast silently for evaluate, finetune and resume;
  5. a resume that reported C10 pass and did not restore the scheduler;
  6. a checkpoint bound to a tokenizer, loaded without one;
  7. a pickle that hides an import from the torch-free safety sniff;
  8. malformed envelopes, a vacuous context check, a stale KV cache, loading under "inspect".
"""
import copy
import io
import json
import os
import pathlib
import pickle
import warnings
import zipfile

import pytest

torch = pytest.importorskip("torch")

from model_management.config_schema import tiny_model_config  # noqa: E402
from model_management.model_manager import ModelManager  # noqa: E402
from model_management.model_saver import ModelSaver  # noqa: E402

from cevahir_ai.adapters import envelopes, tokenizers  # noqa: E402
from cevahir_ai.adapters.inspection import Target, inspect_checkpoint  # noqa: E402
from cevahir_ai.errors import IncompatibleCheckpoint  # noqa: E402
from cevahir_ai.keys.cli import main as keys_cli  # noqa: E402
from cevahir_ai.model import checkpoint as governed  # noqa: E402
from cevahir_ai.model.checkpoint import open_checkpoint  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOKENS = torch.tensor([[1, 7, 9, 3]])
REGISTRY = json.loads((ROOT / "governance/registry/tokenizers.json").read_text(encoding="utf-8"))["tokenizers"][0]
TRACKED_TOK = next(alias["key"].rsplit(":", 1)[1] for alias in REGISTRY["aliases"] if alias["how"] == "read_only=absent, merges=LF")
FOREIGN_TOK = "ab" * 32
LOADS = ("evaluate", "finetune", "resume", "serve", "publish")
# Loading a file that carries no tokenizer identity warns, as the legacy loader does; the test that
# is about that warning asserts it with pytest.warns. torch warns about pickle protocols above 2.
pytestmark = [pytest.mark.filterwarnings("ignore:checkpoint has no tokenizer identity:RuntimeWarning"),
              pytest.mark.filterwarnings("ignore:Detected pickle protocol:UserWarning")]


def manager(seed=0, optimizer=True, **overrides):
    torch.manual_seed(seed)
    result = ModelManager(tiny_model_config(**overrides))
    result.initialize(build_optimizer=optimizer, build_criterion=False, build_scheduler=optimizer)
    result.model.eval()
    return result


def snapshot(model):
    return copy.deepcopy(model.state_dict())


def assert_untouched(model, before):
    live = model.state_dict()
    assert sorted(live) == sorted(before)
    for name, tensor in live.items():
        assert tensor.dtype == before[name].dtype and torch.equal(tensor, before[name]), name


def logits(model):
    with torch.no_grad():
        return model(TOKENS)[0]


def training_checkpoint(directory, source, identity, steps=7):
    """What production training writes (training_management.v2 CheckpointManager), scheduler wrapped."""
    from training_management.v2.utils.checkpoint_manager import CheckpointManager
    for _ in range(steps):
        source.scheduler.step(1.0)
    source.model._tokenizer_identity = identity
    extra = {"format_version": 1, "tokenizer_identity": identity,
             "scheduler": {"scheduler_type": "ReduceLROnPlateau", "scheduler_state": source.scheduler.state_dict()},
             "loop": {"global_step": 3}, "best_val_loss": 1.0}
    if identity is None:
        extra.pop("tokenizer_identity")
    return CheckpointManager(checkpoint_model_dir=str(directory)).save(source.model, source.optimizer, 3, metric=1.5, extra_state=extra)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """One seeded 4-head tiny model in the shapes the tests start from."""
    directory = tmp_path_factory.mktemp("review")
    source = manager()
    paths = {"source": source, "directory": directory}
    paths["mm"] = source.save(str(directory / "mm.pth"), epoch=2)            # complete model_config, no tokenizer identity
    paths["tm2"] = training_checkpoint(directory / "tm2", manager(), TRACKED_TOK)
    paths["tm2-foreign"] = training_checkpoint(directory / "tm2-foreign", manager(), FOREIGN_TOK)
    paths["saver"] = ModelSaver.save_checkpoint(source.model, optimizer=source.optimizer, scheduler=source.scheduler,
                                                epoch=1, save_dir=str(directory / "saver"))
    paths["bare"] = ModelSaver.save_weights_only(source.model, save_dir=str(directory / "bare"))
    return paths


def refused(path, **options):
    with pytest.raises(IncompatibleCheckpoint) as refusal:
        open_checkpoint(path, **options)
    assert refusal.value.exit_code == 5 and not refusal.value.verdict.ok and refusal.value.report is not None
    return refusal.value


# ============================================================ 1. gate pass, failed load, half a model
def test_learned_positional_table_of_another_length_is_refused_before_the_target_is_touched(tmp_path):
    """Reviewer's evidence: pe_mode="learned", file trained at max_seq_length 32, target built at 64.

    The architecture keys were equal (max_seq_length was a serving field only), the gate passed,
    load_state_dict raised a raw RuntimeError and 14 of 26 target tensors were already overwritten.
    """
    source = manager(pe_mode="learned", max_seq_length=32, max_cache_len=32)
    path = source.save(str(tmp_path / "learned32.pth"), epoch=1)
    target = manager(seed=3, pe_mode="learned", max_seq_length=64)
    assert len(target.model.state_dict()) == 26
    before = snapshot(target.model)
    report = inspect_checkpoint(path, purpose="evaluate", target=Target(model=target.model))
    assert report.facts["C4"].status == "fail" and report.facts["C5"].status == "fail"
    assert "pos_encoding.pe_embed.weight: [32, 32] in the file, [64, 32] expected" in report.facts["C4"].detail
    assert "layout differs from the target model" in report.facts["C4"].detail
    assert "pe_max_len: checkpoint=32, target=64" in report.facts["C5"].detail
    error = refused(path, model=target.model)
    assert {"C4", "C5"} <= set(error.verdict.failing)
    assert_untouched(target.model, before)
    # The same file into a target of its own length loads, and computes what the source computes.
    same = manager(seed=3, pe_mode="learned", max_seq_length=32, max_cache_len=32)
    opened = open_checkpoint(path, model=same.model)
    assert torch.equal(logits(opened.model), logits(source.model))
    assert open_checkpoint(path).report.explain["arch"]["pe_max_len"] == 32             # built from the declaration
    # The legacy loader refuses the 64 target too (validate_state_shapes); the gate is no weaker now.
    with pytest.raises(RuntimeError):
        target.load(path, weights_only=True)
    assert_untouched(target.model, before)


def test_the_live_target_layout_is_checked_even_when_the_architecture_keys_agree(saved):
    """C4 compares the file with the target model itself, not only with what its declaration builds."""
    target = manager(seed=3)
    target.model.output_norm.scale = torch.nn.Parameter(torch.ones(2, 32))            # the model was altered after construction
    before = snapshot(target.model)
    report = inspect_checkpoint(saved["mm"], purpose="evaluate", target=Target(model=target.model))
    assert report.facts["C5"].status == "pass"                                        # same declared architecture
    assert report.facts["C4"].status == "fail"
    assert "layout differs from the target model: shape of output_norm.scale: [32] in the file, [2, 32] expected" \
        in report.facts["C4"].detail
    assert refused(saved["mm"], model=target.model).verdict.failing == ("C4",)
    assert_untouched(target.model, before)
    grown = manager(seed=3)
    grown.model.register_buffer("extra_statistics", torch.zeros(3))
    report = inspect_checkpoint(saved["mm"], purpose="evaluate", target=Target(model=grown.model))
    assert report.facts["C4"].status == "fail" and "missing tensors: extra_statistics" in report.facts["C4"].detail
    passing = inspect_checkpoint(saved["mm"], purpose="evaluate", target=Target(model=manager(seed=3).model))
    assert passing.facts["C4"].detail == "names, shapes and ties match the declared architecture and the target model"


def test_a_load_that_fails_after_the_gate_passed_leaves_the_target_as_it_was(saved, monkeypatch):
    """Whatever makes load_state_dict fail half-way, the caller gets a refusal and an unchanged model."""
    target = manager(seed=9)
    before = snapshot(target.model)
    original = type(target.model).load_state_dict

    def half_way(self, state, strict=True):
        live = self.state_dict()
        with torch.no_grad():
            for name in sorted(state)[:14]:
                live[name].copy_(state[name])
        raise RuntimeError("size mismatch for some.tensor: simulated")

    monkeypatch.setattr(type(target.model), "load_state_dict", half_way)
    error = refused(saved["mm"], model=target.model)
    assert error.verdict.failing == ("C4",) and "the target model refused the weights: RuntimeError: size mismatch" in str(error)
    assert isinstance(error.__cause__, RuntimeError) and error.report.facts["C4"].status == "fail"
    assert_untouched(target.model, before)
    monkeypatch.setattr(type(target.model), "load_state_dict", original)
    assert torch.equal(logits(open_checkpoint(saved["mm"], model=target.model).model), logits(saved["source"].model))


def test_a_resume_that_cannot_restore_its_training_state_changes_nothing(saved, monkeypatch):
    target = manager(seed=9)
    before, optimizer_before = snapshot(target.model), copy.deepcopy(target.optimizer.state_dict())
    scheduler_before = copy.deepcopy(target.scheduler.state_dict())

    load, calls = target.optimizer.load_state_dict, []

    def broken(state):                                         # refuses the checkpoint's state, as torch does on a misfit
        calls.append(state)
        if len(calls) == 1:
            raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")
        return load(state)

    monkeypatch.setattr(target.optimizer, "load_state_dict", broken)
    error = refused(saved["tm2"], model=target.model, optimizer=target.optimizer, scheduler=target.scheduler, purpose="resume")
    assert error.verdict.failing == ("C10",) and "the training state could not be restored: ValueError" in str(error)
    monkeypatch.undo()
    assert_untouched(target.model, before)                                             # the weights went back too
    assert str(target.optimizer.state_dict()) == str(optimizer_before)
    assert target.scheduler.state_dict() == scheduler_before


# ====================================================== 2. a head count nobody stated is not declared
def partial_envelopes(saved, tmp_path, drop=("num_heads", "num_kv_heads")):
    """The 4-head weights with a stored config that leaves ``drop`` to the normaliser's defaults (8 heads)."""
    checkpoint = torch.load(saved["mm"], map_location="cpu", weights_only=True)
    state, complete = checkpoint["state_dict"], checkpoint["model_config"]
    partial = {name: value for name, value in complete.items() if name not in drop}
    files = {}
    for name, document in (
            ("config", {"state_dict": state, "model_state_dict": state, "config": partial}),
            ("additional_info.config", {"model_state_dict": state, "additional_info": {"config": partial, "epoch": 1}}),
            ("train_config", {"epoch": 1, "model_state": state, "optimizer_state": None, "scheduler_state": None,
                              "metrics": {}, "train_config": partial})):
        files[name] = tmp_path / f"{name}.pth"
        torch.save(document, files[name])
    return files, complete


def test_a_config_that_leaves_the_head_count_to_a_default_declares_no_architecture(saved, tmp_path, capsys):
    """Reviewer's evidence: 4-head weights, stored config without num_heads/num_kv_heads.

    inspect printed heads=8, C5 passed against an 8-head model, the weights loaded and the logits
    differed from the source; with no model given an 8-head model was built for 4-head weights.
    """
    files, _ = partial_envelopes(saved, tmp_path)
    eight, four = manager(seed=5, num_heads=8), manager(seed=5)
    before = snapshot(eight.model)
    for source, path in files.items():
        report = inspect_checkpoint(path, purpose="evaluate", target=Target(model=eight.model))
        assert "arch" not in report.keys and "arch" not in report.explain and "model" not in report.keys, source
        assert report.facts["C4"].status == "pass"                                    # the shapes do fit eight heads
        assert report.facts["C5"].status == "unknown" and report.verdict.failing == ("C5",)
        assert f"the stored config ({source}) declares no architecture: it does not state num_heads, num_kv_heads" \
            in report.facts["C5"].detail
        assert any("defaults are not a declaration" in note for note in report.notes)
        for target in (eight, four):                                                  # not even into the right model: no proof
            assert "architecture unknown" in str(refused(path, model=target.model))
        assert set(refused(path).verdict.failing) == {"C4", "C5"}                     # and nothing is built from a guess
        assert keys_cli(["inspect", str(path)]) == 5
        text = capsys.readouterr().out
        assert "heads=" not in text and "arch        unknown (the stored config" in text
    assert_untouched(eight.model, before)
    # What the gate prevents: the same weights under eight heads are another function.
    eight.model.load_state_dict(saved["source"].model.state_dict())
    assert not torch.allclose(logits(eight.model), logits(saved["source"].model))


def test_every_function_defining_field_must_be_stated(saved, tmp_path):
    for field in ("num_layers", "pe_mode", "sliding_window", "tie_weights", "logit_soft_cap", "use_moe"):
        files, _ = partial_envelopes(saved, tmp_path, drop=(field,))
        report = inspect_checkpoint(files["config"], purpose="evaluate")
        assert report.facts["C5"].status == "unknown" and f"does not state {field}" in report.facts["C5"].detail, field
    # Fields of a feature that is switched off say nothing about the function: leaving them out is complete.
    files, _ = partial_envelopes(saved, tmp_path, drop=("num_experts", "moe_top_k", "rope_scaling_factor"))
    report = inspect_checkpoint(files["config"], purpose="evaluate")
    assert report.facts["C5"].status == "pass" and report.keys["arch"] == inspect_checkpoint(saved["mm"], purpose="inspect").keys["arch"]


def test_a_training_config_is_not_an_architecture(saved, tmp_path):
    """tm3 stores ``train_config``. It declares an architecture only when it is a complete model config."""
    state = saved["source"].model.state_dict()
    envelope = {"epoch": 1, "model_state": state, "optimizer_state": None, "scheduler_state": None, "metrics": {}}
    cases = {
        "training settings only": {"batch_size": 8, "epochs": 3},
        "complete, plus a field no model config has": {**tiny_model_config(), "swa_anneal_strategy": "cos"},
        "the constructor record (runtime fields included)": torch.load(saved["mm"], weights_only=True)["model_config"],
    }
    for label, train_config in cases.items():
        path = tmp_path / "tm3.pth"
        torch.save({**envelope, "train_config": train_config}, path)
        report = inspect_checkpoint(path, purpose="evaluate")
        assert report.envelope == "tm3" and "arch" not in report.keys, label
        assert report.facts["C5"].status == "unknown" and not report.verdict.ok, label
    torch.save({**envelope, "train_config": dict(tiny_model_config())}, tmp_path / "tm3.pth")
    complete = inspect_checkpoint(tmp_path / "tm3.pth", purpose="evaluate")
    assert complete.facts["C5"].status == "pass" and complete.explain["arch"]["num_heads"] == 4


# ============================================================= 3. an unreadable target is not no target
class Delegating(torch.nn.Module):
    """A wrapper that forwards the state dict of the model it holds, and has no construction record."""

    def __init__(self, inner):
        super().__init__()
        object.__setattr__(self, "inner", inner)

    def state_dict(self, *args, **kwargs):
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.inner.load_state_dict(*args, **kwargs)

    def parameters(self, recurse=True):
        return self.inner.parameters(recurse)

    def named_parameters(self, *args, **kwargs):
        return self.inner.named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self.inner.named_buffers(*args, **kwargs)


def test_a_target_whose_architecture_cannot_be_read_is_refused(saved):
    """Reviewer's evidence: a declared 4-head checkpoint loaded into an 8-head target with _ctor_cfg
    emptied, and into an 8-head model behind a delegating wrapper; C5 said "no target model given"."""
    emptied = manager(seed=5, num_heads=8)
    emptied.model._ctor_cfg = {}
    wrapped = Delegating(manager(seed=5, num_heads=8).model)
    incomplete = manager(seed=5, num_heads=8)
    del incomplete.model._ctor_cfg["num_heads"]
    for target, reason in ((emptied.model, "carries no construction record"), (wrapped, "carries no construction record"),
                           (incomplete.model, "construction record of CevahirNeuralNetwork is incomplete")):
        before = snapshot(target)
        report = inspect_checkpoint(saved["mm"], purpose="evaluate", target=Target(model=target))
        assert report.facts["C4"].status == "pass"                                    # shapes fit: only C5 can stop it
        assert report.facts["C5"].status == "unknown" and reason in report.facts["C5"].detail
        assert "no target model given" not in report.facts["C5"].detail
        for purpose in LOADS:
            assert "C5" in refused(saved["mm"], model=target, purpose=purpose).verdict.failing
        assert_untouched(target, before)
    readable = manager(seed=5)
    assert torch.equal(logits(open_checkpoint(saved["mm"], model=readable.model).model), logits(saved["source"].model))


# =================================================================== 4. fp16 is never cast silently
def test_fp16_weights_are_refused_for_every_purpose_that_loads(saved, tmp_path):
    checkpoint = torch.load(saved["tm2"], map_location="cpu", weights_only=True)
    checkpoint["model_state_dict"] = {name: tensor.half() for name, tensor in checkpoint["model_state_dict"].items()}
    path = tmp_path / "half.pth"
    torch.save(checkpoint, path)
    target = manager(seed=9)
    before = snapshot(target.model)
    for purpose in LOADS:
        report = inspect_checkpoint(path, purpose=purpose, target=Target(model=target.model, optimizer=target.optimizer))
        assert report.facts["C9"].status == "fail" and "stored float16, target float32" in report.facts["C9"].detail
        assert "C9" in report.verdict.failing and not report.verdict.ok, purpose
        error = refused(path, model=target.model, optimizer=target.optimizer, purpose=purpose)
        assert "C9" in error.verdict.failing and "loading would cast silently" in str(error)
    assert set(refused(path).verdict.failing) == {"C9"}                                # the default purpose, built model
    assert_untouched(target.model, before)
    # The caller can allow the up-cast; it is then stated in the check, and what loads is the cast file.
    opened = open_checkpoint(path, model=target.model, allow_cast=True)
    assert "up-cast to float32 allowed by policy" in opened.report.facts["C9"].detail
    for name, tensor in target.model.state_dict().items():
        assert tensor.dtype == torch.float32 and torch.equal(tensor, checkpoint["model_state_dict"][name].float()), name
    with pytest.raises(IncompatibleCheckpoint):                                        # never down: fp32 file, fp16 target
        open_checkpoint(saved["tm2"], model=manager(seed=9).model.half(), allow_cast=True)


# ===================================================================== 5. resume restores the scheduler
class Log:
    def log_info(self, *args, **kwargs):
        pass

    log_debug = log_error = log_warning = log_info


def test_resume_restores_the_scheduler_from_the_training_envelope(saved):
    """Reviewer's evidence: checkpoint scheduler at last_epoch 7; the resumed ReduceLROnPlateau (what
    ModelManager builds) stayed at 0 and gained the attributes scheduler_type and scheduler_state."""
    stored = torch.load(saved["tm2"], weights_only=True)["extra_state"]["scheduler"]
    inner = stored["scheduler_state"]
    assert set(stored) == {"scheduler_type", "scheduler_state"} and inner["last_epoch"] == 7 and inner["num_bad_epochs"] > 0
    resumed = manager(seed=9)
    assert type(resumed.scheduler).__name__ == "ReduceLROnPlateau" and resumed.scheduler.last_epoch == 0
    opened = open_checkpoint(saved["tm2"], model=resumed.model, optimizer=resumed.optimizer, scheduler=resumed.scheduler,
                             purpose="resume")
    fact = opened.report.facts["C10"]
    assert fact.status == "pass" and "optimizer and scheduler state fits the target" in fact.detail
    assert resumed.scheduler.last_epoch == 7 and resumed.scheduler.num_bad_epochs == inner["num_bad_epochs"]
    assert not hasattr(resumed.scheduler, "scheduler_type") and not hasattr(resumed.scheduler, "scheduler_state")
    restored = resumed.scheduler.state_dict()
    assert {name: restored[name] for name in inner if not name.startswith("_")} == {n: v for n, v in inner.items() if not n.startswith("_")}
    assert resumed.optimizer.state_dict()["param_groups"][0]["lr"] == torch.load(
        saved["tm2"], weights_only=True)["optimizer_state_dict"]["param_groups"][0]["lr"]
    assert torch.equal(logits(resumed.model), logits(saved["source"].model))
    # Without a target scheduler the stored state stays in the file, and the check says so.
    partial = manager(seed=9)
    opened = open_checkpoint(saved["tm2"], model=partial.model, optimizer=partial.optimizer, purpose="resume")
    assert "the stored scheduler state has no target and will not be restored" in opened.report.facts["C10"].detail


def test_resume_restores_the_scheduler_across_wrapper_shapes(saved):
    from training_management.v2.utils.training_scheduler import TrainingScheduler
    plain = torch.load(saved["saver"], weights_only=True)["scheduler_state_dict"]
    wrapped = torch.load(saved["tm2"], weights_only=True)["extra_state"]["scheduler"]
    assert "scheduler_state" not in plain and "scheduler_state" in wrapped
    for path, expected in ((saved["saver"], plain["last_epoch"]), (saved["tm2"], 7)):
        # the training wrapper as the target (production training), whatever shape the file stores
        resumed = manager(seed=9)
        wrapper = TrainingScheduler(resumed.optimizer, "ReduceLROnPlateau", logger=Log())
        open_checkpoint(path, model=resumed.model, optimizer=resumed.optimizer, scheduler=wrapper, purpose="resume")
        assert wrapper.scheduler.last_epoch == expected and wrapper.state_dict()["scheduler_type"] == "ReduceLROnPlateau"
        # a plain torch scheduler as the target
        resumed = manager(seed=9)
        open_checkpoint(path, model=resumed.model, optimizer=resumed.optimizer, scheduler=resumed.scheduler, purpose="resume")
        assert resumed.scheduler.last_epoch == expected and not hasattr(resumed.scheduler, "scheduler_state")


def test_a_scheduler_that_drops_the_state_fails_the_resume_and_everything_is_put_back(saved):
    """The training wrapper swallows its own load errors; torch's is a dict update. Neither reports a
    state that did not arrive, so the restore is read back."""
    resumed = manager(seed=9)
    before, optimizer_before = snapshot(resumed.model), copy.deepcopy(resumed.optimizer.state_dict())

    class Deaf:
        def __init__(self, scheduler):
            self.scheduler = scheduler

        def state_dict(self):
            return {"scheduler_type": "ReduceLROnPlateau", "scheduler_state": self.scheduler.state_dict()}

        def load_state_dict(self, state):
            pass                                                                       # "loaded", says nothing

    deaf = Deaf(resumed.scheduler)
    error = refused(saved["tm2"], model=resumed.model, optimizer=resumed.optimizer, scheduler=deaf, purpose="resume")
    assert error.verdict.failing == ("C10",) and "the scheduler does not hold the stored state" in str(error)
    assert "last_epoch" in str(error)
    assert_untouched(resumed.model, before)
    assert str(resumed.optimizer.state_dict()) == str(optimizer_before) and resumed.scheduler.last_epoch == 0
    with pytest.raises(ValueError, match="does not hold the stored state"):
        governed.restore_scheduler(deaf, {"last_epoch": 3, "best": float("nan")})
    governed.restore_scheduler(resumed.scheduler, {**resumed.scheduler.state_dict(), "last_epoch": 3, "best": float("nan")})
    assert resumed.scheduler.last_epoch == 3                                           # NaN equals NaN when read back


# ============================================================== 6. a bound tokenizer must be verified
def test_a_checkpoint_bound_to_a_tokenizer_is_not_loaded_without_one(saved):
    """Reviewer's evidence: a tm2 checkpoint carrying a fingerprint loaded with no tokenizer; C7 was
    "unknown", downgraded to a warning. ModelManager.load refuses the same file."""
    target = manager(seed=9)
    before = snapshot(target.model)
    report = inspect_checkpoint(saved["tm2-foreign"], purpose="evaluate", target=Target(model=target.model))
    assert report.facts["C7"].status == "fail" and "nothing verified" in report.facts["C7"].detail
    for purpose in LOADS:
        assert "C7" in refused(saved["tm2-foreign"], model=target.model, optimizer=target.optimizer, purpose=purpose).verdict.failing
    assert refused(saved["tm2-foreign"], model=target.model).verdict.failing == ("C7",)
    assert_untouched(target.model, before)
    with pytest.raises(RuntimeError) as legacy:                                        # the legacy contract: same answer
        target.load(saved["tm2-foreign"], weights_only=True)
    assert "requires a tokenizer to verify token identity" in str(legacy.value.__cause__)
    assert_untouched(target.model, before)
    # A file with no identity at all is a legacy file: it loads, and says that nothing could be verified.
    with pytest.warns(RuntimeWarning, match="no tokenizer identity; token semantics cannot be verified"):
        opened = open_checkpoint(saved["mm"], model=target.model)
    assert opened.verdict.warnings == ("C1", "C3", "C7") and opened.report.facts["C7"].status == "unknown"
    # An identity the tracked registry knows needs no warning and no tokenizer argument.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opened = open_checkpoint(saved["tm2"], model=manager(seed=9).model)
    assert opened.report.facts["C7"].status == "pass" and "C7" not in opened.verdict.warnings


def test_a_malformed_or_foreign_tokenizer_identity_is_a_failed_check(saved, tmp_path):
    checkpoint = torch.load(saved["tm2"], map_location="cpu", weights_only=True)
    checkpoint["tokenizer_identity"] = checkpoint["extra_state"]["tokenizer_identity"] = "not-a-digest"
    path = tmp_path / "malformed.pth"
    torch.save(checkpoint, path)
    report = inspect_checkpoint(path, purpose="evaluate")
    assert report.facts["C7"].status == "fail" and "not a 64-hex digest" in report.facts["C7"].detail and not report.verdict.ok
    with pytest.raises(RuntimeError) as legacy:
        manager(seed=9).load(str(path), weights_only=True)
    assert "Invalid checkpoint tokenizer identity" in str(legacy.value.__cause__)
    # A CEV-11 checkpoint written with a tokenizer class of the study: the product has no key for it.
    state = saved["source"].model.state_dict()
    study = {"format": "cev11-checkpoint", "version": 1, "cfg": {"config": dict(tiny_model_config())}, "state_dict": state,
             "tokenizer": {"file": "model.tokenizer.json", "identity": "cd" * 32, "class": "FusedFreqBPE"}}
    torch.save(study, tmp_path / "study.pth")
    unsupplied = inspect_checkpoint(tmp_path / "study.pth", purpose="evaluate")
    assert "tok" not in unsupplied.keys and unsupplied.facts["C7"].status == "fail"
    assert "belongs to the study" in unsupplied.facts["C7"].detail and unsupplied.verdict.failing == ("C7",)

    class FusedFreqBPE:
        def __init__(self, identity):
            self._identity = identity

        def identity(self):
            return self._identity

    matching = inspect_checkpoint(tmp_path / "study.pth", purpose="evaluate", target=Target(tokenizer=FusedFreqBPE("cd" * 32)))
    assert matching.facts["C7"].status == "pass" and matching.verdict.ok and "model" not in matching.keys
    other = inspect_checkpoint(tmp_path / "study.pth", purpose="evaluate", target=Target(tokenizer=FusedFreqBPE("ef" * 32)))
    assert other.facts["C7"].status == "fail" and "does not have the stored identity" in other.facts["C7"].detail


def test_a_tracked_tokenizer_passes_only_while_its_files_are_the_recorded_ones(saved, tmp_path, monkeypatch):
    """"Tracked" is a verification, not a lookup: the files in the checkout must be the registered ones."""
    from cevahir_ai.keys.schemes.file import file_utf8lf_key
    (tmp_path / "governance" / "registry").mkdir(parents=True)
    (tmp_path / "vocab.json").write_text('{"a": {"id": 0}}\n', encoding="utf-8")
    (tmp_path / "merges.txt").write_bytes(b"a b\nc d\n")
    entry = {**REGISTRY, "files": {name: str(file_utf8lf_key(tmp_path / name)) for name in ("vocab.json", "merges.txt")}}
    (tmp_path / "governance" / "registry" / "tokenizers.json").write_text(json.dumps({"tokenizers": [entry]}), encoding="utf-8")
    monkeypatch.setattr(tokenizers, "repo_root", lambda: tmp_path)
    assert tokenizers.tracked_files_changed("default") == []
    assert inspect_checkpoint(saved["tm2"], purpose="evaluate").facts["C7"].status == "pass"
    (tmp_path / "merges.txt").write_bytes(b"a b\r\nc d\r\n")                          # a CRLF checkout: the same tokenizer
    assert tokenizers.tracked_files_changed("default") == []
    (tmp_path / "merges.txt").write_bytes(b"\xef\xbb\xbfa b\nc d\n")                  # a BOM breaks the first merge
    assert tokenizers.tracked_files_changed("default") == ["merges.txt"]
    report = inspect_checkpoint(saved["tm2"], purpose="evaluate")
    assert report.facts["C7"].status == "fail" and "not the recorded ones: merges.txt" in report.facts["C7"].detail
    assert "C7" in refused(saved["tm2"], model=manager(seed=9).model).verdict.failing
    (tmp_path / "merges.txt").write_bytes(b"a b\nc e\n")                              # another merge
    assert tokenizers.tracked_files_changed("default") == ["merges.txt"]
    (tmp_path / "merges.txt").unlink()
    (tmp_path / "vocab.json").write_text('{"a": {"id": 1}}\n', encoding="utf-8")
    assert tokenizers.tracked_files_changed("default") == ["merges.txt", "vocab.json"]
    monkeypatch.undo()
    assert tokenizers.tracked_files_changed("default") == []                           # the repository's own files


def test_legacy_digest_variants_keep_a_byte_order_mark(tmp_path):
    """A BOM in merges.txt is another tokenizer (the first merge never applies), not a newline variant."""
    vocab = {"<PAD>": {"id": 0}, "a": {"id": 1}, "b": {"id": 2}}
    plain, marked = tmp_path / "plain.txt", tmp_path / "marked.txt"
    plain.write_bytes(b"a b\r\nb a\r\n")
    marked.write_bytes(b"\xef\xbb\xbfa b\nb a\n")
    variants = {path: tokenizers.legacy_candidates(tokenizers._Variant(vocab, {"read_only": True}, str(path)))
                for path in (plain, marked)}
    assert all(len(candidates) == 6 for candidates in variants.values())
    assert not set(variants[plain]) & set(variants[marked])
    for path, candidates in variants.items():                                          # the file as it is, is one of its variants
        assert tokenizers._digest(tokenizers._Variant(vocab, {"read_only": True}, str(path))) in candidates


# ======================================================================= 7. the torch-free safety sniff
class Recorder(pickle.Unpickler):
    """The real unpickler, with imports recorded instead of performed."""

    def __init__(self, data):
        super().__init__(io.BytesIO(data))
        self.imports = []

    def find_class(self, module, name):
        self.imports.append(f"{module}.{name}")
        return lambda *args, **kwargs: None                   # harmless whatever the stream calls it with


def text(value):
    data = value.encode("utf-8")
    return b"\x8c" + bytes([len(data)]) + data                # SHORT_BINUNICODE


OS_SYSTEM = text("os") + text("system")
DECOYS = text("collections") + text("OrderedDict")
CALL = b"\x93" + text("echo should-never-run") + b"\x85R."    # STACK_GLOBAL, TUPLE1, REDUCE, STOP
HIDDEN = {
    "decoys pushed and popped (the reviewer's stream)": b"\x80\x04" + OS_SYSTEM + DECOYS + b"00" + CALL,
    "decoys under a mark": b"\x80\x04" + OS_SYSTEM + b"(" + DECOYS + b"1" + CALL,
    "names fetched back from the memo": b"\x80\x04" + text("os") + b"\x94" + text("system") + b"\x94" + b"00" + DECOYS + b"00"
                                        + b"h\x00h\x01" + CALL,
    "explicit memo slots": b"\x80\x04" + text("os") + b"q\x07" + text("system") + b"q\x09" + b"00" + DECOYS + b"00"
                           + b"h\x07h\x09" + CALL,
    "a duplicated name": b"\x80\x04" + text("os") + text("system") + b"2" + b"0" + DECOYS + b"00" + CALL,
    "protocol-2 byte strings": b"\x80\x02U\x02osU\x06systemU\x0bcollectionsU\x0bOrderedDict00" + CALL,
    "the GLOBAL opcode after decoys": b"\x80\x02" + DECOYS + b"00" + b"cos\nsystem\n" + text("echo") + b"\x85R.",
}


@pytest.mark.parametrize("label", sorted(HIDDEN))
def test_safety_sniff_resolves_a_global_the_way_the_unpickler_does(label):
    """Reviewer's evidence: os.system behind popped decoy strings was reported as ``unsafe: []``.

    The sniff must name exactly what the real unpickler imports, whatever the stream does to hide it.
    """
    stream = HIDDEN[label]
    real = Recorder(stream)
    real.load()
    assert real.imports == ["os.system"]
    assert envelopes._pickle_globals(stream) == ["os.system"]
    assert not envelopes.is_safe_global("os.system")


def test_safety_sniff_fails_closed_on_what_it_cannot_resolve():
    computed = b"\x80\x04" + text("builtins") + text("str") + b"\x93" + b")R" + text("system") + b"\x93."     # str() as module
    assert [name for name in envelopes._pickle_globals(computed) if not envelopes.is_safe_global(name)] == \
        ["<unresolved global: its name is computed at run time>"]
    extension = b"\x80\x02\x82\x01."
    assert envelopes._pickle_globals(extension) == ["<unresolved global: extension code 1>"]
    for garbage in (b"not a pickle", b"\x80\x04\x93.", b"\x80\x04" + text("os") + b"\x93.", b"\x80\x04h\x05."[:3], b""):
        found = envelopes._pickle_globals(garbage)
        assert found and not any(envelopes.is_safe_global(name) for name in found), garbage
    # A dotted tail is an attribute walk (protocol 4 resolves it): never one of the safe rebuild functions.
    assert envelopes.is_safe_global("torch._utils._rebuild_tensor_v2")
    for walk in ("torch._utils._rebuild_tensor_v2.__globals__", "torch._utils._rebuild_.x", "torch.FloatStorage.__init__",
                 "builtins.str.join", "collections.OrderedDict.__subclasses__"):
        assert not envelopes.is_safe_global(walk), walk


def test_a_crafted_checkpoint_is_refused_at_file_level(tmp_path, saved):
    """The reviewer's file: ``inspect --level file`` said "C12 pass: weights-only loadable ... all in the safe set"."""
    path = tmp_path / "crafted.pth"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", HIDDEN["decoys pushed and popped (the reviewer's stream)"])
        archive.writestr("archive/version", "3\n")
    sniff = envelopes.sniff_pickle_globals(path)
    assert sniff["container"] == "torch-zip" and sniff["unsafe"] == ["os.system"]
    for level in ("file", "weights", "compat"):
        for purpose in ("inspect", "export", "evaluate"):
            report = inspect_checkpoint(path, level=level, purpose=purpose)
            assert report.facts["C12"].status == "fail" and "os.system" in report.facts["C12"].detail
            assert report.verdict.verdict == "incompatible" and report.state is None
    assert keys_cli(["inspect", str(path), "--level", "file", "--purpose", "inspect"]) == 5
    # No false alarm on what torch.save writes, under every pickle protocol. The scan says what the
    # file imports; whether torch's weights-only reader takes the stream is torch's answer, at
    # weights level (it refuses protocol 4 and 5 streams that its restricted unpickler does not cover).
    state = saved["source"].model.state_dict()
    for protocol in (2, 3, 4, 5):
        torch.save({"model_state_dict": state, "model_config": dict(tiny_model_config()), "epoch": 1}, path, pickle_protocol=protocol)
        sniff = envelopes.sniff_pickle_globals(path)
        assert sniff["unsafe"] == [] and "collections.OrderedDict" in sniff["globals"], protocol
        scanned = inspect_checkpoint(path, level="file", purpose="inspect")
        assert scanned.facts["C12"].status == "pass" and "all in the weights-only safe set" in scanned.facts["C12"].detail
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.load(path, map_location="cpu", weights_only=True)
            readable = True
        except Exception:
            readable = False
        assert (inspect_checkpoint(path, purpose="inspect").facts["C12"].status == "pass") == readable, protocol
    assert protocol == 5


# ============================================================================== 8. smaller findings
def test_malformed_envelopes_get_a_verdict_not_a_traceback(saved, tmp_path, capsys):
    """Reviewer's evidence: a non-list param_groups, a CEV-11 cfg that is a string and a list
    extra_state each ended in an AttributeError traceback and exit 1."""
    good = torch.load(saved["tm2"], map_location="cpu", weights_only=True)
    state = good["model_state_dict"]
    cases = {
        "param_groups": ({**good, "optimizer_state_dict": {"param_groups": 5, "state": {}}}, "optimizer param_groups must be a list"),
        "param_group": ({**good, "optimizer_state_dict": {"param_groups": [1, 2]}}, "optimizer param_groups must be a list"),
        "optimizer": ({**good, "optimizer_state_dict": [1]}, "optimizer state must be a mapping"),
        "extra_state": ({**good, "extra_state": [1, 2]}, "extra_state must be a mapping"),
        "scheduler": ({**good, "extra_state": {"scheduler": {"scheduler_type": "x", "scheduler_state": [1]}}},
                      "the wrapped scheduler state must be a mapping"),
        "model_config": ({**good, "model_config": [1]}, "model_config must be a mapping"),
        "state": ({**good, "model_state_dict": [state]}, "model state must be a mapping"),
        "names": ({**good, "model_state_dict": {1: torch.zeros(1)}}, "model state has names that are not strings"),
        "cev11 cfg": ({"format": "cev11-checkpoint", "cfg": "a string", "state_dict": state}, "cfg must be a mapping"),
        "cev11 tokenizer": ({"format": "cev11-checkpoint", "cfg": {}, "state_dict": state, "tokenizer": "x"},
                            "tokenizer must be a mapping"),
        "tm3 train_config": ({"model_state": state, "train_config": [1]}, "train_config must be a mapping"),
    }
    for label, (document, reason) in cases.items():
        path = tmp_path / "malformed.pth"
        torch.save(document, path)
        report = inspect_checkpoint(path, purpose="evaluate")
        assert report.facts["C12"].status == "fail" and f"malformed envelope: {reason}" in report.facts["C12"].detail, label
        assert report.verdict.verdict == "incompatible" and report.state is None, label
        assert keys_cli(["inspect", str(path)]) == 5 and keys_cli(["inspect", str(path), "--json"]) == 5, label
        assert "Traceback" not in capsys.readouterr().err
        with pytest.raises(IncompatibleCheckpoint, match="malformed envelope"):
            open_checkpoint(path)
    # Descriptive fields of the wrong type are dropped, not fatal: the weights are still what they are.
    torch.save({**good, "epoch": "three", "metric": torch.ones(2)}, tmp_path / "odd.pth")
    odd = inspect_checkpoint(tmp_path / "odd.pth", purpose="evaluate")
    assert odd.verdict.ok and odd.explain["epoch"] is None and odd.explain["metric"] is None
    assert keys_cli(["inspect", str(tmp_path / "odd.pth"), "--json"]) == 0
    json.loads(capsys.readouterr().out)
    # A tokenizer "file name" that is a path is not followed.
    cev = {"format": "cev11-checkpoint", "cfg": {"config": dict(tiny_model_config())}, "state_dict": state,
           "tokenizer": {"file": "../elsewhere.json", "identity": "cd" * 32, "class": "ByteLevelBPE"}}
    torch.save(cev, tmp_path / "cev.pth")
    assert envelopes.read_envelope(tmp_path / "cev.pth").tokenizer_file is None


def test_the_context_check_is_not_vacuous_for_serving(saved):
    """Reviewer: C11 was "skipped" (counted as pass) unless the caller passed max_seq_length."""
    keys = inspect_checkpoint(saved["saver"], purpose="inspect").keys
    expect = {"expect_file": keys["file"], "expect_weights": keys["weights"]}
    trained = inspect_checkpoint(saved["saver"], purpose="serve", **expect)
    assert trained.facts["C11"].status == "pass" and "none requested: the trained context applies" in trained.facts["C11"].detail
    longer = manager(seed=9, max_seq_length=128, max_cache_len=128)                    # RoPE: the same architecture key
    report = inspect_checkpoint(saved["saver"], purpose="serve", target=Target(model=longer.model), **expect)
    assert report.facts["C5"].status == "pass" and report.facts["C11"].status == "fail"
    assert "serving context 128 (of the target) vs trained 64" in report.facts["C11"].detail
    assert "C11" in report.verdict.failing
    assert inspect_checkpoint(saved["saver"], purpose="serve", target=Target(model=longer.model, max_seq_length=64),
                              **expect).facts["C11"].status == "pass"                  # an explicit request wins
    assert inspect_checkpoint(saved["saver"], purpose="serve", target=Target(max_seq_length=65), **expect).facts["C11"].status == "fail"
    assert inspect_checkpoint(saved["bare"], purpose="serve").facts["C11"].status == "unknown"
    assert inspect_checkpoint(saved["saver"], purpose="evaluate", target=Target(model=longer.model)).verdict.ok   # serving only


def test_open_checkpoint_does_what_the_legacy_loaders_do_after_a_load(saved, monkeypatch):
    """Reviewer: open_checkpoint did not clear the KV cache. The identity stamp is the other legacy step."""
    target = manager(seed=9)
    cleared = []
    monkeypatch.setattr(type(target.model), "clear_kv_cache", lambda self: cleared.append(True))
    target.model._tokenizer_identity = "ee" * 32                                       # left over from other weights
    open_checkpoint(saved["tm2"], model=target.model)
    assert cleared == [True] and target.model._tokenizer_identity == TRACKED_TOK       # what the next legacy save stamps
    with pytest.warns(RuntimeWarning):
        open_checkpoint(saved["mm"], model=target.model)
    assert cleared == [True, True] and target.model._tokenizer_identity is None
    with pytest.raises(IncompatibleCheckpoint):
        open_checkpoint(saved["bare"], model=target.model)
    assert cleared == [True, True]                                                     # a refusal touches nothing


def test_open_checkpoint_loads_only_under_a_purpose_that_checks_the_weights(saved):
    """"inspect" and "export" require file-level checks only; loading under them would bypass the gate."""
    eight = manager(seed=5, num_heads=8)
    before = snapshot(eight.model)
    assert inspect_checkpoint(saved["bare"], purpose="inspect", target=Target(model=eight.model)).verdict.ok
    for purpose in ("inspect", "export", "deploy"):
        with pytest.raises(ValueError, match="open_checkpoint loads weights into a model"):
            open_checkpoint(saved["bare"], model=eight.model, purpose=purpose)
    assert_untouched(eight.model, before)
    assert governed.LOAD_PURPOSES == LOADS


def test_no_review_test_wrote_into_the_repository():
    assert not [entry for entry in os.listdir(ROOT) if entry.endswith((".pth", ".pt"))]
