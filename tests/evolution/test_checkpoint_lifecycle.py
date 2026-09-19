"""Cross-entrypoint restore and non-shape architecture identity regressions."""
import copy
import random
import pytest
import torch
from model_management.config_schema import tiny_model_config
from model_management.model_manager import ModelManager
from model_management.model_loader import ModelLoader
from model_management.checkpoint_contract import unpack_checkpoint
from training_system.data_split import split_training_records


def manager(**overrides):
    obj = ModelManager(tiny_model_config(**overrides))
    obj.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    obj.model.eval()
    return obj


def test_saved_construction_loads_without_external_config_through_both_loaders(tmp_path):
    source = manager(parallel_residual=True)
    path = source.save(str(tmp_path / "model.pth"))
    tokens = torch.tensor([[1, 3, 5]])
    with torch.no_grad():
        expected = source.model(tokens)[0]
        direct = ModelLoader.load_model(type(source.model), path, device="cpu").eval()
        bundled, _, _, metadata = ModelLoader.load_all(type(source.model), path, device="cpu")
        torch.testing.assert_close(direct(tokens)[0], expected)
        torch.testing.assert_close(bundled.eval()(tokens)[0], expected)
    assert metadata["config"]["parallel_residual"] is True


def test_same_shape_different_computation_rejected_before_mutation(tmp_path):
    source, target = manager(parallel_residual=True), manager(parallel_residual=False)
    path = source.save(str(tmp_path / "model.pth"))
    before = copy.deepcopy(target.model.state_dict())
    with pytest.raises(RuntimeError, match="yükleme") as error:
        target.load(path, weights_only=True)
    assert "parallel_residual" in str(error.value.__cause__)
    for name, value in target.model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert target.config["parallel_residual"] is False


def test_loading_clears_attention_state(tmp_path):
    obj = manager()
    path = obj.save(str(tmp_path / "model.pth"))
    with torch.no_grad():
        obj.model(torch.tensor([[1, 2]]), use_cache=True)
    assert obj.model.layers[0].attn.kv_cache.seen_tokens == 2
    obj.load(path, weights_only=True)
    assert obj.model.layers[0].attn.kv_cache.seen_tokens == 0


def test_legacy_envelope_preserves_training_state():
    state, opt, sched, meta = unpack_checkpoint({"state_dict": {"x": torch.ones(1)},
        "optimizer_state": {"a": 1}, "scheduler_state": {"b": 2},
        "additional_info": {"epoch": 7, "config": {"embed_dim": 16}}})
    assert opt == {"a": 1} and sched == {"b": 2} and meta["epoch"] == 7


def test_common_split_keeps_transitive_duplicate_sources_and_rng():
    records = [([1,2],[2,3],"a"), ([4,5],[5,6],"a"),
               ([4,5],[5,6],"b"), ([7,8],[8,9],"b"),
               ([7,8],[8,9],"c"), ([10,11],[11,12],"d")]
    before = random.getstate()
    train, val = split_training_records(records, .5)
    assert random.getstate() == before
    assert sorted([len(train), len(val)]) == [1, 5]
    assert {tuple(x.tolist()) for x,y in train}.isdisjoint({tuple(x.tolist()) for x,y in val})


def test_ordinary_forward_does_not_collect_scalar_diagnostics(monkeypatch):
    obj = manager()
    def forbidden(*args):
        raise AssertionError("hot path scalar statistics")
    monkeypatch.setattr(obj.model, "_tensor_stats", forbidden)
    obj.model(torch.tensor([[1,2]]))
    assert obj.model.get_last_snapshot()["diagnostics_collected"] is False


def test_manager_respects_outer_no_grad_and_forwards_wrapper_kwargs():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, tokens, **kwargs):
            self.received = kwargs
            return self.weight * torch.ones(*tokens.shape, 8), None
    obj = ModelManager({"device": "cpu"}, model_class=Wrapper)
    obj.model = Wrapper()
    with torch.no_grad():
        output = obj.forward(torch.tensor([[1,2]]), use_cache=True, collect_diagnostics=True)[0]
    assert not output.requires_grad
    assert obj.model.received["use_cache"] and obj.model.received["collect_diagnostics"]


def test_cache_identity_bypass_is_rejected(tmp_path):
    from training_system.data_cache import DataCache
    cache = DataCache(str(tmp_path), str(tmp_path / "cache"))
    cache.save_cached_data("old", "old", [([1], [2])])
    with pytest.raises(ValueError, match="identity bypass"):
        cache.get_cached_data("new", "new", allow_cache_key_mismatch=True)


def test_explicit_model_checkpoint_failure_cannot_fall_back(monkeypatch, tmp_path):
    from model.cevahir import Cevahir, CevahirConfig, CevahirInitializationError
    obj = object.__new__(Cevahir)
    obj.config = CevahirConfig(model=tiny_model_config(), device="cpu",
        load_model_path=str(tmp_path / "missing.pth"))
    obj._tokenizer_core = None
    with pytest.raises(CevahirInitializationError, match="checkpoint not found"):
        obj._init_model()


def test_memory_notes_and_summaries_are_scoped():
    from types import SimpleNamespace
    from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
    from cognitive_management.config import CognitiveManagerConfig
    from cognitive_management.v2.utils.request_scope import memory_scope
    service = MemoryServiceV2(CognitiveManagerConfig())
    writes = []
    service._vector_memory_enabled = True
    service._embedding_adapter = SimpleNamespace(encode_single=lambda text: [1., 0.])
    service._vector_store = SimpleNamespace(add=lambda **kwargs: writes.append(kwargs))
    with memory_scope("alice/session-a"):
        service.add_note("private")
        service._persist_summary_to_vector_store("alice summary", 6)
        assert service.notes() == ["private"]
    with memory_scope("bob/session-b"):
        assert service.notes() == []
        service.add_note("bob note")
        service.clear_notes()
        service._persist_summary_to_vector_store("bob summary", 6)
    with memory_scope("alice/session-a"):
        assert service.notes() == ["private"]
    assert writes[0]["ids"] != writes[1]["ids"]
    assert [w["metadata"][0]["scope"] for w in writes] == ["alice/session-a", "bob/session-b"]
    assert service.revision == 5


def test_tokenizer_permutation_rejected_before_weight_load(tmp_path):
    from types import SimpleNamespace
    from model_management.checkpoint_contract import tokenizer_identity
    first = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    permuted = SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 0})
    source = manager()
    source.tokenizer = first
    path = source.save(str(tmp_path / "tokens.pth"))
    target = manager()
    target.tokenizer = permuted
    before = copy.deepcopy(target.model.state_dict())
    with pytest.raises(RuntimeError) as error:
        target.load(path, weights_only=True)
    assert "tokenizer identity mismatch" in str(error.value.__cause__)
    for name, value in target.model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    for load in [ModelLoader.load_model, ModelLoader.load_all]:
        with pytest.raises(ValueError, match="requires a tokenizer"):
            load(type(source.model), path, device="cpu")
        with pytest.raises(ValueError, match="identity mismatch"):
            load(type(source.model), path, device="cpu", tokenizer=permuted)
        loaded = load(type(source.model), path, device="cpu", tokenizer=first)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        assert model.embedding.embedding.weight.shape == source.model.embedding.embedding.weight.shape
    envelope = {"model_state_dict": source.model.state_dict(),
        "extra_state": {"tokenizer_identity": tokenizer_identity(first)}}
    assert unpack_checkpoint(envelope)[3]["tokenizer_identity"] == tokenizer_identity(first)


def test_resume_rejects_tokenizer_change_before_touching_model(tmp_path):
    from types import SimpleNamespace
    from training_management.v2.core.training_manager import TrainingManager
    from model_management.checkpoint_contract import tokenizer_identity
    first = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    other = SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 0})
    obj = object.__new__(TrainingManager)
    obj.device = torch.device("cpu")
    obj.config = {"tokenizer_identity": tokenizer_identity(other)}
    obj.model = torch.nn.Linear(2, 2)
    before = copy.deepcopy(obj.model.state_dict())
    path = tmp_path / "resume.pth"
    torch.save({"model_state_dict": {k: torch.zeros_like(v) for k,v in before.items()},
        "extra_state": {"tokenizer_identity": tokenizer_identity(first)}}, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        obj.resume_from_checkpoint(path)
    for name, value in obj.model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_direct_resave_preserves_token_identity(tmp_path):
    from types import SimpleNamespace
    from model_management.model_saver import ModelSaver
    from model_management.checkpoint_contract import tokenizer_identity
    core = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    source = manager()
    source.tokenizer = core
    first = source.save(str(tmp_path / "first.pth"))
    loaded = ModelLoader.load_model(type(source.model), first, device="cpu", tokenizer=core)
    ModelSaver.save_model(loaded, save_dir=str(tmp_path), model_name="second.pth")
    saved = torch.load(tmp_path / "second.pth", weights_only=True)
    assert unpack_checkpoint(saved)[3]["tokenizer_identity"] == tokenizer_identity(core)
    source.tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 0})
    with pytest.raises(ValueError, match="identity mismatch"):
        source.save(str(tmp_path / "rebranded.pth"))
    assert not (tmp_path / "rebranded.pth").exists()
