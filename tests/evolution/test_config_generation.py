"""Executable contracts for shared configuration and existing generation APIs."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from model_management.config_schema import (
    CevahirConfig as TypedConfig, ModelArchConfig,
    normalize_model_config, tiny_model_config,
)
from model_management.model_manager import ModelManager
from model.cevahir import CevahirConfig, CevahirModelAPI
from cognitive_management.cognitive_types import DecodingConfig


@pytest.mark.parametrize("field,value", [("num_heads", 0), ("num_kv_heads", 0), ("drop_path_rate", 1), ("sliding_window", 0)])
def test_invalid_architecture_raises_validation_error(field, value):
    with pytest.raises(ValueError):
        ModelArchConfig(**{field: value}).validate()


def test_migration_preserves_explicit_settings_and_extras():
    legacy = {"d_model": 64, "n_heads": 4, "num_layers": 3, "dropout": .25, "custom_training_setting": 17}
    original = copy.deepcopy(legacy)
    cfg = normalize_model_config(legacy)
    assert legacy == original
    assert cfg["embed_dim"] == cfg["seq_proj_dim"] == 64
    assert cfg["num_heads"] == 4 and cfg["dropout"] == .25
    assert cfg["custom_training_setting"] == 17
    assert cfg["config_version"] == 1
    assert normalize_model_config(cfg) == cfg


def test_alias_conflict_future_version_and_unknown_strict_rejected():
    for config in ({"d_model": 32, "embed_dim": 64}, {"config_version": 999}, {"num_layres": 3}):
        with pytest.raises(ValueError):
            normalize_model_config(config, strict=True)


def test_nested_typed_roundtrip_keeps_dataclasses_and_capabilities():
    cfg = TypedConfig.from_flat_dict(tiny_model_config(parallel_residual=True, use_qk_norm=True, custom="kept"))
    restored = TypedConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert isinstance(restored.arch, ModelArchConfig)
    restored.validate_all()
    assert restored.arch.parallel_residual and restored.arch.use_qk_norm
    assert restored.extras["custom"] == "kept"


@pytest.mark.parametrize("kind", ["none", "int8_dynamic", "fp16", "bf16", "int8"])
def test_quantization_configuration_has_one_effective_type(kind):
    cfg = TypedConfig.from_flat_dict(tiny_model_config(quantization_type=kind))
    cfg.validate_all()
    assert normalize_model_config(cfg.to_dict())["quantization_type"] == kind
    assert normalize_model_config({"quant_type": kind})["quantization_type"] == kind


def test_unintegrated_quantization_loading_flags_fail_explicitly():
    for flag in ("load_in_8bit", "load_in_4bit"):
        with pytest.raises(ValueError, match="not integrated"):
            normalize_model_config({flag: True})


def test_facade_and_manager_share_model_dimensions_and_state_roundtrip():
    cfg = tiny_model_config()
    facade = CevahirConfig(model=cfg)
    manager = ModelManager(cfg)
    manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    manager.eval_mode()
    assert facade.model["num_layers"] == len(manager.model.layers)
    assert manager.model.embedding.embedding.weight.shape == (128, 32)
    tokens = torch.tensor([[2, 4, 6]])
    before = manager.forward(tokens, inference=True)[0]
    state = copy.deepcopy(manager.model.state_dict())
    manager.model.load_state_dict(state, strict=True)
    torch.testing.assert_close(manager.forward(tokens, inference=True)[0], before)


def test_manager_preserves_left_and_noncontiguous_padding_positions():
    class Capture(torch.nn.Module):
        def forward(self, inputs, mask=None, **kwargs):
            self.mask = mask
            return torch.zeros(*inputs.shape, 8)
    manager = ModelManager({"device": "cpu"}, model_class=Capture)
    manager.model = Capture()
    manager.forward(torch.tensor([[0, 1, 0, 2]]), mask=torch.tensor([[0, 1, 0, 1]]))
    assert manager.model.mask[0, 0].tolist() == [True, False, True, False]


class ToyTokenizer:
    def encode(self, text, **kwargs):
        return ["p"], [1]

    def get_vocab(self):
        return {"<EOS>": 3}

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if i != 3)


class ToyManager:
    device = torch.device("cpu")
    is_initialized = True

    def __init__(self, cache=False, eos=False):
        self.config = {"use_kv_cache": cache}
        self.calls = []
        self.eos = eos

    def eval_mode(self):
        pass

    def clear_kv_cache(self):
        pass

    def forward(self, tokens, **kwargs):
        self.calls.append((tokens.clone(), kwargs))
        logits = torch.zeros(1, tokens.shape[1], 4)
        logits[..., 3 if self.eos else 2] = 5
        return logits, None


def test_zero_token_limit_does_not_run_model():
    manager = ToyManager()
    api = CevahirModelAPI(manager, ToyTokenizer())
    assert api.generate("p", DecodingConfig(max_new_tokens=0)) == ""
    assert not manager.calls


def test_zero_temperature_is_greedy_without_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("greedy must not sample")
    monkeypatch.setattr(torch, "multinomial", forbidden)
    manager = ToyManager()
    api = CevahirModelAPI(manager, ToyTokenizer())
    assert api.generate("p", DecodingConfig(max_new_tokens=3, temperature=0, repetition_penalty=1)) == "2 2 2"
    assert [tokens.shape[1] for tokens, _ in manager.calls] == [1, 2, 3]
    assert all(not kwargs["use_cache"] for _, kwargs in manager.calls)


def test_eos_and_explicit_minimum_are_respected():
    manager = ToyManager(eos=True)
    api = CevahirModelAPI(manager, ToyTokenizer())
    assert api.generate("p", DecodingConfig(max_new_tokens=10, temperature=0)) == ""
    assert len(manager.calls) == 1
    manager.calls.clear()
    api.generate("p", DecodingConfig(max_new_tokens=10, min_new_tokens=2, temperature=0))
    assert len(manager.calls) == 3


def test_beam_search_dispatch_uses_independent_full_prefixes():
    manager = ToyManager(cache=True)
    api = CevahirModelAPI(manager, ToyTokenizer())
    result = api.generate("p", DecodingConfig(max_new_tokens=3, num_beams=2, repetition_penalty=1))
    assert result == "2 2 2"
    assert all(not kwargs["use_cache"] for _, kwargs in manager.calls)
    assert len(manager.calls) > 3
