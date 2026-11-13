"""CEV-11-facing surfaces of the product code: importable, unchanged, behaving identically.

The frozen study (research/living_learning_paradigm_2026_11_13) builds its models through
``tiny_model_config``, ``ModelManager(cfg).initialize(...)``, ``resolve_ffn_dim`` and
``tokenizer_management.byte_bpe``; its loader requires the rebuilt config to equal the saved one
exactly, and its weight digest hashes state-dict names and shapes. A change to any of these would
stop existing CEV-11 checkpoints from reloading. These tests name CEV-11 when that happens.

The goldens below are literal on purpose: no generator can rewrite them.
"""
import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import warnings

import pytest

torch = pytest.importorskip("torch")

from cevahir_ai.evidence import closure  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
MESSAGE = "CEV-11 dependency surface changed"

TINY_CONFIG = {
    "architecture_version": "cevahir-capabilities-1", "attn_logit_cap": 0.0, "causal_mask": True,
    "checkpointing_strategy": "selective", "config_version": 1, "device": "cpu", "drop_path_rate": 0.0,
    "dropout": 0.0, "embed_dim": 32, "ffn_dim": 64, "kv_eviction_strategy": "sliding_window",
    "kv_num_sink_tokens": 4, "learning_rate": 0.0002, "log_level": 50, "logit_soft_cap": 30.0,
    "max_cache_len": 64, "max_seq_length": 64, "moe_jitter_noise": 0.01, "moe_load_balance_alpha": 0.01,
    "moe_top_k": 2, "num_experts": 8, "num_heads": 4, "num_kv_heads": None, "num_layers": 2,
    "parallel_residual": False, "pe_dropout": 0.0, "pe_max_len": 64, "pe_mode": "rope", "pre_norm": True,
    "precision": "auto", "quantization_type": "none", "rope_original_max_len": 2048,
    "rope_scaling_factor": 1.0, "rope_scaling_type": "none", "seq_proj_dim": 32, "sliding_window": None,
    "tie_weights": True, "use_advanced_checkpointing": False, "use_flash_attention": False,
    "use_gradient_checkpointing": False, "use_kv_cache": True, "use_moe": False, "use_pytorch_sdpa": True,
    "use_qk_norm": False, "use_rmsnorm": True, "use_swiglu": True, "vocab_size": 128,
}

_LAYER = (("attn.key_proj.weight", (32, 32)), ("attn.norm.bias", (32,)), ("attn.norm.weight", (32,)),
          ("attn.out_proj.weight", (32, 32)), ("attn.query_proj.weight", (32, 32)), ("attn.value_proj.weight", (32, 32)),
          ("ffn.fc2.weight", (32, 64)), ("ffn.gate_up_proj.weight", (128, 32)), ("norm1.scale", (32,)), ("norm2.scale", (32,)))
TINY_STATE = {
    "embedding.embedding.weight": (128, 32), "embedding.norm.bias": (32,), "embedding.norm.weight": (32,),
    **{f"layers.{index}.{name}": shape for index in (0, 1) for name, shape in _LAYER},
    "output_layer.weight": (128, 32), "output_norm.scale": (32,),
}


def build(config=None, seed=0):
    from model_management.config_schema import tiny_model_config
    from model_management.model_manager import ModelManager
    torch.manual_seed(seed)
    manager = ModelManager(dict(config or tiny_model_config()))
    manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    return manager


# ------------------------------------------------------------------------------- import surfaces
def test_frozen_import_paths_and_names():
    byte_bpe = importlib.import_module("tokenizer_management.byte_bpe")
    for name in ("ByteLevelBPE", "pretokenize", "PRETOKENIZE_PATTERN", "FORMAT_NAME", "FORMAT_VERSION", "SPECIAL_TOKENS",
                 "PAD_ID", "UNK_ID", "BOS_ID", "EOS_ID", "bytes_to_unicode"):
        assert hasattr(byte_bpe, name), f"{MESSAGE}: tokenizer_management.byte_bpe.{name}"
    assert (byte_bpe.FORMAT_NAME, byte_bpe.FORMAT_VERSION) == ("cevahir-byte-bpe", 1)
    assert tuple(byte_bpe.SPECIAL_TOKENS) == ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
    for module, name in (("model_management.model_manager", "ModelManager"), ("model_management.config_schema", "tiny_model_config"),
                         ("model_management.config_schema", "normalize_model_config"),
                         ("src.neural_network_module.architecture_contracts", "resolve_ffn_dim"),
                         ("src.neural_network", "CevahirNeuralNetwork")):
        assert callable(getattr(importlib.import_module(module), name)), f"{MESSAGE}: {module}.{name}"


def test_tiny_model_config_is_exactly_the_recorded_47_keys():
    from model_management.config_schema import ModelArchConfig, tiny_model_config
    from dataclasses import fields
    assert tiny_model_config() == TINY_CONFIG, f"{MESSAGE}: tiny_model_config() output (cev11.load_checkpoint needs exact equality)"
    assert len(TINY_CONFIG) == 47 and len(fields(ModelArchConfig)) == 40
    assert tiny_model_config(num_heads=8) == {**TINY_CONFIG, "num_heads": 8}


def test_model_manager_surface_and_effective_config():
    manager = build()
    assert manager.model is not None and manager.effective_config == TINY_CONFIG, f"{MESSAGE}: ModelManager.effective_config"
    assert type(manager.model).__name__ == "CevahirNeuralNetwork" and manager.optimizer is None
    # The CEV-11 primary configuration is rebuilt to itself, key for key.
    from cevahir_ai.keys import schemas
    primary = build(schemas.preset_config("cev11-primary"))
    again = build(dict(primary.effective_config))
    assert again.effective_config == primary.effective_config and len(primary.effective_config) == 47
    assert primary.effective_config["num_kv_heads"] == 4 and primary.effective_config["ffn_dim"] is None


def test_tiny_state_dict_names_shapes_dtypes_and_tie():
    model = build().model
    state = model.state_dict()
    assert {name: tuple(tensor.shape) for name, tensor in state.items()} == TINY_STATE, \
        f"{MESSAGE}: state_dict names or shapes (cev11 model_sha256 hashes them)"
    assert len(state) == 25 and {str(tensor.dtype) for tensor in state.values()} == {"torch.float32"}
    assert model.output_layer.weight is model.embedding.embedding.weight, f"{MESSAGE}: tied embeddings"
    assert sorted(name for name, _ in model.named_buffers() if name not in state) == ["pos_encoding.pos_idx", "pos_encoding.rope_freqs"]


def test_attributes_read_by_the_cev11_architecture_readout():
    model = build().model
    layer = model.layers[0]
    for name in ("num_heads", "num_kv_heads", "head_dim", "embed_dim", "use_rope", "use_qk_norm", "sliding_window",
                 "attn_logit_cap", "use_kv_cache", "max_cache_len", "kv_eviction_strategy"):
        assert hasattr(layer.attn, name), f"{MESSAGE}: layers[i].attn.{name}"
    for name in ("ffn_dim", "activation_name", "use_bias", "fc2"):
        assert hasattr(layer.ffn, name), f"{MESSAGE}: layers[i].ffn.{name}"
    for name in ("norm1", "pre_norm", "parallel_residual", "use_moe", "use_gradient_checkpointing", "drop_path_rate"):
        assert hasattr(layer, name), f"{MESSAGE}: layers[i].{name}"
    for name in ("output_norm", "logit_soft_cap", "causal_mask", "pos_encoding", "tie_weights", "_ctor_cfg"):
        assert hasattr(model, name), f"{MESSAGE}: model.{name}"
    assert (layer.attn.num_heads, layer.attn.head_dim, layer.ffn.ffn_dim, model.pos_encoding.mode) == (4, 8, 64, "rope")
    assert type(layer.norm1).__name__ == "RMSNorm" and layer.ffn.activation_name == "swiglu"


def test_resolve_ffn_dim_geometry():
    from src.neural_network_module.architecture_contracts import resolve_ffn_dim
    assert [resolve_ffn_dim(d) for d in (32, 192, 512)] == [256, 512, 1536], f"{MESSAGE}: resolve_ffn_dim"
    assert resolve_ffn_dim(192, gated=False) == 768 and resolve_ffn_dim(192, 64) == 64


# ------------------------------------------------------------------- behaviour: bit-identical
def _fingerprint(config=None, seed=0):
    model = build(config, seed).model.eval()
    tokens = torch.tensor([[1, 5, 9, 2, 7]])
    with torch.no_grad():
        logits = model(tokens)[0]
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}, logits.clone()


def _same(left, right):
    return sorted(left[0]) == sorted(right[0]) and all(torch.equal(left[0][name], right[0][name]) for name in left[0]) \
        and torch.equal(left[1], right[1])


def test_same_seed_gives_identical_weights_and_logits_before_and_after_using_the_roof(tmp_path):
    """Default behaviour is bit-identical: the roof's readouts never perturb a seeded build."""
    from cevahir_ai.adapters import model_build
    from cevahir_ai.adapters.inspection import inspect_checkpoint
    from cevahir_ai.keys import schemas
    from cevahir_ai.keys.torchkeys import weights
    before = _fingerprint()
    assert _same(before, _fingerprint()) and not _same(before, _fingerprint(seed=1))
    # Use every part of the roof that touches models: meta builds, layout checks, keys, inspection.
    for preset in schemas.PRESETS:
        model_build.expected_layout(schemas.preset_config(preset))
    assert schemas.check() == []
    weights.weights_key(before[0])
    torch.save(before[0], tmp_path / "w.pth")
    inspect_checkpoint(tmp_path / "w.pth", purpose="inspect")
    assert _same(before, _fingerprint()), "using the roof changed a seeded model build"
    # A readout in the middle of a seeded sequence does not shift the stream either.
    torch.manual_seed(123)
    first = torch.rand(3)
    torch.manual_seed(123)
    model_build.expected_layout(schemas.preset_config("tiny"))
    assert torch.equal(torch.rand(3), first)


def test_roof_build_path_is_the_legacy_build_path():
    import cevahir_ai.model as roof_model
    import cevahir_ai.model.config as roof_config
    from cevahir_ai.adapters import model_build
    from model_management.model_manager import ModelManager
    assert roof_model.ModelManager is ModelManager
    assert roof_config.tiny_model_config() == TINY_CONFIG
    legacy = _fingerprint()
    torch.manual_seed(0)
    through_roof = model_build.build_model(roof_config.tiny_model_config()).eval()
    with torch.no_grad():
        logits = through_roof(torch.tensor([[1, 5, 9, 2, 7]]))[0]
    assert _same(legacy, ({n: t for n, t in through_roof.state_dict().items()}, logits))
    primary = roof_config.tiny_model_config(vocab_size=4096, embed_dim=192, seq_proj_dim=192, num_heads=4, num_kv_heads=4,
                                            num_layers=6, ffn_dim=None)
    assert _same(_fingerprint(primary, seed=3), _fingerprint(primary, seed=3))


def test_byte_bpe_is_untouched_by_the_roof_facade():
    import cevahir_ai.tokenizer as roof_tokenizer
    from tokenizer_management import byte_bpe
    assert roof_tokenizer.ByteLevelBPE is byte_bpe.ByteLevelBPE
    tokenizer = byte_bpe.ByteLevelBPE.train(["merhaba dünya"] * 10, vocab_size=270)
    assert roof_tokenizer.tokenizer_key(tokenizer).digest == tokenizer.identity()
    assert tokenizer.decode(tokenizer.encode("merhaba dünya")) == "merhaba dünya"


# ---------------------------------------------------------------------------------- closure pin
FROZEN_MODULES = ("model_management.model_manager", "model_management.config_schema", "model_management.checkpoint_contract",
                  "model_management.model_initializer", "tokenizer_management.byte_bpe", "src.neural_network",
                  "src.neural_network_module.architecture_contracts",
                  "src.neural_network_module.ortak_katman_module.attention_manager_module.multi_head_attention")


def test_closure_contains_every_product_module_the_study_reaches():
    modules = dict(closure.compute_closure(ROOT))
    for name in FROZEN_MODULES:
        assert name in modules, f"{name} is missing from the CEV-11 closure"
    assert {name.split(".")[0] for name in modules} == {"src", "model_management", "tokenizer_management", "training_system"}
    assert all((ROOT / path).is_file() for path in modules.values())
    assert not any(name.startswith(("research", "tests", "cevahir_ai")) for name in modules)
    pinned = closure.load_pin(ROOT)
    assert pinned["format"] == "cevahir-closure-pin" and pinned["digest"] == "utf8-lf" and pinned["history"]
    assert pinned["closure_key"].startswith("cev:code:tree-v1:") and len(pinned["modules"]) >= 30


def test_closure_pin_reports_drift_on_the_checkout():
    """Byte drift of a CEV-11 dependency is reported; it fails only under CEVAHIR_CLOSURE_STRICT=1.

    Other work legitimately edits closure modules while keeping default behaviour bit-identical
    (the behaviour tests above are the hard guard). The pin says which files changed, so that the
    study's verification is re-run and the pin re-recorded with a reason.
    """
    findings = closure.check(ROOT)
    if findings and os.environ.get("CEVAHIR_CLOSURE_STRICT") == "1":
        pytest.fail("; ".join(findings) + " -- " + closure.VERIFY_HINT)
    if findings:
        warnings.warn("CEV-11 closure drift: " + "; ".join(findings) + " -- " + closure.VERIFY_HINT)


def _copy_closure(target):
    study = closure.STUDIES["CEV-11"]
    for path in sorted((ROOT / study).rglob("*.py")):
        if "results" in path.relative_to(ROOT / study).parts:
            continue
        destination = target / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for _, relative in closure.compute_closure(ROOT):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return target


def test_closure_pin_names_cev11_and_the_file_when_a_dependency_changes(tmp_path):
    copy = _copy_closure(tmp_path / "repo")
    document = closure.record("test pin", root=copy)
    assert closure.check(copy) == [] and document["history"][-1]["reason"] == "test pin"
    assert dict(closure.compute_closure(copy)) == dict(closure.compute_closure(ROOT))
    # CRLF checkout: nothing changed.
    schema = copy / "model_management/config_schema.py"
    schema.write_bytes(schema.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert closure.check(copy) == []
    # A real edit of a dependency.
    schema.write_bytes(schema.read_bytes() + b"\r\n# edited\r\n")
    assert closure.check(copy) == ["CEV-11 dependency changed: model_management/config_schema.py"]
    # A new import edge.
    (copy / "model_management/extra_dep.py").write_text("X = 1\n", encoding="utf-8")
    exceptions = copy / "model_management/exceptions.py"
    exceptions.write_text(exceptions.read_text(encoding="utf-8") + "\n\ndef _lazy():\n    from model_management import extra_dep\n", encoding="utf-8")
    findings = closure.check(copy)
    assert "CEV-11 dependency added (new import edge): model_management/extra_dep.py" in findings
    with pytest.raises(ValueError, match="reason"):
        closure.record("  ", root=copy)
    repinned = closure.record("reviewed: comment only", root=copy)
    assert closure.check(copy) == [] and [entry["reason"] for entry in repinned["history"]] == ["test pin", "reviewed: comment only"]
    assert repinned["closure_key"] != document["closure_key"]


def test_closure_cli_exit_codes():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-m", "cevahir_ai.evidence.closure", "show"], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)["study"] == "CEV-11"
    check = subprocess.run([sys.executable, "-m", "cevahir_ai.evidence.closure", "check"], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    assert check.returncode in (0, 5)
    assert ("closure unchanged" in check.stdout) if check.returncode == 0 else ("CEV-11 dependency" in check.stderr)
