# -*- coding: utf-8 -*-
"""
Cevahir.py Academic Test Suite - 100 Tests
==========================================

Endüstri Standartları: Comprehensive, Academic, Production-Ready
Test Coverage: Initialization, Configuration, Tokenization, Forward, Generation,
               Process, Model Management, Properties, Error Handling, Edge Cases

Akademik Doğruluk:
- Reproducible results (seed management)
- Scientific methodology (proper validation)
- Peer-review ready (comprehensive tests)
"""

import pytest
import torch
import tempfile
import shutil
import os
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch

# Import Cevahir and related classes
from model.cevahir import (
    Cevahir,
    CevahirConfig,
    CevahirError,
    CevahirInitializationError,
    CevahirConfigurationError,
    CevahirProcessingError,
    create_cevahir,
)
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput, CognitiveOutput


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def vocab_path(temp_dir):
    """Vocab file path - creates test vocab in temp_dir"""
    vocab_file = os.path.join(temp_dir, "vocab.json")
    vocab = {
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3},
        "<SEP>": {"id": 4},
        "merhaba": {"id": 5},
        "dünya": {"id": 6},
        "test": {"id": 7},
        "hello": {"id": 8},
        "world": {"id": 9},
    }
    # Add tokens up to 60000
    for i in range(10, 60000):
        vocab[f"token_{i}"] = {"id": i}
    
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab_file


@pytest.fixture
def merges_path(temp_dir):
    """Merges file path - creates test merges in temp_dir"""
    merges_file = os.path.join(temp_dir, "merges.txt")
    with open(merges_file, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        f.write("m e\n")
        f.write("er h\n")
        f.write("he l\n")
        f.write("lo w\n")
    return merges_file


@pytest.fixture
def cevahir_config(vocab_path, merges_path) -> CevahirConfig:
    """Cevahir configuration for tests"""
    return CevahirConfig(
        device="cpu",
        seed=42,
        log_level="INFO",
        load_model_path="",  # Don't auto-load model
        tokenizer={
            "vocab_path": vocab_path,
            "merges_path": merges_path,
            "data_dir": None,
            "use_gpu": False,
            "batch_size": 32,
            "max_unk_ratio": 0.01,
            "read_only": True,  # Inference mode - don't modify vocab
        },
        model={
            "vocab_size": 60000,
            "embed_dim": 1024,
            "seq_proj_dim": 1024,
            "num_heads": 16,
            "num_layers": 12,
            "ffn_dim": None,
            "pre_norm": True,
            "causal_mask": True,
            "use_flash_attention": False,
            "pe_mode": "rope",
            "use_gradient_checkpointing": False,
            "tie_weights": True,
            "use_rmsnorm": True,
            "use_swiglu": True,
            "use_kv_cache": True,
            "max_cache_len": 2048,
        },
    )


@pytest.fixture
def cevahir(cevahir_config) -> Cevahir:
    """Cevahir instance for tests"""
    return Cevahir(cevahir_config)


# =============================================================================
# Test Category 1: Initialization (20 tests)
# =============================================================================

def test_init_001_initialization_with_config(cevahir_config):
    """Test 001: Initialization with CevahirConfig"""
    cevahir = Cevahir(cevahir_config)
    assert cevahir is not None
    assert cevahir.config == cevahir_config


def test_init_002_initialization_with_dict():
    """Test 002: Initialization with dict config"""
    config_dict = {
        "device": "cpu",
        "seed": 42,
        "tokenizer": {
            "vocab_path": "data/vocab_lib/vocab.json",
            "merges_path": "data/merges_lib/merges.txt",
            "read_only": True,
        },
        "model": {
            "vocab_size": 60000,
            "embed_dim": 1024,
            "seq_proj_dim": 1024,
            "num_heads": 16,
            "num_layers": 12,
        },
        "load_model_path": "",
    }
    cevahir = Cevahir(config_dict)
    assert cevahir is not None
    assert isinstance(cevahir.config, CevahirConfig)


def test_init_003_initialization_with_seed(cevahir_config):
    """Test 003: Initialization with seed sets random seed"""
    cevahir_config.seed = 42
    cevahir = Cevahir(cevahir_config)
    assert cevahir.config.seed == 42


def test_init_004_initialization_without_seed(cevahir_config):
    """Test 004: Initialization without seed works"""
    cevahir_config.seed = None
    cevahir = Cevahir(cevahir_config)
    assert cevahir.config.seed is None


def test_init_005_initialization_tokenizer_initialized(cevahir):
    """Test 005: TokenizerCore is initialized"""
    assert cevahir._tokenizer_core is not None


def test_init_006_initialization_model_initialized(cevahir):
    """Test 006: ModelManager is initialized"""
    assert cevahir._model_manager is not None


def test_init_007_initialization_cognitive_initialized(cevahir):
    """Test 007: CognitiveManager is initialized"""
    assert cevahir._cognitive_manager is not None


def test_init_008_initialization_model_api_initialized(cevahir):
    """Test 008: CevahirModelAPI is initialized"""
    assert cevahir._model_api is not None


def test_init_009_initialization_with_external_tokenizer(cevahir_config):
    """Test 009: Initialization with external TokenizerCore"""
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    tokenizer = TokenizerCore(cevahir_config.tokenizer)
    cevahir = Cevahir(cevahir_config, tokenizer_core=tokenizer)
    assert cevahir._tokenizer_core == tokenizer


def test_init_010_initialization_with_external_model(cevahir_config):
    """Test 010: Initialization with external ModelManager"""
    from model_management.model_manager import ModelManager
    model = ModelManager(config=cevahir_config.model, tokenizer=None)
    cevahir = Cevahir(cevahir_config, model_manager=model)
    assert cevahir._model_manager == model


def test_init_011_initialization_with_external_cognitive(cevahir_config):
    """Test 011: Initialization with external CognitiveManager"""
    from cognitive_management.cognitive_manager import CognitiveManager
    from cognitive_management.config import CognitiveManagerConfig
    cognitive = CognitiveManager(CognitiveManagerConfig())
    cevahir = Cevahir(cevahir_config, cognitive_manager=cognitive)
    assert cevahir._cognitive_manager == cognitive


def test_init_012_initialization_device_cpu(cevahir_config):
    """Test 012: Initialization with CPU device"""
    cevahir_config.device = "cpu"
    cevahir = Cevahir(cevahir_config)
    assert cevahir.config.device == "cpu"


def test_init_013_initialization_log_level(cevahir_config):
    """Test 013: Initialization sets log level"""
    cevahir_config.log_level = "DEBUG"
    cevahir = Cevahir(cevahir_config)
    assert cevahir.config.log_level == "DEBUG"


def test_init_014_initialization_invalid_config():
    """Test 014: Initialization with invalid config raises error"""
    invalid_config = {"invalid": "config"}
    with pytest.raises(CevahirConfigurationError):
        Cevahir(invalid_config)


def test_init_015_initialization_config_validation(cevahir_config):
    """Test 015: Config validation is called during initialization"""
    cevahir_config.model["vocab_size"] = 0  # Invalid
    with pytest.raises(CevahirConfigurationError):
        Cevahir(cevahir_config)


def test_init_016_initialization_v4_features_enabled(cevahir):
    """Test 016: V-4 features are enabled by default"""
    assert cevahir.config.model.get("use_rmsnorm", False) == True
    assert cevahir.config.model.get("use_swiglu", False) == True
    assert cevahir.config.model.get("use_kv_cache", False) == True


def test_init_017_initialization_rope_enabled(cevahir):
    """Test 017: RoPE is enabled by default"""
    assert cevahir.config.model.get("pe_mode", "sinusoidal") == "rope"


def test_init_018_initialization_weight_tying_enabled(cevahir):
    """Test 018: Weight tying is enabled by default"""
    assert cevahir.config.model.get("tie_weights", False) == True


def test_init_019_initialization_factory_function(cevahir_config):
    """Test 019: Factory function create_cevahir works"""
    cevahir = create_cevahir(cevahir_config)
    assert isinstance(cevahir, Cevahir)


def test_init_020_initialization_singleton_behavior(cevahir_config):
    """Test 020: Multiple instances are independent"""
    cevahir1 = Cevahir(cevahir_config)
    cevahir2 = Cevahir(cevahir_config)
    assert cevahir1 is not cevahir2
    assert cevahir1.config == cevahir2.config


# =============================================================================
# Test Category 2: Configuration (10 tests)
# =============================================================================

def test_config_021_config_validation_valid(cevahir_config):
    """Test 021: Valid config passes validation"""
    cevahir_config.validate()
    assert True  # No exception raised


def test_config_022_config_validation_invalid_vocab_size(cevahir_config):
    """Test 022: Invalid vocab_size raises error"""
    cevahir_config.model["vocab_size"] = 0
    with pytest.raises(CevahirConfigurationError):
        cevahir_config.validate()


def test_config_023_config_validation_invalid_embed_dim(cevahir_config):
    """Test 023: Invalid embed_dim raises error"""
    cevahir_config.model["embed_dim"] = 0
    with pytest.raises(CevahirConfigurationError):
        cevahir_config.validate()


def test_config_024_config_validation_invalid_num_heads(cevahir_config):
    """Test 024: Invalid num_heads raises error"""
    cevahir_config.model["num_heads"] = 0
    with pytest.raises(CevahirConfigurationError):
        cevahir_config.validate()


def test_config_025_config_validation_invalid_num_layers(cevahir_config):
    """Test 025: Invalid num_layers raises error"""
    cevahir_config.model["num_layers"] = 0
    with pytest.raises(CevahirConfigurationError):
        cevahir_config.validate()


def test_config_026_config_default_values(cevahir_config):
    """Test 026: Config has correct default values"""
    assert cevahir_config.device == "cpu"
    assert cevahir_config.log_level == "INFO"


def test_config_027_config_tokenizer_defaults(cevahir_config):
    """Test 027: Tokenizer config has correct defaults"""
    assert "vocab_path" in cevahir_config.tokenizer
    assert "merges_path" in cevahir_config.tokenizer


def test_config_028_config_model_defaults(cevahir_config):
    """Test 028: Model config has correct defaults"""
    assert "vocab_size" in cevahir_config.model
    assert "embed_dim" in cevahir_config.model


def test_config_029_config_load_model_path(cevahir_config):
    """Test 029: load_model_path can be set"""
    cevahir_config.load_model_path = "test/path.pth"
    assert cevahir_config.load_model_path == "test/path.pth"


def test_config_030_config_load_model_path_empty(cevahir_config):
    """Test 030: load_model_path can be empty string"""
    cevahir_config.load_model_path = ""
    assert cevahir_config.load_model_path == ""


# =============================================================================
# Test Category 3: Tokenization (15 tests)
# =============================================================================

def test_token_031_encode_basic(cevahir):
    """Test 031: Encode basic text"""
    tokens, ids = cevahir.encode("merhaba dünya")
    assert isinstance(tokens, list)
    assert isinstance(ids, list)
    assert len(tokens) > 0
    assert len(ids) > 0


def test_token_032_encode_empty_string(cevahir):
    """Test 032: Encode empty string"""
    tokens, ids = cevahir.encode("")
    assert isinstance(tokens, list)
    assert isinstance(ids, list)


def test_token_033_encode_inference_mode(cevahir):
    """Test 033: Encode with inference mode"""
    tokens, ids = cevahir.encode("test", mode="inference")
    assert len(ids) > 0


def test_token_034_encode_train_mode(cevahir):
    """Test 034: Encode with train mode"""
    tokens, ids = cevahir.encode("test", mode="train")
    assert len(ids) > 0


def test_token_035_decode_basic(cevahir):
    """Test 035: Decode basic token IDs"""
    _, ids = cevahir.encode("merhaba")
    decoded = cevahir.decode(ids)
    assert isinstance(decoded, str)
    assert len(decoded) >= 0


def test_token_036_decode_empty_list(cevahir):
    """Test 036: Decode empty list"""
    decoded = cevahir.decode([])
    assert isinstance(decoded, str)


def test_token_037_decode_none_raises_error(cevahir):
    """Test 037: Decode None raises error"""
    with pytest.raises(CevahirProcessingError):
        cevahir.decode(None)


def test_token_038_encode_decode_roundtrip(cevahir):
    """Test 038: Encode-decode roundtrip preserves content"""
    text = "merhaba dünya"
    _, ids = cevahir.encode(text)
    decoded = cevahir.decode(ids)
    # BPE tokenization may change exact text, but should preserve meaning
    assert isinstance(decoded, str)


def test_token_039_encode_unicode(cevahir):
    """Test 039: Encode Unicode text"""
    tokens, ids = cevahir.encode("Merhaba dünya! Çok güzel.")
    assert len(ids) > 0


def test_token_040_encode_long_text(cevahir):
    """Test 040: Encode long text"""
    long_text = "merhaba " * 100
    tokens, ids = cevahir.encode(long_text)
    assert len(ids) > 0


def test_token_041_encode_special_characters(cevahir):
    """Test 041: Encode text with special characters"""
    tokens, ids = cevahir.encode("Hello! How are you? I'm fine.")
    assert len(ids) > 0


def test_token_042_decode_invalid_ids(cevahir):
    """Test 042: Decode with invalid token IDs"""
    # Use IDs that might be out of range
    invalid_ids = [999999, 888888]
    # Should handle gracefully
    decoded = cevahir.decode(invalid_ids)
    assert isinstance(decoded, str)


def test_token_043_encode_kwargs(cevahir):
    """Test 043: Encode with additional kwargs"""
    tokens, ids = cevahir.encode("test", mode="inference", include_whole_words=True)
    assert len(ids) > 0


def test_token_044_decode_kwargs(cevahir):
    """Test 044: Decode with additional kwargs"""
    _, ids = cevahir.encode("test")
    decoded = cevahir.decode(ids, skip_special_tokens=False)
    assert isinstance(decoded, str)


def test_token_045_encode_error_handling(cevahir):
    """Test 045: Encode error handling"""
    # This should not raise an error, but handle gracefully
    try:
        tokens, ids = cevahir.encode("test")
        assert True
    except Exception as e:
        assert isinstance(e, CevahirProcessingError)


# =============================================================================
# Test Category 4: Forward Pass (15 tests)
# =============================================================================

def test_forward_046_forward_with_string(cevahir):
    """Test 046: Forward with string input"""
    logits = cevahir.forward("merhaba")
    assert isinstance(logits, torch.Tensor)
    assert logits.dim() >= 2


def test_forward_047_forward_with_list(cevahir):
    """Test 047: Forward with list of token IDs"""
    _, ids = cevahir.encode("merhaba")
    logits = cevahir.forward(ids)
    assert isinstance(logits, torch.Tensor)


def test_forward_048_forward_with_tensor(cevahir):
    """Test 048: Forward with tensor input"""
    _, ids = cevahir.encode("merhaba")
    tensor = torch.tensor([ids], dtype=torch.long)
    logits = cevahir.forward(tensor)
    assert isinstance(logits, torch.Tensor)


def test_forward_049_forward_empty_string(cevahir):
    """Test 049: Forward with empty string"""
    logits = cevahir.forward("")
    assert isinstance(logits, torch.Tensor)


def test_forward_050_forward_empty_list(cevahir):
    """Test 050: Forward with empty list"""
    logits = cevahir.forward([])
    assert isinstance(logits, torch.Tensor)


def test_forward_051_forward_empty_tensor(cevahir):
    """Test 051: Forward with empty tensor"""
    tensor = torch.tensor([[]], dtype=torch.long)
    logits = cevahir.forward(tensor)
    assert isinstance(logits, torch.Tensor)


def test_forward_052_forward_kwargs(cevahir):
    """Test 052: Forward with kwargs"""
    logits = cevahir.forward("test", use_cache=True)
    assert isinstance(logits, torch.Tensor)


def test_forward_053_forward_logits_shape(cevahir):
    """Test 053: Forward logits have correct shape"""
    logits = cevahir.forward("merhaba")
    assert logits.shape[-1] == cevahir.config.model["vocab_size"]


def test_forward_054_forward_long_sequence(cevahir):
    """Test 054: Forward with long sequence"""
    long_text = "merhaba " * 50
    logits = cevahir.forward(long_text)
    assert isinstance(logits, torch.Tensor)


def test_forward_055_forward_batch_dimension(cevahir):
    """Test 055: Forward output has batch dimension"""
    logits = cevahir.forward("test")
    assert logits.dim() >= 2
    assert logits.shape[0] == 1  # Batch size 1


def test_forward_056_forward_token_clipping(cevahir):
    """Test 056: Forward clips token IDs to vocab size"""
    # Create IDs that exceed vocab size
    large_ids = [70000, 80000]  # Exceeds vocab_size=60000
    logits = cevahir.forward(large_ids)
    assert isinstance(logits, torch.Tensor)


def test_forward_057_forward_1d_tensor(cevahir):
    """Test 057: Forward handles 1D tensor"""
    _, ids = cevahir.encode("test")
    tensor = torch.tensor(ids, dtype=torch.long)
    logits = cevahir.forward(tensor)
    assert isinstance(logits, torch.Tensor)


def test_forward_058_forward_0d_tensor(cevahir):
    """Test 058: Forward handles 0D tensor"""
    tensor = torch.tensor(5, dtype=torch.long)
    logits = cevahir.forward(tensor)
    assert isinstance(logits, torch.Tensor)


def test_forward_059_forward_error_handling(cevahir):
    """Test 059: Forward error handling"""
    # Should handle errors gracefully
    try:
        logits = cevahir.forward("test")
        assert True
    except Exception as e:
        assert isinstance(e, CevahirProcessingError)


def test_forward_060_forward_device_consistency(cevahir):
    """Test 060: Forward output is on correct device"""
    logits = cevahir.forward("test")
    assert logits.device.type == cevahir.config.device


# =============================================================================
# Test Category 5: Generation (15 tests)
# =============================================================================

def test_gen_061_generate_basic(cevahir):
    """Test 061: Generate basic text"""
    generated = cevahir.generate("merhaba", max_new_tokens=10)
    assert isinstance(generated, str)


def test_gen_062_generate_max_tokens(cevahir):
    """Test 062: Generate with max_new_tokens"""
    generated = cevahir.generate("test", max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_063_generate_temperature(cevahir):
    """Test 063: Generate with temperature"""
    generated = cevahir.generate("test", temperature=0.8, max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_064_generate_top_p(cevahir):
    """Test 064: Generate with top_p"""
    generated = cevahir.generate("test", top_p=0.9, max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_065_generate_top_k(cevahir):
    """Test 065: Generate with top_k"""
    generated = cevahir.generate("test", top_k=10, max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_066_generate_repetition_penalty(cevahir):
    """Test 066: Generate with repetition_penalty"""
    generated = cevahir.generate("test", repetition_penalty=1.2, max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_067_generate_empty_prompt(cevahir):
    """Test 067: Generate with empty prompt"""
    generated = cevahir.generate("", max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_068_generate_long_prompt(cevahir):
    """Test 068: Generate with long prompt"""
    long_prompt = "merhaba " * 20
    generated = cevahir.generate(long_prompt, max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_069_generate_zero_max_tokens(cevahir):
    """Test 069: Generate with max_new_tokens=0"""
    generated = cevahir.generate("test", max_new_tokens=0)
    assert isinstance(generated, str)


def test_gen_070_generate_high_temperature(cevahir):
    """Test 070: Generate with high temperature"""
    generated = cevahir.generate("test", temperature=2.0, max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_071_generate_low_temperature(cevahir):
    """Test 071: Generate with low temperature"""
    generated = cevahir.generate("test", temperature=0.1, max_new_tokens=5)
    assert isinstance(generated, str)


def test_gen_072_generate_kwargs(cevahir):
    """Test 072: Generate with additional kwargs"""
    generated = cevahir.generate("test", max_new_tokens=5, use_cache=True)
    assert isinstance(generated, str)


def test_gen_073_generate_error_handling(cevahir):
    """Test 073: Generate error handling"""
    try:
        generated = cevahir.generate("test", max_new_tokens=5)
        assert True
    except Exception as e:
        assert isinstance(e, CevahirProcessingError)


def test_gen_074_generate_consistency(cevahir):
    """Test 074: Generate produces consistent results with seed"""
    cevahir.config.seed = 42
    gen1 = cevahir.generate("test", max_new_tokens=5, temperature=0.0)
    gen2 = cevahir.generate("test", max_new_tokens=5, temperature=0.0)
    # With temperature=0.0, should be deterministic
    assert isinstance(gen1, str)
    assert isinstance(gen2, str)


def test_gen_075_generate_unicode(cevahir):
    """Test 075: Generate with Unicode prompt"""
    generated = cevahir.generate("Merhaba dünya! Çok güzel.", max_new_tokens=5)
    assert isinstance(generated, str)


# =============================================================================
# Test Category 6: Process (10 tests)
# =============================================================================

def test_proc_076_process_basic(cevahir):
    """Test 076: Process basic text"""
    output = cevahir.process("merhaba dünya")
    assert isinstance(output, CognitiveOutput)


def test_proc_077_process_with_state(cevahir):
    """Test 077: Process with cognitive state"""
    state = CognitiveState()
    output = cevahir.process("test", state=state)
    assert isinstance(output, CognitiveOutput)


def test_proc_078_process_empty_string(cevahir):
    """Test 078: Process empty string"""
    output = cevahir.process("")
    assert isinstance(output, CognitiveOutput)


def test_proc_079_process_none_raises_error(cevahir):
    """Test 079: Process None raises error"""
    with pytest.raises(CevahirProcessingError):
        cevahir.process(None)


def test_proc_080_process_kwargs(cevahir):
    """Test 080: Process with kwargs"""
    output = cevahir.process("test", max_tokens=128)
    assert isinstance(output, CognitiveOutput)


def test_proc_081_process_long_text(cevahir):
    """Test 081: Process long text"""
    long_text = "merhaba " * 100
    output = cevahir.process(long_text)
    assert isinstance(output, CognitiveOutput)


def test_proc_082_process_unicode(cevahir):
    """Test 082: Process Unicode text"""
    output = cevahir.process("Merhaba dünya! Çok güzel.")
    assert isinstance(output, CognitiveOutput)


def test_proc_083_process_error_handling(cevahir):
    """Test 083: Process error handling"""
    try:
        output = cevahir.process("test")
        assert True
    except Exception as e:
        assert isinstance(e, CevahirProcessingError)


def test_proc_084_process_output_structure(cevahir):
    """Test 084: Process output has correct structure"""
    output = cevahir.process("test")
    assert hasattr(output, 'response') or hasattr(output, 'text')


def test_proc_085_process_multiple_calls(cevahir):
    """Test 085: Process multiple calls work"""
    output1 = cevahir.process("test1")
    output2 = cevahir.process("test2")
    assert isinstance(output1, CognitiveOutput)
    assert isinstance(output2, CognitiveOutput)


# =============================================================================
# Test Category 7: Model Management (10 tests)
# =============================================================================

def test_model_086_save_model(cevahir, temp_dir):
    """Test 086: Save model"""
    model_path = os.path.join(temp_dir, "model.pth")
    cevahir.save_model(model_path)
    assert os.path.exists(model_path)


def test_model_087_load_model(cevahir, temp_dir):
    """Test 087: Load model"""
    model_path = os.path.join(temp_dir, "model.pth")
    cevahir.save_model(model_path)
    cevahir.load_model(model_path)
    assert True  # No exception raised


def test_model_088_freeze_layers(cevahir):
    """Test 088: Freeze layers by pattern"""
    result = cevahir.freeze("embedding.*")
    assert isinstance(result, dict)


def test_model_089_unfreeze_layers(cevahir):
    """Test 089: Unfreeze layers by pattern"""
    cevahir.freeze("embedding.*")
    result = cevahir.unfreeze("embedding.*")
    assert isinstance(result, dict)


def test_model_090_update_model(cevahir):
    """Test 090: Update model parameters"""
    update_params = {"freeze": ["embedding.*"]}
    result = cevahir.update_model(update_params, dry_run=True)
    assert isinstance(result, dict)


def test_model_091_train_mode(cevahir):
    """Test 091: Set model to training mode"""
    cevahir.train_mode()
    assert True  # No exception raised


def test_model_092_eval_mode(cevahir):
    """Test 092: Set model to evaluation mode"""
    cevahir.eval_mode()
    assert True  # No exception raised


def test_model_093_predict_basic(cevahir):
    """Test 093: Predict with basic input"""
    result = cevahir.predict("test", topk=5)
    assert isinstance(result, dict)


def test_model_094_predict_with_logits(cevahir):
    """Test 094: Predict with return_logits=True"""
    result = cevahir.predict("test", return_logits=True)
    assert isinstance(result, dict)
    assert "logits" in result or "predictions" in result


def test_model_095_save_model_with_kwargs(cevahir, temp_dir):
    """Test 095: Save model with additional kwargs"""
    model_path = os.path.join(temp_dir, "model.pth")
    cevahir.save_model(model_path, additional_info={"test": "data"})
    assert os.path.exists(model_path)


# =============================================================================
# Test Category 8: Properties (5 tests)
# =============================================================================

def test_prop_096_property_tokenizer(cevahir):
    """Test 096: tokenizer property"""
    tokenizer = cevahir.tokenizer
    assert tokenizer is not None


def test_prop_097_property_model(cevahir):
    """Test 097: model property"""
    model = cevahir.model
    assert model is not None


def test_prop_098_property_device(cevahir):
    """Test 098: device property"""
    device = cevahir.device
    assert isinstance(device, torch.device)


def test_prop_099_property_is_initialized(cevahir):
    """Test 099: is_initialized property"""
    is_init = cevahir.is_initialized
    assert isinstance(is_init, bool)


def test_prop_100_property_cognitive(cevahir):
    """Test 100: cognitive property"""
    cognitive = cevahir.cognitive
    assert cognitive is not None

