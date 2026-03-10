import pytest
import torch
import logging

from neural_network_module.ortak_katman_module.memory_manager import MemoryManager


# ---------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------
@pytest.fixture
def memory_manager():
    # Not: normalization_type / scaling_type gibi parametreler MemoryManager API'sinde yok.
    return MemoryManager(init_type="xavier", log_level=logging.DEBUG)


# ---------------------------------------------------------------------
# Temel fonksiyonlar
# ---------------------------------------------------------------------
def test_allocate_memory(memory_manager):
    tensor = memory_manager.allocate_memory((10, 20, 64))
    assert tensor.shape == (10, 20, 64)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device("cpu")


def test_initialize_memory_preserves_shape_and_device(memory_manager):
    t0 = memory_manager.allocate_memory((10, 20, 64), dtype=torch.float32)
    t1 = memory_manager.initialize_memory(t0)
    assert t1.shape == t0.shape
    assert str(t1.device) == str(t0.device)


def test_initialize_memory_dtype_conversion_to_fp32(memory_manager):
    # MemoryInitializer mevcut davranışta dtype'ı (low_precision=False iken) float32'ye çevirir
    t0 = memory_manager.allocate_memory((4, 8, 16), dtype=torch.float64)
    t1 = memory_manager.initialize_memory(t0)
    assert t1.dtype == torch.float32


def test_optimize_memory(memory_manager):
    t0 = memory_manager.allocate_memory((10, 20, 64))
    t0 = memory_manager.initialize_memory(t0)
    t1 = memory_manager.optimize_memory(t0)
    assert t1.shape == t0.shape
    assert t1.dtype == t0.dtype
    assert str(t1.device) == str(t0.device)


# ---------------------------------------------------------------------
# Attention bridge
# ---------------------------------------------------------------------
def test_bridge_attention(memory_manager):
    mem = memory_manager.allocate_memory((10, 20, 64))
    att = torch.randn(10, 20, 64)
    out = memory_manager.bridge_attention(mem, att)
    assert out.shape == (10, 20, 128)
    assert out.dtype == mem.dtype
    assert str(out.device) == str(mem.device)


def test_bridge_attention_alignment_different_dims(memory_manager):
    # attention dim < memory dim -> hizalanıp son boyut iki kat MAX dim olur (64*2=128)
    mem = memory_manager.allocate_memory((10, 20, 64))
    att = torch.randn(10, 20, 32)
    out = memory_manager.bridge_attention(mem, att)
    assert out.shape == (10, 20, 128)


def test_bridge_attention_with_masks_via_underlying_bridge(memory_manager):
    # MemoryManager.bridge_attention imzası maskeleri almıyor.
    # Maskeleri alt köprü üzerinden test ediyoruz.
    mem = memory_manager.allocate_memory((10, 20, 64))
    att = torch.randn(10, 20, 64)
    memory_mask = torch.ones(10, 20, dtype=torch.bool)
    attention_mask = torch.zeros(10, 20, dtype=torch.bool)

    out = memory_manager.attention_bridge.bridge_attention(
        mem, att, memory_mask=memory_mask, attention_mask=attention_mask
    )
    assert out.shape == (10, 20, 128)


# ---------------------------------------------------------------------
# Store / Retrieve / Clear
# ---------------------------------------------------------------------
def test_store_and_retrieve(memory_manager):
    x = torch.randn(2, 3, 5)
    memory_manager.store("foo", x)
    y = memory_manager.retrieve("foo")
    assert torch.equal(y, x.detach())


def test_store_redundant(memory_manager, caplog):
    caplog.set_level(logging.WARNING)
    x = torch.randn(2, 3, 5)
    memory_manager.store("dup", x)
    memory_manager.store("dup", x)  # redundant
    assert "Redundant tensor detected" in caplog.text


def test_retrieve_missing_key_raises(memory_manager):
    with pytest.raises(KeyError):
        _ = memory_manager.retrieve("missing_key")


def test_clear_memory(memory_manager):
    memory_manager.store("k1", torch.randn(1, 2, 3))
    memory_manager.store("k2", torch.randn(1, 2, 3))
    memory_manager.clear_memory()
    with pytest.raises(KeyError):
        _ = memory_manager.retrieve("k1")


def test_enforce_strict_gc(memory_manager):
    # Sadece hata atmaması beklenir
    memory_manager.enforce_strict_gc()


# ---------------------------------------------------------------------
# Dtype / boyut / hatalar
# ---------------------------------------------------------------------
def test_memory_allocation_with_different_dtypes():
    mgr = MemoryManager(log_level=logging.DEBUG)
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        t = mgr.allocate_memory((3, 4, 5), dtype=dtype)
        assert t.dtype == dtype


def test_allocate_memory_invalid_size_raises(memory_manager):
    with pytest.raises(ValueError):
        _ = memory_manager.allocate_memory((10, -1, 64))


def test_store_invalid_key_raises(memory_manager):
    with pytest.raises(TypeError):
        memory_manager.store("", torch.randn(2, 2))
    with pytest.raises(TypeError):
        memory_manager.store(None, torch.randn(2, 2))  # type: ignore


def test_validate_tensor_errors(memory_manager):
    with pytest.raises(TypeError):
        memory_manager.validate_tensor("not a tensor")  # type: ignore
    with pytest.raises(ValueError):
        memory_manager.validate_tensor(torch.tensor([1.0]))  # 1D -> < 2 dim


def test_memory_optimizer_does_not_increase_memory(memory_manager):
    t = memory_manager.allocate_memory((10, 20, 64))
    t = memory_manager.initialize_memory(t)
    before = t.element_size() * t.nelement()
    t_opt = memory_manager.optimize_memory(t)
    after = t_opt.element_size() * t_opt.nelement()
    assert after <= before
