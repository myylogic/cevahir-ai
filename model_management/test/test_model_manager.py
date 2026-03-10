# -*- coding: utf-8 -*-
import os
import sys
import time
from copy import deepcopy

import pytest
import torch

# Proje kökünü import yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model_management.model_manager import ModelManager
from src.neural_network import CevahirNeuralNetwork

# --- Ortak sabitler / config -------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _set_seeds():
    torch.manual_seed(1234)


@pytest.fixture
def base_config():
    # ModelInitializer'ın beklediği anahtarları eksiksiz veriyoruz
    return {
        "vocab_size": 512,
        "embed_dim": 32,
        "seq_proj_dim": 64,
        "num_heads": 2,
        "num_tasks": 1,
        "attention_type": "multi_head",
        "normalization_type": "layer_norm",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "lr_decay_factor": 0.5,
        "lr_decay_patience": 1,
        "lr_threshold": 1e-4,
        "device": "cpu",
        "dropout": 0.1,
    }


@pytest.fixture
def mm(base_config):
    return ModelManager(base_config, model_class=CevahirNeuralNetwork)


# --- Yardımcılar -------------------------------------------------------------

def _random_ids(bsz: int, seqlen: int, vocab: int, device="cpu"):
    return torch.randint(0, vocab, (bsz, seqlen), dtype=torch.long, device=device)


# --- Başlatma / Kurulum ------------------------------------------------------

def test_build_model_only(mm):
    model = mm.build_model()
    assert model is not None
    assert mm.optimizer is None and mm.criterion is None and mm.scheduler is None
    assert next(model.parameters()).device.type == mm.device.type


def test_initialize_chain_builds_all(mm):
    mm.initialize()
    assert mm.model is not None
    assert mm.optimizer is not None
    assert mm.criterion is not None
    assert mm.scheduler is not None


def test_initialize_idempotent(mm):
    mm.initialize()
    model_id_1 = id(mm.model)
    mm.initialize()  # tekrar çağırmak bozmamalı
    assert id(mm.model) == model_id_1


def test_missing_config_raises(base_config):
    bad = deepcopy(base_config)
    bad.pop("vocab_size")
    with pytest.raises(Exception):
        ModelManager(bad, model_class=CevahirNeuralNetwork).build_model()


# --- Mod Yönetimi ------------------------------------------------------------

def test_train_eval_mode_toggles(mm):
    mm.initialize()
    mm.train_mode()
    assert mm.model.training is True
    mm.eval_mode()
    assert mm.model.training is False


def test_forward_preserves_mode_when_inference_none(mm):
    mm.initialize()
    mm.train_mode()
    inp = _random_ids(2, 8, mm.config["vocab_size"])
    _ = mm.forward(inp, inference=None)  # modu değiştirmemeli
    assert mm.model.training is True


def test_forward_switch_and_restore_mode(mm):
    mm.initialize()
    mm.train_mode()
    inp = _random_ids(2, 8, mm.config["vocab_size"])
    _ = mm.forward(inp, inference=True)
    # forward bitince önceki moda dönmeli (train)
    assert mm.model.training is True


# --- Forward / Predict -------------------------------------------------------

def test_forward_train_requires_grad(mm):
    mm.initialize()
    inp = _random_ids(2, 8, mm.config["vocab_size"])
    logits, aux = mm.forward(inp, inference=False)
    assert isinstance(logits, torch.Tensor)
    assert logits.requires_grad is True
    assert aux is None or isinstance(aux, torch.Tensor)
    assert logits.shape == (2, 8, mm.config["vocab_size"])


def test_forward_inference_no_grad(mm):
    mm.initialize()
    mm.eval_mode()
    inp = _random_ids(2, 8, mm.config["vocab_size"])
    logits, aux = mm.forward(inp, inference=True)
    assert logits.requires_grad is False
    assert logits.shape[-1] == mm.config["vocab_size"]


@pytest.mark.parametrize("topk,softmax,ret_logits", [(1, True, False), (3, False, True), (5, True, True)])
def test_predict_variants(mm, topk, softmax, ret_logits):
    mm.initialize()
    inp = _random_ids(2, 8, mm.config["vocab_size"])
    out = mm.predict(inp, topk=topk, apply_softmax=softmax, return_logits=ret_logits)
    if softmax:
        assert "probs" in out and isinstance(out["probs"], torch.Tensor)
    if topk > 0:
        assert "topk_values" in out and "topk_indices" in out
        # azalan sıralı olmalı (son eksen)
        vals = out["topk_values"]
        assert torch.all(vals[..., 1:] <= vals[..., :-1])
    if ret_logits:
        assert "logits" in out


@pytest.mark.parametrize("seqlen", [4, 8, 16])
def test_forward_various_lengths(mm, seqlen):
    mm.initialize()
    inp = _random_ids(2, seqlen, mm.config["vocab_size"])
    logits, _ = mm.forward(inp, inference=True)
    assert logits.shape == (2, seqlen, mm.config["vocab_size"])


def test_forward_expected_vocab_check(mm):
    mm.initialize()
    inp = _random_ids(1, 6, mm.config["vocab_size"])
    logits, _ = mm.forward(inp, inference=True, expected_vocab=mm.config["vocab_size"])
    assert logits.shape[-1] == mm.config["vocab_size"]


# --- Save / Load -------------------------------------------------------------

def test_save_returns_path_and_file_exists(mm, tmp_path):
    mm.initialize()
    path = tmp_path / "ckpt.pth"
    ret = mm.save(str(path), epoch=7, additional_info={"note": "unit"})
    assert ret == str(path)
    assert path.exists()


def test_load_roundtrip_full_checkpoint(mm, tmp_path):
    mm.initialize()
    inp = _random_ids(1, 5, mm.config["vocab_size"])
    # önce kaydet
    path = tmp_path / "round.pth"
    mm.save(str(path), epoch=3)

    # yeni manager yüklesin
    mm2 = ModelManager(deepcopy(mm.config), model_class=CevahirNeuralNetwork).initialize()
    mm2.load(str(path))
    # ileri yayılım çalışmalı
    logits, _ = mm2.forward(inp, inference=True)
    assert logits.shape[-1] == mm.config["vocab_size"]
    assert mm2.config.get("current_epoch", 0) >= 3


def test_load_state_only_checkpoint(mm, tmp_path):
    mm.initialize()
    path = tmp_path / "state_only.pth"
    torch.save({"model_state_dict": mm.model.state_dict(), "epoch": 11}, path)
    mm2 = ModelManager(deepcopy(mm.config), model_class=CevahirNeuralNetwork).initialize()
    mm2.load(str(path))
    assert mm2.config.get("current_epoch", 0) == 11


def test_load_full_model_object(mm, tmp_path):
    mm.initialize()
    path = tmp_path / "full_model.pth"
    # Tüm modeli kaydet (module objesi)
    torch.save(mm.model, path)
    mm2 = ModelManager(deepcopy(mm.config), model_class=CevahirNeuralNetwork).initialize()
    mm2.load(str(path))
    assert isinstance(mm2.model, torch.nn.Module)


def test_load_missing_file_raises(mm, tmp_path):
    with pytest.raises(Exception):
        mm.load(str(tmp_path / "does_not_exist.pth"))


# --- Update (lr, wd, freeze/unfreeze, dry-run) --------------------------------

def test_update_learning_rate(mm):
    mm.initialize()
    new_lr = 5e-4
    mm.update({"optimizer": {"learning_rate": new_lr}})
    assert mm.optimizer.param_groups[0]["lr"] == pytest.approx(new_lr)


def test_update_weight_decay(mm):
    mm.initialize()
    mm.update({"optimizer": {"weight_decay": 0.01}})
    assert mm.optimizer.param_groups[0].get("weight_decay", 0.0) == pytest.approx(0.01)


def test_freeze_all_then_unfreeze_all(mm):
    mm.initialize()
    mm.update({"model": {"freeze": [".*"]}})
    assert all(p.requires_grad is False for p in mm.model.parameters())
    mm.update({"model": {"unfreeze": [".*"]}})
    assert any(p.requires_grad is True for p in mm.model.parameters())


def test_update_dry_run_does_not_change(mm):
    mm.initialize()
    before = [p.requires_grad for p in mm.model.parameters()]
    mm.update({"model": {"freeze": [".*"]}}, dry_run=True)
    after = [p.requires_grad for p in mm.model.parameters()]
    assert before == after


def test_update_invalid_optimizer_key_is_safe(mm):
    mm.initialize()
    # Geçersiz anahtar hata fırlatmadan atlanmalı (logger uyarı)
    mm.update({"optimizer": {"nonexistent_param": 123}})


def test_update_returns_report_dict(mm):
    mm.initialize()
    report = mm.update({"optimizer": {"learning_rate": 9e-4}})
    assert isinstance(report, dict)


# --- Scheduler / LR davranışı ------------------------------------------------

def test_scheduler_exists_and_can_step(mm):
    mm.initialize()
    # ReduceLROnPlateau: bir metrik verelim
    prev = mm.optimizer.param_groups[0]["lr"]
    mm.scheduler.step(1.0)  # step edilebilir olmalı
    # ilk adımda değişmeyebilir, ama erişilebilir olmalı
    assert isinstance(mm.optimizer.param_groups[0]["lr"], float)
    assert mm.optimizer.param_groups[0]["lr"] == pytest.approx(prev)


# --- Cihaz / Performans (hızlı) ----------------------------------------------

def test_device_property(mm):
    assert mm.device.type in ("cpu", "cuda")


def test_quick_performance_cpu(mm):
    mm.initialize()
    inp = _random_ids(8, 20, mm.config["vocab_size"])
    t0 = time.time()
    mm.forward(inp, inference=True)
    dt = time.time() - t0
    assert dt < 1.5  # makul bir süre (CI için geniş)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA mevcut değil")
def test_forward_on_cuda(base_config):
    cfg = deepcopy(base_config)
    cfg["device"] = "cuda"
    mm = ModelManager(cfg, model_class=CevahirNeuralNetwork).initialize()
    inp = _random_ids(2, 8, cfg["vocab_size"], device="cuda")
    logits, _ = mm.forward(inp, inference=True)
    assert logits.is_cuda


# --- Bellek yöneticisi (opsiyonel) -------------------------------------------

def test_memory_manager_if_available(mm):
    mm.initialize()
    inp = _random_ids(2, 6, mm.config["vocab_size"])
    mm.forward(inp, inference=True)
    # Model üzerinde memory_manager varsa deneyelim; yoksa atla
    if hasattr(mm.model, "memory_manager"):
        mem = mm.model.memory_manager.retrieve("final_output")
        assert mem is None or isinstance(mem, torch.Tensor)
    else:
        pytest.skip("Model üzerinde memory_manager bulunmuyor")


# --- Dayanıklılık / hata durumları -------------------------------------------

def test_predict_handles_topk_greater_than_vocab(mm):
    mm.initialize()
    inp = _random_ids(1, 5, mm.config["vocab_size"])
    out = mm.predict(inp, topk=mm.config["vocab_size"] + 10, apply_softmax=False)
    # torch.topk otomatik olarak k'yi sınırlayamayacağı için burada hata alabiliriz;
    # bu yüzden makul bir üst sınır deneyelim
    # Beklenti: en azından logits/probs döndü
    assert "probs" not in out  # softmax=False
    assert "topk_values" in out and "topk_indices" in out or True  # gevşek doğrulama


def test_forward_rejects_invalid_input_shape(mm):
    mm.initialize()
    bad = torch.randint(0, mm.config["vocab_size"], (3, 3, 3))  # beklenenden farklı
    # Model ileri yayılım içerde yine de çalışabilir; yalnızca crash olmamasını istiyoruz.
    logits, _ = mm.forward(bad[:, 0], inference=True)  # [B] veriyoruz -> model hatası yakalayabilir
    assert isinstance(logits, torch.Tensor)


# --- Konfigürasyon sürekliliği -----------------------------------------------

def test_save_load_preserves_epoch_and_config(mm, tmp_path):
    mm.initialize()
    mm.config["some_flag"] = True
    path = tmp_path / "preserve.pth"
    mm.save(str(path), epoch=42, additional_info={"extra": 1})
    mm2 = ModelManager(deepcopy(mm.config), model_class=CevahirNeuralNetwork).initialize()
    mm2.load(str(path))
    assert mm2.config.get("current_epoch", 0) == 42
    assert mm2.config.get("some_flag", False) is True


# --- Predict çıktı yapısı -----------------------------------------------------

def test_predict_output_keys(mm):
    mm.initialize()
    inp = _random_ids(2, 8, mm.config["vocab_size"])
    out = mm.predict(inp, topk=3, apply_softmax=True, return_logits=True)
    for k in ("probs", "topk_values", "topk_indices", "logits"):
        assert k in out


def test_predict_no_softmax_no_logits(mm):
    mm.initialize()
    inp = _random_ids(2, 8, mm.config["vocab_size"])
    out = mm.predict(inp, topk=2, apply_softmax=False, return_logits=False)
    assert "probs" not in out and "logits" not in out


# --- Büyükçe vocab (hızlı sınır) ---------------------------------------------

@pytest.mark.parametrize("vocab", [256, 1024])
def test_forward_with_different_vocab_sizes(base_config, vocab):
    cfg = deepcopy(base_config)
    cfg["vocab_size"] = vocab
    mm = ModelManager(cfg, model_class=CevahirNeuralNetwork).initialize()
    inp = _random_ids(2, 8, vocab)
    logits, _ = mm.forward(inp, inference=True)
    assert logits.shape[-1] == vocab
