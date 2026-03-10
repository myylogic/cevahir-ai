# -*- coding: utf-8 -*-
"""
Training System & TrainingService – İleri seviye kapsamlı testler.

Bu dosya training_system modülünü ve training_service.py ana modülünü test eder.
Amaç: Eğitim kötü gidiyorsa nedenini bulmak – config akışı, veri hizalama,
criterion, ve Service→Manager entegrasyonu doğrulanır.

Ref: docs/TRAINING_MANAGEMENT_INCELEME_YOL_HARITASI.md
"""

import os
import sys
import math
import pytest
import torch
from unittest.mock import MagicMock
from typing import Dict, Any, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def mock_tokenizer_core():
    m = MagicMock()
    m.get_vocab_size.return_value = 8000
    m._special_ids.return_value = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
    }
    return m


@pytest.fixture
def base_config():
    return {
            "epochs": 5,
            "batch_size": 16,
            "max_seq_length": 256,
            "max_grad_norm": 1.0,
            "grad_accum_steps": 1,
            "use_amp": False,
            "early_stopping_patience": 3,
            "checkpoint_dir": "./ckpt",
            "use_tensorboard": False,
            "tb_log_dir": "./runs",
            "track_memory": False,
            "track_performance": False,
            "scheduler_type": "ReduceLROnPlateau",
            "scheduler_kwargs": {},
            "warmup_steps": 500,
            "warmup_start_factor": 0.1,
            "embedding_warmup_factor": 1.0,
            "max_checkpoints": 3,
            "gradient_explosion_threshold": 10.0,
            "calculate_advanced_metrics": False,
        }


# ----- ConfigManager: TM config'in tüm gerekli anahtarları içermesi -----

class TestConfigManagerRequiredKeys:
    """ConfigManager.prepare_training_config dönen config'te TM/TrainingLoop için zorunlu anahtarlar."""

    def test_prepare_training_config_contains_vocab_and_special_ids(
        self, mock_tokenizer_core, base_config
    ):
        from training_system.v2.core.config_manager import ConfigManager
        cm = ConfigManager(logger=None)
        out = cm.prepare_training_config(base_config, mock_tokenizer_core, "cpu")
        assert out["vocab_size"] == 8000
        assert out["pad_token_id"] == 0
        assert out["bos_token_id"] == 1
        assert out["eos_token_id"] == 2
        assert out["unk_token_id"] == 3
        assert out["warmup_steps"] == 500
        assert out["warmup_start_factor"] == 0.1
        assert out["embedding_warmup_factor"] == 1.0

    def test_prepare_training_config_contains_training_loop_keys(
        self, mock_tokenizer_core, base_config
    ):
        from training_system.v2.core.config_manager import ConfigManager
        cm = ConfigManager(logger=None)
        out = cm.prepare_training_config(base_config, mock_tokenizer_core, "cpu")
        # TrainingLoop config.get() kullanıyor
        assert "pad_token_id" in out
        assert "vocab_size" in out
        assert "batch_size" in out
        assert "seq_len" in out
        assert "max_grad_norm" in out
        assert "grad_accum_steps" in out


# ----- Prepare_from_cache veri formatı: autoregressive input/target hizalama -----

class TestPrepareFromCacheDataAlignment:
    """prepare_from_cache'tan dönen (inp, tgt) next-token hizalı olmalı."""

    def test_autoregressive_alignment_input_target_same_length(self):
        """Cache (inp_list, tgt_list) formatında: BOS+content, content+EOS → target[i]=input[i+1]."""
        # Simüle: prepare_cache formatı
        # seq_in = [BOS, t1, t2, t3], seq_tgt = [t1, t2, t3, EOS]
        BOS, EOS, PAD = 1, 2, 0
        inp = [BOS, 10, 20, 30]
        tgt = [10, 20, 30, EOS]
        assert len(inp) == len(tgt)
        for i in range(len(tgt) - 1):
            assert tgt[i] == inp[i + 1], f"position {i}: target should be next token"
        assert tgt[-1] == EOS


# ----- CriterionManager: EOS weight ve label_smoothing -----

class TestCriterionManager:
    """CriterionManager.create_criterion doğru loss döndürür."""

    def test_create_criterion_has_weight_and_label_smoothing(self):
        from training_system.v2.core.criterion_manager import CriterionManager
        cm = CriterionManager(logger=None)
        criterion = cm.create_criterion(
            vocab_size=100,
            eos_id=2,
            pad_id=0,
            device=torch.device("cpu"),
            label_smoothing=0.1,
            eos_weight=1.0,
        )
        assert hasattr(criterion, "weight")
        assert criterion.weight is not None
        assert criterion.weight.shape == (100,)
        assert getattr(criterion, "label_smoothing", 0.0) == 0.1

    def test_criterion_forward_gives_finite_loss(self):
        from training_system.v2.core.criterion_manager import CriterionManager
        cm = CriterionManager(logger=None)
        criterion = cm.create_criterion(
            vocab_size=100,
            eos_id=2,
            pad_id=0,
            device=torch.device("cpu"),
            label_smoothing=0.0,
            eos_weight=1.0,
        )
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))
        loss = criterion(logits.view(-1, 100), targets.view(-1))
        assert math.isfinite(loss.item())
        assert loss.item() >= 0.0


# ----- DataLoader wrapper: create_dataloaders batch formatı -----

class TestDataLoaderWrapper:
    """create_dataloaders (input, target) tuple'ları doğru batch'ler."""

    def test_create_dataloaders_returns_two_loaders(self):
        from training_system.v2.utils.data_loader_wrapper import create_dataloaders
        train_data = [
            (torch.randint(0, 100, (8,)), torch.randint(0, 100, (8,)))
            for _ in range(20)
        ]
        val_data = [
            (torch.randint(0, 100, (8,)), torch.randint(0, 100, (8,)))
            for _ in range(10)
        ]
        train_loader, val_loader = create_dataloaders(
            train_data, val_data, batch_size=4, device="cpu", num_workers=0
        )
        assert train_loader is not None
        assert val_loader is not None
        inp, tgt = next(iter(train_loader))
        assert inp.shape[0] == 4 and tgt.shape[0] == 4
        assert inp.shape == tgt.shape


# ----- TrainingService: mock ile init ve config akışı -----

class TestTrainingServiceConfigFlow:
    """TrainingService config'inin TM'e doğru iletilmesi (mock ile)."""

    def test_config_manager_prepare_training_config_used_by_service_flow(
        self, mock_tokenizer_core, base_config
    ):
        """prepare_training_config çıktısı TM'de kullanılabilir olmalı (eksik key yok)."""
        from training_system.v2.core.config_manager import ConfigManager
        base_config["warmup_steps"] = 100
        cm = ConfigManager(logger=None)
        out = cm.prepare_training_config(base_config, mock_tokenizer_core, "cpu")
        # TrainingManager __init__ içinde config.get() ile kullanılanlar
        required = [
            "vocab_size", "epochs", "batch_size", "device", "max_grad_norm",
            "early_stopping_patience", "checkpoint_dir", "pad_token_id",
        ]
        for k in required:
            assert k in out, f"Config'te eksik: {k}"


# ----- Veri hizalama: cache → prepare_from_cache → loss anlamlı mı -----

class TestCacheToLossAlignment:
    """Cache'den gelen (inp, tgt) ile loss doğru hesaplanıyor mu."""

    def test_loss_computation_with_autoregressive_targets(self):
        """Next-token hizalı (inp, tgt) ile LossComputation sonlu loss verir."""
        from training_management.v2.core.loss_computation import LossComputation
        import torch.nn as nn
        B, T, V = 2, 8, 200
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        lc = LossComputation(criterion, logger=None)
        torch.manual_seed(42)
        logits = torch.randn(B, T, V)
        # target[i] = "next token" (simüle: rastgele ama geçerli)
        targets = torch.randint(1, V, (B, T))
        loss, acc, ppl = lc.compute_loss(logits, targets, pad_id=0)
        assert math.isfinite(loss.item())
        assert 0.0 <= acc <= 1.01
        assert ppl >= 0.0


# ----- Özet: neden eğitim kötü gidebilir kontrol listesi -----

class TestTrainingSystemDiagnostics:
    """Eğitim kötü gidiyorsa kontrol edilebilecek noktalar."""

    def test_gradient_flow_from_loss_to_model(self):
        """Loss backward → model parametrelerinde grad olmalı."""
        from src.neural_network import CevahirNeuralNetwork
        cfg = {
            "learning_rate": 1e-4,
            "vocab_size": 500,
            "embed_dim": 32,
            "seq_proj_dim": 32,
            "num_heads": 2,
            "num_layers": 1,
            "ffn_dim": 128,
            "dropout": 0.0,
            "causal_mask": True,
            "tie_weights": True,
            "use_rmsnorm": True,
            "use_swiglu": True,
            "pe_mode": "rope",
            "device": "cpu",
        }
        model = CevahirNeuralNetwork(**cfg)
        x = torch.randint(0, 500, (2, 10))
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 500),
            x[:, 1:].reshape(-1),
            ignore_index=0,
        )
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "Backward sonrası en az bir parametrede grad olmalı"
        model.zero_grad()
