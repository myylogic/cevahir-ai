# -*- coding: utf-8 -*-
"""
Training Management – Endüstri seviyesi kapsamlı testler.

Bu dosya training_management modülünü ve ana akışı (TrainingManager, TrainingLoop,
LossComputation, BatchProcessor) endüstri standartlarına göre test eder.
Veri akışı, loss–hedef hizalama (next-token), tek epoch çalışması ve gradient
yönetimi doğrulanır.

Ref: docs/TRAINING_MANAGEMENT_INCELEME_YOL_HARITASI.md
"""

import os
import sys
import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Proje kökü (conftest ile aynı mantık)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ----- Fixtures: mock logger, config, dataloaders -----

@pytest.fixture
def mock_logger():
    class MockLogger:
        def log_info(self, msg): pass
        def log_error(self, msg, exc_info=False): pass
        def log_warning(self, msg): pass
        def log_debug(self, msg): pass
    return MockLogger()


@pytest.fixture
def tm_config(minimal_model_config):
    """TrainingManager için gerekli config (minimal_model_config ile uyumlu)."""
    c = minimal_model_config.copy()
    c.update({
        "vocab_size": c["vocab_size"],
        "device": torch.device("cpu"),
        "epochs": 2,
        "early_stopping_patience": 3,
        "checkpoint_dir": "./checkpoints_test_tm",
        "enable_file_logging": False,
        "max_grad_norm": 1.0,
        "track_memory": False,
        "track_performance": False,
        "enable_training_analytics": False,
        "pad_token_id": 0,
        "use_amp": False,
        "grad_accum_steps": 1,
        "use_progress_bar": False,
        "batch_size": 2,
        "seq_len": 8,
        "warmup_steps": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "unk_token_id": 3,
        "learning_rate": 1e-4,
        "max_checkpoints": 2,
        "enable_tensorboard": False,
        "tensorboard_log_dir": "./runs_test_tm",
        "calculate_advanced_metrics": False,
    })
    return c


class NextTokenDataset(Dataset):
    """Next-token hedefli küçük dataset: target[t] = input[t+1], son pozisyon EOS."""

    def __init__(self, num_samples, seq_len, vocab_size, eos_id=2, pad_id=0, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.data = []
        for _ in range(num_samples):
            # [1, ..., seq_len-1] token'ları, sonra EOS
            seq = torch.randint(1, vocab_size, (seq_len,), generator=g)
            seq[seq == eos_id] = (eos_id + 1) % max(vocab_size, eos_id + 2)
            inp = seq.clone()
            tgt = torch.roll(seq, -1)
            tgt[-1] = eos_id
            self.data.append((inp, tgt))
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _collate(batch):
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return inputs, targets


@pytest.fixture
def next_token_train_loader(tm_config):
    V = tm_config["vocab_size"]
    T = tm_config["seq_len"]
    B = tm_config["batch_size"]
    ds = NextTokenDataset(num_samples=16, seq_len=T, vocab_size=V, seed=43)
    return DataLoader(ds, batch_size=B, shuffle=False, collate_fn=_collate)


@pytest.fixture
def next_token_val_loader(tm_config):
    V = tm_config["vocab_size"]
    T = tm_config["seq_len"]
    B = tm_config["batch_size"]
    ds = NextTokenDataset(num_samples=8, seq_len=T, vocab_size=V, seed=44)
    return DataLoader(ds, batch_size=B, shuffle=False, collate_fn=_collate)


@pytest.fixture
def training_manager_minimal(minimal_model, tm_config, next_token_train_loader, next_token_val_loader, mock_logger):
    """Minimal TrainingManager: CevahirNeuralNetwork + next-token dataloaders."""
    from training_management.v2.core.training_manager import TrainingManager
    model = minimal_model
    optimizer = torch.optim.Adam(model.parameters(), lr=tm_config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tm_config["pad_token_id"], reduction="mean")
    config = {**tm_config}
    config["device"] = torch.device("cpu")
    tm = TrainingManager(
        model=model,
        train_loader=next_token_train_loader,
        val_loader=next_token_val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        start_epoch=1,
        logger=mock_logger,
    )
    return tm


# ----- S1: Initialization -----

class TestTrainingManagerInitialization:
    """S1: TrainingManager ve bileşenlerin doğru oluşturulması."""

    def test_tm_initializes_with_required_components(self, training_manager_minimal):
        tm = training_manager_minimal
        assert tm.model is not None
        assert tm.train_loader is not None
        assert tm.val_loader is not None
        assert tm.optimizer is not None
        assert tm.criterion is not None
        assert tm.batch_processor is not None
        assert tm.loss_computation is not None
        assert tm.gradient_manager is not None
        assert tm.training_loop is not None
        assert tm.epochs == 2
        assert tm.vocab_size == training_manager_minimal.config["vocab_size"]

    def test_tm_config_has_vocab_size_and_device(self, tm_config):
        assert "vocab_size" in tm_config
        assert tm_config["vocab_size"] > 0
        assert tm_config["device"] is not None


# ----- S2: Contract (train_epoch / validate_epoch return types) -----

class TestTrainingLoopContract:
    """S2: train_epoch ve validate_epoch dönüş sözleşmesi."""

    def test_train_epoch_returns_three_floats(self, training_manager_minimal):
        tm = training_manager_minimal
        train_result = tm.training_loop.train_epoch(tm.train_loader, epoch=1)
        assert isinstance(train_result, (tuple, list))
        assert len(train_result) >= 3
        train_loss, train_acc, avg_grad_norm = train_result[0], train_result[1], train_result[2]
        assert isinstance(train_loss, (float, torch.Tensor))
        assert isinstance(train_acc, (float, torch.Tensor))
        assert isinstance(avg_grad_norm, (float, torch.Tensor))
        if isinstance(train_loss, torch.Tensor):
            assert train_loss.dim() == 0
        assert float(train_loss) >= 0.0
        assert 0.0 <= float(train_acc) <= 1.01

    def test_validate_epoch_returns_at_least_two_floats(self, training_manager_minimal):
        tm = training_manager_minimal
        val_result = tm.training_loop.validate_epoch(tm.val_loader, epoch=1)
        assert isinstance(val_result, (tuple, list))
        assert len(val_result) >= 2
        val_loss, val_acc = val_result[0], val_result[1]
        assert isinstance(val_loss, (float, torch.Tensor))
        assert isinstance(val_acc, (float, torch.Tensor))
        assert float(val_loss) >= 0.0
        assert 0.0 <= float(val_acc) <= 1.01


# ----- L1 / L2: Loss computation ve next-token hizalama -----

class TestLossComputation:
    """L1: Loss shape ve padding. L2: Next-token ile anlamlı loss."""

    def test_compute_loss_shapes(self, minimal_model, tm_config, mock_logger):
        from training_management.v2.core.loss_computation import LossComputation
        B, T, V = 2, 8, tm_config["vocab_size"]
        criterion = nn.CrossEntropyLoss(ignore_index=tm_config["pad_token_id"])
        lc = LossComputation(criterion, logger=mock_logger)
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss, acc, ppl = lc.compute_loss(logits, targets, pad_id=tm_config["pad_token_id"])
        assert loss.dim() == 0
        assert 0.0 <= acc <= 1.01
        assert ppl >= 0.0

    def test_loss_differs_for_different_targets(self, minimal_model, tm_config, mock_logger):
        from training_management.v2.core.loss_computation import LossComputation
        B, T, V = 2, 8, tm_config["vocab_size"]
        criterion = nn.CrossEntropyLoss(ignore_index=tm_config["pad_token_id"])
        lc = LossComputation(criterion, logger=mock_logger)
        logits = torch.randn(B, T, V)
        t1 = torch.randint(1, V, (B, T))
        t2 = (t1 + 1) % V
        t2[t2 == 0] = 1
        loss1, _, _ = lc.compute_loss(logits, t1, pad_id=tm_config["pad_token_id"])
        loss2, _, _ = lc.compute_loss(logits, t2, pad_id=tm_config["pad_token_id"])
        assert abs(loss1.item() - loss2.item()) > 1e-6


# ----- B1: BatchProcessor -----

class TestBatchProcessorFormats:
    """B1: parse_batch – tensor, tuple, dict formatları."""

    def test_parse_batch_tensor(self, mock_logger):
        from training_management.v2.core.batch_processor import BatchProcessor
        bp = BatchProcessor(logger=mock_logger)
        x = torch.randint(0, 100, (2, 8))
        inp, tgt = bp.parse_batch(x, allow_missing_target=False)
        assert inp is tgt
        assert inp.shape == (2, 8)

    def test_parse_batch_tuple_inputs_targets(self, mock_logger):
        from training_management.v2.core.batch_processor import BatchProcessor
        bp = BatchProcessor(logger=mock_logger)
        inp = torch.randint(0, 100, (2, 8))
        tgt = torch.randint(0, 100, (2, 8))
        i, t = bp.parse_batch((inp, tgt))
        assert i.shape == inp.shape and t.shape == tgt.shape
        assert torch.equal(i, inp) and torch.equal(t, tgt)

    def test_parse_batch_dict_input_ids_labels(self, mock_logger):
        from training_management.v2.core.batch_processor import BatchProcessor
        bp = BatchProcessor(logger=mock_logger)
        inp = torch.randint(0, 100, (2, 8))
        tgt = torch.randint(0, 100, (2, 8))
        i, t = bp.parse_batch({"input_ids": inp, "labels": tgt})
        assert i.shape == inp.shape and t.shape == tgt.shape


# ----- I1: Tek epoch entegrasyonu -----

class TestOneEpochIntegration:
    """I1: Tek epoch baştan sona çalışır, loss/accuracy sayısal."""

    def test_one_full_epoch_completes(self, training_manager_minimal):
        tm = training_manager_minimal
        train_loss, train_acc, grad_norm = tm.training_loop.train_epoch(tm.train_loader, epoch=1)
        assert math.isfinite(float(train_loss))
        assert 0.0 <= float(train_acc) <= 1.01
        assert float(grad_norm) >= 0.0

    def test_validate_epoch_completes(self, training_manager_minimal):
        tm = training_manager_minimal
        val_loss, val_acc = tm.training_loop.validate_epoch(tm.val_loader, epoch=1)[:2]
        assert math.isfinite(float(val_loss))
        assert 0.0 <= float(val_acc) <= 1.01


# ----- G1: Gradient -----

class TestGradientManagement:
    """G1: clip_grad_norm_ sonrası norm sınırlı, NaN/Inf yok."""

    def test_gradient_norm_finite_after_step(self, training_manager_minimal):
        tm = training_manager_minimal
        tm.training_loop.train_epoch(tm.train_loader, epoch=1)
        total_norm = 0.0
        for p in tm.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        assert math.isfinite(total_norm)


# ----- Train() bir epoch ve epoch callback -----

class TestTrainingManagerTrainOneEpoch:
    """TrainingManager.train() ile 1 epoch ve callback."""

    def test_train_returns_final_losses(self, training_manager_minimal):
        tm = training_manager_minimal
        # Sadece 1 epoch çalıştır
        orig_epochs = tm.epochs
        tm.epochs = 1
        try:
            final_train, final_val = tm.train(epoch_callback=None)
            assert math.isfinite(final_train)
            assert math.isfinite(final_val)
        finally:
            tm.epochs = orig_epochs

    def test_epoch_callback_invoked(self, training_manager_minimal):
        tm = training_manager_minimal
        orig_epochs = tm.epochs
        tm.epochs = 1
        called = []

        def cb(epoch, train_loss, val_loss):
            called.append((epoch, float(train_loss), float(val_loss)))

        try:
            tm.train(epoch_callback=cb)
            assert len(called) == 1
            assert called[0][0] == 1
            assert math.isfinite(called[0][1]) and math.isfinite(called[0][2])
        finally:
            tm.epochs = orig_epochs


# ----- L2: Next-token veri ile loss anlamlı (pipeline) -----

class TestLossNextTokenAlignment:
    """L2: Next-token hizalı veri ile forward+loss pipeline doğruluğu."""

    def test_next_token_data_produces_finite_loss(self, training_manager_minimal):
        """NextTokenDataset ile bir batch forward + compute_loss sonucu sonlu ve pozitif."""
        tm = training_manager_minimal
        batch = next(iter(tm.train_loader))
        inputs, targets = batch[0], batch[1]
        tm.model.train()
        logits, _ = tm.model(inputs)
        pad_id = tm.training_loop.pad_token_id
        loss, acc, _ = tm.loss_computation.compute_loss(logits, targets, pad_id=pad_id)
        assert math.isfinite(loss.item()) and loss.item() >= 0.0
        assert 0.0 <= acc <= 1.01

    def test_same_input_different_targets_different_loss(self, minimal_model, tm_config, mock_logger):
        """Aynı logits, farklı hedefler → farklı loss (kayıp hedefe duyarlı)."""
        from training_management.v2.core.loss_computation import LossComputation
        B, T, V = 2, 6, tm_config["vocab_size"]
        criterion = nn.CrossEntropyLoss(ignore_index=tm_config["pad_token_id"])
        lc = LossComputation(criterion, logger=mock_logger)
        torch.manual_seed(99)
        logits = torch.randn(B, T, V)
        t1 = torch.randint(1, V, (B, T))
        t2 = (t1 + 7) % V
        t2[t2 == 0] = 1
        l1, _, _ = lc.compute_loss(logits, t1, pad_id=tm_config["pad_token_id"])
        l2, _, _ = lc.compute_loss(logits, t2, pad_id=tm_config["pad_token_id"])
        assert abs(l1.item() - l2.item()) > 1e-5


