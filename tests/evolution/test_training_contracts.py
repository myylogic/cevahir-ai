"""Small CPU regression tests for the supported training pipeline."""
import copy
import logging
import math

import pytest
import torch
from torch import nn

from training_management.contracts import resolve_precision, validate_training_backend
from training_management.v2.core.batch_processor import BatchProcessor
from training_management.v2.core.gradient_manager import GradientManager
from training_management.v2.core.loss_computation import LossComputation
from training_management.v2.core.training_loop import TrainingLoop
from training_management.v2.core.training_manager import TrainingManager
from training_management.v2.utils.training_scheduler import TrainingScheduler
from training_system.v2.core.criterion_manager import CriterionManager
from training_system.v3.core.training_service_v3 import TrainingServiceV3
from training_system.v3.data.collator_v3 import DynamicPaddingCollator
from training_system.v3.data.dataloader_v3 import create_dataloaders_v3


class QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class TinyLM(nn.Module):
    def __init__(self, dropout=0.0, auxiliary=False):
        super().__init__()
        self.embed = nn.Embedding(9, 5)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(5, 9)
        self.route = nn.Parameter(torch.tensor(0.7))
        self.auxiliary = auxiliary
        self.pending = None

    def forward(self, inputs):
        self.pending = 0.05 * self.route.square() if self.auxiliary else None
        # Second tuple item is attention metadata, never an auxiliary objective.
        return self.proj(self.dropout(self.embed(inputs))), torch.tensor(99.0)

    def get_and_reset_moe_loss(self):
        loss, self.pending = self.pending, None
        return loss


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(12)
    yield
    torch.set_num_threads(previous)


def batches():
    return [
        (torch.tensor([[1, 3, 4, 0]]), torch.tensor([[3, 4, 2, 0]])),
        (torch.tensor([[1, 5, 0, 0]]), torch.tensor([[5, 2, 0, 0]])),
        (torch.tensor([[1, 6, 7, 8]]), torch.tensor([[6, 7, 8, 2]])),
    ]


def make_loop(model, accumulation=1, scheduler=None, optimizer=None):
    logger = QuietLogger()
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = CriterionManager().create_criterion(9, 2, device=torch.device("cpu"), label_smoothing=0)
    return TrainingLoop(
        model, optimizer, LossComputation(criterion), GradientManager(1000, logger=logger),
        BatchProcessor(logger=logger), torch.device("cpu"),
        {"vocab_size": 9, "pad_token_id": 0, "grad_accum_steps": accumulation,
         "use_progress_bar": False, "use_amp": False},
        logger=logger, scheduler=scheduler,
    )


def test_criterion_entropy_signature_and_masked_objective():
    criterion = CriterionManager().create_criterion(9, 2, device=torch.device("cpu"), label_smoothing=0, entropy_coeff=0.2)
    logits = torch.randn(1, 3, 9, requires_grad=True)
    targets = torch.tensor([[3, 2, 0]])
    loss, _, _ = LossComputation(criterion).compute_loss(logits, targets, 0)
    valid_logits = logits[:, :2].reshape(-1, 9)
    lp = valid_logits.log_softmax(-1)
    expected = nn.functional.cross_entropy(valid_logits, targets[:, :2].reshape(-1)) + 0.2 * (lp.exp() * lp).sum(-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[:, 2].abs().sum() == 0


@pytest.mark.parametrize("accumulation", [3, 4])
def test_accumulation_matches_token_weighted_full_batch_including_tail(accumulation):
    small = TinyLM()
    full = copy.deepcopy(small)
    small_loop = make_loop(small, accumulation)
    full_loop = make_loop(full)
    micro = batches()
    small_result = small_loop.train_epoch(micro)
    full_result = full_loop.train_epoch([(torch.cat([x for x, _ in micro]), torch.cat([y for _, y in micro]))])
    for a, b in zip(small.parameters(), full.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    assert small_result[0] == pytest.approx(full_result[0], abs=1e-6)
    assert small_loop.optimizer_steps == full_loop.optimizer_steps == 1


def test_auxiliary_objective_is_consumed_once_and_attention_is_ignored():
    model = TinyLM(auxiliary=True)
    loop = make_loop(model, accumulation=3)
    initial = model.route.item()
    loop.train_epoch(batches())
    assert model.route.item() == pytest.approx(initial - 0.05 * 0.1 * initial, abs=1e-6)
    assert model.pending is None


def test_tiny_training_reduces_loss():
    model = TinyLM()
    loop = make_loop(model)
    initial = loop.validate_epoch(batches())[0]
    for _ in range(4):
        loop.train_epoch(batches())
    assert loop.validate_epoch(batches())[0] < initial
    assert loop.optimizer_steps == 12


def test_scheduler_kwargs_and_tail_step_count():
    model = TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = TrainingScheduler(optimizer, "StepLR", logger=QuietLogger(), scheduler_kwargs={"step_size": 2, "gamma": 0.3}, warmup_steps=5)
    assert scheduler.scheduler.base_scheduler.step_size == 2
    assert scheduler.scheduler.base_scheduler.gamma == 0.3
    loop = make_loop(model, accumulation=2, optimizer=optimizer, scheduler=scheduler)
    loop.train_epoch(batches())
    assert loop.optimizer_steps == scheduler.scheduler.step_count == 2


def test_onecycle_advances_per_optimizer_update():
    model = TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = TrainingScheduler(optimizer, "OneCycleLR", logger=QuietLogger(), total_steps=4, max_lr=0.05)
    loop = make_loop(model, accumulation=2, optimizer=optimizer, scheduler=scheduler)
    loop.train_epoch(batches())
    before = scheduler.scheduler.last_epoch
    scheduler.step_epoch(metric=1.0)
    assert scheduler.scheduler.last_epoch == before == 2


def manager(model, tmp_path, epochs):
    return TrainingManager(
        model, batches(), batches(), torch.optim.AdamW(model.parameters(), lr=0.005),
        CriterionManager().create_criterion(9, 2, device=torch.device("cpu"), label_smoothing=0),
        {"vocab_size": 9, "device": "cpu", "epochs": epochs, "grad_accum_steps": 2,
         "pad_token_id": 0, "use_amp": False, "use_progress_bar": False,
         "enable_tensorboard": False, "track_memory": False, "track_performance": False,
         "enable_training_analytics": False, "checkpoint_dir": str(tmp_path),
         "scheduler_type": "StepLR", "scheduler_kwargs": {"step_size": 1, "gamma": 0.7}},
        logger=QuietLogger(),
    )


def test_checkpoint_resume_matches_uninterrupted_dropout_training(tmp_path):
    initial = TinyLM(dropout=0.2)
    whole = manager(copy.deepcopy(initial), tmp_path / "whole", 2)
    torch.manual_seed(456)
    whole.train()
    first = manager(copy.deepcopy(initial), tmp_path / "split", 1)
    torch.manual_seed(456)
    first.train()
    path = tmp_path / "split" / "last.pth"
    assert path.exists()
    payload = torch.load(path, weights_only=True)
    assert payload["epoch"] == 1
    assert payload["extra_state"]["loop"]["optimizer_steps"] == 2
    resumed = manager(TinyLM(dropout=0.2), tmp_path / "split", 1)
    assert resumed.resume_from_checkpoint(path) == 2
    resumed.train()
    for a, b in zip(whole.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert resumed.training_loop.optimizer_steps == whole.training_loop.optimizer_steps == 4
    assert resumed.training_history == whole.training_history
    assert resumed.scheduler.get_last_lr() == whole.scheduler.get_last_lr()


def test_padding_is_trimmed_without_losing_eos():
    collate = DynamicPaddingCollator(pad_id=0)
    x, y = collate([(torch.tensor([1, 3, 0, 0]), torch.tensor([3, 2, 0, 0]))])
    assert x.tolist() == [[1, 3]]
    assert y.tolist() == [[3, 2]]


def test_bucket_epoch_changes_and_repeats_deterministically():
    data = [(torch.tensor([1, i % 6 + 3]), torch.tensor([i % 6 + 3, 2])) for i in range(40)]
    train, _ = create_dataloaders_v3(data, data[:2], 2, device="cpu", num_buckets=2)
    sampler = train.batch_sampler
    sampler.set_epoch(1)
    a = list(sampler)
    sampler.set_epoch(2)
    b = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) == a and a != b


def test_split_rejects_single_source_and_mixed_metadata():
    service = object.__new__(TrainingServiceV3)
    service.config = {}
    service.logger = logging.getLogger("test")
    with pytest.raises(ValueError, match="two independent"):
        service._source_id_aware_split([([1, 3], [3, 2], 0)] * 2)
    with pytest.raises(ValueError, match="mixed"):
        service._source_id_aware_split([([1, 3], [3, 2], 0), ([1, 3], [3, 2])])


@pytest.mark.parametrize("with_ids", [False, True])
def test_duplicate_content_cannot_cross_training_validation(with_ids):
    service = object.__new__(TrainingServiceV3)
    service.config = {"train_val_split": .5}
    service.logger = logging.getLogger("test")
    data = [([1, 4], [4, 2], "a"), ([1, 4, 0], [4, 2, 0], "b"),
            ([1, 5], [5, 2], "b"), ([1, 6], [6, 2], "c")]
    if not with_ids:
        data = [item[:2] for item in data]
    train, val = service._source_id_aware_split(data)
    def signatures(items):
        return {tuple(int(v) for v in x if v != 0) for x, _ in items}
    assert not signatures(train) & signatures(val)
    assert len(train) + len(val) == len(data)
    if with_ids:
        assert ({(1, 4), (1, 5)} <= signatures(train)) or ({(1, 4), (1, 5)} <= signatures(val))
    repeated = service._source_id_aware_split(data)
    assert signatures(repeated[0]) == signatures(train)


def test_transitively_duplicated_sources_are_not_independent():
    service = object.__new__(TrainingServiceV3)
    service.config = {}
    service.logger = logging.getLogger("test")
    data = [([1, 4], [4, 2], "a"), ([1, 4], [4, 2], "b"),
            ([1, 5], [5, 2], "b"), ([1, 5], [5, 2], "c")]
    with pytest.raises(ValueError, match="independent"):
        service._source_id_aware_split(data)


def test_explicit_supported_backend_and_cpu_precision():
    validate_training_backend({"training_backend": "v2"})
    with pytest.raises(ValueError, match="Unsupported"):
        validate_training_backend({"use_ema": True})
    with pytest.raises(ValueError, match="integrated"):
        validate_training_backend({"training_backend": "v3"})
    assert resolve_precision({"precision": "fp16"}, "cpu") == "fp32"
    assert resolve_precision({"precision": "auto", "use_amp": True}, "cpu") == "fp32"


def test_empty_epoch_cannot_report_success():
    loop = make_loop(TinyLM())
    with pytest.raises(ValueError, match="No valid"):
        loop.train_epoch([])
    with pytest.raises(ValueError, match="No valid"):
        loop.validate_epoch([])


def test_v3_service_initializes_real_core_and_runs_supported_training(tmp_path, monkeypatch):
    # Only the tokenizer asset boundary is substituted; the model, criterion,
    # optimizer, scheduler, training manager and checkpoint writer are real.
    import training_system.v3.core.training_service_v3 as module
    from model_management.config_schema import tiny_model_config
    class SmallTokenizer:
        def __init__(self, config):
            pass
        def get_vocab(self):
            return {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
                    "a": 4, "b": 5, "c": 6, "d": 7, "e": 8}
        def get_vocab_size(self):
            return 9
        def _special_ids(self):
            return {k: v for k, v in self.get_vocab().items() if k.startswith("<")}
    monkeypatch.setattr(module, "TokenizerCore", SmallTokenizer)
    monkeypatch.setattr(module.BPEValidator, "validate_files", lambda *args: None)
    config = tiny_model_config(num_layers=1, data_dir=str(tmp_path),
        vocab_path=str(tmp_path / "vocab.json"), merges_path=str(tmp_path / "merges.txt"),
        cache_dir=str(tmp_path / "cache"), enable_data_cache=False,
        checkpoint_dir=str(tmp_path / "checkpoints"), epochs=1, test_prompts=[],
        scheduler_type="StepLR", scheduler_kwargs={"step_size": 1, "gamma": .7},
        entropy_coeff=.01, use_amp=False, enable_tensorboard=False,
        use_progress_bar=False, track_memory=False, track_performance=False,
        enable_training_analytics=False)
    service = TrainingServiceV3(config)
    before = service.model_manager.model.output_layer.weight.detach().clone()
    monkeypatch.setattr(service, "_test_model_inline", lambda *args: None)
    training = service.config_manager.prepare_training_config(service.config, service.tokenizer_core, "cpu")
    losses = service._run_training(batches()[:1], batches()[:1], training, service.model_manager.optimizer)
    assert all(math.isfinite(loss) for loss in losses)
    assert not torch.equal(before, service.model_manager.model.output_layer.weight)
    assert (tmp_path / "checkpoints" / "last.pth").exists()
