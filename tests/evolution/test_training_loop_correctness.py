"""CPU regression tests for training-loop correctness on the supported V2 path.

Covers resume semantics (continuation learning rate, total epoch target, model selection after
resuming a checkpoint whose best is an objective value, param group names), the shipped config
on the V3 path and the V2 fallback, fp16 GradScaler overflow handling (simulated with a CPU
GradScaler), the warmup wrapper's embedding groups and the reported main learning rate, and the
reported metrics (plain NLL/perplexity, objective logged apart, analytics summary, batch timing
and token throughput).
"""
import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from training_management.v2.core.batch_processor import BatchProcessor
from training_management.v2.core.gradient_manager import GradientManager
from training_management.v2.core.loss_computation import LossComputation
from training_management.v2.core.training_loop import TrainingLoop
from training_management.v2.core.training_manager import TrainingManager
from training_management.v2.monitoring.performance_tracker import PerformanceTracker
from training_management.v2.utils.training_scheduler import (
    EMBEDDING_GROUP_NAME, TrainingScheduler, apply_continuation_learning_rate, mark_embedding_param_groups,
)
from training_system.v2.core.criterion_manager import CriterionManager

PAD, EOS, VOCAB = 0, 2, 9


class QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class RecordingLogger:
    def __init__(self):
        self.records = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        level = name[4:] if name.startswith("log_") else name
        return lambda message="", *args, **kwargs: self.records.append((level, str(message)))

    def messages(self, level):
        return [message for kind, message in self.records if kind == level]


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 5)
        self.proj = nn.Linear(5, VOCAB)

    def forward(self, inputs):
        return self.proj(self.embedding(inputs))


class FakeClock:
    """Deterministic clock: every reading advances time by `step` seconds."""

    def __init__(self, step=0.25):
        self.now, self.step = 0.0, step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


class FakeTensorBoard:
    enabled = True

    def __init__(self):
        self.scalars = {}

    def log_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, {})[step] = float(value)

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def batches():
    # Non-pad target tokens per batch: 3, 2, 4.
    return [
        (torch.tensor([[1, 3, 4, 0]]), torch.tensor([[3, 4, 2, 0]])),
        (torch.tensor([[1, 5, 0, 0]]), torch.tensor([[5, 2, 0, 0]])),
        (torch.tensor([[1, 6, 7, 8]]), torch.tensor([[6, 7, 8, 2]])),
    ]


def criterion(label_smoothing=0.0, entropy_coeff=0.0, eos_weight=1.0):
    return CriterionManager().create_criterion(VOCAB, EOS, device=torch.device("cpu"), label_smoothing=label_smoothing,
                                               entropy_coeff=entropy_coeff, eos_weight=eos_weight)


def make_loop(model, optimizer=None, scheduler=None, loss_criterion=None, logger=None):
    logger = logger or QuietLogger()
    return TrainingLoop(
        model, optimizer or torch.optim.SGD(model.parameters(), lr=0.05),
        LossComputation(loss_criterion or criterion()), GradientManager(1000, logger=logger),
        BatchProcessor(logger=logger), torch.device("cpu"),
        {"vocab_size": VOCAB, "pad_token_id": PAD, "grad_accum_steps": 1, "use_progress_bar": False, "use_amp": False},
        logger=logger, scheduler=scheduler,
    )


def make_manager(model, checkpoint_dir, epochs, optimizer=None, loss_criterion=None, logger=None, **config):
    settings = {"vocab_size": VOCAB, "device": "cpu", "epochs": epochs, "grad_accum_steps": 1, "pad_token_id": PAD,
                "use_amp": False, "use_progress_bar": False, "enable_tensorboard": False, "track_memory": False,
                "track_performance": False, "enable_training_analytics": False,
                "checkpoint_dir": str(checkpoint_dir), "scheduler_type": "ReduceLROnPlateau"}
    settings.update(config)
    tensorboard = settings.pop("tensorboard_manager", None)
    return TrainingManager(
        model, batches(), batches(), optimizer or torch.optim.AdamW(model.parameters(), lr=2e-4),
        loss_criterion or criterion(), settings, logger=logger or QuietLogger(), tensorboard_manager=tensorboard,
    )


def two_group_optimizer(model, embedding_lr, other_lr):
    return torch.optim.AdamW([{"params": list(model.embedding.parameters()), "lr": embedding_lr},
                              {"params": list(model.proj.parameters()), "lr": other_lr}])


def record_lrs(optimizer):
    """Record the learning rates every optimizer step actually uses."""
    seen, step = [], optimizer.step

    def recording_step(*args, **kwargs):
        seen.append([float(group["lr"]) for group in optimizer.param_groups])
        return step(*args, **kwargs)

    optimizer.step = recording_step
    return seen


def record_epochs(manager):
    seen, train_epoch = [], manager.training_loop.train_epoch

    def recording_train_epoch(loader, epoch=None):
        seen.append(epoch)
        return train_epoch(loader, epoch=epoch)

    manager.training_loop.train_epoch = recording_train_epoch
    return seen


# --------------------------------------------------------------------------- P4: resume semantics

def test_continuation_lr_is_applied_after_optimizer_and_scheduler_restore(tmp_path):
    first = make_manager(TinyLM(), tmp_path, 1, warmup_steps=2)
    first.train()
    assert first.optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)  # what load_state_dict restores
    resumed = make_manager(TinyLM(), tmp_path, 2, warmup_steps=2, continuation_learning_rate=5e-5)
    assert resumed.resume_from_checkpoint(tmp_path / "last.pth") == 2
    assert [group["lr"] for group in resumed.optimizer.param_groups] == [pytest.approx(5e-5)]
    warmup = resumed.scheduler.scheduler
    assert warmup.step_count == warmup.warmup_steps == 2
    assert warmup._base_lrs == [pytest.approx(5e-5)]
    assert warmup.base_scheduler._last_lr == [pytest.approx(5e-5)]
    used = record_lrs(resumed.optimizer)
    trained = record_epochs(resumed)
    resumed.train()
    assert trained == [2]
    assert used == [[pytest.approx(5e-5)]] * 3
    assert torch.load(tmp_path / "last.pth", weights_only=True)["epoch"] == 2


def test_continuation_lr_keeps_group_ratios_and_ends_an_unfinished_warmup_at_the_target(tmp_path):
    first_model = TinyLM()
    first = make_manager(first_model, tmp_path, 1, optimizer=two_group_optimizer(first_model, 1e-4, 2e-4),
                         warmup_steps=6)
    assert first.optimizer.param_groups[0]["name"] == EMBEDDING_GROUP_NAME
    first.train()
    assert first.scheduler.scheduler.step_count == 3  # checkpoint written in the middle of warmup
    model = TinyLM()
    resumed = make_manager(model, tmp_path, 2, optimizer=two_group_optimizer(model, 1e-4, 2e-4), warmup_steps=6,
                           continuation_learning_rate=5e-5)
    resumed.resume_from_checkpoint(tmp_path / "last.pth")
    warmup = resumed.scheduler.scheduler
    assert warmup._base_lrs == [pytest.approx(2.5e-5), pytest.approx(5e-5)]
    assert warmup._warmup_start_lrs == [pytest.approx(2.5e-6), pytest.approx(5e-6)]
    used = record_lrs(resumed.optimizer)
    resumed.train()
    main = [lrs[1] for lrs in used]
    assert main == sorted(main) and max(main) < 5e-5  # the remaining ramp climbs towards the target
    assert [group["lr"] for group in resumed.optimizer.param_groups] == [pytest.approx(2.5e-5), pytest.approx(5e-5)]


def test_continuation_lr_applies_to_a_legacy_checkpoint_without_runtime_state(tmp_path):
    source = TinyLM()
    optimizer = torch.optim.AdamW(source.parameters(), lr=2e-4)
    path = tmp_path / "legacy.pth"
    torch.save({"epoch": 1, "model_state_dict": source.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, path)
    resumed = make_manager(TinyLM(), tmp_path, 2, continuation_learning_rate=5e-5)
    assert resumed.resume_from_checkpoint(path) == 2
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)


def test_continuation_lr_rescales_scheduler_base_lrs_and_rejects_invalid_values():
    model = TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = TrainingScheduler(optimizer, "StepLR", logger=QuietLogger(), step_size=1, gamma=0.5)
    optimizer.step()  # no gradients: a no-op update that keeps PyTorch's step-order check quiet
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert apply_continuation_learning_rate(optimizer, 0.01, scheduler) == pytest.approx(0.2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert scheduler.scheduler.base_lrs == [pytest.approx(0.02)]
    scheduler.step()  # the schedule continues from the new learning rate
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    for bad in (0.0, -1e-4, float("nan")):
        with pytest.raises(ValueError, match="continuation_learning_rate"):
            apply_continuation_learning_rate(optimizer, bad, scheduler)
    one_cycle = TrainingScheduler(torch.optim.SGD(model.parameters(), lr=0.1), "OneCycleLR", logger=QuietLogger(),
                                  total_steps=4, max_lr=0.1)
    with pytest.raises(ValueError, match="OneCycleLR"):
        apply_continuation_learning_rate(one_cycle.optimizer, 0.01, one_cycle)


def test_epochs_is_the_total_target_when_resuming(tmp_path):
    make_manager(TinyLM(), tmp_path, 1).train()
    resumed = make_manager(TinyLM(), tmp_path, 3)
    resumed.resume_from_checkpoint(tmp_path / "last.pth")
    trained = record_epochs(resumed)
    resumed.train()
    assert trained == [2, 3]
    payload = torch.load(tmp_path / "last.pth", weights_only=True)
    assert payload["epoch"] == 3 and len(payload["training_history"]["train_loss"]) == 3
    before = (tmp_path / "last.pth").read_bytes()
    logger = RecordingLogger()
    finished = make_manager(TinyLM(), tmp_path, 3, logger=logger)
    finished.resume_from_checkpoint(tmp_path / "last.pth")
    trained = record_epochs(finished)
    history = payload["training_history"]
    assert finished.train() == (pytest.approx(history["train_loss"][-1]), pytest.approx(history["val_loss"][-1]))
    assert trained == []
    assert (tmp_path / "last.pth").read_bytes() == before
    assert any("already reaches the target epochs=3" in message for message in logger.messages("warning"))


class SpecialIdsTokenizer:
    def _special_ids(self):
        return {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def get_vocab_size(self):
        return VOCAB


def test_config_manager_forwards_resume_and_warmup_options():
    from training_system.v3.core.config_manager_v3 import ConfigManagerV3

    manager = ConfigManagerV3(logger_instance=QuietLogger())
    defaults = manager.prepare_training_config({}, SpecialIdsTokenizer(), "cpu")
    assert defaults["continuation_learning_rate"] is None and defaults["embedding_warmup_factor"] is None
    given = manager.prepare_training_config({"continuation_learning_rate": "5e-5", "embedding_warmup_factor": 0.5},
                                            SpecialIdsTokenizer(), "cpu")
    assert given["continuation_learning_rate"] == 5e-5 and given["embedding_warmup_factor"] == 0.5


def test_shipped_train_config_passes_through_both_config_managers():
    from training_system.train import TRAIN_CONFIG, normalize_config
    from training_system.v2.core.config_manager import ConfigManager
    from training_system.v3.core.config_manager_v3 import ConfigManagerV3

    precision = torch.get_float32_matmul_precision()
    try:
        shipped = normalize_config(TRAIN_CONFIG)  # what train.py hands to the V3 service or the V2 fallback
    finally:
        torch.set_float32_matmul_precision(precision)  # normalize_config changes this process-wide setting
    v3 = ConfigManagerV3(logger_instance=QuietLogger()).prepare_training_config(shipped, SpecialIdsTokenizer(), "cpu")
    v2 = ConfigManager().prepare_training_config(shipped, SpecialIdsTokenizer(), "cpu")
    assert TRAIN_CONFIG["embedding_warmup_factor"] is None
    assert v3["embedding_warmup_factor"] is None and v2["embedding_warmup_factor"] is None
    given = ConfigManager().prepare_training_config({"embedding_warmup_factor": "0.5"}, SpecialIdsTokenizer(), "cpu")
    assert given["embedding_warmup_factor"] == 0.5


def test_v2_fallback_applies_the_continuation_lr_after_loading_the_weights(tmp_path, monkeypatch):
    import training_system.v2.core.training_service as v2_service

    model = TinyLM()
    optimizer = two_group_optimizer(model, 1e-4, 2e-4)
    loads = []

    class WeightsOnlyModelManager:
        def __init__(self):
            self.model, self.optimizer = model, optimizer

        def load(self, path, weights_only=None):
            loads.append((path, weights_only))

    def service(checkpoint_dir):
        # Only what TrainingService._initialize_model uses; the rest of the service is not needed here.
        instance = object.__new__(v2_service.TrainingService)
        instance.config = {"checkpoint_dir": str(checkpoint_dir), "continuation_learning_rate": 5e-5}
        instance.model_manager, instance.logger = WeightsOnlyModelManager(), QuietLogger()
        return instance

    monkeypatch.setattr(v2_service, "MODEL_SAVE_PATH", str(tmp_path / "missing.pth"))
    service(tmp_path / "empty")._initialize_model()
    assert loads == [] and [group["lr"] for group in optimizer.param_groups] == [1e-4, 2e-4]  # nothing loaded
    found = tmp_path / "runs"
    found.mkdir()
    (found / "last.pth").write_bytes(b"")  # the stub manager does not read it
    service(found)._initialize_model()
    assert loads == [(str(found / "last.pth"), True)]
    assert [group["lr"] for group in optimizer.param_groups] == [pytest.approx(2.5e-5), pytest.approx(5e-5)]


def objective_scale_run(directory, loss_criterion):
    """Train one epoch, then rewrite last.pth/best.pth the way the previous version wrote them.

    That version reported the training objective as the validation loss: best_val_loss, the
    history, the checkpoint metric and the plateau's best were objective values, and there was no
    best_val_metric. Returns that recorded objective-scale best.
    """
    model = TinyLM()
    make_manager(model, directory, 1, optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
                 loss_criterion=loss_criterion).train()
    fresh = torch.load(directory / "last.pth", weights_only=True)
    assert fresh["extra_state"]["best_val_metric"] == "nll"
    assert fresh["extra_state"]["best_val_loss"] == fresh["training_history"]["val_loss"][-1]
    recorded = None
    for name in ("last.pth", "best.pth"):
        payload = torch.load(directory / name, weights_only=True)
        history, extra = payload["training_history"], payload["extra_state"]
        history["train_loss"] = history.pop("train_objective")
        history["val_loss"] = history.pop("val_objective")
        recorded = history["val_loss"][-1]
        del extra["best_val_metric"]
        extra["best_val_loss"] = payload["metric"] = recorded
        extra["scheduler"]["scheduler_state"]["best"] = recorded
        torch.save(payload, directory / name)
    return recorded


def resume_objective_scale_run(directory, loss_criterion, logger=None, epochs=2):
    model = TinyLM()
    manager = make_manager(model, directory, epochs, optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
                           loss_criterion=loss_criterion, logger=logger)
    manager.resume_from_checkpoint(directory / "last.pth")
    return manager


def test_an_older_checkpoints_objective_best_is_not_compared_with_nll(tmp_path):
    # Label smoothing and an EOS weight put the objective well above the plain NLL.
    regularized = criterion(label_smoothing=0.3, eos_weight=3.0)

    # The same weights (LR 0): their NLL is below the recorded objective-scale best, so comparing
    # the two would call the unchanged model a new best.
    same = tmp_path / "same"
    recorded = objective_scale_run(same, regularized)
    best_before = (same / "best.pth").read_bytes()
    logger = RecordingLogger()
    resumed = resume_objective_scale_run(same, regularized, logger=logger)
    assert resumed.best_val_metric == "objective" and resumed.best_val_loss == pytest.approx(recorded)
    assert any("validation objective value" in message for message in logger.messages("warning"))
    for group in resumed.optimizer.param_groups:
        group["lr"] = 0.0
    resumed.train()
    assert resumed.training_history["val_loss"][-1] < recorded
    assert (same / "best.pth").read_bytes() == best_before
    assert resumed.early_stopping_counter == 1 and resumed.best_val_metric == "objective"
    plateau = resumed.scheduler.scheduler
    assert plateau.best == pytest.approx(recorded) and plateau.num_bad_epochs == 1
    extra = torch.load(same / "last.pth", weights_only=True)["extra_state"]
    assert extra["best_val_metric"] == "objective" and extra["best_val_loss"] == pytest.approx(recorded)

    # A worse model (restored logits sharpened, LR 0): its NLL is above the old best's NLL and
    # still below the recorded objective-scale best. It must not replace best.pth.
    worse = tmp_path / "worse"
    recorded = objective_scale_run(worse, regularized)
    best_before = (worse / "best.pth").read_bytes()
    resumed = resume_objective_scale_run(worse, regularized)
    old_best_nll = reference_metrics(resumed.model, batches(), regularized)[0]
    with torch.no_grad():
        resumed.model.proj.weight.mul_(2.0)
        resumed.model.proj.bias.mul_(2.0)
    for group in resumed.optimizer.param_groups:
        group["lr"] = 0.0
    resumed.train()
    assert old_best_nll < resumed.training_history["val_loss"][-1] < recorded
    assert (worse / "best.pth").read_bytes() == best_before
    assert resumed.early_stopping_counter == 1


def test_beating_an_older_checkpoints_objective_best_switches_model_selection_to_nll(tmp_path):
    regularized = criterion(label_smoothing=0.3, eos_weight=3.0)
    recorded = objective_scale_run(tmp_path, regularized)
    resumed = resume_objective_scale_run(tmp_path, regularized)
    resumed.train()  # one more epoch at the same LR improves the model
    history = resumed.training_history
    assert history["val_objective"][-1] < recorded
    nll = history["val_loss"][-1]
    assert resumed.best_val_metric == "nll" and resumed.best_val_loss == nll and resumed.early_stopping_counter == 0
    plateau = resumed.scheduler.scheduler
    assert plateau.best == nll and plateau.num_bad_epochs == 0  # rebased onto the NLL scale
    best = torch.load(tmp_path / "best.pth", weights_only=True)
    assert best["epoch"] == 2
    assert best["extra_state"]["best_val_metric"] == "nll" and best["extra_state"]["best_val_loss"] == nll
    logger = RecordingLogger()
    again = resume_objective_scale_run(tmp_path, regularized, logger=logger, epochs=3)
    assert again.best_val_metric == "nll" and again.best_val_loss == nll
    assert not any("validation objective value" in message for message in logger.messages("warning"))


def test_resume_keeps_param_group_names_that_the_checkpoint_lacks(tmp_path):
    source = TinyLM()
    untagged = two_group_optimizer(source, 1e-4, 2e-4)  # saved without the "embedding" tag, as older runs did
    path = tmp_path / "untagged.pth"
    torch.save({"epoch": 1, "model_state_dict": source.state_dict(),
                "optimizer_state_dict": untagged.state_dict()}, path)
    model = TinyLM()
    resumed = make_manager(model, tmp_path, 2, optimizer=two_group_optimizer(model, 1e-4, 2e-4), warmup_steps=2)
    resumed.resume_from_checkpoint(path)
    groups = resumed.optimizer.param_groups
    assert [group.get("name") for group in groups] == [EMBEDDING_GROUP_NAME, None]
    assert [group["lr"] for group in groups] == [pytest.approx(1e-4), pytest.approx(2e-4)]
    assert resumed.scheduler.get_last_lr() == pytest.approx(2e-4)  # the main group, not the embedding group


# --------------------------------------------------------------------------- P5: fp16 GradScaler overflow

def overflow_once(parameter):
    """Turn the first backward gradient of `parameter` into inf, as an fp16 overflow would."""
    remaining = [1]

    def hook(grad):
        if remaining[0]:
            remaining[0] -= 1
            return torch.full_like(grad, float("inf"))
        return grad

    return parameter.register_hook(hook)


def warmup_loop(model, logger=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = TrainingScheduler(optimizer, "StepLR", logger=QuietLogger(), step_size=100, warmup_steps=5)
    return make_loop(model, optimizer=optimizer, scheduler=scheduler, logger=logger), scheduler


def test_fp16_overflow_skips_the_step_backs_off_the_scale_and_training_continues():
    initial = TinyLM()
    model = copy.deepcopy(initial)
    logger = RecordingLogger()
    loop, scheduler = warmup_loop(model, logger=logger)
    loop.scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 8)  # stands in for the CUDA fp16 scaler
    overflow_once(model.proj.weight)
    loss, accuracy, grad_norm = loop.train_epoch(batches())
    assert math.isfinite(loss) and math.isfinite(grad_norm)
    assert loop.skipped_steps == 1 and loop.optimizer_steps == 2
    assert loop.last_train_stats["skipped_steps"] == 1 and loop.last_train_stats["optimizer_steps"] == 2
    assert loop.scaler.get_scale() == 2.0 ** 7  # backoff_factor 0.5
    assert scheduler.scheduler.step_count == 2  # a skipped update does not advance warmup
    assert any("skipped" in message for message in logger.messages("warning"))
    assert loop.state_dict()["skipped_steps"] == 1
    # The skipped first update leaves exactly the updates of batches 2 and 3 (same warmup schedule).
    reference = copy.deepcopy(initial)
    warmup_loop(reference)[0].train_epoch(batches()[1:])
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_nonfinite_gradients_without_a_scaler_raise_a_clear_error():
    model = TinyLM()
    before = copy.deepcopy(model.state_dict())
    loop = make_loop(model)
    overflow_once(model.proj.weight)
    with pytest.raises(RuntimeError, match="batch 1") as error:
        loop.train_epoch(batches())
    assert isinstance(error.value.__cause__, FloatingPointError)
    assert "no loss scaler" in str(error.value.__cause__)
    assert loop.optimizer_steps == 0 and loop.skipped_steps == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


# --------------------------------------------------------------------------- P6: warmup wrapper

def warmup_scheduler(optimizer, **kwargs):
    return TrainingScheduler(optimizer, "none", logger=QuietLogger(), warmup_steps=10, warmup_start_factor=0.1, **kwargs)


def test_warmup_applies_the_embedding_factor_to_explicitly_named_groups():
    model = TinyLM()
    optimizer = two_group_optimizer(model, 1e-5, 2e-4)
    assert mark_embedding_param_groups(optimizer, model) == 1
    scheduler = warmup_scheduler(optimizer, embedding_warmup_factor=0.5)
    assert [group["lr"] for group in optimizer.param_groups] == [pytest.approx(5e-6), pytest.approx(2e-5)]
    assert scheduler.get_last_lr() == pytest.approx(2e-5)  # main LR: first group that is not an embedding group
    for _ in range(10):
        scheduler.step_batch()
    assert [group["lr"] for group in optimizer.param_groups] == [pytest.approx(1e-5), pytest.approx(2e-4)]


def test_warmup_does_not_infer_embedding_groups_from_a_low_learning_rate():
    model = TinyLM()
    fine_tuning = torch.optim.AdamW(model.parameters(), lr=1e-5)
    warmup_scheduler(fine_tuning)
    assert fine_tuning.param_groups[0]["lr"] == pytest.approx(1e-6)
    untagged = two_group_optimizer(model, 1e-5, 2e-4)
    warmup_scheduler(untagged, embedding_warmup_factor=1.0)
    assert [group["lr"] for group in untagged.param_groups] == [pytest.approx(1e-6), pytest.approx(2e-5)]


def test_embedding_groups_warm_up_like_the_rest_when_no_factor_is_configured():
    model = TinyLM()
    optimizer = two_group_optimizer(model, 2e-4, 2e-4)
    mark_embedding_param_groups(optimizer, model)
    warmup_scheduler(optimizer)
    assert [group["lr"] for group in optimizer.param_groups] == [pytest.approx(2e-5)] * 2
    with pytest.raises(ValueError, match="embedding_warmup_factor"):
        warmup_scheduler(two_group_optimizer(model, 2e-4, 2e-4), embedding_warmup_factor=-0.5)


def test_only_groups_made_of_embedding_weights_are_tagged():
    from model_management.config_schema import tiny_model_config
    from model_management.model_manager import ModelManager

    model = TinyLM()
    mixed = torch.optim.SGD(model.parameters(), lr=0.1)
    assert mark_embedding_param_groups(mixed, model) == 0 and "name" not in mixed.param_groups[0]
    named = torch.optim.SGD([{"params": list(model.embedding.parameters()), "name": "custom"}], lr=0.1)
    assert mark_embedding_param_groups(named, model) == 0 and named.param_groups[0]["name"] == "custom"
    manager = ModelManager(tiny_model_config(num_layers=1, vocab_size=50, device="cpu"))
    manager.initialize(build_optimizer=True, build_criterion=False, build_scheduler=False)
    groups = manager.optimizer.param_groups
    assert mark_embedding_param_groups(manager.optimizer, manager.model) == 1
    assert [group.get("name") for group in groups] == [EMBEDDING_GROUP_NAME, None, None]
    assert len(groups[0]["params"]) == 1 and groups[0]["params"][0] is manager.model.embedding.embedding.weight


def test_manager_applies_the_configured_embedding_warmup_factor(tmp_path):
    model = TinyLM()
    manager = make_manager(model, tmp_path, 1, optimizer=two_group_optimizer(model, 1e-4, 2e-4),
                           warmup_steps=4, warmup_start_factor=0.1, embedding_warmup_factor=0.5)
    assert [group["lr"] for group in manager.optimizer.param_groups] == [pytest.approx(5e-5), pytest.approx(2e-5)]


def test_logged_learning_rate_is_the_main_group_not_the_largest(tmp_path):
    model = TinyLM()
    board = FakeTensorBoard()
    logger = RecordingLogger()
    manager = make_manager(model, tmp_path, 1, optimizer=two_group_optimizer(model, 2e-4, 2e-4), warmup_steps=10,
                           warmup_start_factor=0.1, embedding_warmup_factor=1.0, tensorboard_manager=board,
                           logger=logger)
    manager.train()
    embedding_lr, main_lr = (group["lr"] for group in manager.optimizer.param_groups)
    assert embedding_lr == pytest.approx(2e-4)  # kept out of warmup, above the main group
    assert main_lr == pytest.approx(2e-5 + 0.3 * (2e-4 - 2e-5))  # 3 of 10 warmup steps
    assert manager.scheduler.get_last_lr() == pytest.approx(main_lr)
    assert board.scalars["LR"][1] == pytest.approx(main_lr)
    info = logger.messages("info")
    assert "  Learning Rate (main): 0.00002000" in info  # warmup start of the main group
    assert f"LR (warmup, main): {main_lr:.8f} (warmup devam ediyor)" in info


def test_v3_service_tags_embedding_groups_and_forwards_the_factor(tmp_path, monkeypatch):
    import training_system.v3.core.training_service_v3 as module
    from model_management.config_schema import tiny_model_config

    class SmallTokenizer:
        def __init__(self, config):
            pass

        def get_vocab(self):
            return {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "a": 4, "b": 5, "c": 6, "d": 7, "e": 8}

        def get_vocab_size(self):
            return VOCAB

        def _special_ids(self):
            return {key: value for key, value in self.get_vocab().items() if key.startswith("<")}

    monkeypatch.setattr(module, "TokenizerCore", SmallTokenizer)
    monkeypatch.setattr(module.BPEValidator, "validate_files", lambda *args: None)
    monkeypatch.setattr("training_system.cache_identity.tokenizer_digest", lambda tokenizer: "0" * 64)
    captured = {}

    def capture(self, epoch_callback=None):
        captured["manager"] = self
        return 1.0, 1.0

    monkeypatch.setattr(TrainingManager, "train", capture)
    config = tiny_model_config(num_layers=1, data_dir=str(tmp_path), vocab_path=str(tmp_path / "vocab.json"),
                               merges_path=str(tmp_path / "merges.txt"), cache_dir=str(tmp_path / "cache"),
                               enable_data_cache=False, checkpoint_dir=str(tmp_path / "checkpoints"), epochs=1,
                               test_prompts=[], warmup_steps=4, warmup_start_factor=0.1,
                               embedding_warmup_factor=0.5, use_amp=False, enable_tensorboard=False,
                               use_progress_bar=False, track_memory=False, track_performance=False)
    service = module.TrainingServiceV3(config)
    training = service.config_manager.prepare_training_config(service.config, service.tokenizer_core, "cpu")
    optimizer = service.model_manager.optimizer
    base = [group["lr"] for group in optimizer.param_groups]
    service._run_training(batches()[:1], batches()[:1], training, optimizer)
    assert optimizer.param_groups[0]["name"] == EMBEDDING_GROUP_NAME
    warmup = captured["manager"].scheduler.scheduler
    assert warmup._warmup_start_lrs == [pytest.approx(0.5 * base[0])] + [pytest.approx(0.1 * lr) for lr in base[1:]]


# --------------------------------------------------------------------------- P7: reported metrics

def reference_metrics(model, data, loss_criterion):
    """Token-weighted plain NLL and training objective of `data`, computed independently."""
    nll_sum = objective_sum = 0.0
    tokens = 0
    with torch.no_grad():
        for inputs, targets in data:
            logits = model(inputs).reshape(-1, VOCAB)
            flat = targets.reshape(-1)
            valid = flat != PAD
            count = int(valid.sum())
            nll = F.cross_entropy(logits[valid], flat[valid])
            per_token = F.cross_entropy(logits, flat, weight=loss_criterion.weight, reduction="none",
                                        label_smoothing=loss_criterion.label_smoothing)
            objective = per_token[valid].mean()
            log_probs = logits.log_softmax(-1)
            entropy = -(log_probs.exp() * log_probs).sum(-1)
            objective = objective - loss_criterion.entropy_coeff * entropy[valid].mean()
            nll_sum += nll.item() * count
            objective_sum += objective.item() * count
            tokens += count
    return nll_sum / tokens, objective_sum / tokens


def test_compute_loss_keeps_the_objective_and_reports_plain_nll_perplexity():
    regularized = criterion(label_smoothing=0.1, entropy_coeff=0.01, eos_weight=3.0)
    logits = torch.randn(2, 4, VOCAB, requires_grad=True)
    targets = torch.tensor([[3, 4, 2, 0], [5, 2, 2, 0]])
    details = LossComputation(regularized).compute_loss_details(logits, targets, PAD)
    valid = targets != PAD
    per_token = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), weight=regularized.weight,
                                reduction="none", label_smoothing=0.1)[valid.reshape(-1)]
    log_probs = logits[valid].log_softmax(-1)
    objective = per_token.mean() - 0.01 * (-(log_probs.exp() * log_probs).sum(-1)).mean()
    torch.testing.assert_close(details.loss, objective)
    plain = F.cross_entropy(logits[valid], targets[valid]).item()
    assert details.nll == pytest.approx(plain, rel=1e-6)
    assert details.perplexity == pytest.approx(math.exp(plain), rel=1e-6)
    loss, accuracy, perplexity = LossComputation(regularized).compute_loss(logits, targets, PAD)
    torch.testing.assert_close(loss, objective)
    assert perplexity == pytest.approx(math.exp(plain), rel=1e-6)
    assert not math.isclose(loss.item(), plain, rel_tol=1e-3)  # the regularized objective differs
    plain_details = LossComputation(criterion()).compute_loss_details(logits, targets, PAD)
    assert plain_details.nll == pytest.approx(plain_details.loss.item())


def test_epochs_report_plain_nll_and_keep_the_objective_apart():
    model = TinyLM()
    regularized = criterion(label_smoothing=0.1, entropy_coeff=0.01, eos_weight=3.0)
    loop = make_loop(model, optimizer=torch.optim.SGD(model.parameters(), lr=0.0), loss_criterion=regularized)
    expected_nll, expected_objective = reference_metrics(model, batches(), regularized)
    assert not math.isclose(expected_nll, expected_objective, rel_tol=1e-3)
    train_loss, _, _ = loop.train_epoch(batches())
    assert train_loss == pytest.approx(expected_nll, rel=1e-6)
    assert loop.last_train_stats["nll"] == pytest.approx(expected_nll, rel=1e-6)
    assert loop.last_train_stats["perplexity"] == pytest.approx(math.exp(expected_nll), rel=1e-6)
    assert loop.last_train_stats["objective"] == pytest.approx(expected_objective, rel=1e-6)
    val_loss, _ = loop.validate_epoch(batches())
    assert val_loss == pytest.approx(expected_nll, rel=1e-6)
    assert loop.last_val_stats["objective"] == pytest.approx(expected_objective, rel=1e-6)


def test_training_analytics_summarizes_the_reported_nll(tmp_path):
    regularized = criterion(label_smoothing=0.3, eos_weight=3.0)
    model = TinyLM()
    logger = RecordingLogger()
    manager = make_manager(model, tmp_path, 1, optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
                           loss_criterion=regularized, logger=logger, enable_training_analytics=True,
                           analytics_log_every_n_batches=1)
    manager.train()
    per_batch = [reference_metrics(model, [batch], regularized)[0] for batch in batches()]
    assert manager.training_analytics.loss_history == pytest.approx(per_batch, rel=1e-6)
    summary = next(message for message in logger.messages("info") if message.startswith("Loss: avg="))
    average, lowest, highest = (float(part.split("=")[1]) for part in summary[len("Loss: "):].split(", "))
    assert lowest <= average <= highest
    assert average == pytest.approx(manager.training_history["train_loss"][-1], abs=1e-6)


def test_validation_times_each_batch_once_with_actual_tokens():
    tracker = PerformanceTracker(enabled=True)
    tracker.clock = FakeClock(0.25)
    loop = make_loop(TinyLM())
    loop.performance_tracker = tracker
    tracker.start_batch()  # a training batch that was never closed must not leak into validation
    tracker.clock.now += 100.0
    loop.validate_epoch(batches())
    assert tracker.batch_times == [0.25, 0.25, 0.25]
    assert tracker.batch_tokens == [3, 2, 4] and tracker.batch_samples == [1, 1, 1]
    assert loop.last_val_stats["performance"]["tokens_per_sec"] == pytest.approx(9 / 0.75)
    tracker.start_batch()
    assert tracker.end_batch(batch_size=1, seq_len=4, tokens=3) is not None
    assert tracker.end_batch(batch_size=1, seq_len=4, tokens=3) is None  # one end per start
    assert len(tracker.batch_times) == 4


def test_manager_logs_nll_perplexity_objective_and_actual_token_throughput(tmp_path):
    regularized = criterion(label_smoothing=0.1, entropy_coeff=0.01)
    board = FakeTensorBoard()
    manager = make_manager(TinyLM(), tmp_path, 1, loss_criterion=regularized, track_performance=True,
                           tensorboard_manager=board, batch_size=64, seq_len=512)
    manager.training_loop.performance_tracker.clock = FakeClock(0.25)
    manager.train()
    scalars = {tag: values[1] for tag, values in board.scalars.items()}
    val_nll, val_objective = reference_metrics(manager.model, batches(), regularized)
    assert scalars["Loss/Validation"] == pytest.approx(val_nll, rel=1e-6)
    assert scalars["Perplexity/Validation"] == pytest.approx(math.exp(val_nll), rel=1e-6)
    assert scalars["Objective/Validation"] == pytest.approx(val_objective, rel=1e-6)
    assert scalars["Loss/Train"] == pytest.approx(manager.training_loop.last_train_stats["nll"])
    assert scalars["Objective/Train"] == pytest.approx(manager.training_loop.last_train_stats["objective"])
    assert manager.training_history["val_loss"] == [pytest.approx(val_nll, rel=1e-6)]
    assert manager.training_history["val_objective"] == [pytest.approx(val_objective, rel=1e-6)]
    # 3 training batches timed at 0.25 s each, holding 3 samples and 3 + 2 + 4 non-pad target tokens.
    assert scalars["Performance/AvgBatchTime"] == pytest.approx(0.25)
    assert scalars["Performance/SamplesPerSec"] == pytest.approx(3 / 0.75)
    assert scalars["Performance/TokensPerSec"] == pytest.approx(9 / 0.75)
    assert "Train/SkippedSteps" not in scalars
