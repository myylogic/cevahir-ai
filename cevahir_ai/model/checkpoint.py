"""Governed checkpoint loading.

``open_checkpoint`` is the load path of the roof. It never trusts shapes alone:

  1. the file is sniffed without unpickling and read with ``weights_only=True``;
  2. the compatibility gate decides for the stated purpose (architecture declared completely and
     equal to the target, layout as the current code builds it *and* as the live target has it,
     tokenizer identity, dtype policy, optimizer and scheduler signatures for a resume);
  3. only then are the weights loaded, strictly;
  4. the weights now in the model are hashed again and compared with what passed the gate.

The load is all or nothing. A refusal raises ``IncompatibleCheckpoint`` before the target model
is touched. Should the load itself fail after the gate passed (a model that refuses the weights,
an optimizer that refuses its state, an attestation mismatch), the model, the optimizer and the
scheduler are put back as they were and the failure is raised as a refusal: no caller is left
with a half-overwritten model. The price is one transient copy of the target's weights.

The legacy loaders (``ModelManager.load``, ``ModelLoader``) are unchanged and remain available
below; they are not governed.
"""
import copy
import warnings
from dataclasses import dataclass

from cevahir_ai._lazy import lazy_exports
from cevahir_ai._roof import EXPORTS
from cevahir_ai.errors import AttestationError, IncompatibleCheckpoint

__all__ = sorted([*EXPORTS["model.checkpoint"], "open_checkpoint", "read_envelope", "inspect_checkpoint",
                  "OpenedCheckpoint", "LOAD_PURPOSES"])
__getattr__, _legacy_dir = lazy_exports(__name__, EXPORTS["model.checkpoint"])

# The purposes that put weights into a model. "inspect" and "export" require only file-level
# checks, so loading under them would walk around the gate: ``open_checkpoint`` refuses them.
LOAD_PURPOSES = ("evaluate", "finetune", "resume", "serve", "publish")


def __dir__():
    return sorted(set(_legacy_dir()) | set(__all__))


@dataclass
class OpenedCheckpoint:
    model: object
    report: object       # cevahir_ai.adapters.inspection.Report (keys, facts, explain)
    verdict: object      # cevahir_ai.domain.verdicts.Verdict

    @property
    def keys(self):
        return self.report.keys


def read_envelope(path):
    """Read any known checkpoint shape with ``weights_only=True``; see ``cevahir_ai.adapters.envelopes``."""
    from cevahir_ai.adapters.envelopes import read_envelope as read
    return read(path)


def inspect_checkpoint(path, **options):
    """Facts, keys and the gate's verdict for a file; nothing is loaded into a model."""
    from cevahir_ai.adapters.inspection import inspect_checkpoint as inspect
    return inspect(path, **options)


def _refusal(report, gate, check_id, detail):
    """Turn a failure found while loading into a gate refusal: the fact fails, the verdict follows."""
    from cevahir_ai.domain.verdicts import CheckResult
    from cevahir_ai.keys.gate import check_compatibility
    report.facts[check_id] = CheckResult(check_id, "fail", detail)
    report.verdict = check_compatibility(report.facts, report.purpose, gate)
    error = IncompatibleCheckpoint(report.verdict)
    error.report = report
    return error


def _first_line(error):
    text = str(error).strip()
    return f"{type(error).__name__}: {text.splitlines()[0] if text else ''}".rstrip(": ")


class _Undo:
    """What the load is about to overwrite, kept so that a failed load changes nothing."""

    def __init__(self, model, optimizer, scheduler):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.weights = None if model is None else {name: tensor.detach().clone()
                                                   for name, tensor in model.state_dict().items()}
        self.optimizer_state = None if optimizer is None else copy.deepcopy(optimizer.state_dict())
        self.scheduler_state = None if scheduler is None else copy.deepcopy(scheduler.state_dict())

    def restore(self):
        """Put everything back. Returns what could not be put back (normally nothing), as text."""
        import torch
        lost = []
        if self.weights is not None:
            live = self.model.state_dict()
            with torch.no_grad():                      # in place: no hook of the model takes part
                for name, tensor in self.weights.items():
                    live[name].copy_(tensor)
        for label, holder, state in (("optimizer", self.optimizer, self.optimizer_state),
                                     ("scheduler", self.scheduler, self.scheduler_state)):
            if state is not None:
                try:
                    holder.load_state_dict(state)
                except Exception as error:
                    lost.append(f"; the {label} could NOT be put back ({_first_line(error)})")
        return "".join(lost)


def _plain(value):
    if isinstance(value, (list, tuple)):
        return all(_plain(item) for item in value)
    return value is None or isinstance(value, (bool, int, float, str))


def _same(left, right):
    """Equality of plain values; a tuple equals the list it was saved as, NaN equals NaN."""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) and left != left:
        return right != right
    return left == right


def restore_scheduler(scheduler, stored):
    """Load a stored scheduler state into ``scheduler`` and check that it arrived.

    The training envelope stores ``{"scheduler_type", "scheduler_state"}`` (the shape of
    ``TrainingScheduler``); a torch scheduler takes the inner state. ``LRScheduler.load_state_dict``
    is a plain ``__dict__.update`` and the training wrapper swallows its own errors, so neither
    reports a state that did not fit: every plain public entry is read back and compared.
    """
    from cevahir_ai.adapters.envelopes import unwrap_scheduler_state
    inner, _, wrapper = unwrap_scheduler_state(stored)
    _, live_type, live_wrapper = unwrap_scheduler_state(scheduler.state_dict())
    if live_wrapper:
        payload = stored if wrapper else {"scheduler_type": live_type, "scheduler_state": inner}
    else:
        payload = inner
    scheduler.load_state_dict(copy.deepcopy(payload))
    after = unwrap_scheduler_state(scheduler.state_dict())[0]
    lost = sorted(str(name) for name, value in inner.items()
                  if not str(name).startswith("_") and _plain(value)
                  and not (name in after and _same(after[name], value)))
    if lost:
        raise ValueError(f"the scheduler does not hold the stored state after loading it: {', '.join(lost[:6])}")


def open_checkpoint(path, *, model=None, tokenizer=None, purpose="evaluate", optimizer=None, scheduler=None,
                    gate=None, expect_file=None, expect_weights=None, dtype=None, allow_cast=False,
                    allow_migrations=(), max_seq_length=None):
    """Load ``path`` through the gate. Returns ``OpenedCheckpoint``; raises ``IncompatibleCheckpoint``.

    ``model``: the live model to load into. When omitted, the model is built from the architecture
    the checkpoint declares (through ``ModelManager(config).initialize``), which requires one.
    ``tokenizer``: the tokenizer that will be used with the model (``TokenizerCore`` or
    ``ByteLevelBPE``). A checkpoint that carries a tokenizer identity is refused without it, unless
    the identity is verified by the tokenizer file next to the checkpoint or the tracked registry.
    ``purpose``: one of ``LOAD_PURPOSES``; "resume" also restores optimizer and scheduler state
    when they are given. ``allow_cast``: permit up-casting stored weights to the target dtype
    (fp16 into fp32); without it a dtype difference is a refusal for every purpose.

    A verdict with warnings loads; the facts that could not be established are in
    ``verdict.warnings``. A checkpoint without any tokenizer identity also emits the
    ``RuntimeWarning`` the legacy loader emits: its token semantics cannot be verified.
    """
    from cevahir_ai.adapters.inspection import Target, inspect_checkpoint as inspect
    from cevahir_ai.adapters.model_build import build_from_declared
    from cevahir_ai.keys.torchkeys.tensors import dtype_name, tensor_digests, tensor_table
    if purpose not in LOAD_PURPOSES:
        raise ValueError(f"open_checkpoint loads weights into a model: purpose must be one of {LOAD_PURPOSES}, "
                         f"not {purpose!r} (use inspect_checkpoint to look at a file)")
    if dtype is None:
        first = next(iter(model.parameters()), None) if model is not None else None
        dtype = dtype_name(first.dtype) if first is not None else "float32"
    target = Target(model=model, tokenizer=tokenizer, optimizer=optimizer, scheduler=scheduler, dtype=dtype,
                    allow_cast=allow_cast, allow_migrations=tuple(allow_migrations), max_seq_length=max_seq_length)
    report = inspect(path, level="compat", purpose=purpose, target=target, expect_file=expect_file,
                     expect_weights=expect_weights, gate=gate)
    if not report.verdict.ok:
        error = IncompatibleCheckpoint(report.verdict)
        error.report = report
        raise error
    view, state = report.view, report.state
    if state is None or report.table is None:           # only a custom gate policy gets this far
        raise _refusal(report, gate, "C3", "the file holds no weights that a weight scheme covers")
    resume = purpose == "resume"
    undo = _Undo(model, optimizer if resume else None, scheduler if resume else None)
    if model is None:
        try:
            model = build_from_declared(view.model_config)
        except Exception as error:
            raise _refusal(report, gate, "C4", f"the declared architecture cannot be built: {_first_line(error)}") from error
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:
        lost = undo.restore()
        raise _refusal(report, gate, "C4", f"the target model refused the weights: {_first_line(error)}{lost}") from error
    live = model.state_dict()
    # The digests that passed the gate are reused unless a migration renamed tensors or an allowed
    # cast changed their dtype; then the expected digests are those of the migrated, cast tensors.
    as_gated = set(state) == {row[0] for row in report.table} and all(
        tensor.dtype == live[name].dtype for name, tensor in state.items())
    expected = tensor_digests(report.table if as_gated else tensor_table(
        {name: tensor.to(live[name].dtype) for name, tensor in state.items()}))
    found = tensor_digests(tensor_table(live))
    changed = sorted(name for name in expected if found.get(name) != expected[name])
    if changed or set(found) != set(expected):
        lost = undo.restore()
        raise AttestationError("weights in the model after loading are not the weights that passed the gate "
                               "(the model was put back as it was): "
                               + ", ".join(changed[:6] or sorted(set(found) ^ set(expected))[:6]) + lost)
    if resume:
        try:
            if optimizer is not None and view.optimizer_state is not None:
                optimizer.load_state_dict(view.optimizer_state)
            if scheduler is not None and view.scheduler_state is not None:
                restore_scheduler(scheduler, view.scheduler_state)
        except Exception as error:
            lost = undo.restore()
            raise _refusal(report, gate, "C10",
                           f"the training state could not be restored: {_first_line(error)}{lost}") from error
    # What the legacy loaders do after a load: stale KV entries belong to the old weights, and the
    # legacy writers stamp the next checkpoint with the identity found on the model.
    if hasattr(model, "clear_kv_cache"):
        model.clear_kv_cache()
    model._tokenizer_identity = view.tokenizer[1] if view.tokenizer and view.tokenizer[0] == "legacy-v3" else None
    if hasattr(model, "eval") and purpose in ("evaluate", "serve", "publish"):
        model.eval()
    if "C7" in report.verdict.warnings:
        warnings.warn("checkpoint has no tokenizer identity; token semantics cannot be verified", RuntimeWarning, stacklevel=2)
    return OpenedCheckpoint(model=model, report=report, verdict=report.verdict)
