"""Governed checkpoint loading.

``open_checkpoint`` is the load path of the roof. It never trusts shapes alone:

  1. the file is sniffed without unpickling and read with ``weights_only=True``;
  2. the compatibility gate decides for the stated purpose (architecture known and equal to the
     target, layout as the current code builds it, tokenizer identity, dtype policy, optimizer and
     scheduler signatures for a resume);
  3. only then are the weights loaded, strictly;
  4. the weights now in the model are hashed again and compared with what passed the gate.

A refusal raises ``IncompatibleCheckpoint`` before the target model is touched. The legacy
loaders (``ModelManager.load``, ``ModelLoader``) are unchanged and remain available below.
"""
from dataclasses import dataclass

from cevahir_ai._lazy import lazy_exports
from cevahir_ai._roof import EXPORTS
from cevahir_ai.errors import AttestationError, IncompatibleCheckpoint

__all__ = sorted([*EXPORTS["model.checkpoint"], "open_checkpoint", "read_envelope", "inspect_checkpoint",
                  "OpenedCheckpoint"])
__getattr__, _legacy_dir = lazy_exports(__name__, EXPORTS["model.checkpoint"])


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


def open_checkpoint(path, *, model=None, tokenizer=None, purpose="evaluate", optimizer=None, scheduler=None,
                    gate=None, expect_file=None, expect_weights=None, dtype=None, allow_cast=False,
                    allow_migrations=(), max_seq_length=None):
    """Load ``path`` through the gate. Returns ``OpenedCheckpoint``; raises ``IncompatibleCheckpoint``.

    ``model``: the live model to load into. When omitted, the model is built from the architecture
    the checkpoint declares (through ``ModelManager(config).initialize``), which requires one.
    ``tokenizer``: the tokenizer that will be used with the model (``TokenizerCore`` or ``ByteLevelBPE``).
    ``purpose``: "evaluate", "finetune", "resume", "serve" ... ; "resume" also restores optimizer
    and scheduler state when they are given.
    """
    from cevahir_ai.adapters.inspection import Target, inspect_checkpoint as inspect
    from cevahir_ai.adapters.model_build import build_from_declared
    from cevahir_ai.keys.torchkeys.tensors import dtype_name, tensor_digests, tensor_table
    if dtype is None:
        dtype = dtype_name(next(model.parameters()).dtype) if model is not None else "float32"
    target = Target(model=model, tokenizer=tokenizer, optimizer=optimizer, scheduler=scheduler, dtype=dtype,
                    allow_cast=allow_cast, allow_migrations=tuple(allow_migrations), max_seq_length=max_seq_length)
    report = inspect(path, level="compat", purpose=purpose, target=target, expect_file=expect_file,
                     expect_weights=expect_weights, gate=gate)
    if not report.verdict.ok:
        error = IncompatibleCheckpoint(report.verdict)
        error.report = report
        raise error
    state = report.state
    if model is None:
        model = build_from_declared(report.view.model_config)
    model.load_state_dict(state, strict=True)
    live = model.state_dict()
    expected = tensor_digests(tensor_table({name: tensor.to(live[name].dtype) for name, tensor in state.items()}))
    found = tensor_digests(tensor_table(live))
    changed = sorted(name for name in expected if found.get(name) != expected[name])
    if changed or set(found) != set(expected):
        raise AttestationError("weights in the model after loading are not the weights that passed the gate: "
                               + ", ".join(changed[:6] or sorted(set(found) ^ set(expected))[:6]))
    if purpose == "resume":
        view = report.view
        if optimizer is not None and view.optimizer_state is not None:
            optimizer.load_state_dict(view.optimizer_state)
        if scheduler is not None and view.scheduler_state is not None:
            scheduler.load_state_dict(view.scheduler_state)
    if hasattr(model, "eval") and purpose in ("evaluate", "serve", "publish"):
        model.eval()
    return OpenedCheckpoint(model=model, report=report, verdict=report.verdict)
