"""Envelope readers: every checkpoint shape that exists today, read without running pickled code.

| id              | writer                                                   |
|-----------------|----------------------------------------------------------|
| tm2             | training_management.v2 CheckpointManager.save (training) |
| mm-save         | ModelManager.save / ModelSaver.save_model                |
| saver-4.1       | ModelSaver.save_checkpoint (+ .sha256 sidecar)           |
| bare            | ModelSaver.save_weights_only (a plain state dict)        |
| module          | ModelSaver.save_full_model (pickled nn.Module): refused  |
| tm3             | training_management.v3 CheckpointManager                 |
| cev11-v1        | CEV-11 save_checkpoint (research, read-only)             |
| cev11-ledger-v1 | CEV-11 Ledger.export (research, read-only)               |

``sniff_pickle_globals`` is torch-free: it lists the globals a checkpoint would import, without
unpickling. ``read_envelope`` always loads with ``weights_only=True``.
"""
import pickletools
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from cevahir_ai.errors import EnvelopeError

# Globals that ``torch.load(weights_only=True)`` accepts for tensors and plain containers.
_SAFE_EXACT = {
    "collections.OrderedDict", "collections.defaultdict", "builtins.set", "builtins.frozenset",
    "builtins.dict", "builtins.list", "builtins.tuple", "builtins.int", "builtins.float",
    "builtins.complex", "builtins.bool", "builtins.str", "builtins.bytes", "builtins.bytearray",
    "builtins.slice", "builtins.range", "_codecs.encode", "torch.Size", "torch.device", "torch.Tensor",
    "torch.nn.parameter.Parameter", "torch._tensor._rebuild_from_type_v2",
    "torch.serialization._get_layout", "torch.torch_version.TorchVersion",
}
# One plain attribute of torch._utils: a dotted tail (``_rebuild_tensor.__globals__``) is an
# attribute walk, which protocol 4 resolves, and is never safe.
_SAFE_REBUILD = re.compile(r"torch\._utils\._rebuild_\w+")
_TORCH_DTYPES = {"float16", "float32", "float64", "bfloat16", "half", "float", "double", "int8", "int16",
                 "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "long", "int", "short",
                 "complex64", "complex128", "strided"}
ENVELOPES = ("tm2", "mm-save", "saver-4.1", "bare", "module", "tm3", "cev11-v1", "cev11-ledger-v1")


def is_safe_global(name):
    if name in _SAFE_EXACT or _SAFE_REBUILD.fullmatch(name):
        return True
    module, _, attribute = name.rpartition(".")
    if module == "torch" and (attribute.endswith("Storage") or attribute in _TORCH_DTYPES):
        return True
    return False


class _Opaque:
    """A stack item whose value the sniff does not follow (anything but a string constant)."""

    def __repr__(self):
        return "<opaque>"


_OPAQUE, _MARK = _Opaque(), _Opaque()
_UNICODE_OPS = ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE")
_BYTE_STRING_OPS = ("STRING", "BINSTRING", "SHORT_BINSTRING")
_MEMO_PUT, _MEMO_GET = ("BINPUT", "LONG_BINPUT", "PUT"), ("BINGET", "LONG_BINGET", "GET")


def _pickle_globals(data):
    """Names of the globals a pickle stream imports, without executing it.

    The opcodes are applied to an abstract copy of the unpickler's stack and memo on which only
    string constants keep their value. ``STACK_GLOBAL`` is therefore resolved from the two items
    the real unpickler pops, not from the strings written last: decoy strings that are pushed and
    popped again, memo round trips, ``DUP`` and mark tricks do not hide a name. Whatever cannot
    be resolved statically (operands computed at run time, extension codes, a stack underflow, a
    malformed stream) is reported as ``<unresolved ...>``, which is outside the safe set.
    """
    found, stack, memo = [], [], {}

    def top():
        if not stack or stack[-1] is _MARK:
            raise ValueError("stack underflow")
        return stack[-1]

    def pop():
        top()
        return stack.pop()

    try:
        for opcode, arg, _ in pickletools.genops(data):
            name = opcode.name
            if name == "MARK":
                stack.append(_MARK)
            elif name == "POP" and stack and stack[-1] is _MARK:
                stack.pop()
            elif name in _UNICODE_OPS:
                stack.append(arg)
            elif name in _BYTE_STRING_OPS:
                # Decoded by the unpickler with its ``encoding`` argument: only ASCII is certain.
                stack.append(arg if isinstance(arg, str) and arg.isascii() else _OPAQUE)
            elif name == "DUP":
                stack.append(top())
            elif name == "MEMOIZE":
                memo[len(memo)] = top()
            elif name in _MEMO_PUT:
                memo[int(arg)] = top()
            elif name in _MEMO_GET:
                stack.append(memo.get(int(arg), _OPAQUE))
            elif name == "STACK_GLOBAL":
                attribute, module = pop(), pop()
                if isinstance(module, str) and isinstance(attribute, str):
                    found.append(f"{module}.{attribute}")
                else:
                    found.append("<unresolved global: its name is computed at run time>")
                stack.append(_OPAQUE)
            else:
                if name in ("GLOBAL", "INST"):
                    module, _, attribute = arg.partition(" ")
                    found.append(f"{module}.{attribute}")
                elif name in ("EXT1", "EXT2", "EXT4"):
                    found.append(f"<unresolved global: extension code {arg}>")
                before = list(opcode.stack_before)
                if pickletools.markobject in before:
                    while stack and stack[-1] is not _MARK:
                        stack.pop()
                    if not stack:
                        raise ValueError("no mark on the stack")
                    stack.pop()
                    before = before[:before.index(pickletools.markobject)]
                for _ in before:
                    pop()
                stack.extend(_OPAQUE for _ in opcode.stack_after)
    except Exception as error:      # pickletools raises ValueError; garbage can raise others too
        found.append(f"<unresolved: malformed pickle stream ({type(error).__name__})>")
    return found


def sniff_pickle_globals(path):
    """Torch-free safety sniff. Returns ``{"container", "globals", "unsafe"}``.

    ``container`` is ``"torch-zip"`` for the zip format ``torch.save`` writes, else ``"unknown"``.
    ``unsafe`` lists the globals outside the ``weights_only`` safe set, and whatever the sniff
    could not resolve: a non-empty list means the file needs code execution to load (a pickled
    module, or an arbitrary object) or cannot be shown not to.
    """
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return {"container": "unknown", "globals": [], "unsafe": []}
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/data.pkl") or name == "data.pkl"]
        if len(members) != 1:
            return {"container": "unknown", "globals": [], "unsafe": []}
        found = _pickle_globals(archive.read(members[0]))
    unique = sorted(set(found))
    return {"container": "torch-zip", "globals": unique, "unsafe": [name for name in unique if not is_safe_global(name)]}


@dataclass
class EnvelopeView:
    envelope: str
    state: dict = None                 # name -> tensor
    model_config: dict = None          # the stored architecture config, as written
    config_source: str = None          # which envelope field it came from
    tokenizer: tuple = None            # (scheme, 64-hex), e.g. ("legacy-v3", "…")
    tokenizer_problem: str = None      # an identity is stored, but no product scheme covers it
    tokenizer_file: str = None         # sibling tokenizer file name (CEV-11)
    optimizer_state: dict = None
    scheduler_state: dict = None
    extra_state: dict = None
    epoch: int = None
    metric: float = None
    declared: dict = field(default_factory=dict)   # digests and readouts the file states about itself
    notes: list = field(default_factory=list)
    top_keys: list = field(default_factory=list)


def _is_state(mapping):
    import torch
    return isinstance(mapping, dict) and bool(mapping) and all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in mapping.items())


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _mapping(value, what, required=False):
    """``value`` as a dict (``{}`` for an absent optional field); anything else is a malformed envelope."""
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise EnvelopeError(f"malformed envelope: {what} must be a mapping, not {type(value).__name__}")
    return value


def _state(value, what):
    state = _mapping(value, what, required=True)
    if not all(isinstance(name, str) for name in state):
        raise EnvelopeError(f"malformed envelope: {what} has names that are not strings")
    return dict(state)


def _optimizer_state(value):
    """An optimizer ``state_dict`` or None; a value of another structure is a malformed envelope."""
    if value is None:
        return None
    state = _mapping(value, "optimizer state", required=True)
    groups = state.get("param_groups")
    if groups is not None and (not isinstance(groups, (list, tuple)) or not all(isinstance(g, dict) for g in groups)):
        raise EnvelopeError("malformed envelope: optimizer param_groups must be a list of mappings")
    if state.get("state") is not None and not isinstance(state["state"], dict):
        raise EnvelopeError("malformed envelope: optimizer per-parameter state must be a mapping")
    return state


def _scheduler_state(value):
    if value is None:
        return None
    state = _mapping(value, "scheduler state", required=True)
    if "scheduler_type" in state and "scheduler_state" in state:
        _mapping(state["scheduler_state"], "the wrapped scheduler state")
        if not isinstance(state["scheduler_type"], (str, type(None))):
            raise EnvelopeError("malformed envelope: scheduler_type must be a string")
    return state


def unwrap_scheduler_state(state):
    """``(inner state, scheduler type, wrapper name)`` of a stored or live scheduler state.

    ``training_management.v2`` ``TrainingScheduler`` stores ``{"scheduler_type", "scheduler_state"}``
    around the torch scheduler's own state; a plain torch scheduler stores that state directly.
    """
    if isinstance(state, dict) and "scheduler_type" in state and "scheduler_state" in state:
        return state["scheduler_state"] or {}, state["scheduler_type"], "TrainingScheduler"
    return state, None, None


def _int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def classify_envelope(checkpoint):
    """Envelope id of a loaded checkpoint object, by its key set."""
    if not isinstance(checkpoint, dict):
        raise EnvelopeError(f"checkpoint is a {type(checkpoint).__name__}, not a mapping")
    if checkpoint.get("format") == "cev11-checkpoint":
        return "cev11-v1"
    if checkpoint.get("format") == "cev11-ledger":
        return "cev11-ledger-v1"
    if "model_state" in checkpoint:
        return "tm3"
    if "model_state_dict" in checkpoint or "state_dict" in checkpoint:
        metadata = checkpoint.get("metadata")
        if isinstance(metadata, dict) and "checkpoint_format" in metadata:
            return "saver-4.1"
        if "state_dict" in checkpoint:
            return "mm-save"
        return "tm2"
    if _is_state(checkpoint):
        return "bare"
    raise EnvelopeError("checkpoint contains no recognised model state")


def view_of(checkpoint):
    """Facts of a loaded checkpoint object (no file access).

    Raises ``EnvelopeError`` for a field of the wrong structure (a list where a mapping belongs):
    a malformed envelope is refused as a whole, never half-read.
    """
    envelope = classify_envelope(checkpoint)
    view = EnvelopeView(envelope=envelope, top_keys=sorted(str(key) for key in checkpoint)[:64])
    if envelope == "bare":
        view.state = dict(checkpoint)
        view.notes.append("bare state dict: no architecture, no tokenizer identity")
        return view
    if envelope == "cev11-v1":
        cfg = _mapping(checkpoint.get("cfg"), "cfg")
        view.state = _state(checkpoint.get("state_dict"), "state_dict")
        view.model_config, view.config_source = (_mapping(cfg.get("config"), "cfg.config") or None), "cfg.config"
        tokenizer = _mapping(checkpoint.get("tokenizer"), "tokenizer")
        identity, cls = tokenizer.get("identity"), tokenizer.get("class")
        if _hex64(identity) and cls == "ByteLevelBPE":
            view.tokenizer = ("bytebpe-v1", identity)
        elif identity is not None:
            view.tokenizer_problem = (f"tokenizer class {cls!r} belongs to the study, not to the product"
                                      if _hex64(identity) else "the stored tokenizer identity is not a 64-hex digest")
            view.notes.append(view.tokenizer_problem)
        name = tokenizer.get("file")
        if isinstance(name, str) and name and Path(name).name == name:       # a file name, never a path
            view.tokenizer_file = name
        view.declared = {"weights:cev11-fp32-v1": checkpoint.get("model_sha256"), "arch_readout": cfg.get("arch"),
                         "build": cfg.get("build"), "tokenizer_class": cls,
                         "tokenizer_identity": identity, "version": checkpoint.get("version")}
        return view
    if envelope == "cev11-ledger-v1":
        view.declared = {"identity": checkpoint.get("identity"), "version": checkpoint.get("version")}
        view.notes.append("a memory ledger, not model weights")
        return view
    if envelope == "tm3":
        view.state = _state(checkpoint.get("model_state"), "model_state")
        view.optimizer_state = _optimizer_state(checkpoint.get("optimizer_state"))
        view.scheduler_state = _scheduler_state(checkpoint.get("scheduler_state"))
        train_config = _mapping(checkpoint.get("train_config"), "train_config")
        if train_config:
            view.model_config, view.config_source = train_config, "train_config"
        view.notes.append("training_management.v3 envelope: no tokenizer identity; its train_config counts as an "
                          "architecture only when it states every function-defining field and no unknown one")
        metrics = _mapping(checkpoint.get("metrics"), "metrics")
        view.epoch = _int(checkpoint.get("epoch", metrics.get("epoch")))
        return view
    info = _mapping(checkpoint.get("additional_info"), "additional_info")
    view.state = _state(checkpoint.get("model_state_dict", checkpoint.get("state_dict")), "model state")
    for source, value in (("model_config", checkpoint.get("model_config")), ("config", checkpoint.get("config")),
                          ("additional_info.config", info.get("config"))):
        if value:
            view.model_config, view.config_source = _mapping(value, source, required=True), source
            break
    extra = _mapping(checkpoint.get("extra_state"), "extra_state")
    identity = checkpoint.get("tokenizer_identity") or info.get("tokenizer_identity") or extra.get("tokenizer_identity")
    if identity is not None:
        if _hex64(identity):
            view.tokenizer = ("legacy-v3", identity)
        else:
            view.tokenizer_problem = "the stored tokenizer_identity is not a 64-hex digest"
            view.notes.append(view.tokenizer_problem)
    view.optimizer_state = _optimizer_state(checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer_state")))
    view.scheduler_state = _scheduler_state(checkpoint.get("scheduler_state_dict", checkpoint.get("scheduler_state")))
    if view.scheduler_state is None and extra.get("scheduler") is not None:
        view.scheduler_state = _scheduler_state(extra["scheduler"])
    view.extra_state = extra or None
    view.epoch = _int(checkpoint.get("epoch", info.get("epoch")))
    view.metric = _number(checkpoint.get("metric"))
    if envelope == "saver-4.1":
        metadata = checkpoint["metadata"]
        view.declared = {"cevahir_version": metadata.get("cevahir_version"),
                         "checkpoint_format": metadata.get("checkpoint_format")}
        if metadata.get("sha256"):
            view.declared["embedded_sha256"] = metadata["sha256"]
            view.notes.append("embedded metadata.sha256 is self-referential (known defect): ignored")
    return view


def read_envelope(path):
    """Load ``path`` with ``weights_only=True`` and return its ``EnvelopeView``.

    A file that needs code execution (a pickled module) is refused before torch is asked to read it.
    """
    import torch
    sniff = sniff_pickle_globals(path)
    if sniff["unsafe"]:
        raise EnvelopeError("refused: the file needs code execution to load (envelope 'module' or an "
                            f"arbitrary pickle); unsafe globals: {', '.join(sniff['unsafe'][:6])}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise EnvelopeError(f"cannot be read as a weights-only checkpoint: {type(error).__name__}: "
                            f"{str(error).splitlines()[0] if str(error) else ''}") from error
    return view_of(checkpoint)
