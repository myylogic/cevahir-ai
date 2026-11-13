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
_SAFE_PREFIXES = ("torch._utils._rebuild_",)
_TORCH_DTYPES = {"float16", "float32", "float64", "bfloat16", "half", "float", "double", "int8", "int16",
                 "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "long", "int", "short",
                 "complex64", "complex128", "strided"}
ENVELOPES = ("tm2", "mm-save", "saver-4.1", "bare", "module", "tm3", "cev11-v1", "cev11-ledger-v1")


def is_safe_global(name):
    if name in _SAFE_EXACT or name.startswith(_SAFE_PREFIXES):
        return True
    module, _, attribute = name.rpartition(".")
    if module == "torch" and (attribute.endswith("Storage") or attribute in _TORCH_DTYPES):
        return True
    return False


def _pickle_globals(data):
    """Names of the globals a pickle stream imports (GLOBAL, STACK_GLOBAL), without executing it."""
    found, memo, recent, last_string, memo_index = [], {}, [], None, 0
    for opcode, arg, _ in pickletools.genops(data):
        name = opcode.name
        pushed = None
        if name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"):
            pushed = arg
            recent.append(arg)
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            value = memo.get(int(arg))
            if isinstance(value, str):
                pushed = value
                recent.append(value)
        elif name == "MEMOIZE":
            memo[memo_index] = last_string
            memo_index += 1
            pushed = last_string
        elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[int(arg)] = last_string
            pushed = last_string
        elif name == "GLOBAL":
            module, _, attribute = arg.partition(" ")
            found.append(f"{module}.{attribute}")
        elif name == "STACK_GLOBAL":
            if len(recent) < 2:
                found.append("<unresolved>")
            else:
                found.append(f"{recent[-2]}.{recent[-1]}")
        last_string = pushed
        del recent[:-8]
    return found


def sniff_pickle_globals(path):
    """Torch-free safety sniff. Returns ``{"container", "globals", "unsafe"}``.

    ``container`` is ``"torch-zip"`` for the zip format ``torch.save`` writes, else ``"unknown"``.
    ``unsafe`` lists the globals outside the ``weights_only`` safe set: a non-empty list means the
    file needs code execution to load (a pickled module, or an arbitrary object).
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
    """Facts of a loaded checkpoint object (no file access)."""
    envelope = classify_envelope(checkpoint)
    view = EnvelopeView(envelope=envelope, top_keys=sorted(str(key) for key in checkpoint)[:64])
    if envelope == "bare":
        view.state = dict(checkpoint)
        view.notes.append("bare state dict: no architecture, no tokenizer identity")
        return view
    if envelope == "cev11-v1":
        cfg = checkpoint.get("cfg") or {}
        view.state = dict(checkpoint["state_dict"])
        view.model_config, view.config_source = cfg.get("config"), "cfg.config"
        tokenizer = checkpoint.get("tokenizer") or {}
        if _hex64(tokenizer.get("identity")):
            scheme = "bytebpe-v1" if tokenizer.get("class") == "ByteLevelBPE" else None
            view.tokenizer = (scheme, tokenizer["identity"]) if scheme else None
            if not scheme:
                view.notes.append(f"tokenizer class {tokenizer.get('class')!r} belongs to the study, not to the product")
        view.tokenizer_file = tokenizer.get("file")
        view.declared = {"weights:cev11-fp32-v1": checkpoint.get("model_sha256"), "arch_readout": cfg.get("arch"),
                         "build": cfg.get("build"), "tokenizer_class": tokenizer.get("class"),
                         "tokenizer_identity": tokenizer.get("identity"), "version": checkpoint.get("version")}
        return view
    if envelope == "cev11-ledger-v1":
        view.declared = {"identity": checkpoint.get("identity"), "version": checkpoint.get("version")}
        view.notes.append("a memory ledger, not model weights")
        return view
    if envelope == "tm3":
        view.state = dict(checkpoint["model_state"])
        view.optimizer_state, view.scheduler_state = checkpoint.get("optimizer_state"), checkpoint.get("scheduler_state")
        train_config = checkpoint.get("train_config")
        if isinstance(train_config, dict) and train_config:
            view.model_config, view.config_source = train_config, "train_config"
        view.notes.append("training_management.v3 envelope: no tokenizer identity, architecture is best effort")
        metrics = checkpoint.get("metrics")
        view.epoch = checkpoint.get("epoch", metrics.get("epoch") if isinstance(metrics, dict) else None)
        return view
    info = checkpoint.get("additional_info") or {}
    if not isinstance(info, dict):
        raise EnvelopeError("checkpoint additional_info must be a mapping")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise EnvelopeError("checkpoint model state must be a mapping")
    view.state = dict(state)
    for source, value in (("model_config", checkpoint.get("model_config")), ("config", checkpoint.get("config")),
                          ("additional_info.config", info.get("config"))):
        if value:
            view.model_config, view.config_source = value, source
            break
    extra = checkpoint.get("extra_state") or {}
    identity = checkpoint.get("tokenizer_identity") or info.get("tokenizer_identity") or extra.get("tokenizer_identity")
    if identity is not None:
        if _hex64(identity):
            view.tokenizer = ("legacy-v3", identity)
        else:
            view.notes.append("tokenizer_identity is not a 64-hex digest")
    view.optimizer_state = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer_state"))
    view.scheduler_state = checkpoint.get("scheduler_state_dict", checkpoint.get("scheduler_state"))
    if view.scheduler_state is None and isinstance(extra.get("scheduler"), dict):
        view.scheduler_state = extra["scheduler"]
    view.extra_state = extra or None
    view.epoch = checkpoint.get("epoch", info.get("epoch"))
    view.metric = checkpoint.get("metric")
    metadata = checkpoint.get("metadata")
    if envelope == "saver-4.1":
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
