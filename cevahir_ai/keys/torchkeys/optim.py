"""Content keys of optimizer, scheduler and remaining training state.

The state is reduced to canonical data: tensors become ``{"tensor": [dtype, shape, sha256]}``,
numpy arrays likewise, tuples become lists and mapping keys become strings. Two resume points
with equal weights therefore still have different checkpoint keys.
"""
import hashlib

import torch

from cevahir_ai.keys.canonical import structured_digest
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.torchkeys.tensors import dtype_name, tensor_sha256


def canonical_state(value, depth=0):
    if depth > 60:
        raise ValueError("training state nested deeper than 60")
    if isinstance(value, torch.Tensor):
        return {"tensor": [dtype_name(value.dtype), [int(size) for size in value.shape], tensor_sha256(value)]}
    if value is None or isinstance(value, (bool, int, str)):
        return value if not isinstance(value, int) or isinstance(value, bool) or -2**63 <= value < 2**63 \
            else {"bigint": str(value)}
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else {"float": repr(value)}
    if isinstance(value, dict):
        return {str(key): canonical_state(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_state(item, depth + 1) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return {"bytes": hashlib.sha256(bytes(value)).hexdigest()}
    if hasattr(value, "tobytes") and hasattr(value, "dtype") and hasattr(value, "shape"):   # numpy array / scalar
        return {"array": [str(value.dtype), [int(size) for size in value.shape],
                          hashlib.sha256(value.tobytes()).hexdigest()]}
    return {"repr": f"{type(value).__name__}:{value!r}"}


def _content_key(kind, state):
    return Key(kind, "sha256-v1", structured_digest(kind, "sha256-v1", {"state": canonical_state(state)}))


def optim_content_key(state):
    return _content_key("optim", state)


def sched_content_key(state):
    return _content_key("sched", state)


def tstate_content_key(state):
    return _content_key("tstate", state)
