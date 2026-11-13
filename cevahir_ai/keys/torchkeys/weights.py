"""Weight keys.

``weights:sha256-v1`` hashes the per-tensor table, so the key is independent of the container
(a training envelope, a saver envelope or a bare state dict with the same tensors have one key),
dtype is identity (an fp16 copy is another key) and a diff can name the tensors that changed.

``weights:cev11-fp32-v1`` is a byte-exact re-implementation of ``cev11.model.model_sha256``, the
weight digest stored in every CEV-11 checkpoint. It casts to fp32, so it erases dtype; it is kept
as a legacy alias only.
"""
import hashlib

import torch

from cevahir_ai.errors import UnsupportedTensor
from cevahir_ai.keys.canonical import structured_digest
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.torchkeys.tensors import check_supported, tensor_table


def weights_key(state):
    """Returns ``(key, table)`` for a state dict (or a module's ``state_dict()``)."""
    table = tensor_table(state)
    return Key("weights", "sha256-v1", structured_digest("weights", "sha256-v1", {"tensors": table})), table


def weights_cev11_fp32_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        check_supported(name, state[name])
        tensor = state[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(name.encode("utf-8") + b"\0" + repr(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def weights_cev11_fp32_key(state):
    return Key("weights", "cev11-fp32-v1", weights_cev11_fp32_digest(state))


def dtype_summary(table):
    """Floating dtypes present, most frequent first, and the parameter count (aliases counted once)."""
    counts, n_params = {}, 0
    for _, dtype, shape, alias_of, _ in table:
        if alias_of:
            continue
        size = 1
        for dim in shape:
            size *= dim
        n_params += size
        if dtype.startswith(("float", "bfloat", "complex")):
            counts[dtype] = counts.get(dtype, 0) + size
    return sorted(counts, key=lambda name: (-counts[name], name)), n_params


def diff_tables(left, right):
    """Per-tensor differences between two tables: added, removed and changed names."""
    a = {row[0]: row for row in left}
    b = {row[0]: row for row in right}
    resolve_a = {row[0]: (a[row[3]][4] if row[3] else row[4]) for row in left}
    resolve_b = {row[0]: (b[row[3]][4] if row[3] else row[4]) for row in right}
    changed = []
    for name in sorted(a.keys() & b.keys()):
        reasons = []
        if a[name][1] != b[name][1]:
            reasons.append(f"dtype {a[name][1]} -> {b[name][1]}")
        if a[name][2] != b[name][2]:
            reasons.append(f"shape {a[name][2]} -> {b[name][2]}")
        if not reasons and resolve_a[name] != resolve_b[name]:
            reasons.append("values")
        if a[name][3] != b[name][3]:
            reasons.append(f"alias {a[name][3]} -> {b[name][3]}")
        if reasons:
            changed.append([name, ", ".join(reasons)])
    return {"only_left": sorted(a.keys() - b.keys()), "only_right": sorted(b.keys() - a.keys()), "changed": changed}


__all__ = ["weights_key", "weights_cev11_fp32_key", "weights_cev11_fp32_digest", "dtype_summary",
           "diff_tables", "UnsupportedTensor"]
