"""Tensor digests, alias groups and the per-tensor table of a state dict."""
import hashlib
import sys

import torch

from cevahir_ai.errors import UnsupportedTensor

_QUANTIZED = {torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4}


def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def check_supported(name, value):
    if not isinstance(value, torch.Tensor):
        raise UnsupportedTensor(f"state_dict entry {name!r} is not a tensor ({type(value).__name__})")
    if value.is_sparse or value.layout != torch.strided:
        raise UnsupportedTensor(f"state_dict entry {name!r} is sparse; no weight scheme covers it")
    if value.is_quantized or value.dtype in _QUANTIZED:
        raise UnsupportedTensor(f"state_dict entry {name!r} is quantized ({dtype_name(value.dtype)}); "
                                "no weight scheme covers quantized tensors")
    if value.device.type == "meta":
        raise UnsupportedTensor(f"state_dict entry {name!r} is a meta tensor (it has no data)")


def tensor_sha256(tensor):
    """sha256 of the raw little-endian bytes in logical (row-major) order.

    Independent of device, contiguity and memory layout; dependent on dtype (an fp16 copy of an
    fp32 tensor has another digest). A zero-element tensor gives ``sha256(b"")``.
    """
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    if flat.dtype == torch.bool:
        flat = flat.to(torch.uint8)
    array = flat.numpy()
    if sys.byteorder == "big" and array.dtype.itemsize > 1:
        array = array.byteswap()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _storage_id(tensor):
    if tensor.numel() == 0:
        return None
    return (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype)


def alias_groups(state):
    """Groups of names that share storage, offset, shape and stride (tied weights).

    Returns ``{member: canonical}`` where the canonical name is the lexicographically smallest.
    """
    by_storage = {}
    for name in sorted(state):
        identity = _storage_id(state[name])
        if identity is not None:
            by_storage.setdefault(identity, []).append(name)
    return {member: names[0] for names in by_storage.values() if len(names) > 1 for member in names[1:]}


def tensor_table(state):
    """Rows ``[name, dtype, shape, alias_of | None, sha256 | None]`` sorted by name.

    An alias member carries no digest of its own: the bytes are hashed once, under the canonical name.
    """
    for name, value in state.items():
        check_supported(name, value)
    aliases = alias_groups(state)
    rows = []
    for name in sorted(state):
        tensor = state[name]
        alias_of = aliases.get(name)
        rows.append([str(name), dtype_name(tensor.dtype), [int(size) for size in tensor.shape], alias_of,
                     None if alias_of else tensor_sha256(tensor)])
    return rows


def tensor_digests(table):
    """``{name: sha256}`` with alias members resolved to their canonical tensor's digest."""
    own = {row[0]: row[4] for row in table}
    return {row[0]: own[row[3]] if row[3] else row[4] for row in table}


def layout_of_table(table):
    """Layout (names, shapes, dtypes, alias groups) of a tensor table."""
    groups = {}
    for name, _, _, alias_of, _ in table:
        if alias_of:
            groups.setdefault(alias_of, [alias_of]).append(name)
    return {"tensors": [[row[0], row[2], row[1]] for row in table], "aliases": sorted(groups.values())}
