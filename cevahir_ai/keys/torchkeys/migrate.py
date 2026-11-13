"""Weight-schema migrations: declarative, pure transforms of a state dict.

A migration is a JSON document (``cevahir_ai/keys/schemas/migrations/*.json``) with a list of
operations. Applying one returns a new dict; the input is never modified and no file is written.

Operations:
  strip_prefix  {"prefix": p}                    drop ``p`` from every name that starts with it
  rename        {"from": a, "to": b}             rename one tensor
  rename_regex  {"pattern": re, "repl": s}       anchored regular-expression rename
  drop          {"name": n, "reason": r}         remove a derived or non-persistent tensor
  tie           {"canonical": a, "member": b}    assert equality, then make ``b`` share ``a``
  untie         {"name": n}                      give ``n`` its own storage
  cast          {"dtype": "float32"}             up-cast floating tensors (never down)
"""
import re

import torch

from cevahir_ai.errors import MigrationError

_FLOAT_RANK = {torch.float16: 1, torch.bfloat16: 1, torch.float32: 2, torch.float64: 3}


def _rename(state, mapping):
    result = {}
    for name, tensor in state.items():
        target = mapping.get(name, name)
        if target in result:
            raise MigrationError(f"migration maps two tensors to {target!r}")
        result[target] = tensor
    return result


def apply_migration(state, migration):
    """Apply ``migration`` (a parsed document) to ``state``. Returns ``(new_state, applied_ops)``."""
    if migration.get("format") != "cevahir-wschema-migration" or migration.get("version") != 1:
        raise MigrationError("not a cevahir-wschema-migration v1 document")
    state, applied = dict(state), []
    for op in migration["ops"]:
        kind = op.get("op")
        if kind == "strip_prefix":
            prefix = op["prefix"]
            mapping = {name: name[len(prefix):] for name in state if name.startswith(prefix)}
            if mapping:
                state = _rename(state, mapping)
                applied.append({"op": kind, "prefix": prefix, "renamed": len(mapping)})
        elif kind == "rename":
            if op["from"] in state:
                state = _rename(state, {op["from"]: op["to"]})
                applied.append({"op": kind, "from": op["from"], "to": op["to"]})
        elif kind == "rename_regex":
            pattern = re.compile(op["pattern"])
            mapping = {name: pattern.sub(op["repl"], name) for name in state if pattern.fullmatch(name)}
            mapping = {old: new for old, new in mapping.items() if old != new}
            if mapping:
                state = _rename(state, mapping)
                applied.append({"op": kind, "pattern": op["pattern"], "renamed": len(mapping)})
        elif kind == "drop":
            if not op.get("reason"):
                raise MigrationError("drop needs a reason")
            if op["name"] in state:
                del state[op["name"]]
                applied.append({"op": kind, "name": op["name"]})
        elif kind == "tie":
            canonical, member = op["canonical"], op["member"]
            if canonical in state and member in state:
                if state[canonical].shape != state[member].shape or not torch.equal(state[canonical], state[member]):
                    raise MigrationError(f"cannot tie {member} to {canonical}: the tensors differ")
                state[member] = state[canonical]
                applied.append({"op": kind, "canonical": canonical, "member": member})
        elif kind == "untie":
            if op["name"] in state:
                state[op["name"]] = state[op["name"]].clone()
                applied.append({"op": kind, "name": op["name"]})
        elif kind == "cast":
            target = getattr(torch, op["dtype"])
            changed, shared = 0, {}
            for name, tensor in list(state.items()):
                if tensor.dtype in _FLOAT_RANK and tensor.dtype != target:
                    if _FLOAT_RANK[tensor.dtype] > _FLOAT_RANK.get(target, 0):
                        raise MigrationError(f"cast would lose precision for {name}: {tensor.dtype} -> {target}")
                    # Tied tensors stay tied: one cast per shared storage.
                    identity = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                                tuple(tensor.shape), tuple(tensor.stride())) if tensor.numel() else id(tensor)
                    if identity not in shared:
                        shared[identity] = tensor.to(target)
                    state[name] = shared[identity]
                    changed += 1
            if changed:
                applied.append({"op": kind, "dtype": op["dtype"], "cast": changed})
        else:
            raise MigrationError(f"unknown migration operation: {kind!r}")
    return state, applied
