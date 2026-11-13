"""Weight-schema, signature, checkpoint and model keys (structured, torch-free).

The tensor work (reading names, shapes, dtypes, alias groups and hashing bytes) is done by
``cevahir_ai.keys.torchkeys``; the functions here hash the resulting plain data.
"""
from cevahir_ai.keys.canonical import structured_digest
from cevahir_ai.keys.grammar import Key

# Public scheduler entries that every type has, or that PyTorch adds or removes between releases
# (the rule of model_management.checkpoint_contract._scheduler_signature; a test keeps them equal).
GENERIC_SCHEDULER_KEYS = frozenset({"last_epoch", "base_lrs", "verbose", "default_min_lr"})
CKPT_MEMBERS = ("weights", "arch", "wschema", "tok", "optim", "sched", "tstate")


def _layout_payload(layout, with_dtype):
    if with_dtype:
        tensors = [[name, list(shape), dtype] for name, shape, dtype in layout["tensors"]]
    else:
        tensors = [[entry[0], list(entry[1])] for entry in layout["tensors"]]
    return {"tensors": sorted(tensors), "aliases": sorted(sorted(group) for group in layout["aliases"])}


def wschema_layout_key(layout):
    """``layout``: {"tensors": [[name, shape] or [name, shape, dtype], ...], "aliases": [[names], ...]}."""
    return Key("wschema", "layout-v1", structured_digest("wschema", "layout-v1", _layout_payload(layout, False)))


def wschema_sd_key(layout):
    return Key("wschema", "sd-v1", structured_digest("wschema", "sd-v1", _layout_payload(layout, True)))


def optimizer_signature(state):
    """Structure of an optimizer ``state_dict``: values of hyperparameters may differ, names may not."""
    groups = [{"n_params": len(group.get("params", ())), "hyper": sorted(str(k) for k in group if k != "params")}
              for group in state.get("param_groups", ())]
    state_keys = sorted({str(key) for entry in (state.get("state") or {}).values()
                         if isinstance(entry, dict) for key in entry})
    return {"groups": groups, "state_keys": state_keys}


def optim_sig_key(state, optimizer_type=None):
    payload = {"type": optimizer_type, **optimizer_signature(state)}
    return Key("optim", "sig-v1", structured_digest("optim", "sig-v1", payload)), payload


def scheduler_signature(state):
    """Type-specific public keys of a scheduler state (private and generic entries ignored)."""
    return sorted(str(key) for key in state if not str(key).startswith("_") and key not in GENERIC_SCHEDULER_KEYS)


def sched_sig_key(state, scheduler_type=None, wrapper=None):
    payload = {"wrapper": wrapper, "type": scheduler_type, "type_keys": scheduler_signature(state)}
    return Key("sched", "sig-v1", structured_digest("sched", "sig-v1", payload)), payload


def ckpt_members_key(members):
    """``members``: {"weights": key, "arch": key|None, "wschema": key, "tok": key|None, "optim", "sched", "tstate"}."""
    unknown = set(members) - set(CKPT_MEMBERS)
    if unknown:
        raise ValueError(f"unknown checkpoint members: {sorted(unknown)}")
    if not members.get("weights") or not members.get("wschema"):
        raise ValueError("a checkpoint key needs a weights key and a weight-schema key")
    payload = {name: (None if members.get(name) is None else str(Key.parse(str(members[name]))))
               for name in CKPT_MEMBERS}
    return Key("ckpt", "members-v1", structured_digest("ckpt", "members-v1", payload))


def model_triple_key(weights, arch, tok):
    """The servable identity. It needs a known architecture and a canonical tokenizer key."""
    weights, arch, tok = (Key.parse(str(key)) for key in (weights, arch, tok))
    if (weights.kind, weights.scheme) != ("weights", "sha256-v1") or (arch.kind, arch.scheme) != ("arch", "cfg-v1"):
        raise ValueError("a model key needs weights:sha256-v1 and arch:cfg-v1")
    if tok.kind != "tok" or tok.scheme not in ("bpe-sem-v1", "bytebpe-v1"):
        raise ValueError("a model key needs a canonical tokenizer key (bpe-sem-v1 or bytebpe-v1)")
    payload = {"weights": str(weights), "arch": str(arch), "tok": str(tok)}
    return Key("model", "triple-v1", structured_digest("model", "triple-v1", payload))
