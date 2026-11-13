"""Weight-schema layouts: the names, shapes, dtypes and alias groups of a ``state_dict``.

The expected layout of an architecture is read from the constructor: the model is built under
``torch.device("meta")``, which allocates no memory (the 100M-parameter default model builds in
milliseconds), so no hand-written table can drift from the code.
"""
import torch

from cevahir_ai.keys.schemes.ckpt import wschema_layout_key, wschema_sd_key
from cevahir_ai.keys.torchkeys.tensors import dtype_name, layout_of_table, tensor_table


def layout_of_module(model):
    """Layout of a live (or meta) module. Alias groups are parameters that are the same object."""
    state = model.state_dict()
    owners = {}
    for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if name in state:
            owners.setdefault(id(tensor), []).append(name)
    tensors = sorted([name, [int(size) for size in tensor.shape], dtype_name(tensor.dtype)]
                     for name, tensor in state.items())
    aliases = sorted(sorted(names) for names in owners.values() if len(names) > 1)
    non_persistent = sorted(name for name, _ in model.named_buffers() if name not in state)
    return {"tensors": tensors, "aliases": aliases, "non_persistent": non_persistent}


def layout_of_state(state):
    """Layout of a state dict read from a file (alias groups from shared storage)."""
    return layout_of_table(tensor_table(state))


def layout_keys(layout):
    return wschema_layout_key(layout), wschema_sd_key(layout)


def compare_layouts(instance, expected):
    """Differences that make ``instance`` unloadable into ``expected`` without a partial load.

    Returns a list of human-readable findings; empty means the layouts match (names, shapes and
    alias groups; dtypes are the dtype check's business).
    """
    have = {entry[0]: list(entry[1]) for entry in instance["tensors"]}
    want = {entry[0]: list(entry[1]) for entry in expected["tensors"]}
    findings = []
    missing, unexpected = sorted(want.keys() - have.keys()), sorted(have.keys() - want.keys())
    if missing:
        findings.append(f"missing tensors: {_short(missing)}")
    if unexpected:
        findings.append(f"unexpected tensors: {_short(unexpected)}")
    for name in sorted(have.keys() & want.keys()):
        if have[name] != want[name]:
            findings.append(f"shape of {name}: {have[name]} in the file, {want[name]} expected")
            if len(findings) > 8:
                findings.append("...")
                break
    return findings


def broken_ties(state, expected):
    """Alias groups of the expected layout whose tensors in ``state`` are neither shared nor equal."""
    broken = []
    for group in expected.get("aliases", ()):
        present = [name for name in group if name in state]
        for name in present[1:]:
            if state[name].shape != state[present[0]].shape or not torch.equal(state[name], state[present[0]]):
                broken.append([present[0], name])
    return broken


def _short(names, limit=6):
    return ", ".join(names[:limit]) + (f" (+{len(names) - limit} more)" if len(names) > limit else "")
