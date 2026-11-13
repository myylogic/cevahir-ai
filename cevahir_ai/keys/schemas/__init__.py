"""The weight-schema registry: golden layouts of pinned architectures, and migrations.

A layout snapshot records what the constructor builds for a fixed configuration: every
``state_dict`` name with its shape and dtype, the alias groups (tied weights) and the buffers that
are not persisted. ``python -m cevahir_ai.keys.schemas --check`` rebuilds each preset on the meta
device and compares; a code change that renames a tensor or changes a pinned layout fails it. The
author then reverts, or bumps ``ARCHITECTURE_VERSION`` and adds a migration, and only then runs
``--write``.

Presets are fixed configurations owned by this registry (they do not follow later edits of
``TRAIN_CONFIG``): ``tiny`` is ``tiny_model_config()``, ``train-default`` is the model section of
``training_system/train.py: TRAIN_CONFIG`` as of this registry's creation (vocabulary size and
context length from the schema defaults), ``cev11-primary`` is the configuration of the CEV-11
base checkpoints.
"""
import json
from pathlib import Path

LAYOUT_FORMAT, MIGRATION_FORMAT = "cevahir-wschema-layout", "cevahir-wschema-migration"
_HERE = Path(__file__).resolve().parent
LAYOUT_DIR, MIGRATION_DIR = _HERE / "layouts", _HERE / "migrations"
ALLOWED_DTYPES = ("float32", "bfloat16", "float16")

PRESET_CONFIGS = {
    "tiny": None,     # tiny_model_config(), resolved at use
    "train-default": {
        "embed_dim": 512, "seq_proj_dim": 512, "num_heads": 8, "num_kv_heads": 2, "num_layers": 8,
        "ffn_dim": None, "pre_norm": True, "causal_mask": True, "pe_mode": "rope", "tie_weights": True,
        "use_rmsnorm": True, "use_swiglu": True, "use_moe": False, "num_experts": 8, "moe_top_k": 2,
        "sliding_window": 512, "rope_scaling_type": "yarn", "rope_scaling_factor": 2.0,
        "device": "cpu", "log_level": 50,
    },
    "cev11-primary": {
        "vocab_size": 4096, "embed_dim": 192, "num_heads": 4, "num_kv_heads": 4, "num_layers": 6,
        "ffn_dim": None, "max_seq_length": 64, "max_cache_len": 64, "dropout": 0.0,
        "use_gradient_checkpointing": False, "device": "cpu", "log_level": 50,
    },
}
PRESETS = tuple(PRESET_CONFIGS)


def preset_config(preset):
    """The raw configuration of a preset (pass it to ``adapters.model_build``)."""
    if preset not in PRESET_CONFIGS:
        raise KeyError(f"unknown preset: {preset!r}; known: {', '.join(PRESETS)}")
    if preset == "tiny":
        from cevahir_ai.adapters.model_build import config_schema
        return config_schema().tiny_model_config()
    return dict(PRESET_CONFIGS[preset])


def layout_path(preset):
    return LAYOUT_DIR / f"{preset}.json"


def load_layout(preset):
    return json.loads(layout_path(preset).read_text(encoding="utf-8"))


def generate_layout(preset):
    """Build the snapshot document of a preset from the current code (needs torch)."""
    from cevahir_ai.adapters.model_build import arch_of_config, expected_layout
    from cevahir_ai.keys.schemes.arch import FAMILY
    from cevahir_ai.keys.schemes.ckpt import wschema_layout_key, wschema_sd_key
    config = preset_config(preset)
    arch_key, fields, normalized = arch_of_config(config)
    layout = expected_layout(config)
    dtypes = sorted({entry[2] for entry in layout["tensors"]})
    return {
        "format": LAYOUT_FORMAT, "version": 1, "preset": preset,
        "family": FAMILY, "family_version": normalized["architecture_version"],
        "arch": str(arch_key), "arch_fields": fields,
        "layout": {"tensors": layout["tensors"], "aliases": layout["aliases"]},
        "non_persistent": layout["non_persistent"],
        "layout_key": str(wschema_layout_key(layout)), "sd_key": str(wschema_sd_key(layout)),
        "default_dtype": dtypes[0] if len(dtypes) == 1 else dtypes, "allowed_dtypes": list(ALLOWED_DTYPES),
    }


def dump(document):
    return json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def check(presets=PRESETS):
    """Compare every snapshot with the current code. Returns a list of findings (empty = green)."""
    findings = []
    for preset in presets:
        if not layout_path(preset).is_file():
            findings.append(f"{preset}: snapshot file is missing")
            continue
        recorded, current = load_layout(preset), generate_layout(preset)
        if recorded == current:
            continue
        have = {entry[0]: entry[1:] for entry in recorded["layout"]["tensors"]}
        want = {entry[0]: entry[1:] for entry in current["layout"]["tensors"]}
        detail = []
        if set(have) != set(want):
            detail.append(f"tensor names changed (removed {sorted(set(have) - set(want))[:4]}, "
                          f"added {sorted(set(want) - set(have))[:4]})")
        changed = [name for name in sorted(set(have) & set(want)) if have[name] != want[name]]
        if changed:
            detail.append(f"shape or dtype changed for {changed[:4]}")
        for name in ("arch", "family_version", "non_persistent", "layout_key", "sd_key"):
            if recorded.get(name) != current.get(name):
                detail.append(f"{name} changed")
        if recorded["layout"]["aliases"] != current["layout"]["aliases"]:
            detail.append("alias groups changed")
        findings.append(f"{preset}: the current code no longer builds the recorded layout: " + "; ".join(detail or ["document differs"]))
    return findings


def write(presets=PRESETS):
    from cevahir_ai._compat.fsio import atomic_write
    for preset in presets:
        atomic_write(layout_path(preset), dump(generate_layout(preset)).encode("utf-8"))


def preset_of_layout_key(key):
    """Name of the preset whose recorded layout key equals ``key`` (torch-free), or None."""
    for preset in PRESETS:
        path = layout_path(preset)
        if path.is_file() and load_layout(preset)["layout_key"] == str(key):
            return preset
    return None


def migration_ids():
    return sorted(json.loads(path.read_text(encoding="utf-8"))["id"] for path in MIGRATION_DIR.glob("*.json"))


def load_migration(migration_id):
    for path in sorted(MIGRATION_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["id"] == migration_id:
            if document.get("format") != MIGRATION_FORMAT:
                raise ValueError(f"{path.name} is not a {MIGRATION_FORMAT} document")
            return document
    raise KeyError(f"unknown migration: {migration_id!r}")
