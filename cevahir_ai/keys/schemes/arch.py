"""Architecture keys.

``arch:cfg-v1`` is the typed projection of a *normalised* model config (the unchanged output of
``model_management.config_schema.normalize_model_config``) onto the 25 fields that define the
function or the weight layout: 24 ``ModelArchConfig`` fields and ``pe_max_len``, the length of the
positional table. Derived fields are resolved the way the constructor resolves them, and fields of
inactive features are nulled, so two configs that build the same function have one key and two
configs that differ in, say, ``num_heads`` never do, even when every weight has the same shape.

``pe_max_len`` counts only for ``pe_mode="learned"``, where it is the shape of a weight
(``pos_encoding.pe_embed.weight``). RoPE and sinusoidal tables are not weights; their context
length is a serving matter (``max_seq_length``, check C11).
"""
from cevahir_ai.keys.canonical import COERCE, structured_digest
from cevahir_ai.keys.classify import ARCH_FDF_V1, ARCH_SERVE_V1
from cevahir_ai.keys.grammar import Key

FAMILY = "cevahir-decoder"


def resolve_ffn_dim(embed_dim, ffn_dim=None, *, gated=True):
    """Copy of ``src.neural_network_module.architecture_contracts.resolve_ffn_dim`` (a test keeps
    the two equal), so the key stays torch-free and import-free."""
    if ffn_dim is not None:
        return ffn_dim
    if not gated:
        return 4 * embed_dim
    raw = int(2 / 3 * 4 * embed_dim)
    return ((raw + 255) // 256) * 256


def inactive_fields(fields):
    """Names of the fields that belong to a feature ``fields`` switches off.

    They are nulled in the key, so two configs that differ only there have one key; and a stored
    config need not state them to be a complete declaration.
    """
    inactive = set()
    if fields["pe_mode"] != "rope":
        inactive.add("rope_scaling_type")
    if fields["pe_mode"] != "rope" or fields["rope_scaling_type"] in (None, "none"):
        inactive.update(("rope_scaling_factor", "rope_original_max_len"))
    if not fields["use_moe"]:
        inactive.update(("num_experts", "moe_top_k"))
    if fields["pe_mode"] != "learned":
        inactive.add("pe_max_len")
    return inactive


def function_defining_fields(normalized):
    """FDF-v1 of a normalised config: resolved, conditionally nulled and typed."""
    cfg = dict(normalized)
    fields = {name: cfg.get(name) for name in ARCH_FDF_V1}
    fields["seq_proj_dim"] = fields["embed_dim"] if fields["seq_proj_dim"] is None else fields["seq_proj_dim"]
    fields["num_kv_heads"] = fields["num_kv_heads"] or fields["num_heads"]
    fields["ffn_dim"] = resolve_ffn_dim(fields["embed_dim"], fields["ffn_dim"], gated=bool(fields["use_swiglu"]))
    fields["tie_weights"] = bool(fields["tie_weights"]) and fields["seq_proj_dim"] == fields["embed_dim"]
    if fields["pe_max_len"] is None:
        fields["pe_max_len"] = cfg.get("max_seq_length")
    for name in inactive_fields(fields):
        fields[name] = None
    if fields["pe_mode"] == "learned" and fields["pe_max_len"] is None:
        raise ValueError("a learned positional table needs its length (pe_max_len or max_seq_length)")
    return {name: COERCE[kind](fields[name], name) for name, kind in ARCH_FDF_V1.items()}


def arch_cfg_key(normalized):
    """Returns ``(key, fields)`` for a normalised config (47 keys; see adapters.model_build.normalize)."""
    fields = function_defining_fields(normalized)
    payload = {"family": FAMILY, "family_version": str(normalized["architecture_version"]), "fields": fields}
    return Key("arch", "cfg-v1", structured_digest("arch", "cfg-v1", payload)), fields


def arch_ctor_v0_key(model_config):
    """Identity of a stored ``model_config`` (constructor arguments + tie_weights), minus ``device``."""
    payload = {key: value for key, value in dict(model_config).items() if key != "device"}
    return Key("arch", "ctor-v0", structured_digest("arch", "ctor-v0", {"config": payload}))


def serve_profile_key(normalized, *, dtype="float32"):
    """Serving-time fields. ``quantization_type`` lives here, not in the architecture key: it changes
    neither the layout nor the function of a freshly built model."""
    payload = {name: normalized.get(name) for name in ARCH_SERVE_V1}
    payload["precision"] = normalized.get("precision")
    payload["dtype"] = dtype
    return Key("serveprof", "cfg-v1", structured_digest("serveprof", "cfg-v1", payload)), payload


def diff_fields(left, right):
    """Field-by-field differences between two FDF dicts: ``{name: (left, right)}``."""
    return {name: (left.get(name), right.get(name)) for name in ARCH_FDF_V1 if left.get(name) != right.get(name)}
