"""Architecture keys.

``arch:cfg-v1`` is the typed projection of a *normalised* model config (the unchanged output of
``model_management.config_schema.normalize_model_config``) onto the 24 fields that define the
function or the weight layout. Derived fields are resolved the way the constructor resolves them,
and fields of inactive features are nulled, so two configs that build the same function have one
key and two configs that differ in, say, ``num_heads`` never do, even when every weight has the
same shape.
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


def function_defining_fields(normalized):
    """FDF-v1 of a normalised config: resolved, conditionally nulled and typed."""
    cfg = dict(normalized)
    fields = {name: cfg.get(name) for name in ARCH_FDF_V1}
    fields["seq_proj_dim"] = fields["embed_dim"] if fields["seq_proj_dim"] is None else fields["seq_proj_dim"]
    fields["num_kv_heads"] = fields["num_kv_heads"] or fields["num_heads"]
    fields["ffn_dim"] = resolve_ffn_dim(fields["embed_dim"], fields["ffn_dim"], gated=bool(fields["use_swiglu"]))
    fields["tie_weights"] = bool(fields["tie_weights"]) and fields["seq_proj_dim"] == fields["embed_dim"]
    if fields["pe_mode"] != "rope":
        fields["rope_scaling_type"] = None
    if fields["rope_scaling_type"] in (None, "none"):
        fields["rope_scaling_factor"] = fields["rope_original_max_len"] = None
    if not fields["use_moe"]:
        fields["num_experts"] = fields["moe_top_k"] = None
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
