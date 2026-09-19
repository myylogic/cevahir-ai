"""Shared checkpoint envelope and model-identity rules for save, load and resume."""
from collections.abc import Mapping
from copy import deepcopy
import torch


def unpack_checkpoint(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping")
    info = checkpoint.get("additional_info") or {}
    if not isinstance(info, Mapping):
        raise ValueError("Checkpoint additional_info must be a mapping")
    if "model_state_dict" in checkpoint or "state_dict" in checkpoint:
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint model state must be a mapping")
        return state, checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer_state")), checkpoint.get("scheduler_state_dict", checkpoint.get("scheduler_state")), {
            "config": checkpoint.get("model_config") or checkpoint.get("config") or info.get("config") or {},
            "epoch": checkpoint.get("epoch", info.get("epoch")),
            "tokenizer_identity": checkpoint.get("tokenizer_identity") or info.get("tokenizer_identity") or (checkpoint.get("extra_state") or {}).get("tokenizer_identity"),
        }
    if checkpoint and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
        return checkpoint, None, None, {"config": {}, "epoch": None}
    raise ValueError("Checkpoint contains no recognized model state")


def model_config(model):
    """Capture construction truth, independently of mutable manager/training options."""
    config = deepcopy(getattr(model, "_ctor_cfg", {}))
    if config:
        config["tie_weights"] = bool(model.tie_weights)
        config["device"] = "cpu"
    return config


def validate_model_identity(model, saved_config):
    current = model_config(model)
    if not current or not saved_config:
        return
    from .config_schema import normalize_model_config
    saved = normalize_model_config(saved_config, legacy_profile="model_manager")
    live = normalize_model_config(current, legacy_profile="model_manager")
    for cfg in (saved, live):
        cfg["num_kv_heads"] = cfg.get("num_kv_heads") or cfg["num_heads"]
        cfg["tie_weights"] = bool(cfg.get("tie_weights")) and cfg.get("seq_proj_dim") in (None, cfg["embed_dim"])
    # These can change computation without changing any state_dict shape.
    fields = ("embed_dim", "num_heads", "num_kv_heads", "num_layers", "vocab_size",
        "use_swiglu", "use_moe", "num_experts", "moe_top_k", "pre_norm", "use_rmsnorm",
        "parallel_residual", "pe_mode", "rope_theta", "rope_scaling_type", "rope_scaling_factor",
        "tie_weights", "use_qk_norm", "rope_original_max_len", "causal_mask", "attn_logit_cap", "logit_soft_cap", "sliding_window")
    differences = [f"{k}: checkpoint={saved.get(k)!r}, model={live.get(k)!r}" for k in fields if saved.get(k) != live.get(k)]
    if differences:
        raise ValueError("Checkpoint/model architecture mismatch: " + "; ".join(differences))


def validate_state_shapes(model, state, strict=True):
    """Reject before load_state_dict can partially overwrite live parameters."""
    live = model.state_dict()
    missing, extra = live.keys() - state.keys(), state.keys() - live.keys()
    if strict and (missing or extra):
        raise ValueError(f"Checkpoint state keys mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for key in live.keys() & state.keys():
        if isinstance(live[key], torch.Tensor) and isinstance(state[key], torch.Tensor) and live[key].shape != state[key].shape:
            raise ValueError(f"Checkpoint shape mismatch for {key}: {tuple(state[key].shape)} != {tuple(live[key].shape)}")


def validate_tokenizer_identity(saved_identity, current_identity):
    """A fingerprinted checkpoint must not be used with unverified token semantics."""
    if saved_identity is None:
        if current_identity is not None:
            import warnings
            warnings.warn("Legacy checkpoint has no tokenizer identity; token semantics cannot be verified", RuntimeWarning, stacklevel=2)
        return
    if not isinstance(saved_identity, str) or len(saved_identity) != 64:
        raise ValueError("Invalid checkpoint tokenizer identity")
    if current_identity is None:
        raise ValueError("Checkpoint requires a tokenizer to verify token identity")
    if saved_identity != current_identity:
        raise ValueError("Checkpoint/tokenizer identity mismatch; vocabulary size alone is insufficient")


def tokenizer_identity(tokenizer):
    if tokenizer is None:
        return None
    from training_system.cache_identity import tokenizer_digest
    return tokenizer_digest(tokenizer)
