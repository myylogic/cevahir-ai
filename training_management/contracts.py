"""Capabilities of the supported training path and portable precision selection."""
import torch


def validate_training_backend(config):
    backend = config.get("training_backend", "v2")
    if backend != "v2":
        raise ValueError("Only training_backend='v2' is integrated; the standalone v3 manager is experimental.")
    unsupported = [key for key in (
        "use_ema", "use_swa", "use_sam", "use_lookahead", "use_llrd",
        "use_curriculum", "use_scheduled_sampling", "use_focal_loss",
        "use_agc", "use_gradient_noise", "use_cosine_restarts",
    ) if config.get(key, False)]
    if unsupported:
        raise ValueError("Unsupported enabled training options: " + ", ".join(unsupported))
    if str(config.get("distributed_strategy", "none")).lower() != "none":
        raise ValueError("The training pipeline does not yet support distributed sampling/checkpointing.")


def resolve_precision(config, device):
    requested = str(config.get("precision", "auto")).lower()
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    if requested == "auto":
        requested = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if cuda and config.get("use_amp", False) else "fp32"
    # CPU has a reliable float32 fallback; explicit bf16 uses CPU autocast.
    if requested == "fp16" and not cuda:
        requested = "fp32"
    if requested == "bf16" and cuda and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requested on a CUDA device that does not support it")
    return requested


def consume_model_auxiliary_loss(model):
    """Consume the current forward's already-weighted MoE objective once."""
    visited = set()
    while id(model) not in visited:
        visited.add(id(model))
        getter = getattr(model, "get_and_reset_moe_loss", None)
        if getter is not None:
            return getter()
        wrapped = getattr(model, "module", None)
        if wrapped is None:
            wrapped = getattr(model, "_orig_mod", None)
        if wrapped is None:
            return None
        model = wrapped
    return None
