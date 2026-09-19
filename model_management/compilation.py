"""Optional compiled forward with observable fallback and unchanged state keys."""
from functools import wraps
import logging

import torch


def configure_compilation(model, config):
    """Compile the existing forward callable; preserve model/checkpoint identity.

    Only compiler failures trigger fallback. Model input errors are propagated.
    Eager is the default, and this helper does not promise a speed improvement.
    """
    status = {"requested": bool(config.get("torch_compile", False)), "active": False, "fallback_reason": None}
    model.compilation_status = status
    if not status["requested"]:
        return model
    if not hasattr(torch, "compile"):
        status["fallback_reason"] = "torch.compile is unavailable"
        return model
    original = model.forward
    options = {
        "mode": config.get("torch_compile_mode", "default"),
        "dynamic": bool(config.get("torch_compile_dynamic", False)),
        "fullgraph": bool(config.get("torch_compile_fullgraph", False)),
    }
    try:
        compiled = torch.compile(original, **options)
    except Exception as exc:
        status["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        logging.getLogger(__name__).warning("Compile initialization failed: %s", exc)
        return model
    from torch._dynamo.exc import BackendCompilerFailed, Unsupported
    status["active"] = True

    @wraps(original)
    def forward(*args, **kwargs):
        try:
            return compiled(*args, **kwargs)
        except (BackendCompilerFailed, Unsupported) as exc:
            status["active"] = False
            status["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            model.forward = original
            logging.getLogger(__name__).warning("Compiler failed; subsequent calls use eager: %s", exc)
            return original(*args, **kwargs)

    model.forward = forward
    return model
