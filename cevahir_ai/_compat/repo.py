"""Repository root discovery and the single seam through which the roof imports legacy packages."""
import importlib
import sys
from pathlib import Path

LEGACY_PACKAGES = ("src", "model", "model_management", "tokenizer_management", "training_system",
                   "training_management", "data_loader_management", "cognitive_management",
                   "chatting_management", "api", "database")


def repo_root():
    """The checkout that contains this package (the directory holding ``pyproject.toml``)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"cevahir_ai is not running from a Cevahir AI checkout: {root}")
    return root


def import_legacy(name):
    """Import a legacy module by its existing path; the checkout root joins ``sys.path`` once.

    The roof never copies or wraps the module: the returned object is the legacy module itself,
    so ``cevahir_ai.model.ModelManager is model_management.model_manager.ModelManager``.
    """
    if name.split(".")[0] in LEGACY_PACKAGES:
        root = str(repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
    return importlib.import_module(name)
