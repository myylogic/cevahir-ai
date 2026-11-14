"""Cevahir AI roof package.

One installable top-level name over today's packages (``src``, ``model_management``,
``tokenizer_management``, ``training_system``, ``training_management``, ``cognitive_management``,
``model``, ``api`` ...). It holds the new subsystems (``keys``: the central key system and the
compatibility gate; ``store``: registry, audit log and lineage under ``CEVAHIR_HOME``; ``ops``: one
function per user-level action; ``evidence``: the CEV-11 closure pin) and unversioned lazy facades
(``neural``, ``model``, ``tokenizer``, ``data``, ``training``, ``cognition``, ``serving``).

Importing this package has no side effects: no logging configuration, no torch, no legacy package.
Old import paths keep working because they still are the implementations; nothing is moved.
"""
import importlib

__version__ = "0.3.0"

_SUBPACKAGES = ("keys", "domain", "adapters", "evidence", "errors", "store", "ops",
                "neural", "model", "tokenizer", "data", "training", "cognition", "serving")

__all__ = ["__version__", *_SUBPACKAGES]


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
