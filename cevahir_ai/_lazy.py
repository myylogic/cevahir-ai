"""Lazy re-exports for the facade packages (PEP 562).

``table`` maps a public name to ``"module:attribute"`` (or ``"module"`` for the module itself).
Nothing is imported until the name is first read. There is no fallback: a missing target raises
``ImportError`` or ``AttributeError`` at first use, never a silent substitute.
"""
import importlib
import sys


def resolve(target):
    """Resolve ``"module:attribute"`` through the legacy import seam."""
    from cevahir_ai._compat.repo import import_legacy
    module_name, _, attribute = target.partition(":")
    module = import_legacy(module_name)
    return getattr(module, attribute) if attribute else module


def lazy_exports(module_name, table):
    def __getattr__(name):
        if name not in table:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        value = resolve(table[name])
        setattr(sys.modules[module_name], name, value)
        return value

    def __dir__():
        return sorted(set(table) | {name for name in vars(sys.modules[module_name]) if not name.startswith("_")})

    return __getattr__, __dir__


def lazy_submodules(module_name, names):
    """PEP 562 access to sub-modules of the roof itself (no legacy import)."""
    def __getattr__(name):
        if name in names:
            return importlib.import_module(f"{module_name}.{name}")
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    return __getattr__
