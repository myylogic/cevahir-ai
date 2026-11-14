"""The Cevahir console: a strict subset of PowerShell, implemented in Python.

Verb-Noun cmdlets, an object pipeline, ``$variables``, ``-Parameter value`` binding, ``.ps1``
scripts, an interactive prompt with history and Tab completion, and help in Turkish and English.
It runs on Windows, Linux and Colab with the standard library only, and starts without torch.

    parser      text -> syntax tree; rejects everything outside the subset (``NotInCevahirSubset``)
    expr        the expression language of script blocks: data in, data out, no code execution
    binder      parameters: named, positional, switches, pipeline input
    runtime     Session: variables, streams, pipelines of generators, statements, exit codes
    registry    the cmdlet declarations (``commands/``) that drive binding, help and completion
    format      tables, lists and the default view of each record type
    help        Get-Help text generated from the registry, tr and en
    completion  Tab completion as a pure function
    repl, main  the prompt and the ``cevahir`` entry point

Importing this package loads no torch and no legacy package.
"""
import importlib

_EXPORTS = {"Session": "runtime", "parse": "parser", "ParserError": "errors", "complete": "completion", "main": "main"}
_SUBMODULES = ("syntax", "parser", "expr", "binder", "runtime", "registry", "format", "help", "completion", "repl",
               "main", "errors", "lineedit_win", "commands", "tools")

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_SUBMODULES))
