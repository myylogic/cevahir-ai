"""The command registry: every cmdlet is declared once, here, by the modules in ``commands/``.

The declaration drives everything else: parameter binding, ``Get-Command``, ``Get-Help`` (the
SYNTAX section is generated from it), tab completion and the generated PowerShell module.

A cmdlet is a generator function. ``kind`` says how it meets the pipeline:

  "process"  called once per input object (or once when there is no input); parameters marked
             ``pipeline`` are bound from each object, by value or by property name
  "stream"   called once with the whole upstream iterator (filters, sorters, formatters)

Help text is bilingual: ``("English", "Türkçe")``.
"""
import importlib
from dataclasses import dataclass, field

COMMAND_MODULES = ("core", "output", "artifacts", "checkpoints", "audit", "session")
TYPES = ("string", "int", "float", "bool", "switch", "list", "any", "objects", "scriptblock", "path", "hashtable")
ACTION_CLASSES = ("read", "mutate", "destructive", "publish", "governance", "acknowledge")


@dataclass
class Param:
    name: str
    type: str = "string"
    position: int = None
    mandatory: bool = False
    pipeline: str = None             # None | "value" | "property" | "both"
    aliases: tuple = ()
    choices: tuple = None
    default: object = None
    help: tuple = ("", "")
    remaining: bool = False          # takes every remaining positional argument
    raw: bool = False                # values are passed as written (a number keeps its spelling: 0193)
    secret: bool = False             # never written to the audit log

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"unknown parameter type: {self.type}")


@dataclass
class Cmdlet:
    name: str
    function: object
    params: list
    kind: str = "process"
    output: str = None
    action_class: str = "read"
    module: bool = False             # a Cevahir cmdlet: also callable with the Cv prefix (Get-CvArtifact)
    aliases: tuple = ()
    synopsis: tuple = ("", "")
    description: tuple = ("", "")
    examples: list = field(default_factory=list)      # [(code, English, Türkçe)]
    whatif: bool = False             # supports -WhatIf
    torch: str = ""                  # when it needs torch, in words ("" = never)

    @property
    def verb(self):
        return self.name.partition("-")[0]

    @property
    def noun(self):
        return self.name.partition("-")[2]

    def param(self, name):
        lowered = name.lower()
        for parameter in self.params:
            if parameter.name.lower() == lowered or lowered in (alias.lower() for alias in parameter.aliases):
                return parameter
        return None


_COMMANDS = {}
_ALIASES = {}
_loaded = False


def cmdlet(name, params=(), **options):
    """Decorator that registers a cmdlet function."""
    def register(function):
        command = Cmdlet(name, function, list(params), **options)
        if command.kind not in ("process", "stream") or command.action_class not in ACTION_CLASSES:
            raise ValueError(f"bad declaration of {name}")
        _COMMANDS[name.lower()] = command
        for alias in command.aliases:
            _ALIASES[alias.lower()] = command
        return function
    return register


def load():
    global _loaded
    if not _loaded:
        _loaded = True
        for module in COMMAND_MODULES:
            importlib.import_module(f"cevahir_ai.console.commands.{module}")


def commands():
    load()
    return sorted(_COMMANDS.values(), key=lambda command: (command.noun.lower(), command.verb.lower()))


def aliases():
    load()
    return dict(_ALIASES)


def find(name):
    """The cmdlet called ``name``: its name, an alias, or the ``Cv``-prefixed form of a Cevahir cmdlet."""
    load()
    lowered = name.lower()
    command = _COMMANDS.get(lowered) or _ALIASES.get(lowered)
    if command is None and "-cv" in lowered:
        verb, _, noun = lowered.partition("-cv")
        command = _COMMANDS.get(f"{verb}-{noun}")
        if command is not None and not command.module:
            command = None
    return command
