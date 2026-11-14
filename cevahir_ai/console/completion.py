"""Tab completion as a pure function: ``complete(line, cursor, session)``.

It completes command names and aliases, parameter names, the allowed values of a parameter,
variables, artefact keys in their short form (read from the manifests, no torch) and file paths.
The REPL feeds it to readline; ``Complete-Command '<partial line>'`` exposes it everywhere, which
also makes it testable and usable from the PowerShell module.
"""
import os
import re
from dataclasses import dataclass

from cevahir_ai.console import binder, registry

MAX_RESULTS = 200
_KEY_PARAMETERS = ("key", "ref", "reference", "difference", "target", "id")
_AUTOMATIC = ("true", "false", "null", "_", "PSItem", "Error", "LASTEXITCODE", "CevahirHome", "PolicyKey", "PWD",
              "ErrorActionPreference")
_SHORT_PREFIX = re.compile(r"^[a-z][a-z0-9]{1,15}:[0-9a-f:\-v]*$")


@dataclass(frozen=True)
class Completion:
    text: str
    kind: str
    description: str = ""


def split_path(word):
    """``(directory part including its separator, name prefix)``. Both ``/`` and ``\\`` separate, as
    PowerShell accepts both on Windows; the directory part is returned as typed."""
    cut = max(word.rfind("/"), word.rfind("\\"))
    return word[:cut + 1], word[cut + 1:]


def current_word(text):
    """``(segment, word, quote)``: the command segment under the cursor, the word being typed and the
    quote character it started with (or '')."""
    start, quote, word_start = 0, "", 0
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote, word_start = char, index + 1
        elif char in "|;({=":
            start = word_start = index + 1
        elif char in " \t,":
            word_start = index + 1
    return text[start:].lstrip(), text[word_start:], quote


def _paths(word, limit=MAX_RESULTS):
    directory, prefix = split_path(word)
    base = os.path.expanduser(directory) if directory else "."
    if os.sep == "/" and "\\" in base and not os.path.isdir(base):
        base = base.replace("\\", "/")
    try:
        names = sorted(os.listdir(base), key=str.lower)
    except OSError:
        return []
    found = []
    for name in names:
        if name.lower().startswith(prefix.lower()) and (prefix.startswith(".") or not name.startswith(".")):
            is_dir = os.path.isdir(os.path.join(base, name))
            separator = ("\\" if "\\" in directory and "/" not in directory else os.sep) if is_dir else ""
            found.append(Completion(directory + name + separator, "Directory" if is_dir else "File"))
            if len(found) >= limit:
                break
    return found


def _keys(session, word):
    if session is None:
        return []
    try:
        artifacts = session.store.catalog().artifacts.values()
    except Exception:                     # completion must never fail the prompt
        return []
    found = []
    for artifact in artifacts:
        short = artifact.short()
        if short.startswith(word.lower()) or artifact.key.startswith(word.lower()):
            found.append(Completion(short if not word.startswith("cev:") else artifact.key, "Key", artifact.kind))
    return sorted(found, key=lambda item: item.text)


def complete(line, cursor=None, session=None):
    """Completions for the word under ``cursor`` in ``line``; each ``text`` replaces that word."""
    text = line if cursor is None else line[:cursor]
    segment, word, quote = current_word(text)
    words = segment.split()
    ends_with_space = text[-1:] in (" ", "\t") if text else True
    results = []
    lowered = word.lower()
    if not quote and (not words or (len(words) == 1 and not ends_with_space)):
        if word.startswith("$"):
            return _variables(session, word)
        culture = session.culture if session is not None else "en"
        from cevahir_ai.console.help import pick
        for command in registry.commands():
            if command.name.lower().startswith(lowered):
                results.append(Completion(command.name, "Command", pick(command.synopsis, culture)))
        for alias, command in sorted(registry.aliases().items()):
            if alias.startswith(lowered) and word:
                results.append(Completion(alias, "Alias", command.name))
        return results[:MAX_RESULTS]
    command = registry.find(words[0]) if words else None
    previous = (words[-1] if ends_with_space else words[-2] if len(words) > 1 else "") if words else ""
    if not quote and word.startswith("$"):
        return _variables(session, word)
    if not quote and word.startswith("-") and command is not None:
        return [Completion(f"-{parameter.name}", "Parameter", parameter.type)
                for parameter in binder.parameters_of(command) if parameter.name.lower().startswith(lowered[1:])]
    parameter = None
    if command is not None and previous.startswith("-"):
        try:
            parameter = binder.resolve(command, previous[1:].rstrip(":"))
        except Exception:
            parameter = None
    if parameter is not None and parameter.choices:
        return [Completion(choice, "Value", f"-{parameter.name}") for choice in parameter.choices
                if choice.lower().startswith(lowered)]
    if command is not None and command.name in ("Get-Help", "Get-Command") and not previous.startswith("-"):
        return [Completion(item.name, "Command") for item in registry.commands() if item.name.lower().startswith(lowered)]
    wants_key = (parameter is not None and parameter.name.lower() in _KEY_PARAMETERS) or \
        (word and _SHORT_PREFIX.match(lowered))
    if not wants_key and parameter is None and command is not None and command.module and not word:
        first = next((item for item in command.params if item.position == 0), None)
        wants_key = first is not None and first.name.lower() in _KEY_PARAMETERS
    if wants_key:
        results = _keys(session, word)
        if results:
            return results[:MAX_RESULTS]
    return _paths(word)


def _variables(session, word):
    names = list(_AUTOMATIC)
    if session is not None:
        names += [name for name in session.variables if name not in ("erroractionpreference",)]
    prefix = word[1:].lower()
    found = sorted({name for name in names if name.lower().startswith(prefix)}, key=str.lower)
    return [Completion("$" + name, "Variable") for name in found]
