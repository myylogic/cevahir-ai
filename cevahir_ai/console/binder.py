"""Parameter binding: named (exact, alias or unique prefix), positional, switches, ``-Name:value``,
and pipeline input by value or by property name. Values are converted to the declared type; a
script-block parameter accepts only a parsed ``{ ... }`` block, never text."""
import os

from cevahir_ai.console import syntax as s
from cevahir_ai.console.errors import ParameterBindingError
from cevahir_ai.console.expr import plain_number, to_number, to_text, truthy
from cevahir_ai.console.parser import ArgFloat, ArgInt
from cevahir_ai.console.registry import Param
from cevahir_ai.domain.records import Record

ERROR_ACTIONS = ("Continue", "Stop", "SilentlyContinue", "Ignore")
COMMON = (
    Param("Verbose", "switch", help=("Write verbose messages.", "Ayrıntılı iletileri yaz.")),
    Param("ErrorAction", choices=ERROR_ACTIONS, aliases=("ea",),
          help=("What to do on a non-terminating error.", "Sonlandırmayan hatada ne yapılacağı.")),
)
WHATIF = Param("WhatIf", "switch", help=("Show what would happen; change nothing.",
                                         "Ne olacağını göster; hiçbir şeyi değiştirme."))
NO_VALUE = object()


def parameters_of(command):
    return list(command.params) + ([WHATIF] if command.whatif else []) + list(COMMON)


def _choice(parameter, value, command):
    text = to_text(value)
    for choice in parameter.choices:
        if choice.lower() == text.lower():
            return choice
    raise ParameterBindingError(
        f"{command.name}: -{parameter.name} cannot be {text!r}; allowed values: {', '.join(parameter.choices)}",
        error_id="ParameterArgumentValidationError")


def convert(parameter, value, command):
    """``value`` as the parameter's declared type. Raises ``ParameterBindingError``."""
    kind = parameter.type

    def refuse(expected):
        shown = "$null" if value is None else to_text(value)[:60]
        raise ParameterBindingError(f"{command.name}: -{parameter.name} needs {expected}, not {shown!r}",
                                    error_id="ParameterArgumentTransformationError")

    if kind == "scriptblock":
        if not isinstance(value, s.ScriptBlock):
            refuse("a script block { ... } (text is never evaluated as code)")
        return value
    if kind == "switch":
        return truthy(value)
    if kind == "any":
        return value if parameter.raw else plain_number(value)
    if kind == "objects":
        items = value if isinstance(value, (list, tuple)) else [value]
        return list(items) if parameter.raw else [plain_number(item) for item in items]
    if kind == "hashtable":
        if not isinstance(value, dict):
            refuse("a hashtable @{ ... }")
        return value
    if kind == "list":
        items = value if isinstance(value, (list, tuple)) else [value]
        if any(isinstance(item, (dict, Record, s.ScriptBlock)) or item is None for item in items):
            refuse("text values")
        items = [to_text(item) for item in items]
        return [_choice(parameter, item, command) for item in items] if parameter.choices else items
    if isinstance(value, (list, tuple, dict, Record, s.ScriptBlock)) or value is None:
        refuse({"int": "an integer", "float": "a number", "bool": "$true or $false"}.get(kind, "one text value"))
    if kind == "int":
        number = to_number(value) if not isinstance(value, bool) else None
        if not isinstance(number, int) or isinstance(number, bool):
            refuse("an integer")
        return int(number)
    if kind == "float":
        number = to_number(value) if not isinstance(value, bool) else None
        if number is None:
            refuse("a number")
        return float(number)
    if kind == "bool":
        if isinstance(value, str) and value.lower() in ("$true", "true", "$false", "false"):
            return value.lower() in ("$true", "true")
        if not isinstance(value, bool):
            refuse("$true or $false")
        return value
    text = to_text(value)
    if parameter.choices:
        return _choice(parameter, text, command)
    return os.path.expanduser(text) if kind == "path" else text


def resolve(command, name):
    """The parameter that ``-name`` means: exact name, alias, or a prefix that only one parameter has."""
    available = parameters_of(command)
    lowered = name.lower()
    for parameter in available:
        if parameter.name.lower() == lowered or lowered in (alias.lower() for alias in parameter.aliases):
            return parameter
    matches = [parameter for parameter in available if parameter.name.lower().startswith(lowered)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ParameterBindingError(f"{command.name}: -{name} is ambiguous: " +
                                    ", ".join(f"-{parameter.name}" for parameter in matches),
                                    error_id="AmbiguousParameter")
    raise ParameterBindingError(f"{command.name}: there is no parameter -{name} "
                                f"(see Get-Help {command.name})", error_id="NamedParameterNotFound")


def bind(command, arguments):
    """Bind evaluated arguments ``[(name or None, value or NO_VALUE)]``. Returns ``{parameter name: value}``.

    Pipeline parameters and mandatory checks are completed per input object by ``bind_input``.
    """
    bound, positional = {}, []
    index = 0
    while index < len(arguments):
        name, value = arguments[index]
        index += 1
        if name is None:
            positional.append(value)
            continue
        if name == "?":
            bound["?"] = True
            continue
        parameter = resolve(command, name)
        if parameter.name in bound:
            raise ParameterBindingError(f"{command.name}: -{parameter.name} is given more than once",
                                        error_id="ParameterAlreadyBound")
        if value is NO_VALUE:
            if parameter.type == "switch":
                bound[parameter.name] = True
                continue
            if index >= len(arguments) or arguments[index][0] is not None:
                raise ParameterBindingError(f"{command.name}: -{parameter.name} needs a value",
                                            error_id="MissingArgument")
            value = arguments[index][1]
            index += 1
        bound[parameter.name] = convert(parameter, value, command)
    slots = sorted((parameter for parameter in command.params if parameter.position is not None
                    and parameter.name not in bound), key=lambda parameter: parameter.position)
    for parameter in slots:
        if not positional:
            break
        if parameter.remaining:
            flat = []
            for value in positional:
                flat += value if isinstance(value, (list, tuple)) else [value]
            bound[parameter.name] = convert(parameter, flat, command)
            positional = []
        else:
            bound[parameter.name] = convert(parameter, positional.pop(0), command)
    if positional:
        shown = to_text(positional[0])[:40]
        raise ParameterBindingError(f"{command.name}: no positional parameter accepts {shown!r} "
                                    f"(see Get-Help {command.name})", error_id="PositionalParameterNotFound")
    return bound


def _property(item, parameter):
    for name in (parameter.name, *parameter.aliases):
        if isinstance(item, Record) and item.has(name):
            return item.get(name)
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(key, str) and key.lower() == name.lower():
                    return value
    return NO_VALUE


def bind_input(command, bound, item=NO_VALUE):
    """The parameters for one input object: pipeline binding, defaults and the mandatory check."""
    values = dict(bound)
    if item is not NO_VALUE:
        for parameter in command.params:
            if parameter.pipeline is None or parameter.name in values:
                continue
            if parameter.pipeline in ("property", "both"):
                value = _property(item, parameter)
                if value is not NO_VALUE and value is not None:
                    values[parameter.name] = convert(parameter, value, command)
                    continue
            by_value = not isinstance(item, (Record, dict)) or parameter.type in ("any", "objects")
            if parameter.pipeline in ("value", "both") and by_value and item is not None:
                values[parameter.name] = convert(parameter, item, command)
    for parameter in parameters_of(command):
        if parameter.name not in values:
            if parameter.mandatory:
                raise ParameterBindingError(f"{command.name}: the mandatory parameter -{parameter.name} is missing",
                                            error_id="MissingMandatoryParameter")
            values[parameter.name] = False if parameter.type == "switch" else parameter.default
    return values


def argv_arguments(argv):
    """Process arguments after ``-File script.ps1`` as binder arguments. Values stay text: nothing
    on the command line is evaluated."""
    arguments = []
    for token in argv:
        if len(token) > 1 and token[0] == "-" and token[1].isalpha():
            name, colon, inline = token[1:].partition(":")
            arguments.append((name, inline if colon else NO_VALUE))
        else:
            arguments.append((None, token))
    return arguments


def bind_script(declarations, arguments, evaluate):
    """Bind arguments ``[(name or None, value)]`` to a script's ``param( )`` block. Returns its variables."""
    from cevahir_ai.console.registry import Cmdlet
    params = [Param(decl.name, decl.type, position=index, mandatory=decl.mandatory)
              for index, decl in enumerate(declarations)]
    script = Cmdlet("script", None, params)
    values = {}
    bound = bind(script, arguments)
    for decl, parameter in zip(declarations, params):
        if parameter.name in bound:
            values[decl.name.lower()] = bound[parameter.name]
        elif decl.mandatory:
            raise ParameterBindingError(f"the script needs -{decl.name}", error_id="MissingMandatoryParameter")
        elif decl.default is not None:
            values[decl.name.lower()] = evaluate(decl.default)
        else:
            values[decl.name.lower()] = False if decl.type == "switch" else None
    return values


__all__ = ["ArgFloat", "ArgInt", "COMMON", "NO_VALUE", "argv_arguments", "bind", "bind_input", "bind_script",
           "convert", "parameters_of", "resolve"]
