"""The expression language: evaluation of a parsed expression over plain data.

This is the whole of what a script block ``{ ... }`` can do. There is no Python ``eval``, no
``getattr`` on a host object, no import, no I/O. The evaluator walks the syntax tree and knows only:

  values      ``$null``, booleans, integers, floats, strings, arrays (lists), hashtables (dicts) and
              records (``cevahir_ai.domain.records.Record``: named properties holding the same kinds)
  members     a property of a record, a key of a hashtable, ``Count``/``Length`` of an array or a
              string, and the same member of every element of an array. Nothing else has members:
              an unknown member is ``$null``, as in PowerShell.
  methods     the string methods listed in ``METHODS``, dispatched from a fixed table
  operators   arithmetic, comparison (PowerShell semantics: case-insensitive, the right operand
              takes the left operand's type, an array on the left filters), ``-and``, ``-or``,
              ``-not``

Limits keep one expression from exhausting the process: results of ``*`` and ``+`` on strings and
arrays are capped, and regular expressions are limited in length.
"""
import fnmatch
import re

from cevahir_ai.console import syntax as s
from cevahir_ai.console.errors import EvaluationError
from cevahir_ai.console.parser import METHODS, ArgFloat, ArgInt
from cevahir_ai.domain.records import Record

MAX_ITEMS = 1_000_000          # characters of a string or elements of an array built by one operation
MAX_PATTERN = 512
_METHOD_NAMES = {name.lower() for name in METHODS}


# ------------------------------------------------------------------------------------- conversions
def truthy(value):
    if value is None or value is False:
        return False
    if isinstance(value, (list, tuple)):
        return truthy(value[0]) if len(value) == 1 else len(value) > 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value != ""
    return True


def to_text(value):
    """PowerShell's conversion to string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (ArgInt, ArgFloat)):
        return value.raw
    if isinstance(value, float):
        return repr(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return " ".join(to_text(item) for item in value)
    if isinstance(value, Record):
        return "@{" + "; ".join(f"{name}={to_text(item)}" for name, item in value.items()) + "}"
    if isinstance(value, dict):
        return "System.Collections.Hashtable"
    if isinstance(value, s.ScriptBlock):
        return value.text
    return str(value)


def to_number(value):
    """A number, or None when ``value`` is not one."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return 0
    if isinstance(value, str):
        from cevahir_ai.console.parser import number_token
        return number_token(value.strip())
    return None


def plain_number(value):
    """Argument numbers lose their spelling once they are data."""
    if isinstance(value, ArgInt):
        return int(value)
    if isinstance(value, ArgFloat):
        return float(value)
    return value


def unroll(value):
    """What a value contributes to a pipeline: the elements of an array, nothing for ``$null``."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# -------------------------------------------------------------------------------------- comparison
def _wildcard(pattern, case_sensitive):
    if len(pattern) > MAX_PATTERN:
        raise EvaluationError(f"the wildcard pattern is longer than {MAX_PATTERN} characters")
    return re.compile(fnmatch.translate(pattern), 0 if case_sensitive else re.IGNORECASE)


def _regex(pattern, case_sensitive):
    if len(pattern) > MAX_PATTERN:
        raise EvaluationError(f"the regular expression is longer than {MAX_PATTERN} characters")
    try:
        return re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
    except re.error as error:
        raise EvaluationError(f"invalid regular expression {pattern!r}: {error}") from None


def _equal(left, right, case_sensitive):
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool):
        return left == truthy(right)
    if isinstance(left, (int, float)):
        number = to_number(right)
        return number is not None and left == number
    if isinstance(left, str):
        text = to_text(right)
        return left == text if case_sensitive else left.casefold() == text.casefold()
    return left == right


def _order(left, right, case_sensitive):
    """-1, 0 or 1; the right operand takes the left operand's type."""
    if left is None:
        return 0 if right is None else -1
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        number = to_number(right)
        if number is None:
            raise EvaluationError(f"cannot compare the number {left!r} with {to_text(right)!r}")
        return (left > number) - (left < number)
    a, b = to_text(left), to_text(right)
    if not case_sensitive:
        a, b = a.casefold(), b.casefold()
    return (a > b) - (a < b)


def _scalar_compare(name, left, right, case_sensitive):
    if name == "eq":
        return _equal(left, right, case_sensitive)
    if name == "ne":
        return not _equal(left, right, case_sensitive)
    if name in ("gt", "ge", "lt", "le"):
        order = _order(left, right, case_sensitive)
        return {"gt": order > 0, "ge": order >= 0, "lt": order < 0, "le": order <= 0}[name]
    if name in ("like", "notlike"):
        matched = _wildcard(to_text(right), case_sensitive).match(to_text(left)) is not None
        return matched if name == "like" else not matched
    if name in ("match", "notmatch"):
        matched = _regex(to_text(right), case_sensitive).search(to_text(left)) is not None
        return matched if name == "match" else not matched
    raise EvaluationError(f"unknown comparison -{name}")


def compare(operator, left, right):
    """Evaluate a comparison operator by name (``eq``, ``clike``, ``notin`` ...)."""
    case_sensitive = operator.startswith("c") and operator != "contains"
    name = operator[1:] if case_sensitive else operator
    left, right = plain_number(left), plain_number(right)
    if name in ("contains", "notcontains"):
        found = any(_equal(item, right, case_sensitive) for item in unroll(left))
        return found if name == "contains" else not found
    if name in ("in", "notin"):
        found = any(_equal(item, left, case_sensitive) for item in unroll(right))
        return found if name == "in" else not found
    if isinstance(left, (list, tuple)):                    # an array on the left filters
        return [item for item in left if _scalar_compare(name, item, right, case_sensitive)]
    return _scalar_compare(name, left, right, case_sensitive)


# -------------------------------------------------------------------------------------- arithmetic
def _capped(size):
    if size > MAX_ITEMS:
        raise EvaluationError(f"the result would have more than {MAX_ITEMS} elements")


def arithmetic(operator, left, right):
    left, right = plain_number(left), plain_number(right)
    if operator == "+":
        if left is None:
            return right
        if isinstance(left, (list, tuple)):
            _capped(len(left) + len(unroll(right)))
            return list(left) + unroll(right)
        if isinstance(left, str):
            text = to_text(right)
            _capped(len(left) + len(text))
            return left + text
        if isinstance(left, dict) and isinstance(right, dict):
            return {**left, **right}
    if operator == "*" and isinstance(left, (str, list, tuple)):
        times = to_number(right)
        if not isinstance(times, int) or times < 0:
            raise EvaluationError("a string or an array can only be multiplied by a non-negative integer")
        _capped(len(left) * times)
        return left * times if isinstance(left, str) else list(left) * times
    a, b = to_number(left), to_number(right)
    if a is None or b is None or isinstance(left, (dict, Record)) or isinstance(right, (dict, Record, list, tuple)):
        raise EvaluationError(f"cannot apply {operator} to {to_text(left)!r} and {to_text(right)!r}")
    try:
        if operator == "+":
            return a + b
        if operator == "-":
            return a - b
        if operator == "*":
            result = a * b
            if isinstance(result, int) and result.bit_length() > 4096:
                raise EvaluationError("the integer is too large")
            return result
        if operator == "/":
            return a // b if isinstance(a, int) and isinstance(b, int) and a % b == 0 else a / b
        if operator == "%":
            if isinstance(a, int) and isinstance(b, int):
                return int(a - b * int(a / b))             # the sign of the dividend, as in PowerShell
            import math
            return math.fmod(a, b)
    except ZeroDivisionError:
        raise EvaluationError("attempted to divide by zero") from None
    except OverflowError:
        raise EvaluationError("the number is too large") from None
    raise EvaluationError(f"unknown operator {operator}")


# ----------------------------------------------------------------------------- members and methods
def member(value, name):
    """The member ``name`` of ``value``; ``$null`` when there is none. Never touches Python attributes."""
    lowered = name.lower()
    if isinstance(value, Record):
        if value.has(name):
            return value.get(name)
        return value.type_name if lowered == "pstypename" else None
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() == lowered:
                return item
        if lowered == "count":
            return len(value)
        if lowered == "keys":
            return list(value)
        if lowered == "values":
            return list(value.values())
        return None
    if isinstance(value, (list, tuple)):
        if lowered in ("count", "length"):
            return len(value)
        found = []                                          # member enumeration: the member of every element
        for item in value:
            inner = member(item, name)
            if inner is not None:
                found += inner if isinstance(inner, list) else [inner]
        return found if found else None
    if isinstance(value, str):
        return len(value) if lowered == "length" else None
    return None


def _text_arg(arguments, index, method):
    if index >= len(arguments):
        raise EvaluationError(f".{method}( ) is missing an argument")
    return to_text(arguments[index])


def _int_arg(arguments, index, method):
    if index >= len(arguments):
        raise EvaluationError(f".{method}( ) is missing an argument")
    number = to_number(arguments[index])
    if not isinstance(number, int):
        raise EvaluationError(f".{method}( ) needs an integer argument")
    return number


def _substring(text, arguments):
    start = _int_arg(arguments, 0, "Substring")
    length = _int_arg(arguments, 1, "Substring") if len(arguments) > 1 else len(text) - start
    if start < 0 or length < 0 or start + length > len(text):
        raise EvaluationError(f".Substring({start}, {length}) is outside the string (length {len(text)})")
    return text[start:start + length]


def _split(text, arguments):
    if not arguments:
        return text.split()
    separator = to_text(arguments[0])
    return text.split(separator) if separator else [text]


_STRING_METHODS = {
    "substring": _substring,
    "startswith": lambda text, a: text.startswith(_text_arg(a, 0, "StartsWith")),
    "endswith": lambda text, a: text.endswith(_text_arg(a, 0, "EndsWith")),
    "contains": lambda text, a: _text_arg(a, 0, "Contains") in text,
    "tolower": lambda text, a: text.lower(),
    "toupper": lambda text, a: text.upper(),
    "trim": lambda text, a: text.strip(_text_arg(a, 0, "Trim")) if a else text.strip(),
    "split": _split,
    "replace": lambda text, a: text.replace(_text_arg(a, 0, "Replace"), _text_arg(a, 1, "Replace")),
}
assert set(_STRING_METHODS) == _METHOD_NAMES


def call(value, name, arguments):
    """Call an allow-listed method. Strings only; anything else is an error, never a lookup."""
    function = _STRING_METHODS.get(name.lower())
    if function is None:
        raise EvaluationError(f"the method .{name}( ) does not exist in the console language",
                              error_id="MethodNotAllowed")
    if isinstance(value, (ArgInt, ArgFloat)):
        value = value.raw
    if not isinstance(value, str):
        kind = "$null" if value is None else type(value).__name__
        raise EvaluationError(f".{name}( ) is a string method; the value is {kind}", error_id="MethodNotAllowed")
    result = function(value, list(arguments))
    if isinstance(result, str):
        _capped(len(result))
    return result


def index(value, key):
    key = plain_number(key)
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key or (isinstance(name, str) and isinstance(key, str) and name.lower() == key.lower()):
                return item
        return None
    if isinstance(value, Record):
        return value.get(key) if isinstance(key, str) else None
    if isinstance(value, (list, tuple, str)):
        if isinstance(key, (list, tuple)):
            return [index(value, item) for item in key]
        number = to_number(key)
        if not isinstance(number, int) or not -len(value) <= number < len(value):
            return None
        return value[number]
    return None


# --------------------------------------------------------------------------------------- evaluator
class Evaluator:
    """Evaluates expression nodes. ``variables(name)`` supplies variable values; ``run_pipeline`` runs
    a parenthesised pipeline (outside script blocks only: inside them the parser allows none)."""

    def __init__(self, variables, run_pipeline=None):
        self.variables, self.run_pipeline = variables, run_pipeline
        # A fixed table from node class to handler: nothing is looked up by a name from the script.
        self._handlers = {s.Lit: self._lit, s.Var: self._var, s.Str: self._str, s.Member: self._member,
                          s.Call: self._call, s.Index: self._index, s.Unary: self._unary, s.Binary: self._binary,
                          s.Array: self._array, s.Hash: self._hash, s.Paren: self._paren,
                          s.ScriptBlock: self._scriptblock}

    def evaluate(self, node, scope=None):
        handler = self._handlers.get(type(node))
        if handler is None:
            raise EvaluationError(f"cannot evaluate {type(node).__name__}")
        return handler(node, scope or {})

    def lookup(self, name, scope):
        if name in scope:
            return scope[name]
        if name in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[name]
        return self.variables(name)

    def _lit(self, node, scope):
        return node.value

    def _var(self, node, scope):
        return self.lookup(node.name, scope)

    def _str(self, node, scope):
        text = "".join(part if isinstance(part, str) else to_text(self.evaluate(part, scope)) for part in node.parts)
        _capped(len(text))
        return text

    def _member(self, node, scope):
        return member(self.evaluate(node.target, scope), node.name)

    def _call(self, node, scope):
        target = self.evaluate(node.target, scope)
        return call(target, node.name, [self.evaluate(argument, scope) for argument in node.args])

    def _index(self, node, scope):
        return index(self.evaluate(node.target, scope), self.evaluate(node.index, scope))

    def _unary(self, node, scope):
        value = self.evaluate(node.operand, scope)
        if node.op == "not":
            return not truthy(value)
        number = to_number(value)
        if number is None or isinstance(value, (list, tuple, dict, Record)):
            raise EvaluationError(f"cannot negate {to_text(value)!r}")
        return -number

    def _binary(self, node, scope):
        if node.op == "and":
            return truthy(self.evaluate(node.left, scope)) and truthy(self.evaluate(node.right, scope))
        if node.op == "or":
            return truthy(self.evaluate(node.left, scope)) or truthy(self.evaluate(node.right, scope))
        left, right = self.evaluate(node.left, scope), self.evaluate(node.right, scope)
        if node.op in ("+", "-", "*", "/", "%"):
            return arithmetic(node.op, left, right)
        return compare(node.op, left, right)

    def _array(self, node, scope):
        items = []
        for item in node.items:
            value = self.evaluate(item, scope)
            if isinstance(item, s.Paren) and len(node.items) == 1:      # @( ... ): always an array
                return unroll(value)
            items.append(plain_number(value))
        return items

    def _hash(self, node, scope):
        return {key: (value if isinstance(value, s.ScriptBlock) else plain_number(self.evaluate(value, scope)))
                for key, value in node.pairs}

    def _paren(self, node, scope):
        if isinstance(node.pipeline, s.Pipeline):
            if self.run_pipeline is None:
                raise EvaluationError("a pipeline cannot run here")
            items = list(self.run_pipeline(node.pipeline, scope))
            return None if not items else items[0] if len(items) == 1 else items
        return self.evaluate(node.pipeline, scope)

    def _scriptblock(self, node, scope):
        return node                                         # a block is a value until a cmdlet applies it

    def apply(self, block, item, scope=None):
        """Evaluate a script block for one pipeline object (``$_``)."""
        if not isinstance(block, s.ScriptBlock):
            raise EvaluationError("a script block { ... } is expected; text is never evaluated as code")
        local = dict(scope or {})
        local["_"] = item
        return self.evaluate(block.body, local)
