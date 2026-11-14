"""Syntax tree of the console language. Plain data: a node holds no behaviour and no Python object
other than literals and other nodes, so a parsed script can be inspected, printed and tested."""
from dataclasses import dataclass, field


@dataclass
class Node:
    pos: int = field(default=0, compare=False, repr=False, kw_only=True)


# -------------------------------------------------------------------------------------- expressions
@dataclass
class Lit(Node):
    value: object


@dataclass
class Str(Node):
    """An expandable string: text parts and embedded expressions (``"$name"``, ``"$( ... )"``)."""
    parts: list


@dataclass
class Var(Node):
    name: str            # lower case; "_" for $_ and $PSItem


@dataclass
class Member(Node):
    target: Node
    name: str


@dataclass
class Call(Node):
    target: Node
    name: str
    args: list


@dataclass
class Index(Node):
    target: Node
    index: Node


@dataclass
class Unary(Node):
    op: str              # "-" | "not"
    operand: Node


@dataclass
class Binary(Node):
    op: str              # "+", "-", "*", "/", "%", "and", "or", or a comparison name ("eq", "clike", ...)
    left: Node
    right: Node


@dataclass
class Array(Node):
    items: list


@dataclass
class Hash(Node):
    pairs: list          # [(key text, expression or ScriptBlock)]


@dataclass
class Paren(Node):
    """``( pipeline )``. Inside a script block the content is an expression, never a command."""
    pipeline: Node


@dataclass
class ScriptBlock(Node):
    """``{ expression }``: one expression of the restricted language, evaluated per input object."""
    body: Node
    text: str = ""


# --------------------------------------------------------------------------------------- statements
@dataclass
class Argument(Node):
    name: str = None     # parameter name without the dash, or None for a positional value
    value: Node = None   # None for a bare ``-Name``


@dataclass
class Command(Node):
    name: str
    args: list


@dataclass
class Pipeline(Node):
    elements: list       # the first may be an expression node, the rest are commands


@dataclass
class Assign(Node):
    name: str
    op: str              # "=" | "+="
    value: Node


@dataclass
class If(Node):
    clauses: list        # [(condition pipeline, [statements])]
    otherwise: list = None


@dataclass
class ForEach(Node):
    name: str
    source: Node
    body: list


@dataclass
class Throw(Node):
    value: Node = None


@dataclass
class Exit(Node):
    value: Node = None


@dataclass
class ParamDecl(Node):
    name: str
    type: str = "string"
    mandatory: bool = False
    default: Node = None


@dataclass
class Script(Node):
    params: list
    statements: list


def dump(node):
    """The tree as nested lists and dicts (for golden tests and ``-Ast`` debugging)."""
    if isinstance(node, Node):
        data = {"node": type(node).__name__}
        for name, value in vars(node).items():
            if name != "pos":
                data[name] = dump(value)
        return data
    if isinstance(node, (list, tuple)):
        return [dump(item) for item in node]
    return node
