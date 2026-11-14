"""Parser of the console language: a strict subset of PowerShell.

    script     = [ "param(" declarations ")" ] { statement (NEWLINE | ";") }
    statement  = "$" NAME ("=" | "+=") pipeline | if | foreach | "throw" [expr] | "exit" [expr] | pipeline
    if         = "if" "(" pipeline ")" block { "elseif" "(" pipeline ")" block } [ "else" block ]
    foreach    = "foreach" "(" "$" NAME "in" pipeline ")" block
    pipeline   = ( command | expression ) { "|" command }           a trailing "|" continues the line
    command    = NAME { "-" PARAM [ ":" value ] | value { "," value } }
    value      = primary | BAREWORD | "{" expression "}"            (ckpt:3f9a1c0b, .\\x.pth, 14d)
    expression = comparison { ("-and" | "-or") comparison }
    comparison = additive { CMP additive }                          -eq -ne -gt -ge -lt -le -like -notlike
                                                                    -match -notmatch -contains -notcontains
                                                                    -in -notin (and their -c forms)
    additive   = term { ("+" | "-") term } ;  term = list { ("*" | "/" | "%") list }
    list       = unary { "," unary } ;  unary = ("-not" | "!" | "-") unary | postfix
    postfix    = primary { "." NAME | "." METHOD "(" args ")" | "[" expression "]" }
    primary    = NUMBER [KB|MB|GB] | 'text' | "text $var $( ... )" | "$" NAME | "$_" | "(" pipeline ")"
               | "@(" [ pipeline ] ")" | "@{" { NAME "=" (expression | "{" expression "}") } "}"

A script block ``{ ... }`` holds exactly one expression. It is not code: inside it there are no
commands, no assignments, no statements; ``( )`` and ``$( )`` hold expressions only; members whose
name starts with ``_`` do not exist; the only callable members are the string methods in
``METHODS``. Everything PowerShell has beyond this grammar (functions, loops other than foreach,
try/catch, casts, type literals, static members, ``&`` and ``.`` invocation, redirection,
here-strings, ``$env:`` and other scoped variables, the ``-f``/``-replace``/``-split`` operators) is
rejected at parse time with the error id ``NotInCevahirSubset``. Nothing is approximated.
"""
import re

from cevahir_ai.console import syntax as s
from cevahir_ai.console.errors import ParserError

METHODS = ("Substring", "StartsWith", "EndsWith", "Contains", "ToLower", "ToUpper", "Trim", "Split", "Replace")
_METHODS = {name.lower(): name for name in METHODS}
_BASE_COMPARISONS = ("eq", "ne", "gt", "ge", "lt", "le", "like", "notlike", "match", "notmatch",
                     "contains", "notcontains", "in", "notin")
COMPARISONS = frozenset(_BASE_COMPARISONS) | frozenset("c" + name for name in _BASE_COMPARISONS)
_CASE_INSENSITIVE = {"i" + name: name for name in _BASE_COMPARISONS}
REJECTED_OPERATORS = frozenset((
    "replace", "creplace", "ireplace", "split", "csplit", "isplit", "join", "f", "is", "isnot", "as", "band", "bor",
    "bxor", "bnot", "shl", "shr", "xor"))
REJECTED_KEYWORDS = frozenset((
    "function", "filter", "class", "enum", "switch", "while", "do", "until", "for", "break", "continue", "try",
    "catch", "finally", "trap", "return", "begin", "process", "end", "data", "dynamicparam", "using", "workflow",
    "parallel", "sequence", "inlinescript", "configuration", "elseif", "else", "in", "define", "from", "var"))
READ_ONLY_VARIABLES = frozenset(("_", "psitem", "true", "false", "null", "error", "?", "lastexitcode", "cevahirhome",
                                 "pshome", "pid", "host", "args", "input", "this"))
PARAM_TYPES = {"string": "string", "int": "int", "double": "float", "bool": "bool", "switch": "switch",
               "string[]": "list"}
_SUFFIX = {"kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3, "tb": 1024 ** 4}
_NUMBER = re.compile(r"(0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([kKmMgGtT][bB])?")
_NAME = re.compile(r"[^\W\d]\w*")           # an identifier: letters of any script (Turkish included), digits, _
_VARIABLE = re.compile(r"\$([^\W\d]\w*)")
_PARAMETER = re.compile(r"-([A-Za-z?][A-Za-z0-9_]*)")
_SINGLE, _DOUBLE = "'\u2018\u2019", '"\u201c\u201d'
_NEWLINE = "\r\n"
_COMMAND_END = set("|;)}\r\n")
_BAREWORD_END = set(" \t|;(){},'\"\r\n" + _SINGLE + _DOUBLE)
MAX_DEPTH = 64
MAX_SOURCE = 4 * 1024 * 1024


class ArgInt(int):
    """A number written as a command argument: it keeps its spelling, as PowerShell does, so that
    ``Get-Run 0193`` passes ``0193`` to a text parameter and ``193`` to a numeric one."""
    raw = ""


class ArgFloat(float):
    raw = ""


def number_token(text):
    """The value of ``text`` when all of it is a number literal (``5``, ``-1.5``, ``10MB``, ``0x1F``), else None."""
    sign = -1 if text.startswith("-") else 1
    body = text[1:] if text[:1] in "+-" else text
    match = _NUMBER.fullmatch(body)
    if not match or len(body) > 40:
        return None
    literal, suffix = match.groups()
    if literal[:2].lower() == "0x":
        value = int(literal, 16)
    elif any(mark in literal for mark in ".eE"):
        value = float(literal)
        if value in (float("inf"), float("-inf")):
            return None
    else:
        value = int(literal)
    value *= sign
    if suffix:
        value *= _SUFFIX[suffix.lower()]
        value = int(value) if float(value).is_integer() else value
    return value


class Parser:
    def __init__(self, text):
        if len(text) > MAX_SOURCE:
            raise ParserError("the script is larger than 4 MiB")
        self.text, self.pos, self.block, self.depth = text, 0, 0, 0

    # ------------------------------------------------------------------------------- primitives
    def fail(self, message, pos=None, subset=False):
        pos = self.pos if pos is None else pos
        line = self.text.count("\n", 0, pos) + 1
        column = pos - (self.text.rfind("\n", 0, pos) + 1) + 1
        raise ParserError(message, pos, line, column, "NotInCevahirSubset" if subset else "SyntaxError")

    def subset(self, what, pos=None):
        self.fail(f"{what}: not in the console language (see Get-Help about_Cevahir_Subset)", pos, subset=True)

    def peek(self, offset=0):
        index = self.pos + offset
        return self.text[index] if index < len(self.text) else ""

    def at_end(self):
        return self.pos >= len(self.text)

    def skip_space(self):
        """Skip blanks, comments and backtick line continuations. Returns True when anything was skipped."""
        start = self.pos
        while not self.at_end():
            c = self.peek()
            if c in " \t\f\v\u00a0":
                self.pos += 1
            elif c == "`" and self.peek(1) in _NEWLINE and self.peek(1):
                self.pos += 3 if self.text.startswith("\r\n", self.pos + 1) else 2
            elif c == "#" and (self.pos == 0 or self.text[self.pos - 1] in " \t\r\n;({|"):
                while not self.at_end() and self.peek() not in _NEWLINE:
                    self.pos += 1
            elif c == "<" and self.peek(1) == "#":
                end = self.text.find("#>", self.pos + 2)
                if end < 0:
                    self.fail("the block comment <# is never closed")
                self.pos = end + 2
            else:
                break
        return self.pos > start

    def skip_blank(self, separators=True):
        """Skip space, newlines and (unless told otherwise) ``;`` statement separators."""
        while True:
            self.skip_space()
            if self.peek() and (self.peek() in _NEWLINE or (separators and self.peek() == ";")):
                self.pos += 1
            else:
                return

    def expect(self, char, what=None):
        if self.peek() != char:
            self.fail(f"expected {what or repr(char)}" + (f", found {self.peek()!r}" if self.peek() else ", found the end"))
        self.pos += 1

    def word(self):
        """The keyword-like word at the cursor (lower case), without consuming it."""
        match = re.compile(r"[A-Za-z]+").match(self.text, self.pos)
        if not match or (self.peek(len(match.group())) and re.match(r"[\w\-.:\\/]", self.peek(len(match.group())))):
            return ""
        return match.group().lower()

    def enter(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            self.fail("the expression is nested too deeply")

    # ------------------------------------------------------------------------------- statements
    def parse_script(self):
        self.skip_blank()
        params = []
        if self.word() == "param":
            self.pos += 5
            self.skip_space()
            if self.peek() == "(":
                params = self.parse_param_block()
            else:
                self.fail("expected ( after param")
        statements = self.parse_statements(None)
        return s.Script(params, statements)

    def parse_statements(self, closing):
        statements = []
        while True:
            self.skip_blank()
            if self.at_end():
                if closing:
                    self.fail(f"missing {closing!r}")
                return statements
            if closing and self.peek() == closing:
                return statements
            statements.append(self.parse_statement())
            self.skip_space()
            if not self.at_end() and self.peek() not in _NEWLINE + ";" and self.peek() != (closing or ""):
                self.unexpected()

    def unexpected(self):
        c = self.peek()
        if c in "<>":
            self.subset("redirection (> >> <)")
        if c == "&":
            self.subset("the & operator (call, background, &&)")
        if c == "=":
            self.fail("unexpected '=': only a variable can be assigned ($name = ...)")
        self.fail(f"unexpected {c!r}")

    def parse_statement(self):
        start = self.pos
        if self.peek() == "$":
            assignment = self.try_assignment()
            if assignment is not None:
                return assignment
        word = self.word()
        if word == "if":
            return self.parse_if()
        if word == "foreach":
            return self.parse_foreach()
        if word in ("throw", "exit"):
            self.pos += len(word)
            self.skip_space()
            value = None
            if not self.at_end() and self.peek() not in _COMMAND_END:
                value = self.parse_expression()
            return (s.Throw if word == "throw" else s.Exit)(value, pos=start)
        if word == "param":
            self.fail("param( ) is only allowed at the very start of a script")
        if word in REJECTED_KEYWORDS:
            self.subset(f"the '{word}' statement")
        return self.parse_pipeline()

    def try_assignment(self):
        start = self.pos
        match = re.compile(r"\$(?:\{([^}\r\n]+)\}|([^\W\d]\w*|\?))").match(self.text, self.pos)
        if not match:
            return None
        end = match.end()
        scoped = self.text[end:end + 1] == ":" and re.match(r"[^\W\d]", self.text[end + 1:end + 2] or " ")
        if scoped:
            end = re.compile(r":\w+").match(self.text, end).end()
        rest = self.text[end:].lstrip(" \t")
        operator = next((op for op in ("+=", "-=", "*=", "/=", "%=", "++", "--", "=") if rest.startswith(op)), None)
        if operator is None:
            return None
        if scoped:
            self.subset("assignment to a scoped variable ($env:, $global:, $script:)", start)
        if operator != "=" and operator != "+=":
            self.subset(f"the {operator} operator", start)
        name = (match.group(1) or match.group(2)).lower()
        if name in READ_ONLY_VARIABLES:
            self.fail(f"${name} is an automatic variable and cannot be assigned", start)
        self.pos = end + (len(self.text[end:]) - len(rest)) + len(operator)
        self.skip_space()
        if self.at_end() or self.peek() in _COMMAND_END:
            self.fail("expected a value after the assignment")
        if self.word() in REJECTED_KEYWORDS or self.word() in ("if", "foreach"):
            self.subset(f"a '{self.word()}' statement as the value of an assignment")
        return s.Assign(name, operator, self.parse_pipeline(), pos=start)

    def parse_block(self):
        self.skip_blank(separators=False)
        self.expect("{", "{ to start a block")
        statements = self.parse_statements("}")
        self.expect("}")
        return statements

    def parse_condition(self):
        self.skip_space()
        self.expect("(", "( after the keyword")
        self.skip_blank(separators=False)
        condition = self.parse_pipeline()
        self.skip_blank(separators=False)
        self.expect(")", ") to close the condition")
        return condition

    def parse_if(self):
        start = self.pos
        self.pos += 2
        clauses = [(self.parse_condition(), self.parse_block())]
        otherwise = None
        while True:
            saved = self.pos
            self.skip_blank(separators=False)
            word = self.word()
            if word == "elseif":
                self.pos += 6
                clauses.append((self.parse_condition(), self.parse_block()))
            elif word == "else":
                self.pos += 4
                otherwise = self.parse_block()
                break
            else:
                self.pos = saved
                break
        return s.If(clauses, otherwise, pos=start)

    def parse_foreach(self):
        start = self.pos
        self.pos += 7
        self.skip_space()
        self.expect("(", "( after foreach")
        self.skip_space()
        match = _VARIABLE.match(self.text, self.pos)
        if not match:
            self.fail("expected a variable after foreach (")
        name = match.group(1).lower()
        if name in READ_ONLY_VARIABLES:
            self.fail(f"${name} is an automatic variable and cannot be a loop variable")
        self.pos = match.end()
        self.skip_space()
        if self.word() != "in":
            self.fail("expected 'in' in foreach ($x in ...)")
        self.pos += 2
        self.skip_space()
        source = self.parse_pipeline()
        self.skip_blank(separators=False)
        self.expect(")", ") to close foreach (")
        return s.ForEach(name, source, self.parse_block(), pos=start)

    def parse_param_block(self):
        self.expect("(")
        declarations = []
        while True:
            self.skip_blank(separators=False)
            if self.peek() == ")":
                self.pos += 1
                return declarations
            start, kind, mandatory = self.pos, "string", False
            while self.peek() == "[":
                depth, begin = 0, self.pos
                while not self.at_end():
                    depth += {"[": 1, "]": -1}.get(self.peek(), 0)
                    self.pos += 1
                    if depth == 0:
                        break
                attribute = re.sub(r"\s+", "", self.text[begin + 1:self.pos - 1]).lower()
                if attribute.startswith("parameter("):
                    mandatory = "mandatory" in attribute and "mandatory=$false" not in attribute
                elif attribute in PARAM_TYPES:
                    kind = PARAM_TYPES[attribute]
                else:
                    self.subset(f"the attribute or type [{self.text[begin + 1:self.pos - 1]}] in param( )", begin)
                self.skip_blank(separators=False)
            match = _VARIABLE.match(self.text, self.pos)
            if not match:
                self.fail("expected a parameter variable ($Name) in param( )")
            self.pos = match.end()
            self.skip_space()
            default = None
            if self.peek() == "=":
                self.pos += 1
                self.skip_space()
                default = self.parse_expression(comma=False)
            declarations.append(s.ParamDecl(match.group(1), kind, mandatory, default, pos=start))
            self.skip_blank(separators=False)
            if self.peek() == ",":
                self.pos += 1
            elif self.peek() != ")":
                self.fail("expected , or ) in param( )")

    # --------------------------------------------------------------------------------- pipeline
    def starts_expression(self):
        c = self.peek()
        if c and (c in "$(@" or c in _SINGLE or c in _DOUBLE or c.isdigit() or c == "!"):
            return True
        if c == "-" and (self.peek(1).isdigit() or self.text[self.pos:self.pos + 4].lower() == "-not"):
            return True
        if c == "." and self.peek(1).isdigit():
            return True
        return False

    def parse_pipeline(self):
        start = self.pos
        c = self.peek()
        if c == "{":
            self.subset("a script block outside a command argument (it is only a filter of Where-Object, "
                        "ForEach-Object, Sort-Object and of calculated properties)")
        if c == "[":
            self.subset("type literals, casts and static members ([type]...)")
        if c == "&" or (c == "." and self.peek(1) in " \t"):
            self.subset("the call (&) and dot-source (.) operators")
        elements = [self.parse_expression() if self.starts_expression() else self.parse_command()]
        while True:
            self.skip_space()
            if self.peek() != "|":
                break
            if self.peek(1) == "|":
                self.subset("the || operator")
            self.pos += 1
            self.skip_blank(separators=False)
            if self.at_end():
                self.fail("a command is expected after |")
            if self.starts_expression() or self.peek() in "{[":
                self.fail("only a command can follow | (an expression can only start a pipeline)")
            elements.append(self.parse_command())
        return s.Pipeline(elements, pos=start)

    def parse_command(self):
        start = self.pos
        while not self.at_end() and self.peek() not in _BAREWORD_END:
            self.pos += 1
        name = self.text[start:self.pos]
        if not name:
            self.pos = start
            self.unexpected()
        if re.search(r"[$`&<>\[\]=]", name):
            if name[0] in "<>&" or ">" in name:
                self.pos = start
                self.unexpected()
            self.fail(f"{name!r} is not a command name", start)
        arguments = []
        while True:
            self.skip_space()
            c = self.peek()
            if not c or c in _COMMAND_END:
                break
            if c in "<>" or c == "&" or (c in "0123456789" and re.match(r"\d>", self.text[self.pos:self.pos + 2])):
                self.subset("redirection and the & operator (> >> 2> < &)")
            position = self.pos
            parameter = _PARAMETER.match(self.text, self.pos) if c == "-" else None
            if parameter and (parameter.end() >= len(self.text) or self.text[parameter.end()] in " \t:\r\n|;)}"):
                self.pos = parameter.end()
                value = None
                if self.peek() == ":":
                    self.pos += 1
                    value = self.parse_value_list()
                arguments.append(s.Argument(parameter.group(1), value, pos=position))
            else:
                arguments.append(s.Argument(None, self.parse_value_list(), pos=position))
        return s.Command(name, arguments, pos=start)

    def parse_value_list(self):
        start = self.pos
        items = [self.parse_value()]
        while True:
            saved = self.pos
            self.skip_space()
            if self.peek() != ",":
                self.pos = saved
                break
            self.pos += 1
            self.skip_blank(separators=False)
            items.append(self.parse_value())
        return items[0] if len(items) == 1 else s.Array(items, pos=start)

    def parse_value(self):
        """One command argument: a quoted string, a variable, ( ), @( ), @{ }, { } or a bareword."""
        start, c = self.pos, self.peek()
        if c == "{":
            return self.parse_script_block()
        if c == "[":
            self.subset("type literals, casts and static members ([type]...)")
        if c == "$" or c == "(" or (c == "@" and self.peek(1) in "({") or c in _SINGLE or c in _DOUBLE:
            node = self.parse_postfix(self.parse_primary())
            if self.at_end() or self.peek() in _BAREWORD_END - set("'\"" + _SINGLE + _DOUBLE + "("):
                return node
            if c != "$":
                self.fail("unexpected text right after a value (separate arguments with a space)")
            self.pos = start                      # "$dir\name.pth": a bareword with a variable inside
        if c == "@" and (self.peek(1) in _SINGLE or self.peek(1) in _DOUBLE):
            self.subset("here-strings (@' ... '@)")
        if c == "@":
            self.subset("splatting (@name)")
        text = []
        while not self.at_end() and self.peek() not in _BAREWORD_END:
            if self.peek() == "`" and self.peek(1) and self.peek(1) not in _NEWLINE:
                text.append(self.peek(1))
                self.pos += 2
                continue
            if self.peek() in "<>&":
                self.subset("redirection and the & operator (> >> 2> < &)")
            text.append(self.peek())
            self.pos += 1
        raw = "".join(text)
        if not raw:
            self.unexpected()
        number = number_token(raw)
        if number is not None:
            token = (ArgInt if isinstance(number, int) else ArgFloat)(number)
            token.raw = raw
            return s.Lit(token, pos=start)
        if "$" in raw:
            return self.expandable(raw, start)
        return s.Lit(raw, pos=start)

    def expandable(self, raw, start):
        """A bareword that contains ``$``: parsed as the inside of a double-quoted string."""
        inner = Parser(raw + '"')
        inner.block = self.block
        try:
            node = inner.parse_double(0, closing='"')
        except ParserError as error:
            self.fail(str(error).split(" (line")[0], start, subset=error.error_id == "NotInCevahirSubset")
        node.pos = start
        return node

    def parse_script_block(self):
        start = self.pos
        self.expect("{")
        self.block += 1
        self.skip_blank()
        if self.peek() == "}":
            self.fail("an empty script block { }")
        try:
            body = self.parse_expression()
        except ParserError as error:
            if error.error_id == "NotInCevahirSubset":
                raise
            self.subset("a script block holds one expression (comparisons, -and/-or, property access, literals); "
                        "commands, assignments and statements are not allowed in it. " + str(error).split(" (line")[0],
                        error.position)
        self.skip_blank(separators=False)
        if self.at_end():
            self.fail("the script block { is never closed", start)
        if self.peek() != "}":
            self.subset("a script block holds one expression; commands, assignments, pipelines and several "
                        "statements are not allowed in it")
        self.pos += 1
        self.block -= 1
        return s.ScriptBlock(body, self.text[start:self.pos], pos=start)

    # ------------------------------------------------------------------------------ expressions
    def operator(self):
        """``(name, end)`` of the ``-name`` operator at the cursor after blanks, or ``(None, pos)``."""
        saved = self.pos
        self.skip_space()
        match = re.compile(r"-([A-Za-z]+)").match(self.text, self.pos)
        position = self.pos
        self.pos = saved
        if not match:
            return None, position
        name = match.group(1).lower()
        return _CASE_INSENSITIVE.get(name, name), match.end()

    def parse_expression(self, comma=True):
        self.enter()
        left = self.parse_comparison(comma)
        while True:
            name, end = self.operator()
            if name not in ("and", "or"):
                break
            self.pos = end
            left = s.Binary(name, left, self.parse_comparison(comma), pos=left.pos)
        if name is not None:
            self.bad_operator(name, end)
        self.depth -= 1
        return left

    def bad_operator(self, name, end):
        position = end - len(name) - 1
        if name in REJECTED_OPERATORS:
            self.subset(f"the -{name} operator", position)
        if name == "not":
            self.fail("-not is a prefix operator: write -not (expression)", position)
        self.fail(f"-{name} is not an operator of the console language", position)

    def parse_comparison(self, comma):
        left = self.parse_additive(comma)
        while True:
            name, end = self.operator()
            if name not in COMPARISONS:
                return left
            self.pos = end
            left = s.Binary(name, left, self.parse_additive(comma), pos=left.pos)

    def parse_additive(self, comma):
        left = self.parse_term(comma)
        while True:
            saved = self.pos
            self.skip_space()
            c = self.peek()
            if c == "+" or (c == "-" and not self.peek(1).isalpha()):
                self.pos += 1
                left = s.Binary(c, left, self.parse_term(comma), pos=left.pos)
            else:
                self.pos = saved
                return left

    def parse_term(self, comma):
        left = self.parse_list(comma)
        while True:
            saved = self.pos
            self.skip_space()
            c = self.peek()
            if c and c in "*/%":
                self.pos += 1
                left = s.Binary(c, left, self.parse_list(comma), pos=left.pos)
            else:
                self.pos = saved
                return left

    def parse_list(self, comma):
        first = self.parse_unary()
        if not comma:
            return first
        items = [first]
        while True:
            saved = self.pos
            self.skip_space()
            if self.peek() != ",":
                self.pos = saved
                break
            self.pos += 1
            self.skip_blank(separators=False)
            items.append(self.parse_unary())
        return first if len(items) == 1 else s.Array(items, pos=first.pos)

    def parse_unary(self):
        self.skip_space()
        start, c = self.pos, self.peek()
        self.enter()
        try:
            if c == "!":
                self.pos += 1
                return s.Unary("not", self.parse_unary(), pos=start)
            if c == "-" and self.text[self.pos:self.pos + 4].lower() == "-not" and not re.match(r"\w", self.peek(4) or " "):
                self.pos += 4
                return s.Unary("not", self.parse_unary(), pos=start)
            if c == "-" and not self.peek(1).isalpha():
                self.pos += 1
                return s.Unary("-", self.parse_unary(), pos=start)
            if c == "+" and (self.peek(1).isdigit() or self.peek(1) == "."):
                self.pos += 1
            return self.parse_postfix(self.parse_primary())
        finally:
            self.depth -= 1

    def parse_primary(self):
        start, c = self.pos, self.peek()
        if c and (c.isdigit() or (c == "." and self.peek(1).isdigit())):
            match = _NUMBER.match(self.text, self.pos)
            after = self.text[match.end():match.end() + 1]
            if after and re.match(r"[A-Za-z_]", after):
                self.fail(f"{self.text[start:match.end() + 1]}… is not a number (quote it if it is text)")
            value = number_token(match.group())
            if value is None:
                self.fail("the number is too large")
            self.pos = match.end()
            return s.Lit(value, pos=start)
        if c and c in _SINGLE:
            return self.parse_single()
        if c and c in _DOUBLE:
            return self.parse_double(self.pos + 1)
        if c == "$":
            return self.parse_variable()
        if c == "(":
            self.pos += 1
            self.skip_blank(separators=False)
            inner = self.parse_expression() if self.block else self.parse_pipeline()
            self.skip_blank(separators=False)
            self.expect(")", ") to close (")
            return s.Paren(inner, pos=start)
        if c == "@" and self.peek(1) == "(":
            self.pos += 2
            self.skip_blank()
            if self.peek() == ")":
                self.pos += 1
                return s.Array([], pos=start)
            inner = self.parse_expression() if self.block else self.parse_pipeline()
            self.skip_blank()
            self.expect(")", ") to close @(")
            return s.Array([s.Paren(inner, pos=start)], pos=start)
        if c == "@" and self.peek(1) == "{":
            return self.parse_hash()
        if c == "@" and (self.peek(1) in _SINGLE or self.peek(1) in _DOUBLE) and self.peek(1):
            self.subset("here-strings (@' ... '@)")
        if c == "{":
            self.subset("a script block as a value (it is only a filter argument of Where-Object, ForEach-Object, "
                        "Sort-Object and of calculated properties)")
        if c == "[":
            self.subset("type literals, casts and static members ([type]...)")
        if c == "&":
            self.subset("the call operator (&)")
        if not c:
            self.fail("an expression is expected, found the end")
        if self.block and re.match(r"[A-Za-z]", c):
            self.subset("a command or bare word inside a script block (it holds one expression; text needs quotes)")
        if re.match(r"[A-Za-z]", c):
            self.fail(f"unexpected word {self.text[start:start + 20].split()[0]!r} in an expression (text needs quotes; "
                      "a command can only start a pipeline)")
        self.fail(f"unexpected {c!r} in an expression")

    def parse_postfix(self, node):
        while True:
            c = self.peek()
            if c == "." and re.match(r"[^\W\d]", self.peek(1) or " "):
                match = _NAME.match(self.text, self.pos + 1)
                name = match.group()
                if name.startswith("_"):
                    self.subset(f"the member .{name} (names that start with _ are not accessible)")
                position = self.pos
                self.pos = match.end()
                if self.peek() == "(":
                    if name.lower() not in _METHODS:
                        self.subset(f"the method .{name}( ) (only these string methods exist: {', '.join(METHODS)})", position)
                    self.pos += 1
                    arguments = []
                    self.skip_blank(separators=False)
                    while self.peek() != ")":
                        arguments.append(self.parse_expression(comma=False))
                        self.skip_blank(separators=False)
                        if self.peek() == ",":
                            self.pos += 1
                            self.skip_blank(separators=False)
                        elif self.peek() != ")":
                            self.fail("expected , or ) in the method arguments")
                    self.pos += 1
                    node = s.Call(node, _METHODS[name.lower()], arguments, pos=position)
                else:
                    node = s.Member(node, name, pos=position)
            elif c == "." and (self.peek(1) in "$('\"" and self.peek(1)):
                self.subset("a computed member name (.$name, .( ), .'text')")
            elif c == "[":
                position = self.pos
                self.pos += 1
                self.skip_space()
                index = self.parse_expression()
                self.skip_space()
                self.expect("]", "] to close the index")
                node = s.Index(node, index, pos=position)
            elif c == ":" and self.peek(1) == ":":
                self.subset("static members (::)")
            else:
                return node

    def parse_variable(self):
        start = self.pos
        self.pos += 1
        c = self.peek()
        if c == "(":
            self.subset("$( ) outside a double-quoted string (use ( ) or @( ))", start)
        if c == "{":
            end = self.text.find("}", self.pos)
            name = self.text[self.pos + 1:end] if end > 0 else ""
            if not _NAME.fullmatch(name):
                self.subset("${...} with anything but a plain variable name", start)
            self.pos = end + 1
        elif c == "?":
            self.pos += 1
            name = "?"
        else:
            match = _NAME.match(self.text, self.pos)
            if not match:
                self.subset(f"the variable ${c}" if c else "a lone $", start)
            name = match.group()
            self.pos = match.end()
            if self.peek() == ":" and re.match(r"[^\W\d]", self.peek(1) or " "):
                self.subset(f"scoped variables (${name}:...; the environment and other scopes are not readable)", start)
        name = name.lower()
        return s.Var("_" if name == "psitem" else name, pos=start)

    def parse_single(self):
        start = self.pos
        self.pos += 1
        text = []
        while True:
            if self.at_end():
                self.fail("the string is never closed", start)
            c = self.peek()
            if c in _SINGLE:
                if self.peek(1) and self.peek(1) in _SINGLE:
                    text.append("'")
                    self.pos += 2
                    continue
                self.pos += 1
                return s.Lit("".join(text), pos=start)
            text.append(c)
            self.pos += 1

    _ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "a": "\a", "b": "\b", "e": "\x1b", "f": "\f", "v": "\v"}

    def parse_double(self, position, closing=None):
        """A double-quoted string from ``position`` (just after the opening quote) to the closing quote."""
        start = position - 1
        self.pos = position
        parts, text = [], []
        closers = closing or _DOUBLE

        def flush():
            if text:
                parts.append("".join(text))
                text.clear()

        while True:
            if self.at_end():
                self.fail("the string is never closed", max(start, 0))
            c = self.peek()
            if c in closers:
                if not closing and self.peek(1) and self.peek(1) in _DOUBLE:
                    text.append('"')
                    self.pos += 2
                    continue
                self.pos += 1
                break
            if c == "`" and self.peek(1):
                text.append(self._ESCAPES.get(self.peek(1), self.peek(1)))
                self.pos += 2
            elif c == "$" and self.peek(1) == "(":
                flush()
                at = self.pos
                self.pos += 2
                self.skip_blank()
                inner = self.parse_expression() if self.block else self.parse_pipeline()
                self.skip_blank()
                self.expect(")", ") to close $(")
                parts.append(s.Paren(inner, pos=at))
            elif c == "$" and (re.match(r"[^\W\d]|[{?]", self.peek(1) or " ")):
                flush()
                parts.append(self.parse_variable())
            else:
                text.append(c)
                self.pos += 1
        flush()
        if all(isinstance(part, str) for part in parts):
            return s.Lit("".join(parts), pos=max(start, 0))
        return s.Str(parts, pos=max(start, 0))

    def parse_hash(self):
        start = self.pos
        self.pos += 2
        pairs = []
        while True:
            self.skip_blank()
            if self.at_end():
                self.fail("the hashtable @{ is never closed", start)
            if self.peek() == "}":
                self.pos += 1
                return s.Hash(pairs, pos=start)
            c = self.peek()
            if c in _SINGLE:
                key = self.parse_single().value
            elif c in _DOUBLE:
                node = self.parse_double(self.pos + 1)
                if not isinstance(node, s.Lit):
                    self.fail("a hashtable key must be a literal")
                key = node.value
            else:
                match = re.compile(r"\w+").match(self.text, self.pos)
                if not match:
                    self.fail("expected a key in the hashtable")
                key = match.group()
                self.pos = match.end()
            self.skip_space()
            self.expect("=", "= after the hashtable key")
            self.skip_space()
            value = self.parse_script_block() if self.peek() == "{" else self.parse_expression()
            pairs.append((str(key), value))
            self.skip_space()
            if not self.at_end() and self.peek() not in _NEWLINE + ";}":
                self.fail("expected ; or a new line between hashtable entries")


def parse(text):
    """Parse a script. Raises ``ParserError`` (``NotInCevahirSubset`` for PowerShell beyond the subset)."""
    parser = Parser(text)
    try:
        return parser.parse_script()
    except RecursionError:
        raise ParserError("the script is nested too deeply") from None


def is_incomplete(text):
    """True when ``text`` only lacks its end (an open quote, brace, parenthesis or a trailing pipe):
    the REPL then asks for another line instead of reporting an error."""
    try:
        parse(text)
    except ParserError as error:
        message = str(error)
        return any(mark in message for mark in ("never closed", "missing '}'", "found the end", "expected after |",
                                                "is expected after |"))
    return False
