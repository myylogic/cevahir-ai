"""The console session: variables, streams, the object pipeline and statement execution.

Objects flow between commands, not text: a cmdlet is a generator, and a pipeline is a chain of
generators, so ``Get-Checkpoint | Where-Object ... | Select-Object -First 1`` stops reading when
it has what it needs. Only what a statement leaves on the pipeline at the end is rendered as text.

Errors. A cmdlet failing on one input object writes an error record and the pipeline goes on
(``$ErrorActionPreference = 'Stop'`` or ``-ErrorAction Stop`` makes it terminating; script files
run with ``Stop``). A parse error, an unknown command, a binding error or ``throw`` ends the
statement. The process exit code is the code of the terminating error, else the value of ``exit``,
else the highest code among the errors written and the failed verifications (``Test-*`` never
throw: a failed verification sets code 5), else 0.

The console never starts another program: a name that is not a registered cmdlet or a ``.ps1``
script of this language is ``CommandNotFoundException``.
"""
import os
import shutil
import sys

from cevahir_ai.console import binder, registry
from cevahir_ai.console import syntax as s
from cevahir_ai.console.errors import (CommandNotFound, ConsoleError, ErrorRecord, EvaluationError, ExitScript,
                                       ParameterBindingError, ParserError, ThrowError)
from cevahir_ai.console.expr import Evaluator, arithmetic, plain_number, to_number, to_text, truthy, unroll
from cevahir_ai.console.format import Formatted, render
from cevahir_ai.console.parser import parse
from cevahir_ai.domain.records import Record
from cevahir_ai.errors import CvError

MAX_SCRIPT_DEPTH = 16
# What a cmdlet may raise for an expected failure; anything else is a defect and is shown as one.
EXPECTED = (CvError, OSError, ValueError, LookupError, ImportError)
_CATEGORIES = ((FileNotFoundError, "ObjectNotFound", 1), (PermissionError, "PermissionDenied", 1),
               (ImportError, "ResourceUnavailable", 1), (OSError, "ResourceUnavailable", 1),
               (LookupError, "ObjectNotFound", 1), (ValueError, "InvalidArgument", 2))


def classify(error):
    """Give a plain Python exception the category and exit code of the error model."""
    if isinstance(error, CvError):
        return error
    for kind, category, code in _CATEGORIES:
        if isinstance(error, kind):
            message = str(error)
            if isinstance(error, KeyError) and error.args:
                message = str(error.args[0])
            if isinstance(error, OSError) and error.filename and error.strerror:
                message = f"{error.strerror}: {error.filename}"
            if isinstance(error, ImportError) and getattr(error, "name", None) == "torch":
                message = "this command needs torch, which is not installed in this Python"
            return ConsoleError(message or kind.__name__, error_id=type(error).__name__, category=category, exit_code=code)
    return ConsoleError(str(error) or type(error).__name__, error_id=type(error).__name__)


class Context:
    """What a cmdlet sees of the session."""

    def __init__(self, session, command, verbose=False, whatif=False, error_action=None):
        self.session, self.command, self.verbose, self.whatif = session, command, verbose, whatif
        self.error_action = error_action or session.error_action

    @property
    def store(self):
        return self.session.store

    @property
    def culture(self):
        return self.session.culture

    def apply(self, block, item):
        return self.session.evaluator.apply(block, item)

    def host(self, text):
        self.session.write(text if text.endswith("\n") else text + "\n")

    def warning(self, message):
        self.session.write_error_stream(f"WARNING: {message}")
        self.session.warnings.append(str(message))

    def verbose_message(self, message):
        if self.verbose:
            self.session.write_error_stream(f"VERBOSE: {message}")

    def error(self, error):
        """Write a non-terminating error; under ``Stop`` it is raised instead."""
        error = classify(error)
        if self.error_action == "Stop":
            error._cevahir_command = self.command.name
            raise error
        if self.error_action in ("SilentlyContinue", "Ignore"):
            if self.error_action == "SilentlyContinue":
                self.session.errors.insert(0, ErrorRecord(error, self.command.name))
            return
        self.session.report(error, self.command.name)

    def failed(self, code=5):
        """A verification failed: not an error record, but the exit code must say so."""
        self.session.codes.append(code)

    def what_if(self, action, target):
        """True when ``-WhatIf`` was given: the cmdlet must then do nothing."""
        if self.whatif:
            self.host(f"What if: {action} \"{target}\"")
        return self.whatif


class Session:
    def __init__(self, store=None, *, home=None, environ=None, stdout=None, stderr=None, width=None, culture=None,
                 error_action="Continue", interactive=False, output="text", parent=None):
        self.environ = os.environ if environ is None else environ
        self._store, self._home = store, home
        self.stdout, self.stderr = stdout or sys.stdout, stderr or sys.stderr
        self.width = width or self._terminal_width()
        self._culture = culture
        self.interactive = interactive
        self.output = output                    # "text" | "json" (objects are collected, not rendered)
        self.collected = []
        self.variables = {"erroractionpreference": error_action}
        self.errors, self.warnings, self.codes = [], [], []
        self.last_ok = True
        self.exit_requested = False
        self.depth = parent.depth + 1 if parent else 0
        self.script_path = None
        self.evaluator = Evaluator(self.variable, self.run_pipeline)

    # ---------------------------------------------------------------------------------- basics
    def _terminal_width(self):
        try:
            return int(self.environ.get("CEVAHIR_WIDTH") or shutil.get_terminal_size((120, 24)).columns)
        except ValueError:
            return 120

    @property
    def store(self):
        if self._store is None:
            from cevahir_ai.store.core import Store
            try:
                self._store = Store(self._home, environ=self.environ)
            except RuntimeError as error:
                raise ConsoleError(str(error), error_id="HomeNotUsable", category="InvalidOperation") from None
        return self._store

    @property
    def culture(self):
        if self._culture is None:
            from cevahir_ai.ops.env import ui_culture
            self._culture = ui_culture(self.environ)
        return self._culture

    @property
    def error_action(self):
        value = to_text(self.variables.get("erroractionpreference"))
        return next((name for name in binder.ERROR_ACTIONS if name.lower() == value.lower()), "Continue")

    def write(self, text):
        self.stdout.write(text)

    def write_error_stream(self, text):
        self.stderr.write(text if text.endswith("\n") else text + "\n")

    def report(self, error, command=None):
        """Write an error record to the error stream and remember it (``$Error``, exit code)."""
        command = command or getattr(error, "_cevahir_command", None)
        record = ErrorRecord(classify(error), command)
        self.errors.insert(0, record)
        self.codes.append(record.exit_code)
        self.write_error_stream(record.render())
        return record

    def exit_code(self):
        return max(self.codes, default=0)

    def variable(self, name):
        if name in self.variables:
            return self.variables[name]
        if name == "error":
            return [Record("Cevahir.ErrorRecord", Message=r.message, Category=r.category,
                           FullyQualifiedErrorId=r.fully_qualified_error_id, Command=r.command) for r in self.errors]
        if name == "lastexitcode":
            return self.exit_code()
        if name == "?":
            return self.last_ok
        if name == "cevahirhome":
            return str(self.store.home)
        if name == "pwd":
            return os.getcwd()
        if name == "policykey":
            from cevahir_ai.ops.governed import BOOTSTRAP_POLICY
            return BOOTSTRAP_POLICY
        if name == "psscriptroot":
            return os.path.dirname(self.script_path) if self.script_path else ""
        return None

    def evaluate(self, node, scope=None):
        try:
            return self.evaluator.evaluate(node, scope)
        except RecursionError:
            raise EvaluationError("the expression is too deep to evaluate") from None

    # ------------------------------------------------------------------------------- pipelines
    def run_pipeline(self, node, scope=None):
        """The output of a pipeline, as an iterator of objects."""
        stream = None
        for position, element in enumerate(node.elements):
            if isinstance(element, s.Command):
                stream = self.run_command(element, stream, scope or {})
            else:
                stream = iter(unroll(plain_number(self.evaluate(element, scope))))
        return stream

    def run_command(self, node, upstream, scope):
        command = registry.find(node.name)
        if command is None:
            if node.name.lower().endswith(".ps1"):
                yield from self.run_child_script(node, scope)
                return
            raise CommandNotFound(
                f"'{node.name}' is not a command of the console. The console runs its own cmdlets and .ps1 scripts "
                "of its own language; it never starts another program.", target=node.name,
                recommended="Get-Command  (lists the commands);  Get-Help about_Cevahir_Subset")
        try:
            arguments = [(argument.name, binder.NO_VALUE if argument.value is None else self.evaluate(argument.value, scope))
                         for argument in node.args]
            bound = binder.bind(command, arguments)
            if bound.pop("?", False):
                from cevahir_ai.console.help import help_text
                yield Formatted(help_text(command, self.culture))
                return
            context = Context(self, command, verbose=bool(bound.get("Verbose")), whatif=bool(bound.get("WhatIf")),
                              error_action=bound.get("ErrorAction"))
            if command.kind == "stream":
                parameters = binder.bind_input(command, bound)
                yield from command.function(context, parameters, upstream if upstream is not None else iter(()))
            elif upstream is None:
                yield from command.function(context, binder.bind_input(command, bound))
            else:
                if not any(parameter.pipeline for parameter in command.params):
                    raise ParameterBindingError(f"{command.name} does not take pipeline input",
                                                error_id="InputObjectNotBound")
                for item in upstream:
                    try:
                        yield from command.function(context, binder.bind_input(command, bound, item))
                    except EXPECTED as error:
                        context.error(error)
        except EXPECTED as error:
            if not hasattr(error, "_cevahir_command"):
                try:
                    error._cevahir_command = command.name
                except AttributeError:
                    pass
            raise

    def run_child_script(self, node, scope):
        if self.depth >= MAX_SCRIPT_DEPTH:
            raise ConsoleError("scripts call each other too deeply", error_id="ScriptDepth")
        base = os.path.dirname(self.script_path) if self.script_path else os.getcwd()
        path = node.name if os.path.isabs(node.name) else os.path.join(base, node.name)
        if os.sep == "/" and not os.path.exists(path):        # .\child.ps1 written for Windows, run on Linux or Colab
            path = path.replace("\\", "/")
        arguments = [(argument.name, binder.NO_VALUE if argument.value is None else self.evaluate(argument.value, scope))
                     for argument in node.args]
        child = Session(self._store, home=self._home, environ=self.environ, stdout=self.stdout, stderr=self.stderr,
                        width=self.width, culture=self._culture, error_action=self.error_action, parent=self)
        child.errors, child.warnings, child.codes = self.errors, self.warnings, self.codes
        script = child.load_script(path)
        child.variables.update(binder.bind_script(script.params, arguments, child.evaluate))
        yield from child.execute(script.statements, {})

    # ------------------------------------------------------------------------------ statements
    def collect(self, node, scope):
        items = [plain_number(item) for item in self.run_pipeline(node, scope)]
        return None if not items else items[0] if len(items) == 1 else items

    def execute(self, statements, scope):
        for statement in statements:
            yield from self.execute_statement(statement, scope)

    def execute_statement(self, node, scope):
        if isinstance(node, s.Pipeline):
            yield from self.run_pipeline(node, scope)
        elif isinstance(node, s.Assign):
            value = self.collect(node.value, scope)
            if node.op == "+=":
                value = arithmetic("+", self.variable(node.name), value)
            self.variables[node.name] = value
        elif isinstance(node, s.If):
            for condition, body in node.clauses:
                if truthy(self.collect(condition, scope)):
                    yield from self.execute(body, scope)
                    return
            if node.otherwise is not None:
                yield from self.execute(node.otherwise, scope)
        elif isinstance(node, s.ForEach):
            for item in unroll(self.collect(node.source, scope)):
                self.variables[node.name] = item
                yield from self.execute(node.body, scope)
        elif isinstance(node, s.Throw):
            value = self.evaluate(node.value, scope) if node.value is not None else None
            raise ThrowError(to_text(value) or "ScriptHalted")
        elif isinstance(node, s.Exit):
            value = self.evaluate(node.value, scope) if node.value is not None else 0
            number = to_number(value)
            if not isinstance(number, int):
                raise EvaluationError(f"exit needs an integer code, not {to_text(value)!r}")
            raise ExitScript(number & 0xFF if number >= 0 else 1)
        else:
            raise EvaluationError(f"cannot execute {type(node).__name__}")

    def emit(self, objects):
        if self.output == "json":
            self.collected.extend(objects)
        elif objects:
            self.write(render(objects, self.width, self.evaluator.apply))

    def run_script(self, script):
        """Execute a parsed script statement by statement. Returns the exit code."""
        for statement in script.statements:
            before, objects = len(self.codes), []
            try:
                try:
                    for item in self.execute_statement(statement, {}):
                        objects.append(item)
                finally:
                    self.emit(objects)
            except ExitScript as request:
                self.exit_requested = True
                return request.code
            except KeyboardInterrupt:
                self.write_error_stream("interrupted")
                return 130
            except EXPECTED as error:
                record = self.report(error)
                if self.error_action == "Stop" or isinstance(error, ThrowError):
                    return record.exit_code
            except RecursionError:
                self.report(EvaluationError("the statement is nested too deeply"))
            self.last_ok = len(self.codes) == before
        return self.exit_code()

    def run(self, text):
        """Parse and run ``text``. Returns the exit code."""
        try:
            script = parse(text)
        except ParserError as error:
            return self.report(error).exit_code
        if script.params:
            try:
                self.variables.update(binder.bind_script(script.params, [], self.evaluate))
            except EXPECTED as error:
                return self.report(error).exit_code
        return self.run_script(script)

    def load_script(self, path):
        try:
            with open(path, "rb") as stream:
                data = stream.read()
        except OSError as error:
            raise ConsoleError(f"cannot read the script {path}: {error.strerror or error}", error_id="ScriptNotFound",
                               category="ObjectNotFound") from None
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ConsoleError(f"{path} is not UTF-8 text (save the script as UTF-8)", error_id="ScriptEncoding",
                               category="InvalidData", exit_code=2) from None
        self.script_path = os.path.abspath(path)
        return parse(text)

    def run_file(self, path, argv=()):
        """Run a ``.ps1`` file with ``-Name value`` arguments for its ``param( )`` block."""
        self.variables["erroractionpreference"] = "Stop"
        try:
            script = self.load_script(path)
            self.variables.update(binder.bind_script(script.params, binder.argv_arguments(argv), self.evaluate))
        except EXPECTED as error:
            return self.report(error).exit_code
        return self.run_script(script)
