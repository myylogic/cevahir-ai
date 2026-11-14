"""Console errors: what PowerShell calls error records, with the exit codes of the roof.

| exit | meaning                                                        |
|------|----------------------------------------------------------------|
| 0    | success                                                        |
| 1    | terminating error (includes not found)                         |
| 2    | parse or usage error (includes ``NotInCevahirSubset``)         |
| 3    | policy denied                                                  |
| 4    | approval required (reserved: no approvals exist yet)           |
| 5    | verification failed: integrity, audit chain or compatibility   |
| 130  | interrupted                                                    |
"""
from cevahir_ai.errors import CvError


class ConsoleError(CvError):
    category = "InvalidOperation"
    error_id = "ConsoleError"
    exit_code = 1

    def __init__(self, message, *, error_id=None, category=None, exit_code=None, target=None, recommended=None):
        super().__init__(message)
        if error_id is not None:
            self.error_id = error_id
        if category is not None:
            self.category = category
        if exit_code is not None:
            self.exit_code = exit_code
        self.target, self.recommended = target, recommended


class ParserError(ConsoleError):
    """The text is not a script of the console language. ``NotInCevahirSubset`` marks PowerShell
    constructs that exist but are deliberately not part of it; they are rejected, never approximated."""
    category = "InvalidArgument"
    error_id = "SyntaxError"
    exit_code = 2

    def __init__(self, message, position=None, line=None, column=None, error_id=None):
        location = f" (line {line}, column {column})" if line else ""
        super().__init__(message + location, error_id=error_id)
        self.position, self.line, self.column = position, line, column


class CommandNotFound(ConsoleError):
    category = "ObjectNotFound"
    error_id = "CommandNotFoundException"


class ParameterBindingError(ConsoleError):
    category = "InvalidArgument"
    error_id = "ParameterBindingException"
    exit_code = 2


class EvaluationError(ConsoleError):
    category = "InvalidOperation"
    error_id = "RuntimeException"


class ThrowError(ConsoleError):
    """``throw`` in a script."""
    category = "OperationStopped"
    error_id = "ScriptThrow"


class ExitScript(BaseException):
    """``exit n``: control flow, not an error."""

    def __init__(self, code=0):
        super().__init__(code)
        self.code = code


class ErrorRecord:
    """One entry of the error stream (and of ``$Error``)."""

    def __init__(self, error, command=None):
        self.exception_type = type(error).__name__
        self.message = str(error) or type(error).__name__
        self.category = getattr(error, "category", "NotSpecified")
        self.exit_code = getattr(error, "exit_code", 1)
        identifier = getattr(error, "error_id", None) or type(error).__name__
        self.command = command
        self.fully_qualified_error_id = f"{identifier},Cevahir.Commands.{(command or 'Console').replace('-', '')}"
        self.recommended = getattr(error, "recommended", None)

    def render(self):
        text = (f"{self.command or 'cevahir'} : [{self.category}] {self.message}\n"
                f"    + FullyQualifiedErrorId : {self.fully_qualified_error_id}")
        if self.recommended:
            text += f"\n    + Recommended : {self.recommended}"
        return text

    def to_dict(self):
        return {"exception_type": self.exception_type, "message": self.message, "category": self.category,
                "fully_qualified_error_id": self.fully_qualified_error_id, "exit": self.exit_code,
                "command": self.command, "recommended_action": self.recommended}
