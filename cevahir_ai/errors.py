"""Error hierarchy of the roof. Every error carries a category and a process exit code.

Exit codes: 1 terminating error, 2 usage or format error, 5 verification failed
(integrity, digest or compatibility).
"""


class CvError(Exception):
    category = "InvalidOperation"
    exit_code = 1


class KeyFormatError(CvError, ValueError):
    """A string is not a valid ``cev:<kind>:<scheme>:<digest>`` key."""
    category = "InvalidArgument"
    exit_code = 2

    def __init__(self, message, position=None):
        super().__init__(message if position is None else f"{message} (at character {position})")
        self.position = position


class UnknownSchemeError(KeyFormatError):
    """The ``(kind, scheme)`` pair is not registered: never a silent pass."""


class CanonicalError(CvError, ValueError):
    """A payload cannot be encoded as CJSON-1."""
    category = "InvalidData"


class UnsupportedTensor(CvError, TypeError):
    """A state-dict entry is outside every weight scheme (non-tensor, sparse, meta, quantized)."""
    category = "InvalidData"
    exit_code = 5


class EnvelopeError(CvError, ValueError):
    """A checkpoint file has no recognised envelope, or it cannot be read safely."""
    category = "InvalidData"
    exit_code = 5


class MigrationError(CvError, ValueError):
    """A weight-schema migration does not apply to the given state dict."""
    category = "InvalidData"


class IncompatibleCheckpoint(CvError):
    """The compatibility gate refused a load. ``verdict`` holds the structured reasons."""
    category = "SecurityError"
    exit_code = 5

    def __init__(self, verdict):
        super().__init__(verdict.summary_en)
        self.verdict = verdict


class AttestationError(CvError):
    """Weights in the model after loading are not the weights that passed the gate."""
    category = "SecurityError"
    exit_code = 5
