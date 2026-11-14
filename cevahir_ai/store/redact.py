"""Redaction of parameters before they enter the audit log.

The log is append-only: a secret written once stays. So a value is replaced, before it is hashed
into the chain, when its parameter is declared secret, when its name looks like a credential, when
it equals the value of a credential environment variable, or when it looks like a token or a
private key. A ``cev:`` key is never redacted ("key" is also the artefact noun). Hashes of secrets
are not stored either: a low-entropy secret could be guessed from its hash.
"""
import re

from cevahir_ai.keys.grammar import KEY_RE

NAME_RE = re.compile(r"(?i)secret|password|passwd|token|api[_-]?key|private|credential")
SECRET_ENV = ("JWT_SECRET_KEY", "SECRET_KEY", "ADMIN_PASSWORD", "API_KEY", "CEVAHIR_API_KEY", "DATABASE_URL",
              "POSTGRES_PASSWORD", "HF_TOKEN", "OPENAI_API_KEY", "WANDB_API_KEY")
VALUE_RES = (
    re.compile(r"^eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}$"),            # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"^(?:sk|ghp|gho|ghs|hf|xox[bap])[-_][A-Za-z0-9_-]{16,}$"),                  # common API token shapes
)
REDACTED = {"redacted": True}


def _is_key(value):
    return isinstance(value, str) and bool(KEY_RE.match(value))


def _secret_value(value, env_values):
    if not isinstance(value, str) or _is_key(value):
        return None
    for name, secret in env_values.items():
        if secret and len(secret) >= 6 and secret in value:
            return {"redacted": True, "ref": f"env:{name}"}
    if any(pattern.search(value) for pattern in VALUE_RES):
        return dict(REDACTED)
    return None


def _walk(value, env_values, depth=0):
    if depth > 16:
        return dict(REDACTED)
    if isinstance(value, dict):
        return {str(name): (dict(REDACTED) if NAME_RE.search(str(name)) and not _is_key(item)
                            else _walk(item, env_values, depth + 1)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_walk(item, env_values, depth + 1) for item in value]
    hidden = _secret_value(value, env_values)
    if hidden is not None:
        return hidden
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def redact_params(params, secret=(), environ=None):
    """A copy of ``params`` that is safe to log. ``secret`` names parameters declared secret."""
    import os
    environ = os.environ if environ is None else environ
    env_values = {name: environ.get(name) for name in SECRET_ENV if environ.get(name)}
    declared = {name.lower() for name in secret}
    result = {}
    for name, value in (params or {}).items():
        if name.lower() in declared or (NAME_RE.search(name) and not _is_key(value)):
            result[name] = dict(REDACTED)
        else:
            result[name] = _walk(value, env_values)
    return result
