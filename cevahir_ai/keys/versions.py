"""The versions table: every format or identity counter that exists in the code, with its owner.

The counters were introduced independently and nothing tied them together. A test reads each
owner by AST and compares it with this table, so bumping a counter without editing the table
fails; the edited entry must then say what the change is: a ``migration`` id or a ``break``.

``how`` is how the value appears in the owner file:
  ("const", NAME)        a module-level ``NAME = <literal>``
  ("local", NAME)        ``NAME = <literal>`` anywhere in the file (a function-local constant)
  ("dictkey", KEY)       a dict literal entry ``"KEY": <literal>``
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class VersionEntry:
    counter: str
    owner: str        # repository-relative path
    how: tuple
    value: object
    change: str = ""  # "migration: <id>" or "break: <reason>" when the value was last changed


VERSIONS = (
    VersionEntry("saver cevahir_version", "model_management/model_saver.py", ("const", "_CEVAHIR_CKPT_VERSION"), "4.1"),
    VersionEntry("saver checkpoint_format", "model_management/model_saver.py", ("const", "_CEVAHIR_FORMAT_VERSION"), 2),
    VersionEntry("loader current format", "model_management/model_loader.py", ("local", "_CURRENT_FORMAT"), 2),
    VersionEntry("training extra_state.format_version", "training_management/v2/core/training_manager.py",
                 ("dictkey", "format_version"), 1),
    VersionEntry("cache record format", "training_system/cache_identity.py", ("const", "CACHE_FORMAT_VERSION"), 2),
    VersionEntry("tokenizer identity_version", "training_system/cache_identity.py", ("dictkey", "identity_version"), 3),
    VersionEntry("model config_version", "model_management/config_schema.py", ("const", "CONFIG_VERSION"), 1),
    VersionEntry("architecture_version", "model_management/config_schema.py", ("const", "ARCHITECTURE_VERSION"),
                 "cevahir-capabilities-1"),
    VersionEntry("byte-BPE format version", "tokenizer_management/byte_bpe.py", ("const", "FORMAT_VERSION"), 1),
)


def read_owner_value(source, how):
    """Find the literal a ``how`` rule points at in Python ``source``. Raises ``LookupError`` if absent."""
    import ast
    kind, name = how
    tree = ast.parse(source)
    nodes = tree.body if kind == "const" else list(ast.walk(tree))
    for node in nodes:
        if kind in ("const", "local"):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(target, ast.Name) and target.id == name for target in targets) and node.value is not None:
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == name:
                    try:
                        return ast.literal_eval(value)
                    except ValueError:
                        continue
    raise LookupError(f"{how} not found")


def check(root):
    """Compare the table with the code under ``root``. Returns findings (empty = in step)."""
    from pathlib import Path
    findings = []
    for entry in VERSIONS:
        try:
            source = Path(root, entry.owner).read_text(encoding="utf-8-sig")
            actual = read_owner_value(source, entry.how)
        except (OSError, LookupError, SyntaxError) as error:
            findings.append(f"{entry.counter}: cannot read {entry.owner} {entry.how}: {error}")
            continue
        if actual != entry.value:
            findings.append(f"{entry.counter}: {entry.owner} says {actual!r}, the versions table says {entry.value!r}; "
                            "update cevahir_ai/keys/versions.py and state the migration or the break")
    return findings
