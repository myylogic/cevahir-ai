"""The CEV-11 closure pin: which product modules a frozen study depends on, and their digests.

The study's own freeze manifest covers the study folder only. The product modules it imports
(``src``, ``model_management``, ``tokenizer_management.byte_bpe`` ...) are outside it, so a change
there went unnoticed. The closure is every product module reachable from the study's ``.py`` files
by any import edge (eager, lazy or guarded; ``TYPE_CHECKING`` and ``__main__`` blocks excluded),
with package ``__init__`` files included. Each module is pinned by its utf8-lf digest, so CRLF and
LF checkouts agree.

  python -m cevahir_ai.evidence.closure check              exit 5 and name the files that changed
  python -m cevahir_ai.evidence.closure record --reason …  re-pin after a reviewed change

The pin lives in ``governance/freeze/cev11-closure.json``: outside ``research/`` (byte-protected)
and outside the study's own tests. A drift is not an error by itself; it says "a dependency of
CEV-11 changed: re-run the study's verification, then re-pin with the reason".
"""
import argparse
import ast
import json
import sys
from pathlib import Path

from cevahir_ai._compat.repo import LEGACY_PACKAGES, repo_root
from cevahir_ai.keys.schemes.code import code_tree_key
from cevahir_ai.keys.text import utf8lf_digest

STUDIES = {"CEV-11": "research/living_learning_paradigm_2026_11_13"}
PIN_PATH = ("governance", "freeze", "cev11-closure.json")
RULE = ("product modules reachable from the study's .py files by any import edge (eager, lazy, guarded); "
        "excluding TYPE_CHECKING and __main__ blocks; package __init__ included")
VERIFY_HINT = ("re-run the study's verification (python research/living_learning_paradigm_2026_11_13/verify.py --check "
               "and tests/evolution/test_cev11_*.py), then: python -m cevahir_ai.evidence.closure record --reason \"…\"")


def _excluded_block(node):
    """``if TYPE_CHECKING:`` and ``if __name__ == "__main__":`` bodies are not import edges."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators))


def _imports(tree, package):
    """Absolute module names imported anywhere in ``tree`` (``package``: dotted package of the file)."""
    found, stack = set(), [tree]
    while stack:
        node = stack.pop()
        if _excluded_block(node):
            stack.extend(node.orelse)
            continue
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join([*parts, base] if base else parts)
            if base:
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        stack.extend(ast.iter_child_nodes(node))
    return found


def _module_file(root, name):
    relative = Path(*name.split("."))
    for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def compute_closure(root=None, study="CEV-11"):
    """Returns the sorted list of ``(module name, posix path)`` in the study's product closure."""
    root = Path(root) if root else repo_root()
    study_root = root / STUDIES[study]
    if not study_root.is_dir():
        raise FileNotFoundError(f"study folder not found: {study_root}")
    pending, product = set(), {}
    for path in sorted(study_root.rglob("*.py")):
        pending |= _imports(ast.parse(path.read_text(encoding="utf-8-sig")), "")
    while pending:
        name = pending.pop()
        if name.split(".")[0] not in LEGACY_PACKAGES or name in product:
            continue
        path = _module_file(root, name)
        if path is None:
            continue                       # "from package import attribute": the attribute is not a module
        product[name] = path.relative_to(root).as_posix()
        parts = name.split(".")
        pending.update(".".join(parts[:index]) for index in range(1, len(parts)))
        package = name if path.name == "__init__.py" else ".".join(parts[:-1])
        for imported in _imports(ast.parse(path.read_text(encoding="utf-8-sig")), package):
            pending.add(imported)
            # ``src/neural_network.py`` puts its own directory on sys.path and imports
            # ``neural_network_module...`` without the ``src.`` prefix: resolve such names against
            # the importing module's ancestor packages as well.
            if imported.split(".")[0] not in LEGACY_PACKAGES:
                pending.update(f"{'.'.join(parts[:index])}.{imported}" for index in range(1, len(parts)))
    return sorted(product.items())


def closure_document(root=None, study="CEV-11", history=()):
    root = Path(root) if root else repo_root()
    modules = [{"module": name, "path": path, "sha256": utf8lf_digest(root / path)}
               for name, path in compute_closure(root, study)]
    key = code_tree_key([[entry["path"], entry["sha256"]] for entry in modules])
    return {"format": "cevahir-closure-pin", "version": 1, "study": study, "study_root": STUDIES[study],
            "rule": RULE, "digest": "utf8-lf", "modules": modules, "closure_key": str(key), "history": list(history)}


def pin_path(root=None):
    return Path(root or repo_root()).joinpath(*PIN_PATH)


def load_pin(root=None):
    return json.loads(pin_path(root).read_text(encoding="utf-8"))


def check(root=None, study="CEV-11"):
    """Compare the recorded pin with the checkout. Returns findings; empty means nothing changed."""
    pinned, current = load_pin(root), closure_document(root, study)
    old = {entry["path"]: entry["sha256"] for entry in pinned["modules"]}
    new = {entry["path"]: entry["sha256"] for entry in current["modules"]}
    findings = [f"{study} dependency changed: {path}" for path in sorted(old.keys() & new.keys()) if old[path] != new[path]]
    findings += [f"{study} dependency added (new import edge): {path}" for path in sorted(new.keys() - old.keys())]
    findings += [f"{study} dependency removed (import edge gone): {path}" for path in sorted(old.keys() - new.keys())]
    return findings


def record(reason, root=None, study="CEV-11", commit=None):
    """Re-pin the closure. The previous pin stays in ``history`` together with the stated reason."""
    from cevahir_ai._compat.fsio import atomic_write
    if not reason or not reason.strip():
        raise ValueError("a re-pin needs a reason")
    path = pin_path(root)
    history = load_pin(root).get("history", []) if path.is_file() else []
    document = closure_document(root, study)
    entry = {"closure_key": document["closure_key"], "modules": len(document["modules"]), "reason": reason.strip()}
    if commit:
        entry["commit"] = commit
    document["history"] = [*history, entry]
    atomic_write(path, (json.dumps(document, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m cevahir_ai.evidence.closure", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "record", "show"):
        sub = commands.add_parser(name)
        sub.add_argument("--study", default="CEV-11", choices=sorted(STUDIES))
        if name == "record":
            sub.add_argument("--reason", required=True)
            sub.add_argument("--commit", help="commit the pin describes (recorded in history)")
    args = parser.parse_args(argv)
    if args.command == "show":
        print(json.dumps(closure_document(study=args.study), ensure_ascii=False, indent=1))
        return 0
    if args.command == "record":
        document = record(args.reason, study=args.study, commit=args.commit)
        print(f"pinned {len(document['modules'])} modules: {document['closure_key']}")
        return 0
    findings = check(study=args.study)
    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print(VERIFY_HINT, file=sys.stderr)
        return 5
    pinned = load_pin()
    print(f"{args.study} closure unchanged: {len(pinned['modules'])} modules, {pinned['closure_key']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
