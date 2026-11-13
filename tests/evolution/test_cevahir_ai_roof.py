"""The roof package: import cost, layering, facades, packaging and the state directory.

The roof must import without torch or any legacy package, must configure no logging, and its
facade names must be the legacy objects themselves (no copies, no silent fallbacks).
"""
import ast
import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROOF = ROOT / "cevahir_ai"
LEGACY = ("src", "model", "model_management", "tokenizer_management", "training_system", "training_management",
          "data_loader_management", "cognitive_management", "chatting_management", "api", "database")
HEAVY = ("torch", "numpy")
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}


def run(code, *args):
    return subprocess.run([sys.executable, "-c", code, *args], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=300)


def roof_modules():
    return sorted(path for path in ROOF.rglob("*.py"))


# ------------------------------------------------------------------------------------- import cost
def test_roof_imports_without_torch_or_legacy_and_configures_no_logging():
    code = r"""
import json, logging, sys, time
start = time.perf_counter()
import cevahir_ai, cevahir_ai.keys, cevahir_ai.evidence, cevahir_ai.domain
import cevahir_ai.neural, cevahir_ai.model, cevahir_ai.model.config, cevahir_ai.model.checkpoint
import cevahir_ai.tokenizer, cevahir_ai.data, cevahir_ai.training, cevahir_ai.cognition, cevahir_ai.serving
from cevahir_ai.keys import Key, cjson, structured_digest, check_compatibility, SCHEMES
import cevahir_ai.keys.cli, cevahir_ai.keys.versions, cevahir_ai.keys.classify, cevahir_ai.evidence.closure
elapsed = time.perf_counter() - start
root = logging.getLogger()
print(json.dumps({"elapsed": elapsed, "modules": sorted(sys.modules), "handlers": len(root.handlers),
                  "level": root.level, "version": cevahir_ai.__version__}))
"""
    result = run(code)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    loaded = {name.split(".")[0] for name in report["modules"]}
    assert not loaded & set(HEAVY), "the roof imported torch or numpy"
    assert not loaded & set(LEGACY), f"the roof imported a legacy package: {sorted(loaded & set(LEGACY))}"
    assert report["handlers"] == 0 and report["level"] == 30, "importing the roof configured logging"
    assert report["elapsed"] < 1.0, f"roof import took {report['elapsed']:.2f}s"
    assert report["version"] == "0.2.0"


def test_architecture_key_is_computed_without_torch():
    code = r"""
import sys
from cevahir_ai.adapters.model_build import arch_of_config, config_schema
key, fields, normalized = arch_of_config(config_schema().tiny_model_config())
assert "torch" not in sys.modules and "model_management" not in sys.modules
print(key, len(normalized), fields["num_heads"])
"""
    result = run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["cev:arch:cfg-v1:7ccdfc96fc5c534e31826c2a568e87bbc25932d594e8e554690aa8b161aa81fe", "47", "4"]


def test_version_entry_point():
    result = subprocess.run([sys.executable, "-m", "cevahir_ai", "--version"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "0.2.0"
    usage = subprocess.run([sys.executable, "-m", "cevahir_ai", "nonsense"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert usage.returncode == 2
    schemes = subprocess.run([sys.executable, "-m", "cevahir_ai", "keys", "schemes"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert schemes.returncode == 0 and "tok:legacy-v3" in schemes.stdout and "weights:sha256-v1" in schemes.stdout


# ---------------------------------------------------------------------------------------- layering
def _module_level_imports(tree):
    """Imports executed at import time: everything outside function bodies."""
    found, stack = [], list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            found.append(node.module or "")
        stack.extend(ast.iter_child_nodes(node))
    return found


def test_layering_torch_and_legacy_imports_only_where_allowed():
    torch_allowed = ("keys/torchkeys/",)
    for path in roof_modules():
        relative = path.relative_to(ROOF).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = {name.split(".")[0] for name in _module_level_imports(tree)}
        if not relative.startswith(torch_allowed):
            assert not top & set(HEAVY), f"{relative} imports torch/numpy at module level"
        assert not top & set(LEGACY), f"{relative} imports a legacy package at module level"
        everywhere = {name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                      for name in ([alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""])}
        assert not everywhere & set(LEGACY), f"{relative} imports a legacy package directly; use import_legacy"
        assert "research" not in everywhere, f"{relative} imports from research/"


def test_roof_never_imports_a_deprecated_or_unexported_module():
    from cevahir_ai._roof import DEPRECATED, EXPORTS, NOT_EXPORTED
    targets = {target.partition(":")[0] for table in EXPORTS.values() for target in table.values()}
    banned = set(DEPRECATED) | set(NOT_EXPORTED)
    assert not {target for target in targets if any(target == b or target.startswith(b + ".") for b in banned)}
    for path in roof_modules():
        source = path.read_text(encoding="utf-8")
        if path.name == "_roof.py":
            continue
        for name in banned:
            assert f'import_legacy("{name}' not in source, f"{path.name} imports {name}"


def test_no_silent_fallback_in_facades():
    """A facade has no try/except around its targets: a missing target must raise at first use."""
    for package in ("neural", "model", "tokenizer", "data", "training", "cognition", "serving"):
        for path in (ROOF / package).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if path.name == "checkpoint.py":
                continue
            assert not [node for node in ast.walk(tree) if isinstance(node, ast.Try)], f"{path} has a try block"
    import cevahir_ai.neural as neural
    with pytest.raises(AttributeError):
        neural.DoesNotExist


# ---------------------------------------------------------------------------------------- facades
def test_every_facade_export_is_the_legacy_object_itself():
    pytest.importorskip("torch")
    from cevahir_ai._roof import EXPORTS
    for facade, table in EXPORTS.items():
        module = importlib.import_module(f"cevahir_ai.{facade}")
        assert sorted(table) == sorted(name for name in module.__all__ if name in table)
        for name, target in table.items():
            legacy_module, _, attribute = target.partition(":")
            legacy = importlib.import_module(legacy_module)
            expected = getattr(legacy, attribute) if attribute else legacy
            assert getattr(module, name) is expected, f"cevahir_ai.{facade}.{name} is not {target}"
            assert name in dir(module)


def test_duplicate_legacy_names_get_distinct_roof_names():
    pytest.importorskip("torch")
    import cevahir_ai.model.config as config
    import cevahir_ai.serving as serving
    assert config.SystemConfig.__name__ == serving.FacadeConfig.__name__ == "CevahirConfig"
    assert config.SystemConfig is not serving.FacadeConfig
    import cevahir_ai.training as training
    assert training.TrainingService.__module__ == "training_system.v3.core.training_service_v3"
    assert training.TrainingEngine.__module__ == "training_management.v2.core.training_manager"


def test_responsibility_register_is_consistent():
    from cevahir_ai._roof import EXPORTS, RESPONSIBILITIES
    ids = [entry.id for entry in RESPONSIBILITIES]
    assert len(ids) == len(set(ids))
    implemented = {module for entry in RESPONSIBILITIES for module in entry.implementations}
    for entry in RESPONSIBILITIES:
        assert entry.status in ("owned", "facade", "planned")
        for module in entry.implementations:
            relative = Path(*module.split("."))
            assert (ROOT / relative.with_suffix(".py")).is_file() or (ROOT / relative / "__init__.py").is_file(), module
        owner = Path(*entry.owner.split("."))
        exists = (ROOT / owner.with_suffix(".py")).is_file() or (ROOT / owner / "__init__.py").is_file()
        assert exists == (entry.status != "planned"), f"{entry.id}: owner {entry.owner} vs status {entry.status}"
    # Every legacy module a facade exposes belongs to some responsibility's package.
    packages = {module.split(".")[0] for module in implemented}
    for table in EXPORTS.values():
        for target in table.values():
            assert target.split(".")[0] in packages, target


# ----------------------------------------------------------------- the live training entry point
def test_train_entry_point_still_selects_the_v3_service():
    pytest.importorskip("torch")
    result = run("import training_system.train as t; print(t._TRAINING_SYSTEM_V3_AVAILABLE, t.TrainingServiceV3.__name__)")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-2:] == ["True", "TrainingServiceV3"]


def test_training_management_v3_is_imported_only_by_the_train_entry_point():
    """Why the package is not deleted yet: one live entry point still imports it (for a log flag).

    If this list changes, the reachability proof in docs/architecture/CATI_VE_ANAHTAR_SISTEMI.md
    must be redone before anything is removed.
    """
    importers = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("training_management/v3/", "research/", ".")) or "/." in relative:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(name == "training_management.v3" or name.startswith("training_management.v3.") for name in names):
                importers.append(relative)
    assert sorted(set(importers)) == ["training_system/train.py"]


def test_removed_v3_monitoring_was_unreachable_and_the_rest_still_imports():
    """training_management/v3/monitoring (5 files, 2117 lines) was deleted: no module imported it.

    The proof is repeated here: no import of it anywhere in the tree (absolute or relative), and the
    package that train.py imports still imports, so ``_V3_AVAILABLE`` keeps its value.
    """
    assert not (ROOT / "training_management/v3/monitoring").exists()
    offenders = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("research/", ".")) or "/." in relative or relative.startswith("tests/evolution/test_cevahir_ai_"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        inside_v3 = relative.startswith("training_management/v3/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names] + \
                    [f"{node.module or ''}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in names:
                if "v3.monitoring" in name or (inside_v3 and name.split(".")[0] == "monitoring"):
                    offenders.append((relative, name))
    assert offenders == []
    result = run("import training_management.v3 as v3; from training_system import train; "
                 "print(v3.TrainingManager.__name__, train._V3_AVAILABLE)")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-2:] == ["TrainingManager", "True"]


# ------------------------------------------------------------------------- packaging and hygiene
def test_packaging_metadata():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "cevahir-ai" and project["dependencies"] == [] and project["dynamic"] == ["version"]
    assert project["scripts"]["cevahir"] == "cevahir_ai.__main__:main"
    assert data["tool"]["setuptools"]["packages"]["find"]["include"] == ["cevahir_ai*"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "cevahir_ai.__version__"}
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    minimum = tuple(int(part) for part in project["requires-python"].removeprefix(">=").split("."))
    assert minimum <= (3, 11), "must not exceed the Colab runtime"


def test_no_roof_file_is_swallowed_by_gitignore():
    files = [path.relative_to(ROOT).as_posix() for base in (ROOF, ROOT / "governance") for path in base.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts]
    for path in files:
        assert not {"runs", "logs", ".cache"} & set(Path(path).parts), f"{path} is under a git-ignored directory name"
    try:
        result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=ROOT, input="\n".join(files),
                                capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        pytest.skip("git is not available")
    assert result.returncode == 1 and not result.stdout.strip(), f"git would ignore: {result.stdout}"


def test_roof_sources_are_lf_only():
    for base in (ROOF, ROOT / "governance"):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".json", ".md"):
                assert b"\r" not in path.read_bytes(), f"{path} has CR bytes"
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "/cevahir_ai/** text eol=lf" in attributes and "/governance/** text eol=lf" in attributes


# ------------------------------------------------------------------------------- state directory
def test_home_is_outside_the_checkout(tmp_path):
    from cevahir_ai._compat import home
    assert home.cevahir_home({"CEVAHIR_HOME": str(tmp_path / "h")}) == (tmp_path / "h").resolve()
    with pytest.raises(RuntimeError, match="outside the checkout"):
        home.cevahir_home({"CEVAHIR_HOME": str(ROOT / "state")})
    assert home.cevahir_home({"CEVAHIR_HOME": str(ROOT / "state"), "CEVAHIR_ALLOW_HOME_IN_REPO": "1"}) == (ROOT / "state").resolve()
    assert home.default_home({"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}, "win32").name == "Cevahir"
    assert home.default_home({"XDG_STATE_HOME": "/s"}, "linux") == Path("/s/cevahir")
    assert home.default_home({"COLAB_RELEASE_TAG": "r"}, "linux").name == ".cevahir"


def test_atomic_write_and_streaming_digest(tmp_path):
    import hashlib
    from cevahir_ai._compat.fsio import atomic_write, sha256_file
    target = tmp_path / "a" / "b.bin"
    data = os.urandom(3 * 1024 * 1024 + 17)
    atomic_write(target, data)
    assert target.read_bytes() == data and sha256_file(target) == hashlib.sha256(data).hexdigest()
    atomic_write(target, b"x")
    assert target.read_bytes() == b"x" and [p.name for p in target.parent.iterdir()] == ["b.bin"]
