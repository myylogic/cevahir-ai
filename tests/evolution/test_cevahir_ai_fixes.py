"""Safety-net fixes that came with the roof.

1. ``training_system/health_check.py`` reported "production ready" for a file it could not run:
   a wrong import path led to a stub model, a missing tokenizer led to canned sample answers, and
   the canned answers passed every threshold. It now fails closed.
2. ``scripts/check_publications.py`` checked with bare ``assert``: under ``python -O`` every check
   vanished and the script still printed "passed". Every check now raises.
"""
import ast
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}


# ---------------------------------------------------------------------------------- health check
@pytest.fixture()
def health(monkeypatch):
    pytest.importorskip("torch")
    import logging
    monkeypatch.setattr(logging, "basicConfig", lambda *args, **kwargs: None)       # the checker configures logging
    from training_system import health_check
    return health_check


def _checkpoint(tmp_path, payload):
    import torch
    path = tmp_path / "model.pth"
    torch.save(payload, path)
    return str(path)


def test_health_check_does_not_call_an_unrunnable_checkpoint_ready(health, tmp_path, capsys):
    """Before the fix this exact file was reported as production_ready=True (stub model + canned answers)."""
    import torch
    path = _checkpoint(tmp_path, {"model_state": {"w": torch.zeros(3)}, "epoch": 1})
    report = health.HealthChecker(path).run()
    assert report["production_ready"] is False and report["load_success"] is False
    assert "responses" not in report or not any("yardımcı" in response for response in report["responses"])
    capsys.readouterr()


def test_health_check_import_failure_builds_no_stub(health, tmp_path, monkeypatch):
    import torch
    monkeypatch.setitem(sys.modules, "tokenizer_management.core.tokenizer_core", None)   # import raises ImportError
    checker = health.HealthChecker(_checkpoint(tmp_path, {"model_state": {"w": torch.zeros(3)}}))
    assert checker._initialize_model_from_checkpoint({"model_state": {"w": torch.zeros(3)}}) == (None, None)
    report = checker.run()
    assert report["production_ready"] is False and report["failure_reason"] == "Model yüklenemedi"


def test_health_check_uses_the_real_tokenizer_module_path():
    source = (ROOT / "training_system/health_check.py").read_text(encoding="utf-8")
    assert "from tokenizer_management.core.tokenizer_core import TokenizerCore" in source
    assert "tokenizer_management.tokenizer_core" not in source
    assert (ROOT / "tokenizer_management/core/tokenizer_core.py").is_file()
    assert not (ROOT / "tokenizer_management/tokenizer_core.py").exists()


def test_health_check_without_a_tokenizer_never_simulates_answers(health, tmp_path, monkeypatch, capsys):
    import torch
    checker = health.HealthChecker(_checkpoint(tmp_path, {"model_state": {}}))

    def loaded_without_tokenizer():
        checker._model, checker._tokenizer = torch.nn.Linear(1, 1), None
        return True

    monkeypatch.setattr(checker, "_load_model", loaded_without_tokenizer)
    monkeypatch.setattr(checker, "_simulate_responses", lambda: pytest.fail("simulated answers were used"))
    report = checker.run()
    assert report["responses"] == ["[INFERENCE YOK]"] * len(health.TEST_PROMPTS)
    assert report["production_ready"] is False and report["responses_from_model"] is False
    assert report["metrics"]["empty_ratio"] == 1.0
    capsys.readouterr()


def test_canned_answers_pass_the_thresholds_but_can_no_longer_decide_readiness(health, tmp_path, monkeypatch, capsys):
    """The thresholds alone accept the sample answers; readiness also needs answers from the real model."""
    import torch
    checker = health.HealthChecker(_checkpoint(tmp_path, {"model_state": {}}))
    canned = health.HealthChecker._simulate_responses(checker)
    assert checker._is_production_ready(checker._compute_quality_metrics(canned)) is True
    monkeypatch.setattr(checker, "_load_model", lambda: True)
    monkeypatch.setattr(checker, "_run_inference", lambda: canned)                   # answers not produced by the model
    report = checker.run()
    assert report["production_ready"] is False and report["metrics"]["production_ready"] is False

    class Tokenizer:
        def encode(self, prompt, return_tensors=None):
            return torch.tensor([[1, 2]])

    checker._model, checker._tokenizer = torch.nn.Linear(1, 1), Tokenizer()
    monkeypatch.undo()
    monkeypatch.setattr(health.logging, "basicConfig", lambda *args, **kwargs: None)
    answers = iter(canned)
    monkeypatch.setattr(checker, "_generate_response", lambda prompt, max_new_tokens=50: next(answers))
    assert checker._run_inference() == canned and checker._responses_from_model is True

    def failing(prompt, max_new_tokens=50):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(checker, "_generate_response", failing)
    assert checker._run_inference() == ["[HATA]"] * len(health.TEST_PROMPTS) and checker._responses_from_model is False
    capsys.readouterr()


def test_health_check_cli_exits_non_zero_for_an_unrunnable_checkpoint(tmp_path):
    pytest.importorskip("torch")
    import torch
    path = _checkpoint(tmp_path, {"model_state": {"w": torch.zeros(3)}, "epoch": 1})
    output = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(ROOT / "training_system/health_check.py"), "--model", path,
                             "--output", str(output)], cwd=tmp_path, env=ENV, capture_output=True, text=True, timeout=300)
    assert result.returncode == 1, result.stdout[-2000:]
    assert json.loads(output.read_text(encoding="utf-8"))["production_ready"] is False
    assert "ÜRETIM HAZIR\n" not in result.stdout.replace("ÜRETIM HAZIR DEĞİL", "")


def test_health_check_keeps_crlf_line_endings():
    data = (ROOT / "training_system/health_check.py").read_bytes()
    assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")


# -------------------------------------------------------------------------- publication checks
def test_check_publications_has_no_assert_statements():
    tree = ast.parse((ROOT / "scripts/check_publications.py").read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_require"]
    assert len(calls) >= 16 and all(len(call.args) == 2 for call in calls)


@pytest.mark.parametrize("flags", [[], ["-O"], ["-OO"]])
def test_check_publications_passes_on_the_repository_with_and_without_optimisation(flags):
    result = subprocess.run([sys.executable, *flags, str(ROOT / "scripts/check_publications.py")], cwd=ROOT, env=ENV,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "passed"


@pytest.mark.parametrize("flags", [[], ["-O"]])
def test_check_publications_fails_on_a_broken_tree_even_under_python_O(tmp_path, flags):
    """A copy of the checker over a tree whose author metadata is wrong must exit non-zero under -O."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("check_publications.py", "build_publications.py"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    publications = tmp_path / "docs" / "publications"
    publications.mkdir(parents=True)
    (publications / "catalog.json").write_text(json.dumps({"author": "Someone Else", "version": "1", "papers": []}), encoding="utf-8")
    (publications / "release.json").write_text(json.dumps({"author": "Someone Else", "version": "1", "doi": "", "status": "draft"}), encoding="utf-8")
    (tmp_path / ".zenodo.json").write_text(json.dumps({"creators": [], "version": "1"}), encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text("", encoding="utf-8")
    result = subprocess.run([sys.executable, *flags, str(scripts / "check_publications.py")], cwd=tmp_path, env=ENV,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode != 0 and "passed" not in result.stdout
    assert "publication check failed: catalog/release author" in result.stderr


def test_require_raises_under_optimisation():
    code = ("import sys; sys.path.insert(0, 'scripts'); import check_publications as c\n"
            "c._require(True, 'fine')\n"
            "try:\n    c._require(False, 'broken link')\nexcept SystemExit as error:\n    print(error.code); sys.exit(7)\n")
    result = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert result.returncode == 7 and "publication check failed: broken link" in result.stdout
