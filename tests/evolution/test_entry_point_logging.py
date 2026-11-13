"""Importing the training entry modules must not configure logging; only their main() does.

Each check runs in a fresh interpreter: pytest itself installs root handlers, and a module already
imported by another test would not execute its import-time code again.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IMPORT_THEN_MAIN = r"""
import importlib, json, logging
calls = []
original = logging.basicConfig
def spy(*args, **kwargs):
    calls.append(1)
    return original(*args, **kwargs)
logging.basicConfig = spy
root = logging.getLogger()
before = [len(root.handlers), root.level]
modules = [importlib.import_module(name) for name in ("training_system.prepare_cache", "training_system.train")]
after_import = [len(root.handlers), root.level, len(calls)]
train = modules[1]
def stop(*args, **kwargs):
    raise KeyboardInterrupt  # main() turns this into SystemExit(130) before any training work
train.log_env_info = stop
try:
    train.main()
except SystemExit as exit_:
    code = exit_.code
print(json.dumps({"before": before, "after_import": after_import, "exit": code,
                  "after_main": [len(root.handlers), root.level, len(calls)]}))
"""

PREPARE_MAIN = r"""
import json, logging, sys
import training_system.prepare_cache as prepare
sys.argv = ["prepare_cache.py", "--help"]
try:
    prepare.main()
except SystemExit:
    pass
root = logging.getLogger()
print(json.dumps([len(root.handlers), root.level]))
"""


def run_python(code):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_entry_modules_leaves_root_logger_alone_and_main_configures_it():
    report = run_python(IMPORT_THEN_MAIN)
    assert report["before"] == [0, 30]
    assert report["after_import"] == [0, 30, 0], "import configured the root logger"
    assert report["exit"] == 130
    handlers, level, calls = report["after_main"]
    assert handlers >= 1 and level == 20 and calls >= 1, "train.main() must configure logging at INFO"


def test_prepare_cache_main_configures_logging():
    handlers, level = run_python(PREPARE_MAIN)
    assert handlers >= 1 and level == 20
