"""``CEVAHIR_HOME``: the state directory of the roof, always outside the git checkout.

The registry (manifests, locations), the audit chains and the console history live there
(``cevahir_ai.store``). Windows: ``%LOCALAPPDATA%\\Cevahir``; Linux and macOS:
``${XDG_STATE_HOME:-~/.local/state}/cevahir``; Colab: on Drive when it is mounted. The
``CEVAHIR_HOME`` environment variable overrides the default.
"""
import os
import sys
from pathlib import Path

from cevahir_ai._compat.repo import repo_root


def default_home(environ=None, platform=None):
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if environ.get("COLAB_RELEASE_TAG"):
        drive = Path("/content/drive/MyDrive")
        return drive / ".cevahir" if drive.is_dir() else Path("/content/.cevahir")
    if platform.startswith("win"):
        base = environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Cevahir"
    base = environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "cevahir"


def cevahir_home(environ=None):
    """Resolve the home directory. A home inside the checkout is refused, so state is never committed."""
    environ = os.environ if environ is None else environ
    home = Path(environ["CEVAHIR_HOME"]).expanduser() if environ.get("CEVAHIR_HOME") else default_home(environ)
    home = home.resolve()
    try:
        root = repo_root().resolve()
    except RuntimeError:                 # installed without a checkout: there is no checkout to protect
        return home
    if (home == root or root in home.parents) and environ.get("CEVAHIR_ALLOW_HOME_IN_REPO") != "1":
        raise RuntimeError(f"CEVAHIR_HOME must be outside the checkout ({home} is inside {root}); "
                           "set CEVAHIR_ALLOW_HOME_IN_REPO=1 to override")
    return home


def host_file(home, environ=None):
    """Where this machine's identity is kept. On Colab it is per VM, not on the shared Drive home,
    so two runtimes that share one home never append to one audit chain."""
    environ = os.environ if environ is None else environ
    if environ.get("CEVAHIR_HOST_FILE"):
        return Path(environ["CEVAHIR_HOST_FILE"]).expanduser()
    if environ.get("COLAB_RELEASE_TAG"):
        return Path("/content/.cevahir-host.json")
    return Path(home) / "host.json"
