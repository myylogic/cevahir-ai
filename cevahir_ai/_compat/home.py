"""``CEVAHIR_HOME``: the state directory of the roof, always outside the git checkout.

Nothing in this phase writes there; the resolver exists so that the registry, the audit log and
the lineage store of the next phase have one definition of where state lives.
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
    root = repo_root().resolve()
    if (home == root or root in home.parents) and environ.get("CEVAHIR_ALLOW_HOME_IN_REPO") != "1":
        raise RuntimeError(f"CEVAHIR_HOME must be outside the checkout ({home} is inside {root}); "
                           "set CEVAHIR_ALLOW_HOME_IN_REPO=1 to override")
    return home
