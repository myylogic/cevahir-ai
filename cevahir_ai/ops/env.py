"""Environment and responsibility read-outs (torch is reported from package metadata, never imported)."""
import locale
import os
import platform
import sys

from cevahir_ai.domain.records import Record


def ui_culture(environ=None):
    """``tr`` or ``en``: ``CEVAHIR_UI_CULTURE``, else the OS locale (``tr*`` -> ``tr``), else ``en``."""
    environ = os.environ if environ is None else environ
    value = (environ.get("CEVAHIR_UI_CULTURE") or "").strip().lower()
    if not value:
        for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
            value = (environ.get(name) or "").strip().lower()
            if value:
                break
    if not value:
        try:
            value = (locale.getlocale()[0] or "").lower()
        except (ValueError, TypeError):
            value = ""
    return "tr" if value.startswith(("tr", "turkish")) else "en"


def environment(store):
    from importlib import metadata
    from cevahir_ai import __version__
    from cevahir_ai.ops.governed import BOOTSTRAP_POLICY, actor
    from cevahir_ai.store.audit import AuditLog
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = None
    log = AuditLog(store)
    host = store.host_id()
    return Record("Cevahir.Environment", {
        "Version": __version__, "Python": platform.python_version(), "Platform": platform.system(),
        "Machine": platform.machine(), "Home": str(store.home), "HostId": host,
        "Artifacts": len(store.catalog().artifacts), "AuditChains": log.hosts(),
        "Policy": BOOTSTRAP_POLICY, "PolicyMode": "bootstrap: read and mutate allowed and recorded; nothing else exists yet",
        "Actor": actor(store.environ)["name"], "ActorKind": actor(store.environ)["kind"],
        "Torch": torch_version, "TorchLoaded": "torch" in sys.modules, "Culture": ui_culture(store.environ),
        "Utf8Mode": bool(sys.flags.utf8_mode), "StdoutEncoding": getattr(sys.stdout, "encoding", None)})


def responsibilities(identifier=None):
    from cevahir_ai._roof import RESPONSIBILITIES
    return [Record("Cevahir.Responsibility", Id=entry.id, Title=entry.title, Owner=entry.owner, Status=entry.status,
                   Implementations=list(entry.implementations))
            for entry in RESPONSIBILITIES if identifier is None or entry.id.lower() == str(identifier).lower()]
