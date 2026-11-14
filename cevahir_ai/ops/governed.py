"""The enforcement point, in its bootstrap form: every acting operation is recorded.

``governed(...)`` wraps one user-level action. It asks the policy for a decision, runs the block
and appends one audit event that carries the decision and the outcome, also when the block fails.

There is no policy engine yet. The built-in bootstrap policy is two rules: ``read`` and ``mutate``
actions are allowed and recorded; ``destructive``, ``publish`` and ``governance`` actions are
refused, because the approvals that must guard them do not exist yet. Policies as data, modes,
approvals and signatures replace this module's ``decide`` in the policy phase.
"""
import getpass
import os
import platform
from contextlib import contextmanager

from cevahir_ai.errors import CvError, PolicyDenied

BOOTSTRAP_POLICY = "builtin:bootstrap-v0"
ACTION_CLASSES = ("read", "mutate", "destructive", "publish", "governance", "acknowledge")
_ALLOWED = ("read", "mutate")


def actor(environ=None):
    """Who acts: the OS user and the kind of session (``CEVAHIR_ACTOR``: human, agent or ci)."""
    environ = os.environ if environ is None else environ
    try:
        name = getpass.getuser()
    except Exception:
        name = "unknown"
    kind = environ.get("CEVAHIR_ACTOR") or ("ci" if environ.get("CI") else "human")
    return {"kind": kind if kind in ("human", "agent", "ci") else "human", "name": name, "platform": platform.system()}


def decide(action, action_class):
    """The bootstrap decision. Pure."""
    if action_class not in ACTION_CLASSES:
        raise ValueError(f"unknown action class: {action_class!r}")
    if action_class in _ALLOWED:
        return {"effect": "allow", "mode": "bootstrap", "policy": BOOTSTRAP_POLICY, "rules": ["B-1"]}
    return {"effect": "deny", "mode": "bootstrap", "policy": BOOTSTRAP_POLICY, "rules": ["B-2"],
            "reason": f"{action_class} actions need the policy layer and an approval; neither exists yet"}


class Action:
    """What the block reports back: targets, outputs and lineage of the action."""

    def __init__(self):
        self.targets, self.outputs, self.lineage, self.data = [], [], None, None
        self.event = None                # the appended audit event, set when the block has finished


@contextmanager
def governed(store, action, *, event, action_class="mutate", targets=(), params=None, secret=()):
    """Run the block as the governed action ``action``; append ``event`` with decision and outcome."""
    from cevahir_ai.store.audit import AuditLog
    from cevahir_ai.store.redact import redact_params
    log = AuditLog(store)
    decision = decide(action, action_class)
    common = {"action": action, "action_class": action_class, "actor": actor(store.environ),
              "params": redact_params(params or {}, secret, store.environ), "decision": decision}
    if decision["effect"] != "allow":
        log.append("policy.decision", targets=list(targets), outcome={"status": "denied"}, **common)
        raise PolicyDenied(f"{action} is denied by {BOOTSTRAP_POLICY} ({decision['rules'][0]}): {decision['reason']}")
    handle = Action()
    handle.targets = list(targets)
    try:
        yield handle
    except BaseException as error:
        outcome = {"status": "failed", "error": type(error).__name__, "message": str(error)[:500],
                   "exit": error.exit_code if isinstance(error, CvError) else 1}
        log.append("action.failed", targets=handle.targets, outcome=outcome, **common)
        raise
    outcome = {"status": "ok", "exit": 0}
    if handle.outputs:
        outcome["outputs"] = handle.outputs
    handle.event = log.append(event, targets=handle.targets, outcome=outcome, lineage=handle.lineage,
                              data=handle.data, **common)


def record(store, event, action, *, targets=(), params=None, outcome=None, data=None):
    """Append a read-class event that is worth keeping: a verification verdict."""
    from cevahir_ai.store.audit import AuditLog
    from cevahir_ai.store.redact import redact_params
    return AuditLog(store).append(event, action=action, action_class="read", actor=actor(store.environ),
                                  targets=list(targets), params=redact_params(params or {}, (), store.environ),
                                  decision=decide(action, "read"), outcome=outcome, data=data)
