"""Audit operations: read events, verify chains."""
from cevahir_ai._compat.clock import parse_since
from cevahir_ai.domain.records import Record
from cevahir_ai.errors import NotFound
from cevahir_ai.store.audit import AuditLog


def event_record(event):
    from cevahir_ai.ops.artifacts import short
    decision, outcome, actor = event.get("decision") or {}, event.get("outcome") or {}, event.get("actor") or {}
    targets = [target.get("key") for target in event.get("targets") or ()]
    return Record("Cevahir.AuditEvent", {
        "Seq": event.get("seq"), "Time": event.get("ts"), "Host": event.get("host"), "Event": event.get("event"),
        "Action": event.get("action"), "Class": event.get("class"), "Actor": actor.get("name"),
        "ActorKind": actor.get("kind"), "Decision": decision.get("effect"), "Policy": decision.get("policy"),
        "Outcome": outcome.get("status"), "Target": short(targets[0]) if targets else None, "Targets": targets,
        "Params": event.get("params") or {}, "Lineage": event.get("lineage"), "Data": event.get("data"),
        "OutcomeDetail": outcome, "Hash": event.get("hash"), "Prev": event.get("prev")})


def get_events(store, last=None, since=None, event=None, action=None, target=None, host=None):
    """Events of this host's chain (or ``host``'s), oldest first, after filtering.

    ``since``: a duration (``14d``, ``2h``) or a date; ``event`` and ``action``: wildcard patterns
    (``artifact.*``); ``target``: a key, an alias or a short form.
    """
    from fnmatch import fnmatchcase
    log = AuditLog(store)
    host = host or store.host_id()
    if host is not None and host not in log.hosts():
        raise NotFound(f"no audit chain of host {host} in {store.home}")
    floor = parse_since(since) if since is not None else None
    wanted = None
    if target is not None:
        from cevahir_ai.ops.artifacts import resolve
        try:
            wanted = resolve(store, target).key
        except NotFound:
            wanted = str(target)
    found = []
    for item in log.events(host):
        if floor is not None and str(item.get("ts", "")) < floor:
            continue
        if event is not None and not fnmatchcase(str(item.get("event", "")).lower(), str(event).lower()):
            continue
        if action is not None and not fnmatchcase(str(item.get("action", "")).lower(), str(action).lower()):
            continue
        if wanted is not None and wanted not in [entry.get("key") for entry in item.get("targets") or ()]:
            continue
        found.append(item)
    if last is not None:
        found = found[-int(last):] if int(last) > 0 else []
    return [event_record(item) for item in found]


def verdict_record(store, verdict):
    return Record("Cevahir.AuditVerdict", {
        "Host": verdict["host"], "Own": verdict["host"] == store.host_id(), "Ok": verdict["ok"],
        "Events": verdict["events"], "Segments": verdict["segments"], "Mac": verdict["mac"],
        "FirstBadSeq": verdict["first_bad_seq"], "HeadSeq": (verdict["head"] or {}).get("seq"),
        "Findings": [Record("Cevahir.AuditFinding", Segment=item["segment"], Line=item["line"], Seq=item["seq"],
                            Problem=item["problem"]) for item in verdict["findings"]],
        "Notes": list(verdict["notes"])})


def test_log(store, host=None, repair=False):
    """Verify this host's chain and every imported chain (or only ``host``'s). One verdict per chain."""
    log = AuditLog(store)
    own = store.host_id()
    hosts = [host] if host else ([own] if own and own not in log.hosts() else []) + log.hosts()
    if host and host not in log.hosts():
        raise NotFound(f"no audit chain of host {host} in {store.home}")
    if not hosts:
        return [verdict_record(store, log.verify(None))]
    verdicts = []
    for name in hosts:
        verdict = log.verify(name)
        if repair and name == own and not verdict["ok"]:
            if log.repair() is not None:
                verdict = log.verify(name)
                verdict["notes"].append("crash damage was repaired and recorded (log.repaired / log.checked)")
            else:
                verdict["notes"].append("not repaired: the findings are not crash damage; they stay as evidence")
        verdicts.append(verdict_record(store, verdict))
    return verdicts
