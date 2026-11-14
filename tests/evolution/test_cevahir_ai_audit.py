"""The audit log: an append-only, hash-chained chain per host.

The tamper suite edits the chain the ways an editor could (one byte, a deleted line, swapped lines,
an inserted line, a cut tail, a rewritten chain without the key) and expects each one to be
reported, with the sequence number where it is knowable. Crash damage (a torn last line, a missing
head) is repaired into quarantine and recorded; tampering is never repaired. Everything here is
torch-free.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cevahir_ai.errors import AuditError, PolicyDenied
from cevahir_ai.keys.canonical import cjson
from cevahir_ai.ops import audit as audit_ops
from cevahir_ai.ops.governed import BOOTSTRAP_POLICY, decide, governed
from cevahir_ai.store import audit as audit_module
from cevahir_ai.store.audit import ZERO, AuditLog, event_hash, verify_chain
from cevahir_ai.store.core import Store
from cevahir_ai.store.redact import redact_params

ROOT = Path(__file__).resolve().parents[2]
KEY = "cev:ckpt:members-v1:" + "ab" * 32


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CEVAHIR_LOCK_DIR", str(tmp_path / "locks"))


@pytest.fixture
def store(tmp_path):
    ticks = iter(range(10**6))                              # a deterministic clock: one microsecond per event
    return Store(tmp_path / "home", clock=lambda: f"2026-11-13T12:00:00.{next(ticks):06d}Z")


def filled(store, count=6):
    log = AuditLog(store)
    for index in range(count):
        log.append("test.event", action="Test-Thing", action_class="read", targets=[{"kind": "ckpt", "key": KEY}],
                   data={"index": index, "text": "ğüşıöçİ"})
    return log


def segment(log):
    return log.segments(log.store.host_id())[0]


def problems(verdict):
    return [item["problem"] for item in verdict["findings"]]


# ------------------------------------------------------------------------------------------ chain
def test_chain_starts_with_genesis_and_every_event_links_to_the_one_before(store):
    log = filled(store, 3)
    events = list(log.events())
    assert [event["event"] for event in events] == ["log.genesis", "test.event", "test.event", "test.event"]
    assert [event["seq"] for event in events] == [1, 2, 3, 4] and events[0]["prev"] == ZERO
    for before, after in zip(events, events[1:]):
        assert after["prev"] == before["hash"] == event_hash(before)
    host = store.host_id()
    assert host.startswith("h-") and len(host) == 18 and {event["host"] for event in events} == {host}
    lines = segment(log).read_bytes().split(b"\n")
    assert lines[-1] == b"" and all(cjson(json.loads(line)) == line for line in lines[:-1]), "one canonical line per event"
    assert "ğüşıöçİ".encode("utf-8") in lines[1], "text is stored as UTF-8, not escaped"
    head = log.head(host)
    assert head["head"] == {"seq": 4, "hash": events[-1]["hash"], "seg": 1} and head["mac"].startswith("hmac-sha256:kid=")
    verdict = log.verify()
    assert verdict["ok"] and verdict["events"] == 4 and verdict["mac"] == "verified" and verdict["findings"] == []


def test_hash_covers_everything_but_hash_and_mac(store):
    event = list(filled(store, 1).events())[-1]
    body = {name: value for name, value in event.items() if name not in ("hash", "mac")}
    assert event["hash"] == "sha256:" + hashlib.sha256(cjson(body)).hexdigest()
    assert event_hash({**event, "mac": "other"}) == event["hash"] and event_hash({**event, "ts": "x"}) != event["hash"]


def test_empty_home_verifies_and_reading_creates_nothing(store):
    verdict = AuditLog(store).verify()
    assert verdict["ok"] and verdict["events"] == 0 and "no audit chain yet" in verdict["notes"][0]
    assert list(AuditLog(store).events()) == [] and not store.home.exists()
    record = audit_ops.test_log(store)[0]
    assert record.Ok and record.Events == 0 and not store.home.exists()


# ---------------------------------------------------------------------------------- tamper suite
def test_editing_one_byte_of_a_line_is_detected(store):
    log = filled(store)
    path = segment(log)
    data = path.read_bytes()
    target = data.index(b'"index":3')
    path.write_bytes(data[:target] + b'"index":9' + data[target + len(b'"index":3'):])
    verdict = log.verify()
    assert not verdict["ok"] and verdict["first_bad_seq"] == 5
    assert problems(verdict) == ["hash mismatch: the event was edited"]
    assert verdict["findings"][0]["line"] == 5 and verdict["findings"][0]["segment"] == "seg-000001.jsonl"


def test_deleting_a_line_is_detected(store):
    log = filled(store)
    path = segment(log)
    lines = path.read_bytes().split(b"\n")
    path.write_bytes(b"\n".join(lines[:3] + lines[4:]))
    verdict = log.verify()
    assert not verdict["ok"] and verdict["first_bad_seq"] == 5
    assert any("broken link" in text for text in problems(verdict))
    assert any("sequence number 5 where 4 was expected" in text for text in problems(verdict))


def test_swapping_two_lines_is_detected(store):
    log = filled(store)
    path = segment(log)
    lines = path.read_bytes().split(b"\n")
    lines[2], lines[3] = lines[3], lines[2]
    path.write_bytes(b"\n".join(lines))
    verdict = log.verify()
    assert not verdict["ok"] and verdict["first_bad_seq"] == 4 and any("broken link" in text for text in problems(verdict))


def test_inserting_a_forged_line_is_detected_even_with_a_recomputed_hash(store):
    log = filled(store)
    path = segment(log)
    lines = path.read_bytes().split(b"\n")
    forged = json.loads(lines[2])
    forged["data"] = {"index": 99}
    forged["hash"] = event_hash(forged)                    # a consistent hash, but not this host's MAC
    lines.insert(3, cjson(forged))
    path.write_bytes(b"\n".join(lines))
    texts = problems(log.verify())
    assert any("MAC mismatch" in text for text in texts) and any("broken link" in text for text in texts)


def test_rewriting_a_line_with_other_whitespace_is_detected(store):
    log = filled(store)
    path = segment(log)
    lines = path.read_bytes().split(b"\n")
    lines[2] = json.dumps(json.loads(lines[2]), ensure_ascii=False).encode("utf-8")      # same event, other bytes
    path.write_bytes(b"\n".join(lines))
    assert problems(log.verify()) == ["bytes are not the canonical form of the event: the line was rewritten"]


def test_cutting_lines_from_the_end_is_detected_through_the_head(store):
    log = filled(store)
    path = segment(log)
    lines = path.read_bytes().split(b"\n")
    path.write_bytes(b"\n".join(lines[:5]) + b"\n")         # events 6 and 7 removed; what is left links perfectly
    verdict = log.verify()
    assert not verdict["ok"]
    assert problems(verdict) == ["the chain ends at seq 5 but its head records seq 7: events were removed from the end"]
    with pytest.raises(AuditError, match="events were removed from the end"):
        log.append("test.event")                           # nothing is appended to hide the cut
    assert audit_ops.test_log(store, repair=True)[0].Ok is False, "a cut is not crash damage: it is never repaired"
    assert "not repaired" in audit_ops.test_log(store, repair=True)[0].Notes[-1]


def test_removing_the_whole_log_or_the_head_is_detected(store):
    log = filled(store)
    host = store.host_id()
    head = log._head_path(host)
    saved = head.read_bytes()
    head.unlink()
    assert any("head record (chain.json) is missing" in text for text in problems(log.verify()))
    with pytest.raises(AuditError, match="no head record"):
        log.append("test.event")
    head.write_bytes(saved)
    segment(log).unlink()
    with pytest.raises(AuditError, match="no segment files"):
        log.append("test.event")


def test_a_chain_rewritten_without_the_key_does_not_verify(store):
    """Someone who can write the log but cannot read the key rebuilds a fully consistent chain."""
    log = filled(store, 4)
    host = store.host_id()
    path = segment(log)
    previous, rebuilt = ZERO, []
    for line in path.read_bytes().split(b"\n")[:-1]:
        event = json.loads(line)
        event.pop("mac", None)
        if event["event"] == "test.event":
            event["data"] = {"index": "rewritten"}
        event["prev"] = previous
        event["hash"] = previous = event_hash(event)
        rebuilt.append(cjson(event))
    path.write_bytes(b"\n".join(rebuilt) + b"\n")
    last = json.loads(rebuilt[-1])
    log._head_path(host).write_text(json.dumps({"format": "cevahir-audit-chain", "version": 1, "host": host,
                                                "segments": [path.name],
                                                "head": {"seq": last["seq"], "hash": last["hash"], "seg": 1}}))
    verdict = log.verify()
    texts = problems(verdict)
    assert not verdict["ok"] and verdict["mac"] == "failed"
    assert any("no MAC" in text for text in texts) and any("head record does not carry this host's MAC" in text for text in texts)
    # The same chain held on a machine without the key (an imported copy) links correctly: what protects it
    # there is the copy itself, not the MAC. This is the stated limit of the chain.
    segments = [(path.name, path.read_bytes())]
    foreign = verify_chain(host, segments, log.head(host), key=None)
    assert foreign["ok"] and foreign["mac"] == "absent"


def test_an_event_moved_to_another_chain_is_detected(tmp_path, store):
    log = filled(store, 2)
    other = Store(tmp_path / "other", environ={})
    other_log = AuditLog(other)
    other_log.append("test.event")
    stolen = segment(log).read_bytes().split(b"\n")[1]
    target = segment(other_log)
    target.write_bytes(target.read_bytes() + stolen + b"\n")
    texts = problems(other_log.verify())
    assert any("another host" in text for text in texts) and any("MAC mismatch" in text for text in texts)


def test_a_missing_audit_key_is_not_mistaken_for_a_foreign_chain(store):
    """Deleting the key and stripping the MACs must not turn this host's own chain into an unchecked one."""
    log = filled(store, 3)
    host = store.host_id()
    key_file = store.home / "trust" / "hmac" / f"{host}.bin"
    assert len(key_file.read_bytes()) == 32
    if os.name == "posix":
        assert key_file.stat().st_mode & 0o777 == 0o600
    saved = key_file.read_bytes()
    key_file.unlink()
    verdict = log.verify()
    assert not verdict["ok"] and verdict["mac"] == "failed" and "audit key is missing" in problems(verdict)[-1]
    with pytest.raises(AuditError, match="audit key of h-[0-9a-f]{16} is missing"):
        log.append("test.event")
    assert not key_file.exists(), "a new key is never made for an existing chain"
    key_file.write_bytes(saved)
    assert log.verify()["ok"] and log.append("test.event")["seq"] == 5


# ---------------------------------------------------------------------------------- crash repair
def test_a_torn_last_line_is_quarantined_and_recorded_by_the_next_append(store):
    log = filled(store, 3)
    path = segment(log)
    torn = b'{"v":1,"seq":5,"ts":"2026-11-13T12:00:00.0'
    with open(path, "ab") as stream:
        stream.write(torn)                                  # a crash in the middle of a write
    verdict = log.verify()
    assert not verdict["ok"] and "incomplete last line" in problems(verdict)[0]
    appended = log.append("test.event", data={"after": "crash"})
    events = list(log.events())
    assert [event["event"] for event in events[-2:]] == ["log.repaired", "test.event"] and appended["seq"] == 6
    repaired = events[-2]["data"]
    assert repaired["sha256"] == hashlib.sha256(torn).hexdigest() and repaired["bytes"] == len(torn)
    quarantined = path.parent / "quarantine" / repaired["quarantined"]
    assert quarantined.read_bytes() == torn, "nothing is dropped: the torn bytes are kept"
    assert log.verify()["ok"]


def test_repair_switch_fixes_crash_damage_only(store):
    log = filled(store, 2)
    with open(segment(log), "ab") as stream:
        stream.write(b'{"half')
    record = audit_ops.test_log(store, repair=True)[0]
    assert record.Ok and "repaired" in record.Notes[-1]
    assert [event["event"] for event in log.events()][-2:] == ["log.repaired", "log.checked"]
    log._head_path(store.host_id()).unlink()                # a crash between the first write and the head update
    record = audit_ops.test_log(store, repair=True)[0]
    assert record.Ok and list(log.events())[-1]["data"]["repaired"] == ["the head record (chain.json) is missing: a cut "
                                                                        "at the end cannot be ruled out"]


def test_a_stale_head_after_a_crash_is_not_tampering(store):
    log = filled(store, 3)
    host = store.host_id()
    stale = log._head_path(host).read_bytes()
    log.append("test.event")
    log._head_path(host).write_bytes(stale)                 # the line was written, the head update was lost
    verdict = log.verify()
    assert verdict["ok"] and "head record is behind the chain" in verdict["notes"][-1]
    log.append("test.event")
    assert log.verify()["ok"] and not log.verify()["notes"]


# -------------------------------------------------------------------------------------- segments
def test_segments_roll_and_the_chain_continues_across_them(store):
    log = AuditLog(store, segment_bytes=600)
    for index in range(12):
        log.append("test.event", data={"index": index})
    host = store.host_id()
    names = [path.name for path in log.segments(host)]
    assert len(names) >= 3 and names[0] == "seg-000001.jsonl"
    events = list(log.events())
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    first_of_second = json.loads(log.segments(host)[1].read_bytes().split(b"\n")[0])
    assert first_of_second["event"] == "log.segment" and first_of_second["data"]["previous_segment"] == names[0]
    assert log.verify()["ok"] and log.verify()["segments"] == len(names)
    log.segments(host)[1].unlink()
    texts = problems(log.verify())
    assert any("segment 000002 is missing" in text for text in texts) and any("broken link" in text for text in texts)


# ----------------------------------------------------------------------------------- concurrency
WRITER = r"""
import sys
from cevahir_ai.store.audit import AuditLog
from cevahir_ai.store.core import Store
log = AuditLog(Store(sys.argv[1]))
for index in range(int(sys.argv[3])):
    log.append("test.event", data={"writer": sys.argv[2], "index": index})
"""


def test_processes_appending_at_the_same_time_give_one_valid_chain(tmp_path, store):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CEVAHIR_LOCK_DIR": str(tmp_path / "locks")}
    count, writers = 40, 4
    running = [subprocess.Popen([sys.executable, "-c", WRITER, str(store.home), str(number), str(count)], cwd=ROOT, env=env,
                                stderr=subprocess.PIPE, text=True) for number in range(writers)]
    for process in running:
        _, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr
    log = AuditLog(store)
    verdict = log.verify()
    assert verdict["ok"], verdict["findings"][:3]
    events = list(log.events())
    assert len(events) == writers * count + 1 and [event["seq"] for event in events] == list(range(1, len(events) + 1))
    per_writer = {}
    for event in events[1:]:
        per_writer.setdefault(event["data"]["writer"], []).append(event["data"]["index"])
    assert sorted(per_writer) == ["0", "1", "2", "3"] and all(indices == list(range(count)) for indices in per_writer.values())
    assert len({event["host"] for event in events}) == 1, "four processes on a fresh home agreed on one host id"


def test_file_lock_excludes_and_times_out(tmp_path):
    from cevahir_ai._compat.locks import FileLock, lock_path
    path = lock_path(tmp_path / "home", "unit")
    assert path.parent.parent == tmp_path / "locks" and tmp_path / "home" not in path.parents, "locks live outside the home"
    with FileLock(path):
        with pytest.raises(TimeoutError):
            with FileLock(path, timeout=0.2):
                pass
    with FileLock(path, timeout=0.2):                       # released: it can be taken again
        pass


def test_sharing_violations_are_retried(tmp_path, store, monkeypatch):
    """Windows antivirus and sync clients hold files open for a moment: ``PermissionError`` is retried."""
    log = filled(store, 1)
    real_open, failures = open, {"left": 3, "seen": 0}

    def flaky(path, mode="r", *args, **kwargs):
        if str(path).endswith(".jsonl") and "a" in mode and failures["left"]:
            failures["left"] -= 1
            failures["seen"] += 1
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(audit_module, "open", flaky, raising=False)
    monkeypatch.setattr(audit_module.time, "sleep", lambda seconds: None)
    log.append("test.event", data={"after": "retries"})
    assert failures["seen"] == 3 and log.verify()["ok"]
    failures["left"] = 100
    with pytest.raises(PermissionError):
        log.append("test.event")
    failures["left"] = 0
    assert log.verify()["ok"], "a failed append leaves the chain as it was"


# ------------------------------------------------------------------------------------- redaction
def test_secrets_never_reach_the_log():
    environ = {"JWT_SECRET_KEY": "s3cr3t-value-123", "HOME": "/home/x"}
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    params = {"Path": "/data/model.pth", "Password": "hunter2", "ApiKey": "abc", "Key": KEY, "Token": KEY,
              "Note": "uses s3cr3t-value-123 inside", "Header": jwt, "Nested": {"db_password": "x", "ok": [1, jwt]},
              "Pem": "-----BEGIN OPENSSH PRIVATE KEY-----\nabc", "Level": "File", "Count": 3}
    safe = redact_params(params, secret=("Level",), environ=environ)
    assert safe["Path"] == "/data/model.pth" and safe["Count"] == 3
    assert safe["Password"] == safe["ApiKey"] == safe["Header"] == safe["Pem"] == {"redacted": True}
    assert safe["Key"] == KEY and safe["Token"] == KEY, "a cev: key is an artefact name, never a secret"
    assert safe["Note"] == {"redacted": True, "ref": "env:JWT_SECRET_KEY"}
    assert safe["Nested"] == {"db_password": {"redacted": True}, "ok": [1, {"redacted": True}]}
    assert safe["Level"] == {"redacted": True}, "a parameter declared secret is always redacted"
    text = json.dumps(safe)
    assert "hunter2" not in text and "s3cr3t" not in text and "eyJ" not in text and "PRIVATE" not in text


def test_governed_actions_are_recorded_with_decision_and_outcome(store):
    with governed(store, "Register-Artifact", event="artifact.registered", params={"Path": "x", "Password": "p"}) as action:
        action.targets = [{"kind": "ckpt", "key": KEY}]
        action.lineage = {"output": KEY, "inputs": []}
    event = action.event
    assert event["event"] == "artifact.registered" and event["class"] == "mutate" and event["action"] == "Register-Artifact"
    assert event["decision"] == {"effect": "allow", "mode": "bootstrap", "policy": BOOTSTRAP_POLICY, "rules": ["B-1"]}
    assert event["outcome"] == {"status": "ok", "exit": 0} and event["params"]["Password"] == {"redacted": True}
    assert event["actor"]["kind"] in ("human", "agent", "ci") and event["actor"]["name"]
    with pytest.raises(ValueError, match="boom"):
        with governed(store, "Register-Artifact", event="artifact.registered"):
            raise ValueError("boom")
    failed = list(AuditLog(store).events())[-1]
    assert failed["event"] == "action.failed" and failed["outcome"]["status"] == "failed" and failed["outcome"]["error"] == "ValueError"


def test_bootstrap_policy_refuses_what_needs_an_approval(store):
    """There is no approval layer yet, so the actions that must have one are refused, and the refusal is recorded."""
    assert decide("Get-Artifact", "read")["effect"] == decide("Register-Artifact", "mutate")["effect"] == "allow"
    for action_class in ("destructive", "publish", "governance", "acknowledge"):
        assert decide("X", action_class)["effect"] == "deny"
    with pytest.raises(PolicyDenied) as info:
        with governed(store, "Publish-Model", event="model.published", action_class="publish"):
            raise AssertionError("the block of a denied action must not run")
    assert info.value.exit_code == 3 and info.value.category == "PermissionDenied"
    denied = list(AuditLog(store).events())[-1]
    assert denied["event"] == "policy.decision" and denied["decision"]["effect"] == "deny" and denied["outcome"]["status"] == "denied"
    with pytest.raises(ValueError):
        decide("X", "nonsense")


# ------------------------------------------------------------------------------------ operations
def test_get_audit_event_filters(store):
    log = AuditLog(store)
    log.append("artifact.registered", action="Register-Artifact", targets=[{"kind": "ckpt", "key": KEY}])
    log.append("artifact.verified", action="Test-Checkpoint", targets=[{"kind": "ckpt", "key": KEY}],
               outcome={"status": "failed"})
    log.append("artifact.exported", action="Export-Artifact")
    names = lambda records: [record.Event for record in records]          # noqa: E731
    assert names(audit_ops.get_events(store)) == ["log.genesis", "artifact.registered", "artifact.verified", "artifact.exported"]
    assert names(audit_ops.get_events(store, last=2)) == ["artifact.verified", "artifact.exported"]
    assert names(audit_ops.get_events(store, event="artifact.*", action="test-*")) == ["artifact.verified"]
    assert names(audit_ops.get_events(store, target=KEY)) == ["artifact.registered", "artifact.verified"]
    assert names(audit_ops.get_events(store, since="2026-11-13")) == names(audit_ops.get_events(store))
    assert audit_ops.get_events(store, since="2099-01-01") == []
    record = audit_ops.get_events(store, last=2)[0]
    assert record.Seq == 3 and record.Outcome == "failed" and record.Targets == [KEY] and record.Hash.startswith("sha256:")
    with pytest.raises(ValueError):
        audit_ops.get_events(store, since="yesterday")


def test_since_accepts_durations_and_dates():
    from cevahir_ai._compat.clock import parse_since, utc_now
    now = "2026-11-13T12:00:00.000000Z"
    assert parse_since("14d", now) == "2026-10-30T12:00:00.000000Z" and parse_since("2h", now) == "2026-11-13T10:00:00.000000Z"
    assert parse_since("2026-11-01") == "2026-11-01T00:00:00.000000Z"
    assert parse_since("2026-11-01T03:00:00+03:00") == "2026-11-01T00:00:00.000000Z"
    stamp = utc_now()
    assert len(stamp) == 27 and stamp.endswith("Z") and stamp[10] == "T"
