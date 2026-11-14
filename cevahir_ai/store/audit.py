"""The audit log: one append-only, hash-chained chain per host.

    audit/<host-id>/seg-000001.jsonl   one event per line: CJSON-1 followed by "\\n"
    audit/<host-id>/chain.json         the head: {seq, hash} of the last event (with a MAC)
    audit/<host-id>/quarantine/        torn tails moved aside by a repair, never deleted

Every event carries ``prev`` (the hash of the event before it) and ``hash``, the sha256 of its own
canonical bytes without ``hash`` and ``mac``. So editing a line breaks its hash, and deleting,
inserting or reordering lines breaks a link. Removing lines from the end breaks nothing inside the
chain; that is what the head record is for: a chain that ends before its recorded head was cut.
``mac`` is HMAC-SHA256 of the hash under the host's audit key, so someone who can write the log but
cannot read the key cannot produce a chain that verifies.

What this does not give: a process running as the same user can read the key and rewrite chain,
head and MACs consistently. Detecting that needs a signed copy of the head kept off the machine
(anchors), which is not implemented yet.

Writing: one writer at a time under a local lock; the line is written with one ``write``, flushed
and fsynced before the head is replaced, so after a crash the chain is at or ahead of its head,
never behind it. A torn last line (a crash in the middle of a write) is moved to ``quarantine/``
and recorded as ``log.repaired`` by the next append.
"""
import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path

from cevahir_ai._compat.fsio import atomic_write
from cevahir_ai.errors import AuditError
from cevahir_ai.keys.canonical import cjson
from cevahir_ai.store.core import sanitize

EVENT_VERSION = 1
ZERO = "sha256:" + "0" * 64
SEGMENT_BYTES = 64 * 1024 * 1024
SEGMENT_RE = re.compile(r"^seg-(\d{6})\.jsonl$")
HOST_RE = re.compile(r"^h-[0-9a-f]{16}$")
_TAIL_WINDOW = 64 * 1024
RESERVED = ("v", "seq", "ts", "host", "seg", "prev", "event", "hash", "mac")


def event_hash(event):
    """``sha256:<hex>`` of the event's canonical bytes without ``hash`` and ``mac``."""
    body = {name: value for name, value in event.items() if name not in ("hash", "mac")}
    return "sha256:" + hashlib.sha256(cjson(body)).hexdigest()


def _open_retry(path, mode, attempts=8):
    """``open`` that waits out Windows sharing violations (antivirus, indexers, sync clients)."""
    for attempt in range(attempts):
        try:
            return open(path, mode)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


def _tail(path):
    """``(last complete line or None, torn bytes after the last newline)`` of a segment."""
    size = path.stat().st_size
    if size == 0:
        return None, b""
    window = _TAIL_WINDOW
    with _open_retry(path, "rb") as stream:
        while True:
            start = max(0, size - window)
            stream.seek(start)
            data = stream.read(size - start)
            end = data.rfind(b"\n")
            if end < 0:
                if start == 0:
                    return None, data
            else:
                torn, body = data[end + 1:], data[:end]
                begin = body.rfind(b"\n")
                if begin >= 0 or start == 0:
                    return body[begin + 1:], torn
            window *= 4


class AuditLog:
    def __init__(self, store, segment_bytes=None):
        self.store = store
        configured = store.environ.get("CEVAHIR_AUDIT_SEGMENT_BYTES")
        self.segment_bytes = int(segment_bytes or configured or SEGMENT_BYTES)

    # ----------------------------------------------------------------------------------- layout
    def directory(self, host):
        if not HOST_RE.match(host or ""):
            raise AuditError(f"not a host id: {host!r}")
        return self.store.home / "audit" / host

    def hosts(self):
        """Hosts whose chains are in this home: this machine's own and imported ones."""
        base = self.store.home / "audit"
        if not base.is_dir():
            return []
        return sorted(path.name for path in base.iterdir() if path.is_dir() and HOST_RE.match(path.name)
                      and self.segments(path.name))

    def segments(self, host):
        directory = self.directory(host)
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.iterdir() if SEGMENT_RE.match(path.name))

    def _key_path(self, host):
        base = self.store.environ.get("CEVAHIR_AUDIT_KEY_DIR")
        if not base and self.store.environ.get("COLAB_RELEASE_TAG"):
            base = "/content/.cevahir-keys"          # never on the shared Drive home
        return (Path(base) if base else self.store.home / "trust" / "hmac") / f"{host}.bin"

    def key(self, host, create=False):
        """The host's audit MAC key, or None (a foreign chain: its key never leaves its machine)."""
        path = self._key_path(host)
        if not path.is_file():
            if not create:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(path, os.urandom(32))
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        return path.read_bytes()

    def _head_path(self, host):
        return self.directory(host) / "chain.json"

    def head(self, host):
        path = self._head_path(host)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _write_head(self, host, event, key):
        document = {"format": "cevahir-audit-chain", "version": 1, "host": host,
                    "segments": [path.name for path in self.segments(host)],
                    "head": {"seq": event["seq"], "hash": event["hash"], "seg": event["seg"]}}
        if key:
            document["mac"] = _mac(key, cjson(document).decode("utf-8"))
        atomic_write(self._head_path(host), json.dumps(document, indent=1, sort_keys=True).encode("utf-8") + b"\n")

    # ----------------------------------------------------------------------------------- append
    def _write_line(self, path, event, key):
        event["hash"] = event_hash(event)
        if key:
            event["mac"] = _mac(key, event["hash"])
        line = cjson(event) + b"\n"
        with _open_retry(path, "ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return event

    def _last_event(self, host, segments, key, rebuild_head=False):
        """The last event of the chain, after moving a torn tail aside. Refuses a damaged tail."""
        path = segments[-1]
        line, torn = _tail(path)
        repaired = None
        if torn:
            quarantine = self.directory(host) / "quarantine"
            quarantine.mkdir(exist_ok=True)
            digest = hashlib.sha256(torn).hexdigest()
            name = f"{path.stem}-{digest[:16]}.part"
            atomic_write(quarantine / name, torn)
            os.truncate(path, path.stat().st_size - len(torn))
            repaired = {"segment": path.name, "quarantined": name, "sha256": digest, "bytes": len(torn)}
        if line is None and len(segments) > 1:
            raise AuditError(f"audit segment {path.name} of {host} is empty: run Test-AuditLog")
        last = None
        if line is not None:
            try:
                last = json.loads(line)
                intact = isinstance(last, dict) and event_hash(last) == last.get("hash")
            except (ValueError, TypeError):
                intact = False
            if not intact:
                raise AuditError(f"the last event of the audit chain of {host} is damaged; nothing is appended to "
                                 "a chain that does not verify: run Test-AuditLog")
        head = self.head(host)
        last_seq = last["seq"] if last else 0
        if head is None and not rebuild_head:
            raise AuditError(f"the audit chain of {host} has no head record (chain.json): a cut at the end cannot "
                             "be ruled out; run Test-AuditLog, then Test-AuditLog -Repair")
        if head and head["head"]["seq"] > last_seq:
            raise AuditError(f"the audit chain of {host} ends at seq {last_seq} but its head records seq "
                             f"{head['head']['seq']}: events were removed from the end; run Test-AuditLog")
        return last, repaired

    def append(self, event, *, action=None, action_class=None, actor=None, targets=(), params=None, decision=None,
               outcome=None, lineage=None, data=None, _rebuild_head=False):
        """Append one event to this host's chain and return it (with ``seq``, ``hash`` and ``mac``).

        Refuses (``AuditError``) when the end of the chain does not verify: a damaged last event, a
        chain shorter than its head, or a missing head. Nothing is ever appended to hide a cut.
        """
        host = self.store.host_id(create=True)
        directory = self.directory(host)
        fields = {"event": str(event)}
        for name, value in (("action", action), ("class", action_class), ("actor", actor),
                            ("targets", list(targets) or None), ("params", params), ("decision", decision),
                            ("outcome", outcome), ("lineage", lineage), ("data", data)):
            if value is not None:
                fields[name] = sanitize(value)
        with self.store.lock(f"audit-{host}"):
            directory.mkdir(parents=True, exist_ok=True)
            segments = self.segments(host)
            key = self.key(host, create=not segments)       # a key is made once, with the chain's first event
            if key is None:
                raise AuditError(f"the audit key of {host} is missing ({self._key_path(host)}): its chain cannot "
                                 "be extended or checked without it; restore the key file")
            if not segments and self.head(host) is not None:
                raise AuditError(f"the audit chain of {host} has a head record but no segment files: the log was "
                                 "removed; run Test-AuditLog")
            last, repaired = self._last_event(host, segments, key, _rebuild_head) if segments else (None, None)

            def stamp(body, segment_index):
                previous = last
                return {"v": EVENT_VERSION, "seq": previous["seq"] + 1 if previous else 1, "ts": self.store.clock(),
                        "host": host, "seg": segment_index, "prev": previous["hash"] if previous else ZERO, **body}

            if last is None:
                path = directory / "seg-000001.jsonl"
                last = self._write_line(path, stamp({"event": "log.genesis", "data": {
                    "format": "cevahir-audit/1", "host": host}}, 1), key)
                segments = [path]
            path = segments[-1]
            index = int(SEGMENT_RE.match(path.name).group(1))
            if repaired:
                last = self._write_line(path, stamp({"event": "log.repaired", "data": repaired}, index), key)
            if path.stat().st_size >= self.segment_bytes:
                index += 1
                previous_name, path = path.name, directory / f"seg-{index:06d}.jsonl"
                last = self._write_line(path, stamp({"event": "log.segment", "data": {
                    "previous_segment": previous_name, "previous_hash": last["hash"]}}, index), key)
            last = self._write_line(path, stamp(fields, index), key)
            self._write_head(host, last, key)
        return last

    # ------------------------------------------------------------------------------------- read
    def events(self, host=None):
        """Events of one chain in order, as they parse (no verification: see ``verify``)."""
        host = host or self.store.host_id()
        if host is None:
            return
        for path in self.segments(host):
            with _open_retry(path, "rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        break
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict):
                        yield event

    def all_events(self):
        """Events of every chain in this home, merged by ``(ts, host, seq)``."""
        merged = [event for host in self.hosts() for event in self.events(host)]
        return sorted(merged, key=lambda event: (str(event.get("ts")), str(event.get("host")), event.get("seq") or 0))

    # ----------------------------------------------------------------------------------- verify
    def verify(self, host=None):
        """Recompute the whole chain of ``host``. Returns a verdict dict; ``ok`` is False on any finding."""
        host = host or self.store.host_id()
        if host is None or not self.segments(host):
            return dict(_empty_verdict(host), notes=["no audit chain yet: nothing has been recorded on this host"])
        segments = []
        for path in self.segments(host):
            with _open_retry(path, "rb") as stream:
                segments.append((path.name, stream.read()))
        verdict = verify_chain(host, segments, self.head(host), self.key(host))
        if host == self.store.host_id() and self.key(host) is None:
            # Without this, deleting the key and rewriting the chain without MACs would pass as "a foreign chain".
            verdict["findings"].append({"segment": "", "line": 0, "seq": None, "problem":
                                        "this host's audit key is missing: the MACs of its own chain cannot be checked"})
            verdict["ok"], verdict["mac"] = False, "failed"
        return verdict

    def repair(self):
        """Move a torn tail of this host's chain to quarantine and record it; rebuild a missing head.

        Only crash damage is repaired. An edited, removed or reordered event is evidence, not damage:
        it stays as it is and keeps failing verification.
        """
        host = self.store.host_id()
        if host is None or not self.segments(host):
            return None
        before = self.verify(host)
        problems = [item["problem"] for item in before["findings"]]
        repairable = all(text.startswith(("incomplete last line", "the head record (chain.json) is missing"))
                         for text in problems)
        if not problems or not repairable:
            return None
        return self.append("log.checked", data={"repaired": problems}, _rebuild_head=True)


def _empty_verdict(host):
    return {"host": host, "ok": True, "events": 0, "segments": 0, "first_bad_seq": None, "findings": [],
            "notes": [], "mac": "absent", "head": None, "last": None}


def _mac(key, text):
    kid = hashlib.sha256(key).hexdigest()[:16]
    return f"hmac-sha256:kid={kid}:" + hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_chain(host, segments, head, key=None):
    """Verify one chain given as ``[(segment name, bytes)]`` in order, its head record and (if held) its key.

    Findings name the segment, the line and (when the line parses) the ``seq``. After a finding the
    scan continues from what the file says, so one edit gives one or two findings, not one per
    remaining line. Without the key the MACs cannot be checked; everything else still is.
    """
    verdict = _empty_verdict(host)
    findings = verdict["findings"]

    def finding(problem, segment, line, seq=None):
        findings.append({"segment": segment, "line": line, "seq": seq, "problem": problem})
        if seq is not None and verdict["first_bad_seq"] is None:
            verdict["first_bad_seq"] = seq

    previous_hash, expected_seq, last = ZERO, 1, None
    mac_states = set()
    expected_index = 1
    for name, data in segments:
        index = int(SEGMENT_RE.match(name).group(1))
        verdict["segments"] += 1
        if index != expected_index:
            finding(f"segment {expected_index:06d} is missing", name, 0)
        expected_index = index + 1
        lines = data.split(b"\n")
        if lines[-1] != b"":
            finding(f"incomplete last line ({len(lines[-1])} bytes): a crash during a write, or a cut file; "
                    "Test-AuditLog -Repair moves it to quarantine", name, len(lines))
        for number, line in enumerate(lines[:-1], start=1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("not an object")
            except ValueError:
                finding("line is not a JSON object", name, number)
                continue
            seq = event.get("seq") if isinstance(event.get("seq"), int) else None
            verdict["events"] += 1
            try:
                canonical, recomputed = cjson(event), event_hash(event)
            except Exception:
                canonical, recomputed = None, None
            if recomputed != event.get("hash"):
                finding("hash mismatch: the event was edited", name, number, seq)
            elif canonical != line:
                finding("bytes are not the canonical form of the event: the line was rewritten", name, number, seq)
            if event.get("prev") != previous_hash:
                finding("broken link to the previous event: a line was removed, inserted or reordered", name, number, seq)
            if seq != expected_seq:
                finding(f"sequence number {seq} where {expected_seq} was expected", name, number, seq)
            if event.get("host") != host or event.get("seg") != index:
                finding("event belongs to another host or segment", name, number, seq)
            if number == 1 and index == 1 and event.get("event") != "log.genesis":
                finding("the chain does not start with log.genesis", name, number, seq)
            if number == 1 and index > 1 and event.get("event") != "log.segment":
                finding("the segment does not start with log.segment", name, number, seq)
            mac = event.get("mac")
            if key is None:
                mac_states.add("no-key" if mac else "absent")
            elif not mac:
                mac_states.add("failed")
                finding("event has no MAC although this host has an audit key", name, number, seq)
            elif not hmac.compare_digest(str(mac), _mac(key, str(event.get("hash")))):
                mac_states.add("failed")
                finding("MAC mismatch: the event was not written with this host's audit key", name, number, seq)
            else:
                mac_states.add("verified")
            previous_hash = event.get("hash")
            expected_seq = (seq if seq is not None else expected_seq) + 1
            last = event
    verdict["mac"] = "failed" if "failed" in mac_states else "verified" if mac_states == {"verified"} else \
        "no-key" if "no-key" in mac_states else "absent"
    if key is None:
        verdict["notes"].append("no audit key for this host here: MACs cannot be checked (a chain imported from "
                                "another machine)")
    verdict["head"] = head["head"] if head else None
    last_seq = last.get("seq") if last else 0
    verdict["last"] = {"seq": last_seq, "hash": last.get("hash")} if last else None
    if head is None:
        finding("the head record (chain.json) is missing: a cut at the end cannot be ruled out", "chain.json", 0)
    else:
        recorded = head["head"]
        if key is not None:
            body = {name: value for name, value in head.items() if name != "mac"}
            if not head.get("mac") or not hmac.compare_digest(head["mac"], _mac(key, cjson(body).decode("utf-8"))):
                finding("the head record does not carry this host's MAC", "chain.json", 0)
        if not isinstance(last_seq, int) or recorded["seq"] > last_seq:
            finding(f"the chain ends at seq {last_seq} but its head records seq {recorded['seq']}: "
                    "events were removed from the end", "chain.json", 0, last_seq)
        elif recorded["seq"] == last_seq and recorded["hash"] != last.get("hash"):
            finding("the head records another hash for the last event", "chain.json", 0, last_seq)
        elif recorded["seq"] < last_seq:
            verdict["notes"].append(f"the head record is behind the chain (seq {recorded['seq']} < {last_seq}): "
                                    "a crash between a write and the head update; the next append refreshes it")
    verdict["ok"] = not findings
    return verdict
