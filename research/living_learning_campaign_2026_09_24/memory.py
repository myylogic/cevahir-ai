"""Bounded memory/correction experiments; no learner sees scoring truth."""
from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

K = 16
CAPACITY = 96
WINDOW = 6
DECAY = 0.99
POLICIES = ("count", "fifo", "reservoir", "window", "last", "decay",
            "revision_delta", "revision_append", "revision_ignore", "revision_rebuild")


def hypotheses():
    return json.loads(Path(__file__).with_name("catalog_memory.json").read_text(encoding="utf-8"))


def vote(zero, one):
    return 1.0 if one > zero else 0.0 if zero > one else 0.5


def nested_tuple(value):
    return tuple(nested_tuple(x) for x in value) if isinstance(value, list) else value


class Memory:
    def __init__(self, policy, seed=0):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.policy = policy
        self.counts = [[0.0, 0.0] for _ in range(K)]
        self.records = []
        self.windows = [[] for _ in range(K)]
        self.latest = [None] * K
        self.events = {}
        self.seen = 0
        self.operations = 0
        self.rng = random.Random(seed)

    def predict(self, key):
        self.operations += 3  # two count reads, comparison
        if self.policy == "last":
            return 0.5 if self.latest[key] is None else float(self.latest[key])
        if self.policy == "revision_rebuild":
            # The strong equal-evidence comparator reconstructs all counts.
            fresh = [[0, 0] for _ in range(K)]
            self.operations += 2 * K
            for stored_key, label in self.events.values():
                fresh[stored_key][label] += 1
                self.operations += 3
            return vote(*fresh[key])
        return vote(*self.counts[key])

    def observe(self, event):
        # Public event is only (kind, id, key, observed_label).
        kind, eid, key, label = event
        self.operations += 4  # event decoding
        if kind == "revise":
            if self.policy in ("revision_delta", "revision_rebuild"):
                old_key, old_label = self.events[eid]
                if key != old_key:
                    raise ValueError("Revision may change a label, never its key")
                self.operations += 3  # id read and old tuple reads
                self.events[eid] = [key, label]
                self.operations += 3
                if self.policy == "revision_delta":
                    self.counts[key][old_label] -= 1
                    self.counts[key][label] += 1
                    self.operations += 4
            elif self.policy == "revision_append":
                self.counts[key][label] += 1
                self.operations += 2
            # Other policies deliberately cannot retract an identified old event.
            return
        if kind != "observe":
            raise ValueError(kind)
        self.seen += 1
        self.operations += 1
        if self.policy in ("revision_delta", "revision_rebuild"):
            self.events[eid] = [key, label]
            self.operations += 3
        if self.policy == "revision_rebuild":
            return
        if self.policy == "last":
            self.latest[key] = label
            self.operations += 1
            return
        if self.policy == "decay":
            for pair in self.counts:
                pair[0] *= DECAY
                pair[1] *= DECAY
            self.operations += 4 * K
        if self.policy in ("fifo", "reservoir"):
            keep = True
            if len(self.records) >= CAPACITY:
                if self.policy == "fifo":
                    position = 0
                else:
                    position = self.rng.randrange(self.seen)
                    self.operations += 1  # one supplied RNG primitive
                if position >= CAPACITY:
                    keep = False
                else:
                    old_key, old_label = self.records.pop(position)
                    self.counts[old_key][old_label] -= 1
                    self.operations += 4
                    # List movement is included, not treated as a free deletion.
                    self.operations += len(self.records) - position
            if not keep:
                return
            self.records.append([key, label])
            self.operations += 3
        elif self.policy == "window":
            window = self.windows[key]
            if len(window) >= WINDOW:
                old_label = window.pop(0)
                self.counts[key][old_label] -= 1
                self.operations += 3 + len(window)
            window.append(label)
            self.operations += 1
        self.counts[key][label] += 1
        self.operations += 2

    def memory_slots(self):
        # Live learner numeric/key fields, not Python bytes. RNG state's integers
        # are included for reservoir. Program and interpreter cost are excluded.
        if self.policy == "last":
            return K + 2
        if self.policy == "revision_rebuild":
            return 3 * len(self.events) + 2
        slots = 2 * K + 2
        if self.policy in ("fifo", "reservoir"):
            slots += 2 * len(self.records)
            if self.policy == "reservoir":
                slots += len(self.rng.getstate()[1]) + 2
        if self.policy == "window":
            slots += sum(map(len, self.windows)) + K
        if self.policy == "revision_delta":
            slots += 3 * len(self.events)
        return slots

    def snapshot(self):
        return {"policy": self.policy, "counts": self.counts, "records": self.records,
                "windows": self.windows, "latest": self.latest, "events": self.events,
                "seen": self.seen, "operations": self.operations,
                "rng_state": self.rng.getstate()}

    @classmethod
    def restore(cls, encoded):
        data = json.loads(encoded) if isinstance(encoded, str) else encoded
        obj = cls(data["policy"])
        for field in ("counts", "records", "windows", "latest", "seen", "operations"):
            setattr(obj, field, data[field])
        obj.events = {int(key): value for key, value in data["events"].items()}
        obj.rng.setstate(nested_tuple(data["rng_state"]))
        return obj


def make_stream(seed, regime):
    """Environment alone owns truth; predictors receive query key, then event."""
    rng = random.Random(seed * 1009 + 1907)
    base = [rng.randrange(2) for _ in range(K)]
    initial = []
    stream = []
    n = 512 if regime == "revisions" else 1024
    ids = list(range(256))
    rng.shuffle(ids)
    for t in range(n):
        query_key = rng.randrange(K)
        truth = base[query_key]
        revised_group = (query_key < 6 if regime == "revisions" else
                         query_key % 2 == 0 if regime == "abrupt" else
                         regime == "staggered")
        if regime == "abrupt" and t >= 512 and query_key % 2 == 0:
            truth ^= 1
        if regime == "staggered" and t >= 256 + query_key * 32:
            truth ^= 1
        if regime == "revisions" and t >= 256:
            eid = ids[t - 256]
            key = initial[eid][2]
            event = ("revise", eid, key, base[key])
        else:
            if regime == "clustered" and t >= 128:
                key = ((t - 128) // 56) % K
            elif regime == "rare" and t >= 128:
                key = rng.randrange(4) if rng.random() < 0.96 else rng.randrange(4, K)
            elif regime in ("clustered", "rare"):
                key = t % K
            else:
                key = rng.randrange(K)
            label = base[key]
            if regime == "abrupt" and t >= 512 and key % 2 == 0:
                label ^= 1
            if regime == "staggered" and t >= 256 + key * 32:
                label ^= 1
            noise = (0.85 if key < 6 else 0.10) if regime == "revisions" else 0.25
            if rng.random() < noise:
                label ^= 1
            event = ("observe", t, key, label)
            initial.append(event)
        stream.append({"query": query_key, "truth": truth, "event": event,
                       "group": "revised_keys" if revised_group else "other_keys",
                       "phase": "first" if t < n // 2 else "second"})
    return stream


def evaluate(stream, policy, seed):
    learner = Memory(policy, seed=seed + 424242)
    answer_digest = hashlib.sha256()
    loss = 0.0
    errors = ties = 0
    groups = {key: {"n": 0, "loss": 0.0, "errors": 0, "ties": 0}
              for key in ("revised_keys", "other_keys", "first", "second")}
    peak = learner.memory_slots()
    for row in stream:
        p = learner.predict(row["query"])
        answer_digest.update(bytes([int(2 * p)]))
        item_loss = (p - row["truth"]) ** 2
        wrong = int(p != 0.5 and int(p) != row["truth"])
        tie = int(p == 0.5)
        loss += item_loss
        errors += wrong
        ties += tie
        for key in (row["group"], row["phase"]):
            groups[key]["n"] += 1
            groups[key]["loss"] += item_loss
            groups[key]["errors"] += wrong
            groups[key]["ties"] += tie
        learner.observe(row["event"])
        peak = max(peak, learner.memory_slots())
    return {"brier": loss / len(stream), "cost": learner.operations,
            "cost_per_event": learner.operations / len(stream),
            "errors": errors, "ties": ties, "groups": groups,
            "memory_peak_slots": peak, "memory_final_slots": learner.memory_slots(),
            "n": len(stream), "answer_sha256": answer_digest.hexdigest()}


def run(seed):
    active = [h for h in hypotheses() if h["status"] == "wave1"]
    streams = {h["regime"]: make_stream(seed, h["regime"]) for h in active}
    results = {}
    cache = {}
    for h in active:
        sides = {}
        for side, policy in h["contrast"].items():
            key = (h["regime"], policy)
            if key not in cache:
                cache[key] = evaluate(streams[h["regime"]], policy, seed)
            sides[side] = cache[key]
        c, b = sides["candidate"], sides["control"]
        score = "cost_per_event" if h["metric"] == "virtual_operations_per_event" else "brier"
        results[h["id"]] = {
            "candidate": c[score], "control": b[score],
            "costs": {"candidate": c["cost"], "control": b["cost"]},
            "diagnostics": {"candidate": c, "control": b,
                            "same_stream": True, "truth_only_used_by_scorer": True}}
    delta = cache[("revisions", "revision_delta")]
    replay = cache[("revisions", "revision_rebuild")]
    return {"hypotheses": results,
            "checks": {"finite_bounded_errors": all(0 <= r["brier"] <= 1 for r in cache.values()),
                       "revision_delta_replay_equal_loss": delta["brier"] == replay["brier"],
                       "revision_delta_replay_equal_groups": delta["groups"] == replay["groups"],
                       "revision_delta_replay_equal_answers": delta["answer_sha256"] == replay["answer_sha256"]},
            "meta": {"module": "memory", "seed": seed, "keys_supplied": K,
                     "stream_regimes": len(streams), "paired_policy_runs": len(cache),
                     "evaluated_queries": sum(r["n"] for r in cache.values()),
                     "metric_note": "Brier against latent noiseless label; cost claim also records Brier",
                     "cost_note": "Declared virtual primitives, not time; no byte-equal memory claim"}}


def selfcheck():
    checks = 0
    details = []
    # Independent scalar reference recomputes counts from public evidence only.
    base_events = [("observe", i, (i * 7) % K, (i // 3) % 2) for i in range(120)]
    for policy in POLICIES:
        m = Memory(policy, 73)
        evidence = {}
        raw = []
        for t, event in enumerate(base_events):
            m.observe(event)
            evidence[event[1]] = [event[2], event[3]]
            raw.append(event)
            for key in range(K):
                if policy == "last":
                    seen = [e[3] for e in raw if e[2] == key]
                    expected = float(seen[-1]) if seen else 0.5
                elif policy in ("fifo", "reservoir"):
                    labels = [label for k, label in m.records if k == key]
                    expected = vote(labels.count(0), labels.count(1))
                elif policy == "window":
                    labels = [e[3] for e in raw if e[2] == key][-WINDOW:]
                    expected = vote(labels.count(0), labels.count(1))
                elif policy == "decay":
                    sums = [sum(DECAY ** (t - i) for i, e in enumerate(raw)
                                if e[2] == key and e[3] == label) for label in (0, 1)]
                    # Check real weights independently; avoid testing numerical ties.
                    assert all(abs(a - b) < 1e-10 for a, b in zip(sums, m.counts[key]))
                    expected = vote(*sums)
                else:
                    labels = [label for k, label in evidence.values() if k == key]
                    expected = vote(labels.count(0), labels.count(1))
                assert m.predict(key) == expected, (policy, t, key)
                checks += 1
        saved = json.dumps(m.snapshot())
        restored = Memory.restore(saved)
        for i in range(120, 190):
            key = (i * 11) % K
            assert m.predict(key) == restored.predict(key)
            event = ("observe", i, key, i % 2)
            m.observe(event)
            restored.observe(event)
            assert json.dumps(m.snapshot(), sort_keys=True) == json.dumps(restored.snapshot(), sort_keys=True)
            checks += 2
        details.append(policy + ": scalar checks and serialized continuation")
    # Retracting exactly the identified observation, not an arbitrary old label.
    delta, rebuild, ignore = Memory("revision_delta"), Memory("revision_rebuild"), Memory("revision_ignore")
    append = Memory("revision_append")
    appended_evidence = []
    independent = {}
    for e in base_events:
        for m in (delta, rebuild, ignore, append):
            m.observe(e)
        independent[e[1]] = [e[2], e[3]]
        appended_evidence.append([e[2], e[3]])
    before = [ignore.predict(k) for k in range(K)]
    for i in range(120):
        key, label = independent[i]
        e = ("revise", i, key, 1 - label)
        independent[i] = [key, 1 - label]
        appended_evidence.append([key, 1 - label])
        for m in (delta, rebuild, ignore, append):
            m.observe(e)
        for k in range(K):
            labels = [v for q, v in independent.values() if q == k]
            expected = vote(labels.count(0), labels.count(1))
            assert delta.predict(k) == rebuild.predict(k) == expected
            assert ignore.predict(k) == before[k]
            append_labels = [v for q, v in appended_evidence if q == k]
            assert append.predict(k) == vote(append_labels.count(0), append_labels.count(1))
            checks += 3
    # Removing corrections makes dependency bookkeeping observationally inert.
    plain = Memory("count")
    ablated = Memory("revision_delta")
    for e in base_events:
        plain.observe(e)
        ablated.observe(e)
        assert all(plain.predict(k) == ablated.predict(k) for k in range(K))
        checks += 1
    # Restore after revisions; event identifiers survive JSON key conversion.
    restored = Memory.restore(json.dumps(delta.snapshot()))
    for i in range(40):
        key, label = independent[i]
        event = ("revise", i, key, label)
        delta.observe(event)
        restored.observe(event)
        assert delta.snapshot() == restored.snapshot()
        checks += 1
    return {"checks": checks, "passed": True, "details": details + [
        "Independent label retraction reference and correction-disabled ablation",
        "No campaign seed or main-run performance used in selfcheck"]}


if __name__ == "__main__":
    print(json.dumps(selfcheck(), ensure_ascii=False, indent=2))
