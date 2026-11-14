"""Lineage: which keys produced which, projected from the audit events.

There is no second log. Every event that records an artefact (``artifact.registered``,
``artifact.derived``, ``artifact.imported``) carries ``lineage = {"output": key, "inputs":
[{"role", "key"}]}``, copied from the manifest it recorded. The edges are read back from every
chain in the home (this host's and imported ones), so lineage and audit cannot say different things;
``diverging`` checks the projection against the manifests.

An edge ``source -> target`` reads "source went into target" (tokenizer -> checkpoint -> model,
corpus -> cache). A key that is an alias of a registered artefact (a legacy tokenizer digest) is
shown as that artefact's key.
"""
from dataclasses import dataclass

DIRECTIONS = ("Up", "Down", "Both")
LINEAGE_EVENTS = ("artifact.registered", "artifact.derived", "artifact.imported")


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    role: str
    how: str             # the event that recorded it
    host: str
    seq: int
    ts: str


def edges(store):
    """Every lineage edge in the home, first record of each ``(source, target, role)`` kept."""
    from cevahir_ai.store.audit import AuditLog
    catalog = store.catalog()
    seen, found = set(), []
    for event in AuditLog(store).all_events():
        lineage = event.get("lineage")
        if event.get("event") not in LINEAGE_EVENTS or not isinstance(lineage, dict) or not lineage.get("output"):
            continue
        target = catalog.canonical(lineage["output"])
        for item in lineage.get("inputs") or ():
            source = catalog.canonical(item["key"])
            token = (source, target, item.get("role", ""))
            if token not in seen:
                seen.add(token)
                found.append(Edge(source, target, item.get("role", ""), event["event"], event.get("host", ""),
                                  event.get("seq", 0), event.get("ts", "")))
    return found


def walk(store, key, direction="Up", depth=None):
    """Edges reachable from ``key``: ``Up`` towards what it was made from, ``Down`` towards what was
    made from it. Returns ``[(level, direction, Edge)]``, breadth first, each edge once."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction: {direction!r}; expected one of {DIRECTIONS}")
    start = store.catalog().canonical(key)
    every = edges(store)
    result = []
    for way in (("Up", "Down") if direction == "Both" else (direction,)):
        frontier, visited, level = [start], {start}, 1
        while frontier and (depth is None or level <= depth):
            following = []
            for edge in every:
                near, far = (edge.target, edge.source) if way == "Up" else (edge.source, edge.target)
                if near in frontier:
                    result.append((level, way, edge))
                    if far not in visited:
                        visited.add(far)
                        following.append(far)
            frontier, level = following, level + 1
    return result


def diverging(store):
    """Manifest inputs that the audit projection does not contain (empty when they agree)."""
    catalog = store.catalog()
    projected = {(edge.source, edge.target, edge.role) for edge in edges(store)}
    missing = []
    for artifact in catalog.artifacts.values():
        for item in artifact.inputs:
            token = (catalog.canonical(item["key"]), artifact.key, item["role"])
            if token not in projected:
                missing.append(token)
    return sorted(missing)
