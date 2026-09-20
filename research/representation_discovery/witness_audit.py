"""Falsification audit for witness-gated deterministic representation search.

Run: python -m research.representation_discovery.witness_audit --output PATH

Both searches receive the same ordered representation language and observations.
The generic baseline stops checking each candidate on its first contradiction.
The gated method first checks genuine training contradictions of the base
representation, then uses that same baseline check. Filtering is a necessary
condition, not an additional source of information. Completed searches must
therefore return exactly the same compatible representations and fitted tables.

This is a bounded, synthetic equivalence audit, not a benchmark of general
concept discovery, causal identification, or Cevahir's trained model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import time
import tracemalloc


@dataclass(frozen=True)
class Event:
    target: str
    succeeded: bool


@dataclass(frozen=True)
class Context:
    history: tuple[Event, ...]
    target: str


@dataclass(frozen=True)
class Observation:
    context: Context
    outcome: bool


class LimitReached(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    candidates: int = 64
    observations: int = 256
    history: int = 8
    search_steps: int = 100_000
    total_steps: int = 1_000_000
    seconds: float = 10.0
    traced_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for name in ("candidates", "observations", "history", "search_steps",
                     "total_steps", "traced_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seconds, (int, float)) or not 0 < self.seconds <= 60:
            raise ValueError("seconds must be positive and at most 60")


class Guard:
    """Cooperative wall/work limits; optional traced-Python-memory monitoring.

    Memory monitoring is not an OS RSS cap. Work/cardinality bounds constrain
    this fixed synthetic workload independently of tracemalloc availability.
    """

    def __init__(self, limits: Limits):
        self.limits = limits
        self.started = time.monotonic()
        self.steps = 0

    def charge(self, steps: int = 0):
        self.steps += steps
        if self.steps > self.limits.total_steps:
            raise LimitReached("total_steps")
        if time.monotonic() - self.started > self.limits.seconds:
            raise LimitReached("wall_time")
        if tracemalloc.is_tracing():
            if tracemalloc.get_traced_memory()[1] > self.limits.traced_bytes:
                raise LimitReached("traced_python_memory")


@dataclass
class Meter:
    guard: Guard
    steps: int = 0
    candidates_considered: int = 0
    representation_calls: int = 0
    witness_rows: int = 0
    witness_pairs_checked: int = 0
    witness_rejected: int = 0
    consistency_rows: int = 0

    def charge(self, steps: int):
        self.steps += steps
        self.guard.charge(steps)
        if self.steps > self.guard.limits.search_steps:
            raise LimitReached("search_steps")

    def report(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__
                if name != "guard"}


def active_targets(context: Context, meter: Meter) -> set[str]:
    meter.charge(1 + len(context.history))
    return {event.target for event in context.history if event.succeeded}


def base_key(context: Context, meter: Meter) -> bool:
    return context.target in active_targets(context, meter)


@dataclass(frozen=True)
class Predicate:
    term: str
    relation: str
    threshold: int

    def __post_init__(self):
        if self.term not in ("active_count", "success_count", "failure_count", "attempt_count"):
            raise ValueError("unknown term")
        if self.relation not in ("lt", "eq"):
            raise ValueError("unknown relation")
        if type(self.threshold) is not int or self.threshold not in (0, 1, 2):
            raise ValueError("threshold outside the fixed grammar")

    def evaluate(self, context: Context, meter: Meter) -> bool:
        meter.charge(2)
        if self.term == "active_count":
            value = len(active_targets(context, meter))
        else:
            meter.charge(1 + len(context.history))
            if self.term == "success_count":
                value = sum(event.succeeded for event in context.history)
            elif self.term == "failure_count":
                value = sum(not event.succeeded for event in context.history)
            else:
                value = len(context.history)
        return value < self.threshold if self.relation == "lt" else value == self.threshold


@dataclass(frozen=True)
class Representation:
    predicate: Predicate | None = None

    @property
    def name(self) -> str:
        if self.predicate is None:
            return "target_active"
        p = self.predicate
        return f"target_active+({p.term} {p.relation} {p.threshold})"

    def key(self, context: Context, meter: Meter) -> tuple[bool, ...]:
        meter.representation_calls += 1
        base = base_key(context, meter)
        return (base,) if self.predicate is None else (base, self.predicate.evaluate(context, meter))


def language() -> tuple[Representation, ...]:
    # General observable counters, with the same fixed order for both searches.
    # No world label, object name, hidden capacity, or evaluator feedback enters.
    return (Representation(),) + tuple(
        Representation(Predicate(term, relation, threshold))
        for term in ("active_count", "success_count", "failure_count", "attempt_count")
        for relation in ("lt", "eq") for threshold in (0, 1, 2)
    )


@dataclass(frozen=True)
class Model:
    representation: Representation
    table: tuple[tuple[tuple[bool, ...], bool], ...]

    def predict(self, context: Context, meter: Meter) -> bool | None:
        # An unseen key is an abstention, never a guessed default label.
        return dict(self.table).get(self.representation.key(context, meter))


@dataclass(frozen=True)
class SearchResult:
    status: str
    compatible: tuple[Model, ...]
    costs: dict
    witnesses: int = 0
    limit: str | None = None

    @property
    def selected(self) -> Model | None:
        return self.compatible[0] if self.status == "complete" and self.compatible else None


def validate_observations(rows: tuple[Observation, ...], limits: Limits):
    if not rows or len(rows) > limits.observations:
        raise ValueError("observation count outside limits")
    for row in rows:
        if type(row.outcome) is not bool or len(row.context.history) > limits.history:
            raise ValueError("invalid outcome or excessive history")
        if not isinstance(row.context.target, str) or not 0 < len(row.context.target) <= 64:
            raise ValueError("invalid target")
        for event in row.context.history:
            if type(event.succeeded) is not bool or not isinstance(event.target, str) or not 0 < len(event.target) <= 64:
                raise ValueError("invalid event")


def find_witnesses(rows: tuple[Observation, ...], meter: Meter) -> tuple[tuple[int, int], ...]:
    """At most one real opposite-outcome pair per base key, training data only.

    This small subset is necessary but generally NOT sufficient for consistency.
    All setup work is charged to the gated method; no free witness oracle.
    """
    representatives = {}
    found = {}
    for index, row in enumerate(rows):
        meter.witness_rows += 1
        meter.charge(1)
        key = base_key(row.context, meter)
        opposite = representatives.get((key, not row.outcome))
        if opposite is not None and key not in found:
            found[key] = (opposite, index)
        representatives.setdefault((key, row.outcome), index)
    return tuple(found.values())


def search(rows: tuple[Observation, ...], candidates: tuple[Representation, ...],
           *, gated: bool, guard: Guard) -> SearchResult:
    validate_observations(rows, guard.limits)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be nonempty and unique")
    meter = Meter(guard)
    witnesses = ()
    compatible = []
    try:
        guard.charge()
        witnesses = find_witnesses(rows, meter) if gated else ()
        for candidate in candidates:
            if meter.candidates_considered >= guard.limits.candidates:
                raise LimitReached("candidates")
            meter.candidates_considered += 1
            rejected = False
            for left, right in witnesses:
                meter.witness_pairs_checked += 1
                meter.charge(1)
                if candidate.key(rows[left].context, meter) == candidate.key(rows[right].context, meter):
                    meter.witness_rejected += 1
                    rejected = True
                    break
            if rejected:
                continue
            table = {}
            for row in rows:
                meter.consistency_rows += 1
                meter.charge(1)
                key = candidate.key(row.context, meter)
                if key in table and table[key] != row.outcome:
                    rejected = True
                    break
                table[key] = row.outcome
            if not rejected:
                compatible.append(Model(candidate, tuple(sorted(table.items()))))
        guard.charge()
    except LimitReached as exc:
        # A prefix of the candidate set cannot establish completed equivalence.
        return SearchResult("inconclusive", (), meter.report(), len(witnesses), str(exc))
    return SearchResult("complete", tuple(compatible), meter.report(), len(witnesses))


def fixtures(capacity: int | None, names: tuple[str, ...], depth: int = 2) -> tuple[Observation, ...]:
    """Private synthetic oracle. Only its observable histories reach search().

    Each history is a separate reset episode; each successful claim holds one
    unit until reset. None means independent capacity per object. This grammar
    and these worlds are deliberately tiny and are not a general ontology test.
    """
    if capacity is not None and (type(capacity) is not int or capacity < 1):
        raise ValueError("invalid capacity")
    if not 2 <= len(names) <= 6 or len(set(names)) != len(names) or not 0 <= depth <= 2:
        raise ValueError("fixture outside the tiny workload bounds")
    if sum(len(names) ** size for size in range(depth + 1)) * len(names) > 256:
        raise ValueError("too many fixture observations")
    rows = []
    for size in range(depth + 1):
        for actions in itertools.product(names, repeat=size):
            active = set()
            history = []
            for target in actions:
                ok = target not in active and (capacity is None or len(active) < capacity)
                history.append(Event(target, ok))
                if ok:
                    active.add(target)
            for target in names:
                ok = target not in active and (capacity is None or len(active) < capacity)
                rows.append(Observation(Context(tuple(history), target), ok))
    return tuple(rows)


def score_predictions(predictions: tuple[bool | None, ...], rows: tuple[Observation, ...]) -> dict:
    if len(predictions) != len(rows) or not rows:
        raise ValueError("prediction/observation length mismatch")
    correct = sum(pred is not None and pred == row.outcome for pred, row in zip(predictions, rows))
    answered = sum(pred is not None for pred in predictions)
    return {"observations": len(rows), "correct": correct, "abstentions": len(rows) - answered,
            "accuracy_including_abstentions": correct / len(rows), "coverage": answered / len(rows)}


def compare_case(train: tuple[Observation, ...], heldout: tuple[Observation, ...],
                 *, candidates: tuple[Representation, ...], limits: Limits,
                 guard: Guard | None = None) -> dict:
    validate_observations(heldout, limits)
    guard = guard or Guard(limits)
    generic = search(train, candidates, gated=False, guard=guard)
    gated = search(train, candidates, gated=True, guard=guard)
    result = {
        "training_observations": len(train),
        "generic": {"status": generic.status, "limit": generic.limit, "costs": generic.costs},
        "gated": {"status": gated.status, "limit": gated.limit, "costs": gated.costs,
                  "witness_count": gated.witnesses},
    }
    if generic.status != "complete" or gated.status != "complete":
        result["comparison"] = "inconclusive"
        return result
    equal = generic.compatible == gated.compatible
    result["compatible_sets_and_tables_equal"] = equal
    for name, search_result in (("generic", generic), ("gated", gated)):
        result[name]["compatible_representations"] = [m.representation.name for m in search_result.compatible]
        result[name]["selected"] = search_result.selected.representation.name if search_result.selected else None
    # Freeze all predictions before evaluator labels are used for scoring.
    predictions = []
    evaluation_meter = Meter(guard)
    try:
        for search_result in (generic, gated):
            selected = search_result.selected
            predictions.append(tuple(selected.predict(row.context, evaluation_meter)
                                     if selected else None for row in heldout))
    except LimitReached as exc:
        # A proved counterexample remains a counterexample when later work stops.
        result.update(comparison="inconclusive" if equal else "mismatch", evaluation_limit=str(exc))
        return result
    result["predictions_and_abstentions_equal"] = predictions[0] == predictions[1]
    for name, values in zip(("generic", "gated"), predictions):
        result[name]["heldout"] = score_predictions(values, heldout)
    result["comparison"] = "equivalent" if equal and predictions[0] == predictions[1] else "mismatch"
    return result


def overall_status(comparisons: list[str], *, interrupted: bool = False) -> str:
    if "mismatch" in comparisons:
        return "mismatch"
    if interrupted or not comparisons or "inconclusive" in comparisons:
        return "inconclusive"
    return "equivalent"


def run_audit(limits: Limits = Limits()) -> dict:
    owns_tracer = not tracemalloc.is_tracing()
    if owns_tracer:
        tracemalloc.start()
    guard = Guard(limits)
    report = {"schema_version": 1, "kind": "synthetic_representation_search_equivalence_audit",
              "limits": asdict(limits), "worlds": [], "candidate_count": len(language()),
              "empirical_model_quality_validated": False, "scientific_novelty_established": False,
              "training_or_model_execution": False,
              "scope": "Fixed 25-candidate language; deterministic claim worlds; no active query policy.",
              "memory_measure": "Peak traced Python allocations; not total process RAM or an OS cap.",
              "tracing_scope": "this_audit" if owns_tracer else "existing_session_including_prior_work",
              "cost_scope": "Each prefix is independently refitted. Steps count representation work and row/pair checks; they are not CPU instructions or measured speed. The shared total/time guard is an audit watchdog, not a per-method performance budget. Limited comparisons cannot establish a method advantage.",
              "observation_scope": "Counts labeled query contexts, including supplied past action outcomes; not physical environment interactions."}
    try:
        for label, capacity in (("independent", None), ("shared_capacity_1", 1), ("shared_capacity_2", 2)):
            guard.charge()
            train = fixtures(capacity, ("train_17", "train_03", "train_91"))
            heldout = fixtures(capacity, ("eval_42", "eval_08", "eval_63", "eval_25", "eval_79"))
            world = {"world": label, "train_objects": 3, "heldout_objects": 5,
                     "prefixes": []}
            report["worlds"].append(world)
            for size in (3, 6, 12, 24, len(train)):
                world["prefixes"].append(compare_case(train[:size], heldout, candidates=language(),
                                                       limits=limits, guard=guard))
                guard.charge()
        comparisons = [p["comparison"] for w in report["worlds"] for p in w["prefixes"]]
        report["status"] = overall_status(comparisons)
    except LimitReached as exc:
        comparisons = [p["comparison"] for w in report["worlds"] for p in w["prefixes"]]
        report.update(status=overall_status(comparisons, interrupted=True), limit=str(exc))
    finally:
        report["coverage_complete"] = (
            len(report["worlds"]) == 3
            and all(len(w["prefixes"]) == 5 and all(
                p["comparison"] != "inconclusive" and "evaluation_limit" not in p
                for p in w["prefixes"]) for w in report["worlds"])
            and "limit" not in report)
        report["resources"] = {"seconds": time.monotonic() - guard.started,
                               "accounted_steps": guard.steps,
                               "peak_traced_python_bytes": tracemalloc.get_traced_memory()[1]}
        if owns_tracer:
            tracemalloc.stop()
    report["interpretation"] = {
        "equivalent": "No observational advantage in this completed-search formulation. A necessary-condition witness filter preserves the compatible set. Costs include witness construction.",
        "inconclusive": "A resource limit prevented complete comparison. Do not interpret partial results as a win, failure, or equivalence.",
        "mismatch": "The necessary-filter invariant failed; investigate the harness before claiming a research gain.",
    }[report["status"]]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "resources": report["resources"], "output": str(args.output)}))
    return 0 if report["status"] == "equivalent" else 2


if __name__ == "__main__":
    raise SystemExit(main())
