"""Information-boundary and falsification tests; stdlib only, no model loads."""

from dataclasses import replace
import itertools
import unittest
from unittest.mock import patch

from research.representation_discovery.witness_audit import (
    Context, Event, Guard, LimitReached, Limits, Meter, Model, Observation, Predicate, Representation,
    SearchResult, compare_case, find_witnesses, fixtures, language, overall_status, run_audit, search,
)


class WitnessAuditTests(unittest.TestCase):
    def search(self, rows, *, gated=False, candidates=None, limits=None):
        return search(tuple(rows), language() if candidates is None else candidates,
                      gated=gated, guard=Guard(limits or Limits()))

    def test_all_label_assignments_preserve_entire_compatible_set_and_tables(self):
        # Includes impossible deterministic labels and cases the grammar cannot fit.
        contexts = (
            Context((), "b"), Context((Event("a", True),), "b"),
            Context((Event("a", True), Event("c", True)), "b"),
            Context((Event("b", True),), "b"),
        )
        for labels in itertools.product((False, True), repeat=len(contexts)):
            rows = tuple(Observation(context, outcome) for context, outcome in zip(contexts, labels))
            for ordered in (rows, tuple(reversed(rows))):
                with self.subTest(labels=labels, reverse=ordered != rows):
                    generic = self.search(ordered)
                    gated = self.search(ordered, gated=True)
                    self.assertEqual((generic.status, gated.status), ("complete", "complete"))
                    self.assertEqual(generic.compatible, gated.compatible)

    def test_candidate_order_changes_selection_identically(self):
        rows = fixtures(1, ("a", "b", "c"))
        for candidates in (language(), tuple(reversed(language()))):
            generic = self.search(rows, candidates=candidates)
            gated = self.search(rows, gated=True, candidates=candidates)
            self.assertEqual(generic.compatible, gated.compatible)
            self.assertIsNotNone(generic.selected)

    def test_filter_alone_is_insufficient_and_full_check_rejects_false_candidate(self):
        # Earliest witness compares count=0 with count=2. count<1 separates them
        # but merges the later count=1 success with that count=2 failure.
        rows = (
            Observation(Context((), "c"), True),
            Observation(Context((Event("a", True), Event("b", True)), "c"), False),
            Observation(Context((Event("a", True),), "c"), True),
        )
        candidate = Representation(Predicate("active_count", "lt", 1))
        meter = Meter(Guard(Limits()))
        self.assertEqual(find_witnesses(rows, meter), ((0, 1),))
        self.assertNotEqual(candidate.key(rows[0].context, meter), candidate.key(rows[1].context, meter))
        result = self.search(rows, gated=True, candidates=(candidate,))
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.compatible, ())
        self.assertEqual(result.costs["witness_rejected"], 0)

    def test_identical_context_conflicting_labels_has_no_deterministic_solution(self):
        context = Context((), "b")
        rows = (Observation(context, True), Observation(context, False))
        for gated in (False, True):
            result = self.search(rows, gated=gated)
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.compatible, ())
            self.assertIsNone(result.selected)

    def test_unseen_key_is_abstention_and_counted_as_incorrect(self):
        train = (Observation(Context((), "a"), True),)
        heldout = (Observation(Context((Event("z", True),), "z"), False),)
        report = compare_case(train, heldout, candidates=language(), limits=Limits())
        self.assertEqual(report["comparison"], "equivalent")
        self.assertEqual(report["generic"]["heldout"]["abstentions"], 1)
        self.assertEqual(report["generic"]["heldout"]["accuracy_including_abstentions"], 0)

    def test_heldout_labels_cannot_change_selection_or_predictions(self):
        train = fixtures(2, ("a", "b", "c"))
        heldout = fixtures(2, ("x", "y", "z", "v", "w"))
        flipped = tuple(replace(row, outcome=not row.outcome) for row in heldout)
        original = compare_case(train, heldout, candidates=language(), limits=Limits())
        altered = compare_case(train, flipped, candidates=language(), limits=Limits())
        for method in ("generic", "gated"):
            self.assertEqual(original[method]["selected"], altered[method]["selected"])
            self.assertEqual(original[method]["compatible_representations"], altered[method]["compatible_representations"])
            self.assertEqual(original[method]["costs"], altered[method]["costs"])
            self.assertEqual(original[method]["heldout"]["accuracy_including_abstentions"], 1.0)
            self.assertEqual(altered[method]["heldout"]["accuracy_including_abstentions"], 0.0)

    def test_representation_is_invariant_to_bijective_object_renaming(self):
        context = Context((Event("a", True), Event("a", False), Event("b", True)), "b")
        renamed = Context((Event("x", True), Event("x", False), Event("z", True)), "z")
        meter = Meter(Guard(Limits()))
        for candidate in language():
            self.assertEqual(candidate.key(context, meter), candidate.key(renamed, meter))

    def test_base_representation_is_insufficient_in_shared_world(self):
        for capacity in (1, 2):
            rows = fixtures(capacity, ("a", "b", "c"))
            base = self.search(rows, candidates=(Representation(),))
            expanded = self.search(rows)
            self.assertEqual(base.compatible, ())
            self.assertIsNotNone(expanded.selected)

    def test_witness_construction_is_charged_even_without_contradictions(self):
        rows = fixtures(None, ("a", "b", "c"))
        generic = self.search(rows)
        gated = self.search(rows, gated=True)
        self.assertEqual(gated.witnesses, 0)
        self.assertEqual(gated.costs["witness_rows"], len(rows))
        self.assertGreater(gated.costs["steps"], generic.costs["steps"])

    def test_candidate_and_step_limits_never_return_partial_selection(self):
        rows = fixtures(1, ("a", "b", "c"))
        for limits in (Limits(candidates=1), Limits(search_steps=1), Limits(total_steps=1)):
            for gated in (False, True):
                result = self.search(rows, gated=gated, limits=limits)
                self.assertEqual(result.status, "inconclusive")
                self.assertEqual(result.compatible, ())
                self.assertIsNone(result.selected)
                self.assertIsNotNone(result.limit)

    def test_comparison_at_limit_cannot_claim_equivalence(self):
        rows = fixtures(None, ("a", "b", "c"))
        report = compare_case(rows, rows, candidates=language(), limits=Limits(candidates=1))
        self.assertEqual(report["comparison"], "inconclusive")
        self.assertNotIn("compatible_sets_and_tables_equal", report)

    def test_limits_and_grammar_reject_unbounded_inputs(self):
        with self.assertRaises(ValueError):
            Limits(seconds=float("inf"))
        with self.assertRaises(ValueError):
            Predicate("hidden_capacity", "eq", 1)
        with self.assertRaises(ValueError):
            fixtures(1, tuple(str(i) for i in range(20)))
        with self.assertRaises(ValueError):
            self.search(fixtures(1, ("a", "b", "c")), limits=Limits(observations=2))

    def test_known_table_mismatch_survives_later_evaluation_limit(self):
        rows = (Observation(Context((), "a"), True),)
        correct = Model(Representation(), (((False,), True),))
        faulty = Model(Representation(), (((False,), False),))
        with patch("research.representation_discovery.witness_audit.search", side_effect=(
                SearchResult("complete", (correct,), {}), SearchResult("complete", (faulty,), {}))):
            with patch.object(Model, "predict", side_effect=LimitReached("wall_time")):
                report = compare_case(rows, rows, candidates=language(), limits=Limits())
        self.assertEqual(report["comparison"], "mismatch")
        self.assertEqual(report["evaluation_limit"], "wall_time")

    def test_known_counterexample_survives_later_global_limit(self):
        with patch("research.representation_discovery.witness_audit.compare_case", side_effect=(
                {"comparison": "mismatch"}, LimitReached("wall_time"))):
            report = run_audit()
        self.assertEqual(report["status"], "mismatch")
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(report["limit"], "wall_time")
        self.assertEqual(overall_status(["mismatch", "inconclusive"]), "mismatch")

    def test_expired_wall_limit_and_traced_memory_limit_are_inconclusive(self):
        rows = (Observation(Context((), "a"), True),)
        guard = Guard(Limits())
        guard.started -= 61
        self.assertEqual(search(rows, language(), gated=False, guard=guard).limit, "wall_time")
        with patch("research.representation_discovery.witness_audit.tracemalloc.is_tracing", return_value=True):
            with patch("research.representation_discovery.witness_audit.tracemalloc.get_traced_memory", return_value=(100, 100)):
                result = search(rows, language(), gated=False, guard=Guard(Limits(traced_bytes=1)))
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.limit, "traced_python_memory")


if __name__ == "__main__":
    unittest.main()
