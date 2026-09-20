"""Cheap experience-learning contract tests: stdlib only, no models or training.

Run with ``python -m unittest discover -s tests/evolution
-p test_research_experience.py``. Fixture outcomes are synthetic contract probes,
not evidence of real model quality or a validated causal routing improvement.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import unittest
from unittest.mock import patch

from cognitive_management.research.experience import ExperienceStore


IDENTITY = {"model_revision": "model-a", "tokenizer_revision": "tokens-a"}


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperienceStore()
        self.sequence = 0

    def observe(self, strategy="direct", *, store=None, **changes):
        self.sequence += 1
        fields = dict(scope="alice/session-1", identity=IDENTITY, task_key="math/low/budget-v1",
                      planned_strategy=strategy, executed_strategy=strategy, status="observed",
                      cost=0, source_ids=("source-1",), request_id=f"request-{self.sequence}",
                      sample_id=f"independent-input-{self.sequence}")
        fields.update(changes)
        return (store or self.store).observe(**fields)

    def feedback(self, experience_id, value, *, store=None, **changes):
        self.sequence += 1
        fields = dict(experience_id=experience_id, scope="alice/session-1", identity=IDENTITY,
                      value=value, source="evaluator", event_id=f"evaluation-{self.sequence}")
        fields.update(changes)
        return (store or self.store).feedback(**fields)

    def recommendation(self, *, store=None, **changes):
        fields = dict(scope="alice/session-1", identity=IDENTITY, task_key="math/low/budget-v1",
                      baseline="direct", allowed=("direct", "think1"))
        fields.update(changes)
        return (store or self.store).recommend(**fields)

    def examples(self, strategy, value, count=3, *, store=None, **changes):
        ids = []
        for _ in range(count):
            experience_id = self.observe(strategy, store=store, **changes)
            self.feedback(experience_id, value, store=store)
            ids.append(experience_id)
        return ids

    def contrasted_examples(self, *, store=None):
        baseline = self.examples("direct", 0, store=store)
        candidate = self.examples("think1", 1, store=store)
        return baseline, candidate

    def test_observations_do_not_self_verify(self):
        for strategy in ("direct", "think1"):
            for _ in range(10):
                self.observe(strategy)
        result = self.recommendation()
        self.assertEqual(result["strategy"], "direct")
        self.assertEqual(result["support"], 0)
        self.assertEqual(result["scores"]["think1"]["support"], 0)

    def test_explicit_verified_independent_support_can_change_preference(self):
        self.contrasted_examples()
        result = self.recommendation()
        self.assertEqual(result["strategy"], "think1")
        self.assertEqual(result["support"], 3)
        self.assertEqual(result["reason"], "supported_observed_preference")
        self.assertGreater(result["scores"]["think1"]["lower"], result["scores"]["direct"]["upper"])

    def test_baseline_and_candidate_both_require_independent_support(self):
        self.examples("think1", 1, 6)
        self.examples("direct", 0, 2)
        self.assertEqual(self.recommendation()["strategy"], "direct")
        self.feedback(self.observe("direct"), 0)
        self.assertEqual(self.recommendation()["strategy"], "think1")

    def test_no_counterfactual_support_for_planned_but_unexecuted_route(self):
        self.examples("direct", 0)
        self.examples("direct", 1, planned_strategy="think1")
        self.assertEqual(self.recommendation()["scores"]["think1"]["support"], 0)

    def test_repeated_sample_and_feedback_sources_do_not_inflate_support(self):
        for index in range(8):
            experience_id = self.observe("think1", sample_id="same-prompt")
            for source in ("evaluator", "tool_verified", "user"):
                self.feedback(experience_id, 1, source=source)
        self.examples("direct", 0)
        result = self.recommendation()
        self.assertEqual(result["scores"]["think1"]["support"], 1)
        self.assertEqual(result["strategy"], "direct")

    def test_missing_sample_identity_fails_conservatively(self):
        self.examples("direct", 0, sample_id=None)
        self.examples("think1", 1, sample_id=None)
        result = self.recommendation()
        self.assertEqual(result["scores"]["direct"]["support"], 1)
        self.assertEqual(result["strategy"], "direct")

    def test_latest_unverified_repeat_retracts_sample_support(self):
        self.examples("direct", 0)
        self.examples("think1", 1, 2)
        self.feedback(self.observe("think1", sample_id="repeat"), 1)
        self.assertEqual(self.recommendation()["strategy"], "think1")
        self.observe("think1", sample_id="repeat")
        self.assertEqual(self.recommendation()["strategy"], "direct")

    def test_scope_identity_task_and_allowed_actions_are_isolated(self):
        self.contrasted_examples()
        for changes in ({"scope": "bob/session-1"}, {"task_key": "different-budget"},
                        {"identity": {**IDENTITY, "model_revision": "model-b"}},
                        {"identity": {**IDENTITY, "tokenizer_revision": "tokens-b"}}):
            with self.subTest(changes=changes):
                self.assertEqual(self.recommendation(**changes)["support"], 0)
        restricted = self.recommendation(allowed=("direct",))
        self.assertEqual(restricted["strategy"], "direct")
        self.assertNotIn("think1", restricted["scores"])
        with self.assertRaises(ValueError):
            self.recommendation(allowed=("think1",))

    def test_foreign_feedback_and_critic_sources_are_rejected(self):
        experience_id = self.observe()
        for changes in ({"scope": "bob/session-1"}, {"source": "critic"},
                        {"identity": {**IDENTITY, "model_revision": "model-b"}},
                        {"identity": {**IDENTITY, "tokenizer_revision": "tokens-b"}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.feedback(experience_id, 1, **changes)
        self.assertEqual(self.store.revision, 1)

    def test_feedback_replay_correction_and_supersession_provenance(self):
        _, candidate = self.contrasted_examples()
        experience_id = candidate[0]
        original_quality = self.recommendation()["scores"]["think1"]["mean_quality"]
        self.feedback(experience_id, 0, source="user", event_id="bad")
        # One corrected observation reduces aggregate utility; it need not erase
        # a preference still supported by the other independently observed tasks.
        self.assertLess(self.recommendation()["scores"]["think1"]["mean_quality"], original_quality)
        self.feedback(experience_id, 1, source="user", event_id="corrected")
        self.assertAlmostEqual(self.recommendation()["scores"]["think1"]["mean_quality"], original_quality)
        revision = self.store.revision
        self.assertFalse(self.feedback(experience_id, 0, source="user", event_id="bad"))
        self.assertEqual(self.store.revision, revision)
        self.assertEqual(self.recommendation()["strategy"], "think1")
        with self.assertRaises(ValueError):
            self.feedback(experience_id, 1, source="user", event_id="bad")
        row = next(r for r in self.store.snapshot()["records"] if r["experience_id"] == experience_id)
        self.assertEqual(row["feedback"][-1]["supersedes"], "bad")

    def test_feedback_sources_do_not_overrule_an_uncorrected_negative(self):
        _, candidate = self.contrasted_examples()
        for experience_id in candidate:
            self.feedback(experience_id, 0, source="user")
            self.feedback(experience_id, 1, source="tool_verified")
        self.assertEqual(self.recommendation()["strategy"], "direct")

    def test_failed_limited_and_cached_outcomes_cannot_supply_positive_support(self):
        self.examples("direct", 0)
        for status in ("error", "budget_limited", "cache"):
            self.examples("think1", 1, status=status)
        result = self.recommendation()
        self.assertEqual(result["scores"]["think1"]["support"], 0)
        self.assertEqual(result["strategy"], "direct")

    def test_negative_excluded_status_only_vetoes_until_corrected_or_forgotten(self):
        self.contrasted_examples()
        experience_id = self.observe("think1", status="budget_limited")
        self.feedback(experience_id, 0)
        result = self.recommendation()
        self.assertEqual(result["strategy"], "direct")
        self.assertEqual(result["scores"]["think1"]["support"], 3)
        self.assertEqual(result["scores"]["think1"]["negative_evidence"], 1)
        self.feedback(experience_id, 1)
        self.assertEqual(self.recommendation()["strategy"], "think1")
        self.feedback(experience_id, 0)
        self.store.forget(scope="alice/session-1", experience_id=experience_id)
        self.assertEqual(self.recommendation()["strategy"], "think1")

    def test_cost_penalty_uses_executed_observations(self):
        store = ExperienceStore(cost_weight=1)
        self.examples("direct", 0.9, 4, store=store, cost=9)
        self.examples("think1", 0.9, 4, store=store, cost=0)
        self.assertEqual(self.recommendation(store=store)["strategy"], "think1")
        expensive = ExperienceStore(cost_weight=1)
        self.examples("direct", 0.5, 4, store=expensive, cost=0)
        self.examples("think1", 1, 4, store=expensive, cost=99)
        self.assertEqual(self.recommendation(store=expensive)["strategy"], "direct")

    def test_close_quality_estimates_stay_with_baseline(self):
        self.examples("direct", 0.7)
        self.examples("think1", 0.8)
        self.assertEqual(self.recommendation()["strategy"], "direct")

    def test_old_evidence_decays_and_cannot_create_a_confident_preference(self):
        with patch("cognitive_management.research.experience.time.time", return_value=1000):
            self.contrasted_examples()
            self.assertEqual(self.recommendation()["strategy"], "think1")
        with patch("cognitive_management.research.experience.time.time", return_value=1000 + 604800 * 30):
            result = self.recommendation()
            self.assertEqual(result["strategy"], "direct")
            self.assertEqual(result["scores"]["think1"]["support"], 3)
            self.assertLess(result["scores"]["think1"]["effective_support"], 0.001)

    def test_request_retries_are_idempotent_but_conflicts_rejected(self):
        experience_id = self.observe(request_id="request-replay", sample_id="sample")
        revision = self.store.revision
        self.assertEqual(self.observe(request_id="request-replay", sample_id="sample"), experience_id)
        self.assertEqual(self.store.revision, revision)
        with self.assertRaises(ValueError):
            self.observe(request_id="request-replay", sample_id="changed")

    def test_event_replay_cannot_migrate_to_a_different_observation(self):
        first, second = self.observe(), self.observe()
        self.feedback(first, 1, event_id="shared-event")
        with self.assertRaises(ValueError):
            self.feedback(second, 1, event_id="shared-event")

    def test_forget_retracts_support_and_all_feedback_and_bumps_revision(self):
        _, candidates = self.contrasted_examples()
        revision = self.store.revision
        self.assertEqual(self.store.forget(scope="bob", experience_id=candidates[0]), 0)
        self.assertEqual(self.store.revision, revision)
        self.assertEqual(self.store.forget(scope="alice/session-1", experience_id=candidates[0]), 1)
        self.assertGreater(self.store.revision, revision)
        self.assertEqual(self.recommendation()["strategy"], "direct")
        with self.assertRaises(ValueError):
            self.feedback(candidates[0], 1)
        self.assertEqual(self.store.forget(scope="alice/session-1"), 5)
        self.assertEqual(self.store.snapshot()["records"], [])

    def test_scope_record_and_feedback_history_are_bounded(self):
        store = ExperienceStore(max_records_per_scope=2, min_support=1, max_scopes=1,
                                max_feedback_events_per_record=2)
        first = self.observe(store=store)
        second = self.observe(store=store)
        self.feedback(first, 1, store=store, event_id="evicted-event")
        third = self.observe(store=store)
        self.assertEqual([r["experience_id"] for r in store.snapshot()["records"]], [second, third])
        with self.assertRaises(ValueError):
            self.feedback(first, 1, store=store)
        with self.assertRaises(ValueError):
            self.observe(store=store, scope="new-scope")
        self.feedback(third, 1, store=store, event_id="one")
        self.feedback(third, 0, store=store, event_id="two")
        self.assertFalse(self.feedback(third, 1, store=store, event_id="one"))
        with self.assertRaises(ValueError):
            self.feedback(third, 1, store=store, event_id="three")
        store.forget(scope="alice/session-1")
        self.observe(store=store, scope="new-scope")
        self.assertEqual(len(store.snapshot()["records"]), 1)

    def test_records_are_frozen_and_snapshot_does_not_contain_raw_task_or_sample(self):
        experience_id = self.observe(task_key="private task bucket", sample_id="private sample")
        record = self.store._records[experience_id]
        with self.assertRaises(FrozenInstanceError):
            record.cost = 8
        snapshot = self.store.snapshot()
        serialized = json.dumps(snapshot, allow_nan=False)
        self.assertNotIn("private task bucket", serialized)
        self.assertNotIn("private sample", serialized)
        self.assertNotIn("prompt", snapshot["records"][0])
        snapshot["records"][0]["cost"] = 999
        self.assertEqual(self.store.snapshot()["records"][0]["cost"], 0)

    def test_snapshot_roundtrip_preserves_bounds_provenance_scores_and_replay(self):
        source = ExperienceStore(max_records_per_scope=20, max_scopes=2, min_support=3,
                                 max_feedback_events_per_record=4, cost_weight=0.2,
                                 decay_half_life_seconds=100)
        with patch("cognitive_management.research.experience.time.time", return_value=1000):
            _, candidates = self.contrasted_examples(store=source)
            self.feedback(candidates[0], 0, store=source, event_id="old")
            self.feedback(candidates[0], 1, store=source, event_id="correction")
            snapshot = json.loads(json.dumps(source.snapshot(), allow_nan=False))
            target = ExperienceStore()
            target.restore(snapshot)
            self.assertEqual(target.snapshot()["config"], source.snapshot()["config"])
            self.assertEqual(target.snapshot()["records"], source.snapshot()["records"])
            self.assertEqual(self.recommendation(store=target)["scores"], self.recommendation(store=source)["scores"])
            self.assertFalse(self.feedback(candidates[0], 0, store=target, event_id="old"))
            self.assertEqual(self.recommendation(store=target)["strategy"], "think1")
            revision = target.revision
            target.restore(snapshot)
            self.assertGreater(target.revision, revision)

    def test_invalid_snapshots_are_rejected_atomically(self):
        experience_id = self.observe()
        self.feedback(experience_id, 1)
        good = self.store.snapshot()
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(schema_version=999),
            lambda d: d.update(revision=-1),
            lambda d: d.update(extra="not-in-schema"),
            lambda d: d["config"].update(max_scopes=0),
            lambda d: d["config"].update(min_support=10000),
            lambda d: d["records"].append(deepcopy(d["records"][0])),
            lambda d: d["records"][0].update(cost=float("nan")),
            lambda d: d["records"][0].update(status="verified"),
            lambda d: d["records"][0].update(task_fingerprint="raw text"),
            lambda d: d["records"][0].update(identity={"model_revision": "missing tokenizer"}),
            lambda d: d["records"][0].update(source_ids=["duplicate", "duplicate"]),
            lambda d: d["records"][0]["feedback"][0].update(source="critic"),
            lambda d: d["records"][0]["feedback"][0].update(value=2),
            lambda d: d["records"][0]["feedback"][0].update(created_at=0),
            lambda d: d["records"][0]["feedback"][0].update(supersedes="nonexistent"),
        ]
        for mutate in mutations:
            bad = deepcopy(good)
            mutate(bad)
            with self.subTest(snapshot=bad), self.assertRaises(ValueError):
                self.store.restore(bad)
            self.assertEqual(self.store.snapshot(), good)

    def test_duplicate_events_requests_and_scope_overflow_rejected_on_restore(self):
        self.feedback(self.observe(), 1, event_id="event-a")
        self.feedback(self.observe(), 1, event_id="event-b")
        good = self.store.snapshot()
        def duplicate_event(d):
            d["records"][1]["feedback"][0]["event_id"] = "event-a"
        def duplicate_request(d):
            d["records"][1]["request_id"] = d["records"][0]["request_id"]
        def per_scope_overflow(d):
            d["config"].update(max_records_per_scope=1, min_support=1)
        def scope_overflow(d):
            d["config"]["max_scopes"] = 1
            d["records"][1]["scope"] = "other-scope"
        for mutation in (duplicate_event, duplicate_request, per_scope_overflow, scope_overflow):
            bad = deepcopy(good)
            mutation(bad)
            with self.assertRaises(ValueError):
                self.store.restore(bad)
            self.assertEqual(self.store.snapshot(), good)

    def test_numeric_and_identity_validation_is_strict(self):
        experience_id = self.observe()
        for value in (True, float("nan"), float("inf"), -0.1, 1.1, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.feedback(experience_id, value)
        for cost in (False, float("inf"), -1, "0"):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                self.observe(cost=cost)
        for identity in ({}, {"model_revision": "a"}, {**IDENTITY, "extra": "x"},
                         {**IDENTITY, "model_revision": ""}):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.observe(identity=identity)
        for kwargs in ({"max_scopes": True}, {"min_support": 0}, {"cost_weight": 2},
                       {"decay_half_life_seconds": 0}, {"max_feedback_events_per_record": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExperienceStore(**kwargs)

    def test_clock_rollback_cannot_break_feedback_snapshot_chronology(self):
        with patch("cognitive_management.research.experience.time.time", return_value=100):
            experience_id = self.observe()
            self.feedback(experience_id, 0)
        with patch("cognitive_management.research.experience.time.time", return_value=50):
            self.feedback(experience_id, 1)
        target = ExperienceStore()
        target.restore(self.store.snapshot())
        events = target.snapshot()["records"][0]["feedback"]
        self.assertEqual([e["created_at"] for e in events], [100, 100])

    def test_concurrent_observations_feedback_and_snapshots_remain_consistent(self):
        store = ExperienceStore(max_records_per_scope=256)
        def worker(index):
            experience_id = store.observe(scope="concurrent", identity=IDENTITY, task_key="task",
                planned_strategy="direct", executed_strategy="direct", status="observed", cost=0,
                request_id=f"req-{index}", sample_id=f"sample-{index}")
            kwargs = dict(experience_id=experience_id, scope="concurrent", identity=IDENTITY,
                          value=1, source="evaluator", event_id=f"event-{index}")
            self.assertTrue(store.feedback(**kwargs))
            self.assertFalse(store.feedback(**kwargs))
            target = ExperienceStore()
            target.restore(store.snapshot())
            return experience_id
        with ThreadPoolExecutor(max_workers=8) as executor:
            ids = list(executor.map(worker, range(80)))
        self.assertEqual(len(set(ids)), 80)
        self.assertEqual(store.revision, 160)
        self.assertEqual(len(store.snapshot()["records"]), 80)
        result = store.recommend(scope="concurrent", identity=IDENTITY, task_key="task",
                                 baseline="direct", allowed=["direct"])
        self.assertEqual(result["support"], 80)


if __name__ == "__main__":
    unittest.main()
