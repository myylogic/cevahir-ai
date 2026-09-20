"""Small table-only replay tests. No model, tokenizer, tensor, or dataset load."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.research.experience_replay import evaluate_rows, load_rows, main, validate_rows
from cognitive_management.research.experience import ExperienceStore


def synthetic_rows():
    """Deliberately invented labels exercising support, selection, and fallback."""
    rows = [{"sample_id": f"synthetic-train-{index}", "source_group": f"synthetic-source-{index}",
             "task_key": "synthetic-known", "split": "train", "baseline": "direct",
             "outcomes": {"direct": {"quality": 0.05, "cost": 0.1},
                          "think1": {"quality": 0.95, "cost": 0.5}}}
            for index in range(4)]
    rows.extend([
        {"sample_id": "synthetic-eval-known", "source_group": "synthetic-heldout-1",
         "task_key": "synthetic-known", "split": "eval", "baseline": "direct",
         "outcomes": {"direct": {"quality": 0.1, "cost": 0.1}, "think1": {"quality": 0.9, "cost": 0.5}}},
        {"sample_id": "synthetic-eval-new", "source_group": "synthetic-heldout-2",
         "task_key": "synthetic-new", "split": "eval", "baseline": "direct",
         "outcomes": {"direct": {"quality": 0.6, "cost": 0.1}, "think1": {"quality": 0.8, "cost": 0.5}}},
    ])
    return rows


class ExperienceReplayTests(unittest.TestCase):
    def test_controlled_table_exercises_supported_preference_and_fallback(self):
        report = evaluate_rows(synthetic_rows())
        known, novel = report["paired_samples"]
        self.assertEqual(known["policies"]["adaptive"]["action"], "think1")
        self.assertEqual(novel["policies"]["adaptive"]["action"], "direct")
        self.assertEqual(novel["policies"]["adaptive"]["reason"], "insufficient_baseline_support")
        self.assertAlmostEqual(known["deltas_vs_frozen"]["adaptive"]["quality"], 0.8)
        self.assertAlmostEqual(known["deltas_vs_frozen"]["adaptive"]["cost"], 0.4)
        self.assertEqual(report["policies"]["adaptive"]["action_distribution"], {"direct": 1, "think1": 1})
        self.assertFalse(report["empirical_model_quality_validated"])
        self.assertEqual(report["controls"]["eval_feedback_events"], 0)
        json.dumps(report, allow_nan=False)

    def test_ablation_omits_negative_training_feedback_without_making_it_positive(self):
        report = evaluate_rows(synthetic_rows())
        self.assertEqual(report["controls"]["train_feedback_counts"]["adaptive"], 8)
        self.assertEqual(report["controls"]["train_feedback_counts"]["no_negative_feedback"], 4)
        row = report["paired_samples"][0]
        self.assertEqual(row["policies"]["no_negative_feedback"]["action"], "direct")
        self.assertEqual(row["policies"]["no_negative_feedback"]["reason"], "insufficient_baseline_support")

    def test_eval_labels_cannot_change_recommendations_or_training_feedback(self):
        observed = []
        feedback = []

        class RecordingStore(ExperienceStore):
            def observe(self, **kwargs):
                observed.append(deepcopy(kwargs))
                return super().observe(**kwargs)

            def feedback(self, **kwargs):
                feedback.append(deepcopy(kwargs))
                return super().feedback(**kwargs)

        rows = synthetic_rows()
        original = evaluate_rows(rows, store_factory=RecordingStore)
        self.assertEqual(len(observed), 24)
        self.assertTrue(all(row["sample_id"].startswith("synthetic-train-") for row in observed))
        self.assertTrue(all(row["source"] == "evaluator" for row in feedback))
        changed = deepcopy(rows)
        for row in changed:
            if row["split"] == "eval":
                for observation in row["outcomes"].values():
                    observation["quality"] = 1 - observation["quality"]
                    observation["cost"] = 1 - observation["cost"]
        modified = evaluate_rows(changed)
        for first, second in zip(original["paired_samples"], modified["paired_samples"]):
            self.assertEqual({k: v["action"] for k, v in first["policies"].items()},
                             {k: v["action"] for k, v in second["policies"].items()})
        self.assertEqual(original["controls"]["train_feedback_counts"], modified["controls"]["train_feedback_counts"])

    def test_shuffle_is_seeded_and_uses_only_training_label_multiset(self):
        stores = []

        class RecordingStore(ExperienceStore):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.labels = []
                stores.append(self)

            def feedback(self, **kwargs):
                self.labels.append(kwargs["value"])
                return super().feedback(**kwargs)

        evaluate_rows(synthetic_rows(), seed=17, store_factory=RecordingStore)
        first = stores[1].labels
        self.assertEqual(sorted(first), [0.05] * 4 + [0.95] * 4)
        self.assertNotEqual(first, stores[0].labels)
        stores.clear()
        evaluate_rows(synthetic_rows(), seed=17, store_factory=RecordingStore)
        self.assertEqual(first, stores[1].labels)

    def test_row_order_does_not_allow_early_eval_feedback(self):
        rows = synthetic_rows()
        report = evaluate_rows(rows[-2:] + rows[:-2])
        self.assertEqual(report["paired_samples"][0]["policies"]["adaptive"]["action"], "think1")
        self.assertEqual(report["controls"]["train_outcome_count"], 8)

    def test_source_group_cannot_cross_splits(self):
        rows = synthetic_rows()
        rows[-1]["source_group"] = rows[0]["source_group"]
        with self.assertRaisesRegex(ValueError, "source_group crosses"):
            evaluate_rows(rows)

    def test_duplicate_sample_ids_are_rejected(self):
        rows = synthetic_rows()
        rows[1]["sample_id"] = rows[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            validate_rows(rows)

    def test_bad_outcomes_and_missing_baseline_are_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            rows = synthetic_rows()
            rows[0]["outcomes"]["direct"]["quality"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_rows(rows)
        rows = synthetic_rows()
        del rows[0]["outcomes"]["direct"]
        with self.assertRaisesRegex(ValueError, "baseline outcome"):
            validate_rows(rows)

    def test_dataset_and_stream_caps_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "max_rows"):
            validate_rows(iter(synthetic_rows()), max_rows=3)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("x" * 40, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_line_chars"):
                load_rows(path, max_line_chars=16)
            path.write_text('\n'.join(json.dumps(row) for row in synthetic_rows()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_rows"):
                load_rows(path, max_rows=2)

    def test_duplicate_json_keys_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"sample_id":"first","sample_id":"second"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_rows(path)

    def test_store_cannot_return_unobserved_action(self):
        class InvalidStore(ExperienceStore):
            def recommend(self, **kwargs):
                return {"strategy": "tot", "support": 10, "reason": "unsupported"}

        with self.assertRaisesRegex(ValueError, "unobserved action"):
            evaluate_rows(synthetic_rows(), store_factory=InvalidStore)

    def test_evaluation_does_not_mutate_frozen_store(self):
        class MutatingStore(ExperienceStore):
            def recommend(self, **kwargs):
                result = super().recommend(**kwargs)
                self.forget(scope=kwargs["scope"])
                return result

        with self.assertRaisesRegex(RuntimeError, "modified a frozen"):
            evaluate_rows(synthetic_rows(), store_factory=MutatingStore)

    def test_cli_writes_labeled_report_and_cannot_overwrite_input(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "synthetic.jsonl"
            output_path = Path(directory) / "report.json"
            input_path.write_text('\n'.join(json.dumps(row) for row in synthetic_rows()), encoding="utf-8")
            self.assertEqual(main(["--input", str(input_path), "--output", str(output_path)]), 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("offline", report["evidence_type"])
            with self.assertRaises(SystemExit):
                main(["--input", str(input_path), "--output", str(input_path)])


if __name__ == "__main__":
    unittest.main()
