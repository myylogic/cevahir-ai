"""Bounded research-runtime contracts using only stdlib fake backends."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field, replace
import json
from threading import Event, Lock
import unittest

from cognitive_management.research.runtime import (
    BudgetedModelAPI, BudgetExhausted, BudgetLimits, RequestRuntime,
    current_runtime, request_runtime, runtime_stage,
)


@dataclass(frozen=True)
class Decoding:
    max_new_tokens: int = 20
    min_new_tokens: int | None = None
    num_beams: int = 1
    tags: list = field(default_factory=list)


class FakeBackend:
    supports_routing_bias = True

    def __init__(self):
        self.calls = []
        self.lock = Lock()
        self.fail = False
        self.entropy = 0.5

    def _record(self, *args):
        with self.lock:
            self.calls.append(args)
        if self.fail:
            raise RuntimeError("fake failure")

    def generate(self, prompt, config, **kwargs):
        self._record("generate", prompt, config, kwargs)
        return "answer:" + prompt

    def score(self, prompt, candidate, **kwargs):
        self._record("score", prompt, candidate, kwargs)
        return 0.75

    def entropy_estimate(self, prompt, **kwargs):
        self._record("entropy", prompt, kwargs)
        return self.entropy


def make_runtime(**overrides):
    return RequestRuntime("user/session", {"model_revision": "test"},
                          replace(BudgetLimits(), **overrides))


class ResearchRuntimeTests(unittest.TestCase):
    def test_unscoped_proxy_keeps_raw_arguments_and_custom_attributes(self):
        raw = FakeBackend()
        proxy = BudgetedModelAPI(raw)
        config = Decoding()
        self.assertEqual(proxy.generate("x", config, marker=1), "answer:x")
        self.assertIs(raw.calls[0][2], config)
        self.assertEqual(raw.calls[0][3], {"marker": 1})
        self.assertIs(proxy.calls, raw.calls)
        self.assertIsNone(current_runtime())

    def test_nested_runtime_and_stage_restore_on_exception(self):
        first, second = make_runtime(), make_runtime()
        with request_runtime(first), runtime_stage("deliberation"):
            self.assertEqual(first.stage, "deliberation")
            with self.assertRaises(RuntimeError):
                with request_runtime(second), runtime_stage("answer"):
                    self.assertIs(current_runtime(), second)
                    raise RuntimeError("exit")
            self.assertIs(current_runtime(), first)
            self.assertEqual(first.stage, "deliberation")
        self.assertIsNone(current_runtime())

    def test_answer_call_and_token_reservation_survives_deliberation(self):
        raw = FakeBackend()
        proxy = BudgetedModelAPI(raw)
        runtime = make_runtime(max_generate_calls=2, max_output_tokens=100, reserve_answer_tokens=40)
        with request_runtime(runtime):
            with runtime_stage("deliberation"):
                proxy.generate("thought", Decoding(90, 80))
                self.assertEqual(raw.calls[-1][2].max_new_tokens, 60)
                self.assertEqual(raw.calls[-1][2].min_new_tokens, 60)
                with self.assertRaises(BudgetExhausted):
                    proxy.generate("another thought", Decoding())
            with runtime_stage("answer"):
                proxy.generate("final", Decoding(90))
                self.assertEqual(raw.calls[-1][2].max_new_tokens, 40)
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["reserved_output_tokens"], 100)
        self.assertEqual(snapshot["calls"]["generate"], 2)
        self.assertFalse(snapshot["answer_reservation_held"])
        self.assertEqual(runtime.last_answer, "answer:final")

    def test_input_allowance_is_shared_and_protected_for_answer(self):
        raw = FakeBackend()
        proxy = BudgetedModelAPI(raw)
        runtime = make_runtime(max_input_chars=20, reserve_answer_input_chars=8)
        with request_runtime(runtime):
            proxy.score("12345", "67890")
            proxy.entropy_estimate("12")
            with self.assertRaises(BudgetExhausted) as denied:
                proxy.score("x", "")
            self.assertEqual(denied.exception.reason, "input_chars_reserved_for_answer")
            with runtime_stage("answer"):
                proxy.generate("12345678", Decoding())
        self.assertEqual(runtime.snapshot()["input_chars"], 20)

    def test_answer_prompt_larger_than_reserved_share_can_be_denied(self):
        proxy = BudgetedModelAPI(FakeBackend())
        runtime = make_runtime(max_input_chars=10, reserve_answer_input_chars=4)
        with request_runtime(runtime):
            proxy.score("123456", "")
            with runtime_stage("answer"), self.assertRaises(BudgetExhausted):
                proxy.generate("12345", Decoding())
        self.assertEqual(runtime.snapshot()["calls"]["generate"], 0)

    def test_beams_charge_input_and_output_and_clamp_without_mutation(self):
        raw = FakeBackend()
        proxy = BudgetedModelAPI(raw)
        runtime = make_runtime(max_output_tokens=60, reserve_answer_tokens=0)
        config = Decoding(100, 80, 3, tags=["original"])
        with request_runtime(runtime), runtime_stage("answer"):
            proxy.generate("abcd", config)
        passed = raw.calls[0][2]
        self.assertEqual((passed.max_new_tokens, passed.min_new_tokens), (20, 20))
        passed.tags.append("backend mutation")
        self.assertEqual(config.tags, ["original"])
        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(runtime.snapshot()["reserved_output_tokens"], 60)
        self.assertEqual(runtime.snapshot()["input_chars"], 12)

    def test_dict_config_is_copied(self):
        raw = FakeBackend()
        config = {"max_new_tokens": 100, "min_new_tokens": 80, "num_beams": 2}
        runtime = make_runtime(max_output_tokens=30, reserve_answer_tokens=0)
        with request_runtime(runtime), runtime_stage("answer"):
            BudgetedModelAPI(raw).generate("x", config)
        self.assertEqual(config["max_new_tokens"], 100)
        self.assertEqual(raw.calls[0][2]["min_new_tokens"], 15)

    def test_errors_and_cancellation_keep_reservations(self):
        raw = FakeBackend()
        raw.fail = True
        proxy = BudgetedModelAPI(raw)
        runtime = make_runtime(max_generate_calls=1)
        with request_runtime(runtime), runtime_stage("answer"):
            with self.assertRaises(RuntimeError):
                proxy.generate("x", Decoding())
            with self.assertRaises(BudgetExhausted):
                proxy.generate("x", Decoding())
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["errors"], 1)
        self.assertEqual(snapshot["reserved_output_tokens"], 20)
        self.assertEqual(snapshot["calls"]["generate"], 1)
        self.assertFalse(snapshot["answer_reservation_held"])
        runtime.cancel("client disconnected")
        with request_runtime(runtime), self.assertRaises(BudgetExhausted) as denied:
            proxy.score("x", "y")
        self.assertEqual(denied.exception.reason, "cancelled")

    def test_successful_answer_is_preserved_when_refinement_fails(self):
        raw = FakeBackend()
        proxy = BudgetedModelAPI(raw)
        runtime = make_runtime()
        with request_runtime(runtime):
            with runtime_stage("answer"):
                proxy.generate("useful", Decoding())
            raw.fail = True
            with runtime_stage("refinement"), self.assertRaises(RuntimeError):
                proxy.generate("revision", Decoding())
        self.assertEqual(runtime.last_answer, "answer:useful")

    def test_concurrent_calls_cannot_spend_reserved_answer_slot(self):
        raw = FakeBackend()
        proxy = BudgetedModelAPI(raw)
        runtime = make_runtime(max_generate_calls=3)

        def attempt():
            try:
                proxy.generate("x", Decoding())
                return True
            except BudgetExhausted:
                return False

        with request_runtime(runtime), runtime_stage("deliberation"):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(copy_context().run, attempt) for _ in range(20)]
                self.assertEqual(sum(f.result() for f in futures), 2)
            with runtime_stage("answer"):
                proxy.generate("final", Decoding())
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["calls"]["generate"], 3)
        self.assertEqual(snapshot["denied"], 18)

    def test_inflight_completion_after_cancel_is_not_a_new_answer(self):
        started, finish = Event(), Event()

        class BlockingBackend(FakeBackend):
            def generate(self, prompt, config, **kwargs):
                started.set()
                if not finish.wait(2):
                    raise RuntimeError("test synchronization failed")
                return "late answer"

        runtime = make_runtime()
        with request_runtime(runtime), runtime_stage("answer"):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(copy_context().run, BudgetedModelAPI(BlockingBackend()).generate, "x", Decoding())
                try:
                    self.assertTrue(started.wait(2))
                    runtime.cancel()
                finally:
                    finish.set()
                self.assertEqual(future.result(), "late answer")
        self.assertIsNone(runtime.last_answer)
        self.assertEqual(runtime.snapshot()["attempts"][0]["status"], "completed_after_cancel")
        self.assertEqual(runtime.snapshot()["reserved_output_tokens"], 20)

    def test_limits_and_decoding_reject_invalid_counts_without_backend_calls(self):
        for value in (-1, float("nan"), float("inf"), 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BudgetLimits(max_generate_calls=value)
        with self.assertRaises(ValueError):
            BudgetLimits(max_input_chars=10, reserve_answer_input_chars=11)
        with self.assertRaises(ValueError):
            BudgetLimits(max_output_tokens=10, reserve_answer_tokens=11)
        raw = FakeBackend()
        with request_runtime(make_runtime()), runtime_stage("answer"):
            for config in (Decoding(0), Decoding(float("nan")), Decoding(-1),
                           Decoding(1, 2), Decoding(10, -1), Decoding(10, None, 0),
                           Decoding(10, None, float("inf")), Decoding(True)):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    BudgetedModelAPI(raw).generate("x", config)
            with self.assertRaises(ValueError):
                BudgetedModelAPI(raw).generate("x", Decoding(), max_new_tokens=999)
        self.assertEqual(raw.calls, [])

    def test_plan_prior_reaches_generation_and_scoring_but_not_entropy(self):
        raw = FakeBackend()
        runtime = make_runtime()
        runtime.set_metadata(plan={"routing_bias": [0.2, -0.2]})
        proxy = BudgetedModelAPI(raw)
        with request_runtime(runtime):
            proxy.entropy_estimate("x")
            with runtime_stage("answer"):
                proxy.generate("x", Decoding())
            proxy.score("x", "y")
        self.assertEqual(raw.calls[0][2], {})
        self.assertEqual(raw.calls[1][3]["routing_bias"], [[0.2, -0.2]])
        self.assertEqual(raw.calls[2][3]["routing_bias"], [[0.2, -0.2]])
        raw.calls[1][3]["routing_bias"][0][0] = 9
        self.assertEqual(runtime.plan["routing_bias"], [0.2, -0.2])
        json.dumps(runtime.snapshot())

    def test_prior_rejects_unsupported_backend_or_malformed_values(self):
        raw = FakeBackend()
        proxy = BudgetedModelAPI(raw)
        runtime = make_runtime()
        with request_runtime(runtime), runtime_stage("answer"):
            for bias in ([], [[1]], [float("nan")], [True], ["x"]):
                runtime.plan = {"routing_bias": bias}
                with self.subTest(bias=bias), self.assertRaises(ValueError):
                    proxy.generate("x", Decoding())
            runtime.plan = {"routing_bias": [1, -1]}
            raw.supports_routing_bias = False
            with self.assertRaisesRegex(ValueError, "does not support"):
                proxy.score("x", "y")
        self.assertEqual(raw.calls, [])

    def test_invalid_entropy_is_error_not_an_observation(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "uncertain"):
            with self.subTest(value=value):
                raw = FakeBackend()
                raw.entropy = value
                runtime = make_runtime()
                with request_runtime(runtime), self.assertRaises(ValueError):
                    BudgetedModelAPI(raw).entropy_estimate("x")
                self.assertEqual(runtime.snapshot()["errors"], 1)
                self.assertEqual(runtime.snapshot()["calls"]["entropy"], 1)

    def test_snapshots_do_not_expose_mutable_ledger(self):
        runtime = make_runtime()
        runtime.set_metadata(plan={"samples": [1]})
        snapshot = runtime.snapshot()
        snapshot["plan"]["samples"].append(2)
        snapshot["calls"]["score"] = 999
        self.assertEqual(runtime.snapshot()["plan"], {"samples": [1]})
        self.assertEqual(runtime.snapshot()["calls"]["score"], 0)


class AsyncResearchRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_to_thread_propagates_request_and_stage_without_cross_request_leak(self):
        proxy = BudgetedModelAPI(FakeBackend())
        runtimes = [make_runtime(), make_runtime()]

        async def run(runtime, text):
            with request_runtime(runtime), runtime_stage("answer"):
                await asyncio.to_thread(proxy.generate, text, Decoding())
                self.assertIs(current_runtime(), runtime)

        await asyncio.gather(run(runtimes[0], "first"), run(runtimes[1], "second"))
        self.assertEqual(runtimes[0].last_answer, "answer:first")
        self.assertEqual(runtimes[1].last_answer, "answer:second")
        self.assertEqual([r.snapshot()["calls"]["generate"] for r in runtimes], [1, 1])
        self.assertIsNone(current_runtime())


if __name__ == "__main__":
    unittest.main()
