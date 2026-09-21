"""Book examples using real generation, retrieval and tool orchestration code.

The fixed-logit manager and hand-assigned vectors are teaching fixtures, not a
trained language model or an embedding-quality benchmark. Default mode verifies
the stored evidence; --write explicitly records a new evidence file.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from cognitive_management.cognitive_types import DecodingConfig
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.v2.components.memory_service_v2 import MemoryServiceV2
from cognitive_management.v2.components.vector_store.memory_vector_store import MemoryVectorStore
from cognitive_management.v2.components.tool_executor_v2 import ToolExecutorV2
from cognitive_management.v2.components.tool_policy_v2 import ToolPolicyV2
from cognitive_management.v2.utils.request_scope import memory_scope
from model.cevahir import CevahirModelAPI

OUTPUT = ROOT / "docs/book/evidence/runtime_walkthrough.json"


class FixtureTokenizer:
    def encode(self, text, **kwargs):
        return ["p", "q"], [0, 1]

    def get_vocab(self):
        return {"<EOS>": 3}

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if i != 3)


class FixedLogitManager:
    device = torch.device("cpu")
    is_initialized = True

    def __init__(self, logits, cache=True):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.config = {"use_kv_cache": cache}
        self.calls = []
        self.clears = 0

    def eval_mode(self):
        pass

    def clear_kv_cache(self):
        self.clears += 1

    def forward(self, tokens, **kwargs):
        position = kwargs.get("cache_position")
        self.calls.append({"length": tokens.shape[1], "use_cache": kwargs.get("use_cache"),
                           "positions": position.tolist() if position is not None else None})
        return self.logits.repeat(1, tokens.shape[1], 1), None


def sampling_examples():
    base = [.5, .3, .15, .05]
    recorded = {}
    for name, temperature, top_k, top_p, expected in [
        ("nucleus_only", 1., 0, .6, [.625, .375, 0., 0.]),
        ("top_k_then_nucleus", 1., 2, .6, [1., 0., 0., 0.]),
        ("temperature_two", 2., 0, 1., None),
    ]:
        manager = FixedLogitManager([math.log(p) for p in base])
        api = CevahirModelAPI(manager, FixtureTokenizer())
        captured = []

        def capture(probs, num_samples):
            captured.append(probs.clone())
            return torch.argmax(probs).reshape(1)

        cfg = DecodingConfig(max_new_tokens=1, temperature=temperature, top_k=top_k,
                             top_p=top_p, repetition_penalty=1.)
        with patch("torch.multinomial", side_effect=capture):
            api.generate("fixture", cfg)
        assert len(captured) == 1
        if expected is None:
            expected = [math.sqrt(p)/sum(math.sqrt(x) for x in base) for p in base]
        torch.testing.assert_close(captured[0], torch.tensor(expected))
        recorded[name] = {"temperature": temperature, "top_k": top_k, "top_p": top_p,
                          "probabilities": captured[0].tolist()}
    return {"base_probabilities": base, "cases": recorded,
            "note": "Production filtering; multinomial intercepted to inspect its input, not a random-sampling experiment."}


def generation_protocol_examples():
    result = {}
    for cache in (True, False):
        manager = FixedLogitManager([0., 0., 5., 0.], cache)
        api = CevahirModelAPI(manager, FixtureTokenizer())
        with patch("torch.multinomial", side_effect=AssertionError("Greedy must not sample")):
            output = api.generate("fixture", DecodingConfig(max_new_tokens=3, temperature=0., repetition_penalty=1.))
        lengths = [c["length"] for c in manager.calls]
        assert lengths == ([2, 1, 1] if cache else [2, 3, 4])
        assert output == "2 2 2"
        result["cache" if cache else "no_cache"] = {"text": output, "calls": manager.calls, "cache_clears": manager.clears}
    manager = FixedLogitManager([0., 0., 0., 5.])
    api = CevahirModelAPI(manager, FixtureTokenizer())
    output = api.generate("fixture", DecodingConfig(max_new_tokens=10, min_new_tokens=2,
                          temperature=0., repetition_penalty=1.))
    assert output == "0 0" and len(manager.calls) == 3
    result["minimum_then_eos"] = {"text": output, "forward_calls": len(manager.calls), "minimum_new_tokens": 2}
    manager = FixedLogitManager([0., 0., 5., 0.])
    api = CevahirModelAPI(manager, FixtureTokenizer())
    assert api.generate("fixture", DecodingConfig(max_new_tokens=0)) == ""
    assert manager.calls == []
    result["zero_limit_forward_calls"] = 0
    manager = FixedLogitManager([0., 0., 5., 0.])
    api = CevahirModelAPI(manager, FixtureTokenizer())
    api.generate("fixture", DecodingConfig(max_new_tokens=3, num_beams=2, temperature=0.))
    assert manager.calls and all(c["use_cache"] is False for c in manager.calls)
    result["beam_calls"] = manager.calls
    result["note"] = "Fixed logits verify adapter call protocol only; real neural cache equivalence is tested separately."
    return result


def entropy_example():
    probabilities = [.5, .3, .15, .05]
    api = CevahirModelAPI(FixedLogitManager([math.log(p) for p in probabilities]), FixtureTokenizer())
    details = api.entropy_details("fixture")
    expected = -sum(p * math.log(p) for p in probabilities)/math.log(4)
    assert details["available"] and math.isclose(details["value"], expected, abs_tol=1e-6)
    return {"probabilities": probabilities, "normalized_entropy": details["value"], "available": details["available"],
            "note": "Token distribution entropy; not the probability that an answer is true."}


def scoped_retrieval_example():
    cfg = CognitiveManagerConfig()
    cfg.memory.enable_vector_memory = False
    cfg.memory.enable_rag = False
    cfg.memory.hybrid_search_alpha = 1.
    cfg.memory.rag_score_threshold = 0.
    memory = MemoryServiceV2(cfg)
    store = MemoryVectorStore(cfg, dimension=2)
    store.add(texts=["other exact match", "my related context"], embeddings=[[1., 0.], [.8, .2]],
              metadata=[{"scope": "other"}, {"scope": "mine"}], ids=["other-record", "my-record"])
    memory._vector_memory_enabled = True
    memory._vector_store = store
    memory._embedding_adapter = SimpleNamespace(encode_single=lambda text: [1., 0.])
    with memory_scope("mine"):
        results = memory.retrieve_context("query", top_k=1)
    assert len(results) == 1 and results[0]["content"] == "my related context"
    score = float(results[0]["score"])
    raw_cosine = .8/math.sqrt(.68)
    assert math.isclose(score, (raw_cosine + 1.) / 2., abs_tol=1e-6)
    return {"query_vector": [1., 0.], "other_vector": [1., 0.], "mine_vector": [.8, .2],
            "top_k": 1, "scope": "mine", "returned_content": results[0]["content"],
            "raw_cosine": raw_cosine, "score": score,
            "note": "Assigned vectors exercise scope filtering before top-k, not embedding quality or authentication."}


def tool_semantics_example():
    cfg = CognitiveManagerConfig()
    cfg.tools.enable_tools = True
    cfg.tools.allow = ["calculator"]
    executor = ToolExecutorV2(cfg)
    policy = ToolPolicyV2(cfg, executor)
    inferred = policy.infer_tool_parameters("calculator", "2+3*4", {})
    parsed_result = executor.execute("calculator", inferred)
    explicit_result = executor.execute("calculator", {"operation": "2+3*4"})
    successes = executor.get_tool_metrics("calculator")["success_count"]
    assert inferred == {"operation": "2+3*4"}
    assert parsed_result == "14" and explicit_result == "14" and successes == 2
    assert policy.infer_tool_parameters("calculator", "2 ve 3", {}) == {}
    try:
        executor.execute("calculator", {"operation": "2/0"})
    except Exception as error:
        failure_type = type(error).__name__
    else:
        raise AssertionError("Division by zero must fail")
    assert failure_type == "ToolPolicyError"
    return {"request": "2+3*4", "inferred_parameters": inferred, "inferred_result": parsed_result,
            "explicit_result": explicit_result, "successful_executions_before_failure": successes,
            "division_by_zero_error": failure_type,
            "ambiguous_numbers_inferred_parameters": {},
            "note": "Complete expression preserved after fix; execution success alone still cannot certify user intent."}


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            compare(a, e)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, abs_tol=2e-6, rel_tol=2e-5), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    cases = {"sampling": sampling_examples(), "generation_protocol": generation_protocol_examples(),
             "entropy": entropy_example(), "scoped_retrieval": scoped_retrieval_example(),
             "tool_semantics": tool_semantics_example()}
    if args.write:
        evidence = {"date": "2026-09-21", "kind": "executed_book_examples",
                    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
                    "cases": cases, "comparison_tolerance": {"absolute": 2e-6, "relative": 2e-5},
                    "limits": ["Fixed logits and assigned vectors, no trained-model quality claim",
                               "Actual production selection, memory and tool methods execute",
                               "Calculator inference corrected in this revision; model weights and historical research records unchanged"]}
        OUTPUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    else:
        compare(cases, json.loads(OUTPUT.read_text(encoding="utf-8"))["cases"])
    print(json.dumps({"status": "passed", "example_groups": len(cases), "mode": "record" if args.write else "check"}))


if __name__ == "__main__":
    main()
