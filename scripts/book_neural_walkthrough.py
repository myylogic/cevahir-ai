"""Small CPU examples using Cevahir's real neural components (no trained assets).

Default: reproduce assertions and compare with recorded educational evidence.
--write: explicitly record reviewed results; never train or rewrite stored models.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from src.neural_network import CevahirNeuralNetwork
from src.neural_network_module.architecture_contracts import resolve_ffn_dim
from src.neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding
from src.neural_network_module.dil_katmani_module.positional_encoding import PositionalEncoding
from src.neural_network_module.ortak_katman_module.feed_forward_network import FeedForwardNetwork
from src.neural_network_module.ortak_katman_module.rms_norm import RMSNorm
from src.neural_network_module.ortak_katman_module.transformer_encoder_layer import TransformerEncoderLayer
from training_management.v2.core.loss_computation import LossComputation

OUTPUT = ROOT / "docs/book/evidence/neural_walkthrough.json"


def tiny(**overrides):
    config = dict(learning_rate=.001, dropout=0., vocab_size=41, embed_dim=16,
                  seq_proj_dim=16, num_heads=4, num_layers=2, ffn_dim=24,
                  use_gradient_checkpointing=False, use_advanced_checkpointing=False,
                  log_level=50, max_cache_len=32, pe_max_len=32, use_moe=False)
    return CevahirNeuralNetwork(**{**config, **overrides})


def max_error(left, right):
    return float((left.detach() - right.detach()).abs().max())


def embedding_example():
    layer = LanguageEmbedding(6, 3, dropout=0., norm_type="none", log_level=50)
    with torch.no_grad():
        layer.embedding.weight.copy_(torch.arange(18).reshape(6, 3) / 10.)
    ids = torch.tensor([[1, 2, 1]])
    actual = layer(ids)
    explicit = F.one_hot(ids, 6).float() @ layer.embedding.weight
    torch.testing.assert_close(actual, explicit)
    actual.sum().backward()
    lookup_rows = layer.embedding.weight.grad.abs().sum(-1).nonzero().flatten().tolist()
    assert lookup_rows == [1, 2]
    layer.zero_grad(set_to_none=True)
    head = nn.Linear(3, 6, bias=False)
    head.weight = layer.embedding.weight
    logits = head(layer(ids))
    F.cross_entropy(logits.reshape(-1, 6), torch.tensor([2, 3, 4])).backward()
    output_rows = layer.embedding.weight.grad.abs().sum(-1).nonzero().flatten().tolist()
    assert 5 in output_rows and head.weight is layer.embedding.weight
    return {"ids": ids.tolist(), "embedding": actual.detach().tolist(),
            "lookup_one_hot_max_error": max_error(actual, explicit),
            "lookup_only_gradient_rows": lookup_rows,
            "tied_output_gradient_rows": output_rows,
            "tied_same_parameter_object": head.weight is layer.embedding.weight,
            "note": "Deliberately assigned teaching weights, not learned semantics; norm/dropout disabled."}


def scalar_backprop_example():
    w, b, v, c = [torch.tensor(x, dtype=torch.float64, requires_grad=True)
                  for x in (.5, 0., 2., 0.)]
    prediction = v * torch.relu(w * 2 + b) + c
    loss = .5 * prediction.square()
    loss.backward()
    gradients = [float(p.grad) for p in (w, b, v, c)]
    assert gradients == [8., 4., 2., 2.]
    with torch.no_grad():
        for p in (w, b, v, c):
            p -= .01 * p.grad
        after = v * torch.relu(w * 2 + b) + c
    assert math.isclose(float(after), 1.564, abs_tol=1e-12)
    return {"prediction_before": float(prediction.detach()), "loss_before": float(loss.detach()),
            "gradients_w_b_v_c": gradients, "learning_rate": .01,
            "prediction_after": float(after), "loss_after": float(.5 * after.square()),
            "note": "Scalar chain-rule teaching example, not Cevahir's architecture."}


def attention_hand_example():
    attention = tiny().eval().layers[0].attn
    q = torch.tensor([1., 2., 0.]).reshape(1, 1, 3, 1)
    values = torch.tensor([1., 3., 2.]).reshape(1, 1, 3, 1)
    # Already-projected Q/K/V supplied to the actual low-level method.
    out, weights = attention.scaled_dot_product_attention(
        q, q, values, causal_mask=True, apply_dropout=False, return_attention_weights=True)
    expected = torch.tensor([1., 2.761594156, 2.]).reshape_as(out)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    return {"q_equals_k": [1, 2, 0], "v": [1, 3, 2],
            "weights": weights[0, 0].tolist(), "output": out.flatten().tolist(),
            "note": "Hand-chosen projected vectors; bypasses learned projection and RoPE."}


def attention_cache_examples():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    cases = []
    for kv_heads in (1, 2, 4):
        model = tiny(num_kv_heads=kv_heads, use_pytorch_sdpa=True).eval()
        with torch.no_grad():
            full = model(ids)[0]
            manual, weights = model(ids, return_attention_weights=True)
            torch.testing.assert_close(full, manual, atol=2e-6, rtol=2e-5)
            changed = ids.clone()
            changed[:, 3:] = (changed[:, 3:] + 7) % 41
            altered = model(changed)[0]
            torch.testing.assert_close(full[:, :3], altered[:, :3])
            chunks = [model(ids[:, start:end], use_cache=True,
                            cache_position=torch.arange(start, end))[0]
                      for start, end in ((0, 2), (2, 5), (5, 6))]
            cached = torch.cat(chunks, dim=1)
            torch.testing.assert_close(full, cached, atol=2e-6, rtol=2e-5)
            cache_shapes = [list(layer.attn.kv_cache.get()[0].shape) for layer in model.layers]
            assert all(shape == [2, kv_heads, 6, 4] for shape in cache_shapes)
            blocked = model.layers[0].attn(torch.randn(1, 3, 16),
                                         mask=torch.ones(3, 3, dtype=torch.bool), causal_mask=True)
            assert torch.count_nonzero(blocked) == 0
        cases.append({"kv_heads": kv_heads, "query_heads": 4,
                      "full_vs_manual_max_error": max_error(full, manual),
                      "full_vs_chunked_max_error": max_error(full, cached),
                      "future_change_prefix_max_error": max_error(full[:, :3], altered[:, :3]),
                      "weight_shape": list(weights.shape), "cache_key_shapes": cache_shapes,
                      "fully_blocked_output_nonzero": int(torch.count_nonzero(blocked))})
    return cases


def block_examples():
    counts = {}
    for activation in ("gelu", "swiglu"):
        ffn = FeedForwardNetwork(8, 16, dropout=0., activation=activation, use_bias=False)
        counts[activation] = sum(p.numel() for p in ffn.parameters())
    assert counts == {"gelu": 256, "swiglu": 384}
    width = resolve_ffn_dim(512, gated=True)
    assert width == 1536
    layer = TransformerEncoderLayer(embed_dim=16, num_heads=4, ffn_dim=24,
                                    dropout=0., pre_norm=True, parallel_residual=False,
                                    use_gradient_checkpointing=False, log_level=50).eval()
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        normalized = layer.norm1(x)
        intermediate = x + layer.attn(normalized, normalized, normalized, causal_mask=True)
        manual = intermediate + layer.ffn(layer.norm2(intermediate))
        actual = layer(x, causal_mask=True)[0]
    torch.testing.assert_close(manual, actual)
    rms = RMSNorm(2, log_level=50)
    vector = torch.tensor([[3., 4.]])
    normalized = rms(vector)
    explicit = vector / torch.sqrt(vector.square().mean(-1, keepdim=True) + rms.eps)
    torch.testing.assert_close(normalized, explicit)
    return {"bias_free_ffn_parameters_D8_F16": counts, "resolved_gated_width_D512": width,
            "serial_pre_norm_max_error": max_error(actual, manual),
            "rmsnorm_input": vector.tolist(), "rmsnorm_output": normalized.detach().tolist(),
            "rmsnorm_eps": rms.eps, "rmsnorm_formula_max_error": max_error(normalized, explicit)}


def rope_example():
    rope = PositionalEncoding(8, max_len=32, dropout=0., mode="rope", num_heads=2, log_level=50)
    q, k = torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4)
    def rotate(vec, position):
        return rope.apply_rotary_pos_emb(vec, torch.tensor([position]))
    dot = (rotate(q, 2) * rotate(k, 5)).sum()
    shifted = (rotate(q, 9) * rotate(k, 12)).sum()
    torch.testing.assert_close(dot, shifted, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(q.square().sum(), rotate(q, 2).square().sum())
    return {"positions": [2, 5], "shifted_positions": [9, 12],
            "dot_product": float(dot), "shifted_dot_product": float(shifted),
            "common_shift_max_error": max_error(dot, shifted),
            "norm_preservation_max_error": max_error(q.square().sum(), rotate(q, 2).square().sum())}


def model_update_example():
    model = tiny().eval()
    ids = torch.tensor([[1, 2, 3, 4], [6, 7, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5], [7, 3, 4, -100]])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    trace, handles = {}, []
    def capture(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            trace[name] = list(tensor.shape)
        return hook
    for name, module in [("embedding", model.embedding),
                         *[(f"layer_{i}", layer) for i, layer in enumerate(model.layers)],
                         ("output_norm", model.output_norm), ("output_projection", model.output_layer)]:
        handles.append(module.register_forward_hook(capture(name)))
    try:
        logits = model(ids)[0]
    finally:
        for handle in handles:
            handle.remove()
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    raw = model.embedding.embedding(ids)
    assert torch.equal(raw[0, 2], raw[1, 2])
    assert not torch.allclose(logits[0, 2], logits[1, 2])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.)
    objective = LossComputation(criterion)
    loss, accuracy, ppl = objective.compute_loss(logits, targets)
    valid = targets != -100
    reference = F.cross_entropy(logits[valid], targets[valid])
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert changed and model.output_layer.weight is model.embedding.embedding.weight
    with torch.no_grad():
        after = objective.compute_loss(model(ids)[0], targets)[0]
    return {"input_ids": ids.tolist(), "targets": targets.tolist(), "trace_shapes": trace,
            "logits_shape": list(logits.shape), "parameter_count": sum(p.numel() for p in model.parameters()),
            "tied_same_parameter_object": True, "forward_parameters_unchanged": True,
            "backward_parameters_unchanged": True, "changed_parameter_tensors_after_sgd": len(changed),
            "valid_target_tokens": int(valid.sum()), "loss_before": float(loss.detach()),
            "loss_after_one_sgd_step": float(after), "accuracy_before": accuracy, "perplexity_before": ppl,
            "same_token_initial_vectors_equal": True,
            "same_token_different_context_logits_max_difference": max_error(logits[0, 2], logits[1, 2]),
            "note": "Random tiny core; in-memory SGD only. No language-quality or generalization claim."}


def run():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20260921)
    try:
        return {"embedding": embedding_example(), "scalar_backprop": scalar_backprop_example(),
                "attention_hand": attention_hand_example(), "attention_cache": attention_cache_examples(),
                "block": block_examples(), "rope": rope_example(), "model_update": model_update_example()}
    finally:
        torch.set_num_threads(previous_threads)


def compare(actual, expected, path="cases"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-6), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    cases = run()
    if args.write:
        result = {"date": "2026-09-21", "kind": "executed_book_examples",
                  "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
                  "seed": 20260921, "cases": cases,
                  "comparison_tolerance": {"absolute": 2e-6, "relative": 2e-5},
                  "limits": ["No trained checkpoint, vocabulary or research record modified",
                             "Synthetic data, random tiny core and explicitly assigned teaching weights",
                             "No GPU, timing, large-model or language-quality benchmark",
                             "Assertions also execute independently of stored numerical evidence"]}
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    else:
        compare(cases, json.loads(OUTPUT.read_text(encoding="utf-8"))["cases"])
    print(json.dumps({"status": "passed", "example_groups": len(cases), "cache_variants": len(cases["attention_cache"]),
                      "device": "cpu", "mode": "record" if args.write else "check"}))


if __name__ == "__main__":
    main()
