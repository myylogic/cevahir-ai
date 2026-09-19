"""Small, reproducible measurements of the real Cevahir core (CPU by default).

Examples: python benchmarks/core.py --label current --output benchmarks/results/current.json
          python benchmarks/core.py --compare baseline.json current.json --output delta.json
Synthetic next-token loss is a correctness probe, not a language-quality evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import inspect
import json
import logging
import math
from pathlib import Path
import platform
import statistics
import sys
import time


def compare(baseline, experiment):
    differences = {}
    for key, value in baseline["metrics"].items():
        other = experiment["metrics"].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(other, (int, float)):
            differences[key] = {"baseline": value, "experiment": other, "delta": other - value,
                                "percent": 100 * (other - value) / value if value else None}
    return {"BASELINE": baseline, "EXPERIMENT": experiment, "DELTA": differences,
            "interpretation": "Single-host tiny synthetic run. Timing is noisy; architectural correctness and quality are separate."}


def run(args):
    source = Path(args.source_root).resolve() if args.source_root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))
    import torch
    from src.neural_network import CevahirNeuralNetwork
    from model_management.profiler import ModelProfiler
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu")
    config = dict(learning_rate=2e-4, dropout=0.0, vocab_size=128, embed_dim=32,
                  seq_proj_dim=32, num_heads=4, num_layers=2, ffn_dim=64,
                  pe_max_len=64, max_cache_len=64, device=str(device), log_level=50,
                  use_gradient_checkpointing=False, use_kv_cache=True, use_flash_attention=False,
                  use_pytorch_sdpa=not args.manual, num_kv_heads=args.kv_heads,
                  use_moe=args.moe, num_experts=4, moe_top_k=2)
    model = CevahirNeuralNetwork(**config).to(device)
    stats = ModelProfiler.count_parameters(model, log=False)
    data = torch.randint(1, 128, (2, 17), device=device)
    train_x, train_y = data[:, :-1], data[:, 1:]
    val = torch.randint(1, 128, (2, 17), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(fn):
        sync(); start = time.perf_counter(); result = fn(); sync()
        return result, time.perf_counter() - start

    def aux_loss():
        getter = getattr(model, "get_and_reset_moe_loss", None)
        return getter() if getter else None

    def objective(x, y):
        logits = model(x)[0]
        ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.reshape(-1))
        auxiliary = aux_loss()
        return ce if auxiliary is None else ce + auxiliary

    def clear_cache():
        model.clear_kv_cache()

    # Warm up both execution paths; initialization and optimizer state allocation
    # are excluded from the measured steady-state training loop.
    model.train()
    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True); objective(train_x, train_y).backward(); optimizer.step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forwards, backwards, train_times, grad_norms = [], [], [], []
    nonfinite = 0
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        sync(); start = time.perf_counter()
        loss, forward = timed(lambda: objective(train_x, train_y))
        _, backward = timed(loss.backward)
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        norm = torch.stack([g.float().norm().square() for g in grads]).sum().sqrt().item()
        nonfinite += int(not torch.isfinite(loss).item()) + sum(int((~torch.isfinite(g)).sum()) for g in grads)
        optimizer.step(); sync()
        train_times.append(time.perf_counter() - start)
        forwards.append(forward); backwards.append(backward); grad_norms.append(norm)
    model.eval()
    with torch.no_grad():
        validation_loss = torch.nn.functional.cross_entropy(model(val[:, :-1])[0].flatten(0, 1), val[:, 1:].reshape(-1)).item()
        aux_loss()
        prefill, decode = [], []
        for _ in range(args.steps):
            clear_cache()
            _, duration = timed(lambda: model(train_x[:1], use_cache=True, cache_position=torch.arange(16, device=device)))
            prefill.append(duration)
            def decode_tokens():
                for t in range(8):
                    model(train_x[:1, t:t+1], use_cache=True, cache_position=torch.tensor([16 + t], device=device))
            _, duration = timed(decode_tokens)
            decode.append(duration)
        clear_cache()
        full = model(train_x[:1])[0]
        pieces = [model(train_x[:1, t:t+1], use_cache=True, cache_position=torch.tensor([t], device=device))[0] for t in range(16)]
        cache_error = (full - torch.cat(pieces, dim=1)).abs().max().item()
        cache = [layer.attn.kv_cache for layer in model.layers]
        used_cache_bytes = sum((c.key_cache[:, :, :c.cache_len].numel() + c.value_cache[:, :, :c.cache_len].numel()) * c.key_cache.element_size() for c in cache)
        allocated_cache_bytes = sum((c.key_cache.numel() + c.value_cache.numel()) * c.key_cache.element_size() for c in cache)
        clear_cache()

    # Separate diagnostic pass. Manual attention returns weights; this pass is not
    # included in the SDPA timing numbers.
    attention_entropies, router_entropies, utilizations = [], [], []
    handles, linear_flops = [], [0]
    def count_linear(module, inputs, output):
        linear_flops[0] += 2 * output.numel() * module.in_features
    def capture_router(module, inputs, output):
        weights, indices, logits = output
        p = torch.softmax(logits.detach().float(), -1)
        router_entropies.append(float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean()))
        counts = torch.bincount(indices.reshape(-1), minlength=logits.shape[-1]).float()
        utilizations.append((counts / counts.sum()).tolist())
    def capture_attention(module, inputs, output):
        if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
            p = output[1].detach().float()
            attention_entropies.append(float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean()))
    old_routes = []
    for layer in model.layers:
        old_routes.append(layer.attn.use_pytorch_sdpa)
        layer.attn.use_pytorch_sdpa = False
        handles.append(layer.attn.register_forward_hook(capture_attention))
        if hasattr(layer.ffn, "router"):
            handles.append(layer.ffn.router.register_forward_hook(capture_router))
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(count_linear))
    diagnostic_kwargs = {}
    if "return_attention_weights" in inspect.signature(model.forward).parameters:
        diagnostic_kwargs["return_attention_weights"] = True
    with torch.no_grad():
        model(train_x, **diagnostic_kwargs)
        aux_loss()
    for handle in handles:
        handle.remove()
    for layer, route in zip(model.layers, old_routes):
        layer.attn.use_pytorch_sdpa = route
    active = stats.total
    for layer in model.layers:
        if hasattr(layer.ffn, "experts"):
            expert_size = sum(p.numel() for p in layer.ffn.experts[0].parameters())
            active -= (len(layer.ffn.experts) - layer.ffn.top_k) * expert_size
    buffer = io.BytesIO()
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config}, buffer)
    tokens = train_x.numel()
    attn_flops = config["num_layers"] * 4 * train_x.shape[0] * config["num_heads"] * 16 * 16 * 8
    source_hash = hashlib.sha256()
    for path in sorted((source / "src").rglob("*.py")):
        source_hash.update(path.relative_to(source).as_posix().encode()); source_hash.update(path.read_bytes())
    return {
        "schema_version": 1, "label": args.label,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": str(device), "threads": args.threads},
        "configuration": config, "seed": args.seed, "warmup": args.warmup, "measurement_steps": args.steps,
        "source_sha256": source_hash.hexdigest(),
        "data": {"kind": "seeded synthetic tokens", "sha256": hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest(), "language_quality_evaluation": False},
        "metrics": {
            "total_parameters": stats.total, "trainable_parameters": stats.trainable,
            "active_parameters_per_token_estimate": active,
            "forward_flops_per_token_estimate": (linear_flops[0] + attn_flops) / tokens,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "kv_cache_bytes_per_token": used_cache_bytes / 16, "kv_cache_allocated_bytes": allocated_cache_bytes,
            "prefill_tokens_per_second": 16 / statistics.median(prefill),
            "decode_tokens_per_second": 8 / statistics.median(decode),
            "training_tokens_per_second": tokens / statistics.median(train_times),
            "forward_latency_ms": statistics.median(forwards) * 1000,
            "backward_latency_ms": statistics.median(backwards) * 1000,
            "validation_loss": validation_loss, "perplexity": math.exp(validation_loss),
            "gradient_norm": statistics.mean(grad_norms), "nan_inf_occurrences": nonfinite,
            "router_expert_utilization": utilizations or None,
            "router_entropy": statistics.mean(router_entropies) if router_entropies else None,
            "attention_entropy": statistics.mean(attention_entropies) if attention_entropies else None,
            "checkpoint_bytes": buffer.tell(), "cache_equivalence_max_abs_error": cache_error,
        },
        "notes": ["GPU metrics are null on CPU; router metrics null for dense FFN.",
                  "Active parameters estimate includes dense/tied weights plus top-k experts, not unique embedding row accesses.",
                  "FLOPs counts observed linear work and dense QK/AV matmuls; excludes normalization, nonlinearity, rotary and optimizer work.",
                  "Attention entropy is a separate manual diagnostic. No claim about trained language quality or GPU speed."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--label", default="current")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if min(args.steps, args.threads) < 1 or args.warmup < 0:
        parser.error("steps/threads must be positive; warmup non-negative")
    result = compare(*(json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)) if args.compare else run(args)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(str(path.resolve()))


if __name__ == "__main__":
    main()
