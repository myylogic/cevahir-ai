"""Execute small, temporary Cevahir checkpoint examples for book chapter 7."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from model_management.config_schema import tiny_model_config
from model_management.model_manager import ModelManager
from model_management.model_saver import ModelSaver
from model_management.model_loader import ModelLoader
from model_management.checkpoint_contract import validate_state_shapes

OUTPUT = ROOT / "docs/book/evidence/lifecycle_walkthrough.json"


def manager(*, parallel=False, reverse_tokens=False):
    obj = ModelManager(tiny_model_config(vocab_size=16, embed_dim=8, num_heads=2,
        num_layers=1, ffn_dim=16, parallel_residual=parallel, logit_soft_cap=0.))
    obj.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    obj.model.eval()
    obj.tokenizer = SimpleNamespace(get_vocab=lambda: {
        f"token_{i}": 15-i if reverse_tokens else i for i in range(16)})
    obj.optimizer = torch.optim.AdamW(obj.model.parameters(), lr=.003)
    return obj


def step(obj, ids, targets):
    obj.optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(obj.model(ids)[0].reshape(-1, 16), targets.flatten())
    loss.backward()
    obj.optimizer.step()
    return float(loss.detach())


def unchanged(model, before):
    return all(torch.equal(t, before[k]) for k,t in model.state_dict().items())


def rejected_restore(target, path):
    before = copy.deepcopy(target.model.state_dict())
    try:
        target.load(str(path), weights_only=True)
    except RuntimeError as exc:
        assert unchanged(target.model, before)
        return {"rejected": True, "parameters_unchanged": True,
                "cause": str(exc.__cause__)}
    raise AssertionError("Expected a rejected checkpoint")


def run():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20260921)
    try:
        with tempfile.TemporaryDirectory(prefix="cevahir-book-lifecycle-") as directory:
            folder = Path(directory)
            source = manager()
            ids, targets = torch.tensor([[1,2,3]]), torch.tensor([[2,3,4]])
            step(source, ids, targets)
            path = source.save(str(folder / "manager.pth"), epoch=2)
            with torch.no_grad():
                expected = source.model(ids)[0]
            resumed = manager()
            # None keeps framework restricted-loading default, while the manager
            # additionally restores optimizer state because bool(None) is false.
            resumed.load(path, weights_only=None)
            fresh_optimizer = manager()
            fresh_optimizer.load(path, weights_only=True)
            with torch.no_grad():
                actual = resumed.model(ids)[0]
            torch.testing.assert_close(actual, expected)
            assert len(resumed.optimizer.state) > 0 and not fresh_optimizer.optimizer.state
            source_next = step(source, ids, targets)
            resumed_next = step(resumed, ids, targets)
            step(fresh_optimizer, ids, targets)
            continuing_error = max(float((p-resumed.model.state_dict()[k]).abs().max())
                for k,p in source.model.state_dict().items())
            reset_error = max(float((p-fresh_optimizer.model.state_dict()[k]).abs().max())
                for k,p in source.model.state_dict().items())
            assert continuing_error == 0. and reset_error > 0.
            with torch.no_grad():
                resumed.model(ids, use_cache=True)
            cache_before = resumed.model.layers[0].attn.kv_cache.seen_tokens
            resumed.load(path, weights_only=True)
            cache_after = resumed.model.layers[0].attn.kv_cache.seen_tokens
            assert cache_before == 3 and cache_after == 0
            mismatches = {"architecture": rejected_restore(manager(parallel=True), path),
                          "tokenizer": rejected_restore(manager(reverse_tokens=True), path)}
            sidecar_path = ModelSaver.save_checkpoint(source.model, optimizer=source.optimizer,
                epoch=3, save_dir=directory, filename="sidecar.pth", create_latest_marker=False)
            raw = Path(sidecar_path).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            saved = torch.load(sidecar_path, map_location="cpu", weights_only=True)
            sidecar_matches = Path(sidecar_path+".sha256").read_text() == digest
            embedded_matches = saved["metadata"]["sha256"] == digest
            assert sidecar_matches and not embedded_matches
            # Mutate only this freshly generated teaching checkpoint.
            with Path(sidecar_path).open("ab") as stream:
                stream.write(b"book-corruption-probe")
            try:
                ModelLoader.load_checkpoint_raw(sidecar_path, device="cpu", weights_only=True)
            except Exception as exc:
                checksum_rejected = type(exc).__name__
            else:
                raise AssertionError("Sidecar failed to reject changed bytes")
            linear = torch.nn.Linear(2,1,bias=False)
            dtype_state = {"weight": torch.tensor([[1.25,2.5]],dtype=torch.float64)}
            validate_state_shapes(linear, dtype_state)
            linear.load_state_dict(dtype_state, strict=True)
            assert linear.weight.dtype == torch.float32
            return {
                "roundtrip": {"saved_epoch":2,"loaded_epoch":resumed.config["current_epoch"],
                    "logits_shape":list(actual.shape),"max_logit_error":float((actual-expected).abs().max()),
                    "optimizer_state_entries":len(resumed.optimizer.state),
                    "next_loss_uninterrupted":source_next,"next_loss_resumed":resumed_next,
                    "next_step_parameter_max_error":continuing_error,
                    "fresh_optimizer_next_step_parameter_max_difference":reset_error},
                "cache": {"seen_before_load":cache_before,"seen_after_load":cache_after},
                "rejected_restores":mismatches,
                "checksums": {"manager_save_created_sidecar":Path(path+".sha256").exists(),
                    "save_checkpoint_sidecar_matches_final_bytes":sidecar_matches,
                    "embedded_sha256_matches_final_bytes":embedded_matches,
                    "altered_scratch_file_rejected_with":checksum_rejected},
                "dtype": {"source":"float64","destination":"float32","strict_shape_check_passed":True,
                    "loaded_values":linear.weight.detach().tolist()}}
    finally:
        torch.set_num_threads(old_threads)


def compare(actual, expected):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected: compare(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected): compare(a,b)
    elif isinstance(expected,float):
        assert math.isclose(actual,expected,rel_tol=2e-5,abs_tol=2e-6),(actual,expected)
    else: assert actual==expected,(actual,expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    logging.disable(logging.CRITICAL)
    results=run()
    if args.write:
        evidence={"date":"2026-09-21","kind":"executed_book_examples",
            "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu"},
            "seed":20260921,"cases":results,
            "limits":["Only synthetic tiny models and automatically removed temporary checkpoint files",
                "Same-process CPU, no dropout, fixed batches; not full V2 epoch resume or cross-hardware reproducibility",
                "No production model, tokenizer asset, or historical research record modified"]}
        OUTPUT.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    else:
        compare(results,json.loads(OUTPUT.read_text(encoding="utf-8"))["cases"])
    print(json.dumps({"cases":len(results),"mode":"write" if args.write else "check","scratch_checkpoints_removed":True}))


if __name__ == "__main__":
    main()
