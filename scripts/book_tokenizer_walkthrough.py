"""Read-only, real-asset tokenizer examples for the technical book.

Run at repository root; --write explicitly records the educational evidence.
No tokenizer training, vocabulary update, model loading or network access.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizer_management.core.tokenizer_core import TokenizerCore, TokenizerCoreError

OUTPUT = ROOT / "docs/book/evidence/tokenizer_walkthrough.json"
ASSETS = ("data/vocab_lib/vocab.json", "data/merges_lib/merges.txt")


def asset_fingerprints():
    # Portable across Git text checkouts; not the runtime checkpoint digest.
    return {p: hashlib.sha256((ROOT/p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in ASSETS}


def make_core(**overrides):
    return TokenizerCore({"vocab_path": str(ROOT/ASSETS[0]), "merges_path": str(ROOT/ASSETS[1]),
                          "use_gpu": False, "read_only": True,
                          "bpe_config": {"auto_vocab_update": False, **overrides}})


def probe(core, text, **options):
    manager = core.tokenizer
    tokens, ids, stats = core.encode_with_stats(text, **options)
    # Diagnostic re-evaluation of existing stages, not a separate tokenizer.
    normalized = manager._normalize_text(manager._strip_or_map_tags(text, map_to_special=False),
                                         lowercase=manager.config.get("normalize_lowercase", False))
    pre = manager.pretokenizer.tokenize(normalized)
    per_token = [{"piece": t, "ids": manager.encoder._encode_token_to_ids(t)} for t in tokens]
    assert [i for item in per_token for i in item["ids"]] == ids
    return {"input": text, "options": options, "after_manager_normalization": normalized,
            "pretokenizer_output": pre, "pieces": tokens, "ids": ids,
            "id_spellings": [manager.decoder.reverse_vocab[i] for i in ids],
            "per_piece_ids": per_token, "stats": stats, "decoded": core.decode(ids)}


def run():
    before = {p: (ROOT/p).read_bytes() for p in ASSETS}
    core = make_core()
    cases = {
        "inference_explicit_no_specials": probe(core, "Merhaba dünya!", mode="inference", add_special_tokens=False),
        "inference_defaults": probe(core, "Merhaba dünya!"),
        "train_defaults": probe(core, "Merhaba dünya!", mode="train"),
        "new_word": probe(core, "cevahirleştirilemeyenlerden", add_special_tokens=False),
        "normalization": probe(core, "  [USER] Merhaba\t  dünya!  ", add_special_tokens=False),
        "emoji_warn": probe(core, "Merhaba 🧬 dünya!", add_special_tokens=False),
    }
    assert cases["inference_explicit_no_specials"]["decoded"] == "Merhaba dünya!"
    assert cases["normalization"]["decoded"] == "Merhaba dünya!"
    assert cases["emoji_warn"]["stats"]["unk_count"] == 0
    assert "🧬" not in cases["emoji_warn"]["decoded"]
    plain = make_core(cleanup_punctuation_spaces=False)
    ids = cases["inference_explicit_no_specials"]["ids"]
    assert plain.encode("Merhaba dünya!", add_special_tokens=False)[1] == ids
    strict = make_core(text_loss_policy="error")
    try:
        strict.encode("Merhaba 🧬 dünya!", add_special_tokens=False)
    except TokenizerCoreError as error:
        strict_result = {"outer_exception": type(error).__name__,
                         "cause": type(error.__cause__).__name__,
                         "dropped_codepoints": error.__cause__.dropped_codepoints}
    else:
        raise AssertionError("Strict preprocessing loss was not rejected")
    batch_inputs = ["Merhaba", "🧬", "dünya"]
    batch = strict.batch_encode(batch_inputs, add_special_tokens=False)
    assert len(batch) == 2
    try:
        strict.batch_encode(batch_inputs, add_special_tokens=False, skip_invalid=False)
    except TokenizerCoreError:
        batch_raises = True
    else:
        raise AssertionError("Strict non-skipping batch must raise")
    unchanged = all((ROOT/p).read_bytes() == original for p, original in before.items())
    assert unchanged
    return {"kind": "executed_book_examples", "date": "2026-09-21", "device": "CPU",
            "asset_sha256_utf8_lf": asset_fingerprints(), "asset_bytes_unchanged": unchanged,
            "vocabulary_entries": len(core.get_vocab()), "loaded_merge_rules": len(core.get_merges()),
            "cases": cases, "same_ids_without_punctuation_cleanup": plain.decode(ids),
            "strict_preprocessing_loss": strict_result,
            "batch": {"inputs": batch_inputs, "returned_items": len(batch),
                      "decoded": [strict.decode(ids) for _, ids in batch],
                      "skip_invalid_false_raises": batch_raises},
            "limits": ["Worked examples, not a multilingual quality or throughput benchmark",
                       "Preprocessing stages re-evaluated diagnostically through the same methods",
                       "Asset fingerprints normalize line endings; not a replacement for tokenizer_digest",
                       "No GPU execution or tokenizer/model training"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    result = run()
    if args.write:
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    else:
        assert result == json.loads(OUTPUT.read_text(encoding="utf-8"))
    print(json.dumps({"examples": len(result["cases"]), "asset_bytes_unchanged": True,
                      "vocabulary_entries": result["vocabulary_entries"],
                      "loaded_merge_rules": result["loaded_merge_rules"], "mode": "record" if args.write else "check"}))


if __name__ == "__main__":
    main()
