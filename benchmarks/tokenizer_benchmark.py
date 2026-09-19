"""CPU diagnostic benchmark of the existing Cevahir tokenizer and vocabulary.

The short, hand-annotated suffix probes are not a linguistic evaluation corpus.
No training, vocabulary updates, or alternate tokenization backend is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS = {
    "turkish": [
        "Merhaba dünya! Bugün Türkçe bir metin okuyoruz.",
        "Evlerimizden arkadaşlarımıza kitaplarımızı götürüyoruz.",
        "İstanbul'da Iğdır'ı, Şırnak'ı ve Çanakkale'yi konuştuk.",
        "İ ı Ş ş Ğ ğ Ü ü Ö ö Ç ç I i",
        "Çocukların öğretmenleri bilimsel araştırmaları destekliyor.",
        "  iki   boşluk\nve\tbir sekme  ",
        "I\u0307stanbul ve s\u0327eker",
        "Yeni sözcük: zırıldatılamayanlarımızdan; emoji: 🧬🚀; 漢字.",
    ],
    "english": [
        "Hello world! Today we are reading a short English text.",
        "The children brought their books from our houses.",
        "Researchers compare reproducible experiments and measurements.",
        "Uppercase ABC and lowercase abc remain different.",
        "  two   spaces\nand\tone tab  ",
        "A new word: uncharacteristically; emoji: 🧬🚀; 漢字.",
    ],
}
MORPHEME_PROBES = {
    "turkish": ["ev|ler|imiz|den", "kitap|lar|ımız|ı", "göz|lük|çü", "çocuk|lar|ın"],
    "english": ["book|s", "read|ing", "help|ful", "kind|ness"],
}


def normalized(text):
    """Only NFC/whitespace normalization; case and punctuation stay significant."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emitted_surfaces(ids, inverse):
    return [inverse.get(i, "<MISSING_ID>") for i in ids]


def morpheme_probe(manager, annotated, inverse):
    parts = annotated.split("|")
    word = "".join(parts)
    _, ids = manager.encode(word, mode="inference", add_special_tokens=False)
    surfaces = emitted_surfaces(ids, inverse)
    pieces = [t.replace("</w>", "") for t in surfaces if not t.startswith("<")]
    reconstructed = "".join(pieces)
    aligned = reconstructed == word
    position = 0
    spans = []
    for piece in pieces:
        spans.append((position, position + len(piece)))
        position += len(piece)
    boundaries = {sum(map(len, parts[:i])) for i in range(1, len(parts))}
    token_boundaries = {end for _, end in spans[:-1]}
    suffix_counts = []
    start = len(parts[0])
    for suffix in parts[1:]:
        end = start + len(suffix)
        suffix_counts.append(sum(a < end and b > start for a, b in spans) if aligned else None)
        start = end
    return {
        "annotation": annotated, "text": word, "ids": ids, "emitted_tokens": surfaces,
        "surface_alignment": aligned,
        "morpheme_boundaries": len(boundaries),
        "preserved_boundaries": len(boundaries & token_boundaries) if aligned else None,
        "boundary_preservation": len(boundaries & token_boundaries) / len(boundaries) if aligned else None,
        "tokens_overlapping_each_suffix": suffix_counts,
        "note": "Token overlap is counted; a token spanning root and suffix counts once. Unalignable outputs are null.",
    }


def run_benchmark(vocab_path=None, merges_path=None, repeats=5):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    from tokenizer_management.bpe.bpe_manager import BPEManager

    vocab_path = Path(vocab_path or ROOT / "data/vocab_lib/vocab.json").resolve()
    merges_path = Path(merges_path or ROOT / "data/merges_lib/merges.txt").resolve()
    before_hashes = {"vocab": digest(vocab_path), "merges": digest(merges_path)}
    t0 = time.perf_counter()
    manager = BPEManager(vocab_file=str(vocab_path), merges_file=str(merges_path),
                         use_gpu=False, config={"read_only": True})
    initialization_seconds = time.perf_counter() - t0
    vocab = manager.get_vocab()
    inverse = {int(meta["id"]): token for token, meta in vocab.items()}
    unknown_ids = {int(vocab[t]["id"]) for t in ("<UNK>", "<UNK_CHAR>") if t in vocab}
    language_results = {}
    for language, texts in CORPUS.items():
        examples, usage = [], Counter()
        for text in texts:
            started = time.perf_counter()
            _, ids = manager.encode(text, mode="inference", add_special_tokens=False)
            cold_encode_seconds = time.perf_counter() - started
            decoded = manager.decode(ids)
            usage.update(ids)
            examples.append({
                "text": text, "ids": ids, "emitted_tokens": emitted_surfaces(ids, inverse),
                "decoded": decoded, "exact_round_trip": decoded == text,
                "normalized_round_trip": normalized(decoded) == normalized(text),
                "unknown_tokens": sum(i in unknown_ids for i in ids),
                "cold_encode_seconds": cold_encode_seconds,
                "missing_input_characters": dict(Counter(normalized(text)) - Counter(normalized(decoded))),
            })
        started = time.perf_counter()
        for _ in range(repeats):
            for text in texts:
                manager.encode(text, mode="inference", add_special_tokens=False)
        encode_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(repeats):
            for example in examples:
                manager.decode(example["ids"])
        decode_seconds = time.perf_counter() - started
        tokens = sum(len(e["ids"]) for e in examples)
        words = sum(len(re.findall(r"\w+(?:['’]\w+)*", text, re.UNICODE)) for text in texts)
        chars = sum(map(len, texts))
        probes = [morpheme_probe(manager, p, inverse) for p in MORPHEME_PROBES[language]]
        aligned = [p for p in probes if p["surface_alignment"]]
        known_boundary_count = sum(p["morpheme_boundaries"] for p in aligned)
        suffix_counts = [n for p in aligned for n in p["tokens_overlapping_each_suffix"]]
        language_results[language] = {
            "samples": len(texts), "words": words, "tokens": tokens,
            "token_fertility": tokens / words if words else None,
            "characters_per_token": chars / tokens if tokens else None,
            "utf8_bytes_per_token": sum(len(t.encode("utf-8")) for t in texts) / tokens if tokens else None,
            "exact_round_trip_accuracy": sum(e["exact_round_trip"] for e in examples) / len(examples),
            "normalized_round_trip_accuracy": sum(e["normalized_round_trip"] for e in examples) / len(examples),
            "unknown_tokens": sum(e["unknown_tokens"] for e in examples),
            "vocabulary_utilization": len(usage) / len(vocab), "used_token_count": len(usage),
            "warm_encode_tokens_per_second": tokens * repeats / encode_seconds,
            "warm_encode_chars_per_second": chars * repeats / encode_seconds,
            "warm_decode_tokens_per_second": tokens * repeats / decode_seconds,
            "suffix_fragmentation_mean_overlapping_tokens": sum(suffix_counts) / len(suffix_counts) if suffix_counts else None,
            "morpheme_boundary_preservation": sum(p["preserved_boundaries"] for p in aligned) / known_boundary_count if known_boundary_count else None,
            "aligned_morpheme_probe_fraction": len(aligned) / len(probes),
            "morpheme_probes": probes, "examples": examples,
        }
    unicode_probes = {}
    for ch in "İıŞşĞğÜüÖöÇçIi":
        _, ids = manager.encode(ch, mode="inference", add_special_tokens=False)
        decoded = manager.decode(ids)
        unicode_probes[ch] = {"ids": ids, "decoded": decoded, "correct": decoded == ch}
    fallback_probes = {}
    for text in ("🧬", "漢字", "İ", "\u0378", "", "   "):
        _, pipeline_ids = manager.encode(text, mode="inference", add_special_tokens=False)
        direct_ids = manager.encoder.encode_sequence([text]) if text else []
        fallback_probes[text] = {
            "pipeline_ids": pipeline_ids, "pipeline_decoded": manager.decode(pipeline_ids),
            "encoder_ids": direct_ids,
            "pipeline_unknown_count": sum(i in unknown_ids for i in pipeline_ids),
            "encoder_unknown_count": sum(i in unknown_ids for i in direct_ids),
            "dropped_before_encoding": bool(text.strip()) and not pipeline_ids,
        }
    after_hashes = {"vocab": digest(vocab_path), "merges": digest(merges_path)}
    if before_hashes != after_hashes:
        raise RuntimeError("Tokenizer benchmark modified vocabulary artifacts")
    return {
        "schema_version": 1, "kind": "tokenizer_diagnostic", "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu"},
        "artifacts": {"vocab_path": str(vocab_path), "merges_path": str(merges_path), "sha256": before_hashes, "unchanged": True},
        "policy": {"pipeline": "BPEManager", "mode": "inference", "add_special_tokens": False, "repeats": repeats,
                   "normalization": "NFC and collapsed/trimmed whitespace only", "training_performed": False,
                   "scope": "Small hand-authored diagnostic probes; not a model or linguistic quality claim.",
                   "throughput": "Warm repeated corpus, including encoder cache effects; timing is noisy on shared CPU."},
        "vocab_size": len(vocab), "initialization_seconds": initialization_seconds,
        "languages": language_results, "turkish_unicode_probes": unicode_probes,
        "fallback_probes": fallback_probes,
    }


def comparison(baseline, experiment):
    keys = ("token_fertility", "characters_per_token", "utf8_bytes_per_token", "exact_round_trip_accuracy",
            "normalized_round_trip_accuracy", "unknown_tokens", "vocabulary_utilization",
            "warm_encode_tokens_per_second", "warm_decode_tokens_per_second")
    return {language: {key: experiment["languages"][language][key] - baseline["languages"][language][key]
                       for key in keys} for language in experiment["languages"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/tokenizer_baseline.json")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    result = run_benchmark(repeats=args.repeats)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        result["comparison"] = {"baseline": str(args.baseline), "experiment": str(args.output), "delta": comparison(baseline, result)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "languages": {k: {m: v[m] for m in
          ("token_fertility", "exact_round_trip_accuracy", "normalized_round_trip_accuracy", "unknown_tokens")}
          for k, v in result["languages"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
