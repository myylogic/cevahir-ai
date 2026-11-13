"""Shared content identities for cache preparation and training consumption."""
import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_directory_digest(directory):
    root = Path(directory)
    if not root.exists():
        return ""
    entries = [(path.relative_to(root).as_posix(), file_digest(path))
               for path in sorted(root.rglob("*"))
               if path.is_file() and path.suffix.lower() in {".json", ".txt", ".docx"}]
    if not entries:
        return ""
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def tokenizer_digest(tokenizer_core):
    manager = getattr(tokenizer_core, "tokenizer", None)
    merges_path = getattr(tokenizer_core, "merges_path", None) or getattr(manager, "merges_file", None)
    # Runtime knobs (batch sizes, device, length limits, caches, logging) do not change how text maps
    # to ids. Hashing them made the identity depend on the caller: prepare_cache, train.py and the
    # inference facade pass different batch sizes, so a cache or checkpoint built with byte-identical
    # vocab/merges was rejected. Sequence length is part of the cache key separately (cache_v3).
    bpe_config = {key: value for key, value in (getattr(manager, "config", None) or {}).items()
                  if key not in RUNTIME_ONLY_TOKENIZER_KEYS}
    payload = {
        "identity_version": 3,
        "vocab": tokenizer_core.get_vocab(),
        "merges": file_digest(merges_path) if merges_path else None,
        "bpe_config": bpe_config,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


# Keys of the tokenizer manager config that only affect throughput, devices or diagnostics.
RUNTIME_ONLY_TOKENIZER_KEYS = frozenset({
    "batch_size", "tokenizer_batch_size", "batch_merge_size", "adaptive_batch", "batch_log_interval",
    "gpu_batch_size", "gpu_memory_fraction", "use_gpu", "use_tpu", "device", "parallel_workers",
    "num_workers", "max_seq_length", "cache_size", "chunk_size", "pair_stats_chunk_size",
    "memory_monitoring", "memory_threshold", "gc_interval", "future_timeout",
    "log_interval", "progress_log_interval", "log_level", "verbose", "show_progress",
})


# Layout version of the formatted records that prepare_cache.py writes and training reads. It is part
# of every cache key (training_system/data_cache.py and v3/data/cache_v3.py), so a cache written under an
# older layout is rebuilt instead of silently reused. Bump it whenever cached records change meaning.
# The tokenizer identity above is deliberately separate: checkpoints do not depend on the cache layout.
#   1: BOS ... EOS around every record, including each chunk of a long document.
#   2: BOS only where a document starts and EOS only where it ends; continuation chunks carry neither.
CACHE_FORMAT_VERSION = 2


def cache_format_tag():
    """Cache-key component naming the formatted-record layout."""
    return f"cache_format_{CACHE_FORMAT_VERSION}"
