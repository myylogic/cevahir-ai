"""Field classifications: which configuration fields enter which key.

ARCH_FIELD_CLASS_V1 classifies every ``ModelArchConfig`` field (and the extras that
``normalize_model_config`` adds). A test fails when the dataclass gains an unclassified field.
  fdf    function or weight layout  -> arch:cfg-v1
  train  only matters while training -> the run specification
  serve  only matters while serving  -> serveprof:cfg-v1
  extra  added by the normaliser; ``architecture_version`` becomes the family version, the rest
         (device, log level, ...) identify nothing

TOKENIZER_KEY_CLASS_V1 classifies the legacy BPE manager's config keys. Only ``semantic`` keys can
change how text maps to ids, so only they enter tok:bpe-sem-v1. An unclassified key counts as
semantic: a new key can change the key, never cause a collision.
"""

# name -> (class, type in the typed projection)
ARCH_FDF_V1 = {
    "vocab_size": "int", "embed_dim": "int", "seq_proj_dim": "int", "num_heads": "int",
    "num_kv_heads": "int", "num_layers": "int", "ffn_dim": "int",
    "pre_norm": "bool", "use_rmsnorm": "bool", "use_swiglu": "bool", "use_qk_norm": "bool",
    "parallel_residual": "bool", "causal_mask": "bool",
    "pe_mode": "str", "rope_scaling_type": "str?", "rope_scaling_factor": "float?",
    "rope_original_max_len": "int?", "sliding_window": "int?", "attn_logit_cap": "float",
    "tie_weights": "bool", "logit_soft_cap": "float",
    "use_moe": "bool", "num_experts": "int?", "moe_top_k": "int?",
}
ARCH_TRAIN_V1 = ("dropout", "drop_path_rate", "pe_dropout", "use_gradient_checkpointing",
                 "use_advanced_checkpointing", "checkpointing_strategy", "moe_jitter_noise",
                 "moe_load_balance_alpha")
ARCH_SERVE_V1 = ("quantization_type", "max_seq_length", "use_flash_attention", "use_pytorch_sdpa",
                 "use_kv_cache", "max_cache_len", "kv_eviction_strategy", "kv_num_sink_tokens")
ARCH_EXTRA_V1 = ("config_version", "architecture_version", "pe_max_len", "device", "log_level",
                 "learning_rate", "precision")

ARCH_FIELD_CLASS_V1 = {
    **{name: "fdf" for name in ARCH_FDF_V1},
    **{name: "train" for name in ARCH_TRAIN_V1},
    **{name: "serve" for name in ARCH_SERVE_V1},
    **{name: "extra" for name in ARCH_EXTRA_V1},
}

_SEMANTIC = ("append_eos", "include_morphology", "include_sep", "include_syllables", "include_whole_words",
             "lowercase", "normalize", "normalize_lowercase", "digit_chars", "punctuation_chars",
             "max_onset_size", "max_suffix_chain", "max_suffix_count", "max_syllable_length", "max_syllables",
             "min_nucleus_size", "min_syllable_len", "min_word_length", "unk_handling", "use_syllables_for_oov",
             "protect_specials", "keep_specials_in_train", "max_tokens_per_text", "overlap_tokens",
             "min_chunk_size", "streaming_mode")
_TRAINING = ("max_iter", "target_merges", "min_frequency", "max_vocab_size", "min_vocab_size",
             "initial_vocab_ratio", "vocab_size_buffer", "vocab_strategy", "auto_vocab_update", "precompute_pairs")
_DECODE = ("capitalize_sentence", "remove_tags", "collapse_spaces", "prefer_mode", "remove_specials")
_DIAGNOSTIC = ("vocab_growth_alert", "max_unk_ratio", "text_loss_policy")
_MODE = ("read_only",)
_PATH = ("vocab_path", "merges_path", "data_dir", "vocab_file", "merges_file")
# Copy of training_system.cache_identity.RUNTIME_ONLY_TOKENIZER_KEYS; a test keeps the two equal.
_RUNTIME = ("batch_size", "tokenizer_batch_size", "batch_merge_size", "adaptive_batch", "batch_log_interval",
            "gpu_batch_size", "gpu_memory_fraction", "use_gpu", "use_tpu", "device", "parallel_workers",
            "num_workers", "max_seq_length", "cache_size", "chunk_size", "pair_stats_chunk_size",
            "memory_monitoring", "memory_threshold", "gc_interval", "future_timeout",
            "log_interval", "progress_log_interval", "log_level", "verbose", "show_progress")

TOKENIZER_KEY_CLASS_V1 = {
    **{name: "semantic" for name in _SEMANTIC}, **{name: "training" for name in _TRAINING},
    **{name: "decode" for name in _DECODE}, **{name: "diagnostic" for name in _DIAGNOSTIC},
    **{name: "mode" for name in _MODE}, **{name: "path" for name in _PATH},
    **{name: "runtime" for name in _RUNTIME},
}


def tokenizer_semantic_config(config):
    """The config keys that can change id semantics (unclassified keys included)."""
    return {key: value for key, value in (config or {}).items()
            if TOKENIZER_KEY_CLASS_V1.get(key, "semantic") == "semantic"}
