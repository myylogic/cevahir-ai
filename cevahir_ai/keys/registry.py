"""The scheme registry: every key scheme the system knows, with its status and derivation.

status:
  primary     canonical content-derived key of the roof
  legacy      an identity that exists in persisted artefacts today; computed by calling the
              original code (or a re-implementation proven equal), never recomputed in place
  signature   structure only (used for resume compatibility), not content
  occurrence  identifies one event (a run), not content
``torch`` says whether deriving the key needs torch. ``newline`` is the text policy of the inputs:
``lf`` (TEXT-LF-1), ``bytes`` (raw bytes), ``n/a`` (structured payload).
"""
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Scheme:
    kind: str
    scheme: str
    status: str
    digest_len: int
    torch: bool
    newline: str
    derive: str
    summary: str

    @property
    def name(self):
        return f"{self.kind}:{self.scheme}"


def _table(*rows):
    return {(row[0], row[1]): Scheme(*row) for row in rows}


_S, _A = "cevahir_ai.keys.schemes", "cevahir_ai.adapters"
_T = "cevahir_ai.keys.torchkeys"

SCHEMES = _table(
    ("file", "bytes-v1", "primary", 64, False, "bytes", f"{_S}.file:file_bytes_key",
     "sha256 of the raw bytes; equal to sha256sum and to .sha256 sidecars"),
    ("file", "utf8lf-v1", "primary", 64, False, "lf", f"{_S}.file:file_utf8lf_key",
     "sha256 of the TEXT-LF-1 bytes; equal to the book's utf8-lf fingerprints"),
    ("corpus", "files-v1", "primary", 64, False, "lf", f"{_S}.corpus:corpus_files_key",
     "sorted [relative path, file key] of the training documents"),
    ("corpus", "legacy-dd-v1", "legacy", 64, False, "bytes", f"{_A}.caches:corpus_legacy_dd_key",
     "training_system.cache_identity.data_directory_digest, called"),
    ("evalset", "files-v1", "primary", 64, False, "lf", f"{_S}.corpus:evalset_files_key",
     "as corpus:files-v1, for an evaluation set"),
    ("tok", "bpe-sem-v1", "primary", 64, False, "lf", f"{_S}.tok:tok_bpe_sem_key",
     "token->id map, ordered merges, specials and the id-semantic config keys of the legacy BPE"),
    ("tok", "bytebpe-v1", "primary", 64, False, "n/a", f"{_S}.tok:tok_bytebpe_key",
     "ByteLevelBPE.identity() verbatim (every CEV-11 artefact is already keyed by it)"),
    ("tok", "legacy-v3", "legacy", 64, True, "bytes", f"{_A}.tokenizers:tok_legacy_v3_key",
     "training_system.cache_identity.tokenizer_digest (identity_version 3), called on a live core"),
    ("cache", "recipe-v1", "primary", 64, False, "n/a", f"{_S}.cache:cache_recipe_key",
     "input-addressed: corpus key + tokenizer key + encoding profile + record format + builder"),
    ("cache", "md5-v2", "legacy", 32, False, "n/a", f"{_S}.cache:cache_md5_key",
     "the md5 in cached_data_<md5>_<datahash>.pkl"),
    ("split", "assign-v1", "primary", 64, False, "n/a", f"{_S}.cache:split_assign_key",
     "cache key + validation ratio + seed + method"),
    ("arch", "cfg-v1", "primary", 64, False, "n/a", f"{_S}.arch:arch_cfg_key",
     "typed projection of normalize_model_config onto the 24 function-defining fields"),
    ("arch", "ctor-v0", "legacy", 64, False, "n/a", f"{_S}.arch:arch_ctor_v0_key",
     "a stored model_config (constructor arguments + tie_weights) minus device"),
    ("serveprof", "cfg-v1", "primary", 64, False, "n/a", f"{_S}.arch:serve_profile_key",
     "serving-time fields (quantization, context length, KV cache, attention backend, dtype)"),
    ("wschema", "layout-v1", "primary", 64, False, "n/a", f"{_S}.ckpt:wschema_layout_key",
     "state_dict names + shapes + alias groups"),
    ("wschema", "sd-v1", "primary", 64, False, "n/a", f"{_S}.ckpt:wschema_sd_key",
     "state_dict names + shapes + dtypes + alias groups"),
    ("weights", "sha256-v1", "primary", 64, True, "n/a", f"{_T}.weights:weights_key",
     "per-tensor table: name, dtype, shape, alias, sha256 of the raw little-endian bytes"),
    ("weights", "cev11-fp32-v1", "legacy", 64, True, "n/a", f"{_T}.weights:weights_cev11_fp32_key",
     "CEV-11 model_sha256 (sorted names, shape, fp32-cast bytes), re-implemented and proven equal"),
    ("optim", "sig-v1", "signature", 64, False, "n/a", f"{_S}.ckpt:optim_sig_key",
     "optimizer structure: parameter-group sizes, hyperparameter names, per-parameter state names"),
    ("sched", "sig-v1", "signature", 64, False, "n/a", f"{_S}.ckpt:sched_sig_key",
     "scheduler type-specific state keys (the rule of checkpoint_contract.validate_scheduler_state)"),
    ("optim", "sha256-v1", "primary", 64, True, "n/a", f"{_T}.optim:optim_content_key",
     "full optimizer state, tensors hashed like weights"),
    ("sched", "sha256-v1", "primary", 64, True, "n/a", f"{_T}.optim:sched_content_key",
     "full scheduler state"),
    ("tstate", "sha256-v1", "primary", 64, True, "n/a", f"{_T}.optim:tstate_content_key",
     "remaining training state (loop, best metric, early-stop counter, RNG)"),
    ("ckpt", "members-v1", "primary", 64, False, "n/a", f"{_S}.ckpt:ckpt_members_key",
     "member keys: weights, arch, wschema, tok, optim, sched, tstate"),
    ("model", "triple-v1", "primary", 64, False, "n/a", f"{_S}.ckpt:model_triple_key",
     "the servable identity: weights + arch + canonical tokenizer"),
    ("runspec", "cfg-v1", "primary", 64, False, "n/a", f"{_S}.runs:runspec_key",
     "a training configuration with paths replaced by input keys"),
    ("run", "id-v1", "occurrence", 32, False, "n/a", f"{_S}.runs:new_run_key",
     "one training run (uuid7 as 32 hex)"),
    ("eval", "recipe-v1", "primary", 64, False, "n/a", f"{_S}.runs:eval_recipe_key",
     "model key + evaluation-set key + metric generation + decoding + seed"),
    ("code", "tree-v1", "primary", 64, False, "lf", f"{_S}.code:code_tree_key",
     "sorted [path, utf8-lf digest] of a declared file set"),
    ("env", "rt-v1", "primary", 64, False, "n/a", f"{_S}.code:env_rt_key",
     "python, platform, machine and package versions (no host or user names)"),
)


def scheme(kind, name):
    return SCHEMES[(kind, name)]


def derive(kind, name, *args, **kwargs):
    """Compute a key by its registered derivation. Returns whatever that function returns."""
    module_name, _, function = SCHEMES[(kind, name)].derive.partition(":")
    return getattr(importlib.import_module(module_name), function)(*args, **kwargs)
