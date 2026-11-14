"""Recognising existing artefacts: scan a folder, describe what each file is.

Nothing is moved, renamed or rewritten. A description is the content of a manifest with
provenance ``legacy-import``: only facts read from the file itself are stated, and what the file
does not say (the run that produced a checkpoint, the corpus behind a tokenizer) stays unstated.

  checkpoint   ``.pth/.pt/.ckpt`` -> ``ckpt:members-v1`` (needs torch to read the tensors), and the
               servable ``model:triple-v1`` when architecture and tokenizer are known
  tokenizer    a byte-level BPE file -> ``tok:bytebpe-v1``; ``vocab.json`` + ``merges.txt`` -> ``tok:bpe-sem-v1``
  cache        ``cached_data_<md5>_<datahash>.pkl`` -> ``cache:md5-v2`` (never unpickled)
  corpus       a directory, only when asked for (``kind="corpus"``) -> ``corpus:files-v1``
  file         anything, only when asked for -> ``file:bytes-v1``
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from cevahir_ai.keys.schemes.cache import CACHE_NAME_RE
from cevahir_ai.keys.schemes.file import file_bytes_key, file_utf8lf_key

KINDS = ("auto", "ckpt", "tok", "cache", "corpus", "evalset", "file")
CHECKPOINT_SUFFIXES = (".pth", ".pt", ".ckpt")
CACHE_META_FIELDS = ("version", "encode_mode", "include_whole_words", "include_syllables", "include_sep",
                     "max_seq_length", "alignment_format", "cache_format_version", "sample_count")
_RESERVED_NAMES = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)


@dataclass
class Candidate:
    kind: str                # ckpt | tok-byte | tok-bpe | cache | corpus | evalset | file
    paths: tuple             # the files (or the directory) it is read from, primary first


@dataclass
class Description:
    """What becomes a manifest, plus the local files that hold its payload."""
    kind: str
    key: str
    members: dict = field(default_factory=dict)
    model: str = None
    aliases: list = field(default_factory=list)
    inputs: list = field(default_factory=list)
    payload: list = field(default_factory=list)          # [{"role", "file", "size", "name_hint", ...}]
    attributes: dict = field(default_factory=dict)
    explain: dict = field(default_factory=dict)
    files: list = field(default_factory=list)            # [(file key, path)] on this host
    detail: dict = None                                  # a bulky companion document (per-tensor table, file list)
    notes: list = field(default_factory=list)


def safe_name(name):
    """A file name that is valid on Windows and POSIX, taken from the last component of ``name``.

    Used for name hints in manifests and for members of bundles: a hint recorded on one platform
    must never become a path on another (``..\\..\\x``, ``C:\\x``, ``NUL``, a trailing dot).
    """
    last = re.split(r"[\\/]", str(name))[-1]
    cleaned = re.sub(r'[<>:"|?*\x00-\x1f]', "_", last).strip(" .")
    if not cleaned or _RESERVED_NAMES.match(cleaned):
        cleaned = "_" + cleaned
    return cleaned[:120]


def _payload(role, path, file_key=None, **extra):
    """A payload entry; the file is hashed unless its key is already known."""
    path = Path(path)
    key = file_key or file_bytes_key(path)
    return dict({"role": role, "file": str(key), "size": path.stat().st_size, "name_hint": safe_name(path.name)}, **extra)


def _is_byte_bpe(path):
    try:
        with open(path, "rb") as stream:
            return b'"cevahir-byte-bpe"' in stream.read(512)
    except OSError:
        return False


def classify(path):
    """Kind of one file by its name and first bytes, or None when it is not a recognised artefact."""
    path = Path(path)
    if CACHE_NAME_RE.match(path.name):
        return "cache"
    if path.suffix.lower() in CHECKPOINT_SUFFIXES:
        return "ckpt"
    if path.suffix.lower() == ".json" and _is_byte_bpe(path):
        return "tok-byte"
    return None


def _tokenizer_pairs(files):
    """``vocab.json`` + ``merges.txt`` pairs: in one directory, or the only two under the scanned root."""
    vocabs = [path for path in files if path.name == "vocab.json"]
    merges = [path for path in files if path.name == "merges.txt"]
    pairs, used = [], set()
    for vocab in vocabs:
        partner = next((path for path in merges if path.parent == vocab.parent), None)
        if partner is not None:
            pairs.append((vocab, partner))
            used.update((vocab, partner))
    rest_v, rest_m = [p for p in vocabs if p not in used], [p for p in merges if p not in used]
    if len(rest_v) == 1 and len(rest_m) == 1:
        pairs.append((rest_v[0], rest_m[0]))
    return pairs


def scan(path, recurse=False, kind="auto"):
    """Candidates under ``path``. Returns ``(candidates, skipped)``; ``skipped`` lists unrecognised files."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind!r}; expected one of {KINDS}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    if kind in ("corpus", "evalset"):
        if not path.is_dir():
            raise ValueError(f"a {kind} is a directory: {path}")
        return [Candidate(kind, (path,))], []
    if path.is_file():
        files = [path]
    else:
        files = sorted(item for item in (path.rglob("*") if recurse else path.iterdir()) if item.is_file())
    if kind == "file":
        return [Candidate("file", (item,)) for item in files], []
    candidates, skipped, paired = [], [], set()
    if kind in ("auto", "tok"):
        for vocab, merges in _tokenizer_pairs(files):
            candidates.append(Candidate("tok-bpe", (vocab, merges)))
            paired.update((vocab, merges))
    for item in files:
        if item in paired:
            continue
        found = classify(item)
        wanted = {"auto": found, "ckpt": "ckpt", "cache": "cache", "tok": "tok-byte"}[kind]
        if found is not None and found == wanted:
            candidates.append(Candidate(found, (item,)))
        elif path.is_file() and kind != "auto":
            raise ValueError(f"{item} is not a {kind} artefact")
        else:
            skipped.append(item)
    return candidates, skipped


# ------------------------------------------------------------------------------------ descriptions
def describe_file(path, note=None, **attributes):
    entry = _payload("file", path)
    description = Description("file", entry["file"], payload=[entry], attributes=dict(attributes),
                              files=[(entry["file"], Path(path))])
    if note:
        description.notes.append(note)
    return description


def describe_checkpoint(path):
    """A checkpoint and, when it has one, its servable model. Reads the tensors (torch)."""
    path = Path(path)
    try:
        from cevahir_ai.adapters.inspection import inspect_checkpoint
        report = inspect_checkpoint(path, level="weights", purpose="inspect")
    except ImportError as error:
        return [describe_file(path, note=f"registered at file level only: torch is not available ({error})",
                              unread="torch is not available")]
    keys = report.keys
    if "ckpt" not in keys:
        safety = report.facts.get("C12")
        reason = safety.detail if safety is not None and safety.status != "pass" else \
            "; ".join(report.notes) or "the file holds no readable weights"
        return [describe_file(path, note=f"registered at file level only: {reason}", quarantined=reason,
                              envelope=report.envelope)]
    entry = _payload("container", path, keys["file"], envelope=report.envelope)
    tok = keys.get("tok_canonical") or keys.get("tok")
    members = {"weights": keys["weights"], "wschema": keys["wschema"], "arch": keys.get("arch"), "tok": tok,
               "optim": keys.get("optim"), "sched": keys.get("sched"), "tstate": keys.get("tstate")}
    member_aliases = {}
    for alias in report.aliases:
        member_aliases.setdefault(alias.kind, []).append(str(alias))
    if keys.get("tok") and tok != keys["tok"]:
        member_aliases.setdefault("tok", []).append(str(keys["tok"]))
    weights = report.explain.get("weights", {})
    attributes = {"envelope": report.envelope, "epoch": report.explain.get("epoch"),
                  "metric": report.explain.get("metric"), "dtypes": weights.get("dtypes", []),
                  "parameters": weights.get("parameters"), "tensors": weights.get("tensors"),
                  "member_aliases": {name: sorted(values) for name, values in member_aliases.items()},
                  "signatures": {name: str(keys[name]) for name in ("optim_sig", "sched_sig") if name in keys},
                  "trained_max_seq_length": report.explain.get("trained_max_seq_length")}
    explain = {name: report.explain[name] for name in ("arch", "family_version", "arch_undeclared", "tokenizer")
               if report.explain.get(name) is not None}
    checkpoint = Description("ckpt", str(keys["ckpt"]), members=members, model=keys.get("model"),
                             inputs=[{"role": "tokenizer", "key": str(tok)}] if tok else [], payload=[entry],
                             attributes=attributes, explain=explain, files=[(entry["file"], path)],
                             notes=list(report.notes),
                             detail={"format": "cevahir-tensor-table", "version": 1, "weights": str(keys["weights"]),
                                     "tensors": report.table})
    found = [checkpoint]
    if keys.get("model"):
        found.append(Description(
            "model", str(keys["model"]), members={"weights": keys["weights"], "arch": keys["arch"], "tok": tok},
            inputs=[{"role": "checkpoint", "key": str(keys["ckpt"])}],
            attributes={"parameters": weights.get("parameters"), "dtypes": weights.get("dtypes", []),
                        "trained_max_seq_length": report.explain.get("trained_max_seq_length")},
            explain={name: explain[name] for name in ("arch", "family_version") if name in explain}))
    return found


def describe_byte_bpe(path):
    import json
    from cevahir_ai.keys.schemes.tok import tok_bytebpe_key
    key = tok_bytebpe_key(path)                         # raises when the stored identity disagrees
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = _payload("tokenizer", path)
    return [Description("tok", str(key), payload=[entry], files=[(entry["file"], Path(path))],
                        attributes={"family": "cevahir-byte-bpe", "vocab_size": data.get("vocab_size"),
                                    "merges": len(data.get("merges", ()))})]


def describe_bpe_pair(vocab, merges):
    """The legacy BPE: ``tok:bpe-sem-v1`` from the two files, with the legacy digests the tracked
    registry records for the same tokenizer as aliases."""
    from cevahir_ai.keys.schemes.tok import tok_bpe_sem_key
    key, explain = tok_bpe_sem_key(vocab, merges)
    entries = [_payload("vocab", vocab, text=str(file_utf8lf_key(vocab))),
               _payload("merges", merges, text=str(file_utf8lf_key(merges)))]
    aliases, tracked = [], None
    try:
        from cevahir_ai.adapters.tokenizers import tracked_registry
        for item in tracked_registry()["tokenizers"]:
            if item["key"] == str(key):
                tracked, aliases = item["name"], [alias["key"] for alias in item.get("aliases", ())]
    except (OSError, RuntimeError, ValueError):
        pass
    return [Description("tok", str(key), aliases=aliases, payload=entries,
                        files=[(entries[0]["file"], Path(vocab)), (entries[1]["file"], Path(merges))],
                        attributes={"family": explain["family"], "vocab_size": explain["vocab_size"],
                                    "merges": explain["merges"], "tracked": tracked},
                        explain={"specials": explain["specials"], "id_config": explain["id_config"]})]


def describe_cache(path):
    from cevahir_ai.adapters.caches import recognise_cache
    path = Path(path)
    facts = recognise_cache(path)
    entries = [_payload("records", path, facts["file"])]
    files = [(facts["file"], path)]
    for role, sibling in (("meta", path.with_suffix(".meta.json")), ("sidecar", path.with_suffix(".sha256"))):
        if sibling.is_file():
            entry = _payload(role, sibling)
            entries.append(entry)
            files.append((entry["file"], sibling))
    meta = facts["meta"] or {}
    attributes = {"sidecar": facts["sidecar"], "meta": {name: meta[name] for name in CACHE_META_FIELDS if name in meta}}
    description = Description("cache", facts["cache"], payload=entries, files=files, attributes=attributes,
                              inputs=[{"role": "corpus", "key": facts["corpus"]}] if facts["corpus"] else [])
    if facts["sidecar"] == "stale":
        description.notes.append("the .sha256 sidecar does not match the file (stale sidecar); the file key is the truth")
    return [description]


def describe_corpus(directory, kind="corpus"):
    from cevahir_ai.keys.schemes.corpus import corpus_files_key, evalset_files_key
    key, payload = (corpus_files_key if kind == "corpus" else evalset_files_key)(directory)
    return [Description(kind, str(key), attributes={"files": len(payload["files"]), "suffixes": payload["selector"]["suffixes"]},
                        detail={"format": "cevahir-file-listing", "version": 1, kind: str(key), "files": payload["files"]})]


def describe(candidate):
    """Descriptions for one candidate (a checkpoint may give two: the checkpoint and its model)."""
    kind, paths = candidate.kind, candidate.paths
    if kind == "ckpt":
        return describe_checkpoint(paths[0])
    if kind == "tok-byte":
        return describe_byte_bpe(paths[0])
    if kind == "tok-bpe":
        return describe_bpe_pair(*paths)
    if kind == "cache":
        return describe_cache(paths[0])
    if kind in ("corpus", "evalset"):
        return describe_corpus(paths[0], kind)
    return [describe_file(paths[0])]
