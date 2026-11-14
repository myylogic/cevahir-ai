"""The registry under CEVAHIR_HOME: manifests, locations, registration, lineage and bundles.

Torch-free parts first (content-addressed objects, references, locations, scanning, lineage as a
projection of the audit events, bundles and what they reject). The last section registers real
checkpoints written by the real writers into a temporary folder and tests them against their
manifests. Every test uses a temporary home; nothing is written into the checkout.
"""
import hashlib
import json
import ntpath
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from cevahir_ai.errors import AmbiguousReference, BundleError, NotFound, StoreError
from cevahir_ai.keys.canonical import cjson
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.schemes.ckpt import ckpt_members_key, model_triple_key
from cevahir_ai.keys.schemes.tok import bytebpe_identity
from cevahir_ai.ops import artifacts, checkpoints
from cevahir_ai.ops import audit as audit_ops
from cevahir_ai.ops import keys as key_ops
from cevahir_ai.store import bundles, lineage, locations
from cevahir_ai.store import register as registration
from cevahir_ai.store.audit import AuditLog
from cevahir_ai.store.core import Store, manifest_key, new_manifest, statement

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "research" / "living_learning_paradigm_2026_11_13"
CORPUS = "cev:corpus:files-v1:" + "c0" * 32
TOK = "cev:tok:bpe-sem-v1:" + "70" * 32
CACHE = "cev:cache:recipe-v1:" + "ca" * 32
RUN = "cev:run:id-v1:" + "0193" * 8
WEIGHTS, WSCHEMA = "cev:weights:sha256-v1:" + "11" * 32, "cev:wschema:layout-v1:" + "55" * 32
ARCH = "cev:arch:cfg-v1:" + "a5" * 32
CKPT_MEMBERS = {"weights": WEIGHTS, "wschema": WSCHEMA, "arch": ARCH, "tok": TOK}
MODEL_MEMBERS = {"weights": WEIGHTS, "arch": ARCH, "tok": TOK}
CKPT = str(ckpt_members_key(CKPT_MEMBERS))          # a checkpoint's key is the key of its members
MODEL = str(model_triple_key(WEIGHTS, ARCH, TOK))
CACHE_NAME = "cached_data_" + "ab" * 16 + "_" + "cd" * 32 + ".pkl"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CEVAHIR_LOCK_DIR", str(tmp_path / "locks"))


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "home", environ={})


def other_home(tmp_path, name="colab"):
    """A second machine: its own home, hence its own host id, audit chain and key."""
    return Store(tmp_path / name, environ={})


def byte_bpe_file(path, merges=(("a", "b"), ("ab", "c"))):
    data = {"format": "cevahir-byte-bpe", "version": 1, "pattern": r"\w+|\s+", "special_tokens": ["<PAD>", "<UNK>"],
            "vocab_size": 258 + len(merges), "merges": [list(pair) for pair in merges]}
    data["identity"] = bytebpe_identity(data["pattern"], data["special_tokens"], data["merges"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return Key("tok", "bytebpe-v1", data["identity"])


def events(store):
    return [event["event"] for event in AuditLog(store).events()]


# ------------------------------------------------------------------------------- home and objects
def test_home_layout_follows_the_platform_and_the_environment(tmp_path):
    from cevahir_ai._compat import home
    assert home.cevahir_home({"CEVAHIR_HOME": str(tmp_path / "h")}) == (tmp_path / "h").resolve()
    windows = home.default_home({"LOCALAPPDATA": r"C:\Users\Yasin\AppData\Local"}, "win32")
    assert ntpath.basename(str(windows).replace("/", "\\")) == "Cevahir"
    assert ntpath.dirname(str(windows).replace("/", "\\")) == r"C:\Users\Yasin\AppData\Local"
    assert home.default_home({"XDG_STATE_HOME": "/state"}, "linux") == Path("/state/cevahir")
    assert home.host_file(tmp_path, {}) == tmp_path / "host.json"
    assert home.host_file(tmp_path, {"COLAB_RELEASE_TAG": "r"}) == Path("/content/.cevahir-host.json"), "per VM, not on Drive"
    assert home.host_file(tmp_path, {"CEVAHIR_HOST_FILE": str(tmp_path / "x.json")}) == tmp_path / "x.json"
    with pytest.raises(RuntimeError, match="outside the checkout"):
        Store(None, environ={"CEVAHIR_HOME": str(ROOT / "state")})


def test_objects_are_content_addressed_and_a_modified_object_is_never_used(store):
    document = new_manifest("ckpt", CKPT, inputs=[{"role": "tokenizer", "key": TOK}], created="2026-11-13T00:00:00.000000Z")
    digest = store.put(document)
    assert digest == hashlib.sha256(cjson(document)).hexdigest() == manifest_key(document).digest
    assert store.put(document) == digest and store.object_ids() == [digest], "writing the same statement twice is one object"
    assert store.get(digest) == document
    path = store.object_path(digest)
    assert path == store.home / "objects" / digest[:2] / (digest[2:] + ".json")
    path.write_bytes(path.read_bytes().replace(b"tokenizer", b"tokenizeR"))
    with pytest.raises(StoreError, match="does not hash to its name"):
        store.get(digest)
    store.invalidate()
    assert store.catalog().artifacts == {} and store.catalog().corrupt == [digest]
    with pytest.raises(NotFound):
        store.get("0" * 64)
    with pytest.raises(ValueError):
        store.object_path("../../etc/passwd")


def test_manifest_is_canonical_data_without_paths_or_unknown_keys():
    document = new_manifest("ckpt", CKPT, members={"weights": "cev:weights:sha256-v1:" + "11" * 32, "arch": None},
                            aliases=[CKPT, "cev:tok:legacy-v3:" + "22" * 32], attributes={"metric": float("nan"), "epoch": 3},
                            payload=[{"role": "container", "file": "cev:file:bytes-v1:" + "33" * 32, "name_hint": "last.pth"}])
    assert document["format"] == "cevahir-manifest" and document["version"] == 1 and document["provenance"] == "legacy-import"
    assert document["members"] == {"weights": "cev:weights:sha256-v1:" + "11" * 32}, "an unknown member is absent, not null"
    assert document["aliases"] == ["cev:tok:legacy-v3:" + "22" * 32], "its own key is not its alias"
    assert document["attributes"] == {"metric": "nan", "epoch": 3}, "a NaN metric cannot break the registry"
    assert "created" not in statement(document) and "producer" not in statement(document)
    assert not any("path" in name.lower() for name in document), "where a file is never enters a manifest"
    with pytest.raises(ValueError):
        new_manifest("ckpt", "cev:ckpt:nope-v1:" + "00" * 32)
    with pytest.raises(ValueError):
        new_manifest("ckpt", CKPT, provenance="hearsay")


def test_host_identity_is_created_once_per_home(tmp_path, store):
    assert store.host_id() is None and not store.home.exists(), "reading never creates the home"
    host = store.host_id(create=True)
    assert host == Store(store.home, environ={}).host_id() and len(host) == 18
    assert other_home(tmp_path).host_id(create=True) != host
    (store.home / "host.json").write_text('{"host_id": "../evil"}')
    with pytest.raises(StoreError):
        Store(store.home, environ={}).host_id()


# -------------------------------------------------------------------------------------- locations
def test_byte_identical_copies_are_two_locations_of_one_file(tmp_path, store):
    first, second = tmp_path / "last.pth", tmp_path / "best.pth"
    first.write_bytes(b"same bytes")
    shutil.copyfile(first, second)
    key = locations.file_key_of(store, first)
    assert key == f"cev:file:bytes-v1:{hashlib.sha256(b'same bytes').hexdigest()}"
    assert locations.add_location(store, key, first) and locations.add_location(store, key, second)
    assert not locations.add_location(store, key, first), "recording a known location again is not new"
    assert sorted(entry["path"] for entry in locations.locations_of(store, key)) == sorted([str(second), str(first)])
    assert locations.known_file_key(store, first) == key
    first.write_bytes(b"other bytes!")
    assert locations.known_file_key(store, first) is None, "a changed file is hashed again, never trusted by its path"
    assert [entry["changed"] for entry in locations.live_locations(store, key)].count(True) == 1
    assert locations.find_file(store, key) == second, "the unchanged copy is preferred"
    second.unlink()
    assert [entry["path"] for entry in locations.live_locations(store, key)] == [str(first)]
    document = json.loads((store.home / "locations" / f"{store.host_id()}.json").read_text(encoding="utf-8"))
    assert document["host"] == store.host_id() and list(document["files"]) == [key]


def test_safe_name_never_becomes_a_path_on_any_platform():
    assert registration.safe_name(r"C:\Users\Yasin\saved_models\last.pth") == "last.pth"
    assert registration.safe_name(ntpath.join("C:\\", "Program Files", "cevahir", "model.pt")) == "model.pt"
    assert registration.safe_name("../../etc/passwd") == "passwd" and registration.safe_name(r"..\..\evil.pth") == "evil.pth"
    assert registration.safe_name("NUL") == "_NUL" and registration.safe_name("con.txt") == "_con.txt"
    assert registration.safe_name('a<b>:"c|d?*.pth') == "a_b___c_d__.pth"
    assert registration.safe_name("ends with dot. ") == "ends with dot" and registration.safe_name("..") == "_"
    assert registration.safe_name("türkçe ağırlık.pth") == "türkçe ağırlık.pth"


# ----------------------------------------------------------------------------------- registration
def test_scan_recognises_caches_tokenizers_and_checkpoints_by_name_and_first_bytes(tmp_path):
    root = tmp_path / "data"
    (root / "vocab_lib").mkdir(parents=True)
    (root / "merges_lib").mkdir()
    (root / "vocab_lib" / "vocab.json").write_text('{"<PAD>": 0, "a": 1}', encoding="utf-8")
    (root / "merges_lib" / "merges.txt").write_text("a b\n", encoding="utf-8")
    (root / CACHE_NAME).write_bytes(b"x")
    (root / "model.tokenizer.json").write_text('{"format": "cevahir-byte-bpe"}', encoding="utf-8")
    (root / "notes.json").write_text("{}", encoding="utf-8")
    (root / "weights.pth").write_bytes(b"x")
    found, skipped = registration.scan(root, recurse=True)
    kinds = sorted((item.kind, tuple(path.name for path in item.paths)) for item in found)
    assert kinds == [("cache", (CACHE_NAME,)), ("ckpt", ("weights.pth",)), ("tok-bpe", ("vocab.json", "merges.txt")),
                     ("tok-byte", ("model.tokenizer.json",))]
    assert [path.name for path in skipped] == ["notes.json"]
    assert [item.kind for item in registration.scan(root)[0]] == ["cache", "tok-byte", "ckpt"], "without recurse: one level"
    assert [item.kind for item in registration.scan(root, kind="corpus")[0]] == ["corpus"]
    assert len(registration.scan(root, recurse=True, kind="file")[0]) == 6
    with pytest.raises(FileNotFoundError):
        registration.scan(root / "missing")
    with pytest.raises(ValueError):
        registration.scan(root / "notes.json", kind="ckpt")


def test_register_cache_tokenizers_and_corpus_without_torch(tmp_path, store):
    root = tmp_path / "work"
    (root / "cache").mkdir(parents=True)
    cache = root / "cache" / CACHE_NAME
    cache.write_bytes(b"records that are never unpickled")
    cache.with_suffix(".meta.json").write_text(json.dumps({"encode_mode": "train", "max_seq_length": 64, "sample_count": 12,
                                                           "data_dir": r"C:\Users\Yasin\education"}), encoding="utf-8")
    cache.with_suffix(".sha256").write_text("0" * 64, encoding="utf-8")          # a stale sidecar
    byte_key = byte_bpe_file(root / "tok" / "model.tokenizer.json")
    (root / "tok" / "vocab.json").write_text('{"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3, "ab": 4}', encoding="utf-8")
    (root / "tok" / "merges.txt").write_bytes(b"a b\r\n")
    warnings = []
    records = artifacts.register(store, root, recurse=True, warn=warnings.append)
    by_kind = {record.Kind: record for record in records}
    assert sorted(record.Kind for record in records) == ["cache", "tok", "tok"] and all(record.New for record in records)
    assert by_kind["cache"].Key == "cev:cache:md5-v2:" + "ab" * 16 and by_kind["cache"].Provenance == "legacy-import"
    assert [(item.Role, item.Key) for item in by_kind["cache"].Inputs] == [("corpus", "cev:corpus:legacy-dd-v1:" + "cd" * 32)]
    assert by_kind["cache"].Attributes == {"sidecar": "stale", "meta": {"encode_mode": "train", "max_seq_length": 64,
                                                                        "sample_count": 12}}, "the meta's path is not kept"
    assert any("stale sidecar" in text for text in warnings)
    assert len(by_kind["cache"].Files) == 3 and len(by_kind["cache"].Locations) == 3, "records, meta and sidecar"
    keys = {record.Key for record in records}
    assert str(byte_key) in keys and any(key.startswith("cev:tok:bpe-sem-v1:") for key in keys)
    cached = checkpoints.get_caches(store)[0]
    assert cached.Sidecar == "stale" and cached.Samples == 12 and cached.MaxSeqLength == 64
    assert sorted(record.Family for record in checkpoints.get_tokenizers(store)) == ["cevahir-bpe", "cevahir-byte-bpe"]
    # A corpus is registered only when asked for; CRLF and LF copies of it are one corpus.
    for name, newline in (("lf", b"\n"), ("crlf", b"\r\n")):
        (root / name).mkdir()
        (root / name / "a.txt").write_bytes(b"bir" + newline + b"iki" + newline)
    lf = artifacts.register(store, root / "lf", kind="corpus")[0]
    crlf = artifacts.register(store, root / "crlf", kind="corpus")[0]
    assert lf.Kind == "corpus" and lf.Key == crlf.Key and lf.New and not crlf.New and lf.Attributes["files"] == 1


def test_registry_and_listing_need_no_torch(tmp_path):
    """Registering caches and tokenizers, listing, lineage and the audit check run in a torch-free process."""
    code = r"""
import json, sys
from pathlib import Path
from cevahir_ai.keys.schemes.tok import bytebpe_identity
from cevahir_ai.ops import artifacts, audit, checkpoints
from cevahir_ai.store.core import Store
root = Path(sys.argv[1])
data = {"format": "cevahir-byte-bpe", "version": 1, "pattern": "x", "special_tokens": [], "vocab_size": 257, "merges": [["a", "b"]]}
data["identity"] = bytebpe_identity(data["pattern"], data["special_tokens"], data["merges"])
(root / "work").mkdir()
(root / "work" / "t.tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
(root / "work" / ("cached_data_" + "ab" * 16 + "_" + "cd" * 32 + ".pkl")).write_bytes(b"x")
store = Store(root / "home", environ={})
records = artifacts.register(store, root / "work")
listed = artifacts.get(store) + checkpoints.get_tokenizers(store) + checkpoints.get_caches(store) + checkpoints.get_checkpoints(store)
edges = artifacts.lineage_of(store, records[0].Key, "Both")
verdict = audit.test_log(store)[0]
loaded = sorted(name for name in ("torch", "numpy", "model_management", "src", "training_system") if name in sys.modules)
print(json.dumps({"kinds": sorted(r.Kind for r in records), "listed": len(listed), "ok": verdict.Ok, "loaded": loaded}))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CEVAHIR_LOCK_DIR": str(tmp_path / "locks")}
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"kinds": ["cache", "tok"], "listed": 4, "ok": True, "loaded": []}


def test_registering_again_changes_nothing_and_records_nothing(tmp_path, store):
    root = tmp_path / "tok"
    byte_bpe_file(root / "a.tokenizer.json")
    first = artifacts.register(store, root)
    objects, recorded = store.object_ids(), events(store)
    again = artifacts.register(store, root)
    assert [record.Key for record in again] == [record.Key for record in first] and not again[0].New
    assert store.object_ids() == objects and events(store) == recorded == ["log.genesis", "artifact.registered"]
    shutil.copyfile(root / "a.tokenizer.json", root / "copy.tokenizer.json")
    copied = artifacts.register(store, root)
    assert len(copied) == 1 and not copied[0].New and len(copied[0].Locations) == 2, "a copy is a location, not an artefact"
    assert store.object_ids() == objects and events(store)[-1] == "artifact.located"


def test_a_file_that_is_not_what_it_claims_is_reported_not_registered(tmp_path, store):
    broken = tmp_path / "x.tokenizer.json"
    byte_bpe_file(broken)
    data = json.loads(broken.read_text(encoding="utf-8"))
    data["merges"].append(["x", "y"])                      # the stored identity no longer matches the content
    broken.write_text(json.dumps(data), encoding="utf-8")
    warnings = []
    assert artifacts.register(store, broken, warn=warnings.append) == []
    assert "identity mismatch" in warnings[0] and store.object_ids() == []
    plain = tmp_path / "readme.txt"
    plain.write_text("x", encoding="utf-8")
    assert artifacts.register(store, plain, warn=warnings.append) == [] and "not a recognised artefact" in warnings[-1]
    as_file = artifacts.register(store, plain, kind="file")[0]
    assert as_file.Kind == "file" and as_file.Key == f"cev:file:bytes-v1:{hashlib.sha256(b'x').hexdigest()}"


# ------------------------------------------------------------------------------------- references
def test_references_resolve_by_key_alias_file_manifest_short_form_and_path(tmp_path, store):
    path = tmp_path / "a.tokenizer.json"
    key = byte_bpe_file(path)
    record = artifacts.register(store, path)[0]
    catalog = store.catalog()
    artifact = catalog.resolve(str(key))
    assert artifact.key == str(key) and artifact.kind == "tok" and artifact.short() == key.display() == f"tok:{key.digest[:12]}"
    for reference in (key.display(), f"tok:{key.digest[:6]}", f"tok:bytebpe-v1:{key.digest[:8]}", record.Files[0], record.Manifests[0]):
        assert catalog.resolve(reference).key == str(key), reference
    assert artifacts.resolve(store, str(path)).key == str(key), "the path of a registered file names its artefact"
    with pytest.raises(NotFound, match="no registered artefact"):
        catalog.resolve("tok:ffffff")
    with pytest.raises(NotFound):
        catalog.resolve("not a reference")
    loose = tmp_path / "unregistered.bin"
    loose.write_bytes(b"nobody registered this")
    with pytest.raises(NotFound, match="register it first"):
        artifacts.resolve(store, str(loose))


def test_ambiguous_short_forms_are_refused_not_guessed(store):
    first = new_manifest("ckpt", "cev:ckpt:members-v1:" + "abcd" + "0" * 60)
    second = new_manifest("ckpt", "cev:ckpt:members-v1:" + "abcd" + "1" * 60)
    store.put(first)
    store.put(second)
    with pytest.raises(AmbiguousReference, match="give more digits"):
        store.catalog().resolve("ckpt:abcd")
    assert store.catalog().resolve("ckpt:abcd0").key == first["key"]
    assert [artifact.key for artifact in store.catalog().find("ckpt:abcd")] == [first["key"], second["key"]]


def test_a_member_key_is_not_an_artefact_and_the_error_says_where_it_lives(store):
    weights = "cev:weights:sha256-v1:" + "11" * 32
    store.put(new_manifest("ckpt", CKPT, members={"weights": weights}))
    with pytest.raises(NotFound, match=f"it is a member of {artifacts.short(CKPT)}"):
        artifacts.resolve(store, weights)


# ---------------------------------------------------------------------------------------- lineage
def chain_of_derivations(store, tmp_path, name="last.pth"):
    """What governed training will record: corpus -> cache -> run -> checkpoint -> model, with a tokenizer."""
    payload = tmp_path / name
    payload.write_bytes(b"checkpoint bytes of " + name.encode())
    artifacts.derive(store, "corpus", CORPUS)
    artifacts.derive(store, "tok", TOK, inputs=[{"role": "corpus", "key": CORPUS}])
    artifacts.derive(store, "cache", CACHE, inputs=[{"role": "corpus", "key": CORPUS}, {"role": "tokenizer", "key": TOK}])
    artifacts.derive(store, "run", RUN, inputs=[{"role": "cache", "key": CACHE}, {"role": "tokenizer", "key": TOK}],
                     activity={"kind": "training", "id": RUN})
    checkpoint = artifacts.derive(store, "ckpt", CKPT, inputs=[{"role": "run", "key": RUN}], files=[payload], members=CKPT_MEMBERS,
                                  activity={"kind": "training", "id": RUN}, attributes={"epoch": 5})
    artifacts.derive(store, "model", MODEL, inputs=[{"role": "checkpoint", "key": CKPT}], members=MODEL_MEMBERS)
    return checkpoint, payload


def test_lineage_is_a_projection_of_the_audit_events(tmp_path, store):
    checkpoint, _ = chain_of_derivations(store, tmp_path)
    assert checkpoint.Provenance == "native" and checkpoint.State == "registered"
    up = [(row.Level, row.Source, row.Target, row.Role) for row in artifacts.lineage_of(store, MODEL, "Up")]
    assert up == [(1, CKPT, MODEL, "checkpoint"), (2, RUN, CKPT, "run"), (3, CACHE, RUN, "cache"), (3, TOK, RUN, "tokenizer"),
                  (4, CORPUS, TOK, "corpus"), (4, CORPUS, CACHE, "corpus"), (4, TOK, CACHE, "tokenizer")]
    sources = {row.Source for row in artifacts.lineage_of(store, MODEL, "Up")}
    assert sources == {CKPT, RUN, CACHE, TOK, CORPUS}, "from the model back to the corpus"
    down = {row.Target for row in artifacts.lineage_of(store, CORPUS, "Down")}
    assert down == {TOK, CACHE, RUN, CKPT, MODEL}
    assert {row.Source for row in artifacts.lineage_of(store, MODEL, "Up", depth=1)} == {CKPT}
    both = artifacts.lineage_of(store, CKPT, "Both")
    assert {row.Direction for row in both} == {"Up", "Down"} and MODEL in {row.Target for row in both}
    tree = [row.Tree for row in artifacts.lineage_of(store, MODEL, "Up", depth=2)]
    assert tree == [f"<- {artifacts.short(CKPT)}  [checkpoint]", f"  <- {artifacts.short(RUN)}  [run]"]
    # The projection and the manifests agree, and every edge names the event that recorded it.
    assert lineage.diverging(store) == []
    edges = lineage.edges(store)
    assert all(edge.how == "artifact.derived" and edge.host == store.host_id() and edge.seq > 1 for edge in edges)
    recorded = [event for event in AuditLog(store).events() if event["event"] == "artifact.derived"]
    assert recorded[4]["lineage"] == {"output": CKPT, "inputs": [{"role": "run", "key": RUN}],
                                      "activity": {"kind": "training", "id": RUN}}
    assert artifacts.lineage_of(store, CORPUS, "Up") == [] and lineage.walk(store, "cev:tok:legacy-v3:" + "00" * 32) == []
    with pytest.raises(ValueError):
        lineage.walk(store, CKPT, "Sideways")


def test_lineage_survives_cycles_and_follows_aliases(store):
    legacy = "cev:tok:legacy-v3:" + "5e" * 32
    artifacts.derive(store, "tok", TOK)
    store.put(new_manifest("tok", TOK, aliases=[legacy], created="2099-01-01T00:00:00.000000Z"))
    store.invalidate()
    artifacts.derive(store, "ckpt", CKPT, inputs=[{"role": "tokenizer", "key": legacy}], members=CKPT_MEMBERS)   # by its legacy digest
    assert [row.Source for row in artifacts.lineage_of(store, CKPT, "Up")] == [TOK], "an alias is shown as the artefact"
    assert [row.Target for row in artifacts.lineage_of(store, legacy, "Down")] == [CKPT]
    log = AuditLog(store)
    log.append("artifact.derived", lineage={"output": TOK, "inputs": [{"role": "loop", "key": CKPT}]})
    assert len(artifacts.lineage_of(store, CKPT, "Up")) == 2, "a cycle ends, it does not hang"


# ------------------------------------------------------------------------------------- bundles
def test_checkpoint_from_another_machine_is_recognised_by_its_bytes_with_its_lineage(tmp_path, store, monkeypatch):
    """Colab trains, the workstation receives a manifest-only bundle and, separately, the checkpoint file."""
    colab = other_home(tmp_path)
    _, payload = chain_of_derivations(colab, tmp_path)
    bundle = artifacts.export(colab, [MODEL], tmp_path / "run.cvb", include_audit=True)
    assert bundle.Manifests == 6 and bundle.Payloads == 0 and bundle.Chains == [colab.host_id()], "ancestors travel too"
    imported = artifacts.import_bundle(store, tmp_path / "run.cvb")
    assert {record.Key for record in imported} == {CORPUS, TOK, CACHE, RUN, CKPT, MODEL}
    assert all(record.New and record.State == "imported" and record.Locations == [] for record in imported)
    monkeypatch.setattr(registration, "describe", lambda candidate: pytest.fail("a known file must not be read again"))
    copied = tmp_path / "downloads" / "epoch5.pth"
    copied.parent.mkdir()
    shutil.copyfile(payload, copied)                        # the manual copy from Drive
    located = artifacts.register(store, copied)
    assert [record.Key for record in located] == [CKPT] and not located[0].New and located[0].Locations == [str(copied)]
    assert located[0].State == "registered"
    assert {row.Source for row in artifacts.lineage_of(store, str(copied), "Up")} == {RUN, CACHE, TOK, CORPUS}
    # Both chains are in the workstation's home; the foreign one verifies without its key.
    verdicts = {record.Host: record for record in audit_ops.test_log(store)}
    assert verdicts[colab.host_id()].Ok and verdicts[colab.host_id()].Mac == "no-key" and not verdicts[colab.host_id()].Own
    assert verdicts[store.host_id()].Ok and verdicts[store.host_id()].Mac == "verified"
    assert events(store).count("artifact.imported") == 6 and events(store)[-1] == "artifact.located"
    assert lineage.diverging(store) == []
    again = artifacts.import_bundle(store, tmp_path / "run.cvb")
    assert not any(record.New for record in again) and events(store).count("artifact.imported") == 6, \
        "importing the same bundle again records nothing"


def test_bundle_round_trip_with_payload(tmp_path, store):
    colab = other_home(tmp_path)
    _, payload = chain_of_derivations(colab, tmp_path)
    bundle = artifacts.export(colab, [CKPT], tmp_path / "ckpt.cvb", include_payload=True)
    assert bundle.Payloads == 1 and bundle.Chains == [] and str(bundles.bundle_key(tmp_path / "ckpt.cvb")).endswith(bundle.Id)
    with zipfile.ZipFile(tmp_path / "ckpt.cvb") as archive:
        names = archive.namelist()
        stored = [info.compress_type for info in archive.infolist() if info.filename.startswith("payloads/")]
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert f"payloads/{digest}/last.pth" in names and stored == [zipfile.ZIP_STORED] and "bundle.json" in names
    imported = {record.Key: record for record in artifacts.import_bundle(store, tmp_path / "ckpt.cvb")}
    blob = Path(imported[CKPT].Locations[0])
    assert blob == store.home / "blobs" / digest[:2] / (digest[2:] + ".pth") and blob.read_bytes() == payload.read_bytes()
    assert imported[CKPT].Attributes["epoch"] == 5 and store.catalog().artifacts[CKPT].manifests == \
        colab.catalog().artifacts[CKPT].manifests, "the manifest arrives byte for byte"
    elsewhere = artifacts.import_bundle(other_home(tmp_path, "third"), tmp_path / "ckpt.cvb", destination=tmp_path / "unpacked")
    assert Path({r.Key: r for r in elsewhere}[CKPT].Locations[0]).parent.parent == tmp_path / "unpacked"
    payload.unlink()
    with pytest.raises(BundleError, match="is not on this host"):
        artifacts.export(colab, [CKPT], tmp_path / "again.cvb", include_payload=True)
    assert not (tmp_path / "again.cvb").exists() and not (tmp_path / "again.cvb.part").exists()
    assert events(colab)[-1] == "action.failed", "a failed export is recorded too"


def rewrite(source, target, change):
    """Copy a bundle, letting ``change(name, data) -> data | None`` edit or drop members."""
    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(target, "w") as out:
        for info in archive.infolist():
            data = change(info.filename, archive.read(info))
            if data is not None:
                out.writestr(info.filename, data)


def test_import_rejects_what_does_not_verify_and_writes_nothing(tmp_path, store):
    colab = other_home(tmp_path)
    chain_of_derivations(colab, tmp_path)
    good = tmp_path / "good.cvb"
    artifacts.export(colab, [CKPT], good, include_payload=True, include_audit=True)

    def refused(change, message):
        bad = tmp_path / "bad.cvb"
        rewrite(good, bad, change)
        with pytest.raises(BundleError, match=message):
            artifacts.import_bundle(store, bad)
        assert not store.home.exists() or not any(store.home.rglob("*.json")), "a refused bundle leaves the home untouched"

    refused(lambda name, data: data + b"x" if name.startswith("payloads/") else data, "payload last.pth is corrupted")
    refused(lambda name, data: None if name.startswith("objects/") and CKPT[-8:].encode() in data else data,
            "lists objects/.* but does not contain it")
    refused(lambda name, data: data.replace(b'"epoch":5', b'"epoch":9') if name.startswith("objects/") else data,
            "does not hash to its name")
    refused(lambda name, data: data.replace(b"cevahir-bundle", b"something-else") if name == "bundle.json" else data,
            "not a cevahir-bundle")

    def drop_line(name, data):                              # a gap in the foreign chain, with matching digests
        if name.startswith("audit/") and name.endswith(".jsonl"):
            lines = data.split(b"\n")
            drop_line.new = b"\n".join(lines[:2] + lines[3:])
            return drop_line.new
        return data

    gap = tmp_path / "gap.cvb"
    rewrite(good, gap, drop_line)

    def fix_digest(name, data):
        if name != "bundle.json":
            return data
        description = json.loads(data)
        for segment in description["chains"][0]["segments"]:
            segment["sha256"], segment["bytes"] = hashlib.sha256(drop_line.new).hexdigest(), len(drop_line.new)
        return json.dumps(description).encode("utf-8")

    consistent = tmp_path / "gap2.cvb"
    rewrite(gap, consistent, fix_digest)
    with pytest.raises(BundleError, match="does not match its digest"):
        artifacts.import_bundle(store, gap)
    with pytest.raises(BundleError, match="audit chain of h-[0-9a-f]{16} in the bundle does not verify: broken link"):
        artifacts.import_bundle(store, consistent)
    for junk in (b"not a zip", b""):
        (tmp_path / "junk.cvb").write_bytes(junk)
        with pytest.raises(BundleError, match="not a zip file"):
            artifacts.import_bundle(store, tmp_path / "junk.cvb")
    with pytest.raises(FileNotFoundError):
        artifacts.import_bundle(store, tmp_path / "missing.cvb")
    assert not store.home.exists() or store.object_ids() == []
    assert len(artifacts.import_bundle(store, good)) == 5, "the untouched bundle still imports (checkpoint and ancestors)"


def test_a_crafted_bundle_cannot_write_outside_the_home(tmp_path, store):
    colab = other_home(tmp_path)
    chain_of_derivations(colab, tmp_path)
    good, evil = tmp_path / "good.cvb", tmp_path / "evil.cvb"
    artifacts.export(colab, [CKPT], good, include_payload=True)

    def traversal(name, data):
        if name == "bundle.json":
            description = json.loads(data)
            description["payloads"][0]["name"] = "../../../../evil.pth"
            description["manifests"][0]["id"] = "../../../outside"
            return json.dumps(description).encode("utf-8")
        return data

    rewrite(good, evil, traversal)
    with pytest.raises(BundleError, match="not a sha256"):
        artifacts.import_bundle(store, evil)

    def only_name(name, data):
        if name == "bundle.json":
            description = json.loads(data)
            description["payloads"][0]["name"] = "../../../../evil.pth"
            return json.dumps(description).encode("utf-8")
        return data

    rewrite(good, evil, only_name)
    with pytest.raises(BundleError, match="lists payloads/[0-9a-f]{64}/evil.pth but does not contain it"):
        artifacts.import_bundle(store, evil)                # the name is reduced to its last component before any lookup
    assert not (tmp_path / "evil.pth").exists() and not list(tmp_path.rglob("evil.pth"))


def test_a_manifest_whose_key_is_not_the_key_of_its_members_is_refused(tmp_path, store):
    """A checkpoint key is derived from its members: a bundle cannot attach other weights to a known key."""
    colab = other_home(tmp_path)
    chain_of_derivations(colab, tmp_path)
    good, forged = tmp_path / "good.cvb", tmp_path / "forged.cvb"
    artifacts.export(colab, [CKPT], good)
    other_weights = "cev:weights:sha256-v1:" + "ee" * 32
    state = {}

    def swap(name, data):
        if name.startswith("objects/") and WEIGHTS.encode() in data and b'"kind":"ckpt"' in data:
            state["old"] = name[len("objects/"):-len(".json")]
            state["data"] = data.replace(WEIGHTS.encode(), other_weights.encode())
            state["new"] = hashlib.sha256(state["data"]).hexdigest()
            return None
        return data

    def relist(name, data):
        return data.replace(state["old"].encode(), state["new"].encode()) if name == "bundle.json" else data

    rewrite(good, tmp_path / "step.cvb", swap)
    rewrite(tmp_path / "step.cvb", forged, relist)
    with zipfile.ZipFile(forged, "a") as archive:
        archive.writestr(f"objects/{state['new']}.json", state["data"])      # content-addressed correctly, and still a lie
    with pytest.raises(BundleError, match="contradicts itself: its key is not the key of its members"):
        artifacts.import_bundle(store, forged)
    assert not store.home.exists() or store.object_ids() == []
    with pytest.raises(ValueError, match="its key is not the key of its members"):
        artifacts.derive(store, "ckpt", CKPT, members=dict(CKPT_MEMBERS, weights=other_weights))
    with pytest.raises(ValueError, match="do not form a key"):
        artifacts.derive(store, "model", MODEL, members={"weights": WEIGHTS})
    assert not store.home.exists() or (store.object_ids() == [] and events(store) == []), "nothing was stored or recorded"


def test_two_copies_of_one_chain_that_disagree_are_a_fork(tmp_path, store):
    colab = other_home(tmp_path)
    chain_of_derivations(colab, tmp_path)
    artifacts.export(colab, [CORPUS], tmp_path / "first.cvb", include_audit=True)
    artifacts.import_bundle(store, tmp_path / "first.cvb")
    artifacts.derive(colab, "evalset", "cev:evalset:files-v1:" + "e5" * 32)
    artifacts.export(colab, [CORPUS], tmp_path / "later.cvb", include_audit=True)
    held = AuditLog(store).verify(colab.host_id())["events"]
    artifacts.import_bundle(store, tmp_path / "later.cvb")
    assert AuditLog(store).verify(colab.host_id())["events"] > held, "a longer copy of the same chain extends the held one"
    artifacts.import_bundle(store, tmp_path / "first.cvb")
    assert AuditLog(store).verify(colab.host_id())["events"] > held, "an older copy never shortens it"
    # A second machine that claims the same host id but has another history.
    impostor = Store(tmp_path / "impostor", environ={})
    impostor.home.mkdir()
    (impostor.home / "host.json").write_text(json.dumps({"host_id": colab.host_id()}), encoding="utf-8")
    artifacts.derive(impostor, "corpus", CORPUS)
    artifacts.export(impostor, [CORPUS], tmp_path / "fork.cvb", include_audit=True)
    with pytest.raises(BundleError, match="contradicts the copy already held"):
        artifacts.import_bundle(store, tmp_path / "fork.cvb")


# ---------------------------------------------------------------------------- real checkpoints
@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """A folder as training leaves it: epoch checkpoints with best/last copies, a bare state dict, a
    pickled module and a CEV-11 style checkpoint with its tokenizer file, all written by the real writers."""
    torch = pytest.importorskip("torch")
    from model_management.config_schema import tiny_model_config
    from model_management.model_manager import ModelManager
    from model_management.model_saver import ModelSaver
    from tokenizer_management.byte_bpe import ByteLevelBPE
    from training_management.v2.utils.checkpoint_manager import CheckpointManager
    folder = tmp_path_factory.mktemp("saved_models")
    entry = json.loads((ROOT / "governance/registry/tokenizers.json").read_text(encoding="utf-8"))["tokenizers"][0]
    tracked = next(a["key"].rsplit(":", 1)[1] for a in entry["aliases"] if a["how"] == "read_only=absent, merges=LF")

    def manager(seed):
        torch.manual_seed(seed)
        result = ModelManager(tiny_model_config())
        result.initialize(build_optimizer=True, build_criterion=False, build_scheduler=True)
        return result

    saver = CheckpointManager(checkpoint_model_dir=str(folder / "checkpoints"))
    for epoch, metric in ((1, 2.9), (2, 2.4)):
        trained = manager(epoch)
        trained.model._tokenizer_identity = tracked
        extra = {"format_version": 1, "tokenizer_identity": tracked, "loop": {"global_step": epoch}, "best_val_loss": metric,
                 "scheduler": {"scheduler_type": "ReduceLROnPlateau", "scheduler_state": trained.scheduler.state_dict()},
                 "rng": {"torch": torch.get_rng_state()}}
        saver.save(trained.model, trained.optimizer, epoch, metric=metric, extra_state=extra)
    source = manager(9)
    ModelSaver.save_weights_only(source.model, save_dir=str(folder / "bare"))
    ModelSaver.save_full_model(source.model, save_dir=str(folder / "full"))
    for item in (str(ROOT), str(STUDY)):
        if item not in sys.path:
            sys.path.insert(0, item)
    from cev11 import model as study
    tokenizer = ByteLevelBPE.train(["merhaba dünya merhaba", "dünya güzel bir yer"] * 20, vocab_size=290)
    cev_model, cfg = study.build_model(vocab_size=tokenizer.vocab_size, layers=2, d=32, heads=4, kv_heads=4, max_len=64, seed=0)
    study.save_checkpoint(str(folder / "cev11" / "model.pt"), cev_model, cfg, tokenizer, {"note": "test"})
    return {"folder": folder, "tracked": entry["key"], "tokenizer": tokenizer}


pytestmark = pytest.mark.filterwarnings("ignore:checkpoint has no tokenizer identity:RuntimeWarning")


def test_register_a_folder_of_real_checkpoints(real, store):
    warnings = []
    records = artifacts.register(store, real["folder"], recurse=True, warn=warnings.append)
    kinds = sorted(record.Kind for record in records)
    assert kinds == ["ckpt", "ckpt", "ckpt", "ckpt", "file", "model", "model", "model", "tok"], kinds
    listed = {record.Epoch: record for record in checkpoints.get_checkpoints(store)}
    assert sorted(epoch for epoch in listed if epoch is not None) == [1, 2] and len(listed) == 3
    best, last = listed[2], listed[2]
    assert best.Metric == 2.4 and best.Envelope == "tm2" and best.Dtype == "float32" and best.Parameters == 24928
    names = sorted(Path(path).name for path in best.Locations)
    assert names == ["best.pth", "checkpoint_epoch_0002.pth", "last.pth"], "best.pth and last.pth are locations, not artefacts"
    assert sorted(Path(path).name for path in listed[1].Locations) == ["checkpoint_epoch_0001.pth"] and last is best
    assert best.Tok == artifacts.short(real["tracked"]) and best.Model.startswith("cev:model:triple-v1:")
    assert checkpoints.get_checkpoints(store, best=True)[0].Key == best.Key
    assert checkpoints.get_checkpoints(store, path=best.Path)[0].Key == best.Key
    # The legacy digest stamped by training is kept as an alias of the member, and the canonical key is the member.
    document = store.catalog().artifacts[best.Key].latest
    assert document["members"]["tok"] == real["tracked"]
    assert document["attributes"]["member_aliases"]["tok"][0].startswith("cev:tok:legacy-v3:")
    assert document["explain"]["arch"]["num_heads"] == 4 and document["provenance"] == "legacy-import"
    table = store.get(document["attributes"]["detail"], "tables")
    assert table["weights"] == document["members"]["weights"] and len(table["tensors"]) == 25
    # A bare state dict has no architecture: it is a checkpoint without a model key.
    bare = next(record for record in checkpoints.get_checkpoints(store) if record.Envelope == "bare")
    assert bare.Model is None and bare.Arch is None and bare.Tok is None
    assert any("no architecture" in text for text in warnings)
    # A pickled module is registered by its bytes only, quarantined, and never unpickled.
    unsafe = next(record for record in records if record.Kind == "file")
    assert unsafe.State == "quarantined" and "code execution" in unsafe.Attributes["quarantined"]
    assert unsafe.Attributes["envelope"] == "module"
    # The CEV-11 style checkpoint: byte-BPE tokenizer registered from the file next to it, model key present.
    tokenizer = checkpoints.get_tokenizers(store)[0]
    assert tokenizer.Key == f"cev:tok:bytebpe-v1:{real['tokenizer'].identity()}" and tokenizer.Family == "cevahir-byte-bpe"
    cev = next(record for record in checkpoints.get_checkpoints(store) if record.Envelope == "cev11-v1")
    assert cev.Tok == tokenizer.Short
    models = checkpoints.get_models(store)
    assert len(models) == 3 and all(not model.Published and len(model.Checkpoints) == 1 for model in models)
    up = [(row.Level, artifacts.short(row.Source), row.Role) for row in artifacts.lineage_of(store, cev.Model, "Up")]
    assert up == [(1, cev.Short, "checkpoint"), (2, tokenizer.Short, "tokenizer")]
    assert [row.Target for row in artifacts.lineage_of(store, tokenizer.Key, "Down")] == [cev.Key, cev.Model]
    assert lineage.diverging(store) == []
    recorded = events(store)
    assert recorded.count("artifact.registered") == 9 and recorded.count("artifact.located") == 2
    assert AuditLog(store).verify()["ok"]
    again = artifacts.register(store, real["folder"], recurse=True)
    assert not any(record.New for record in again) and events(store) == recorded, "registering again records nothing"


def test_registered_checkpoint_is_tested_against_its_manifest(real, store, tmp_path):
    folder = tmp_path / "copy"
    shutil.copytree(real["folder"] / "checkpoints", folder)
    artifacts.register(store, folder, recurse=True)
    checkpoint = checkpoints.get_checkpoints(store, best=True)[0]
    file_level = checkpoints.test_checkpoint(store, checkpoint.Key)
    assert file_level.Ok and file_level.Level == "File" and file_level.Purpose == "inspect" and file_level.Registered
    assert (file_level.C1, file_level.C12, file_level.C3) == ("pass", "pass", "-"), "file level reads no tensors"
    assert "file bytes equal the expected key" in next(c.Detail for c in file_level.Checks if c.Id == "C1")
    serve = checkpoints.test_checkpoint(store, checkpoint.Short, level="Compat", purpose="serve",
                                        target=checkpoints.build_target("tiny"))
    assert serve.Ok and serve.Verdict == "compatible", "registered bytes and weights make a legacy checkpoint verifiable"
    assert {check.Id: check.Status for check in serve.Checks}["C3"] == "pass"
    wrong = checkpoints.test_checkpoint(store, checkpoint.Key, level="Compat", purpose="evaluate",
                                        target=checkpoints.build_target("cev11-primary"))
    assert not wrong.Ok and "C5" in wrong.Failing and "architecture differs from the target" in wrong.Summary
    assert artifacts.get(store, checkpoint.Key)[0].State == "verified", "another target is not a reason to quarantine"
    # The same file, unregistered, can only say "unknown" about its integrity.
    loose = tmp_path / "loose.pth"
    shutil.copyfile(checkpoint.Path, loose)
    fresh = Store(tmp_path / "fresh", environ={})
    unknown = checkpoints.test_checkpoint(fresh, path=loose, purpose="serve", level="Weights")
    assert not unknown.Registered and unknown.C1 == "unknown" and not unknown.Ok and unknown.Key.startswith("cev:ckpt:")
    # Tampering after registration: tensor bytes of every copy of the file are changed, and the test says so.
    for path in checkpoint.Locations:
        with open(path, "r+b") as stream:
            stream.seek(os.path.getsize(path) // 2)
            original = stream.read(8)
            stream.seek(-8, os.SEEK_CUR)
            stream.write(bytes(byte ^ 0xFF for byte in original))
    tampered = checkpoints.test_checkpoint(store, checkpoint.Key)
    assert not tampered.Ok and tampered.C1 == "fail" and tampered.Intact == "no" and tampered.Verdict == "incompatible"
    assert artifacts.get(store, checkpoint.Key)[0].State == "quarantined"
    verified = [event for event in AuditLog(store).events() if event["event"] == "artifact.verified"]
    assert len(verified) == 4 and verified[-1]["outcome"] == {
        "status": "failed", "verdict": "incompatible", "failing": ["C1"], "warnings": [], "intact": "no",
        "envelope_safety": "pass"}
    assert verified[-1]["targets"] == [{"kind": "ckpt", "key": checkpoint.Key}] and verified[-1]["class"] == "read"
    assert tampered.AuditSeq == verified[-1]["seq"]
    silent = checkpoints.test_checkpoint(store, checkpoint.Key, record=False)
    assert not silent.Ok and len([e for e in AuditLog(store).events() if e["event"] == "artifact.verified"]) == 4
    with pytest.raises(NotFound, match="not a checkpoint"):
        checkpoints.test_checkpoint(store, checkpoint.Model)
    with pytest.raises(ValueError):
        checkpoints.test_checkpoint(store, checkpoint.Key, level="Deep")
    # A file cut at the end is no longer a readable archive: reported as such, never a traceback.
    other = next(record for record in checkpoints.get_checkpoints(store) if record.Key != checkpoint.Key)
    for path in other.Locations:
        os.truncate(path, os.path.getsize(path) - 64)
    cut = checkpoints.test_checkpoint(store, other.Key)
    assert not cut.Ok and cut.C1 == "fail" and cut.C12 == "unsupported" and artifacts.get(store, other.Key)[0].State == "quarantined"
    os.remove(other.Locations[0])
    with pytest.raises(NotFound, match="has no file on this machine"):
        checkpoints.test_checkpoint(store, other.Key)


def test_compare_key_names_the_fields_and_tensors_that_differ(real, store):
    artifacts.register(store, real["folder"], recurse=True)
    listed = checkpoints.get_checkpoints(store)
    first, second = (next(record for record in listed if record.Epoch == epoch) for epoch in (1, 2))
    cev = next(record for record in listed if record.Envelope == "cev11-v1")
    rows = {row.Member: row for row in key_ops.compare(store, first.Key, second.Short)}
    assert rows["arch"].Equal and rows["tok"].Equal and rows["wschema"].Equal and not rows["weights"].Equal and not rows["key"].Equal
    assert len(rows["weights"].Changed) > 5 and rows["weights"].OnlyReference == [] and "changed:" in rows["weights"].Detail
    assert rows["key"].Reference == first.Short and rows["key"].DifferenceKey == second.Key
    other = {row.Member: row for row in key_ops.compare(store, first.Key, cev.Key)}
    assert not other["arch"].Equal and "vocab_size" in other["arch"].Fields and "128 -> " in other["arch"].Detail
    same = key_ops.compare(store, first.Key, first.Path)
    assert all(row.Equal for row in same), "a registered file and its key are the same artefact"
    info = {record.Name: record for record in key_ops.keys_of(store, first.Path)}
    assert info["file:bytes-v1"].Source == "file" and info["ckpt"].Key == first.Key and info["weights"].Source == "manifest"
    explained = key_ops.keys_of(store, key=first.Short, explain=True)[0]
    assert explained.Key == first.Key and explained.Explain["arch"]["num_layers"] == 2 and explained.Status == "primary"


def test_real_checkpoint_moves_between_machines_in_a_bundle(real, store, tmp_path):
    artifacts.register(store, real["folder"] / "cev11")
    cev = checkpoints.get_checkpoints(store)[0]
    bundle = artifacts.export(store, [cev.Model], tmp_path / "cev.cvb", include_payload=False)
    assert bundle.Manifests == 3, "model, checkpoint and tokenizer manifests"
    artifacts.export(store, [cev.Key, checkpoints.get_tokenizers(store)[0].Key], tmp_path / "cev-full.cvb", include_payload=True)
    workstation = other_home(tmp_path, "workstation")
    imported = artifacts.import_bundle(workstation, tmp_path / "cev-full.cvb")
    assert sorted(record.Kind for record in imported) == ["ckpt", "tok"] and all(record.State == "imported" for record in imported)
    verdict = checkpoints.test_checkpoint(workstation, cev.Short, level="Weights", purpose="inspect")
    assert verdict.Ok and verdict.C1 == "pass" and verdict.C3 == "pass" and verdict.Envelope == "cev11-v1"
    assert artifacts.get(workstation, cev.Key)[0].State == "verified", "imported, then verified here"
