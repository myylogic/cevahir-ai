"""read_only=True: the tokenizer never writes vocab.json / merges.txt and its vocabulary stays fixed.

Every tokenizer here works on temporary copies of the shipped assets whose mtimes are set far in the
past, so any rewrite (even with identical bytes) is visible. Before the fix, a read_only TokenizerCore
rewrote vocab.json on a syllable-only train encode, save(), auto_update_vocab(), set_vocab(),
train_model() and reset() (the last two also rewrote merges.txt).
"""
import hashlib
import os
import shutil
from pathlib import Path

import pytest

from tokenizer_management.bpe.bpe_manager import BPEManager, TokenizerReadOnlyError
from tokenizer_management.core.tokenizer_core import TokenizerCore, TokenizerCoreError
from training_system.cache_identity import tokenizer_digest

ROOT = Path(__file__).resolve().parents[2]
OLD_MTIME_NS = 10**18  # 2001-09-09; a rewrite shows up as a newer mtime even with identical bytes
TEXT = "Merhaba dünya, bugün hava çok güzel. Kitaplarımız masada duruyor."


def snapshot(directory):
    return {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in sorted(directory.iterdir())}


def copy_assets(directory):
    for source, name in (("data/vocab_lib/vocab.json", "vocab.json"), ("data/merges_lib/merges.txt", "merges.txt")):
        target = directory / name
        shutil.copyfile(ROOT / source, target)
        os.utime(target, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    return snapshot(directory)


def read_only_core(directory, **extra):
    return TokenizerCore({"vocab_path": str(directory / "vocab.json"), "merges_path": str(directory / "merges.txt"),
                          "read_only": True, "use_gpu": False, **extra})


def forget_managers(directory):
    """BPEManager caches instances per configuration; drop the ones built on this test's copies."""
    BPEManager._instances = {key: manager for key, manager in BPEManager._instances.items()
                             if not str(manager.vocab_file).startswith(str(directory))}


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    directory = tmp_path_factory.mktemp("read_only_tokenizer")
    original = copy_assets(directory)
    yield directory, original
    forget_managers(directory)


@pytest.fixture(scope="module")
def core(assets):
    return read_only_core(assets[0])


def test_encode_decode_and_reload_never_touch_the_files(assets, core):
    directory, original = assets
    vocab_size, identity = core.get_vocab_size(), tokenizer_digest(core)

    _, ids = core.encode(TEXT)
    assert core.decode(ids)
    core.batch_encode([TEXT, "Ankara İç Anadolu'dadır."])
    core.encode(TEXT, mode="train")
    # Syllable-only train encoding used to add unseen syllables to vocab.json.
    _, syllable_ids = core.encode("kitaplarımızdan yepyenikelimeler", mode="train",
                                  include_whole_words=False, include_syllables=True)
    assert syllable_ids and max(syllable_ids) < vocab_size
    core.decode(syllable_ids)
    core.finalize_vocab()
    core.finalize_vocab(["kitaplarımızdan yepyenikelimeler"])

    assert snapshot(directory) == original
    assert core.get_vocab_size() == vocab_size
    assert tokenizer_digest(core) == identity


def test_auto_vocab_update_is_ignored_when_read_only(assets):
    directory, original = assets
    # The shipped vocabulary already sits at the default max_vocab_size, which would hide the growth.
    core = read_only_core(directory, bpe_config={"auto_vocab_update": True, "max_vocab_size": 10**6})
    vocab_size = core.get_vocab_size()
    _, ids = core.encode("yepyenikelimeler uydurulmuş sözcükler", mode="train")
    assert ids and max(ids) < vocab_size
    assert core.get_vocab_size() == vocab_size
    assert snapshot(directory) == original


@pytest.mark.parametrize("operation", [
    pytest.param(lambda core: core.save(), id="save"),
    pytest.param(lambda core: core.reset(), id="reset"),
    pytest.param(lambda core: core.set_vocab(dict(core.get_vocab())), id="set_vocab"),
    pytest.param(lambda core: core.auto_update_vocab(["yepyenikelime"]), id="auto_update_vocab"),
    pytest.param(lambda core: core.train_model(["merhaba dünya merhaba"] * 3), id="train_model"),
    pytest.param(lambda core: core.tokenizer.save_vocab(), id="BPEManager.save_vocab"),
    pytest.param(lambda core: core.tokenizer.save_merges(), id="BPEManager.save_merges"),
    pytest.param(lambda core: core.tokenizer.ensure_tokens_in_vocab(["yepyenikelimeler"]), id="ensure_tokens_in_vocab"),
])
def test_explicit_writes_are_refused_before_anything_changes(assets, core, operation):
    directory, original = assets
    vocab_size, identity = core.get_vocab_size(), tokenizer_digest(core)
    with pytest.raises(TokenizerReadOnlyError, match="read_only tokenizer cannot"):
        operation(core)
    assert snapshot(directory) == original
    assert core.get_vocab_size() == vocab_size
    assert tokenizer_digest(core) == identity
    _, ids = core.encode(TEXT)
    assert core.decode(ids)  # still usable after the refusal


def test_adding_only_known_tokens_is_a_no_op(assets, core):
    directory, original = assets
    known = [token for token in ("merhaba</w>", "dünya</w>") if token in core.get_vocab()]
    assert known, "shipped vocabulary lacks the probe tokens"
    assert core.auto_update_vocab(known) == 0
    assert snapshot(directory) == original


def test_read_only_creates_no_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        TokenizerCore({"vocab_path": "missing/vocab.json", "merges_path": "missing/merges.txt",
                       "read_only": True, "use_gpu": False})
    assert not list(tmp_path.iterdir())


def test_read_only_does_not_require_a_writable_directory(assets, monkeypatch):
    directory, original = assets
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda path, mode, **kw: False if mode == os.W_OK else real_access(path, mode, **kw))
    with pytest.raises(TokenizerCoreError, match="not writable"):
        TokenizerCore({"vocab_path": str(directory / "vocab.json"), "merges_path": str(directory / "merges.txt"),
                       "use_gpu": False})
    core = read_only_core(directory)
    _, ids = core.encode(TEXT)
    assert ids
    assert snapshot(directory) == original
