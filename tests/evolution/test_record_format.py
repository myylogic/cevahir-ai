"""Formatted training records: BOS only where a document starts, EOS only where it really ends.

Before CACHE_FORMAT_VERSION 2, prepare_cache.py closed every record with EOS, so each chunk of a long
document taught the model that the text ended (34 of the 35 chunks of a 13.7 KB document) although it
continued. The cache key now carries the layout version so caches of the old layout are rebuilt.
"""
import hashlib
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from training_system import cache_identity
from training_system.data_cache import DataCache
from training_system.record_format import format_autoregressive_record, record_boundaries
from training_system.v3.data.cache_v3 import DataCacheV3

ROOT = Path(__file__).resolve().parents[2]
BOS, EOS, PAD, UNK = 2, 3, 0, 1
FORMAT = dict(bos_id=BOS, eos_id=EOS, pad_id=PAD, unk_id=UNK, vocab_size=100, max_seq_length=8)


def content(seq):
    end = len(seq)
    while end and seq[end - 1] == PAD:
        end -= 1
    return seq[:end]


@pytest.mark.parametrize("starts, ends, expected_input, expected_target", [
    (True, True, [BOS, 10, 11, 12], [10, 11, 12, EOS]),    # a document that fits in one record
    (True, False, [BOS, 10, 11], [10, 11, 12]),            # first chunk of a long document
    (False, False, [10, 11], [11, 12]),                    # continuation chunk: neither BOS nor EOS
    (False, True, [10, 11, 12], [11, 12, EOS]),            # last chunk: the only one closed with EOS
])
def test_record_is_the_shift_of_one_sequence(starts, ends, expected_input, expected_target):
    inp, tgt = format_autoregressive_record([10, 11, 12], starts_document=starts, ends_document=ends, **FORMAT)
    assert len(inp) == len(tgt) == FORMAT["max_seq_length"]
    assert content(inp) == expected_input and content(tgt) == expected_target
    assert tgt[:len(expected_target) - 1] == inp[1:len(expected_input)]  # target[i] == input[i + 1]
    assert inp[len(expected_input):] == tgt[len(expected_target):] == [PAD] * (8 - len(expected_input))


def test_out_of_vocabulary_ids_become_unk():
    inp, tgt = format_autoregressive_record([10, 500, -1], starts_document=True, ends_document=True, **FORMAT)
    assert content(inp) == [BOS, 10, UNK, UNK] and content(tgt) == [10, UNK, UNK, EOS]


def test_records_that_do_not_fit_are_rejected_instead_of_cut():
    exact = format_autoregressive_record(list(range(10, 17)), starts_document=True, ends_document=True, **FORMAT)
    assert content(exact[1])[-1] == EOS  # 7 tokens + BOS + EOS use exactly 8 positions
    with pytest.raises(ValueError, match="chunk it first"):
        format_autoregressive_record(list(range(10, 18)), starts_document=True, ends_document=True, **FORMAT)
    with pytest.raises(ValueError, match="no token to predict"):
        format_autoregressive_record([10], starts_document=False, ends_document=False, **FORMAT)


def test_boundaries_follow_runs_of_the_same_source():
    records = [([1], [1], 7), ([2], [2], 7), ([3], [3], 7), ([4], [4], 9), ([5], [5]), ([6], [6], 8)]
    assert record_boundaries(records) == [
        (True, False), (False, False), (False, True), (True, True), (True, True), (True, True)]
    with pytest.raises(ValueError, match="not consecutive"):
        record_boundaries([([1], [1], 1), ([2], [2], 2), ([3], [3], 1)])


def test_cache_key_carries_the_record_layout_version(tmp_path, monkeypatch):
    prepared = DataCache(str(tmp_path), str(tmp_path / "cache"))
    strict = DataCacheV3(str(tmp_path), str(tmp_path / "cache"))
    args = ("train", True, False, False, 64, "tokenizer-digest")
    current = prepared._get_cache_key(*args)
    assert current == strict.get_cache_key(*args)
    monkeypatch.setattr(cache_identity, "CACHE_FORMAT_VERSION", 1)
    old_layout = prepared._get_cache_key(*args)
    assert old_layout == strict.get_cache_key(*args)
    assert old_layout != current, "a cache written under another record layout must not be found"


# --------------------------------------------------------------------- end to end: prepare_cache()
WORDS = ["kitap", "okul", "deniz", "güneş", "ağaç", "şehir", "insan", "zaman", "yol", "ev",
         "su", "toprak", "hava", "bilgi", "dil", "kalem", "masa", "kapı", "pencere", "bahçe"]
LONG_DOCUMENT = " ".join(
    f"Bu belgede {WORDS[i % 20]} ile {WORDS[(3 * i + 1) % 20]} arasında güzel bir {WORDS[(7 * i + 2) % 20]} vardı."
    for i in range(48)
) + " Belgenin son cümlesi burada biter."
SHORT_DOCUMENT = "Kısa belge tek parçadır."
QA = [{"question": "Ankara nerededir?", "answer": "Ankara İç Anadolu Bölgesindedir."},
      {"question": "Deniz ne renktir?", "answer": "Deniz genellikle mavidir."}]
MAX_SEQ_LENGTH = 64  # the tokenizer chunks records longer than 62 tokens with a 20-token overlap


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    from tokenizer_management.bpe.bpe_manager import BPEManager
    import training_system.prepare_cache as prepare_module

    base = tmp_path_factory.mktemp("record_format")
    data, cache, assets = base / "education", base / "cache", base / "assets"
    for directory in (data, assets):
        directory.mkdir()
    (data / "a_uzun.txt").write_text(LONG_DOCUMENT, encoding="utf-8")
    (data / "b_kisa.txt").write_text(SHORT_DOCUMENT, encoding="utf-8")
    (data / "c_soru.json").write_text(json.dumps(QA, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(ROOT / "data/vocab_lib/vocab.json", assets / "vocab.json")
    shutil.copyfile(ROOT / "data/merges_lib/merges.txt", assets / "merges.txt")
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir()}

    patch = pytest.MonkeyPatch()
    patch.setattr(prepare_module, "BPE_CONFIG", {**prepare_module.BPE_CONFIG,
                                                 "vocab_file": str(assets / "vocab.json"),
                                                 "merges_file": str(assets / "merges.txt")})
    try:
        prepare_module.prepare_cache(data_dir=str(data), cache_dir=str(cache), max_seq_length=MAX_SEQ_LENGTH,
                                     include_whole_words=True, include_syllables=False, include_sep=False)
    finally:
        patch.undo()
    [cache_file] = sorted(cache.glob("cached_data_*.pkl"))
    with cache_file.open("rb") as stream:
        records = pickle.load(stream)
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    tokenizer = TokenizerCore({"vocab_path": str(assets / "vocab.json"), "merges_path": str(assets / "merges.txt"),
                               "read_only": True, "use_gpu": False})
    yield {"records": records, "cache_file": cache_file, "assets": assets, "digests": digests, "tokenizer": tokenizer}
    BPEManager._instances = {k: m for k, m in BPEManager._instances.items()
                             if not str(m.vocab_file).startswith(str(base))}


def by_source(records):
    groups = defaultdict(list)
    for inp, tgt, source in records:
        groups[source].append((inp, tgt))
    return groups


def test_every_record_is_a_one_position_shift(prepared):
    records = prepared["records"]
    assert records and all(len(item) == 3 for item in records)
    for inp, tgt, _ in records:
        assert len(inp) == len(tgt) == MAX_SEQ_LENGTH
        inp_content, tgt_content = content(inp), content(tgt)
        assert len(inp_content) == len(tgt_content) >= 1
        assert tgt_content[:-1] == inp_content[1:]
        assert PAD not in inp_content and PAD not in tgt_content


def test_long_document_has_exactly_one_eos_at_its_end_and_one_bos_at_its_start(prepared):
    groups = by_source(prepared["records"])
    long_source = max(groups, key=lambda source: len(groups[source]))
    chunks = groups[long_source]
    assert len(chunks) >= 3, f"the long document should span several records, got {len(chunks)}"
    eos_counts = [content(tgt).count(EOS) for _, tgt in chunks]
    assert eos_counts == [0] * (len(chunks) - 1) + [1]
    assert content(chunks[-1][1])[-1] == EOS
    assert [content(inp)[0] == BOS for inp, _ in chunks] == [True] + [False] * (len(chunks) - 1)
    assert all(BOS not in content(inp)[1:] for inp, _ in chunks)
    last_text = prepared["tokenizer"].decode(content(chunks[-1][0])[1:])
    assert "burada biter" in last_text, last_text  # the EOS follows the document's real last words


def test_short_documents_and_questions_keep_bos_and_eos(prepared):
    groups = by_source(prepared["records"])
    single = {source: chunks for source, chunks in groups.items() if len(chunks) == 1}
    assert len(single) == 3, f"short text + 2 QA pairs expected as one record each: {sorted(single)}"
    for [(inp, tgt)] in single.values():
        assert content(inp)[0] == BOS and content(tgt)[-1] == EOS
        assert content(inp)[1:] == content(tgt)[:-1]
        assert content(tgt).count(EOS) == 1


def test_preparing_the_cache_records_the_layout_and_leaves_the_tokenizer_untouched(prepared):
    meta = json.loads(prepared["cache_file"].with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["cache_format_version"] == cache_identity.CACHE_FORMAT_VERSION == 2
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared["assets"].iterdir()}
    assert digests == prepared["digests"]
