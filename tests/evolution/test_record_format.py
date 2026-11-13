"""Formatted training records: BOS only where a document starts, EOS only where it really ends.

Before CACHE_FORMAT_VERSION 2, prepare_cache.py closed every record with EOS, so each chunk of a long
document taught the model that the text ended (34 of the 35 chunks of a 13.7 KB document) although it
continued. The cache key now carries the layout version so caches of the old layout are rebuilt.
Both writers of that cache (prepare_cache.py and the V2 TrainingService) must produce the same records,
and the "(devam N)" parts of a long answer are one document: only the last part ends with EOS.
"""
import hashlib
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from data_loader_management.data_loader_manager import DataLoaderConfig, DataLoaderManager, continuation_question
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


def test_a_one_token_middle_chunk_has_no_target_and_is_not_an_error():
    # Its only token is context (no BOS before it) and nothing follows it (no EOS): no position to train.
    # The V2 writer used to abort the whole cache build on it while prepare_cache skipped it.
    assert format_autoregressive_record([10], starts_document=False, ends_document=False, **FORMAT) is None
    first = format_autoregressive_record([10], starts_document=True, ends_document=False, **FORMAT)
    last = format_autoregressive_record([10], starts_document=False, ends_document=True, **FORMAT)
    assert (content(first[0]), content(first[1])) == ([BOS], [10])
    assert (content(last[0]), content(last[1])) == ([10], [EOS])


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


# ------------------------------------------- end to end: prepare_cache() and the V2 TrainingService
WORDS = ["kitap", "okul", "deniz", "güneş", "ağaç", "şehir", "insan", "zaman", "yol", "ev",
         "su", "toprak", "hava", "bilgi", "dil", "kalem", "masa", "kapı", "pencere", "bahçe"]
LONG_DOCUMENT = " ".join(
    f"Bu belgede {WORDS[i % 20]} ile {WORDS[(3 * i + 1) % 20]} arasında güzel bir {WORDS[(7 * i + 2) % 20]} vardı."
    for i in range(48)
) + " Belgenin son cümlesi burada biter."
SHORT_DOCUMENT = "Kısa belge tek parçadır."
# A short sentence between two sentences longer than the data loader's chunk limit (198 estimated tokens)
# becomes a chunk of its own (DataLoaderManager._smart_split_text): here the one-token middle chunk "Evet".
LONG_SENTENCE_A = " ".join(WORDS[(3 * i) % 20] for i in range(80))
LONG_SENTENCE_B = " ".join(WORDS[(7 * i + 1) % 20] for i in range(80))
SHORT_SENTENCE_DOCUMENT = f"Giriş cümlesi burada. {LONG_SENTENCE_A}. Evet. {LONG_SENTENCE_B}. Son cümle burada biter."
# The data loader splits a long answer into the pair itself and "<question> (devam 2)".
LONG_ANSWER = " ".join(
    f"{WORDS[i % 20].capitalize()} hakkında söylenecek çok şey var ve {WORDS[(5 * i + 3) % 20]} bunlardan biridir."
    for i in range(9)
) + " Cevabın son cümlesi burada biter."
QA = [{"question": "Ankara nerededir?", "answer": "Ankara İç Anadolu Bölgesindedir."},
      {"question": "Deniz ne renktir?", "answer": "Deniz genellikle mavidir."},
      {"question": "Bu konular hakkında ne düşünüyorsun?", "answer": LONG_ANSWER}]
MAX_SEQ_LENGTH = 64  # the tokenizer chunks records longer than 62 tokens with a 20-token overlap


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from tokenizer_management.bpe.bpe_manager import BPEManager

    base = tmp_path_factory.mktemp("record_format")
    data, assets = base / "education", base / "assets"
    for directory in (data, assets):
        directory.mkdir()
    (data / "a_uzun.txt").write_text(LONG_DOCUMENT, encoding="utf-8")
    (data / "b_kisa.txt").write_text(SHORT_DOCUMENT, encoding="utf-8")
    (data / "c_soru.json").write_text(json.dumps(QA, ensure_ascii=False), encoding="utf-8")
    (data / "d_kisa_cumle.txt").write_text(SHORT_SENTENCE_DOCUMENT, encoding="utf-8")
    shutil.copyfile(ROOT / "data/vocab_lib/vocab.json", assets / "vocab.json")
    shutil.copyfile(ROOT / "data/merges_lib/merges.txt", assets / "merges.txt")
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir()}
    yield {"base": base, "data": data, "assets": assets, "digests": digests}
    BPEManager._instances = {k: m for k, m in BPEManager._instances.items()
                             if not str(m.vocab_file).startswith(str(base))}


@pytest.fixture(scope="module")
def prepared(corpus):
    import training_system.prepare_cache as prepare_module

    cache, assets = corpus["base"] / "cache", corpus["assets"]
    patch = pytest.MonkeyPatch()
    patch.setattr(prepare_module, "BPE_CONFIG", {**prepare_module.BPE_CONFIG,
                                                 "vocab_file": str(assets / "vocab.json"),
                                                 "merges_file": str(assets / "merges.txt")})
    try:
        prepare_module.prepare_cache(data_dir=str(corpus["data"]), cache_dir=str(cache),
                                     max_seq_length=MAX_SEQ_LENGTH, include_whole_words=True,
                                     include_syllables=False, include_sep=False)
    finally:
        patch.undo()
    [cache_file] = sorted(cache.glob("cached_data_*.pkl"))
    with cache_file.open("rb") as stream:
        records = pickle.load(stream)
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    tokenizer = TokenizerCore({"vocab_path": str(assets / "vocab.json"), "merges_path": str(assets / "merges.txt"),
                               "read_only": True, "use_gpu": False})
    return {"records": records, "cache_file": cache_file, "tokenizer": tokenizer}


@pytest.fixture(scope="module")
def v2(corpus):
    """The other writer of the same cache: V2 TrainingService.prepare_from_cache on the same corpus."""
    from model_management.config_schema import tiny_model_config
    from training_system.v2.core.training_service import TrainingService

    cache, assets = corpus["base"] / "v2_cache", corpus["assets"]
    config = tiny_model_config(num_layers=1, max_seq_length=MAX_SEQ_LENGTH, max_cache_len=MAX_SEQ_LENGTH)
    config.update(data_dir=str(corpus["data"]), cache_dir=str(cache), vocab_path=str(assets / "vocab.json"),
                  merges_path=str(assets / "merges.txt"), read_only=True, use_gpu=False, device="cpu",
                  pad_token_id=PAD, train_include_whole_words=True, train_include_syllables=False,
                  train_include_sep=False)
    service = TrainingService(config)
    raw = service.tokenizer_core.load_training_data(encode_mode="train", include_whole_words=True,
                                                    include_syllables=False, include_sep=False)
    service.prepare_from_cache(data_cache=DataCache(str(corpus["data"]), str(cache)),
                               tokenizer_core=service.tokenizer_core, config=service.config)
    [cache_file] = sorted(cache.glob("cached_data_*.pkl"))
    with cache_file.open("rb") as stream:
        return {"raw": raw, "records": pickle.load(stream)}


def by_source(records):
    groups = defaultdict(list)
    for inp, tgt, source in records:
        groups[source].append((inp, tgt))
    return groups


def text_of(tokenizer, inp, tgt):
    """Decoded text of one formatted record: its whole token sequence without BOS and EOS."""
    return tokenizer.decode([t for t in content(inp) + content(tgt)[-1:] if t not in (BOS, EOS)])


def document_with(prepared, phrase):
    """The records of the one document whose text contains ``phrase``."""
    groups = by_source(prepared["records"])
    found = [source for source, chunks in groups.items()
             if any(phrase in text_of(prepared["tokenizer"], inp, tgt) for inp, tgt in chunks)]
    assert len(found) == 1, f"{phrase!r} should be in exactly one document, found in sources {found}"
    return groups[found[0]]


def assert_one_bos_and_one_eos(chunks):
    assert [content(inp)[0] == BOS for inp, _ in chunks] == [True] + [False] * (len(chunks) - 1)
    assert all(BOS not in content(inp)[1:] for inp, _ in chunks)
    assert [content(tgt).count(EOS) for _, tgt in chunks] == [0] * (len(chunks) - 1) + [1]
    assert content(chunks[-1][1])[-1] == EOS


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
    chunks = document_with(prepared, "Bu belgede")
    assert len(chunks) >= 3, f"the long document should span several records, got {len(chunks)}"
    assert_one_bos_and_one_eos(chunks)
    last_text = prepared["tokenizer"].decode(content(chunks[-1][0])[1:])
    assert "Belgenin son" in last_text, last_text  # the EOS follows the document's real last words


def test_short_documents_and_questions_keep_bos_and_eos(prepared):
    groups = by_source(prepared["records"])
    single = {source: chunks for source, chunks in groups.items() if len(chunks) == 1}
    assert len(single) == 3, f"short text + 2 short QA pairs expected as one record each: {sorted(single)}"
    for [(inp, tgt)] in single.values():
        assert content(inp)[0] == BOS and content(tgt)[-1] == EOS
        assert content(inp)[1:] == content(tgt)[:-1]
        assert content(tgt).count(EOS) == 1


def test_the_devam_parts_of_a_long_answer_are_one_document(corpus, prepared):
    # Each part used to get its own source id, so every part but the last ended with a false EOS.
    questions = [question for question, _ in DataLoaderManager(DataLoaderConfig(data_dir=corpus["data"])).load()]
    assert continuation_question(QA[2]["question"], 2) in questions, "the data loader no longer splits the answer"
    chunks = document_with(prepared, "hakkında söylenecek")
    texts = [text_of(prepared["tokenizer"], inp, tgt) for inp, tgt in chunks]
    assert any("devam" in text for text in texts[1:]), texts
    assert_one_bos_and_one_eos(chunks)
    assert "Cevabın son" in texts[-1], texts[-1]


def test_a_one_token_middle_chunk_is_skipped_by_both_writers(prepared, v2):
    raw, boundaries = v2["raw"], record_boundaries(v2["raw"])
    middle_tokens = [list(ids) for (ids, _, _), (starts, ends) in zip(raw, boundaries)
                     if not starts and not ends and len(ids) == 1]
    assert [prepared["tokenizer"].decode(ids).strip() for ids in middle_tokens] == ["Evet"]
    assert len(v2["records"]) == len(raw) - 1  # the V2 writer aborted the whole cache build here
    chunks = document_with(prepared, "Giriş cümlesi")
    assert_one_bos_and_one_eos(chunks)
    assert not any("Evet" in text_of(prepared["tokenizer"], inp, tgt) for inp, tgt in chunks)


def test_both_cache_writers_produce_the_same_records(prepared, v2):
    assert v2["records"] == prepared["records"]


def test_preparing_the_cache_records_the_layout_and_leaves_the_tokenizer_untouched(corpus, prepared, v2):
    meta = json.loads(prepared["cache_file"].with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["cache_format_version"] == cache_identity.CACHE_FORMAT_VERSION == 2
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in corpus["assets"].iterdir()}
    assert digests == corpus["digests"]
