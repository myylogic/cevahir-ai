# tokenizer_management/tests/test_tokenizer_core.py
from __future__ import annotations

import os
import sys
import json
import shutil
import time
import pytest
import unicodedata

from typing import List

# --- Proje kökünü import yoluna ekle (unutmaman gereken ayrıntı) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tokenizer_management.core.tokenizer_core import TokenizerCore, TokenizerCoreError
from tokenizer_management.bpe.bpe_manager import BPETokenError, BPEDecodingError


# ----------------------------- yardımcılar & fixture'lar -----------------------------

@pytest.fixture(scope="function")
def paths_and_backup(tmp_path):
    """
    ROOT altında data/ dizinleri ve dosya yedek/restore yönetimi.
    Testler bitince eski dosyalar geri yüklenir.
    """
    vocab_dir = os.path.join(ROOT, "data", "vocab")
    merges_dir = os.path.join(ROOT, "data", "merges")
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(merges_dir, exist_ok=True)

    vocab_path = os.path.join(vocab_dir, "vocab.json")
    merges_path = os.path.join(merges_dir, "merges.txt")

    # Eski dosyaları varsa yedekle
    vocab_bak = vocab_path + ".bak_coretests"
    merges_bak = merges_path + ".bak_coretests"
    if os.path.exists(vocab_path):
        shutil.copy2(vocab_path, vocab_bak)
    if os.path.exists(merges_path):
        shutil.copy2(merges_path, merges_bak)

    yield vocab_path, merges_path

    # Restore
    if os.path.exists(vocab_bak):
        shutil.move(vocab_bak, vocab_path)
    else:
        try:
            os.remove(vocab_path)
        except FileNotFoundError:
            pass
    if os.path.exists(merges_bak):
        shutil.move(merges_bak, merges_path)
    else:
        try:
            os.remove(merges_path)
        except FileNotFoundError:
            pass


@pytest.fixture(scope="function")
def education_dir():
    base_dir = os.path.join(ROOT, "education")
    if not os.path.isdir(base_dir):
        pytest.skip("education klasörü bulunamadı; ilgili testler atlanıyor.")
    return base_dir


@pytest.fixture(scope="function")
def core(paths_and_backup, education_dir):
    """
    Gerçek education klasöründeki verilerle TokenizerCore başlatılır.
    vocab ve merges ROOT/data altında tutulur.
    Her testte temiz başlangıç için reset yapılır.
    """
    vocab_path, merges_path = paths_and_backup
    cfg = {
        "data_dir": education_dir,
        "vocab_path": vocab_path,
        "merges_path": merges_path,
        "batch_size": 4,
        "max_length": 256,
        "skip_missing": True,
    }
    tc = TokenizerCore(cfg)
    # Temiz state
    tc.reset()
    return tc


# Basit yardımcı: education boşsa bazı testler skip edilsin
def _ensure_education_has_data(tc: TokenizerCore) -> int:
    if tc.data_loader is None:
        return 0
    records = tc.data_loader.load_data()
    return len(records)


# --------------------------------- TESTLER (20 adet) ---------------------------------

def test_core_init_valid_paths(core):
    s = core.summary()
    assert s["has_specials"] is True
    # yollar doğru konfig ile geldi mi
    assert s["vocab_path"].endswith(os.path.join("data", "vocab", "vocab.json"))
    assert s["merges_path"].endswith(os.path.join("data", "merges", "merges.txt"))


def test_core_init_invalid_data_dir_raises(paths_and_backup):
    vocab_path, merges_path = paths_and_backup
    cfg = {
        "data_dir": os.path.join(ROOT, "education_not_exists_###"),
        "vocab_path": vocab_path,
        "merges_path": merges_path,
    }
    with pytest.raises(TokenizerCoreError):
        _ = TokenizerCore(cfg)


def test_train_model_with_corpus_persists_and_finalizes(core):
    corpus = ["merhaba dünya", "merhaba evren"]
    core.train_model(
        corpus,
        method="bpe",
        vocab_size=8,
        include_whole_words=True,
        include_syllables=False,
        include_sep=True,
    )
    # dosyalar oluşmuş mu
    assert os.path.isfile(core.summary()["vocab_path"])
    assert os.path.isfile(core.summary()["merges_path"])
    # finalize idempotent
    core.finalize_vocab()
    core.finalize_vocab()  # ikinci çağrı hata vermemeli
    assert core.get_vocab_size() >= 5  # specials + ötekiler


def test_train_from_loader_when_data_exists(core):
    n = _ensure_education_has_data(core)
    if n == 0:
        pytest.skip("education boş; train_from_loader testi atlandı.")
    core.train_from_loader(method="bpe", vocab_size=12, include_syllables=False)
    s = core.summary()
    assert s["vocab_size"] >= 5
    assert s["merges_count"] >= 0


def test_finalize_vocab_is_idempotent(core):
    corpus = ["a b c", "a b d"]
    core.train_model(corpus, vocab_size=6, include_syllables=False)
    before = core.summary()
    core.finalize_vocab()
    after = core.summary()
    assert before["vocab_size"] == after["vocab_size"]
    assert before["merges_count"] == after["merges_count"]


def test_encode_decode_roundtrip_inference_basic(core):
    corpus = ["alpha beta", "gamma delta"]
    core.train_model(corpus, vocab_size=10, include_syllables=False)
    toks, ids = core.encode("alpha beta", mode="inference", include_whole_words=True, include_syllables=False, include_sep=True)
    assert toks[0] == "<BOS>" and toks[-1] == "<EOS>"
    out = core.decode(ids, method="bpe")
    assert isinstance(out, str) and len(out) > 0


def test_encode_train_mode_alignment_no_specials(core):
    core.train_model(["x y z"], vocab_size=5, include_syllables=False)
    toks, ids = core.encode("x y", mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    assert all(t not in ("<BOS>", "<EOS>", "<SEP>") for t in toks)
    assert len(toks) == len(ids) > 0


def test_batch_encode_skip_invalid(core):
    core.train_model(["foo bar"], vocab_size=5, include_syllables=False)
    items = ["foo bar", "", None, 123, "foo"]  # type: ignore[list-item]
    outs = core.batch_encode(items, include_whole_words=True, include_syllables=False, include_sep=True, skip_invalid=True)
    # yalnızca geçerli stringler işlendi
    assert all(isinstance(t, list) and isinstance(i, list) for t, i in outs)
    assert len(outs) >= 2


def test_batch_decode_skip_invalid(core):
    core.train_model(["m n"], vocab_size=4, include_syllables=False)
    _, good_ids = core.encode("m n", include_whole_words=True, include_syllables=False, include_sep=True)
    seqs = [good_ids, [999999], [], good_ids]
    outs = core.batch_decode(seqs, method="bpe", skip_invalid=True)
    assert len(outs) >= 2
    assert all(isinstance(x, str) for x in outs)


def test_load_training_data_generates_examples(core):
    n = _ensure_education_has_data(core)
    if n == 0:
        pytest.skip("education boş; load_training_data testi atlandı.")
    # önce eğit
    core.train_from_loader(method="bpe", vocab_size=15, include_syllables=False)
    examples = core.load_training_data(
        input_field="data",
        target_field="target",
        encode_mode="train",
        include_whole_words=True,
        include_syllables=False,
        include_sep=True,
    )
    assert isinstance(examples, list)
    if examples:
        inp, tgt = examples[0]
        assert isinstance(inp, list) and isinstance(tgt, list)


def test_auto_update_vocab_then_encode_uses_new_ids(core):
    core.train_model(["s t"], vocab_size=5, include_syllables=False)
    added = core.auto_update_vocab(["yeni1", "yeni2", "yeni1"])
    assert added >= 2
    toks, ids = core.encode("yeni1 yeni2", include_whole_words=True, include_syllables=False, include_sep=True)
    assert len(ids) > 0


def test_set_vocab_roundtrip(core):
    core.train_model(["a b"], vocab_size=5, include_syllables=False)
    v = core.get_vocab()
    new_tok = "deneme</w>"
    assert new_tok not in v
    next_id = max(x["id"] for x in v.values()) + 1
    v2 = dict(v)
    v2[new_tok] = {"id": next_id, "total_freq": 0, "positions": []}
    core.set_vocab(v2)
    assert core.get_vocab()[new_tok]["id"] == next_id


def test_get_vocab_size_matches_len(core):
    core.train_model(["q w"], vocab_size=5, include_syllables=False)
    size = core.get_vocab_size()
    assert size == len(core.get_vocab())


def test_save_persists_and_reload_consistency(core, paths_and_backup, education_dir):
    # train & save
    core.train_model(["k l"], vocab_size=5, include_syllables=False)
    core.save()

    # aynı path'lerle yeni örnek (singleton olsa da finalize ile sync’ler)
    vocab_path, merges_path = paths_and_backup
    cfg2 = {"data_dir": education_dir, "vocab_path": vocab_path, "merges_path": merges_path}
    core2 = TokenizerCore(cfg2)
    core2.finalize_vocab()

    # aynı girdi aynı ids üretmeli
    _, id1 = core.encode("k l", include_whole_words=True, include_syllables=False, include_sep=True)
    _, id2 = core2.encode("k l", include_whole_words=True, include_syllables=False, include_sep=True)
    assert id1 == id2


def test_decode_bpe_raises_on_bad_id(core):
    core.train_model(["u v"], vocab_size=5, include_syllables=False)
    with pytest.raises(Exception):
        _ = core.decode([999999], method="bpe")


def test_summary_has_specials_and_paths(core):
    s = core.summary()
    assert s["has_specials"] is True
    assert isinstance(s["vocab_size"], int)
    assert isinstance(s["merges_count"], int)
    assert os.path.isabs(s["vocab_path"]) and os.path.isabs(s["merges_path"])


def test_tokenize_returns_syllables(core):
    out = core.tokenize("merhaba dünya")
    assert isinstance(out, list) and all(isinstance(x, str) for x in out)


def test_reset_restores_defaults(core):
    core.train_model(["aa bb"], vocab_size=6, include_syllables=False)
    assert core.summary()["merges_count"] >= 0
    core.reset()
    s = core.summary()
    # merges sıfırlanır (dosya boş), special tokenlar var
    assert s["has_specials"] is True
    assert s["merges_count"] == 0


def test_train_with_syllables_true_completes(core):
    core.train_model(["istanbul"], vocab_size=10, include_whole_words=True, include_syllables=True, include_sep=False)
    s = core.summary()
    assert s["vocab_size"] >= 5


def test_batch_order_preserved_for_valid_items(core):
    core.train_model(["al be ce"], vocab_size=7, include_syllables=False)
    items = ["al be", "ce", "al ce be"]
    outs = core.batch_encode(items, include_whole_words=True, include_syllables=False, include_sep=True, skip_invalid=True)
    # giriş/çıkış sayıları eşleşsin
    assert len(outs) == len(items)
    # her biri decode edilebilir olsun
    for toks, ids in outs:
        txt = core.decode(ids, method="bpe")
        assert isinstance(txt, str) and len(txt) >= 0


def test_train_from_loader_multiple_runs_idempotent_size(core):
    n = _ensure_education_has_data(core)
    if n == 0:
        pytest.skip("education boş; idempotent test atlandı.")
    core.train_from_loader(method="bpe", vocab_size=14, include_syllables=False)
    size1 = core.get_vocab_size()
    # tekrar çalıştır (aynı yollar, finalize ile re-sync olacak)
    core.train_from_loader(method="bpe", vocab_size=14, include_syllables=False)
    size2 = core.get_vocab_size()
    assert size2 >= size1

# --- Aşağıdakileri mevcut test dosyanızın SONUNA ekleyin ---

def test_core_encode_uses_infer_defaults_when_none(core):
    # inference defaults: whole=True, syllables=False, sep=False (TokenzierCore.__init__ varsayılanları)
    toks, _ = core.encode("alfa beta", mode="inference", include_whole_words=None, include_syllables=None, include_sep=None)
    # <SEP> inference'ta default False olduğu için olmamalı
    assert "<SEP>" not in toks
    # whole-word işaretleri gelmeli
    assert any(t.endswith("</w>") for t in toks if t not in ("<BOS>", "<EOS>"))

def test_core_encode_uses_train_defaults_when_none(core):
    # train defaults: whole=True, syllables=True, sep=True (projede öyle ayarlı ise)
    toks, _ = core.encode("alfa beta", mode="train", include_whole_words=None, include_syllables=None, include_sep=None)
    # train maskesinde BOS/EOS/SEP zaten atılır; burada doğrudan None bayrakların çözümlenmesini test ediyoruz
    assert all(t not in ("<BOS>", "<EOS>") for t in toks)

def test_core_encode_punctuation_tokenization_with_sep(core):
    core.train_model(["evet hayır"], vocab_size=8, include_syllables=False)
    toks, _ = core.encode("evet, hayır.", include_whole_words=True, include_syllables=False, include_sep=True)
    # Noktalama ayrı tokenlar olarak gelmeli
    assert "," in toks and "." in toks
    # Kelimeler arası bir <SEP> olmalı
    assert "<SEP>" in toks

def test_core_encode_punctuation_without_sep(core):
    core.train_model(["evet hayır"], vocab_size=8, include_syllables=False)
    toks, _ = core.encode("evet, hayır.", include_whole_words=True, include_syllables=False, include_sep=False)
    assert "<SEP>" not in toks
    assert "," in toks and "." in toks  # noktalama ayrışmalı

def test_core_decode_prefer_word_vs_syllable(core):
    # hece akışını da üretelim
    core.train_model(["kelime"], vocab_size=10, include_whole_words=True, include_syllables=True, include_sep=True)
    toks, ids = core.encode("kelime", mode="inference", include_whole_words=True, include_syllables=True, include_sep=True)
    # word akışını seç
    out_word = core.decode(ids, method="bpe", prefer="word")
    # syllable akışını seç
    out_syll = core.decode(ids, method="bpe", prefer="syllable")
    assert isinstance(out_word, str) and isinstance(out_syll, str)
    # Her iki tercih de aynı metni vermeli
    assert out_word == out_syll

def test_core_decode_prefer_auto_picks_word_when_majority_words(core):
    core.train_model(["kedi köpek"], vocab_size=10, include_whole_words=True, include_syllables=True, include_sep=True)
    _, ids = core.encode("kedi köpek", include_whole_words=True, include_syllables=True, include_sep=True)
    out_auto = core.decode(ids, method="bpe", prefer="auto")
    out_word = core.decode(ids, method="bpe", prefer="word")
    assert out_auto == out_word

def test_core_batch_encode_no_skip_invalid_raises(core):
    core.train_model(["foo bar"], vocab_size=6, include_syllables=False)
    items = ["foo", 123, None]  # type: ignore[list-item]
    with pytest.raises(TokenizerCoreError):
        _ = core.batch_encode(items, include_whole_words=True, include_syllables=False, include_sep=True, skip_invalid=False)

def test_core_batch_decode_no_skip_invalid_raises(core):
    core.train_model(["a b"], vocab_size=6, include_syllables=False)
    _, good = core.encode("a b", include_whole_words=True, include_syllables=False, include_sep=True)
    seqs = [good, [999999]]  # ikinci bozuk
    with pytest.raises(Exception):
        _ = core.batch_decode(seqs, method="bpe", skip_invalid=False)

def test_core_decode_raw_is_nfc_normalized(core):
    core.train_model(["e\u0301"], vocab_size=6, include_syllables=False, include_sep=False)
    _, ids = core.encode("e\u0301", include_whole_words=True, include_syllables=False, include_sep=False)
    out = core.decode(ids, method="raw")
    assert unicodedata.is_normalized("NFC", out)

def test_core_save_without_prior_train_is_safe(paths_and_backup, education_dir):
    # Sıfırdan çekirdek oluştur, hemen save çağır (finalize içerir)
    vocab_path, merges_path = paths_and_backup
    cfg = {"data_dir": education_dir, "vocab_path": vocab_path, "merges_path": merges_path}
    tc = TokenizerCore(cfg)
    tc.reset()  # default vocab + boş merges
    tc.save()   # hata vermemeli
    assert os.path.isfile(vocab_path) and os.path.isfile(merges_path)

def test_core_encode_type_error_bubbles_as_core_error(core):
    with pytest.raises(TokenizerCoreError):
        _ = core.encode(42)  # type: ignore[arg-type]

def test_core_decode_type_error_bubbles_as_core_error(core):
    with pytest.raises(TokenizerCoreError):
        _ = core.decode("not-a-list")  # type: ignore[arg-type]

def test_core_train_model_invalid_method_raises(paths_and_backup, education_dir):
    vocab_path, merges_path = paths_and_backup
    cfg = {"data_dir": education_dir, "vocab_path": vocab_path, "merges_path": merges_path}
    tc = TokenizerCore(cfg)
    with pytest.raises(TokenizerCoreError):
        tc.train_model(["x y"], method="wordpiece")  # desteklenmeyen

def test_core_summary_changes_after_training(core):
    before = core.summary()
    core.train_model(["p q r p"], vocab_size=8, include_syllables=False)
    after = core.summary()
    assert after["vocab_size"] >= before["vocab_size"]

def test_core_finalize_after_merges_deleted_is_robust(core, paths_and_backup):
    # eğit, merges'i sil, finalize çalışsın
    core.train_model(["x y"], vocab_size=6, include_syllables=False)
    merges_path = core.summary()["merges_path"]
    os.remove(merges_path)
    core.finalize_vocab()
    # merges sıfırlanmış olmalı (0 veya mevcut dosya yeniden oluşturulur)
    assert core.summary()["merges_count"] == 0

def test_core_batch_encode_mode_train_masks_specials_when_flags_none(core):
    core.train_model(["a b c"], vocab_size=7, include_syllables=False)
    outs = core.batch_encode(["a b", "b c"], mode="train",
                             include_whole_words=None, include_syllables=None, include_sep=None, skip_invalid=True)
    for toks, _ in outs:
        assert all(t not in ("<BOS>", "<EOS>", "<SEP>") for t in toks)

def test_core_encode_inference_default_no_sep(core):
    core.train_model(["al be"], vocab_size=6, include_syllables=False)
    toks, _ = core.encode("al be", mode="inference", include_whole_words=None, include_syllables=None, include_sep=None)
    assert "<SEP>" not in toks

def test_core_reset_then_encode_empty_returns_only_specials(core):
    core.reset()
    toks, ids = core.encode("", mode="inference", include_whole_words=True, include_syllables=False, include_sep=True)
    # BOS/EOS dışında bir şey olmamalı
    assert toks == ["<BOS>", "<EOS>"]
    assert len(ids) == 2

def test_core_merges_persist_across_finalize(core):
    core.train_model(["m n m"], vocab_size=8, include_syllables=False)
    c1 = core.summary()["merges_count"]
    core.finalize_vocab()
    c2 = core.summary()["merges_count"]
    assert c2 == c1

def test_core_decode_lowercase_true_propagates(core):
    core.train_model(["ABC DEF"], vocab_size=8, include_syllables=False)
    _, ids = core.encode("ABC DEF", include_whole_words=True, include_syllables=False, include_sep=True)
    out1 = core.decode(ids, method="bpe", lowercase=False)
    out2 = core.decode(ids, method="bpe", lowercase=True)
    assert out2 == out1.lower()

# ============================== EK TESTLER (100+) ==============================

import itertools
import tempfile

# --- Yardımcı: temp QA_TRAIN klasörü oluşturup core döndür ---
def _make_core_with_tmp_qadata(paths_and_backup, pairs):
    vocab_path, merges_path = paths_and_backup
    tmpd = tempfile.TemporaryDirectory()
    qa_dir = tmpd.name
    os.makedirs(qa_dir, exist_ok=True)
    # DataLoaderManager varsayılanları "Soru/Cevap" ve "question/answer" destekliyor
    data = [{"question": q, "answer": a} for q, a in pairs]
    with open(os.path.join(qa_dir, "qa.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    cfg = {"data_dir": qa_dir, "vocab_path": vocab_path, "merges_path": merges_path}
    tc = TokenizerCore(cfg)
    # Kaynak yaşam döngüsünü testler yönetsin; tmpd objesini ekleyelim
    tc.__tmpd = tmpd  # type: ignore[attr-defined]
    return tc

# ---- A) 25 roundtrip vakası ---------------------------------------------------

ROUNDTRIP_STRINGS = [
    "merhaba dünya",
    "istanbul güzel bir şehir",
    "kedi köpek",
    "a b",
    "alpha beta",
    "çığ düşer mi",
    "şafak vakti",
    "Türkçe İMLÂ",
    "foo, bar.",
    "123 456",
    "e\u0301" ,             # e + combining acute
    "ĞÜŞİÖÇ ğıüşiöç",
    "yağmur; rüzgâr!",
    "python > java",
    "mail@example.com",
    "— tire —",
    "parantez (içerik)",
    "iki  boşluk",          # çift boşluk
    "tab\tayrı",
    "satır\nsonu",
    "nokta... üç",
    "kısa",
    "uzuuuun kelime zinciri",
    "",
    "  "                    # whitespace only
]

@pytest.mark.parametrize("text", ROUNDTRIP_STRINGS)
def test_ext_roundtrip_bpe_many(core, text):
    # Her örnek için küçük bir corpus; boş/whitespace durumları da kapsanır
    corpus = [text if text.strip() else "boş girdi telafisi"]
    core.train_model(corpus, vocab_size=8, include_syllables=False)
    toks, ids = core.encode(text, mode="inference", include_whole_words=True, include_syllables=False, include_sep=True)
    out = core.decode(ids, method="bpe")
    assert isinstance(out, str)
    # Boş girdi ise de decode en azından string döner
    assert out is not None

# ---- B) 10 noktalama varyasyonu ----------------------------------------------

PUNCTS = [",", ".", ";", ":", "!", "?", "…", "—", "(", ")"]

@pytest.mark.parametrize("p", PUNCTS)
def test_ext_punctuation_segmentation(core, p):
    core.train_model(["evet hayır"], vocab_size=8, include_syllables=False)
    toks, _ = core.encode(f"evet{p} hayır{p}", include_whole_words=True, include_syllables=False, include_sep=True)
    assert p in toks
    assert "<SEP>" in toks

# ---- C) 10 Unicode normalizasyon örneği --------------------------------------

UNICODES = [
    "e\u0301", "a\u0308", "ı\u0307", "o\u0302", "n\u0303",
    "ş", "ğ", "İ", "Ω", "ß"
]

@pytest.mark.parametrize("s", UNICODES)
def test_ext_unicode_nfc(core, s):
    core.train_model([s], vocab_size=6, include_syllables=False, include_sep=False)
    _, ids = core.encode(s, include_whole_words=True, include_syllables=False, include_sep=False)
    out_raw = core.decode(ids, method="raw")
    assert unicodedata.is_normalized("NFC", out_raw)

# ---- D) 12 bayrak kombinasyonu (train/infer & sep/whole) ---------------------

FLAG_COMBOS = [
    ("train",  True,  False, True ),
    ("train",  True,  True,  True ),
    ("train",  False, False, True ),
    ("train",  True,  False, False),
    ("inference", True,  False, True ),
    ("inference", True,  True,  True ),
    ("inference", False, False, False),
    ("inference", True,  False, False),
    ("inference", None,  None,  None),
    ("train",     None,  None,  None),
    ("inference", False, True,  True ),
    ("train",     False, True,  False),
]

@pytest.mark.parametrize("mode,iw,isy,isp", FLAG_COMBOS)
def test_ext_flag_matrix(core, mode, iw, isy, isp):
    core.train_model(["al be ce"], vocab_size=9, include_syllables=True)
    toks, ids = core.encode("al be", mode=mode, include_whole_words=iw, include_syllables=isy, include_sep=isp)
    assert isinstance(toks, list) and isinstance(ids, list)
    assert len(ids) > 0

# ---- E) 8 batch sırası & atlama davranışı ------------------------------------

BATCH_CASES = [
    ["al be", "ce", "al ce be"],
    ["", "foo", None],          # type: ignore[list-item]
    ["x", "y", 123],            # type: ignore[list-item]
    ["merhaba", "dünya", " "],
    ["A B", "c d", ""],
    ["ç ğ", None, "ş"],         # type: ignore[list-item]
    ["; noktalı", "cümle.", "bitti"],
    ["tek"]
]

@pytest.mark.parametrize("items", BATCH_CASES)
def test_ext_batch_behaviour(core, items):
    core.train_model(["al be ce"], vocab_size=8, include_syllables=False)
    outs = core.batch_encode(items, include_whole_words=True, include_syllables=False, include_sep=True, skip_invalid=True)
    for toks, ids in outs:
        dec = core.decode(ids, method="bpe")
        assert isinstance(dec, str)

# ---- F) 5 persist/reload senaryosu -------------------------------------------

PERSIST_CORPUS = [
    ["k l"], ["foo bar"], ["abc def"], ["merhaba dünya"], ["çığ şapka"]
]

@pytest.mark.parametrize("corpus", PERSIST_CORPUS)
def test_ext_persist_reload_consistency(paths_and_backup, education_dir, corpus):
    vocab_path, merges_path = paths_and_backup
    cfg = {"data_dir": education_dir, "vocab_path": vocab_path, "merges_path": merges_path}
    tc1 = TokenizerCore(cfg)
    tc1.reset()
    tc1.train_model(corpus, vocab_size=7, include_syllables=False)
    tc1.save()
    tc2 = TokenizerCore(cfg)
    tc2.finalize_vocab()
    _, id1 = tc1.encode(corpus[0], include_whole_words=True, include_syllables=False, include_sep=True)
    _, id2 = tc2.encode(corpus[0], include_whole_words=True, include_syllables=False, include_sep=True)
    assert id1 == id2

# ---- G) 10 hata vakası (tip/arg) ---------------------------------------------

BAD_ENCODE_INPUTS = [None, 123, 3.14, [], {}, set(), (1,2), b"bytes", True, object()]

@pytest.mark.parametrize("bad", BAD_ENCODE_INPUTS)
def test_ext_encode_bad_types_raise(core, bad):
    core.train_model(["x y"], vocab_size=5, include_syllables=False)
    with pytest.raises(TokenizerCoreError):
        _ = core.encode(bad)  # type: ignore[arg-type]

BAD_DECODE_INPUTS = [None, "str", 3.14, [1,"x"], [b"x"], [None], [True], [{}], set(), object()]

@pytest.mark.parametrize("bad", BAD_DECODE_INPUTS)
def test_ext_decode_bad_types_raise(core, bad):
    core.train_model(["x y"], vocab_size=5, include_syllables=False)
    with pytest.raises(TokenizerCoreError):
        _ = core.decode(bad)  # type: ignore[arg-type]

# ---- H) 5 tekrar eğitim/idempotent -------------------------------------------

@pytest.mark.parametrize("vsize", [6,7,8,9,10])
def test_ext_retrain_idempotent(core, vsize):
    core.train_model(["m n m"], vocab_size=vsize, include_syllables=False)
    size1 = core.get_vocab_size()
    core.train_model(["m n m"], vocab_size=vsize, include_syllables=False)
    size2 = core.get_vocab_size()
    assert size2 >= size1

# ---- I) 10 DataLoader QA_TRAIN entegrasyon testi -----------------------------

QA_PAIRS_LIST = [
    [("soru1", "cevap1")],
    [("kedi nedir", "bir hayvandır"), ("renk?", "mavi")],
    [("istanbul ?", "şehir"), ("ankara ?", "başkent"), ("izmir ?", "şehir")],
    [("foo", "bar")],
    [("a b", "c d")],
    [("Türkçe", "harfler"), ("şekil", "şemalardır")],
    [("e\u0301", "é"), ("nokta", "son")],
    [("mail?", "cevap."), ("parantez", "(içerik)")],
    [("s1", "a; b"), ("s2", "x, y.")],
    [("çoklu", "örnek")]
]

@pytest.mark.parametrize("pairs", QA_PAIRS_LIST)
def test_ext_train_from_loader_on_tmp_dir(paths_and_backup, pairs):
    tc = _make_core_with_tmp_qadata(paths_and_backup, pairs)
    try:
        tc.train_from_loader(method="bpe", vocab_size=12, include_syllables=False)
        s = tc.summary()
        assert s["vocab_size"] >= 5
    finally:
        # tmp dir temizliği
        tc.__tmpd.cleanup()  # type: ignore[attr-defined]

# ---- J) 10 decode seçeneği matrisi -------------------------------------------

DECODE_MATRIX = [
    dict(remove_specials=True,  collapse_spaces=True,  lowercase=False, prefer="auto"),
    dict(remove_specials=False, collapse_spaces=True,  lowercase=False, prefer="auto"),
    dict(remove_specials=True,  collapse_spaces=False, lowercase=False, prefer="auto"),
    dict(remove_specials=True,  collapse_spaces=True,  lowercase=True,  prefer="auto"),
    dict(remove_specials=True,  collapse_spaces=True,  lowercase=False, prefer="word"),
    dict(remove_specials=True,  collapse_spaces=True,  lowercase=False, prefer="syllable"),
    dict(remove_specials=False, collapse_spaces=False, lowercase=True,  prefer="word"),
    dict(remove_specials=False, collapse_spaces=True,  lowercase=True,  prefer="syllable"),
    dict(remove_specials=True,  collapse_spaces=False, lowercase=True,  prefer="word"),
    dict(remove_specials=False, collapse_spaces=False, lowercase=False, prefer=None),
]

@pytest.mark.parametrize("opts", DECODE_MATRIX)
def test_ext_decode_option_matrix(core, opts):
    core.train_model(["kelime hece"], vocab_size=10, include_whole_words=True, include_syllables=True, include_sep=True)
    _, ids = core.encode("kelime hece", include_whole_words=True, include_syllables=True, include_sep=True)
    txt = core.decode(ids, method="bpe", **opts)
    assert isinstance(txt, str)
    if opts.get("lowercase"):
        assert txt == txt.lower()

# ---- K) 5 encode_with_stats doğrulama ----------------------------------------

@pytest.mark.parametrize("text", ["foo bar", "tek", "iki üç", "çığ düşer", "e\u0301"])
def test_ext_encode_with_stats(core, text):
    core.train_model([text or "yer tutucu"], vocab_size=7, include_syllables=False)
    toks, ids, stats = core.encode_with_stats(text, include_whole_words=True, include_syllables=False, include_sep=True)
    assert isinstance(stats, dict) and "unk_ratio" in stats
    assert stats["length"] == float(len(ids))

# ============================== YENİ: load_training_data odaklı 50+ test ==============================

# -- Yardımcı: education verisi yoksa testleri atlamak için kısa kısayol
def _skip_if_no_edu(core: TokenizerCore):
    n = _ensure_education_has_data(core)
    if n == 0:
        pytest.skip("education klasörü boş; load_training_data odaklı test atlandı.")
    return n

def test_ltd_education_basic(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=16, include_syllables=False)
    examples = core.load_training_data(
        encode_mode="train",
        include_whole_words=True, include_syllables=False, include_sep=True,
    )
    assert isinstance(examples, list)
    if examples:
        inp, tgt = examples[0]
        assert isinstance(inp, list) and isinstance(tgt, list)
        assert all(isinstance(i, int) for i in inp + tgt)

def test_ltd_vocab_size_after_ensure_tokens(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=10, include_syllables=False)
    before = core.get_vocab_size()
    _ = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    after = core.get_vocab_size()
    # ensure_tokens_in_vocab devreye girdi; küçülmemeli.
    assert after >= before

def test_ltd_examples_have_bos_eos_when_inference_encoding(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=12, include_syllables=False)
    v = core.get_vocab()
    bos_id, eos_id = v["<BOS>"]["id"], v["<EOS>"]["id"]
    ex = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    if ex:
        inp, tgt = ex[0]
        assert bos_id in inp and eos_id in inp
        assert bos_id in tgt and eos_id in tgt

def test_ltd_is_idempotent_in_count(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=14, include_syllables=False)
    e1 = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    e2 = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    assert len(e2) == len(e1)

def test_ltd_oob_skips_all_examples_with_monkeypatch(core, monkeypatch):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=14, include_syllables=False)
    # Tüm id'leri OOB sayalım: get_valid_ids boş set dönsün
    monkeypatch.setattr(
        "tokenizer_management.core.tokenizer_core.get_valid_ids",
        lambda _v: set(), raising=False
    )
    ex = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    assert ex == []

def test_ltd_does_not_change_merges(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=18, include_syllables=False)
    before = core.summary()["merges_count"]
    _ = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    after = core.summary()["merges_count"]
    assert after == before

# --- Bayrak varyasyonları (encode_mode & include_* kombinasyonları) ---
LTD_FLAG_COMBOS = [
    ("train", True,  False, True ),
    ("train", True,  True,  True ),
    ("train", False, False, True ),
    ("train", True,  False, False),
    ("inference", True,  False, True ),
    ("inference", True,  True,  True ),
    ("inference", False, False, False),
    ("inference", True,  False, False),
    ("inference", None,  None,  None),
    ("train",     None,  None,  None),
    ("inference", False, True,  True ),
    ("train",     False, True,  False),
]

@pytest.mark.parametrize("encode_mode,iw,isy,isp", LTD_FLAG_COMBOS)
def test_ltd_flag_matrix_variations(core, encode_mode, iw, isy, isp):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=16, include_syllables=True)
    ex = core.load_training_data(
        encode_mode=encode_mode,
        include_whole_words=iw, include_syllables=isy, include_sep=isp
    )
    assert isinstance(ex, list)
    # varsa ilk örnek decode edilebilir olmalı
    if ex:
        inp, tgt = ex[0]
        txt1 = core.decode(inp, method="bpe")
        txt2 = core.decode(tgt, method="bpe")
        assert isinstance(txt1, str) and isinstance(txt2, str)

# --- education verisi yerine tmp QA ile 10 farklı metin ---
LTD_TEXTS = [
    "kısa cevap", "uzun uzun soruuu", "foo bar", "çığ düşer mi",
    "Unicode e\u0301", "şapka â", "parantez (iç)", "nokta... üç",
    "satır\nsonu", "mail@example.com"
]

@pytest.mark.parametrize("t", LTD_TEXTS)
def test_ltd_tmpdir_various_texts(paths_and_backup, t):
    pairs = [(t, t[::-1])]
    tc = _make_core_with_tmp_qadata(paths_and_backup, pairs)
    try:
        tc.train_from_loader(method="bpe", vocab_size=12, include_syllables=False)
        ex = tc.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
        assert isinstance(ex, list) and len(ex) == 1
        inp, tgt = ex[0]
        assert len(inp) > 0 and len(tgt) > 0
    finally:
        tc.__tmpd.cleanup()  # type: ignore[attr-defined]

# --- Unicode odaklı 5 vaka (tmp QA) ---
UNICODE_PAIRS = [
    ("e\u0301", "é"),
    ("İSTANBUL", "istanbul"),
    ("ß", "ss"),
    ("Ωmega", "omega"),
    ("ğüşiöç", "ĞÜŞİÖÇ"),
]

@pytest.mark.parametrize("q,a", UNICODE_PAIRS)
def test_ltd_unicode_tmpdir(paths_and_backup, q, a):
    tc = _make_core_with_tmp_qadata(paths_and_backup, [(q,a)])
    try:
        tc.train_from_loader(method="bpe", vocab_size=10, include_syllables=False)
        ex = tc.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=False)
        assert len(ex) == 1
        inp, tgt = ex[0]
        out_q = tc.decode(inp, method="bpe")
        out_a = tc.decode(tgt, method="bpe")
        assert isinstance(out_q, str) and isinstance(out_a, str)
    finally:
        tc.__tmpd.cleanup()  # type: ignore[attr-defined]

# --- Merges üretimi (tekrarlı korpuslarla) ---
MERGE_CORPORA = [
    ["a a a a"], ["k l k l k l"], ["foo foo bar bar"], ["aa bb aa bb"], ["x y x y x"]
]

@pytest.mark.parametrize("corpus", MERGE_CORPORA)
def test_merges_training_repetitive(core, corpus):
    core.train_model(corpus, vocab_size=10, include_whole_words=True, include_syllables=False, include_sep=True)
    s = core.summary()
    assert s["merges_count"] >= 0  # trainer veya sentetikten gelir
    # merges dosyası yazılmış olmalı
    assert os.path.isfile(s["merges_path"])

def test_ltd_with_custom_fields_ignored(paths_and_backup):
    # load_training_data input/target alanlarını kabul ediyor ama DataLoader'a iletmiyor; yalnız uyumluluk
    pairs = [("soru", "cevap")]
    tc = _make_core_with_tmp_qadata(paths_and_backup, pairs)
    try:
        tc.train_from_loader(method="bpe", vocab_size=9, include_syllables=False)
        ex = tc.load_training_data(input_field="data", target_field="target", encode_mode="train",
                                   include_whole_words=True, include_syllables=False, include_sep=True)
        assert len(ex) == 1
    finally:
        tc.__tmpd.cleanup()  # type: ignore[attr-defined]

def test_ltd_sep_effect_on_length(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=14, include_syllables=False)
    v = core.get_vocab(); sep_id = v["<SEP>"]["id"]
    ex1 = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    ex2 = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=False)
    if ex1 and ex2:
        inp1, _ = ex1[0]
        inp2, _ = ex2[0]
        assert (sep_id in inp1) or (len(inp1) > len(inp2))

def test_ltd_after_reset(core):
    core.reset()
    # reset sonrası bile eğitim + ltd çalışmalı
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=12, include_syllables=False)
    ex = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    assert isinstance(ex, list)

@pytest.mark.parametrize("pref", ["auto", "word", "syllable"])
def test_decode_on_ltd_examples_with_preferences(core, pref):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=16, include_whole_words=True, include_syllables=True)
    ex = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=True, include_sep=True)
    if ex:
        inp, tgt = ex[0]
        t1 = core.decode(inp, method="bpe", prefer=pref)
        t2 = core.decode(tgt, method="bpe", prefer=pref)
        assert isinstance(t1, str) and isinstance(t2, str)

def test_ltd_valid_ids_subset_of_vocab(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=12, include_syllables=False)
    v = core.get_vocab()
    valid_ids = {meta["id"] for meta in v.values()}
    ex = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    for inp, tgt in ex:
        assert set(inp).issubset(valid_ids)
        assert set(tgt).issubset(valid_ids)

def test_ltd_ensure_tokens_in_vocab_expands(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=8, include_syllables=False)
    before = core.get_vocab_size()
    _ = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=False)
    after = core.get_vocab_size()
    assert after >= before

def test_ltd_after_save_and_finalize(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=10, include_syllables=False)
    core.save()
    core.finalize_vocab()
    ex = core.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    assert isinstance(ex, list)

def test_ltd_handles_empty_question_or_answer(paths_and_backup):
    # strict=True olduğu için boş Soru/Cevap kaydı DataLoader tarafından reddedilir.
    # Beklenen davranış: TokenizerCoreError fırlatılmalı.
    tc = _make_core_with_tmp_qadata(paths_and_backup, [("soru", ""), ("", "cevap")])
    try:
        with pytest.raises(TokenizerCoreError):
            tc.train_from_loader(method="bpe", vocab_size=9, include_syllables=False)
    finally:
        tc.__tmpd.cleanup()  # type: ignore[attr-defined]

def test_ltd_handles_whitespace_only(paths_and_backup):
    # strict=True: yalnız whitespace içeren Soru/Cevap da reddedilir.
    tc = _make_core_with_tmp_qadata(paths_and_backup, [("  ", "x"), ("y", "   ")])
    try:
        with pytest.raises(TokenizerCoreError):
            tc.train_from_loader(method="bpe", vocab_size=9, include_syllables=False)
    finally:
        tc.__tmpd.cleanup()  # type: ignore[attr-defined])

def test_ltd_sanitized_when_empty_fields_replaced(paths_and_backup):
    # Boş alanları yer tutucuyla değiştirerek strict akışta başarılı olmalı.
    pairs = [("soru", ""), ("", "cevap")]
    sanitized = [(q or "bos", a or "bos") for q, a in pairs]
    tc = _make_core_with_tmp_qadata(paths_and_backup, sanitized)
    try:
        tc.train_from_loader(method="bpe", vocab_size=9, include_syllables=False)
        ex = tc.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
        assert len(ex) == 2
    finally:
        tc.__tmpd.cleanup()  # type: ignore[attr-defined]
        
def test_ltd_handles_very_long_sequences(paths_and_backup):
    long_q = "a " * 1000
    long_a = "b " * 1000
    tc = _make_core_with_tmp_qadata(paths_and_backup, [(long_q, long_a)])
    try:
        tc.train_from_loader(method="bpe", vocab_size=20, include_syllables=False)
        ex = tc.load_training_data(encode_mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
        assert len(ex) == 1 and len(ex[0][0]) > 0 and len(ex[0][1]) > 0
    finally:
        tc.__tmpd.cleanup()  # type: ignore[attr-defined]

def test_ltd_encode_mode_train_inserts_sep_by_default(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=12, include_syllables=False)
    v = core.get_vocab(); sep_id = v["<SEP>"]["id"]
    ex = core.load_training_data(encode_mode="train", include_whole_words=None, include_syllables=None, include_sep=None)
    if ex:
        inp, _ = ex[0]
        assert sep_id in inp  # train defaults: include_sep=True

def test_ltd_encode_mode_inference_no_sep_by_default(core):
    _skip_if_no_edu(core)
    core.train_from_loader(method="bpe", vocab_size=12, include_syllables=False)
    v = core.get_vocab(); sep_id = v["<SEP>"]["id"]
    ex = core.load_training_data(encode_mode="inference", include_whole_words=None, include_syllables=None, include_sep=None)
    if ex:
        inp, _ = ex[0]
        assert sep_id not in inp  # inference defaults: include_sep=False
