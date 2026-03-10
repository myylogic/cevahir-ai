# tests/test_bpe_manager.py
from __future__ import annotations

import json
import os
import unicodedata
import pytest

from tokenizer_management.bpe.bpe_manager import (
    BPEManager,
    BPETokenError,
)
from tokenizer_management.bpe.bpe_manager_utils import (
    default_vocab,
    get_valid_ids,
)

# ----------------------------- yardımcılar -----------------------------

def make_manager(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    # Başlangıçta dosyalar yok; manager kendi oluşturur
    m = BPEManager(vocab_file=str(vocab_path), merges_file=str(merges_path))
    return m, vocab_path, merges_path


# ----------------------------- temel/init ------------------------------

def test_init_creates_files_and_components(tmp_path):
    m, vocab_path, merges_path = make_manager(tmp_path)
    assert vocab_path.exists()
    assert merges_path.exists()
    v = m.get_vocab()
    # default specials mevcut ve benzersiz id
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
    ids = [v[s]["id"] for s in specials]
    assert all(s in v for s in specials)
    assert len(set(ids)) == len(ids)


# ---------------------------- encode/decode ----------------------------

def test_encode_inference_and_train_alignment(tmp_path):
    m, _, _ = make_manager(tmp_path)

    # inference (hece kapalı tutalım; deterministik)
    toks, ids = m.encode("merhaba dunya", mode="inference",
                         include_whole_words=True, include_syllables=False, include_sep=True)
    assert toks[0] == "<BOS>" and toks[-1] == "<EOS>"
    # <SEP> araya girmeli
    assert "<SEP>" in toks

    # train hizası: BOS/EOS/SEP çıkar ve ids aynı maske ile filtrelenir
    t2, i2 = m.encode("merhaba dunya", mode="train",
                      include_whole_words=True, include_syllables=False, include_sep=True)
    assert "<BOS>" not in t2 and "<EOS>" not in t2 and "<SEP>" not in t2
    assert len(t2) == len(i2) > 0
    valid_ids = get_valid_ids(m.get_vocab())
    assert all(i in valid_ids for i in i2)


def test_encode_membership_check_no_oob(tmp_path):
    m, _, _ = make_manager(tmp_path)
    # unknown kelime <UNK> id'sine gider, ama üyelik kontrolü geçmeli
    toks, ids = m.encode("bilinmeyen", mode="inference",
                         include_whole_words=True, include_syllables=False, include_sep=False)
    valid_ids = get_valid_ids(m.get_vocab())
    assert all(i in valid_ids for i in ids)


def test_decode_bpe_and_raw_and_error_on_unknown_id(tmp_path):
    m, _, _ = make_manager(tmp_path)
    _, ids = m.encode("merhaba dunya", mode="inference",
                      include_whole_words=True, include_syllables=False, include_sep=True)

    # bpe method
    text = m.decode(ids, method="bpe")
    assert isinstance(text, str) and len(text) > 0

    # raw method
    text_raw = m.decode(ids, method="raw")
    assert isinstance(text_raw, str) and len(text_raw) > 0

    # bilinmeyen id ile bpe decode hata vermeli
    bad_id = max(get_valid_ids(m.get_vocab())) + 9999
    with pytest.raises(Exception):
        m.decode([bad_id], method="bpe")


# ------------------------------- training ------------------------------

def test_train_produces_merges_and_persists(tmp_path):
    m, vocab_path, merges_path = make_manager(tmp_path)

    corpus = ["merhaba dunya", "merhaba evren"]
    m.train(
        corpus,
        target_merges=10,
        max_iter=50,
        min_frequency=1,
        include_whole_words=True,
        include_syllables=False,   # heceyi kapat → sade istatistik
        include_sep=True,
        append_eos=True,
        protect_specials=True,
    )

    # Merges diske yazılmış olmalı ve okunabilmeli
    assert merges_path.exists()
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) > 0
    # manager üstünden de erişelim
    merges = m.get_merges()
    assert len(merges) == len(lines)

    # Vocab kaydı güncellenmiş olmalı (trainer kaynaklı)
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict) and len(data) == len(m.get_vocab())


def test_finalize_vocab_reload_and_sync(tmp_path):
    m, _, _ = make_manager(tmp_path)
    corpus = ["a b c a b", "a b d"]
    m.train(corpus, target_merges=5, include_whole_words=True, include_syllables=False)

    # finalize_vocab yeniden yükler ve sync eder
    m.finalize_vocab()
    # encode/decode hala çalışmalı
    _, ids = m.encode("a b c", mode="inference", include_whole_words=True, include_syllables=False)
    txt = m.decode(ids)
    assert isinstance(txt, str) and len(txt) > 0


# --------------------------- get/set/reset/sync -------------------------

def test_set_get_vocab_roundtrip(tmp_path):
    m, vocab_path, _ = make_manager(tmp_path)
    v = m.get_vocab()
    # küçük bir ek: sahte token
    new_token = "deneme</w>"
    assert new_token not in v
    v2 = dict(v)
    # id olarak benzersiz bir sayı kullan
    next_id = max(x["id"] for x in v2.values()) + 1
    v2[new_token] = {"id": next_id, "total_freq": 0, "positions": []}

    m.set_vocab(v2)
    assert m.get_vocab()[new_token]["id"] == next_id
    # dosyada da olmalı
    with open(vocab_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    assert new_token in j


def test_auto_update_vocab_monotonic_ids(tmp_path):
    m, _, _ = make_manager(tmp_path)
    base_max = max(x["id"] for x in m.get_vocab().values())
    added = m.auto_update_vocab(["tok1", "tok2", "tok2", " ", "", "tok3"])
    assert added == 3
    new_max = max(x["id"] for x in m.get_vocab().values())
    assert new_max >= base_max + 3  # monotonik artış

    # encode sonrası id’ler sözlükte olmalı
    _, ids = m.encode("tok1 tok3", mode="inference", include_whole_words=True, include_syllables=False)
    valid_ids = get_valid_ids(m.get_vocab())
    assert all(i in valid_ids for i in ids)


def test_reset_clears_merges_and_restores_default_vocab(tmp_path):
    m, _, merges_path = make_manager(tmp_path)
    m.train(["x y x"], target_merges=3, include_whole_words=True, include_syllables=False)
    assert len(m.get_merges()) > 0

    m.reset()
    assert len(m.get_merges()) == 0
    # merges dosyası boş ya da yok
    with open(merges_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    assert content == ""
    # vocab default
    dv = default_vocab()
    for k in dv:
        assert k in m.get_vocab()


def test_load_vocab_and_merges_sync(tmp_path):
    m, vocab_path, merges_path = make_manager(tmp_path)
    # eğitim yapıp dosyaları üretelim
    m.train(["merhaba dunya"], target_merges=5, include_whole_words=True, include_syllables=False)
    # yeni manager açmadan, mevcut nesnede dosyadan yükle/senkronize
    m.load_vocab_and_merges()
    # encode/decode çalışmalı
    _, ids = m.encode("merhaba", mode="inference", include_whole_words=True, include_syllables=False)
    assert isinstance(m.decode(ids), str)


# ------------------------------- hata yolları ---------------------------

def test_encode_invalid_inputs_raise(tmp_path):
    m, _, _ = make_manager(tmp_path)
    with pytest.raises(TypeError):
        m.encode(123, mode="inference")  # type: ignore[arg-type]


def test_decode_invalid_inputs_raise(tmp_path):
    m, _, _ = make_manager(tmp_path)
    with pytest.raises(ValueError):
        m.decode([], method="bpe")
    with pytest.raises(Exception):
        m.decode([1, 2, 3], method="invalid")  # method kontrolü


def test_training_invalid_corpus_raises(tmp_path):
    m, _, _ = make_manager(tmp_path)
    with pytest.raises(ValueError):
        m.train([], target_merges=1)
    with pytest.raises(TypeError):
        m.train([123], target_merges=1)  # type: ignore[list-item]


# ---------------------------- seçenek/parametre -------------------------

def test_encode_option_flags_affect_tokens(tmp_path):
    m, _, _ = make_manager(tmp_path)
    # sadece whole word
    toks1, _ = m.encode("iki kelime", include_whole_words=True, include_syllables=False, include_sep=True)
    assert any(t.endswith("</w>") for t in toks1)
    assert "<SEP>" in toks1

    # sep kapalı
    toks2, _ = m.encode("iki kelime", include_whole_words=True, include_syllables=False, include_sep=False)
    assert "<SEP>" not in toks2

    # hece açık (davranış implementasyona bağlı; sadece boş olmamasını doğrulayalım)
    toks3, _ = m.encode("iki", include_whole_words=False, include_syllables=True, include_sep=False)
    assert len(toks3) >= 2  # <BOS> ... <EOS>


def test_decode_normalization_flags(tmp_path):
    m, _, _ = make_manager(tmp_path)
    _, ids = m.encode("Merhaba DUNYA", include_whole_words=True, include_syllables=False)
    txt1 = m.decode(ids, lowercase=False)
    txt2 = m.decode(ids, lowercase=True)  # küçük harfe zorlama
    assert isinstance(txt1, str) and isinstance(txt2, str)


# ----------------------------- ek pratik testler -----------------------------

def test_end_to_end_train_then_infer_no_unk(tmp_path):
    m, _, _ = make_manager(tmp_path)
    corpus = ["kedi köpek", "köpek kedi"]
    m.train(
        corpus,
        target_merges=6,
        include_whole_words=True,
        include_syllables=False,
        include_sep=True,
    )
    _, ids = m.encode(
        "kedi köpek",
        mode="inference",
        include_whole_words=True,
        include_syllables=False,
        include_sep=True,
    )
    unk_id = m.get_vocab()["<UNK>"]["id"]
    assert unk_id not in ids


def test_persistence_consistency_same_paths_ids_equal(tmp_path):
    m1, vocab_path, merges_path = make_manager(tmp_path)
    m1.train(
        ["alp beta"],
        target_merges=4,
        include_whole_words=True,
        include_syllables=False,
        include_sep=True,
    )
    sample = "alp beta"
    _, ids1 = m1.encode(sample, include_whole_words=True, include_syllables=False, include_sep=True)

    # Aynı path'lerle ikinci örnek (aynı süreçte singleton da olmalı)
    m2 = BPEManager(vocab_file=str(vocab_path), merges_file=str(merges_path))
    _, ids2 = m2.encode(sample, include_whole_words=True, include_syllables=False, include_sep=True)
    assert ids1 == ids2


def test_raw_decode_inserts_space_and_keeps_markers(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(
        ["foo bar"],
        target_merges=3,
        include_whole_words=True,
        include_syllables=False,
        include_sep=True,
    )
    _, ids = m.encode("foo bar", include_whole_words=True, include_syllables=False, include_sep=True)
    txt_raw = m.decode(ids, method="raw")
    # raw: </w> işaretleri kalır ve <SEP> yerine boşluk eklenir
    assert "</w>" in txt_raw and " " in txt_raw and len(txt_raw.strip()) > 0


def test_raw_decode_with_unknown_id_returns_empty_string(tmp_path):
    m, _, _ = make_manager(tmp_path)
    # Bilinmeyen id raw için hata vermez; <UNK> filtrelenir → boş string
    out = m.decode([999999], method="raw")
    assert out == ""


def test_train_accepts_token_list_input(tmp_path):
    m, _, _ = make_manager(tmp_path)
    toks = ["ahmet</w>", "<SEP>", "mehmet</w>"]
    m.train(
        [toks],
        target_merges=2,
        include_whole_words=True,
        include_syllables=False,
        include_sep=True,
    )
    _, ids = m.encode("ahmet mehmet", include_whole_words=True, include_syllables=False, include_sep=True)
    out = m.decode(ids, method="bpe")
    assert isinstance(out, str) and len(out) > 0


def test_finalize_keeps_interesting_ids_stable(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(
        ["a b a"],
        target_merges=4,
        include_whole_words=True,
        include_syllables=False,
        include_sep=True,
    )
    interest = [ "a</w>", "b</w>", "<SEP>" ]
    before = {k: m.get_vocab()[k]["id"] for k in interest if k in m.get_vocab()}
    m.finalize_vocab()
    after = {k: m.get_vocab()[k]["id"] for k in interest if k in m.get_vocab()}
    assert before == after


def test_singleton_same_paths_instance_identity(tmp_path):
    m1, vocab_path, merges_path = make_manager(tmp_path)
    m2 = BPEManager(vocab_file=str(vocab_path), merges_file=str(merges_path))
    assert m1 is m2


def test_unicode_normalization_nfc_roundtrip(tmp_path):
    m, _, _ = make_manager(tmp_path)
    word = "e\u0301"  # 'e' + combining acute
    m.train(
        [word],
        target_merges=2,
        include_whole_words=True,
        include_syllables=False,
        include_sep=False,
    )
    _, ids = m.encode(word, include_whole_words=True, include_syllables=False, include_sep=False)
    out = m.decode(ids, method="bpe", lowercase=False)
    assert out == unicodedata.normalize("NFC", word)


def test_syllables_flag_changes_tokenization_but_not_decoded_text(tmp_path):
    m, _, _ = make_manager(tmp_path)
    text = "kelime"
    # önce sözlüğe ekle ki UNK olmasın
    m.train([text], target_merges=2, include_whole_words=True, include_syllables=False, include_sep=False)

    toks_syll, ids_syll = m.encode(text, include_whole_words=False, include_syllables=True, include_sep=False)
    toks_word, ids_word = m.encode(text, include_whole_words=True, include_syllables=False, include_sep=False)

    # heceleme daha çok token üretmeli (BOS/EOS hariç kıyaslayalım)
    core = lambda t: [x for x in t if x not in ("<BOS>", "<EOS>", "<SEP>")]
    assert len(core(toks_syll)) >= len(core(toks_word))

    out1 = m.decode(ids_syll, method="bpe")
    out2 = m.decode(ids_word, method="bpe")
    assert out1 == out2


def test_save_merges_empty_writes_empty_file(tmp_path):
    m, _, merges_path = make_manager(tmp_path)
    m.train(["x y"], target_merges=2, include_whole_words=True, include_syllables=False, include_sep=True)
    # önce dolu olsun
    assert len(m.get_merges()) > 0

    m.save_merges([])  # dosyayı boşalt
    with open(merges_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.strip() == ""


def test_merges_roundtrip_manual_save_and_reload(tmp_path):
    m, _, _ = make_manager(tmp_path)
    merges = [("x</w>", "y</w>"), ("y</w>", "z</w>")]
    m.save_merges(merges)
    m.finalize_vocab()  # dosyadan tekrar yükle
    assert m.get_merges() == merges


def test_tokenize_pipeline_returns_syllables(tmp_path):
    m, _, _ = make_manager(tmp_path)
    sylls = m.tokenize("merhaba dünya")
    assert isinstance(sylls, list) and len(sylls) > 0 and all(isinstance(s, str) for s in sylls)


def test_encode_with_sep_toggle_changes_raw_output(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["alpha beta"], target_merges=3, include_whole_words=True, include_syllables=False, include_sep=True)

    _, ids_with = m.encode("alpha beta", include_whole_words=True, include_syllables=False, include_sep=True)
    _, ids_without = m.encode("alpha beta", include_whole_words=True, include_syllables=False, include_sep=False)

    raw_with = m.decode(ids_with, method="raw")
    raw_without = m.decode(ids_without, method="raw")

    assert " " in raw_with  # <SEP> -> boşluk
    assert " " not in raw_without  # <SEP> yok


def test_decode_lowercase_true_makes_lowercase(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["ABC DEF"], target_merges=3, include_whole_words=True, include_syllables=False, include_sep=True)
    _, ids = m.encode("ABC DEF", include_whole_words=True, include_syllables=False, include_sep=True)

    out_keep = m.decode(ids, method="bpe", lowercase=False)
    out_lower = m.decode(ids, method="bpe", lowercase=True)
    assert out_lower == out_keep.lower()


def test_reset_restores_default_special_ids(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["p q"], target_merges=2, include_whole_words=True, include_syllables=False, include_sep=True)
    assert len(m.get_merges()) > 0

    m.reset()
    dv = default_vocab()
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
    for s in specials:
        assert s in m.get_vocab()
        assert m.get_vocab()[s]["id"] == dv[s]["id"]


# ----------------------------- generative / ek testler -----------------------------

def test_encode_empty_string_returns_only_specials(tmp_path):
    m, _, _ = make_manager(tmp_path)
    toks, ids = m.encode("", include_whole_words=True, include_syllables=False, include_sep=True)
    assert toks == ["<BOS>", "<EOS>"]
    assert ids == [m.get_vocab()["<BOS>"]["id"], m.get_vocab()["<EOS>"]["id"]]


def test_sep_count_equals_word_gaps(tmp_path):
    m, _, _ = make_manager(tmp_path)
    text = "a b c d"
    toks, _ = m.encode(text, include_whole_words=True, include_syllables=False, include_sep=True)
    assert toks.count("<SEP>") == len(text.split()) - 1


def test_raw_decode_sep_only_space(tmp_path):
    m, _, _ = make_manager(tmp_path)
    sep_id = m.get_vocab()["<SEP>"]["id"]
    out = m.decode([sep_id], method="raw")
    assert out == " "


def test_bpe_decode_unknown_returns_empty_tag(tmp_path):
    m, _, _ = make_manager(tmp_path)
    _, ids = m.encode("bilinmeyen", include_whole_words=True, include_syllables=False, include_sep=False)
    out = m.decode(ids, method="bpe")
    assert out == "<empty>"


def test_instance_isolation_with_different_paths(tmp_path):
    p1 = tmp_path / "p1"; p1.mkdir()
    p2 = tmp_path / "p2"; p2.mkdir()

    m1, _, _ = make_manager(p1)
    m2, _, _ = make_manager(p2)
    assert m1 is not m2

    m1.train(["x y"], target_merges=3, include_whole_words=True, include_syllables=False, include_sep=True)
    assert len(m1.get_merges()) > 0
    assert len(m2.get_merges()) == 0  # diğeri etkilenmemeli


def test_load_vocab_and_merges_after_merges_file_deleted(tmp_path):
    m, _, merges_path = make_manager(tmp_path)
    m.train(["a b"], target_merges=3, include_whole_words=True, include_syllables=False, include_sep=True)
    assert merges_path.exists() and len(m.get_merges()) > 0

    os.remove(merges_path)
    m.load_vocab_and_merges()
    assert len(m.get_merges()) == 0


def test_set_vocab_accepts_int_values_normalized(tmp_path):
    m, _, _ = make_manager(tmp_path)
    dv = default_vocab()
    # specials'ları olduğu gibi, yanı sıra int id ile yeni bir token
    v = {k: dv[k]["id"] for k in dv}
    v["custom</w>"] = 999
    m.set_vocab(v)
    assert m.get_vocab()["custom</w>"]["id"] == 999
    # specials id'leri bozulmamış olmalı
    for s in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]:
        assert m.get_vocab()[s]["id"] == dv[s]["id"]


def test_finalize_does_not_change_merges_or_vocab_sizes(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["k l k"], target_merges=4, include_whole_words=True, include_syllables=False, include_sep=True)
    before = (len(m.get_vocab()), len(m.get_merges()))
    m.finalize_vocab()
    after = (len(m.get_vocab()), len(m.get_merges()))
    assert before == after


def test_finalize_preserves_encoding_ids(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["m n"], target_merges=3, include_whole_words=True, include_syllables=False, include_sep=True)
    sample = "m n"
    _, ids1 = m.encode(sample, include_whole_words=True, include_syllables=False, include_sep=True)
    m.finalize_vocab()
    _, ids2 = m.encode(sample, include_whole_words=True, include_syllables=False, include_sep=True)
    assert ids1 == ids2


def test_get_merges_type_and_content_tuples(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["x y z"], target_merges=3, include_whole_words=True, include_syllables=False, include_sep=True)
    merges = m.get_merges()
    assert isinstance(merges, list) and len(merges) > 0
    a, b = merges[0]
    assert isinstance(a, str) and isinstance(b, str)


def test_train_with_include_sep_false_still_writes_merges(tmp_path):
    m, _, merges_path = make_manager(tmp_path)
    m.train(["a b"], target_merges=3, include_whole_words=True, include_syllables=False, include_sep=False)
    assert merges_path.exists()
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) > 0


def test_train_with_negative_target_merges_uses_default(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["p q p"], target_merges=-5, include_whole_words=True, include_syllables=False, include_sep=True)
    assert len(m.get_merges()) >= 1


def test_auto_update_vocab_ignores_specials(tmp_path):
    m, _, _ = make_manager(tmp_path)
    added = m.auto_update_vocab(["<BOS>", "<EOS>", "<PAD>", "<UNK>", "<SEP>"])
    assert added == 0


def test_encode_with_both_whole_and_syllables_produces_both_kinds(tmp_path):
    m, _, _ = make_manager(tmp_path)
    toks, _ = m.encode("iki", include_whole_words=True, include_syllables=True, include_sep=False)
    core = [t for t in toks if t not in ("<BOS>", "<EOS>", "<SEP>")]
    assert any(t.endswith("</w>") for t in core)
    assert any((not t.endswith("</w>")) for t in core)


def test_preprocess_returns_syllables_nonempty(tmp_path):
    m, _, _ = make_manager(tmp_path)
    sylls = m._preprocess("istanbul", mode="diag")
    assert isinstance(sylls, list) and len(sylls) > 0 and all(isinstance(s, str) and s.strip() for s in sylls)


def test_save_vocab_then_reload_preserves_ids(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.auto_update_vocab(["zeta</w>"])
    before = m.get_vocab()["zeta</w>"]["id"]
    m.save_vocab()
    m.load_vocab_and_merges()
    after = m.get_vocab()["zeta</w>"]["id"]
    assert before == after


def test_train_does_not_fail_on_duplicate_lines(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["a a a", "a a a"], target_merges=4, include_whole_words=True, include_syllables=False, include_sep=True)
    assert len(m.get_merges()) >= 1


def test_synthetic_merges_include_sep_when_sep_true(tmp_path):
    m, _, _ = make_manager(tmp_path)
    m.train(["x y x"], target_merges=4, include_whole_words=True, include_syllables=False, include_sep=True)
    merges = m.get_merges()
    assert any(pair[0] == "<SEP>" or pair[1] == "<SEP>" for pair in merges)


def test_decode_raw_ignores_specials_and_unk(tmp_path):
    m, _, _ = make_manager(tmp_path)
    bos = m.get_vocab()["<BOS>"]["id"]
    eos = m.get_vocab()["<EOS>"]["id"]
    unk = m.get_vocab()["<UNK>"]["id"]
    out = m.decode([bos, unk, eos], method="raw")
    assert out == ""


def test_encode_train_mode_masks_counts_in_multiword(tmp_path):
    m, _, _ = make_manager(tmp_path)
    toks, _ = m.encode("a b c d", mode="train", include_whole_words=True, include_syllables=False, include_sep=True)
    assert "<BOS>" not in toks and "<EOS>" not in toks and "<SEP>" not in toks
    assert len(toks) == 4  # sadece whole-word tokenları
