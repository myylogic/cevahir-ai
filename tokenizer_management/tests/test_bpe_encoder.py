import pytest
import logging
from typing import List, Dict, Tuple

from tokenizer_management.bpe.bpe_encoder import BPEEncoder, BPEEncodingError

LOGGER = logging.getLogger("tokenizer_management.bpe.bpe_encoder")


def make_basic_vocab():
    # Sadece bir örnek token + özel tokenlar ekleniyor init sırasında
    return {"foo": {"id": 0, "total_freq": 0, "positions": []}}


def test_init_non_dict_vocab_raises():
    """Vocab dict değilse TypeError fırlatılmalı."""
    with pytest.raises(TypeError):
        BPEEncoder(vocab="not a dict")  # type: ignore


def test_init_empty_vocab_raises():
    """Boş vocab ile init ValueError olmalı."""
    with pytest.raises(ValueError):
        BPEEncoder(vocab={})


def test_init_eksik_special_token_eklenir(caplog):
    """Eksik özel tokenlar init sırasında eklenmeli ve uyarı loglanmalı."""
    caplog.set_level(logging.WARNING)
    vocab = {"foo": {"id": 0, "total_freq": 0, "positions": []}}
    enc = BPEEncoder(vocab=vocab)
    # SPECIAL_TOKENS mutlaka vocab içinde
    for tok in BPEEncoder.SPECIAL_TOKENS:
        assert tok in enc.vocab
    # Uyarı logu:
    assert any("Eksik özel token" in rec.message for rec in caplog.records)


def test_init_tam_special_tokens_no_warning(caplog):
    """Tüm özel tokenlar varsa uyarı değil bilgi logu gelmeli."""
    caplog.set_level(logging.INFO)
    vocab = {
        "<PAD>": {"id": 0, "total_freq": 0, "positions": []},
        "<UNK>": {"id": 1, "total_freq": 0, "positions": []},
        "<BOS>": {"id": 2, "total_freq": 0, "positions": []},
        "<EOS>": {"id": 3, "total_freq": 0, "positions": []},
        "foo": {"id": 4, "total_freq": 0, "positions": []},
    }
    enc = BPEEncoder(vocab=vocab)
    assert not any(rec.levelno == logging.WARNING for rec in caplog.records)
    assert any("eksik yok" in rec.message.lower() for rec in caplog.records)


def test_reverse_vocab_mapping_doğru():
    """reverse_vocab, id→token eşlemesini doğru kurmalı."""
    vocab = {"a": {"id": 5, "total_freq": 0, "positions": []}}
    enc = BPEEncoder(vocab=vocab)
    # Özel tokenlar + 'a'
    rv = enc.reverse_vocab
    assert rv[5] == "a"
    for st in BPEEncoder.SPECIAL_TOKENS:
        assert any(tok == st for tok in rv.values())


def test_set_vocab_invalid_type_raises():
    """set_vocab non-dict tipte TypeError fırlatmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    with pytest.raises(TypeError):
        enc.set_vocab("nope")  # type: ignore


def test_set_vocab_empty_raises():
    """set_vocab boş dict ile ValueError fırlatmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    with pytest.raises(ValueError):
        enc.set_vocab({})


def test_set_vocab_successful_updates(caplog):
    """set_vocab yeni vocab atayıp special tokenları eklemeli."""
    caplog.set_level(logging.WARNING)
    enc = BPEEncoder(vocab=make_basic_vocab())
    new = {"bar": {"id": 10, "total_freq": 0, "positions": []}}
    enc.set_vocab(new)
    # new vocab içinde 'bar' ve özel tokenlar
    assert "bar" in enc.vocab
    for st in BPEEncoder.SPECIAL_TOKENS:
        assert st in enc.vocab
    assert any("Eksik özel token" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("merges", [None, [], [("a","b")]])
def test_set_merges_various(merges):
    """set_merges None, boş ve geçerli tuple listesi ile çalışmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    enc.set_merges(merges)  # hata olmamalı
    expected = merges or []
    assert enc.merges == expected


def test_set_merges_invalid_raises():
    """set_merges hatalı içeriklerde ValueError fırlatmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    with pytest.raises(ValueError):
        enc.set_merges([("a",), ("x","y","z")])  # yanlış tuple uzunluğu


def test_update_reverse_vocab_empty_vocab_raises():
    """_update_reverse_vocab boş vocab ile BPEEncodingError fırlatmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    enc.vocab.clear()
    with pytest.raises(BPEEncodingError):
        enc._update_reverse_vocab()


def test_encode_invalid_mode_raises():
    """encode geçersiz mode ile ValueError fırlatmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    with pytest.raises(ValueError):
        enc.encode(["foo"], mode="badmode")


def test_encode_non_list_raises():
    """encode non-list input ile TypeError fırlatmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    with pytest.raises(TypeError):
        enc.encode("foo")  # type: ignore


def test_encode_empty_list_raises():
    """encode boş liste ile ValueError fırlatmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    with pytest.raises(ValueError):
        enc.encode([], mode="train")


def test_encode_known_token_train_updates_freq_and_positions():
    """train modunda bilinen token frekans/pozisyon güncellenmeli."""
    vocab = {"foo": {"id": 7, "total_freq": 0, "positions": []}}
    enc = BPEEncoder(vocab=vocab)
    ids = enc.encode(["foo", "foo"], mode="train")
    assert ids == [7, 7]
    entry = enc.vocab["foo"]
    assert entry["total_freq"] == 2
    assert entry["positions"] == [0, 1]


def test_encode_unknown_token_train_adds_new_id(caplog):
    """train modunda bilinmeyen token vocab'a eklenmeli."""
    caplog.set_level(logging.INFO)
    enc = BPEEncoder(vocab=make_basic_vocab())
    new_ids = enc.encode(["baz"], mode="train")
    # sadece bir id dönmeli
    assert len(new_ids) == 1
    new_id = new_ids[0]
    # vocab güncellenmiş olmalı
    assert "baz" in enc.vocab
    assert enc.vocab["baz"]["id"] == new_id
    # logda yeni token eklendi mesajı
    assert any("Yeni token eklendi" in rec.message for rec in caplog.records)


def test_encode_unknown_token_inference_uses_UNK():
    """inference modunda bilinmeyenler <UNK> id ile döner, vocab değişmez."""
    base = make_basic_vocab()
    enc = BPEEncoder(vocab=dict(base))  # kopya
    unk_id = enc.vocab["<UNK>"]["id"]
    ids = enc.encode(["nonesuch"], mode="inference")
    assert ids == [unk_id]
    assert "nonesuch" not in enc.vocab


def test_apply_bpe_merge_exact_match():
    """_apply_bpe_merge tam eşleşme durumunda id döndürmeli."""
    vocab = {"abc": {"id": 5, "total_freq": 0, "positions": []}}
    enc = BPEEncoder(vocab=vocab, merges=[("ab","c")])
    assert enc._apply_bpe_merge("abc") == 5


def test_apply_bpe_merge_split():
    """_apply_bpe_merge split-onset durumunda çalışmalı."""
    vocab = {"ac": {"id": 9, "total_freq":0, "positions":[]}}
    enc = BPEEncoder(vocab=vocab, merges=[("a","c")])
    # token 'axc' startswith 'a' endswith 'c', sub 'ac'
    assert enc._apply_bpe_merge("axc") == 9


def test_resolve_token_id_conflicts(caplog):
    """_resolve_token_id_conflicts tekrar eden ID'leri yeni ID ile değiştirmeli."""
    caplog.set_level(logging.WARNING)
    enc = BPEEncoder(vocab={"x": {"id":0,"total_freq":0,"positions":[]},
                            "y": {"id":1,"total_freq":0,"positions":[]},
                            "z": {"id":2,"total_freq":0,"positions":[]}})
    # Basit token_ids listesi
    token_ids = [0,1,0,2,1]
    resolved = enc._resolve_token_id_conflicts(token_ids)
    # boyut değişmemeli, tekrarlı ID yerleri farklı olmalı
    assert len(resolved) == len(token_ids)
    assert resolved[0] == 0
    assert resolved[2] != 0
    assert any("ID çakışması" in rec.message for rec in caplog.records)


def test_reset_restores_only_special_tokens():
    """reset sadece SPECIAL_TOKENS'ı koruyup, vocab/marge sıfırlamalı."""
    vocab = {"foo": {"id":10,"total_freq":0,"positions":[]}}
    enc = BPEEncoder(vocab=vocab, merges=[("a","b")])
    enc.reset()
    # yalnızca özel tokenlar kalmalı
    keys = set(enc.vocab.keys())
    assert keys == set(BPEEncoder.SPECIAL_TOKENS)
    assert enc.merges == []
    # reverse_vocab da sadece onlar olmalı
    assert set(enc.reverse_vocab.values()) == set(BPEEncoder.SPECIAL_TOKENS)


# Ekstra: hatalı token ile _handle_unknown_token
def test_handle_unknown_token_invalid_raises():
    """_handle_unknown_token boş veya non-str ile BPEEncodingError fırlatmalı."""
    enc = BPEEncoder(vocab=make_basic_vocab())
    with pytest.raises(BPEEncodingError):
        enc._handle_unknown_token("", 0)
    with pytest.raises(BPEEncodingError):
        enc._handle_unknown_token(123, 0)  # type: ignore
