from typing import List
import pytest
import logging

from tokenizer_management.bpe.bpe_decoder import (
    BPEDecoder,
    BPEDecodingError,
    DummyPostprocessor,
)

def test_dummy_postprocessor_join():
    """DummyPostprocessor listeyi boşlukla birleştirir."""
    dp = DummyPostprocessor()
    assert dp.process(['a', 'b', 'c']) == 'a b c'

def test_init_non_dict_raises():
    """vocab dict değilse TypeError fırlatır."""
    with pytest.raises(TypeError):
        BPEDecoder(vocab='not a dict')

def test_init_empty_vocab_raises():
    """vocab boş dict ise ValueError fırlatır."""
    with pytest.raises(ValueError):
        BPEDecoder(vocab={})

def test_ensure_special_tokens_added(caplog):
    """Eksik özel tokenlar eklenip warning loglanır."""
    caplog.set_level(logging.WARNING)
    vocab = {'x': {'id': 0, 'total_freq': 0, 'positions': []}}
    dec = BPEDecoder(vocab=vocab)
    for tok in BPEDecoder.SPECIAL_TOKENS:
        assert tok in dec.vocab
    assert "Eksik özel tokenlar eklendi" in caplog.text

def test_ensure_special_tokens_not_added(caplog):
    """Zaten varsa özel tokenlar eklenmez, bilgi loglanır."""
    caplog.set_level(logging.INFO)
    vocab = {
        tok: {'id': i, 'total_freq': 0, 'positions': []}
        for i, tok in enumerate(BPEDecoder.SPECIAL_TOKENS)
    }
    dec = BPEDecoder(vocab=vocab)
    assert "Özel tokenlar kontrol edildi" in caplog.text

def test_reverse_vocab_basic_mapping():
    """reverse_vocab doğru oluşturulur."""
    vocab = {'x': {'id': 5, 'total_freq': 0, 'positions': []}}
    dec = BPEDecoder(vocab=vocab)
    rv = dec.reverse_vocab
    assert rv[5] == 'x'
    for tok in BPEDecoder.SPECIAL_TOKENS:
        tid = dec.vocab[tok]['id']
        assert rv[tid] == tok

def test_reverse_vocab_id_collision(caplog):
    """ID çakışması olursa uyarı ve yeni ID atanır."""
    caplog.set_level(logging.WARNING)
    vocab = {
        'a': {'id': 0, 'total_freq': 0, 'positions': []},
        'b': {'id': 0, 'total_freq': 0, 'positions': []},
    }
    dec = BPEDecoder(vocab=vocab)
    rv = dec.reverse_vocab
    assert rv[0] == 'a'
    b_ids = [tid for tid, tok in rv.items() if tok == 'b']
    assert len(b_ids) == 1 and b_ids[0] != 0
    assert "Decoder ID çakışması" in caplog.text

def test_reverse_vocab_non_int_id_ignored():
    """ID’si int olmayan entry’ler ters sözlükte atlanır."""
    vocab = {
        'a': {'id': 'x', 'total_freq': 0, 'positions': []},
        'b': {'id': 1,   'total_freq': 0, 'positions': []},
    }
    dec = BPEDecoder(vocab=vocab)
    rv = dec.reverse_vocab
    assert 1 in rv and rv[1] == 'b'
    assert 'a' not in rv.values()

def test_set_vocab_invalid_type():
    """set_vocab dict değilse ValueError fırlatır."""
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    with pytest.raises(ValueError):
        dec.set_vocab({})
    with pytest.raises(ValueError):
        dec.set_vocab(None)

def test_set_vocab_successful_update(caplog):
    """set_vocab geçerli dict ile çalışır ve loglanır."""
    caplog.set_level(logging.INFO)
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    new_vocab = {'c': {'id':10,'total_freq':0,'positions':[]}}
    dec.set_vocab(new_vocab)
    assert 10 in dec.reverse_vocab
    assert "BPEDecoder vocab güncellendi" in caplog.text

def test_set_merges_none_clears():
    """set_merges(None) merges listesini temizler."""
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    dec.set_merges(None)
    assert dec.merges == []
    assert dec._merges_set == set()

def test_set_merges_invalid():
    """set_merges hatalı formatta ValueError fırlatır."""
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    with pytest.raises(ValueError):
        dec.set_merges([('a',), ('b','c','d')])

def test_set_merges_valid(caplog):
    """set_merges doğru listeyi kabul eder ve loglar."""
    caplog.set_level(logging.INFO)
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    dec.set_merges([('x','y')])
    assert ('x','y') in dec._merges_set
    assert "BPEDecoder merges listesi güncellendi" in caplog.text

def test_decode_invalid_input_non_list():
    """decode parametresi liste değilse BPEDecodingError fırlatır."""
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    with pytest.raises(BPEDecodingError):
        dec.decode('not a list')

def test_decode_invalid_input_non_int():
    """decode listesinde int olmayan varsa BPEDecodingError fırlatır."""
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    with pytest.raises(BPEDecodingError):
        dec.decode([1, 'x'])

def test_decode_empty_list(caplog):
    """decode([]) boş string döner ve warning loglar."""
    caplog.set_level(logging.WARNING)
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    assert dec.decode([]) == ""
    assert "Boş ID listesi decode" in caplog.text

def test_decode_unknown_id_to_empty():
    """Tüm ID’ler unknown ise '<EMPTY>' döner."""
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    assert dec.decode([999]) == "<EMPTY>"

def test_decode_simple_merge():
    """Ardışık merges pair’leri doğru birleşir."""
    vocab = {
        'a': {'id':0,'total_freq':0,'positions':[]},
        'b': {'id':1,'total_freq':0,'positions':[]},
    }
    merges = [('a','b')]
    dec = BPEDecoder(vocab=vocab, merges=merges)
    assert dec.decode([0,1,0]) == "ab a"

def test_decode_strip_single_dot():
    """Tek nokta sonda temizlenir."""
    vocab = {'x.': {'id':0,'total_freq':0,'positions':[]}}
    dec = BPEDecoder(vocab=vocab)
    assert dec.decode([0]) == "x"

def test_decode_keep_ellipsis():
    """Üç nokta ('...') olduğu sürece korunur."""
    vocab = {'e...': {'id':0,'total_freq':0,'positions':[]}}
    dec = BPEDecoder(vocab=vocab)
    assert dec.decode([0]) == "e..."

def test_decode_filter_special_and_tags():
    """<PAD>,<UNK>,<BOS>,<EOS> ve '__tag__' tokenları filtrelenir."""
    vocab = {
        tok: {'id':i,'total_freq':0,'positions':[]}
        for i,tok in enumerate(BPEDecoder.SPECIAL_TOKENS)
    }
    vocab['foo'] = {'id':len(vocab),'total_freq':0,'positions':[]}
    vocab['__tag__1'] = {'id':len(vocab),'total_freq':0,'positions':[]}
    vocab['bar'] = {'id':len(vocab),'total_freq':0,'positions':[]}
    dec = BPEDecoder(vocab=vocab)
    ids = [
        vocab['<PAD>']['id'],
        vocab['foo']['id'],
        vocab['__tag__1']['id'],
        vocab['bar']['id'],
        vocab['<EOS>']['id'],
    ]
    assert dec.decode(ids) == "foo bar"

def test_decode_custom_postprocessor():
    """Kullanıcı tanımlı postprocessor çalışır."""
    class Custom:
        def process(self, toks: List[str]) -> str:
            return "".join(toks).upper()
    vocab = {
        'h': {'id':0,'total_freq':0,'positions':[]},
        'i': {'id':1,'total_freq':0,'positions':[]},
    }
    dec = BPEDecoder(vocab=vocab, postprocessor=Custom())
    assert dec.decode([0,1]) == "HI"

def test_reset_restores_defaults(caplog):
    """reset() özel tokenlar + merges olmadan sıfırlar."""
    caplog.set_level(logging.WARNING)
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    dec.set_merges([('a','a')])
    dec.reset()
    assert set(dec.vocab.keys()) == set(BPEDecoder.SPECIAL_TOKENS)
    assert dec.merges == []
    bos = dec.vocab['<BOS>']['id']
    eos = dec.vocab['<EOS>']['id']
    assert dec.decode([bos, eos]) == "<EMPTY>"
    assert "BPEDecoder resetleniyor" in caplog.text
