from typing import List
import pytest
import logging
from tokenizer_management.bpe.bpe_manager_utils import DEFAULT_SPECIALS, default_vocab

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

def test_missing_special_tokens_do_not_mutate_vocabulary(caplog):
    vocab = {'x': {'id': 10}}
    dec = BPEDecoder(vocab=vocab, use_gpu=False)
    assert dec.vocab == vocab == {'x': {'id': 10}}
    assert dec.decode([10]) == 'x'
    assert 'eklenmedi' in caplog.text


def test_existing_special_token_ids_are_preserved():
    vocab = {token: {'id': tid} for token, tid in DEFAULT_SPECIALS.items()}
    dec = BPEDecoder(vocab=vocab, use_gpu=False)
    assert dec.vocab == vocab
    assert dec.reverse_vocab == {tid: token for token, tid in DEFAULT_SPECIALS.items()}


def test_reverse_vocab_basic_mapping():
    dec = BPEDecoder(vocab={'x': {'id': 5}}, use_gpu=False)
    assert dec.reverse_vocab == {5: 'x'}


def test_reverse_vocab_id_collision_is_rejected_without_reassigning_ids():
    vocab = {'a': {'id': 0}, 'b': {'id': 0}}
    with pytest.raises(BPEDecodingError, match='çakışması'):
        BPEDecoder(vocab=vocab, use_gpu=False)
    assert vocab['b']['id'] == 0


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
    assert dec.decode([10]) == "c"
    with pytest.raises(BPEDecodingError):
        dec.decode([0])

def test_set_merges_none_clears():
    """set_merges(None) merges listesini temizler."""
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    dec.set_merges(None)
    assert dec.merges == []
    assert dec.decode([0]) == "a"

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
    assert dec.merges == [('x', 'y')]
    assert dec.decode([0]) == 'a'

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
    assert dec.reverse_vocab == {0: "a"}

def test_decode_unknown_id_raises():
    """Unknown IDs must not silently become replacement text."""
    dec = BPEDecoder(vocab={'a': {'id':0,'total_freq':0,'positions':[]}})
    with pytest.raises(BPEDecodingError, match="999"):
        dec.decode([999])

def test_decode_simple_merge():
    """Ardışık merges pair’leri doğru birleşir."""
    vocab = {
        'a': {'id':0,'total_freq':0,'positions':[]},
        'b': {'id':1,'total_freq':0,'positions':[]},
    }
    merges = [('a','b')]
    dec = BPEDecoder(vocab=vocab, merges=merges)
    assert dec.decode([0,1,0]) == "aba"  # No word-end token, no invented space.

def test_decode_preserves_single_dot():
    """Punctuation is content and is retained."""
    vocab = {'x.': {'id':0,'total_freq':0,'positions':[]}}
    dec = BPEDecoder(vocab=vocab)
    assert dec.decode([0]) == "x."

def test_decode_keep_ellipsis():
    """Üç nokta ('...') olduğu sürece korunur."""
    vocab = {'e...': {'id':0,'total_freq':0,'positions':[]}}
    dec = BPEDecoder(vocab=vocab)
    assert dec.decode([0]) == "e..."

def test_decode_filter_special_and_tags():
    """<PAD>,<UNK>,<BOS>,<EOS> ve '__tag__' tokenları filtrelenir."""
    vocab = {
        tok: {'id':i,'total_freq':0,'positions':[]}
        for tok,i in DEFAULT_SPECIALS.items()
    }
    vocab['foo</w>'] = {'id':len(vocab),'total_freq':0,'positions':[]}
    vocab['__tag__1'] = {'id':len(vocab),'total_freq':0,'positions':[]}
    vocab['bar'] = {'id':len(vocab),'total_freq':0,'positions':[]}
    dec = BPEDecoder(vocab=vocab)
    ids = [
        vocab['<PAD>']['id'],
        vocab['foo</w>']['id'],
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
    assert dec.vocab == default_vocab()
    assert dec.merges == []
    bos = dec.vocab['<BOS>']['id']
    eos = dec.vocab['<EOS>']['id']
    assert dec.decode([bos, eos]) == ""
    assert "Resetleniyor" in caplog.text


def test_rejected_vocab_update_preserves_working_decoder():
    dec = BPEDecoder({'a': {'id': 10}}, use_gpu=False)
    with pytest.raises(BPEDecodingError):
        dec.set_vocab({'b': {'id': 20}, 'c': {'id': 20}})
    assert dec.vocab == {'a': {'id': 10}}
    assert dec.decode([10]) == 'a'
