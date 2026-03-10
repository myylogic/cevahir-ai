# tokenizer_management/tests/test_bpe_tokenizer.py
import logging
import pytest
from typing import List, Dict, Any

from tokenizer_management.bpe.bpe_tokenizer import BPETokenizer, BPETokenizerError

# --- Yardımcı Dummy Sınıflar ---
class DummyEncoder:
    def __init__(self):
        self.vocab_set = None
        self.encode_calls = []

    def set_vocab(self, vocab: Dict[str, dict]):
        self.vocab_set = vocab

    def encode(self, text_or_tokens, mode="inference"):
        # encode metodu hem metin hem token listesi alabilir
        self.encode_calls.append((text_or_tokens, mode))
        return [10, 20, 30]

class DummyDecoder:
    def __init__(self):
        self.vocab_set = None
        self.decode_calls = []

    def set_vocab(self, vocab: Dict[str, dict]):
        self.vocab_set = vocab

    def decode(self, token_ids: List[int]) -> str:
        self.decode_calls.append(token_ids)
        return "  sonuc  "  # baş/son boşluklu

class ErrorEncoder:
    def encode(self, *args, **kwargs):
        raise RuntimeError("encode hatası")

class ErrorDecoder:
    def decode(self, *args, **kwargs):
        raise RuntimeError("decode hatası")

# --- Başlangıç Testleri ---
def test_init_hedef_vocab_boşsa_kendi_vocabunu_tutar(caplog):
    enc = DummyEncoder()
    dec = DummyDecoder()
    caplog.set_level(logging.INFO)
    tok = BPETokenizer(enc, dec)
    # Başlangıçta boş vocab
    assert tok.vocab == {}
    # Başlatma log'u
    assert any("BPETokenizer başlatıldı. Vocab size: 0" in rec.message for rec in caplog.records)

def test_init_vocab_verilirse_sync_cagrilir(caplog):
    enc = DummyEncoder()
    dec = DummyDecoder()
    sample_vocab = {"a": {"id":1}}
    tok = BPETokenizer(enc, dec, vocab=sample_vocab)
    # encoder ve decoder set_vocab çağrılmış
    assert enc.vocab_set is sample_vocab
    assert dec.vocab_set is sample_vocab
    assert tok.vocab is sample_vocab
    assert any("senkronize edildi" in rec.message.lower() for rec in caplog.records)

# --- encode Metodu Hataları ---
def test_encode_non_str_raises():
    tok = BPETokenizer(DummyEncoder(), DummyDecoder(), vocab={"x":{}})
    with pytest.raises(BPETokenizerError) as e:
        tok.encode(123, mode="inference")
    assert "text bir str olmalıdır" in str(e.value)

def test_encode_empty_string_raises():
    tok = BPETokenizer(DummyEncoder(), DummyDecoder(), vocab={"x":{}})
    with pytest.raises(BPETokenizerError) as e:
        tok.encode("   ", mode="train")
    assert "Boş metin kodlanamaz" in str(e.value)

def test_encode_empty_vocab_raises():
    tok = BPETokenizer(DummyEncoder(), DummyDecoder())
    with pytest.raises(BPETokenizerError) as e:
        tok.encode("merhaba", mode="inference")
    assert "Vocab boş" in str(e.value)

# --- encode Doğru Çalışma ---
def test_encode_list_donerse_aynısını_gönderir():
    enc = DummyEncoder()
    dec = DummyDecoder()
    vocab = {"a": {"id":1}}
    tok = BPETokenizer(enc, dec, vocab=vocab)
    ids = tok.encode("metin", mode="inference")
    assert ids == [10,20,30]
    # dummy encode çağrısı doğru argümanla
    assert enc.encode_calls == [("metin", "inference")]

def test_encode_tuple_donerse_sadece_idleri_alır():
    class TupleEncoder(DummyEncoder):
        def encode(self, text_or_tokens, mode="inference"):
            return (["tok1","tok2"], [7,8])
    enc = TupleEncoder()
    dec = DummyDecoder()
    vocab = {"a": {"id":1}}
    tok = BPETokenizer(enc, dec, vocab=vocab)
    assert tok.encode("x", mode="train") == [7,8]

def test_encode_encoder_hatasını_sarmalar(caplog):
    tok = BPETokenizer(ErrorEncoder(), DummyDecoder(), vocab={"a":{}})
    caplog.set_level(logging.ERROR)
    with pytest.raises(BPETokenizerError) as e:
        tok.encode("x", mode="train")
    assert "Encode Error: encode hatası" in str(e.value)
    assert any("Kodlama hatası" in rec.message for rec in caplog.records)

# --- alias Metodları ---
def test_get_token_ids_alias():
    enc = DummyEncoder()
    dec = DummyDecoder()
    vocab = {"a": {"id":1}}
    tok = BPETokenizer(enc, dec, vocab=vocab)
    assert tok.get_token_ids("selam") == tok.encode("selam", mode="inference")

# --- decode Metodu Hataları ---
def test_decode_non_list_raises():
    tok = BPETokenizer(DummyEncoder(), DummyDecoder(), vocab={"a":{}})
    with pytest.raises(BPETokenizerError):
        tok.decode("notalist")

def test_decode_empty_list_raises():
    tok = BPETokenizer(DummyEncoder(), DummyDecoder(), vocab={"a":{}})
    with pytest.raises(BPETokenizerError):
        tok.decode([])

def test_decode_empty_vocab_raises():
    tok = BPETokenizer(DummyEncoder(), DummyDecoder())
    with pytest.raises(BPETokenizerError):
        tok.decode([1,2,3])

# --- decode Doğru Çalışma ---
def test_decode_strips_whitespace_and_returns(caplog):
    enc = DummyEncoder()
    dec = DummyDecoder()
    vocab = {"a": {"id":1}}
    tok = BPETokenizer(enc, dec, vocab=vocab)
    res = tok.decode([1,2,3])
    assert res == "sonuc"
    # ham decode sonucu debug loglandı
    assert any("Ham decode sonucu" in rec.message for rec in caplog.records)

def test_decode_decoder_hatasını_sarmalar(caplog):
    tok = BPETokenizer(DummyEncoder(), ErrorDecoder(), vocab={"a":{}})
    caplog.set_level(logging.ERROR)
    with pytest.raises(BPETokenizerError) as e:
        tok.decode([1])
    assert "Decode Error: decode hatası" in str(e.value)
    assert any("Çözümleme hatası" in rec.message for rec in caplog.records)

def test_get_text_alias():
    enc = DummyEncoder()
    dec = DummyDecoder()
    vocab = {"a": {"id":1}}
    tok = BPETokenizer(enc, dec, vocab=vocab)
    assert tok.get_text([9]) == tok.decode([9])

# --- update_vocab Metodu ---
def test_update_vocab_non_dict_raises():
    tok = BPETokenizer(DummyEncoder(), DummyDecoder(), vocab={"x":{}})
    with pytest.raises(BPETokenizerError):
        tok.update_vocab("notadict")

def test_update_vocab_empty_dict_raises():
    tok = BPETokenizer(DummyEncoder(), DummyDecoder(), vocab={"x":{}})
    with pytest.raises(BPETokenizerError):
        tok.update_vocab({})

def test_update_vocab_calls_sync(caplog):
    enc = DummyEncoder()
    dec = DummyDecoder()
    tok = BPETokenizer(enc, dec)
    new_vocab = {"b": {"id":2}}
    tok.update_vocab(new_vocab)
    assert tok.vocab is new_vocab
    assert enc.vocab_set is new_vocab
    assert dec.vocab_set is new_vocab
    assert any("Vocab güncellendi" in rec.message for rec in caplog.records)

# --- post_process Doğrudan Test ---
def test_post_process_strips():
    enc = DummyEncoder()
    dec = DummyDecoder()
    tok = BPETokenizer(enc, dec, vocab={"a":{}})
    # Özel metod doğrudan çağırılır
    assert tok._post_process("  ara  ") == "ara"

# --- Ek Senaryolar ---
def test_encode_train_mode_gecerli_mode_kayit_yapar():
    class ModeRecorder(DummyEncoder):
        def encode(self, text_or_tokens, mode="inference"):
            return [0]
    enc = ModeRecorder()
    tok = BPETokenizer(enc, DummyDecoder(), vocab={"a":{}})
    _ = tok.encode("x", mode="train")
    assert enc.encode_calls[-1][1] == "train"

def test_sync_vocab_sadece_set_vocab_olanlara():
    class OnlyDecoder:
        def __init__(self):
            self.v = None
        def set_vocab(self, v): self.v = v
    enc = DummyEncoder()  # set_vocab var
    dec = OnlyDecoder()   # set_vocab var
    v = {"c": {"id":3}}
    tok = BPETokenizer(enc, dec, vocab=v)
    assert enc.vocab_set is v
    assert dec.v is v

