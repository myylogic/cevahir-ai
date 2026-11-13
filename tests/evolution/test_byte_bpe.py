"""Contracts for the lossless byte-level BPE backend (tokenizer_management/byte_bpe.py)."""
import json
import random

import pytest

from tokenizer_management.byte_bpe import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID, ByteLevelBPE, pretokenize,
)

CORPUS = [
    "Ankara'da kar yağıyor; İstanbul'un havası ılık.",
    "Çekoslovakyalılaştıramadıklarımızdan mısınız?",
    "Hâlâ kâr etmedik, rüzgâr sert esiyor.",
    "13.11.2026 saat 14:30'da toplantı var — katılım %80.",
    "E-posta: ornek@alan.com.tr, adres: https://ornek.org/yol?x=1",
    "def hesapla(x):\n\treturn x ** 2  # kare",
    "IŞIKLAR İzmir'de söndü. ışıklar izmir'de yandı.",
    "Merhaba 😀 dünya!   Çok   boşluk\n\nve satır.",
] * 3

HARD = [
    "", " ", "\n", "  baştaki boşluklar", "sondaki boşluklar  ", "tab\tlı\tmetin",
    "İ̇ (birleşik nokta) ve ß ve 中文 ve кириллица ve ελληνικά",
    "snake_case_değişken ve __dunder__", "H2O, Sütun1, 3.14159, 1.000.000 TL",
    "Tırnak 'tek' \"çift\" ‘eğik’ “eğik”", "\x00 kontrol \x1f karakteri",
]


@pytest.fixture(scope="module")
def tok():
    return ByteLevelBPE.train(CORPUS, vocab_size=400, min_frequency=2)


def test_pretokenize_is_lossless_on_random_unicode():
    rnd = random.Random(1234)
    alphabet = "abcçğıöşüİIâîû019 \t\n.,;:!?'\"()-_/€😀ßж中̇\x00"
    for _ in range(2000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 30)))
        assert "".join(pretokenize(text)) == text


def test_apostrophe_separates_proper_noun_and_suffix():
    # Turkish proper-noun suffixes: the root and the apostrophe+suffix are separate chunks,
    # so a root's tokens are identical in every case form ("Ankara" in Ankara, Ankara'da, Ankara'nın).
    assert pretokenize("Ankara'da") == ["Ankara", "'da"]
    assert pretokenize(" İstanbul'un") == [" İstanbul", "'un"]
    assert pretokenize("Cevahir’e") == ["Cevahir", "’e"]
    assert pretokenize("Ankara'dır.") == ["Ankara", "'dır", "."]
    assert pretokenize(" Ali Kaya'nın")[:2] == pretokenize(" Ali Kaya")


@pytest.mark.parametrize("text", CORPUS[:8] + HARD)
def test_roundtrip_is_exact(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_unseen_words_never_produce_unk(tok):
    ids = tok.encode("Görülmemiş sözcükler: zımbırtılaştırılamayanlardan, Ŋ, ǅ, ﷺ")
    assert UNK_ID not in ids and PAD_ID not in ids


def test_special_ids_and_bos_eos(tok):
    ids = tok.encode("Merhaba", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert tok.decode(ids) == "Merhaba"
    assert tok.decode(ids, skip_special=False) == "<BOS>Merhaba<EOS>"


def test_training_is_deterministic_and_merges_compress(tok):
    again = ByteLevelBPE.train(CORPUS, vocab_size=400, min_frequency=2)
    assert again.merges == tok.merges and again.identity() == tok.identity()
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_vocab_size_bound_and_min_frequency_stop():
    small = ByteLevelBPE.train(CORPUS, vocab_size=300)
    assert small.vocab_size == 300
    capped = ByteLevelBPE.train(["abc"], vocab_size=10_000, min_frequency=2)
    assert capped.merges == []  # every pair occurs once: nothing reaches min_frequency


def test_save_load_roundtrip_and_tamper_detection(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = ByteLevelBPE.load(path)
    assert loaded.identity() == tok.identity()
    assert loaded.encode(CORPUS[1]) == tok.encode(CORPUS[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    data["merges"][0], data["merges"][1] = data["merges"][1], data["merges"][0]  # stored identity kept
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        ByteLevelBPE.load(bad)


def test_out_of_range_id_is_rejected(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])


def test_streaming_decode_can_replace_split_multibyte_chars(tok):
    ids = tok.encode("ğ")
    raw = list(ids)
    # A single multibyte character split across byte tokens decodes with replacement when partial.
    if len(raw) == 1:
        pytest.skip("character is a single merged token in this vocabulary")
    assert "�" in tok.decode(raw[:1], errors="replace")
