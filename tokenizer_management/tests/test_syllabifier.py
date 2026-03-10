import pytest
import unicodedata
import logging
import re

from tokenizer_management.bpe.tokenization.syllabifier import (
    strip_diacritics,
    Syllabifier,
    SyllabificationError
)

# -------------------------------------------------------------------
# _strip_diacritics fonksiyonu
# -------------------------------------------------------------------

def test_strip_diacritics_basit():
    text = "Çeşitli örnek: âêîôû ÄÖÜ"
    stripped = strip_diacritics(text)
    assert "Ç" not in stripped and "C" in stripped
    assert "â" not in stripped and "a" in stripped
    assert "Ö" not in stripped and "O" in stripped
    assert stripped == "Cesitli ornek: aeiou AOU"

def test_strip_diacritics_turkish():
    text = "İıŞşĞğÜüÖöÇç"
    stripped = strip_diacritics(text)
    assert stripped == "IiSsGgUuOoCc"

def test_strip_diacritics_empty():
    assert strip_diacritics("") == ""

# -------------------------------------------------------------------
# Syllabifier.__init__ yapılandırma testleri
# -------------------------------------------------------------------

def test_init_default_config():
    syl = Syllabifier()
    assert isinstance(syl.vowels, set) and set("aeıioöuü") <= syl.vowels
    assert isinstance(syl.consonants, set) and set("bcçdfgğhjklmnprsştvyz") <= syl.consonants
    assert isinstance(syl.allowed_onsets, set) and {"kr", "tr", "kl", "pr"} <= syl.allowed_onsets

@pytest.mark.parametrize("bad", [123, "string", [1, 2, 3]])
def test_init_hata_config_tipi(bad):
    with pytest.raises(TypeError):
        Syllabifier(config=bad)

def test_init_hata_vowels_tipi():
    cfg = {"vowels": 123}
    with pytest.raises(ValueError):
        Syllabifier(config=cfg)

def test_init_hata_consonants_tipi():
    cfg = {"consonants": None}
    with pytest.raises(ValueError):
        Syllabifier(config=cfg)

def test_init_hata_rules_tipi():
    cfg = {"syllabification_rules": "not a dict"}
    with pytest.raises(ValueError):
        Syllabifier(config=cfg)

def test_init_hata_onset_clusters_tipi():
    cfg = {"syllabification_rules": {"allowed_onset_clusters": 999}}
    with pytest.raises(ValueError):
        Syllabifier(config=cfg)

def test_init_gecersiz_onset_cluster_filtre(caplog):
    caplog.set_level(logging.WARNING)
    cfg = {"syllabification_rules": {"allowed_onset_clusters": ["zx", "kr"]}}
    syl = Syllabifier(config=cfg)
    assert "zx" not in syl.allowed_onsets
    assert "kr" in syl.allowed_onsets
    assert any("Geçersiz onset kümesi atlandı" in rec.message for rec in caplog.records)

# -------------------------------------------------------------------
# Syllabifier.split metodu: giriş validasyonları
# -------------------------------------------------------------------

def test_split_type_error_non_list():
    syl = Syllabifier()
    with pytest.raises(TypeError):
        syl.split("not a list")

def test_split_value_error_empty_list():
    syl = Syllabifier()
    with pytest.raises(ValueError):
        syl.split([])

def test_split_type_error_non_str_element():
    syl = Syllabifier()
    with pytest.raises(TypeError):
        syl.split(["abc", 123])

# -------------------------------------------------------------------
# Syllabifier.split: özel token’lar ve sessiz/ünsüzsüz tokenlar
# -------------------------------------------------------------------

def test_split_special_token_passthrough():
    syl = Syllabifier()
    out = syl.split(["<BOS>", "<Test>", "<EOS>"])
    assert out == ["<bos>", "<test>", "<eos>"]

def test_split_token_without_vowel():
    syl = Syllabifier()
    out = syl.split(["XYZ", "123", "bcdf"])
    assert out == ["xyz", "123", "bcdf"]

# -------------------------------------------------------------------
# Syllabifier.syllabify_word temel hece bölme testleri
# -------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("kitap", ["ki", "tap"]),
    ("deneme", ["de", "ne", "me"]),
    ("ekmek", ["ek", "mek"]),
    ("şekil", ["şe", "kil"]),
    ("okul", ["o", "kul"]),
    ("kar", ["kar"]),
    ("aklim", ["ak", "lim"]),  # "kl" izinli onset
    ("kaymak", ["kay", "mak"]),  # "ym" izinli değil
    ("metropol", ["met", "ro", "pol"]),  # "tr" izinli onset
    ("kral", ["kral"]),  # "kr" izinli onset, tek hece
    ("tren", ["tren"]),  # "tr" izinli onset, tek hece
    ("psikoloji", ["psi", "ko", "lo", "ji"]),  # "ps" izinli onset
    ("transkript", ["trans", "kript"]),  # "tr" izinli onset
])
def testsyllabify_word_basit_kelime(word, expected):
    syl = Syllabifier()
    result = syl.syllabify_word(word)
    assert result == expected, f"{word} için beklenen {expected}, ama {result} alındı"

def testsyllabify_word_tek_unlu():
    syl = Syllabifier()
    assert syl.syllabify_word("kar") == ["kar"]
    assert syl.syllabify_word("ben") == ["ben"]

def testsyllabify_word_iki_unlu_arasinda_bos():
    syl = Syllabifier()
    assert syl.syllabify_word("aile") == ["a", "i", "le"]

def testsyllabify_word_uzun_kelime():
    syl = Syllabifier()
    assert syl.syllabify_word("konstantinopolis") == ["kons", "tan", "ti", "no", "po", "lis"]

# -------------------------------------------------------------------
# Syllabifier.split_into_syllables: Türkçe ekler ve karmaşık kelimeler
# -------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("erken", ["er", "ken"]),
    ("eriyor", ["e","ri", "yor"]),
    ("lanıyorum", ["la", "nı", "yo", "rum"]),
    ("dığım", ["dı", "ğım"]),
    ("kullanıyorum", ["kul", "la", "nı", "yo", "rum"]),
    ("kitaplardan", ["ki", "tap", "lar", "dan"]),
    ("bilinçsizce", ["bi", "linç", "siz", "ce"]),
    ("okuyorum", ["o", "ku", "yo", "rum"]),
    ("evlerimizden", ["ev", "le", "ri", "miz", "den"]),
    ("çocuklarımız", ["ço", "cuk", "la", "rı", "mız"]),
])
def test_split_into_syllables_turkish_suffixes(word, expected):
    syl = Syllabifier()
    result = syl.split_into_syllables(word)
    assert result == expected, f"{word} için beklenen {expected}, ama {result} alındı"

def test_split_into_syllables_empty():
    syl = Syllabifier()
    assert syl.split_into_syllables("") == []

def test_split_into_syllables_non_str():
    syl = Syllabifier()
    with pytest.raises(TypeError):
        syl.split_into_syllables(123)

def test_split_into_syllables_simple_sentence():
    syl = Syllabifier()
    text = "Merhaba, dünya! <TAG>"
    parts = syl.split_into_syllables(text)
    assert parts == [
        "mer", "ha", "ba", ",", " ", "dün", "ya", "!", " ", "<tag>"
    ]

def test_split_into_syllables_complex_sentence():
    syl = Syllabifier()
    text = "Kitaplarımı okuyorum, naber? <BOS>"
    parts = syl.split_into_syllables(text)
    assert parts == [
        "ki", "tap", "la", "rı", "mı", " ", "o", "ku", "yo", "rum", ",", " ",
        "na", "ber", "?", " ", "<bos>"
    ]

def test_split_into_syllables_preserves_spaces_and_punctuation():
    syl = Syllabifier()
    text = "a b?c!"
    parts = syl.split_into_syllables(text)
    assert parts == ["a", " ", "b", "?", "c", "!"]

def test_split_into_syllables_numbers():
    syl = Syllabifier()
    text = "12345 test 678"
    parts = syl.split_into_syllables(text)
    assert parts == ["12345", " ", "test", " ", "678"]

def test_split_into_syllables_mixed_content():
    syl = Syllabifier()
    text = "Merhaba! Kitaplar <EOS> 123 naber?"
    parts = syl.split_into_syllables(text)
    assert parts == [
        "mer", "ha", "ba", "!", " ", "ki", "tap", "lar", " ", "<eos>", " ",
        "123", " ", "na", "ber", "?"
    ]

# -------------------------------------------------------------------
# Syllabifier.split: birden fazla kelime ve karmaşık testler
# -------------------------------------------------------------------

def test_split_entire_sentence_mixed():
    syl = Syllabifier()
    toks = ["Kitaplar", "ev", "test", "edildi."]
    parts = syl.split(toks)
    assert parts == ["ki", "tap", "lar", "ev", "test", "e", "dil", "di", "."]

def test_split_complex_sentence_with_suffixes():
    syl = Syllabifier()
    toks = ["kullanıyorum", "kitaplarımı", "okurken", "<TAG>"]
    parts = syl.split(toks)
    assert parts == [
        "kul", "la", "nı", "yo", "rum", "ki", "tap", "la", "rı", "mı",
        "o", "kur", "ken", "<tag>"
    ]

def test_split_multiple_words_with_punctuation():
    syl = Syllabifier()
    toks = ["Merhaba,", "dünya!", "naber?"]
    parts = syl.split(toks)
    assert parts == ["mer", "ha", "ba", ",", "dün", "ya", "!", "na", "ber", "?"]

# -------------------------------------------------------------------
# Hata durumları ve edge case'ler
# -------------------------------------------------------------------

def test_split_heceleme_hatasi_yakalanir(caplog):
    class Broken:
        def split(self, toks):
            raise RuntimeError("bozuk heceleyici")
    syl = Syllabifier()
    syl.syllabifier = Broken()
    caplog.set_level(logging.WARNING)
    out = syl.split(["deneme"])
    assert "bozuk heceleyici" in caplog.text
    assert out == ["deneme"]

def test_sylla_error_if_no_syllables_produced(monkeypatch):
    syl = Syllabifier()
    monkeypatch.setattr(syl, "syllabify_word", lambda w: [])
    with pytest.raises(SyllabificationError):
        syl.split(["test"])

def testsyllabify_word_no_vowel():
    syl = Syllabifier()
    assert syl.syllabify_word("bcdf") == ["bcdf"]

def testsyllabify_word_long_consonant_cluster():
    syl = Syllabifier()
    assert syl.syllabify_word("angstrom") == ["ang", "st","rom"]  # "str" izinli onset

def testsyllabify_word_invalid_chars():
    syl = Syllabifier()
    assert syl.syllabify_word("test#") == ["test", "#"]

def testsyllabify_word_single_char():
    syl = Syllabifier()
    assert syl.syllabify_word("a") == ["a"]
    assert syl.syllabify_word("b") == ["b"]

def test_split_into_syllables_long_complex_word():
    syl = Syllabifier()
    text = "konstantinopolis"
    parts = syl.split_into_syllables(text)
    assert parts == ["kons", "tan", "ti", "no", "po", "lis"]

# -------------------------------------------------------------------
# Türkçe ekler için özel testler (BPEManager hatalarını çözmek için)
# -------------------------------------------------------------------

@pytest.mark.parametrize("suffix,expected", [
    ("iyor", ["i", "yor"]),
    ("dığım", ["dı", "ğım"]),
    ("ken", ["ken"]),
    ("lerimiz", ["le", "ri", "miz"]),
    ("larımızdan", ["la", "rı", "mız", "dan"]),
    ("sizce", ["siz", "ce"]),
])
def test_split_into_syllables_turkish_suffixes_specific(suffix, expected):
    syl = Syllabifier()
    result = syl.split_into_syllables(suffix)
    assert result == expected, f"{suffix} için beklenen {expected}, ama {result} alındı"

# -------------------------------------------------------------------
# Performans ve karmaşık cümle testleri
# -------------------------------------------------------------------

def test_split_into_syllables_long_sentence():
    syl = Syllabifier()
    text = "Merhaba arkadaşlar, kitaplarımı okuyorum ve naber diyorum! <EOS>"
    parts = syl.split_into_syllables(text)
    assert parts == [
        "mer", "ha", "ba", " ", "ar", "ka", "daş", "lar", ",", " ",
        "ki", "tap", "la", "rı", "mı", " ", "o", "ku", "yo", "rum", " ",
        "ve", " ", "na", "ber", " ", "di", "yo", "rum", "!", " ", "<eos>"
    ]

def test_split_into_syllables_mixed_with_numbers_and_punctuation():
    syl = Syllabifier()
    text = "2023 yılında, psikoloji kitapları okuyorum."
    parts = syl.split_into_syllables(text)
    assert parts == [
        "2023", " ", "yı", "lın", "da", ",", " ", "psi", "ko", "lo", "ji", " ",
        "ki", "tap", "la", "rı", " ", "o", "ku", "yo", "rum", "."
    ]

# -------------------------------------------------------------------
# Toplam test sayısı
# -------------------------------------------------------------------
# Toplam: 35 test fonksiyonu