import pytest
import re
import unicodedata
from tokenizer_management.bpe.tokenization.pretokenizer import (
    Pretokenizer,
    PretokenizationError,
)


@pytest.fixture
def pretok():
    return Pretokenizer()


def test_unicode_normalizasyon_ve_diyakritik_temizleme():
    # "Café", "naïve", "São Paulo" gibi örnekler
    p = Pretokenizer(lower=False)
    assert p._normalize_unicode("Café") == "Cafe"
    assert p._normalize_unicode("naïve") == "naive"
    assert p._normalize_unicode("São Paulo") == "Sao Paulo"


def test_ozel_karakter_temizleme_default():
    p = Pretokenizer(lower=False)
    # Default cleanup_pattern, noktalama ve semboller boşluk oluyor
    temiz = p._clean_specials("hello@#world!")
    assert "@" not in temiz and "#" not in temiz and "!" not in temiz
    assert temiz.count(" ") >= 2


def test_split_alphanum_varsayilan():
    p = Pretokenizer(lower=False)
    # Harf ile sayı arasına boşluk ekler
    assert p._split_alphanum("abc123def") == "abc 123 def"
    assert p._split_alphanum("2020 yılında") == "2020  yılında"


def test_whitespace_tokenizasyon_dogru_bolme():
    p = Pretokenizer(lower=False)
    tokens = p._tokenize_whitespace("  bir   iki\tüç\n dört ")
    assert tokens == ["bir", "iki", "üç", "dört"]


@pytest.mark.parametrize("tokens,expected", [
    (["abc", "123", "def"], ["abc", "123", "def"]),  # hepsi geçerli
    (["çğüşİ", "hello"], ["çğüşİ", "hello"]),       # Türkçe karakterler
    (["你好", "123"], ["123"]),                      # Çince atlanır, sayı geçer
    (["!!!"], ["<EMPTY>"]),                          # tamamı silinince <EMPTY>
])
def test_filter_tokens(tokens, expected):
    p = Pretokenizer(lower=False)
    assert p._filter_tokens(tokens) == expected


def test_tokenize_bosluk():
    p = Pretokenizer()
    assert p.tokenize("") == ["<EMPTY>"]
    assert p.tokenize("   ") == ["<EMPTY>"]


def test_tokenize_none_raises():
    p = Pretokenizer()
    with pytest.raises(PretokenizationError):
        p.tokenize(None)


def test_tokenize_non_string_raises():
    p = Pretokenizer()
    with pytest.raises(PretokenizationError):
        p.tokenize(12345)


def test_tokenize_list_input():
    p = Pretokenizer()
    # Listeyi önce join edip tokenize eder
    out = p.tokenize(["Merhaba", "DÜNYA"])
    assert "merhaba" in out and "dunya" in out


def test_tokenize_dict_input():
    p = Pretokenizer()
    data = {"data": "Test 123"}
    out = p.tokenize(data)
    assert "test" in out and "123" in out


def test_lower_case_devre_disi():
    p = Pretokenizer(lower=False)
    out = p.tokenize("ABC ÇĞİ")
    assert any(tok.isupper() for tok in out)


def test_cleanup_pattern_ozel():
    # Rakamları tamamen kaldıralım
    pat = re.compile(r"\d")
    p = Pretokenizer(cleanup_pattern=pat)
    toks = p.tokenize("a1b2c3")
    # rakamlar silindiği için "abc"
    assert toks == ["abc"]


def test_split_pattern_ozel():
    # Harf-harf arasına boşluk ekleyelim
    pat = re.compile(r"(?<=\w)(?=\w)")
    p = Pretokenizer(split_pattern=pat)
    # her karakteri ayrı tokene dönüştürür
    toks = p.tokenize("ab12")
    assert "a" in toks and "b" in toks and "1" in toks and "2" in toks


def test_numeric_alfanum_split_pipeline():
    p = Pretokenizer()
    toks = p.tokenize("ölçü2025test")
    # ölçü 2025 test olarak ayrılmalı
    assert toks[:3] == ["olcu", "2025", "test"]


def test_whitespace_ve_tab_yeni_satir_temizleme():
    p = Pretokenizer()
    toks = p.tokenize("ilk\tikinci\nüçüncü  dört")
    assert toks == ["ilk", "ikinci", "ucuncu", "dort"]


def test_cache_kullanimi_ve_reset():
    p = Pretokenizer()
    first = p.tokenize("cache test")
    assert "cache" in first
    # cache dolu mu?
    assert "cache test" in p.cache
    p.reset()
    assert p.cache == {}


def test_validate_token_true_false():
    p = Pretokenizer()
    assert p.validate_token("abc123") is True
    assert p.validate_token("你好") is False


def test_get_token_offsets_basit():
    p = Pretokenizer()
    text = "merhaba dünya"
    offs = p.get_token_offsets(text)
    # iki token ve pozisyonları
    assert offs[0][0] == "merhaba" and offs[0][1] == 0
    assert offs[1][0] == "dunya" and offs[1][2] == len(text)


def test_syllabifier_injection_split_metodu():
    class FakeSyl:
        def split(self, toks):
            return [t + "_syl" for t in toks]

    p = Pretokenizer(syllabifier=FakeSyl())
    toks = p.tokenize("Merhaba123")
    # heceleyici eklenince sonunda "_syl" olur
    assert all(t.endswith("_syl") for t in toks)


def test_syllabifier_hatasi_yakalanir(caplog):
    class BadSyl:
        def split(self, toks):
            raise RuntimeError("bozuk heceleyici")

    p = Pretokenizer(syllabifier=BadSyl())
    caplog.set_level("WARNING")
    toks = p.tokenize("deneme")
    # hata yakalanıp orijinal tokenlar döner
    assert "deneme" in toks
    assert any("Syllabifier hatası" in rec.message for rec in caplog.records)


def test_valid_characters_ozel_filtre():
    # Sadece rakamı kabul edelim
    p = Pretokenizer(valid_characters=set("0123456789"))
    toks = p.tokenize("a1b2")
    # sadece rakamlar kalır
    assert toks == ["1", "2"]


def test_birden_fazla_gecersiz_token_tumünü_siler():
    p = Pretokenizer()
    # tamamen geçersiz karakter kümesi
    toks = p.tokenize("♪♫")
    assert toks == ["<EMPTY>"]


# Yeni Test Metodları


def test_complex_unicode_chars():
    """Emoji, sembol ve çoklu diyakritik işaretlerin doğru işlenmesini test eder."""
    p = Pretokenizer(lower=False)
    text = "Smiling😊Café🍵a̐ěȋŏû"
    toks = p.tokenize(text)
    assert "smiling" in toks
    assert "cafe" in toks
    assert "aeiou" in toks
    assert "😊" not in ''.join(toks) and "🍵" not in ''.join(toks)


def test_invalid_cleanup_pattern_raises():
    """Geçersiz cleanup_pattern regex'inin hata fırlatmasını test eder."""
    with pytest.raises(re.error):
        Pretokenizer(cleanup_pattern=re.compile(r"[\w+"))  # Eksik kapatma parantezi


def test_punctuation_separation():
    """Noktalama işaretlerinin ayrı tokenlere bölündüğünü test eder."""
    p = Pretokenizer(lower=False)
    text = "Merhaba,dünya!Naber?"
    toks = p.tokenize(text)
    assert toks == ["Merhaba", ",", "dunya", "!", "Naber", "?"]
    assert "," in toks and "!" in toks and "?" in toks


def test_long_text_cache_performance():
    """Çok uzun metinlerin önbellek ile doğru işlenmesini test eder."""
    p = Pretokenizer()
    long_text = "test " * 1000  # 5000 karakter
    first_toks = p.tokenize(long_text)
    assert len(first_toks) == 1000 and all(t == "test" for t in first_toks)
    assert long_text in p.cache
    second_toks = p.tokenize(long_text)
    assert first_toks == second_toks  # Önbellekten gelmeli


def test_syllabifier_empty_tokens():
    """Heceleyiciye boş token listesi verildiğinde davranışını test eder."""
    class EmptySyl:
        def split(self, toks):
            return []

    p = Pretokenizer(syllabifier=EmptySyl())
    toks = p.tokenize("merhaba")
    assert toks == ["<EMPTY>"]  # Heceleyici boş liste dönerse <EMPTY>


def test_token_offsets_with_punctuation():
    """Noktalama işaretleri içeren metinlerde offset doğruluğunu test eder."""
    p = Pretokenizer()
    text = "Merhaba, dünya!"
    offsets = p.get_token_offsets(text)
    expected = [
        ("merhaba", 0, 7),
        (",", 7, 8),
        ("dunya", 9, 14),
        ("!", 14, 15),
    ]
    assert offsets == expected


def test_no_valid_tokens_after_cleanup():
    """Temizleme sonrası hiç geçerli token kalmadığında <EMPTY> test eder."""
    p = Pretokenizer(cleanup_pattern=re.compile(r"[a-zA-Z0-9\s]"))
    toks = p.tokenize("abc123")
    assert toks == ["<EMPTY>"]  # Tüm geçerli karakterler silindi


def test_lower_case_turkish_chars():
    """Küçük harfe çevirirken Türkçe karakterlerin doğru işlenmesini test eder."""
    p = Pretokenizer(lower=True)
    text = "İSTANBUL ÇÖŞ"
    toks = p.tokenize(text)
    assert "istanbul" in toks and "çoş" in toks
    assert not any(c.isupper() for c in ''.join(toks))


def test_multiple_pipeline_runs():
    """Aynı metnin farklı ayarlarla birden fazla tokenize edilmesini test eder."""
    text = "Merhaba DÜNYA 123"
    p1 = Pretokenizer(lower=True)
    p2 = Pretokenizer(lower=False)
    toks1 = p1.tokenize(text)
    toks2 = p2.tokenize(text)
    assert toks1 == ["merhaba", "dunya", "123"]
    assert toks2 == ["Merhaba", "DÜNYA", "123"]
    assert text in p1.cache and text in p2.cache


def test_error_message_content():
    """PretokenizationError hata mesajlarının doğruluğunu test eder."""
    p = Pretokenizer()
    with pytest.raises(PretokenizationError, match="Girdi None olamaz"):
        p.tokenize(None)
    with pytest.raises(PretokenizationError, match="Girdi string olmalı"):
        p.tokenize(123)