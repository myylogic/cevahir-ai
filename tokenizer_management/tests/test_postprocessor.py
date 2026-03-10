# tokenizer_management/tests/test_postprocessor_tr.py

import re
import unicodedata
import pytest

from tokenizer_management.bpe.tokenization.postprocessor import (
    Postprocessor,
    PostProcessingError,
)

@pytest.fixture
def pp():
    """Her test için yeni, varsayılan bir Postprocessor örneği döner."""
    return Postprocessor()

def test_boş_token_listesi_boş_string_döndürür(pp):
    """
    Boş bir token listesi verildiğinde process() metodu boş string döndürmeli,
    ayrıca _apply_special_tokens([]) da boş liste için boş string vermeli.
    """
    assert pp.process([]) == ""
    assert pp._apply_special_tokens([]) == ""

def test_varsayılan_özel_tokenlar_doğru_uygulanır(pp):
    """
    Varsayılan özel tokenlar (<PAD>, <UNK>, <BOS>, <EOS>) doğru şekilde
    silinmeli veya dönüştürülmeli.
    """
    tokens = ["merhaba", "<PAD>", "dünya", "<UNK>", "<BOS>", "<EOS>"]
    sonuc = pp.process(tokens)
    # "<PAD>" ve "<BOS>" silinir, "<UNK>" → [UNK], "<EOS>" → "."
    assert sonuc == "Merhaba dünya [UNK]."

def test_özel_token_haritasi_ile_kendi_ayarlar(pp):
    """
    Kendi special_tokens haritamızı verip, capitalize_sentence=False ile
    cümle başı büyütmeyi iptal edebiliriz.
    """
    ozel = {"<PAD>": "_PAD_", "<EOS>": "[END]"}
    pp2 = Postprocessor(special_tokens=ozel, capitalize_sentence=False)
    assert pp2.process(["<PAD>", "test", "<EOS>"]) == "_PAD_ test [END]"

def test_unicode_nfc_normalizasyonu(pp):
    """
    Dekompoze edilmiş Unicode karakterler NFC formuna çevrilmeli.
    """
    dekompoze = "Cafe\u0301"
    normalized = pp._normalize_unicode(dekompoze)
    assert normalized == "Café"

@pytest.mark.parametrize("giris, beklenen", [
    ("a ,b",   "a,b"),
    ("x . y",  "x. y"),
    ("hey !you", "hey!you"),
    ("do ? it", "do? it"),
    ("a : b",  "a: b"),
    ("foo ;bar", "foo;bar"),
    ("( hi",   "(hi"),
    ("bye )",  "bye)"),
])
def test_noktalama_bosluk_duzeltmeleri(giris, beklenen, pp):
    """
    Noktalama işaretleri etrafındaki gereksiz boşluklar,
    DEFAULT_PUNCTUATION_FIXES ile düzeltilmeli.
    """
    assert pp._fix_punctuation_spacing(giris) == beklenen

@pytest.mark.parametrize("giris, beklenen", [
    ("çok    boşluk",      "çok boşluk"),
    ("  başta ve sonda  ", "başta ve sonda"),
    ("\nçok\nsatır\n",     "çok satır"),
    ("\tçok\tsekme\t",     "çok sekme"),
])
def test_fazla_bosluk_temizleme(giris, beklenen, pp):
    """
    Birden fazla boşluk, tab veya newline tek boşluğa indirgense,
    baş/son boşluklar kırpılsa.
    """
    assert pp._collapse_whitespace(giris) == beklenen

def test_cümle_başlarını_büyük_harf(pp):
    """
    Nokta, ünlem ve soru işaretinden sonra gelen cümleler
    büyük harfle başlamalı.
    """
    txt = "merhaba. bu bir test! doğru mu? evet"
    assert pp._capitalize_sentences(txt) == "Merhaba. Bu bir test! Doğru mu? Evet"

def test_capitalize_sentence_devre_dışı(pp):
    """
    capitalize_sentence=False iken cümle başı büyütme yapılmamalı.
    """
    pp2 = Postprocessor(capitalize_sentence=False)
    assert pp2.process(["merhaba", ".", "<EOS>"]) == "merhaba."

def test_tam_pipeline(pp):
    """
    process() metodu tüm adımları (özel token, normalizasyon,
    noktalama düzeltme, boşluk çökertme, cümle başı) eksiksiz uygulamalı.
    """
    tokens = ["<BOS>", "Cafe\u0301", ",", "nasıl", "?", "iyi", "<EOS>"]
    assert pp.process(tokens) == "Café, nasıl? İyi."

def test_çoklu_cümle_işleme(pp):
    """
    Birden fazla cümle içeren tokenize edilmiş liste
    doğru şekilde işlenmeli.
    """
    tokens = ["merhaba", ",", "nasılsın", "?", "ben", "iyi", ".", "<EOS>"]
    assert pp.process(tokens) == "Merhaba, nasılsın? Ben iyi."

def test_bilinmeyen_tokenlar_değişmeden_geçer(pp):
    """
    Haritada olmayan tokenlar aynen geçmeli.
    """
    assert pp.process(["özel", "token", "."]) == "Özel token."

def test_pad_eos_işleme(pp):
    """
    Arka arkaya gelen <PAD> silinir, <EOS> nokta olarak eklenir.
    """
    assert pp.process(["<PAD>", "son", "<EOS>", "<EOS>"]) == "Son."

def test_reset_varsayilanlara_döner():
    """
    reset() çağrıldığında özelleştirilmiş haritalar silinip,
    varsayılan ayarlara dönmeli.
    """
    custom = {"<PAD>": "X", "<UNK>": "?"}
    pp2 = Postprocessor(special_tokens=custom, capitalize_sentence=False)
    pp2.reset()
    assert pp2.process(["<PAD>", "<UNK>", "ok", "<EOS>"]) == "Ok [UNK]."

def test_geçersiz_regex_istisna():
    """
    punctuation_fixes için hatalı regex örneği PostProcessingError fırlatmalı.
    """
    hatali = {r"(\s+": ","}  # dengesi bozuk parantez
    with pytest.raises(PostProcessingError):
        Postprocessor(punctuation_fixes=hatali)

def test_apply_special_tokens_metodu(pp):
    """
    _apply_special_tokens doğrudan çağrıldığında
    tokenlar doğru eşlenmeli ve boşlar atılmalı.
    """
    tokens = ["x", "<UNK>", "y"]
    assert pp._apply_special_tokens(tokens) == "x [UNK] y"

def test_normalize_fix_collapse_sekans(pp):
    """
    normalize → fix punctuation → collapse whitespace
    adımları ardışık olarak test edilir.
    """
    tokens = ["Cafe\u0301", " ", ",", "hey", "   you", "<EOS>"]
    assert pp.process(tokens) == "Café, hey you."

def test_tab_newline_temizleme(pp):
    """
    Tab ve yeni satır karakterleri de boşluk olarak işlenip kırpılmalı.
    """
    tokens = ["ilk", "\n", "ikinci", "\t", "<EOS>"]
    assert pp.process(tokens) == "İlk ikinci."

def test_arka_arkaya_ozel_token_sil(pp):
    """
    <PAD>, <BOS> gibi arka arkaya gelen özel tokenlar tamamen silinmeli.
    """
    tokens = ["<PAD>", "<BOS>", "<PAD>", "<EOS>"]
    assert pp.process(tokens) == "."

def test_baştaki_ve_sondaki_bosluk_kırp(pp):
    """
    process() sonucu hem baştaki hem sondaki boşluklar kırpılmalı.
    """
    tokens = [" ", "<PAD>", "hi", "<EOS>", " "]
    assert pp.process(tokens) == "Hi."

def test_özel_metotlara_erişim(pp):
    """
    Bütün yardımcı (_ ile başlayan) metodlar callable olmalı.
    """
    assert callable(pp._normalize_unicode)
    assert callable(pp._fix_punctuation_spacing)
    assert callable(pp._collapse_whitespace)
    assert callable(pp._capitalize_sentences)

def test_uzun_cümle_divide_olmaz(pp):
    """
    Noktalama içermeyen uzun bir cümle bir bütün olarak kalmalı.
    """
    tokens = ["bu", "bir", "deneme", "cümlesi", "<EOS>"]
    assert pp.process(tokens) == "Bu bir deneme cümlesi."

def test_cift_eos_nokta_tekrar_yapmaz(pp):
    """
    Arka arkaya gelen <EOS> ikince noktayı eklememeli.
    """
    tokens = ["merhaba", "<EOS>", "<EOS>"]
    assert pp.process(tokens) == "Merhaba."


def test_turkish_capitalization_special_cases(pp):
    """
    Türkçe'ye özgü büyük harf dönüşümleri (i → İ, ı → I) doğru çalışmalı.
    """
    txt = "ıssız bir ada. istanbul çok güzel! izmir mi?"
    assert pp._capitalize_sentences(txt) == "Issız bir ada. İstanbul çok güzel! İzmir mi?"

def test_complex_punctuation_and_spacing(pp):
    """
    Birden fazla noktalama işareti ve boşluk içeren karmaşık girişler düzeltilmeli.
    """
    tokens = ["merhaba", "  ,  ", "nasılsın", " ?  ", "evet", "  !", "<EOS>"]
    assert pp.process(tokens) == "Merhaba, nasılsın? Evet."

def test_foreign_unicode_chars(pp):
    """
    Türkçe dışı Unicode karakterler (örneğin, Arapça veya Farsça) normalizasyonla işlenmeli.
    """
    tokens = ["مرحبا", ",", "dünya", "<EOS>"]  # Arapça "merhaba"
    assert pp.process(tokens) == "مرحبا, dünya."

def test_multiple_special_token_combinations(pp):
    """
    Çoklu özel token kombinasyonları (örneğin, ardışık <PAD>, <UNK>, <EOS>) doğru işlenmeli.
    """
    tokens = ["<PAD>", "<UNK>", "<PAD>", "test", "<EOS>", "<UNK>", "<EOS>"]
    assert pp.process(tokens) == "Test [UNK]."

def test_invalid_unicode_input(pp):
    """
    Bozuk veya geçersiz Unicode karakterleri hata fırlatmadan işlenmeli.
    """
    tokens = ["test", "\ud800", "<EOS>"]  # Geçersiz Unicode karakter
    assert pp.process(tokens) == "Test."  # Hatalı karakter yok sayılmalı veya temizlenmeli

def test_long_complex_sentence(pp):
    """
    Uzun ve karmaşık cümleler (çok sayıda token ve noktalama) doğru işlenmeli.
    """
    tokens = [
        "bu", "<PAD>", "bir", "uzun", ",", "karmasık", "cümle", "!", 
        "naber", "?", "evet", ".", "<BOS>", "devam", "<EOS>"
    ]
    assert pp.process(tokens) == "Bu bir uzun, karmasık cümle! Naber? Evet devam."

def test_invalid_regex_complex_pattern():
    """
    Daha karmaşık hatalı regex pattern'ları PostProcessingError fırlatmalı.
    """
    hatali = {r"[a-z+": ","}  # Eksik kapatma parantezi
    with pytest.raises(PostProcessingError):
        Postprocessor(punctuation_fixes=hatali)