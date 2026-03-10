# tokenizer_management/tests/test_morphology.py

import pytest
from tokenizer_management.bpe.tokenization.morphology import Morphology, MorphologyError

@pytest.fixture
def morph():
    return Morphology()

# === TEMEL TESTLER ===

def test_split_morpheme_basic(morph):
    result = morph._split_morpheme("kitaplarda")
    assert result[0] == "kitap"
    assert "lar" in result or "da" in result

def test_split_morpheme_no_suffix(morph):
    assert morph._split_morpheme("merhaba") == ["merhaba"]

def test_split_morpheme_multi_suffix(morph):
    result = morph._split_morpheme("kitaplarımızdan")
    assert result[0] == "kitap"
    assert any(s in result for s in ["larımız", "dan", "larımızdan"])

def test_split_morpheme_yumusama(morph):
    # Ünsüz yumuşaması (kitabım → kitap)
    result = morph._split_morpheme("kitabım")
    assert result[0] == "kitap"

def test_split_morpheme_uyumlu_ek(morph):
    result = morph._split_morpheme("evimizde")
    assert result[0] == "ev"
    assert "imiz" in result and "de" in result

def test_split_morpheme_vowel_drop(morph):
    # Ünlü düşmesi (oğluyla → oğul)
    result = morph._split_morpheme("oğluyla")
    assert result[0] == "oğul"
    assert any(s in result for s in ["yla", "la"])

def test_split_morpheme_birlesik_ek(morph):
    # Birleşik/bitişik ek zinciri (gördüğümüzden)
    result = morph._split_morpheme("gördüğümüzden")
    assert result[0] == "gör"
    assert any("den" in s or "düğümüz" in s or "ümüzden" in s for s in result)

def test_split_morpheme_edge_case(morph):
    # Yansıma kelimeler
    result = morph._split_morpheme("cıvıldamak")
    assert result[0].startswith("cıvıl")

def test_split_morpheme_invalid(morph):
    with pytest.raises(TypeError):
        morph._split_morpheme(None)
    with pytest.raises(TypeError):
        morph._split_morpheme("")

# === KÖK BULMA ===

def test_find_root_regular(morph):
    assert morph.find_root("kitaplarda") == "kitap"
    assert morph.find_root("geliyorsun").startswith("gel")
    assert morph.find_root("edildi") == "et"

def test_find_root_plural(morph):
    assert morph.find_root("çocuklar") == "çocuk"
    assert morph.find_root("arabalar") == "araba"

def test_find_root_verb_causative(morph):
    assert morph.find_root("koşturdu") == "koş"
    assert morph.find_root("yazdırıldı") == "yaz"

def test_find_root_vowel_softening(morph):
    assert morph.find_root("kitabı") == "kitap"
    assert morph.find_root("ağacı") == "ağaç"
    assert morph.find_root("tadı") == "tat"

def test_find_root_personal_pronouns(morph):
    assert morph.find_root("benim") == "ben"
    assert morph.find_root("senin") == "sen"
    assert morph.find_root("onların") == "on"

def test_find_root_abbreviation(morph):
    assert morph.find_root("tbmm") == "tbmm"
    assert morph.find_root("abd'den") == "abd"

def test_find_root_birlesik_fiil(morph):
    assert morph.find_root("yardım etti") == "yardım et"
    assert morph.find_root("fark etti") == "fark et"

def test_find_root_with_question_particle(morph):
    assert morph.find_root("geldin mi") == "gel"
    assert morph.find_root("aldın mı") == "al"

def test_find_root_with_negation(morph):
    assert morph.find_root("yapmadı") == "yap"
    assert morph.find_root("gelmeyecek") == "gel"

def test_find_root_with_time_adverb(morph):
    assert morph.find_root("geliyordu") == "gel"
    assert morph.find_root("gidecekti") == "git"

def test_find_root_no_suffix(morph):
    assert morph.find_root("merhaba") == "merhaba"
    assert morph.find_root("ifade") == "ifade"

def test_find_root_invalid_input(morph):
    with pytest.raises(MorphologyError):
        morph.find_root(None)
    with pytest.raises(MorphologyError):
        morph.find_root("")

# === MORFOLOJİK ANALİZ & EDGE-CASE ===

def test_analyze_words(morph):
    tokens = ["kitaplar", "evlerden", "çocuk"]
    roots = morph.analyze(tokens)
    assert "kitap" in roots
    assert "ev" in roots
    assert "çocuk" in roots

def test_analyze_composite_suffix(morph):
    roots = morph.analyze(["kitaplarımızdan"])
    assert roots == ["kitap"]

def test_analyze_mixed_tokens(morph):
    tokens = ["kompozisyon", "bilinçsizce", "kitaplarının"]
    roots = morph.analyze(tokens)
    assert roots[0] == "kompozisyon"
    assert any("bilinç" in r for r in roots)
    assert "kitap" in roots[-1]

def test_analyze_invalid_token_in_list(morph):
    with pytest.raises(MorphologyError):
        morph.analyze(["kitap", 123, "ev"])

def test_analyze_edge_cases(morph):
    with pytest.raises(MorphologyError):
        morph.analyze(None)
    with pytest.raises(MorphologyError):
        morph.analyze([])
    with pytest.raises(MorphologyError):
        morph.analyze([123])

def test_analyze_preserves_order(morph):
    tokens = ["kitaplar", "merhaba", "arkadaşlar"]
    roots = morph.analyze(tokens)
    assert roots[0] == "kitap"
    assert roots[1] == "merhaba"
    assert roots[2].startswith("arkadaş")

def test_analyze_empty_string_token(morph):
    roots = morph.analyze([""])
    assert roots == [""]

def test_analyze_repeated_tokens(morph):
    tokens = ["kitap", "kitap", "ev"]
    roots = morph.analyze(tokens)
    assert roots == ["kitap", "kitap", "ev"]

# === ÜNLÜ UYUMU & İSTİSNA KONTROLÜ ===

def test_check_vowel_harmony_valid(morph):
    assert morph._check_vowel_harmony("kitap") is True
    assert morph._check_vowel_harmony("güzellik") is True
    assert morph._check_vowel_harmony("araba") is True

def test_check_vowel_harmony_exceptions(morph):
    assert morph._check_vowel_harmony("elma") is True
    assert morph._check_vowel_harmony("merhaba") is True

def test_check_vowel_harmony_no_vowel(morph):
    assert morph._check_vowel_harmony("rhythm") is True

# === RESET & YARDIMCI FONKSİYONLAR ===

def test_reset_function(morph):
    try:
        morph.reset()
    except Exception as e:
        pytest.fail(f"reset() hata verdi: {e}")
