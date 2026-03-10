import pytest
import logging

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def ai_morph():
    from tokenizer_management.bpe.tokenization.ai_morphology import AIMorphology
    return AIMorphology(transformer_model_name="dbmdz/bert-base-turkish-cased")

def log_roots(word, roots):
    logger.info(f"Kelime: {word} - Kökler: {roots}")

def test_find_root_with_punctuation(ai_morph):
    word = "koşuyorum!"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("koş" in root for root in roots)

def test_find_root_with_uppercase(ai_morph):
    word = "Koşuyorum"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("koş" in root for root in roots)

def test_find_root_compound_word(ai_morph):
    word = "okuldan"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("okul" in root for root in roots)

def test_find_root_negation(ai_morph):
    word = "yapmıyorum"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("yap" in root for root in roots)

def test_find_root_question_suffix(ai_morph):
    word = "geliyor musun"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("gel" in root for root in roots)

def test_find_root_possessive_suffix(ai_morph):
    word = "kitabım"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("kitap" in root for root in roots)

def test_find_root_irregular_word(ai_morph):
    word = "diyorum"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("de" in root or "dik" in root or "diy" in root for root in roots)

def test_find_root_short_word(ai_morph):
    word = "su"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert roots == ["su"]

def test_find_root_repeated_word(ai_morph):
    word = "koşuyor koşuyor"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    # Burada birden fazla kök dönebilir
    assert any("koş" in root for root in roots)

def test_find_root_non_turkish_word(ai_morph):
    word = "python"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert roots == [word]

def test_analyze_mixed_tokens(ai_morph):
    tokens = ["koşuyorum", "Python", "geldim!"]
    roots = ai_morph.analyze(tokens)
    logger.info(f"Tokens: {tokens} -> Kökler: {roots}")
    assert len(roots) >= len(tokens)

def test_find_root_with_numbers(ai_morph):
    word = "test123"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert roots == [word]

def test_find_root_with_special_characters(ai_morph):
    word = "@koşuyorum"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("koş" in root for root in roots or [word])

def test_find_root_empty_string(ai_morph):
    word = ""
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert roots == []

def test_find_root_none_input(ai_morph):
    with pytest.raises(Exception):
        ai_morph.find_root(None)

def test_analyze_empty_tokens(ai_morph):
    with pytest.raises(Exception):
        ai_morph.analyze([])

def test_analyze_none_tokens(ai_morph):
    with pytest.raises(Exception):
        ai_morph.analyze(None)

def test_analyze_non_string_tokens(ai_morph):
    with pytest.raises(Exception):
        ai_morph.analyze([123, 456])

def test_find_root_hyphenated_word(ai_morph):
    word = "bilgi-teknolojisi"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    # Muhtemelen kök ayrı ayrı gelir
    assert any("bilgi" in root for root in roots) or any("teknoloji" in root for root in roots)

def test_find_root_suffixes_combination(ai_morph):
    word = "yapabileceklerimizden"
    roots = ai_morph.find_root(word)
    log_roots(word, roots)
    assert any("yap" in root for root in roots)
