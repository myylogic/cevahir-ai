# tokenizer_management/tests/test_turkish_processor.py

import pytest
from tokenizer_management.preprocessing.turkish_processor import TurkishTextProcessor

@pytest.fixture
def processor():
    return TurkishTextProcessor()

def test_normalize_text_lowercase(processor):
    assert processor.normalize_text("MerHaBa") == "merhaba"

def test_normalize_text_unicode(processor):
    # fi ligatürü → "fi"
    text = "ﬁancé"
    normalized = processor.normalize_text(text)
    assert "fiance" in normalized

def test_normalize_text_strip_whitespace(processor):
    assert processor.normalize_text("  selam dünya  ") == "selam dünya"

def test_normalize_text_collapse_spaces(processor):
    assert processor.normalize_text("a   b\t c\n d") == "a b c d"

def test_remove_punctuation_simple(processor):
    assert processor.remove_punctuation("merhaba!") == "merhaba"

def test_remove_punctuation_multiple(processor):
    assert processor.remove_punctuation(".,!?;:") == ""

def test_remove_punctuation_inside_word(processor):
    assert processor.remove_punctuation("kardeş-im") == "kardeşim"

def test_remove_punctuation_keeps_unicode(processor):
    text = "çüşĞÜŞİ"
    assert processor.remove_punctuation(text) == text

def test_remove_stopwords_removes(processor):
    tokens = ["ve", "dünya", "bir", "selam"]
    assert processor.remove_stopwords(tokens) == ["dünya", "selam"]

def test_remove_stopwords_keeps_non_stopwords(processor):
    tokens = ["merhaba", "selam"]
    assert processor.remove_stopwords(tokens) == ["merhaba", "selam"]

def test_remove_stopwords_empty(processor):
    assert processor.remove_stopwords([]) == []

def test_tokenize_simple_sentence(processor):
    tokens = processor.tokenize("merhaba dünya")
    assert tokens == ["merhaba", "dünya"]

def test_tokenize_with_punctuation(processor):
    tokens = processor.tokenize("merhaba, dünya!")
    assert tokens == ["merhaba", "dünya"]

def test_tokenize_whitespace_variations(processor):
    tokens = processor.tokenize("  selam   dünya  ")
    assert tokens == ["selam", "dünya"]

def test_tokenize_stopwords_removed(processor):
    text = "merhaba ve dünya bir selam"
    assert processor.tokenize(text) == ["merhaba", "dünya", "selam"]

def test_tokenize_empty_string(processor):
    assert processor.tokenize("") == []

def test_tokenize_only_stopwords(processor):
    assert processor.tokenize("ve bir ama") == []

def test_tokenize_unicode_characters(processor):
    text = "ÇİĞDEM ŞİĞIL"
    tokens = processor.tokenize(text)
    assert tokens == ["çiğdem", "şiğil"]

def test_tokenize_numeric_tokens(processor):
    text = "1234 56"
    tokens = processor.tokenize(text)
    assert tokens == ["1234", "56"]

def test_tokenize_underscore(processor):
    text = "foo_bar baz"
    tokens = processor.tokenize(text)
    assert tokens == ["foo_bar", "baz"]

def test_tokenize_mixed_language(processor):
    text = "hello dünya"
    tokens = processor.tokenize(text)
    assert tokens == ["hello", "dünya"]

def test_tokenize_emoji_removed(processor):
    text = "selam 😊 dünya"
    tokens = processor.tokenize(text)
    assert tokens == ["selam", "dünya"]

def test_tokenize_turkish_chars(processor):
    text = "İstanbul Üsküdar"
    tokens = processor.tokenize(text)
    assert tokens == ["istanbul", "üsküdar"]

def test_tokenize_handles_tab_newline(processor):
    text = "merhaba\t\ndünya"
    tokens = processor.tokenize(text)
    assert tokens == ["merhaba", "dünya"]

def test_tokenize_handles_multiple_sentences(processor):
    text = "merhaba. dünya? selam!"
    tokens = processor.tokenize(text)
    assert tokens == ["merhaba", "dünya", "selam"]

def test_tokenize_combined_pipeline(processor):
    raw = "  Merhaba, ben bir     ChatGPT!  "
    tokens = processor.tokenize(raw)
    assert tokens == ["merhaba", "ben", "chatgpt"]

def test_normalize_and_tokenize_roundtrip(processor):
    raw = "ĞÜŞİ şüİç"
    normalized = processor.normalize_text(raw)
    tokens = processor.tokenize(normalized)
    assert tokens == ["ğüşi", "şüiç"]

def test_type_error_on_non_string(processor):
    with pytest.raises(AttributeError):
        # normalize_text will try string methods on None
        processor.tokenize(None)

def test_repeated_stopword_positions(processor):
    text = "ve ve merhaba ve"
    tokens = processor.tokenize(text)
    assert tokens == ["merhaba"]

def test_punctuation_preservation_in_between(processor):
    text = "a.b c"
    tokens = processor.tokenize(text)
    assert tokens == ["ab", "c"]

def test_strip_unicode_marks(processor):
    raw = "çiğdem"  # combining mark sequence
    normalized = processor.normalize_text(raw)
    assert "çiğdem" in normalized
