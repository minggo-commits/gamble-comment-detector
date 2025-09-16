import pytest
from src.preprocessing import clean_text, preprocess_texts

def test_clean_text_lowercase():
    text = "HELLO WORLD"
    result = clean_text(text)
    assert result == "hello world"

def test_clean_text_remove_punctuation():
    text = "Hello!!!"
    result = clean_text(text)
    assert result == "hello"

def test_clean_text_remove_numbers():
    text = "Win 1000 now"
    result = clean_text(text)
    assert result == "win now"

def test_clean_text_remove_url():
    text = "Visit https://example.com now"
    result = clean_text(text)
    assert "http" not in result
    assert result == "visit now"

def test_preprocess_texts_multiple():
    texts = ["Hello WORLD!!!", "Free 1000 $$$", "Check http://abc.com"]
    result = preprocess_texts(texts)
    assert isinstance(result, list)
    assert all(isinstance(r, str) for r in result)
    assert result[0] == "hello world"
