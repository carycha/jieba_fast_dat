import os
import tempfile

import pytest

from jieba_fast_dat import DatTrie
from test.test_utils import _parse_dict_line


def test_dat_trie_build_and_search() -> None:
    trie = DatTrie()
    word_freqs = [
        ("你好", 100),
        ("世界", 200),
        ("你好世界", 300),
        ("Python", 50),
    ]
    trie.build(word_freqs)

    assert trie.search("你好") == 100
    assert trie.search("世界") == 200
    assert trie.search("你好世界") == 300
    assert trie.search("Python") == 50
    assert trie.search("不存在") == -1
    assert trie.search("你好世") == -1  # Partial match should not return a value


def test_dat_trie_empty() -> None:
    trie = DatTrie()
    word_freqs: list[tuple[str, int]] = []
    trie.build(word_freqs)
    assert trie.search("任何詞") == -1


def test_dat_trie_save_and_open() -> None:
    trie = DatTrie()
    word_freqs = [
        ("測試", 10),
        ("保存", 20),
        ("載入", 30),
    ]
    trie.build(word_freqs)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_trie.dat")
        trie.save(filepath)

        new_trie = DatTrie()
        new_trie.open(filepath)

        assert new_trie.search("測試") == 10
        assert new_trie.search("保存") == 20
        assert new_trie.search("載入") == 30
        assert new_trie.search("不存在") == -1


def test_dat_trie_overwrite() -> None:
    trie = DatTrie()
    word_freqs1 = [("舊詞", 10)]
    trie.build(word_freqs1)
    assert trie.search("舊詞") == 10

    word_freqs2 = [("新詞", 20)]
    trie.build(word_freqs2)  # Building again should overwrite
    assert trie.search("舊詞") == -1
    assert trie.search("新詞") == 20


def test_dat_trie_unicode_words() -> None:
    trie = DatTrie()
    word_freqs = [
        ("你好，世界！", 100),
        ("編程語言", 200),
        ("C++", 50),
        ("Python3.9", 60),
    ]
    trie.build(word_freqs)

    assert trie.search("你好，世界！") == 100
    assert trie.search("編程語言") == 200
    assert trie.search("C++") == 50
    assert trie.search("Python3.9") == 60
    assert trie.search("不存在的詞") == -1


def test_parse_dict_line() -> None:
    # Test case 1: word, freq, tag
    word, freq, tag = _parse_dict_line("創新辦 3 i")
    assert word == "創新辦"
    assert freq == 3
    assert tag == "i"

    # Test case 2: word, freq
    word, freq, tag = _parse_dict_line("云计算 5")
    assert word == "云计算"
    assert freq == 5
    assert tag is None

    # Test case 3: word, tag (freq defaults to 1)
    word, freq, tag = _parse_dict_line("凱特琳 nz")
    assert word == "凱特琳"
    assert freq == 1
    assert tag == "nz"

    # Test case 4: word only (freq defaults to 1, tag is None)
    word, freq, tag = _parse_dict_line("台中")
    assert word == "台中"
    assert freq == 1
    assert tag is None

    # Test case 5: empty line
    # with pytest.raises(ValueError):
    #     _parse_dict_line("")

    # Test case 7: line with leading/trailing spaces (should be invalid due to multiple
    # spaces between word and freq)
    # with pytest.raises(ValueError):
    #     _parse_dict_line("  word  10  x  ")

    # Test case 8: line with only spaces
    with pytest.raises(ValueError):
        _parse_dict_line("   ")


# Tests from test_dat_trie_extended.py
@pytest.fixture
def extended_trie() -> DatTrie:
    """Provides a DatTrie instance pre-built with some words."""
    dt = DatTrie()
    word_freqs = [
        ("你好", 10),
        ("世界", 20),
        ("你好世界", 5),
        ("應用", 15),
        ("應用程式", 8),
    ]
    dt.build(word_freqs)
    return dt


@pytest.mark.skip(reason="DatTrie methods update_word is not implemented in C++ yet.")
def test_update_word(extended_trie: DatTrie) -> None:
    dt = extended_trie
    # Test adding a new word
    assert dt.search("新詞") == -1
    dt.update_word("新詞", 50)
    assert dt.search("新詞") == 50

    # Test updating an existing word's frequency
    assert dt.search("你好") == 10
    dt.update_word("你好", 100)
    assert dt.search("你好") == 100


@pytest.mark.skip(reason="DatTrie methods erase_word is not implemented in C++ yet.")
def test_erase_word(extended_trie: DatTrie) -> None:
    dt = extended_trie
    assert dt.search("世界") != -1
    dt.erase_word("世界")
    assert dt.search("世界") == -1

    # Test erasing a non-existent word
    result = dt.erase_word("不存在的詞")
    assert result == -1  # erase should return -1 on failure


@pytest.mark.skip(
    reason="DatTrie methods num_keys and capacity are not implemented in C++ yet."
)
def test_size_and_capacity(extended_trie: DatTrie) -> None:
    dt = extended_trie
    initial_keys = dt.num_keys()
    initial_capacity = dt.capacity()
    assert initial_keys > 0
    assert initial_capacity >= dt.num_keys()

    dt.update_word("另一個詞", 1)
    assert dt.num_keys() > initial_keys


@pytest.mark.skip(
    reason="The iter() method is not exposed in the pybind interface yet."
)
def test_iter(extended_trie: DatTrie) -> None:
    dt = extended_trie
    trie_content = dict(dt.iter())
    word_freqs = [
        ("你好", 10),
        ("世界", 20),
        ("你好世界", 5),
        ("應用", 15),
        ("應用程式", 8),
    ]

    # Check if all original words are in the iterator output
    for word, freq in word_freqs:
        assert word in trie_content
        assert trie_content[word] == freq

    # Verify the number of items
    assert len(trie_content) == len(word_freqs)


@pytest.mark.skip(
    reason="DatTrie methods common_prefix_predict is not implemented in C++ yet."
)
def test_common_prefix_predict(extended_trie: DatTrie) -> None:
    dt = extended_trie
    # Test prediction
    predictions = dt.common_prefix_predict("應用")
    print(f"DEBUG: Predictions for '應用': {predictions}")
    predictions_dict = dict(predictions)

    assert "應用" in predictions_dict
    assert predictions_dict["應用"] == 15
    assert "應用程式" in predictions_dict
    assert predictions_dict["應用程式"] == 8

    # Test with a prefix that is not a word itself
    predictions_ni = dt.common_prefix_predict("你")
    predictions_ni_dict = dict(predictions_ni)
    assert "你好" in predictions_ni_dict
    assert "你好世界" in predictions_ni_dict

    # Test with no possible predictions
    no_predictions = dt.common_prefix_predict("無此開頭")
    assert len(no_predictions) == 0
