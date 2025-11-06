import os
import tempfile
from jieba_fast_dat import DatTrie


def test_dat_trie_build_and_search():
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


def test_dat_trie_empty():
    trie = DatTrie()
    word_freqs = []
    trie.build(word_freqs)
    assert trie.search("任何詞") == -1


def test_dat_trie_save_and_open():
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


def test_dat_trie_overwrite():
    trie = DatTrie()
    word_freqs1 = [("舊詞", 10)]
    trie.build(word_freqs1)
    assert trie.search("舊詞") == 10

    word_freqs2 = [("新詞", 20)]
    trie.build(word_freqs2)  # Building again should overwrite
    assert trie.search("舊詞") == -1
    assert trie.search("新詞") == 20


def test_dat_trie_unicode_words():
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
