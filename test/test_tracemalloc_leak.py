import pytest
import _jieba_fast_dat_functions_py3 as c_funcs


# To use the detailed memory leak check, add this mark to the test function.
# The fixture 'tracemalloc_leaks' is defined in test/conftest.py
@pytest.mark.usefixtures("tracemalloc_leaks")
def test_dat_trie_build_with_tracemalloc():
    """
    This test checks for memory leaks during the DatTrie build process using tracemalloc.
    It repeats the build process multiple times to amplify any potential leaks.
    The 'tracemalloc_leaks' fixture will automatically fail the test if memory
    growth exceeds the threshold defined in conftest.py.
    """
    for _ in range(100):  # Repeat build multiple times
        trie = c_funcs.DatTrie()
        word_freqs = [("word" + str(i), i) for i in range(100)]
        trie.build(word_freqs)
        # In each loop, 'trie' and 'word_freqs' go out of scope and should be garbage collected.


@pytest.mark.usefixtures("tracemalloc_leaks")
def test_segmentation_with_tracemalloc(tokenizer_base):
    """
    Checks for memory leaks during word segmentation.
    """
    sentence = "這我們的民主成果得來不易，是兩千三百萬人堅持用選票寫下的自我證明。是我們的島，也是我們的家。"
    for _ in range(500):
        list(tokenizer_base.cut(sentence, HMM=True))


@pytest.mark.usefixtures("tracemalloc_leaks")
def test_pos_tagging_with_tracemalloc(pos_tokenizer):
    """
    Checks for memory leaks during Part-of-Speech tagging.
    """
    sentence = "每一次國際上的發聲，都代表著我們作為一個獨立社群的清晰意志與存在。"
    for _ in range(500):
        list(pos_tokenizer.cut(sentence, HMM=True))
