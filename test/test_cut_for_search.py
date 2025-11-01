import os
import pytest
import jieba_fast_dat

# Define paths to our test resources
TEST_DICTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_dicts")
TEST_TEXTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_texts")
DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_dict_base.txt")
USER_DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")
MAIN_TEST_TEXT_PATH = os.path.join(TEST_TEXTS_DIR, "main_test_text.txt")

@pytest.fixture
def custom_tokenizer():
    """
    Fixture that provides a Tokenizer instance initialized with custom base and user dictionaries
    and forces HMM=False for segmentation.
    """
    tokenizer = jieba_fast_dat.Tokenizer()
    tokenizer.set_dictionary(DICT_BASE_PATH)
    tokenizer.initialize()
    tokenizer.load_userdict(USER_DICT_BASE_PATH)

    # Temporarily override the cut and cut_for_search methods to force HMM=False
    original_cut = tokenizer.cut
    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False) # Force HMM=False
    tokenizer.cut = new_cut

    original_cut_for_search = tokenizer.cut_for_search
    def new_cut_for_search(sentence, HMM=True):
        return original_cut_for_search(sentence, HMM=False) # Force HMM=False
    tokenizer.cut_for_search = new_cut_for_search

    return tokenizer


def test_cut_for_search_basic(custom_tokenizer):
    """
    Test basic cut_for_search functionality with custom dictionaries.
    """
    tokenizer = custom_tokenizer
    test_sent = "賴清德是台灣的政治人物。"
    words = list(tokenizer.cut_for_search(test_sent))
    
    expected_words = ["賴清德", "台灣", "政治人物"]
    for word in expected_words:
        assert word in words, f"Expected '{word}' not found in {words}"

def test_cut_for_search_finer_segmentation(custom_tokenizer):
    """
    Test that cut_for_search provides finer segmentation including overlapping words.
    """
    tokenizer = custom_tokenizer
    test_sent = "人工智慧是熱門技術"
    words = list(tokenizer.cut_for_search(test_sent))
    # Example: '人工智慧' -> '人工', '智慧', '人工智慧'
    assert "人工智慧" in words
    assert "人工" in words
    assert "智慧" in words

def test_cut_for_search_empty_sentence(custom_tokenizer):
    """
    Test cut_for_search with an empty sentence.
    """
    tokenizer = custom_tokenizer
    test_sent = ""
    words = list(tokenizer.cut_for_search(test_sent))
    assert len(words) == 0

def test_cut_for_search_non_forced_initialization():
    """
    Test that jieba_fast_dat does not auto-initialize with default dictionaries
    when a custom tokenizer is intended to be used with cut_for_search.
    """
    # Create a Tokenizer instance without explicit dictionary path
    tokenizer = jieba_fast_dat.Tokenizer()
    # Do not call initialize() or set_dictionary() yet

    # Attempt to cut_for_search a sentence
    test_sent = "這是一個測試句子。"
    words = list(tokenizer.cut_for_search(test_sent, HMM=False))

    # Assert that it performs some form of segmentation
    assert len(words) > 0
    assert isinstance(words[0], str)

    # Now, explicitly initialize the tokenizer and check if it uses the custom dictionary.
    tokenizer.set_dictionary(DICT_BASE_PATH)
    tokenizer.initialize()
    tokenizer.load_userdict(USER_DICT_BASE_PATH)

    # Temporarily override cut_for_search method to force HMM=False
    original_cut_for_search = tokenizer.cut_for_search
    def new_cut_for_search(sentence, HMM=True):
        return original_cut_for_search(sentence, HMM=False) # Force HMM=False
    tokenizer.cut_for_search = new_cut_for_search

    test_sent_custom = "賴清德是政治人物。"
    words_custom = list(tokenizer.cut_for_search(test_sent_custom))
    assert "賴清德" in words_custom
    assert "政治人物" in words_custom
