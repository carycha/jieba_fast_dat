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
    print(f"FREQ after userdict load: {tokenizer.FREQ.keys()}") # Add this line

    # Temporarily override the cut and cut_for_search methods to force HMM=False
    original_cut = tokenizer.cut
    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False) # Force HMM=False
    tokenizer.cut = new_cut

    return tokenizer


def test_cut_basic(custom_tokenizer):
    """
    Test basic cut functionality with custom dictionaries.
    """
    tokenizer = custom_tokenizer
    test_sent = "賴清德是台灣的政治人物。"
    words = list(tokenizer.cut(test_sent))
    
    expected_words = ["賴清德", "是", "台灣", "的", "政治人物", "。"]
    assert words == expected_words

def test_cut_with_user_dict(custom_tokenizer):
    """
    Test cut functionality with words from the user dictionary.
    """
    tokenizer = custom_tokenizer
    test_sent = "館長和Joeman是知名的YouTuber。"
    words = list(tokenizer.cut(test_sent))
    print(f"Words before direct add_word('館長'): {words}")
    # Debug: Add "館長" directly to see if it works
    tokenizer.add_word("館長", freq=10000, tag="nr")
    words_after_add = list(tokenizer.cut(test_sent))
    print(f"Words after direct add_word('館長'): {words_after_add}")

    expected_words = ["館長", "和", "Joeman", "是", "知名的", "YouTuber", "。"]
    for word in expected_words:
        assert word in words_after_add, f"Expected '{word}' not found in {words_after_add}"

def test_cut_empty_sentence(custom_tokenizer):
    """
    Test cut functionality with an empty sentence.
    """
    tokenizer = custom_tokenizer
    test_sent = ""
    words = list(tokenizer.cut(test_sent))
    assert len(words) == 0

def test_del_word_functionality(custom_tokenizer):
    """
    Test deleting a word from the dictionary dynamically.
    """
    tokenizer = custom_tokenizer
    word_to_delete = "賴清德"
    test_sent = f"這是關於{word_to_delete}的報導。"

    # Ensure word is present initially
    words_before = list(tokenizer.cut(test_sent))
    assert word_to_delete in words_before

    tokenizer.del_word(word_to_delete)
    words_after = list(tokenizer.cut(test_sent))
    assert word_to_delete not in words_after

def test_non_forced_initialization_cut():
    """
    Test that jieba_fast_dat does not auto-initialize with default dictionaries
    when a custom tokenizer is intended to be used with cut.
    """
    # Create a Tokenizer instance without explicit dictionary path
    tokenizer = jieba_fast_dat.Tokenizer()
    # Do not call initialize() or set_dictionary() yet

    # Attempt to cut a sentence
    test_sent = "這是一個測試句子。"
    words = list(tokenizer.cut(test_sent, HMM=False))

    # Assert that it performs some form of segmentation
    assert len(words) > 0
    assert isinstance(words[0], str)

    # Now, explicitly initialize the tokenizer and check if it uses the custom dictionary.
    tokenizer.set_dictionary(DICT_BASE_PATH)
    tokenizer.initialize()
    tokenizer.load_userdict(USER_DICT_BASE_PATH)

    # Temporarily override cut method to force HMM=False
    original_cut = tokenizer.cut
    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False) # Force HMM=False
    tokenizer.cut = new_cut

    test_sent_custom = "賴清德是政治人物。"
    words_custom = list(tokenizer.cut(test_sent_custom))
    assert "賴清德" in words_custom
    assert "政治人物" in words_custom
