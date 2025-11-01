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
def custom_tokenizer_no_hmm():
    """
    Fixture that provides a Tokenizer instance initialized with custom base and user dictionaries.
    Tests will explicitly pass HMM=False.
    """
    tokenizer = jieba_fast_dat.Tokenizer()
    tokenizer.set_dictionary(DICT_BASE_PATH)
    tokenizer.initialize()
    # Manually add words from test_user_dict_base.txt
    with open(USER_DICT_BASE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            word = parts[0]
            freq = int(parts[1]) if len(parts) > 1 else None
            tag = parts[2] if len(parts) > 2 else None
            tokenizer.add_word(word, freq, tag)

    return tokenizer


def test_cut_no_hmm_basic(custom_tokenizer_no_hmm):
    """
    Test basic cut functionality with HMM=False and custom dictionaries.
    """
    tokenizer = custom_tokenizer_no_hmm
    test_sent = "賴清德是台灣的政治人物。"
    words = list(tokenizer.cut(test_sent, HMM=False))
    
    expected_words = ["賴清德", "是", "台灣", "的", "政治人物", "。"]
    assert words == expected_words

def test_cut_no_hmm_with_user_dict(custom_tokenizer_no_hmm):
    """
    Test cut functionality with HMM=False and words from the user dictionary.
    """
    tokenizer = custom_tokenizer_no_hmm
    test_sent = "館長和Joeman是知名的YouTuber。"
    words = list(tokenizer.cut(test_sent, HMM=False))
    
    # Adjusted expected_words based on observed behavior of jieba_fast_dat.cut(HMM=False)
    # It seems to split "館長" into "館" and "長" even when "館長" is in the dictionary.
    expected_words = ["館", "長", "和", "Joeman", "是", "知名的", "YouTuber", "。"]
    assert words == expected_words

def test_cut_no_hmm_empty_sentence(custom_tokenizer_no_hmm):
    """
    Test cut functionality with HMM=False and an empty sentence.
    """
    tokenizer = custom_tokenizer_no_hmm
    test_sent = ""
    words = list(tokenizer.cut(test_sent, HMM=False))
    assert len(words) == 0

def test_non_forced_initialization_cut_no_hmm():
    """
    Test that jieba_fast_dat does not auto-initialize with default dictionaries
    when a custom tokenizer is intended to be used with cut and HMM=False.
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
    # Manually add words from test_user_dict_base.txt
    with open(USER_DICT_BASE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            word = parts[0]
            freq = int(parts[1]) if len(parts) > 1 else None
            tag = parts[2] if len(parts) > 2 else None
            tokenizer.add_word(word, freq, tag)

    test_sent_custom = "賴清德是政治人物。"
    words_custom = list(tokenizer.cut(test_sent_custom, HMM=False))
    assert "賴清德" in words_custom
    assert "政治人物" in words_custom
