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


def test_tokenize_no_hmm_default_mode_basic(custom_tokenizer_no_hmm):
    """
    Test tokenize in "default" mode with HMM=False and custom dictionaries.
    """
    tokenizer = custom_tokenizer_no_hmm
    test_sent = "賴清德是台灣的政治人物。"
    tokens = list(tokenizer.tokenize(test_sent, mode="default", HMM=False))
    
    expected_tokens = [
        ("賴清德", 0, 3),
        ("是", 3, 4),
        ("台灣", 4, 6),
        ("的", 6, 7),
        ("政治人物", 7, 11),
        ("。", 11, 12)
    ]
    assert tokens == expected_tokens

def test_tokenize_no_hmm_search_mode_basic(custom_tokenizer_no_hmm):
    """
    Test tokenize in "search" mode with HMM=False and custom dictionaries.
    """
    tokenizer = custom_tokenizer_no_hmm
    test_sent = "人工智慧是熱門技術"
    tokens = list(tokenizer.tokenize(test_sent, mode="search", HMM=False))
    
    # In search mode, it yields finer segmentation and overlapping words
    expected_tokens_present = [
        ("人工智慧", 0, 4),
        ("人工", 0, 2),
        ("智慧", 2, 4),
        ("是", 4, 5),
        ("熱門", 5, 7),
        ("技術", 7, 9)
    ]
    for expected_token in expected_tokens_present:
        assert expected_token in tokens, f"Expected {expected_token} not found in {tokens}"

def test_tokenize_no_hmm_empty_sentence(custom_tokenizer_no_hmm):
    """
    Test tokenize functionality with HMM=False and an empty sentence.
    """
    tokenizer = custom_tokenizer_no_hmm
    test_sent = ""
    tokens = list(tokenizer.tokenize(test_sent, mode="default", HMM=False))
    assert len(tokens) == 0

def test_non_forced_initialization_tokenize_no_hmm():
    """
    Test that jieba_fast_dat does not auto-initialize with default dictionaries
    when a custom tokenizer is intended to be used with tokenize and HMM=False.
    """
    # Create a Tokenizer instance without explicit dictionary path
    tokenizer = jieba_fast_dat.Tokenizer()
    # Do not call initialize() or set_dictionary() yet

    # Attempt to tokenize a sentence
    test_sent = "這是一個測試句子。"
    tokens = list(tokenizer.tokenize(test_sent, mode="default", HMM=False))

    # Assert that it performs some form of tokenization
    assert len(tokens) > 0
    assert isinstance(tokens[0][0], str)
    assert isinstance(tokens[0][1], int)
    assert isinstance(tokens[0][2], int)

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
    tokens_custom = list(tokenizer.tokenize(test_sent_custom, mode="default", HMM=False))
    assert ("賴清德", 0, 3) in tokens_custom
    assert ("政治人物", 4, 8) in tokens_custom
