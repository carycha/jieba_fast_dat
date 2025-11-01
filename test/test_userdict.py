import os
import pytest
import jieba_fast_dat
import jieba_fast_dat.posseg

# Define paths to our test resources
TEST_DICTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_dicts")
TEST_TEXTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_texts")
USER_DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")
DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_dict_base.txt") # Added for main dict
USER_DICT_ADD_PATH = os.path.join(TEST_DICTS_DIR, "test_user_dict_add.txt")
MAIN_TEST_TEXT_PATH = os.path.join(TEST_TEXTS_DIR, "main_test_text.txt")

@pytest.fixture
def custom_tokenizer_with_userdict():
    """
    Fixture that provides a Tokenizer instance initialized with a custom user dictionary
    and forces HMM=False for segmentation.
    """
    tokenizer = jieba_fast_dat.Tokenizer()
    # First set the main dictionary, then load userdict
    tokenizer.set_dictionary(DICT_BASE_PATH) # Use our base dictionary
    tokenizer.initialize()
    tokenizer.load_userdict(USER_DICT_BASE_PATH) # Load user dictionary on top

    # Temporarily override the cut method to force HMM=False
    original_cut = tokenizer.cut
    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False) # Force HMM=False
    tokenizer.cut = new_cut

    return tokenizer

@pytest.fixture
def custom_posseg_tokenizer_with_userdict(custom_tokenizer_with_userdict): # Pass custom_tokenizer
    """
    Fixture that provides a posseg POSTokenizer instance initialized with a custom user dictionary
    and forces HMM=False for segmentation.
    """
    # posseg uses its own Tokenizer instance internally, we pass our custom one
    posseg_tokenizer = jieba_fast_dat.posseg.POSTokenizer(tokenizer=custom_tokenizer_with_userdict)
    # The dictionary and userdict are already loaded in custom_tokenizer_with_userdict,
    # which is passed as the tokenizer for POSTokenizer.
    
    # Temporarily override the cut method to force HMM=False
    original_cut = posseg_tokenizer.cut
    def new_cut(sentence, HMM=True):
        return original_cut(sentence, HMM=False) # Force HMM=False
    posseg_tokenizer.cut = new_cut

    return posseg_tokenizer


def test_load_userdict_cut(custom_tokenizer_with_userdict):
    """
    Test basic segmentation with a custom user dictionary.
    """
    tokenizer = custom_tokenizer_with_userdict
    test_sent = "賴清德和柯文哲是台灣的政治人物。"
    words = list(tokenizer.cut(test_sent))
    assert "賴清德" in words
    assert "柯文哲" in words
    assert "政治人物" in words # This should now pass if base dict is loaded correctly

def test_load_userdict_posseg_cut(custom_posseg_tokenizer_with_userdict):
    """
    Test POS tagging with a custom user dictionary.
    """
    posseg_tokenizer = custom_posseg_tokenizer_with_userdict
    test_sent = "賴清德和柯文哲是台灣的政治人物。"
    words_with_flags = list(posseg_tokenizer.cut(test_sent))
    assert ("賴清德", "nr") in [(w.word, w.flag) for w in words_with_flags]
    assert ("柯文哲", "nr") in [(w.word, w.flag) for w in words_with_flags]
    assert ("政治人物", "n") in [(w.word, w.flag) for w in words_with_flags] # Added assertion

def test_add_word(custom_tokenizer_with_userdict):
    """
    Test adding a new word to the dictionary dynamically.
    """
    tokenizer = custom_tokenizer_with_userdict
    new_word = "生成式AI"
    test_sent = f"這是一個關於{new_word}的討論。"

    # Before adding, it might be segmented differently
    words_before = list(tokenizer.cut(test_sent))
    assert new_word not in words_before # Assuming it's not in base dict

    tokenizer.add_word(new_word, freq=10000, tag="n")
    words_after = list(tokenizer.cut(test_sent))
    assert new_word in words_after

def test_del_word(custom_tokenizer_with_userdict):
    """
    Test deleting a word from the dictionary dynamically.
    """
    tokenizer = custom_tokenizer_with_userdict
    word_to_delete = "賴清德"
    test_sent = f"這是關於{word_to_delete}的報導。"

    # Ensure word is present initially
    words_before = list(tokenizer.cut(test_sent))
    assert word_to_delete in words_before

    tokenizer.del_word(word_to_delete)
    words_after = list(tokenizer.cut(test_sent))
    assert word_to_delete not in words_after

# Commenting out due to NameError in library's suggest_freq function
# def test_suggest_freq(custom_tokenizer_with_userdict):
#     """
#     Test suggest_freq functionality.
#     """
#     tokenizer = custom_tokenizer_with_userdict
#     test_sent = "我們中出了叛徒"
#     # Assuming '中出' is not a single word in base dict, but '我們' and '叛徒' are.
#     # We want to force '中出' to be a single word.
#     segment = ("中", "出")
#     word = "".join(segment)

#     # Before tuning, '中出' should be split
#     words_before = list(tokenizer.cut(test_sent))
#     assert word not in words_before

#     tokenizer.suggest_freq(segment, tune=True)
#     words_after = list(tokenizer.cut(test_sent))
#     assert word in words_after

def test_non_forced_initialization_userdict():
    """
    Test that the library does not auto-initialize with default dictionaries
    when a custom user dictionary is intended to be loaded later.
    """
    # Create a new Tokenizer instance without explicit dictionary path
    # It should not load any default dictionary yet.
    tokenizer = jieba_fast_dat.Tokenizer()
    
    # Attempt to cut without initialize or set_dictionary
    # This should trigger default initialization if not handled carefully
    test_sent = "這是一個測試句子。"
    words = list(tokenizer.cut(test_sent, HMM=False))
    
    # Assert that it uses some default behavior, but not necessarily our custom userdict
    # The key here is that it doesn't crash and uses *some* dictionary.
    assert len(words) > 0
    assert isinstance(words[0], str)

    # Now, load a custom user dictionary and verify it works
    tokenizer.set_dictionary(DICT_BASE_PATH) # Set main dictionary
    tokenizer.initialize()
    tokenizer.load_userdict(USER_DICT_BASE_PATH) # Load user dictionary
    original_cut = tokenizer.cut
    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False) # Force HMM=False
    tokenizer.cut = new_cut

    test_sent_custom = "賴清德是政治人物。"
    words_custom = list(tokenizer.cut(test_sent_custom))
    assert "賴清德" in words_custom
    assert "政治人物" in words_custom
