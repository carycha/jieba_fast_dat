import os
import pytest
import jieba_fast_dat
import jieba_fast_dat.posseg

# Define paths to our test resources
TEST_DICTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_dicts")
TEST_TEXTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_texts")
DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_dict_base.txt")
USER_DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")
MAIN_TEST_TEXT_PATH = os.path.join(TEST_TEXTS_DIR, "main_test_text.txt")

@pytest.fixture
def custom_tokenizer_for_pos():
    """
    Fixture that provides a Tokenizer instance initialized with custom base and user dictionaries
    and forces HMM=False for segmentation.
    """
    tokenizer = jieba_fast_dat.Tokenizer()
    tokenizer.set_dictionary(DICT_BASE_PATH)
    tokenizer.initialize()
    tokenizer.load_userdict(USER_DICT_BASE_PATH)

    # Temporarily override the cut method to force HMM=False
    original_cut = tokenizer.cut
    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False) # Force HMM=False
    tokenizer.cut = new_cut

    return tokenizer

@pytest.fixture
def custom_posseg_tokenizer(custom_tokenizer_for_pos):
    """
    Fixture that provides a POSTokenizer instance using the custom_tokenizer_for_pos
    and forces HMM=False for segmentation.
    """
    posseg_tokenizer = jieba_fast_dat.posseg.POSTokenizer(tokenizer=custom_tokenizer_for_pos)
    # The dictionary and userdict are already loaded in custom_tokenizer_for_pos.
    
    # Temporarily override the cut method to force HMM=False for posseg
    original_cut = posseg_tokenizer.cut
    def new_cut(sentence, HMM=True):
        return original_cut(sentence, HMM=False) # Force HMM=False (posseg.cut does not have cut_all)
    posseg_tokenizer.cut = new_cut

    return posseg_tokenizer


def test_pos_tagging_basic(custom_posseg_tokenizer):
    """
    Test basic POS tagging with custom dictionaries.
    """
    posseg_tokenizer = custom_posseg_tokenizer
    test_sent = "賴清德是台灣的政治人物。"
    words_with_flags = list(posseg_tokenizer.cut(test_sent))
    
    expected_results = [
        ("賴清德", "nr"),
        ("是", "v"),
        ("台灣", "ns"),
        ("的", "uj"),
        ("政治人物", "n"),
        ("。", "x")
    ]
    for word, flag in expected_results:
        assert (word, flag) in [(w.word, w.flag) for w in words_with_flags], \
            f"Expected ('{word}', '{flag}') not found in {[(w.word, w.flag) for w in words_with_flags]}"
def test_pos_tagging_from_main_text(custom_posseg_tokenizer):
    """
    Test POS tagging with a more complex sentence from main_test_text.
    """
    posseg_tokenizer = custom_posseg_tokenizer
    with open(MAIN_TEST_TEXT_PATH, 'r', encoding='utf-8'):
        pass # Content is not used, so we just open and close the file
    
    # Take a snippet from the main text that includes words from base and user dicts
    test_sent = "台灣的台北是一個充滿活力的城市，這裡有許多電腦和手機的程式設計師。他們正在開發區塊鏈和人工智慧的應用。"
    words_with_flags = list(posseg_tokenizer.cut(test_sent))
    
    assert ("台灣", "ns") in [(w.word, w.flag) for w in words_with_flags]
    assert ("區塊鏈", "n") in [(w.word, w.flag) for w in words_with_flags]
    assert ("人工智慧", "n") in [(w.word, w.flag) for w in words_with_flags]
    assert ("電腦", "n") in [(w.word, w.flag) for w in words_with_flags]
    assert ("程式設計師", "n") in [(w.word, w.flag) for w in words_with_flags] # assuming this is tagged as noun
    assert ("充滿", "v") in [(w.word, w.flag) for w in words_with_flags]

def test_pos_tagging_empty_sentence(custom_posseg_tokenizer):
    """
    Test POS tagging with an empty sentence.
    """
    posseg_tokenizer = custom_posseg_tokenizer
    test_sent = ""
    words_with_flags = list(posseg_tokenizer.cut(test_sent))
    assert len(words_with_flags) == 0

def test_pos_tagging_non_forced_initialization():
    """
    Test that posseg does not auto-initialize with default dictionaries
    when a custom tokenizer is intended to be used.
    """
    # Create a Tokenizer instance without explicit dictionary path
    tokenizer_for_pos = jieba_fast_dat.Tokenizer()
    # Do not call initialize() or set_dictionary() yet

    # Create a POSTokenizer using this uninitialized Tokenizer
    posseg_tokenizer = jieba_fast_dat.posseg.POSTokenizer(tokenizer=tokenizer_for_pos)

    # Attempt to cut a sentence
    test_sent = "這是一個測試。"
    words_with_flags = list(posseg_tokenizer.cut(test_sent, HMM=False))

    # Assert that it performs some form of segmentation
    assert len(words_with_flags) > 0
    assert isinstance(words_with_flags[0].word, str)
    assert isinstance(words_with_flags[0].flag, str)

    # Now, explicitly initialize the tokenizer and check if it uses the custom dictionary.
    tokenizer_for_pos.set_dictionary(DICT_BASE_PATH)
    tokenizer_for_pos.initialize()
    tokenizer_for_pos.load_userdict(USER_DICT_BASE_PATH)

    # Temporarily override cut method for the tokenizer of posseg_tokenizer
    original_cut = posseg_tokenizer.tokenizer.cut
    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False) # Force HMM=False
    posseg_tokenizer.tokenizer.cut = new_cut

    test_sent_custom = "賴清德是政治人物。"
    words_custom = list(posseg_tokenizer.cut(test_sent_custom, HMM=False))
    assert ("賴清德", "nr") in [(w.word, w.flag) for w in words_custom]