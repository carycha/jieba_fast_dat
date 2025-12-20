import os

import jieba as original_jieba
import jieba.posseg as original_jieba_posseg
import pytest

import jieba_fast_dat
import jieba_fast_dat.posseg


def test_load_userdict_cut(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test basic segmentation with a custom user dictionary,
    comparing with original jieba.
    """
    test_sent = "賴清德和柯文哲是台灣的政治人物。"
    fast_words = list(fast_tokenizer.cut(test_sent, HMM=False))
    orig_words = list(orig_tokenizer.cut(test_sent, HMM=False))

    assert fast_words == orig_words, (
        f"test_load_userdict_cut failed:\nFast: {fast_words}\nOrig: {orig_words}"
    )


def test_load_userdict_posseg_cut(
    fast_pos_tokenizer: jieba_fast_dat.posseg.POSTokenizer,
    orig_pos_tokenizer: original_jieba_posseg.POSTokenizer,
) -> None:
    """
    Test POS tagging with a custom user dictionary, comparing with original jieba.
    """
    test_sent = "賴清德和柯文哲是台灣的政治人物。"
    fast_words_with_flags = [
        (w.word, w.flag) for w in fast_pos_tokenizer.cut(test_sent, HMM=False)
    ]
    orig_words_with_flags = [
        (w.word, w.flag) for w in orig_pos_tokenizer.cut(test_sent, HMM=False)
    ]

    assert fast_words_with_flags == orig_words_with_flags, (
        "test_load_userdict_posseg_cut failed:\n"
        f"Fast: {fast_words_with_flags}\n"
        f"Orig: {orig_words_with_flags}"
    )


def test_add_word(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test adding a new word to the dictionary dynamically,
    comparing with original jieba.
    """
    new_word = "生成式AI"
    test_sent = f"這是一個關於{new_word}的討論。"

    # Before adding
    fast_words_before = list(fast_tokenizer.cut(test_sent, HMM=False))
    orig_words_before = list(orig_tokenizer.cut(test_sent, HMM=False))
    assert fast_words_before == orig_words_before, (
        "test_add_word (before) failed:\n"
        f"Fast: {fast_words_before}\n"
        f"Orig: {orig_words_before}"
    )

    fast_tokenizer.add_word(new_word, freq=10000, tag="n")
    orig_tokenizer.add_word(new_word, freq=10000, tag="n")

    # After adding
    fast_words_after = list(fast_tokenizer.cut(test_sent, HMM=False))
    orig_words_after = list(orig_tokenizer.cut(test_sent, HMM=False))
    assert fast_words_after == orig_words_after, (
        "test_add_word (after) failed:\n"
        f"Fast: {fast_words_after}\n"
        f"Orig: {orig_words_after}"
    )


def test_del_word(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test deleting a word from the dictionary dynamically,
    comparing with original jieba.
    """
    word_to_delete = "賴清德"
    test_sent = f"這是關於{word_to_delete}的報導。"

    # Ensure word is present initially
    fast_words_before = list(fast_tokenizer.cut(test_sent, HMM=False))
    orig_words_before = list(orig_tokenizer.cut(test_sent, HMM=False))
    assert fast_words_before == orig_words_before, (
        "test_del_word (before) failed:\n"
        f"Fast: {fast_words_before}\n"
        f"Orig: {orig_words_before}"
    )

    fast_tokenizer.del_word(word_to_delete)
    orig_tokenizer.del_word(word_to_delete)

    fast_words_after = list(fast_tokenizer.cut(test_sent, HMM=False))
    orig_words_after = list(orig_tokenizer.cut(test_sent, HMM=False))
    assert fast_words_after == orig_words_after, (
        "test_del_word (after) failed:\n"
        f"Fast: {fast_words_after}\n"
        f"Orig: {orig_words_after}"
    )


def test_suggest_freq(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test suggest_freq functionality, comparing with original jieba.
    """
    test_sent = "我們中出了叛徒"
    segment = ("中", "出")

    # Before tuning
    fast_words_before = list(fast_tokenizer.cut(test_sent, HMM=False))
    orig_words_before = list(orig_tokenizer.cut(test_sent, HMM=False))
    assert fast_words_before == orig_words_before, (
        "test_suggest_freq (before) failed:\n"
        f"Fast: {fast_words_before}\n"
        f"Orig: {orig_words_before}"
    )

    fast_tokenizer.suggest_freq(segment, tune=True)
    orig_tokenizer.suggest_freq(segment, tune=True)

    # After tuning
    fast_words_after = list(fast_tokenizer.cut(test_sent, HMM=False))
    orig_words_after = list(orig_tokenizer.cut(test_sent, HMM=False))
    assert fast_words_after == orig_words_after, (
        "test_suggest_freq (after) failed:\n"
        f"Fast: {fast_words_after}\n"
        f"Orig: {orig_words_after}"
    )


def test_non_forced_initialization_userdict(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test that a properly initialized tokenizer uses the user dictionary,
    comparing with original jieba.
    """
    test_sent_custom = "賴清德是政治人物。"
    fast_words_custom = list(fast_tokenizer.cut(test_sent_custom, HMM=False))
    orig_words_custom = list(orig_tokenizer.cut(test_sent_custom, HMM=False))
    assert fast_words_custom == orig_words_custom, (
        "test_non_forced_initialization_userdict failed:\n"
        f"Fast: {fast_words_custom}\n"
        f"Orig: {orig_words_custom}"
    )


def test_load_userdict_file_not_found(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Tests that loading a non-existent user dictionary raises FileNotFoundError
    for both jieba_fast_dat and original jieba.
    """
    non_existent_file = "path/to/a/file/that/does/not/exist.txt"

    with pytest.raises(RuntimeError):
        fast_tokenizer.load_userdict(non_existent_file)

    with pytest.raises(FileNotFoundError):
        orig_tokenizer.load_userdict(non_existent_file)


# Tests from test_change_dictpath.py
def test_set_dictionary_changes_behavior_on_instance(
    dict_base_path: str, dict_new_main_path: str
) -> None:
    """
    Tests that tokenizer.set_dictionary() correctly changes an instance's
    behavior, comparing with original jieba. This is a refactored test that
    avoids manipulating global state.
    """
    test_sent = "程式設計師正在研究元宇宙"

    # Create clean tokenizer instances for this test
    fast_tokenizer = jieba_fast_dat.Tokenizer()
    orig_tokenizer = original_jieba.Tokenizer()

    # 1. Initialize with the base dictionary.
    # In base_dict, "程式設計師" is a word, but "元宇宙" is not.
    fast_tokenizer.set_dictionary(dict_base_path)
    fast_tokenizer.initialize()
    orig_tokenizer.set_dictionary(dict_base_path)
    orig_tokenizer.initialize()

    fast_seg_list_base = list(fast_tokenizer.cut(test_sent))
    orig_seg_list_base = list(orig_tokenizer.cut(test_sent))
    assert fast_seg_list_base == orig_seg_list_base, (
        "test_set_dictionary (base dict) failed:\n"
        f"Fast: {fast_seg_list_base}\n"
        f"Orig: {orig_seg_list_base}"
    )

    # 2. Change to the 'add' dictionary.
    # In add_dict, "元宇宙" is a word, but "程式設計師" is not.
    fast_tokenizer.set_dictionary(dict_new_main_path)
    orig_tokenizer.set_dictionary(dict_new_main_path)

    fast_seg_list_add = list(fast_tokenizer.cut(test_sent))
    orig_seg_list_add = list(orig_tokenizer.cut(test_sent))
    assert fast_seg_list_add == orig_seg_list_add, (
        "test_set_dictionary (add dict) failed:\n"
        f"Fast: {fast_seg_list_add}\n"
        f"Orig: {orig_seg_list_add}"
    )

    # 3. No cleanup is needed because we only modified local instances.


def test_load_userdict_base_file_consistency() -> None:
    """
    Tests loading a user dictionary with varied formats (freq-only, tag-only, word-only)
    and ensures consistency between jieba_fast_dat and original jieba for POS tagging.
    """
    # 1. Define paths and test sentences
    current_dir = os.path.dirname(__file__)
    user_dict_path = os.path.join(current_dir, "test_dicts", "test_user_dict_base.txt")

    test_sentences = [
        "我是只有頻次的字",
        "我是只有詞性的字",
        "我是沒有頻次跟詞性的字",
    ]

    # 2. Create clean tokenizer instances
    fast_tokenizer = jieba_fast_dat.Tokenizer()
    fast_tokenizer.initialize()
    fast_pos_tokenizer = jieba_fast_dat.posseg.POSTokenizer(fast_tokenizer)

    orig_tokenizer = original_jieba.Tokenizer()
    orig_tokenizer.initialize()
    orig_pos_tokenizer = original_jieba_posseg.POSTokenizer(orig_tokenizer)

    # 3. Load the custom user dictionary
    fast_tokenizer.load_userdict(user_dict_path)
    orig_tokenizer.load_userdict(user_dict_path)

    # 4. Perform comparison
    for sentence in test_sentences:
        fast_words_with_flags = [
            (w.word, w.flag) for w in fast_pos_tokenizer.cut(sentence, HMM=False)
        ]
        orig_words_with_flags = [
            (w.word, w.flag) for w in orig_pos_tokenizer.cut(sentence, HMM=False)
        ]

        assert fast_words_with_flags == orig_words_with_flags, (
            f"Consistency failed for sentence: '{sentence}'\n"
            f"Fast: {fast_words_with_flags}\n"
            f"Orig: {orig_words_with_flags}"
        )
