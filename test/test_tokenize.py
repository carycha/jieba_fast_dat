import jieba as original_jieba

import jieba_fast_dat


def test_tokenize_default_mode_basic(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test tokenize in "default" mode with custom dictionaries,
    comparing with original jieba.
    """
    test_sent = "賴清德是台灣的政治人物。"
    fast_tokens = list(fast_tokenizer.tokenize(test_sent, mode="default", HMM=False))
    orig_tokens = list(orig_tokenizer.tokenize(test_sent, mode="default", HMM=False))

    assert fast_tokens == orig_tokens, (
        "test_tokenize_default_mode_basic failed:\n"
        f"Fast: {fast_tokens}\n"
        f"Orig: {orig_tokens}"
    )


def test_tokenize_search_mode_basic(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test tokenize in "search" mode with custom dictionaries,
    comparing with original jieba.
    """
    test_sent = "人工智慧是熱門技術"
    fast_tokens = list(fast_tokenizer.tokenize(test_sent, mode="search", HMM=False))
    orig_tokens = list(orig_tokenizer.tokenize(test_sent, mode="search", HMM=False))

    assert fast_tokens == orig_tokens, (
        "test_tokenize_search_mode_basic failed:\n"
        f"Fast: {fast_tokens}\n"
        f"Orig: {orig_tokens}"
    )


def test_tokenize_empty_sentence(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test tokenize functionality with an empty sentence,
    comparing with original jieba.
    """
    test_sent = ""
    fast_tokens = list(fast_tokenizer.tokenize(test_sent, mode="default", HMM=False))
    orig_tokens = list(orig_tokenizer.tokenize(test_sent, mode="default", HMM=False))
    assert fast_tokens == orig_tokens, (
        "test_tokenize_empty_sentence failed:\n"
        f"Fast: {fast_tokens}\n"
        f"Orig: {orig_tokens}"
    )


def test_non_forced_initialization_tokenize(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test that a properly initialized tokenizer uses the custom dictionary
    and compares with original jieba.
    """
    # These assignments are redundant as fixtures already provide initialized instances.
    # They are kept to reflect original test intent if it was to re-assign.
    fast_tokenizer = fast_tokenizer
    orig_tokenizer = orig_tokenizer

    test_sent_custom = "賴清德是政治人物。"
    fast_tokens_custom = list(
        fast_tokenizer.tokenize(test_sent_custom, mode="default", HMM=False)
    )
    orig_tokens_custom = list(
        orig_tokenizer.tokenize(test_sent_custom, mode="default", HMM=False)
    )

    assert fast_tokens_custom == orig_tokens_custom, (
        "test_non_forced_initialization_tokenize (initialized) failed:\n"
        f"Fast: {fast_tokens_custom}\n"
        f"Orig: {orig_tokens_custom}"
    )


# Tests from test_tokenize_no_hmm.py
def test_tokenize_no_hmm_default_mode_basic(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test tokenize in "default" mode with HMM=False and custom dictionaries,
    comparing with original jieba.
    """
    test_sent = "賴清德是台灣的政治人物。"
    fast_tokens = list(fast_tokenizer.tokenize(test_sent, mode="default", HMM=False))
    orig_tokens = list(orig_tokenizer.tokenize(test_sent, mode="default", HMM=False))

    assert fast_tokens == orig_tokens, (
        "test_tokenize_no_hmm_default_mode_basic failed:\n"
        f"Fast: {fast_tokens}\n"
        f"Orig: {orig_tokens}"
    )


def test_tokenize_no_hmm_search_mode_basic(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test tokenize in "search" mode with HMM=False and custom dictionaries,
    comparing with original jieba.
    """
    test_sent = "人工智慧是熱門技術"
    fast_tokens = list(fast_tokenizer.tokenize(test_sent, mode="search", HMM=False))
    orig_tokens = list(orig_tokenizer.tokenize(test_sent, mode="search", HMM=False))

    assert fast_tokens == orig_tokens, (
        "test_tokenize_no_hmm_search_mode_basic failed:\n"
        f"Fast: {fast_tokens}\n"
        f"Orig: {orig_tokens}"
    )


def test_tokenize_no_hmm_empty_sentence(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test tokenize functionality with HMM=False and an empty sentence,
    comparing with original jieba.
    """
    test_sent = ""
    fast_tokens = list(fast_tokenizer.tokenize(test_sent, mode="default", HMM=False))
    orig_tokens = list(orig_tokenizer.tokenize(test_sent, mode="default", HMM=False))
    assert fast_tokens == orig_tokens, (
        "test_tokenize_no_hmm_empty_sentence failed:\n"
        f"Fast: {fast_tokens}\n"
        f"Orig: {orig_tokens}"
    )


def test_non_forced_initialization_tokenize_no_hmm(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: original_jieba.Tokenizer
) -> None:
    """
    Test that a properly initialized tokenizer uses the custom dictionary,
    comparing with original jieba.
    """
    # These assignments are redundant as fixtures already provide initialized instances.
    # They are kept to reflect original test intent if it was to re-assign.
    fast_tokenizer = fast_tokenizer
    orig_tokenizer = orig_tokenizer

    test_sent_custom = "賴清德是政治人物。"
    fast_tokens_custom = list(
        fast_tokenizer.tokenize(test_sent_custom, mode="default", HMM=False)
    )
    orig_tokens_custom = list(
        orig_tokenizer.tokenize(test_sent_custom, mode="default", HMM=False)
    )
    assert fast_tokens_custom == orig_tokens_custom, (
        "test_non_forced_initialization_tokenize_no_hmm failed:\n"
        f"Fast: {fast_tokens_custom}\n"
        f"Orig: {orig_tokens_custom}"
    )
