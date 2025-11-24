from typing import Any

import pytest

import jieba_fast_dat.posseg
from jieba_fast_dat import Tokenizer as FastTokenizer

# Check if original jieba is installed
original_jieba = None  # Initialize to None
try:
    import jieba as original_jieba

    original_jieba_installed = True
except ImportError:
    original_jieba_installed = False


@pytest.mark.skipif(not original_jieba_installed, reason="original jieba not installed")
def test_comparison_with_original_jieba(
    tmp_path: Any,  # noqa: ANN401
    dict_base_path: Any,  # noqa: ANN401
    user_dict_base_path: Any,  # noqa: ANN401
    main_test_text: str,
):
    assert original_jieba is not None, (
        "original jieba should be installed for this test"
    )
    """
    Test consistency between jieba_fast_dat and the original jieba.
    Uses fixtures for dictionary paths and test text.
    """
    sentence = main_test_text

    # 2. Initialize original jieba with custom dictionaries
    # Use a new Tokenizer to avoid global state modification
    tk_orig = original_jieba.Tokenizer(str(dict_base_path))
    tk_orig.load_userdict(str(user_dict_base_path))

    # 3. Initialize jieba_fast_dat with custom dictionaries
    # Use a new Tokenizer and a temporary cache directory
    tk_fast = FastTokenizer(str(dict_base_path))
    tk_fast.tmp_dir = str(tmp_path)  # type: ignore # pyright might complain about type mismatch, but it's assigned
    tk_fast.initialize()  # Force initialization
    tk_fast.load_userdict(str(user_dict_base_path))

    # 4. Compare results for different modes

    # Test cut (accurate mode)
    result_orig = list(tk_orig.cut(sentence))
    result_fast = list(tk_fast.cut(sentence))
    assert result_orig == result_fast, (
        f"Accurate mode failed:\nOriginal: {result_orig}\nFast:     {result_fast}"
    )

    # Test cut (cut_all=True)
    result_orig_all = list(tk_orig.cut(sentence, cut_all=True))
    result_fast_all = list(tk_fast.cut(sentence, cut_all=True))
    assert result_orig_all == result_fast_all, (
        f"Cut all mode failed:\nOriginal: {result_orig_all}\n"
        f"Fast:     {result_fast_all}"
    )

    # Test cut_for_search
    result_orig_search = list(tk_orig.cut_for_search(sentence))
    result_fast_search = list(tk_fast.cut_for_search(sentence))
    assert result_orig_search == result_fast_search, (
        f"Search mode failed:\nOriginal: {result_orig_search}\n"
        f"Fast:     {result_fast_search}"
    )

    # Test POS tagging
    # Original jieba's posseg is usually accessed via jieba.posseg
    # jieba_fast_dat's posseg is accessed via jieba_fast_dat.posseg
    # We need to ensure both are initialized with the same tokenizer.
    if original_jieba is not None and hasattr(
        original_jieba, "posseg"
    ):  # Add None check
        try:
            # Attempt to create a POSTokenizer for original jieba using the same
            # tokenizer instance. This might not be directly supported by older
            # jieba versions or its API.
            # If it fails, we'll fall back to using the global state for original
            # jieba's posseg.
            orig_pos_tokenizer = original_jieba.posseg.POSTokenizer(tokenizer=tk_orig)  # type: ignore
            fast_pos_tokenizer = jieba_fast_dat.posseg.POSTokenizer(tokenizer=tk_fast)

            pos_orig = list(orig_pos_tokenizer.cut(sentence))
            pos_fast = list(fast_pos_tokenizer.cut(sentence))
            assert pos_orig == pos_fast, (
                f"POS tagging failed (instance-based):\n"
                f"Original: {pos_orig}\nFast:     {pos_fast}"
            )
        except (AttributeError, TypeError):
            # Fallback: Use global state for original jieba's posseg
            # This requires original_jieba's global state to be set up with the same
            # dictionaries.
            # We already did tk_orig.set_dictionary and tk_orig.load_userdict,
            # which affects the global dt.
            original_jieba.set_dictionary(str(dict_base_path))
            original_jieba.load_userdict(str(user_dict_base_path))

            fast_pos_tokenizer = jieba_fast_dat.posseg.POSTokenizer(tokenizer=tk_fast)

            pos_orig = list(original_jieba.posseg.cut(sentence))  # type: ignore
            pos_fast = list(fast_pos_tokenizer.cut(sentence))
            assert pos_orig == pos_fast, (
                f"POS tagging failed (global-based):\n"
                f"Original: {pos_orig}\nFast:     {pos_fast}"
            )
