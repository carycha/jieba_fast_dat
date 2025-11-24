from collections.abc import Generator

import pytest

import jieba_fast_dat
import jieba_fast_dat.posseg as pseg

# A long text for testing parallel processing
LONG_TEXT = (
    "這是一個很長很長的句子，用來測試jieba_fast_dat在並行處理模式下的表現。"
    "我們希望確保在啟用多進程加速後，分詞的結果與單進程完全一致，沒有任何差異。"
    "這包括了常規分詞、搜索引擎模式分詞以及詞性標註。"
    "賴清德和柯文哲是台灣的政治人物，這句話也應該被正確地處理。"
    "此外，我們還需要測試英文和數字的混合情況，例如 Python3.9 和 COVID-19。"
) * 10  # Make the text even longer to give parallel processing a chance to work


@pytest.fixture
def parallel_tokenizer(
    dict_base_path: str, user_dict_base_path: str
) -> Generator[None, None, None]:
    """
    This fixture provides a tokenizer instance but does NOT initialize it.
    This is because the parallel processing mechanism pickles the main tokenizer,
    so we need to work with the global `jieba_fast_dat` module functions.
    """
    # Store original global state (temporarily commented out as get_dictionary/
    # get_userdict are not available)
    # original_dict_path = jieba_fast_dat.get_dictionary()
    # original_userdict_path = jieba_fast_dat.get_userdict()

    # Set up the global tokenizer for the purpose of this test module
    jieba_fast_dat.set_dictionary(dict_base_path)
    jieba_fast_dat.load_userdict(user_dict_base_path)
    jieba_fast_dat.initialize()

    yield

    # Teardown: Restore original global state (temporarily commented out)
    # jieba_fast_dat.set_dictionary(original_dict_path)
    # jieba_fast_dat.load_userdict(original_userdict_path)
    # jieba_fast_dat.initialize() # Re-initialize with original settings
    jieba_fast_dat.disable_parallel()  # Ensure parallel is disabled after test


def test_parallel_cut_correctness(parallel_tokenizer: None) -> None:
    """
    Tests that `cut` output is identical between single and parallel processing.
    """
    # 1. Run in single-process mode first
    result_single = list(jieba_fast_dat.cut(LONG_TEXT, HMM=False))

    # 2. Run in parallel mode
    try:
        jieba_fast_dat.enable_parallel(2)
        result_parallel = list(jieba_fast_dat.cut(LONG_TEXT, HMM=False))
    finally:
        jieba_fast_dat.disable_parallel()  # Ensure cleanup

    # 3. Compare results
    assert result_single == result_parallel, (
        "Parallel cut result differs from single process"
    )


def test_parallel_cut_for_search_correctness(parallel_tokenizer: None) -> None:
    """
    Tests that `cut_for_search` output is identical between single and parallel modes.
    """
    # 1. Run in single-process mode
    result_single = list(jieba_fast_dat.cut_for_search(LONG_TEXT, HMM=False))

    # 2. Run in parallel mode
    try:
        jieba_fast_dat.enable_parallel(2)
        result_parallel = list(jieba_fast_dat.cut_for_search(LONG_TEXT, HMM=False))
    finally:
        jieba_fast_dat.disable_parallel()

    # 3. Compare results
    assert result_single == result_parallel, "Parallel cut_for_search result differs"


def test_parallel_posseg_cut_correctness(parallel_tokenizer: None) -> None:
    """
    Tests that `posseg.cut` output is identical between single and parallel modes.
    """
    # 1. Run in single-process mode
    result_single = list(pseg.cut(LONG_TEXT, HMM=False))

    # 2. Run in parallel mode
    try:
        jieba_fast_dat.enable_parallel(2)
        result_parallel = list(pseg.cut(LONG_TEXT, HMM=False))
    finally:
        jieba_fast_dat.disable_parallel()

    # 3. Compare results
    assert result_single == result_parallel, "Parallel posseg.cut result differs"


def test_parallel_with_new_words(parallel_tokenizer: None) -> None:
    """
    Tests that words added after initialization are available to parallel workers.
    """
    jieba_fast_dat.add_word("並行處理模式", freq=1000, tag="n")

    # 1. Run in single-process mode
    result_single = list(jieba_fast_dat.cut(LONG_TEXT, HMM=False))
    assert "並行處理模式" in result_single

    # 2. Run in parallel mode
    try:
        jieba_fast_dat.enable_parallel(2)
        result_parallel = list(jieba_fast_dat.cut(LONG_TEXT, HMM=False))
    finally:
        jieba_fast_dat.disable_parallel()

    # 3. Compare results
    assert "並行處理模式" in result_parallel
    assert result_single == result_parallel
