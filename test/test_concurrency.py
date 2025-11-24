from concurrent.futures import ThreadPoolExecutor, as_completed

import jieba  # ADDED

import jieba_fast_dat


def test_concurrent_initialization(dict_base_path: str) -> None:
    """
    Tests that concurrently initializing multiple Tokenizer instances is thread-safe.
    This replaces the old `test_lock.py` script.
    """
    num_threads = 10
    # Create multiple tokenizer instances that will all use the same cache file path
    fast_tokenizers = [
        jieba_fast_dat.Tokenizer(dictionary=dict_base_path) for _ in range(num_threads)
    ]
    orig_tokenizers = [
        jieba.Tokenizer(dictionary=dict_base_path) for _ in range(num_threads)
    ]

    def init_and_cut(
        tokenizer: jieba_fast_dat.Tokenizer | jieba.Tokenizer,
    ) -> list[str]:
        tokenizer.initialize()
        # Verify it works after initialization
        return list(tokenizer.cut("這是一個測試"))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        fast_futures = [executor.submit(init_and_cut, tk) for tk in fast_tokenizers]
        orig_futures = [executor.submit(init_and_cut, tk) for tk in orig_tokenizers]

        fast_results = []
        for future in as_completed(fast_futures):
            # The test passes if no exceptions are raised during initialization.
            # We also collect results to ensure they are valid.
            fast_results.append(future.result())

        orig_results = []
        for future in as_completed(orig_futures):
            orig_results.append(future.result())

    assert len(fast_results) == num_threads
    assert len(orig_results) == num_threads

    # Compare fast_results with orig_results
    assert fast_results == orig_results, (
        "Concurrent initialization produced inconsistent results.\n"
        f"Fast: {fast_results}\n"
        f"Orig: {orig_results}"
    )


def test_concurrent_cutting(
    fast_tokenizer: jieba_fast_dat.Tokenizer, orig_tokenizer: jieba.Tokenizer
) -> None:
    """
    Tests that concurrently calling .cut() on a single shared Tokenizer
    instance is thread-safe and produces consistent results.
    This replaces the old `test_multithread.py` script.
    """
    num_threads = 20
    test_sentence = "賴清德和柯文哲是台灣的政治人物。"

    def cut_sentence_fast(_: object) -> list[str]:
        return list(fast_tokenizer.cut(test_sentence, HMM=False))

    def cut_sentence_orig(_: object) -> list[str]:
        return list(orig_tokenizer.cut(test_sentence, HMM=False))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        fast_futures = [
            executor.submit(cut_sentence_fast, i) for i in range(num_threads)
        ]
        orig_futures = [
            executor.submit(cut_sentence_orig, i) for i in range(num_threads)
        ]

        fast_results = []
        for future in as_completed(fast_futures):
            fast_results.append(future.result())

        orig_results = []
        for future in as_completed(orig_futures):
            orig_results.append(future.result())

    # All tokenizers should produce the same valid result
    # We will compare fast_results with orig_results directly
    assert fast_results == orig_results, (
        "Concurrent cutting produced inconsistent results.\n"
        f"Fast: {fast_results}\n"
        f"Orig: {orig_results}"
    )
