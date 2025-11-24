import gc
import os

import psutil
import pytest

import jieba_fast_dat


def get_memory_rss() -> int:
    """Get current process's RSS memory in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


@pytest.mark.memory
def test_tokenizer_memory_leak(dict_base_path: str, main_test_text: str) -> None:
    """
    Tests for memory leaks in the Tokenizer by running it in a loop.
    A healthy system should not see significant, sustained memory growth.
    """
    # Warm-up: Run once to initialize any caches or global state
    tokenizer = jieba_fast_dat.Tokenizer(dictionary=dict_base_path)
    tokenizer.lcut(main_test_text)

    gc.collect()
    mem_before = get_memory_rss()

    # Run the core logic many times to amplify any potential leaks
    iterations = 1000
    for _ in range(iterations):
        # to correctly test its lifecycle.
        tokenizer = jieba_fast_dat.Tokenizer(dictionary=dict_base_path)
        tokenizer.lcut(main_test_text)

    gc.collect()
    mem_after = get_memory_rss()

    growth = mem_after - mem_before

    # Allow for a small, fixed amount of growth.
    # After many iterations, memory should be stable.
    # Threshold: 2 MB (a generous buffer)
    threshold = 2 * 1024 * 1024

    assert growth < threshold, (
        "Potential memory leak detected in Tokenizer!\n"
        f"Memory before loop: {mem_before / 1024:.2f} KB\n"
        f"Memory after loop:  {mem_after / 1024:.2f} KB\n"
        f"Growth over {iterations} iterations: {growth / 1024:.2f} KB "
        f"(Threshold: {threshold / 1024:.2f} KB)"
    )
