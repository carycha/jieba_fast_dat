
import jieba_fast_dat
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_concurrent_initialization(dict_base_path):
    """
    Tests that concurrently initializing multiple Tokenizer instances is thread-safe.
    This replaces the old `test_lock.py` script.
    """
    num_threads = 10
    # Create multiple tokenizer instances that will all use the same cache file path
    tokenizers = [jieba_fast_dat.Tokenizer(dictionary=dict_base_path) for _ in range(num_threads)]

    def init_and_cut(tokenizer):
        tokenizer.initialize()
        # Verify it works after initialization
        return list(tokenizer.cut("這是一個測試"))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(init_and_cut, tk) for tk in tokenizers]
        
        results = []
        for future in as_completed(futures):
            # The test passes if no exceptions are raised during initialization.
            # We also collect results to ensure they are valid.
            results.append(future.result())
    
    assert len(results) == num_threads
    # All tokenizers should produce the same valid result
    expected_result = ['這是', '一個', '測試']
    for result in results:
        assert result == expected_result

def test_concurrent_cutting(tokenizer_base):
    """
    Tests that concurrently calling .cut() on a single shared Tokenizer 
    instance is thread-safe and produces consistent results.
    This replaces the old `test_multithread.py` script.
    """
    num_threads = 20
    test_sentence = "賴清德和柯文哲是台灣的政治人物。"
    expected_result = ['賴清德', '和', '柯文哲', '是', '台灣', '的', '政治人物', '。']

    def cut_sentence(_):
        # Each thread cuts the same sentence using the *same* shared tokenizer instance
        return list(tokenizer_base.cut(test_sentence, HMM=False))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(cut_sentence, i) for i in range(num_threads)]
        
        for future in as_completed(futures):
            result = future.result()
            # Assert that every thread gets the exact same, correct result.
            assert result == expected_result, "Concurrent cut produced inconsistent result"
