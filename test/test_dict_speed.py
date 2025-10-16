import time
import os
import pytest
import shutil
import jieba_fast_dat as jieba

# Assuming the default dictionary is used for speed testing
# We'll use the actual dict.txt from the package for this test
# To ensure we're testing the real loading mechanism.

def test_dictionary_loading_speed():
    # Path to the large dictionary
    big_dict_path = os.path.join(os.path.dirname(__file__), "..", "extra_dict", "dict.txt.big")
    big_dict_path = os.path.abspath(big_dict_path)

    # Ensure cache directory is clean before starting the test
    cache_file_path_initial = jieba.get_cache_file_path(big_dict_path)
    cache_dir = os.path.dirname(cache_file_path_initial)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # First initialization - should build the DAT and save the cache for dict.txt.big
    jieba.dt = jieba.Tokenizer() # Ensure a fresh tokenizer instance
    start_time = time.time()
    jieba.dt.initialize(big_dict_path)
    first_load_time = time.time() - start_time
    print(f"First load time (dict.txt.big): {first_load_time:.4f} seconds")
    # dict.txt.big is large, so initial load might take several seconds
    assert first_load_time < 10.0, f"First load of dict.txt.big took too long: {first_load_time:.4f}s"

    # Second initialization - should be very fast due to caching
    jieba.dt = jieba.Tokenizer() # Fresh tokenizer
    start_time = time.time()
    jieba.dt.initialize(big_dict_path)
    second_load_time = time.time() - start_time
    print(f"Second load time (dict.txt.big with cache): {second_load_time:.4f} seconds")
    assert second_load_time < 1.0, f"Second load of dict.txt.big took too long: {second_load_time:.4f}s"
    assert second_load_time < first_load_time, "Second load was not faster than the first load, caching might not be effective."
    assert first_load_time / second_load_time > 3, "Speedup from caching for dict.txt.big is not significant enough."

    # Third initialization - no change, should be instant and similar to second load
    jieba.dt = jieba.Tokenizer() # Fresh tokenizer
    start_time = time.time()
    jieba.dt.initialize(big_dict_path)
    third_load_time = time.time() - start_time
    print(f"Third load time (dict.txt.big no change): {third_load_time:.4f} seconds")
    assert third_load_time < 0.4, f"Third load of dict.txt.big took too long: {third_load_time:.4f}s"
    assert third_load_time <= second_load_time * 1.1, "Third load was significantly slower than the second load, caching might be unstable."

    # Clean up
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
