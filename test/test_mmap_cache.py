import os
import time
import tempfile
import shutil
import pytest
from unittest.mock import patch
import logging


import jieba_fast_dat as jieba

# Helper function to create a temporary dictionary file
def create_temp_dict(content):
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
        tmp.write(content)
    return path

def test_mmap_cache_behavior():
    # Use the default dictionary for performance testing
    default_dict_path = jieba.get_module_res_path(jieba.DEFAULT_DICT_NAME)
    cache_file_path_default = jieba.get_cache_file_path(default_dict_path)
    cache_dir = os.path.dirname(cache_file_path_default)

    # Ensure cache directory is clean before starting the test
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # --- Test 1: First load of default dict (should build DAT and save cache) ---
    jieba.dt = jieba.Tokenizer() # Ensure a fresh tokenizer instance
    start_time = time.time()
    jieba.dt.initialize(default_dict_path)
    first_load_time = time.time() - start_time
    logging.info(f"First load time (default dict): {first_load_time:.4f} seconds")

    assert os.path.exists(cache_file_path_default + ".trie")
    assert os.path.getsize(cache_file_path_default + ".trie") > 0
    assert os.path.exists(cache_file_path_default + ".freq")
    assert os.path.getsize(cache_file_path_default + ".freq") > 0

    # --- Test 2: Second load of default dict (should use cache and be faster) ---
    jieba.dt = jieba.Tokenizer() # Fresh tokenizer
    start_time = time.time()
    jieba.dt.initialize(default_dict_path)
    second_load_time = time.time() - start_time
    logging.info(f"Second load time (default dict with cache): {second_load_time:.4f} seconds")

    assert second_load_time < first_load_time
    assert first_load_time / second_load_time > 2 # Expect significant speedup

    # --- Test 3: Cache invalidation (modify a *temporary* dict, cache should rebuild) ---
    # Create a temporary dictionary file for invalidation test
    invalidation_dict_content = "你好 1000\n世界 1000\n"
    temp_invalidation_dict_path = create_temp_dict(invalidation_dict_content)

    # First load of temp dict to create its cache
    jieba.dt = jieba.Tokenizer() # Fresh tokenizer
    jieba.dt.initialize(temp_invalidation_dict_path)
    cache_file_path_temp_initial = jieba.get_cache_file_path(temp_invalidation_dict_path)
    assert os.path.exists(cache_file_path_temp_initial + ".trie")

    # Modify the temporary dictionary file
    modified_invalidation_dict_content = invalidation_dict_content + "新詞 500\n"
    with open(temp_invalidation_dict_path, 'w', encoding='utf-8') as f:
        f.write(modified_invalidation_dict_content)

    # Get the new cache file path for the modified temporary dict (should be different)
    cache_file_path_temp_modified = jieba.get_cache_file_path(temp_invalidation_dict_path)
    assert cache_file_path_temp_modified != cache_file_path_temp_initial

    jieba.dt = jieba.Tokenizer() # Fresh tokenizer
    start_time = time.time()
    jieba.dt.initialize(temp_invalidation_dict_path)
    third_load_time = time.time() - start_time
    logging.info(f"Third load time (after temp dict modification): {third_load_time:.4f} seconds")

    # The third load should be similar to a first load (rebuilding cache)
    # We can't directly compare to first_load_time because dicts are different sizes
    # But we can assert that new cache files are created
    assert os.path.exists(cache_file_path_temp_modified + ".trie")
    assert os.path.exists(cache_file_path_temp_modified + ".freq")

    # Verify the new word is recognized
    seg_result = jieba.lcut("你好世界新詞")
    assert "新詞" in seg_result

    # Clean up
    os.remove(temp_invalidation_dict_path)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def test_tokenization_with_cache():
    # Create a temporary dictionary file for testing
    initial_dict_content = "你好 1000\n世界 1000\n分詞 1000\n測試 1000\n"
    temp_dict_path = create_temp_dict(initial_dict_content)

    # Ensure cache directory is clean
    cache_file_path = jieba.get_cache_file_path(temp_dict_path)
    cache_dir = os.path.dirname(cache_file_path)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # First load to build cache
    jieba.dt = jieba.Tokenizer()
    jieba.dt.initialize(temp_dict_path)

    # Perform tokenization
    sentence = "你好世界分詞測試"
    expected_result = ["你好", "世界", "分詞", "測試"]
    result = jieba.lcut(sentence)
    assert result == expected_result, f"First tokenization mismatch. Expected: {expected_result}, Got: {result}"

    # Re-initialize tokenizer to load from cache
    jieba.dt = jieba.Tokenizer()
    jieba.dt.initialize(temp_dict_path)

    # Perform tokenization again (should use cache)
    result_from_cache = jieba.lcut(sentence)
    assert result_from_cache == expected_result, f"Tokenization from cache mismatch. Expected: {expected_result}, Got: {result_from_cache}"

    # Clean up
    os.remove(temp_dict_path)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
