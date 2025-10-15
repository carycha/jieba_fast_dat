import time
import os
import pytest
import jieba_fast_dat as jieba

# Assuming the default dictionary is used for speed testing
# We'll use the actual dict.txt from the package for this test
# To ensure we're testing the real loading mechanism.

def test_dictionary_loading_speed():
    # First initialization - should be fast but might involve initial setup
    start_time = time.time()
    jieba.initialize()
    first_load_time = time.time() - start_time
    print(f"First dictionary load time: {first_load_time:.4f} seconds")
    assert first_load_time < 1.0, f"First dictionary load took too long: {first_load_time:.4f}s"

    # Simulate a dictionary change by touching the file
    # This should force a re-initialization
    dict_path = jieba.dt.dictionary # Access the dictionary path from the tokenizer instance
    os.utime(dict_path, None) # Update modification time

    # Second initialization - should be very fast due to caching
    start_time = time.time()
    jieba.initialize()
    second_load_time = time.time() - start_time
    print(f"Second dictionary load time (after touch): {second_load_time:.4f} seconds")
    assert second_load_time < 0.1, f"Second dictionary load took too long: {second_load_time:.4f}s"
    assert second_load_time < first_load_time, "Second load was not faster than the first load, caching might not be effective."

    # Third initialization - no change, should be instant
    start_time = time.time()
    jieba.initialize()
    third_load_time = time.time() - start_time
    print(f"Third dictionary load time (no change): {third_load_time:.4f} seconds")
    assert third_load_time < 0.01, f"Third dictionary load took too long: {third_load_time:.4f}s"
    assert third_load_time < second_load_time, "Third load was not faster than the second load, caching might not be effective."
