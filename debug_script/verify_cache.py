import logging
import os
import shutil
import tempfile
import time

import jieba_fast_dat

# Enable debug logging to see cache messages
jieba_fast_dat.setLogLevel(logging.DEBUG)

# Create a temporary directory for custom dictionary and its cache
temp_dir = tempfile.mkdtemp()
print(f"Using temporary directory: {temp_dir}")

original_default_dict = jieba_fast_dat.DEFAULT_DICT
original_tmp_dir = jieba_fast_dat.dt.tmp_dir

try:
    # --- Test with default dictionary ---
    print("\n--- Testing with default dictionary ---")
    # Set default tokenizer's tmp_dir for clearer cache management.
    jieba_fast_dat.dt.tmp_dir = temp_dir
    jieba_fast_dat.initialize()  # Ensure default dict is initialized and cached

    # 1. First initialization (should build cache)
    start_time = time.time()
    # Re-initialize to see the time, even if it's already cached.
    # We use a fresh Tokenizer instance to ensure it goes through the init process.
    tk_default_1 = jieba_fast_dat.Tokenizer()
    tk_default_1.initialize()
    end_time = time.time()
    print(f"1. Default dict, 1st init (build/verify): {end_time - start_time:.4f} secs")

    # 2. Second initialization (should load from cache)
    start_time = time.time()
    tk_default_2 = jieba_fast_dat.Tokenizer()
    tk_default_2.initialize()
    end_time = time.time()
    print(f"2. Default dict, 2nd init (load cache): {end_time - start_time:.4f} secs")
    print("-" * 20)

    # --- Test with custom dictionary and modification ---
    print("--- Testing with custom dictionary ---")
    dict_path = os.path.join(temp_dir, "temp_dict.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("測試詞 100\n")
        f.write("另一個詞 200\n")

    # 3. Custom dict first initialization (should build cache)
    start_time = time.time()
    tk_custom = jieba_fast_dat.Tokenizer()
    tk_custom.tmp_dir = temp_dir  # Assign custom tokenizer to use our temp_dir
    tk_custom.initialize(dict_path)
    end_time = time.time()
    print(
        f"3. Custom dict, 1st init (build cache): {end_time - start_time:.4f} seconds"
    )
    assert tk_custom.get_freq("測試詞") > 0, "測試詞 should be in the dictionary"

    # 4. Custom dict second initialization (should load from cache)
    start_time = time.time()
    tk_custom_2 = jieba_fast_dat.Tokenizer()
    tk_custom_2.tmp_dir = temp_dir  # Assign custom tokenizer to use our temp_dir
    tk_custom_2.initialize(dict_path)
    end_time = time.time()
    print(f"4. Custom dict, 2nd init (load cache): {end_time - start_time:.4f} secs")
    assert tk_custom_2.get_freq("測試詞") > 0, (
        "測試詞 should still be in the dictionary"
    )

    # Give it a moment to ensure modification time is different
    time.sleep(1)

    # 5. Modify the dictionary
    print("\nModifying the dictionary file...")
    with open(dict_path, "a", encoding="utf-8") as f:
        f.write("新加的詞 300\n")
    # Update modification time of the dictionary file
    os.utime(dict_path, None)

    # 6. Custom dict third initialization (should rebuild cache)
    start_time = time.time()
    tk_custom_3 = jieba_fast_dat.Tokenizer()
    tk_custom_3.tmp_dir = temp_dir  # Assign custom tokenizer to use our temp_dir
    tk_custom_3.initialize(dict_path)
    end_time = time.time()
    print(f"6. Custom dict, 3rd init (modified, rebuild): {end_time - start_time:.4f}s")
    assert tk_custom_3.get_freq("新加的詞") > 0, (
        "新加的詞 should be in the dictionary after rebuild"
    )

    # 7. Custom dict fourth initialization (should load from new cache)
    start_time = time.time()
    tk_custom_4 = jieba_fast_dat.Tokenizer()
    tk_custom_4.tmp_dir = temp_dir  # Assign custom tokenizer to use our temp_dir
    tk_custom_4.initialize(dict_path)
    end_time = time.time()
    print(
        f"7. Custom dict, 4th init (load new cache): {end_time - start_time:.4f} secs"
    )
    assert tk_custom_4.get_freq("新加的詞") > 0, (
        "新加的詞 should still be in the dictionary"
    )

    print("-" * 20)
    # --- Test with load_userdict caching ---
    print("\n--- Testing load_userdict caching ---")
    user_dict_path_for_load = os.path.join(temp_dir, "user_dict_for_load.txt")
    with open(user_dict_path_for_load, "w", encoding="utf-8") as f:
        f.write("自定義詞1 100\n")
        f.write("自定義詞2 200\n")

    # Initialize a tokenizer first to load the main dict
    tk_userdict = jieba_fast_dat.Tokenizer()
    tk_userdict.tmp_dir = temp_dir
    tk_userdict.initialize(dict_path)  # Use the custom dict as base for userdict tests

    # 8. load_userdict first call (should build cache)
    start_time = time.time()
    tk_userdict.load_userdict(user_dict_path_for_load)
    end_time = time.time()
    print(
        f"8. load_userdict, 1st call (build cache): {end_time - start_time:.4f} seconds"
    )
    assert tk_userdict.get_freq("自定義詞1") > 0, (
        "自定義詞1 should be in the dictionary"
    )

    # To ensure a fresh test of loading from cache, use a new tokenizer instance
    tk_userdict_2 = jieba_fast_dat.Tokenizer()
    tk_userdict_2.tmp_dir = temp_dir
    tk_userdict_2.initialize(dict_path)  # Initialize with base dict

    # 9. load_userdict second call (should load from cache)
    start_time = time.time()
    tk_userdict_2.load_userdict(user_dict_path_for_load)
    end_time = time.time()
    print(f"9. load_userdict, 2nd call (load cache): {end_time - start_time:.4f} secs")
    assert tk_userdict_2.get_freq("自定義詞1") > 0, (
        "自定義詞1 should still be in the dictionary"
    )

    # Give it a moment to ensure modification time is different
    time.sleep(1)

    # 10. Modify the user dictionary for load_userdict
    print("\nModifying the user dictionary file for load_userdict...")
    with open(user_dict_path_for_load, "a", encoding="utf-8") as f:
        f.write("新增自定義詞3 300\n")
    os.utime(user_dict_path_for_load, None)  # Update modification time

    # Use a new tokenizer instance to ensure clean state for cache check
    tk_userdict_3 = jieba_fast_dat.Tokenizer()
    tk_userdict_3.tmp_dir = temp_dir
    tk_userdict_3.initialize(dict_path)  # Initialize with base dict

    # 11. load_userdict third call after mod (should rebuild cache)
    start_time = time.time()
    tk_userdict_3.load_userdict(user_dict_path_for_load)
    end_time = time.time()
    print(f"11. load_userdict, 3rd (modified, rebuild): {end_time - start_time:.4f}s")
    assert tk_userdict_3.get_freq("新增自定義詞3") > 0, (
        "新增自定義詞3 should be in the dictionary after rebuild"
    )

    # Use a new tokenizer instance
    tk_userdict_4 = jieba_fast_dat.Tokenizer()
    tk_userdict_4.tmp_dir = temp_dir
    tk_userdict_4.initialize(dict_path)  # Initialize with base dict

    # 12. load_userdict fourth call (should load from new cache)
    start_time = time.time()
    tk_userdict_4.load_userdict(user_dict_path_for_load)
    end_time = time.time()
    print(f"12. load_userdict, 4th (load new cache): {end_time - start_time:.4f}s")
    assert tk_userdict_4.get_freq("新增自定義詞3") > 0, (
        "新增自定義詞3 should still be in the dictionary"
    )
    print("-" * 20)

    print("\nAll cache verification steps passed!")

finally:
    # Clean up the temporary directory
    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)
    # Restore original settings
    jieba_fast_dat.dt.tmp_dir = original_tmp_dir
    jieba_fast_dat.DEFAULT_DICT = original_default_dict
