import logging
import os
import shutil
import tempfile
import time

import pytest

import jieba_fast_dat

# Configure logging for the test environment
# We'll allow debug logs to be printed if pytest is run with -s or --log-cli-level=DEBUG
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _clear_all_jieba_caches(tmp_dir_to_clear: str):
    """
    Helper function to clear caches (adapted from perf_comparison.py).
    """
    for f in os.listdir(tmp_dir_to_clear):
        path = os.path.join(tmp_dir_to_clear, f)
        is_jieba_cache = f.startswith("jieba.") and (
            f.endswith(".cache")
            or f.endswith(".cache.dat")
            or f.endswith(".tmp")  # Added .tmp for temporary files
        )
        is_jieba_fast_dat_cache = f.startswith("jieba_fast_dat.") and (
            f.endswith(".cache")
            or f.endswith(".dat")
            or f.endswith(".idf.dat")
            or f.endswith("hmm_models.bin")
            or f.endswith(".tmp")  # Added .tmp for temporary files
            or f.endswith(".dict")  # Added .dict for temporary user dict content files
        )

        if is_jieba_cache or is_jieba_fast_dat_cache:
            try:
                os.remove(path)
            except OSError:
                pass
        elif f.startswith("jieba.") and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def _clear_all_jieba_caches_global(fixture_temp_dir: str):
    """
    Clears caches from both the fixture's temp_dir and the system's /tmp/.
    """
    # Clear caches from the system's default temp directory
    _clear_all_jieba_caches(tempfile.gettempdir())
    # Clear caches from the fixture's temp directory
    _clear_all_jieba_caches(fixture_temp_dir)


@pytest.fixture
def clean_cache_env():
    """
    Fixture to set up a temporary directory for cache testing and ensure cleanup.
    Yields the path to the temporary directory.
    """
    temp_dir = tempfile.mkdtemp()
    original_default_dict = jieba_fast_dat.DEFAULT_DICT
    original_tmp_dir = jieba_fast_dat.dt.tmp_dir

    # Clear any existing caches that might interfere with cold start tests
    _clear_all_jieba_caches_global(temp_dir)

    # Set dt.tmp_dir for global initializations in case it's used
    jieba_fast_dat.dt.tmp_dir = temp_dir

    yield temp_dir

    # Teardown: Clean up the temporary directory and restore original settings
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    jieba_fast_dat.dt.tmp_dir = original_tmp_dir
    jieba_fast_dat.DEFAULT_DICT = original_default_dict


def _generate_large_dict(filepath: str, num_words: int = 500, prefix_base_idx: int = 0):
    """
    Generates a large dictionary file with dummy words for testing.
    `prefix_base_idx` allows selecting a different starting prefix from base_words.
    """
    base_words = [
        "測試詞",
        "另一個詞",
        "新加的詞",
        "自定義詞",
        "專用詞A",
        "專用詞B",
    ]  # Added more base words
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(num_words):
            word = f"{base_words[(prefix_base_idx + i) % len(base_words)]}{i}"
            freq = 100 + i
            f.write(f"{word} {freq}\n")


class TestCacheMechanism:
    def test_all_cache_scenarios(self, clean_cache_env: str):
        temp_dir = clean_cache_env

        # --- Test with default dictionary ---
        logging.info("\n--- Testing with default dictionary ---")

        # 1. First initialization (should build cache)
        start_time = time.time()
        tk_default_1 = jieba_fast_dat.Tokenizer()
        tk_default_1.tmp_dir = (
            temp_dir  # Ensure this tokenizer uses the fixture's temp_dir
        )
        tk_default_1.initialize()
        end_time = time.time()
        first_init_time = end_time - start_time
        logging.info(
            f"1. Default dict, 1st init (build/verify): {first_init_time:.4f}s"
        )
        assert first_init_time >= 0, (
            "First default dict init time should be non-negative."
        )  # Sanity check

        # 2. Second initialization (should load from cache)
        start_time = time.time()
        tk_default_2 = jieba_fast_dat.Tokenizer()
        tk_default_2.tmp_dir = (
            temp_dir  # Ensure this tokenizer uses the fixture's temp_dir
        )
        tk_default_2.initialize()
        end_time = time.time()
        second_init_time = end_time - start_time
        logging.info(f"2. Default dict, 2nd init (load cache): {second_init_time:.4f}s")
        assert second_init_time < first_init_time, (
            "Second default dict init should be faster due to cache."
        )
        assert second_init_time >= 0, (
            "Second default dict init time should be non-negative."
        )  # Sanity check

        logging.info("-" * 20)

        # --- Test with custom dictionary and modification ---
        logging.info("--- Testing with custom dictionary (Tokenizer.initialize) ---")
        dict_path = os.path.join(temp_dir, "temp_dict.txt")
        _generate_large_dict(
            dict_path, num_words=500, prefix_base_idx=0
        )  # Use 500 words for custom dict

        # 3. Custom dict first initialization (should build cache)
        start_time = time.time()
        tk_custom = jieba_fast_dat.Tokenizer()
        tk_custom.tmp_dir = temp_dir
        tk_custom.initialize(dict_path)
        end_time = time.time()
        custom_init_1_time = end_time - start_time
        logging.info(
            f"3. Custom dict, 1st init (build cache): {custom_init_1_time:.4f}s"
        )
        assert tk_custom.get_freq("測試詞0") > 0, "測試詞0 should be in the dictionary"
        assert custom_init_1_time >= 0, (
            "First custom dict init time should be non-negative."
        )  # Sanity check

        # 4. Custom dict second initialization (should load from cache)
        start_time = time.time()
        tk_custom_2 = jieba_fast_dat.Tokenizer()
        tk_custom_2.tmp_dir = temp_dir
        tk_custom_2.initialize(dict_path)
        end_time = time.time()
        custom_init_2_time = end_time - start_time
        logging.info(
            f"4. Custom dict, 2nd init (load cache): {custom_init_2_time:.4f}s"
        )
        assert tk_custom_2.get_freq("測試詞0") > 0, (
            "測試詞0 should still be in the dictionary"
        )
        assert custom_init_2_time < custom_init_1_time, (
            "Second custom dict init should be faster due to cache."
        )
        assert custom_init_2_time >= 0, (
            "Second custom dict init time should be non-negative."
        )  # Sanity check

        # Give it a moment to ensure modification time is different
        time.sleep(1.1)  # Ensure mtime changes

        # 5. Modify the dictionary
        logging.info("\nModifying the dictionary file...")
        with open(dict_path, "a", encoding="utf-8") as f:
            f.write("新加的詞 300\n")
        os.utime(dict_path, None)  # Update modification time

        # 6. Custom dict third initialization (should rebuild cache)
        start_time = time.time()
        tk_custom_3 = jieba_fast_dat.Tokenizer()
        tk_custom_3.tmp_dir = temp_dir
        tk_custom_3.initialize(dict_path)
        end_time = time.time()
        custom_init_3_time = end_time - start_time
        logging.info(
            f"6. Custom dict, 3rd init (modified, rebuild): {custom_init_3_time:.4f}s"
        )
        assert tk_custom_3.get_freq("新加的詞") > 0, (
            "新加的詞 should be in the dictionary after rebuild"
        )
        assert custom_init_3_time > custom_init_2_time, (
            "Third custom dict init should be slower due to rebuild."
        )
        assert custom_init_3_time >= 0, (
            "Third custom dict init time should be non-negative."
        )  # Sanity check

        # 7. Custom dict fourth initialization (should load from new cache)
        start_time = time.time()
        tk_custom_4 = jieba_fast_dat.Tokenizer()
        tk_custom_4.tmp_dir = temp_dir
        tk_custom_4.initialize(dict_path)
        end_time = time.time()
        custom_init_4_time = end_time - start_time
        logging.info(
            f"7. Custom dict, 4th init (load new cache): {custom_init_4_time:.4f}s"
        )
        assert tk_custom_4.get_freq("新加的詞") > 0, (
            "新加的詞 should still be in the dictionary"
        )
        assert custom_init_4_time < custom_init_3_time, (
            "Fourth custom dict init should be faster due to cache."
        )
        assert custom_init_4_time >= 0, (
            "Fourth custom dict init time should be non-negative."
        )  # Sanity check

        logging.info("-" * 20)
        # --- Test with load_userdict caching ---
        logging.info("\n--- Testing load_userdict caching ---")
        user_dict_path_for_load = os.path.join(temp_dir, "user_dict_for_load.txt")
        _generate_large_dict(
            user_dict_path_for_load, num_words=500, prefix_base_idx=2
        )  # Use 500 words for userdict load test

        # Initialize a tokenizer first to load the main dict
        tk_userdict = jieba_fast_dat.Tokenizer()
        tk_userdict.tmp_dir = temp_dir
        tk_userdict.initialize(
            dict_path
        )  # Use the custom dict as base for userdict tests

        # 8. load_userdict first call (should build cache)
        start_time = time.time()
        tk_userdict.load_userdict(user_dict_path_for_load)
        end_time = time.time()
        load_userdict_1_time = end_time - start_time
        logging.info(
            f"8. load_userdict, 1st call (build cache): {load_userdict_1_time:.4f}s"
        )
        assert tk_userdict.get_freq("新加的詞0") > 0, (
            "新加的詞0 should be in the dictionary"
        )
        assert load_userdict_1_time >= 0, (
            "First load_userdict call should take measurable time."
        )  # Sanity check

        # To ensure a fresh test of loading from cache, use a new tokenizer instance
        # It's important that this new tokenizer also loads the base dict
        tk_userdict_2 = jieba_fast_dat.Tokenizer()
        tk_userdict_2.tmp_dir = temp_dir
        tk_userdict_2.initialize(dict_path)  # Initialize with base dict

        # 9. load_userdict second call (should load from cache)
        start_time = time.time()
        tk_userdict_2.load_userdict(user_dict_path_for_load)
        end_time = time.time()
        load_userdict_2_time = end_time - start_time
        logging.info(
            f"9. load_userdict, 2nd call (load cache): {load_userdict_2_time:.4f}s"
        )
        assert tk_userdict_2.get_freq("新加的詞0") > 0, (
            "新加的詞0 should still be in the dictionary"
        )
        assert load_userdict_2_time < load_userdict_1_time, (
            "Second load_userdict call should be faster due to cache."
        )
        assert load_userdict_2_time >= 0, (
            "Second load_userdict call should be non-negative."
        )  # Sanity check

        # Give it a moment to ensure modification time is different
        time.sleep(1.1)

        # 10. Modify the user dictionary for load_userdict
        logging.info("\nModifying the user dictionary file for load_userdict...")
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
        load_userdict_3_time = end_time - start_time
        logging.info(
            f"11. load_userdict, 3rd (modified, rebuild): {load_userdict_3_time:.4f}s"
        )
        assert tk_userdict_3.get_freq("新增自定義詞3") > 0, (
            "新增自定義詞3 should be in the dictionary after rebuild"
        )
        assert load_userdict_3_time > load_userdict_2_time, (
            "Third load_userdict call should be slower due to rebuild."
        )
        assert load_userdict_3_time >= 0, (
            "Third load_userdict call should be non-negative."
        )  # Sanity check

        # Use a new tokenizer instance
        tk_userdict_4 = jieba_fast_dat.Tokenizer()
        tk_userdict_4.tmp_dir = temp_dir
        tk_userdict_4.initialize(dict_path)  # Initialize with base dict

        # 12. load_userdict fourth call (should load from new cache)
        start_time = time.time()
        tk_userdict_4.load_userdict(user_dict_path_for_load)
        end_time = time.time()
        load_userdict_4_time = end_time - start_time
        logging.info(
            f"12. load_userdict, 4th (load new cache): {load_userdict_4_time:.4f}s"
        )
        assert tk_userdict_4.get_freq("新增自定義詞3") > 0, (
            "新增自定義詞3 should still be in the dictionary"
        )
        assert load_userdict_4_time < load_userdict_3_time, (
            "Fourth load_userdict call should be faster due to cache."
        )
        assert load_userdict_4_time >= 0, (
            "Fourth load_userdict call should be non-negative."
        )  # Sanity check

        logging.info("-" * 20)

        logging.info("\nAll cache verification steps passed!")
