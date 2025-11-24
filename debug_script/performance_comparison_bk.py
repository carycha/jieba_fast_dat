import os
import shutil
import sys
import tempfile
import time
import warnings
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="jieba",
)

# Try to import jieba and jieba_fast_dat
try:
    import jieba
    import jieba.posseg
except ImportError:
    print("Error: jieba is not installed. Please install it using 'pip install jieba'")
    sys.exit(1)

try:
    import jieba_fast_dat
    import jieba_fast_dat.posseg
except ImportError:
    print(
        "Error: jieba_fast_dat is not installed. Please install it using "
        "'pip install jieba_fast_dat'"
    )
    sys.exit(1)


def measure_time(func: Any, *args: Any, **kwargs: Any):  # noqa: ANN401
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    # If the result is an iterator, consume it to ensure all computation is done
    if hasattr(result, "__iter__") and not isinstance(
        result, (list, tuple, str, dict, bytes)
    ):
        _ = list(result)
    end_time = time.perf_counter()
    return end_time - start_time, result


def run_performance_test():
    print("Starting performance comparison between jieba and jieba_fast_dat...")

    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    custom_dict_path = os.path.join(
        current_dir, "..", "extra_dict", "dict.txt.big.tw_nerd.txt"
    )
    user_dict_path = os.path.join(current_dir, "..", "extra_dict", "dict.txt.small")
    test_text_path = os.path.join(current_dir, "..", "extra_dict", "profile_test")

    # Define cache file paths
    def clear_all_caches():
        print("Clearing all caches...")
        # Clear jieba_fast_dat cache
        jieba_fast_dat_cache_dir = tempfile.gettempdir()
        for f in os.listdir(jieba_fast_dat_cache_dir):
            if f.startswith("jieba_fast_dat.u") and (
                f.endswith(".cache") or f.endswith(".cache.dat")
            ):
                os.remove(os.path.join(jieba_fast_dat_cache_dir, f))
            elif f.startswith("jieba_fast_dat.cache") and (
                f.endswith(".cache") or f.endswith(".cache.dat")
            ):
                os.remove(os.path.join(jieba_fast_dat_cache_dir, f))

        # Clear jieba cache
        # Jieba's cache files are usually in /tmp/jieba.*
        tmp_dir = tempfile.gettempdir()
        for f in os.listdir(tmp_dir):
            if f.startswith("jieba.") and (
                f.endswith(".cache") or f.endswith(".cache.dat")
            ):
                os.remove(os.path.join(tmp_dir, f))
            elif f.startswith("jieba.") and os.path.isdir(os.path.join(tmp_dir, f)):
                shutil.rmtree(os.path.join(tmp_dir, f), ignore_errors=True)
        print("All caches cleared.")

    # Check if files exist
    if not os.path.exists(custom_dict_path):
        print(f"Error: Custom dictionary file not found at {custom_dict_path}")
        sys.exit(1)
    if not os.path.exists(user_dict_path):
        print(f"Error: User dictionary file not found at {user_dict_path}")
        sys.exit(1)
    if not os.path.exists(test_text_path):
        print(f"Error: Test text file not found at {test_text_path}")
        sys.exit(1)

    # Read test text
    with open(test_text_path, encoding="utf-8") as f:
        long_text = f.read()

    print(f"\n--- Dictionary Initialization Performance ({custom_dict_path}) ---")

    # --- First Initialization (No Cache) ---
    clear_all_caches()
    print("\n--- First Initialization (No Cache) ---")

    # Jieba first initialization
    jieba.set_dictionary(custom_dict_path)
    jieba_first_init_time, _ = measure_time(jieba.initialize)
    print(f"Jieba first initialize time: {jieba_first_init_time:.4f} seconds")

    # Jieba_fast_dat first initialization
    jieba_fast_dat.set_dictionary(custom_dict_path)
    jieba_fast_dat_first_init_time, _ = measure_time(jieba_fast_dat.initialize)
    print(
        f"Jieba_fast_dat first initialize time: "
        f"{jieba_fast_dat_first_init_time:.4f} seconds"
    )

    # --- Second Initialization (With Cache) ---
    print("\n--- Second Initialization (With Cache) ---")

    # Jieba second initialization
    jieba_second_init_time, _ = measure_time(jieba.initialize)
    print(f"Jieba second initialize time: {jieba_second_init_time:.4f} seconds")

    # Jieba_fast_dat second initialization
    jieba_fast_dat_second_init_time, _ = measure_time(jieba_fast_dat.initialize)
    print(
        f"Jieba_fast_dat second initialize time: "
        f"{jieba_fast_dat_second_init_time:.4f} seconds"
    )

    print(f"\n--- User Dictionary Loading Performance ({user_dict_path}) ---")

    # Measure user dictionary loading time
    jieba_user_dict_load_time, _ = measure_time(jieba.load_userdict, user_dict_path)
    print(f"Jieba user dict load time: {jieba_user_dict_load_time:.4f} seconds")

    jieba_fast_dat_user_dict_load_time, _ = measure_time(
        jieba_fast_dat.load_userdict, user_dict_path
    )
    print(
        f"Jieba_fast_dat user dict load time: "
        f"{jieba_fast_dat_user_dict_load_time:.4f} seconds"
    )

    print(f"\n--- Word Segmentation (cut) on long text ({test_text_path}) ---")

    # Jieba word segmentation
    jieba_cut_time, _ = measure_time(jieba.cut, long_text, HMM=False)
    print(f"Jieba cut time: {jieba_cut_time:.4f} seconds")

    # Jieba_fast_dat word segmentation
    jieba_fast_dat_cut_time, _ = measure_time(jieba_fast_dat.cut, long_text, HMM=False)
    print(f"Jieba_fast_dat cut time: {jieba_fast_dat_cut_time:.4f} seconds")

    print(
        f"\n--- Part-of-Speech Tagging (posseg.cut) on long text ({test_text_path}) ---"
    )

    # Jieba POS tagging
    jieba_pos_time, _ = measure_time(jieba.posseg.cut, long_text, HMM=False)
    print(f"Jieba posseg.cut time: {jieba_pos_time:.4f} seconds")

    # Jieba_fast_dat POS tagging
    jieba_fast_dat_pos_time, _ = measure_time(
        jieba_fast_dat.posseg.cut, long_text, HMM=False
    )
    print(f"Jieba_fast_dat posseg.cut time: {jieba_fast_dat_pos_time:.4f} seconds")

    print(
        f"\n--- Word Segmentation (cut) with HMM=True on long text "
        f"({test_text_path}) ---"
    )

    # Jieba word segmentation with HMM=True
    jieba_cut_hmm_true_time, _ = measure_time(jieba.cut, long_text, HMM=True)
    print(f"Jieba cut (HMM=True) time: {jieba_cut_hmm_true_time:.4f} seconds")

    # Jieba_fast_dat word segmentation with HMM=True
    jieba_fast_dat_cut_hmm_true_time, _ = measure_time(
        jieba_fast_dat.cut, long_text, HMM=True
    )
    print(
        f"Jieba_fast_dat cut (HMM=True) time: "
        f"{jieba_fast_dat_cut_hmm_true_time:.4f} seconds"
    )

    print(
        f"\n--- Part-of-Speech Tagging (posseg.cut) with HMM=True on long text "
        f"({test_text_path}) ---"
    )

    # Jieba POS tagging with HMM=True
    jieba_pos_hmm_true_time, _ = measure_time(jieba.posseg.cut, long_text, HMM=True)
    print(f"Jieba posseg.cut (HMM=True) time: {jieba_pos_hmm_true_time:.4f} seconds")

    # Jieba_fast_dat POS tagging with HMM=True
    jieba_fast_dat_pos_hmm_true_time, _ = measure_time(
        jieba_fast_dat.posseg.cut, long_text, HMM=True
    )
    print(
        f"Jieba_fast_dat posseg.cut (HMM=True) time: "
        f"{jieba_fast_dat_pos_hmm_true_time:.4f} seconds"
    )

    print("\n--- Summary ---")
    print(f"{'Metric':<40} {'Jieba':>15} {'Jieba_fast_dat':>15}")
    print(
        f"{'Initial Main Dict Load (No Cache)':<40} "
        f"{jieba_first_init_time:>15.7f}s "
        f"{jieba_fast_dat_first_init_time:>15.7f}s"
    )
    print(
        f"{'Initial Main Dict Load (With Cache)':<40} "
        f"{jieba_second_init_time:>15.7f}s "
        f"{jieba_fast_dat_second_init_time:>15.7f}s"
    )
    print(
        f"{'User Dict Load':<40} {jieba_user_dict_load_time:>15.7f}s "
        f"{jieba_fast_dat_user_dict_load_time:>15.7f}s"
    )
    print(
        f"{'Word Segmentation (HMM=False)':<40} {jieba_cut_time:>15.7f}s "
        f"{jieba_fast_dat_cut_time:>15.7f}s"
    )
    print(
        f"{'POS Tagging (HMM=False)':<40} {jieba_pos_time:>15.7f}s "
        f"{jieba_fast_dat_pos_time:>15.7f}s"
    )
    print(
        f"{'Word Segmentation (HMM=True)':<40} {jieba_cut_hmm_true_time:>15.7f}s "
        f"{jieba_fast_dat_cut_hmm_true_time:>15.7f}s"
    )
    print(
        f"{'POS Tagging (HMM=True)':<40} {jieba_pos_hmm_true_time:>15.7f}s "
        f"{jieba_fast_dat_pos_hmm_true_time:>15.7f}s"
    )


if __name__ == "__main__":
    run_performance_test()
