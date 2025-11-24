import importlib  # Added for dynamic import
import multiprocessing  # Added for process isolation
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


def measure_time(func: Any, repetitions: int = 1, *args: Any, **kwargs: Any):  # noqa: ANN401
    total_time = 0.0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        # If the result is an iterator, consume it to ensure all computation is done
        if hasattr(result, "__iter__") and not isinstance(
            result, (list, tuple, str, dict, bytes)
        ):
            _ = list(result)
        end_time = time.perf_counter()
        total_time += end_time - start_time
    return total_time / repetitions, result


def clear_all_caches(silent: bool = False):
    if not silent:
        print("Clearing all caches...")
    # Clear jieba_fast_dat cache
    jieba_fast_dat_cache_dir = tempfile.gettempdir()
    for f in os.listdir(jieba_fast_dat_cache_dir):
        if f.startswith("jieba_fast_dat.u") and (
            f.endswith(".cache") or f.endswith(".cache.dat")
        ):
            try:
                os.remove(os.path.join(jieba_fast_dat_cache_dir, f))
            except OSError:
                pass  # Ignore errors if file is already removed or inaccessible
        elif f.startswith("jieba_fast_dat.cache") and (
            f.endswith(".cache") or f.endswith(".cache.dat")
        ):
            try:
                os.remove(os.path.join(jieba_fast_dat_cache_dir, f))
            except OSError:
                pass

    # Clear jieba cache
    # Jieba's cache files are usually in /tmp/jieba.*
    tmp_dir = tempfile.gettempdir()
    for f in os.listdir(tmp_dir):
        if f.startswith("jieba.") and (
            f.endswith(".cache") or f.endswith(".cache.dat")
        ):
            try:
                os.remove(os.path.join(tmp_dir, f))
            except OSError:
                pass
        elif f.startswith("jieba.") and os.path.isdir(os.path.join(tmp_dir, f)):
            shutil.rmtree(os.path.join(tmp_dir, f), ignore_errors=True)
    if not silent:
        print("All caches cleared.")


# This is the new function that will run in a separate process
def _worker_function(
    lib_name: str,
    num_repetitions: int,
    custom_dict_path: str,
    user_dict_path: str,
    test_text: str,
    results_queue: multiprocessing.Queue,
):
    # Ensure a clean state for this process
    clear_all_caches(silent=True)

    lib_module = importlib.import_module(lib_name)
    lib_posseg_module = importlib.import_module(f"{lib_name}.posseg")

    lib_results = {}

    # Dictionary Initialization Performance
    lib_module.set_dictionary(custom_dict_path)
    init_time, _ = measure_time(lib_module.initialize, num_repetitions)
    lib_results["first_init_time"] = init_time

    # Second Initialization (With Cache)
    second_init_time, _ = measure_time(lib_module.initialize, num_repetitions)
    lib_results["second_init_time"] = second_init_time

    # User Dictionary Loading Performance
    user_dict_load_time, _ = measure_time(
        lib_module.load_userdict, num_repetitions, user_dict_path
    )
    lib_results["user_dict_load_time"] = user_dict_load_time

    # Word Segmentation (cut) on long text (HMM=False)
    cut_time, _ = measure_time(lib_module.cut, num_repetitions, test_text, HMM=False)
    lib_results["cut_time_HMM_False"] = cut_time

    # Part-of-Speech Tagging (posseg.cut) on long text (HMM=False)
    pos_time, _ = measure_time(
        lib_posseg_module.cut, num_repetitions, test_text, HMM=False
    )
    lib_results["pos_time_HMM_False"] = pos_time

    # Word Segmentation (cut) with HMM=True
    cut_hmm_true_time, _ = measure_time(
        lib_module.cut, num_repetitions, test_text, HMM=True
    )
    lib_results["cut_time_HMM_True"] = cut_hmm_true_time

    # Part-of-Speech Tagging (posseg.cut) with HMM=True
    pos_hmm_true_time, _ = measure_time(
        lib_posseg_module.cut, num_repetitions, test_text, HMM=True
    )
    lib_results["pos_time_HMM_True"] = pos_hmm_true_time

    results_queue.put({lib_name: lib_results})


def run_performance_test():
    print("Starting performance comparison between jieba and jieba_fast_dat...")

    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    custom_dict_path = os.path.join(current_dir, "..", "extra_dict", "dict.txt.big")
    # custom_dict_path = os.path.join(
    #     current_dir, "..", "test/test_dicts", "test_dict_base.txt"
    # )
    user_dict_path = os.path.join(
        current_dir, "..", "extra_dict", "dict.txt.big.tw_nerd.txt"
    )
    # user_dict_path = os.path.join(
    #     current_dir, "..", "test/test_dicts", "test_user_dict_base.txt"
    # )
    test_text_path = os.path.join(current_dir, "..", "extra_dict", "profile_test")

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

    NUM_REPETITIONS = 10  # Define repetition count for each test

    # Multiprocessing setup
    manager = multiprocessing.Manager()
    results_queue = manager.Queue()

    jieba_process = multiprocessing.Process(
        target=_worker_function,
        args=(
            "jieba",
            NUM_REPETITIONS,
            custom_dict_path,
            user_dict_path,
            long_text,
            results_queue,
        ),
    )
    jieba_fast_dat_process = multiprocessing.Process(
        target=_worker_function,
        args=(
            "jieba_fast_dat",
            NUM_REPETITIONS,
            custom_dict_path,
            user_dict_path,
            long_text,
            results_queue,
        ),
    )

    print(f"\nRunning tests with {NUM_REPETITIONS} repetitions per measurement...")
    print("Tests for jieba and jieba_fast_dat are run in isolated processes.")
    print("This may take a moment...")

    jieba_process.start()
    jieba_fast_dat_process.start()

    jieba_process.join()
    jieba_fast_dat_process.join()

    jieba_results = {}
    jieba_fast_dat_results = {}

    while not results_queue.empty():
        result = results_queue.get()
        if "jieba" in result:
            jieba_results = result["jieba"]
        elif "jieba_fast_dat" in result:
            jieba_fast_dat_results = result["jieba_fast_dat"]

    if not jieba_results or not jieba_fast_dat_results:
        print("\nError: Failed to collect results from one or both processes.")
        sys.exit(1)

    print("\n--- Summary ---")
    print(f"{'Metric':<40} {'Jieba':>15} {'Jieba_fast_dat':>15}")
    print(
        f"{'Initial Main Dict Load (No Cache)':<40} "
        f"{jieba_results['first_init_time']:>15.7f}s "
        f"{jieba_fast_dat_results['first_init_time']:>15.7f}s"
    )
    print(
        f"{'Initial Main Dict Load (With Cache)':<40} "
        f"{jieba_results['second_init_time']:>15.7f}s "
        f"{jieba_fast_dat_results['second_init_time']:>15.7f}s"
    )
    print(
        f"{'User Dict Load':<40} {jieba_results['user_dict_load_time']:>15.7f}s "
        f"{jieba_fast_dat_results['user_dict_load_time']:>15.7f}s"
    )
    print(
        f"{'Word Segmentation (HMM=False)':<40} "
        f"{jieba_results['cut_time_HMM_False']:>15.7f}s "
        f"{jieba_fast_dat_results['cut_time_HMM_False']:>15.7f}s"
    )
    print(
        f"{'POS Tagging (HMM=False)':<40} "
        f"{jieba_results['pos_time_HMM_False']:>15.7f}s "
        f"{jieba_fast_dat_results['pos_time_HMM_False']:>15.7f}s"
    )
    print(
        f"{'Word Segmentation (HMM=True)':<40} "
        f"{jieba_results['cut_time_HMM_True']:>15.7f}s "
        f"{jieba_fast_dat_results['cut_time_HMM_True']:>15.7f}s"
    )
    print(
        f"{'POS Tagging (HMM=True)':<40} "
        f"{jieba_results['pos_time_HMM_True']:>15.7f}s "
        f"{jieba_fast_dat_results['pos_time_HMM_True']:>15.7f}s"
    )


if __name__ == "__main__":
    run_performance_test()
