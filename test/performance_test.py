import time
import os
import sys
import psutil
from jieba_fast_dat import Tokenizer, DEFAULT_DICT_NAME
import logging

def measure_performance():
    logging.info("Starting performance test...")

    # Get the absolute path to the default dictionary
    # Assuming DEFAULT_DICT_NAME is in the same directory as the module
    module_path = os.path.dirname(os.path.abspath(__file__))
    dict_path = os.path.join(module_path, "..", "jieba_fast_dat", DEFAULT_DICT_NAME)

    # Ensure the dictionary file exists
    if not os.path.exists(dict_path):
        logging.info(f"Error: Dictionary file not found at {dict_path}")
        return

    # Measure memory usage before initialization
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
    logging.info(f"Memory usage before initialization: {mem_before:.2f} MB")

    # Measure loading time
    start_time = time.time()
    tokenizer = Tokenizer()
    tokenizer.initialize(dict_path)
    end_time = time.time()

    loading_time = end_time - start_time
    logging.info(f"Dictionary loading time: {loading_time:.4f} seconds")

    # Measure memory usage after initialization
    mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
    logging.info(f"Memory usage after initialization: {mem_after:.2f} MB")
    logging.info(f"Memory increase: {mem_after - mem_before:.2f} MB")

    # Basic segmentation test to ensure functionality
    test_sentence = "我愛北京天安門"
    words = tokenizer.lcut(test_sentence)
    logging.info(f"Test sentence: {test_sentence}")
    logging.info(f"Segmented words: {words}")

    # Unload mmap resources if applicable (Tokenizer should handle this on destruction or re-init)
    # For explicit testing, we can try to unload if the mmap_loader is exposed
    if hasattr(tokenizer, 'mmap_loader') and tokenizer.mmap_loader.is_loaded():
        tokenizer.mmap_loader.unload()
        logging.info("Mmap resources explicitly unloaded.")

if __name__ == "__main__":
    measure_performance()
