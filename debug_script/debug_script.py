import jieba as original_jieba
from jieba import posseg as original_posseg

import jieba_fast_dat
from jieba_fast_dat import posseg


def run_debug():
    """
    Runs a side-by-side comparison of the POS tagging implementations
    and prints their debug output directly to stdout.
    """
    text = "台灣的台北是一個充滿活力的城市，這裡有許多電腦和手機的程式設計師。"
    hmm = True

    print("--- Running Original Jieba ---")
    # Initialize tokenizer
    orig_tokenizer = original_posseg.POSTokenizer(original_jieba.Tokenizer())
    orig_result = orig_tokenizer.lcut(text, HMM=hmm)
    print("--- Finished Original Jieba ---")

    print("\n\n--- Running Fast Jieba ---")
    # Initialize tokenizer
    fast_tokenizer = posseg.POSTokenizer(jieba_fast_dat.Tokenizer())
    fast_result = fast_tokenizer.lcut(text, HMM=hmm)
    print("--- Finished Fast Jieba ---")

    # Final comparison
    orig_pairs = [(p.word, p.flag) for p in orig_result]
    fast_pairs = [(p.word, p.flag) for p in fast_result]

    print("\n\n--- Comparison ---")
    print(f"Original: {orig_pairs}")
    print(f"Fast:     {fast_pairs}")
    if orig_pairs == fast_pairs:
        print("Results are IDENTICAL.")
    else:
        print("Results are DIFFERENT.")


if __name__ == "__main__":
    run_debug()
