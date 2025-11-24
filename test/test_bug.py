import gc  # Add import gc

from jieba.posseg import POSTokenizer as OriginalPOSTokenizer

from jieba_fast_dat.posseg import POSTokenizer as FastPOSTokenizer


def test_bug_repeated_char_segmentation(
    fast_pos_tokenizer: FastPOSTokenizer, orig_pos_tokenizer: OriginalPOSTokenizer
) -> None:
    """
    This test addresses a potential bug in the segmentation of phrases
    with repeated characters like "一是為這".
    It ensures that each character is segmented correctly with its proper POS tag.
    """
    test_sent = "一是為這"

    # Force garbage collection after tokenizer creation to ensure C++ objects
    # are released
    gc.collect()

    # Fast tokenizer
    fast_words = fast_pos_tokenizer.cut(test_sent, HMM=False)
    fast_result = [(w.word, w.flag) for w in fast_words]

    # Original jieba tokenizer
    orig_words = orig_pos_tokenizer.cut(test_sent, HMM=False)
    orig_result = [
        (w.word, w.flag) for w in orig_words
    ]  # Original jieba returns pair objects with .word and .flag

    assert fast_result == orig_result, (
        f"Segmentation of '{test_sent}' failed.\n"
        f"Fast: {fast_result}\n"
        f"Orig: {orig_result}"
    )
