from jieba import Tokenizer as OriginalTokenizer
from jieba.posseg import POSTokenizer as OriginalPOSTokenizer

from jieba_fast_dat import Tokenizer as FastTokenizer
from jieba_fast_dat.posseg import POSTokenizer as FastPOSTokenizer


def test_cut_file_content(
    fast_tokenizer: FastTokenizer,
    orig_tokenizer: OriginalTokenizer,
    main_test_text_path: str,
) -> None:
    """
    Tests that `jieba_fast_dat.cut` can process raw bytes from a file,
    comparing with original jieba.
    """
    with open(main_test_text_path, "rb") as f:
        content_bytes = f.read()
    content_str = content_bytes.decode("utf-8")

    fast_words = list(fast_tokenizer.cut(content_str, HMM=False))
    orig_words = list(orig_tokenizer.cut(content_str, HMM=False))

    assert fast_words == orig_words, (
        f"test_cut_file_content failed:\nFast: {fast_words}\nOrig: {orig_words}"
    )


def test_pos_cut_file_content(
    fast_pos_tokenizer: FastPOSTokenizer,
    orig_pos_tokenizer: OriginalPOSTokenizer,
    main_test_text_path: str,
) -> None:
    """
    Tests that `jieba_fast_dat.posseg.cut` can process raw bytes from a file
    and return correct POS tags, comparing with original jieba.
    """
    with open(main_test_text_path, "rb") as f:
        content_bytes = f.read()
    content_str = content_bytes.decode("utf-8")

    fast_words = list(fast_pos_tokenizer.cut(content_str, HMM=False))
    orig_words = list(orig_pos_tokenizer.cut(content_str, HMM=False))

    fast_words_tuples = [(p.word, p.flag) for p in fast_words]
    orig_words_tuples = [(p.word, p.flag) for p in orig_words]

    assert fast_words_tuples == orig_words_tuples, (
        "test_pos_cut_file_content failed:\n"
        f"Fast: {fast_words_tuples}\n"
        f"Orig: {orig_words_tuples}"
    )
