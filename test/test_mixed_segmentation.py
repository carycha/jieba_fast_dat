import jieba_fast_dat
import jieba_fast_dat.posseg as posseg
import logging

# Configure logging for better visibility during tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

text_complex_mixed = (
    r"""中文 English123 數字456.78 符號!@#$ %^&*()_+-=~`{}[]|\:;"'<>,.?/ 和空格"""
)


def test_mixed_segmentation_hmm_true():
    logging.info(
        f"====== Testing mixed segmentation (HMM=True) for: {text_complex_mixed}"
    )
    cut_result = list(jieba_fast_dat.cut(text_complex_mixed, HMM=True))
    logging.info(f"jieba.cut result (HMM=True): {cut_result}")
    expected_cut_parts = [
        "中文",
        " ",
        "English123",
        " ",
        "數字",
        "456.78",
        " ",
        "符號",
        "!",
        "@",
        "#",
        "$",
        " ",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "_",
        "+",
        "-",
        "=",
        "~",
        "`",
        "{",
        "}",
        "[",
        "]",
        "|",
        "\\",
        ":",
        ";",
        '"',
        "'",
        "<",
        ">",
        ",",
        ".",
        "?",
        "/",
        " ",
        "和",
        "空格",
    ]
    assert cut_result == expected_cut_parts


def test_mixed_segmentation_hmm_false():
    logging.info(
        f"====== Testing mixed segmentation (HMM=False) for: {text_complex_mixed}"
    )
    cut_result = list(jieba_fast_dat.cut(text_complex_mixed, HMM=False))
    logging.info(f"jieba.cut result (HMM=False): {cut_result}")
    expected_cut_parts = [
        "中文",
        " ",
        "English123",
        " ",
        "數字",
        "456.78",
        " ",
        "符號",
        "!",
        "@",
        "#",
        "$",
        " ",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "_",
        "+",
        "-",
        "=",
        "~",
        "`",
        "{",
        "}",
        "[",
        "]",
        "|",
        "\\",
        ":",
        ";",
        '"',
        "'",
        "<",
        ">",
        ",",
        ".",
        "?",
        "/",
        " ",
        "和",
        "空格",
    ]
    assert cut_result == expected_cut_parts


def test_mixed_pos_tagging_hmm_true():
    logging.info(
        f"====== Testing mixed POS tagging (HMM=True) for: {text_complex_mixed}"
    )
    pos_result = list(posseg.cut(text_complex_mixed, HMM=True))
    pos_result_tuples = [(word, flag) for word, flag in pos_result]
    logging.info(f"POS tagged words (HMM=True): {pos_result_tuples}")
    expected_pos_parts = [
        ("中文", "nz"),
        (" ", "x"),
        ("English123", "eng"),
        (" ", "x"),
        ("數字", "n"),
        ("456.78", "m"),
        (" ", "x"),
        ("符號", "n"),
        ("!", "x"),
        ("@", "x"),
        ("#", "x"),
        ("$", "x"),
        (" ", "x"),
        ("%", "x"),
        ("^", "x"),
        ("&", "x"),
        ("*", "x"),
        ("(", "x"),
        (")", "x"),
        ("_", "x"),
        ("+", "x"),
        ("-", "x"),
        ("=", "x"),
        ("~", "x"),
        ("`", "x"),
        ("{", "x"),
        ("}", "x"),
        ("[", "x"),
        ("]", "x"),
        ("|", "x"),
        ("\\", "x"),
        (":", "x"),
        (";", "x"),
        ('"', "x"),
        ("'", "x"),
        ("<", "x"),
        (">", "x"),
        (",", "x"),
        (".", "x"),
        ("?", "x"),
        ("/", "x"),
        (" ", "x"),
        ("和", "c"),
        ("空格", "n"),
    ]
    assert pos_result_tuples == expected_pos_parts


def test_mixed_pos_tagging_hmm_false():
    logging.info(
        f"====== Testing mixed POS tagging (HMM=False) for: {text_complex_mixed}"
    )
    pos_result = list(posseg.cut(text_complex_mixed, HMM=False))
    pos_result_tuples = [(word, flag) for word, flag in pos_result]
    logging.info(f"POS tagged words (HMM=False): {pos_result_tuples}")
    expected_pos_parts = [
        ("中文", "nz"),
        (" ", "x"),
        ("English123", "eng"),
        (" ", "x"),
        ("數字", "n"),
        ("456.78", "m"),
        (" ", "x"),
        ("符號", "n"),
        ("!", "x"),
        ("@", "x"),
        ("#", "x"),
        ("$", "x"),
        (" ", "x"),
        ("%", "x"),
        ("^", "x"),
        ("&", "x"),
        ("*", "x"),
        ("(", "x"),
        (")", "x"),
        ("_", "x"),
        ("+", "x"),
        ("-", "x"),
        ("=", "x"),
        ("~", "x"),
        ("`", "x"),
        ("{", "x"),
        ("}", "x"),
        ("[", "x"),
        ("]", "x"),
        ("|", "x"),
        ("\\", "x"),
        (":", "x"),
        (";", "x"),
        ('"', "x"),
        ("'", "x"),
        ("<", "x"),
        (">", "x"),
        (",", "x"),
        (".", "x"),
        ("?", "x"),
        ("/", "x"),
        (" ", "x"),
        ("和", "c"),
        ("空格", "n"),
    ]
    assert pos_result_tuples == expected_pos_parts
