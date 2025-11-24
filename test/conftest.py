import logging
import os
from collections.abc import Generator

import jieba as original_jieba
import jieba.analyse as original_jieba_analyse
import jieba.posseg as original_jieba_posseg
import pytest

import jieba_fast_dat
import jieba_fast_dat.analyse
import jieba_fast_dat.posseg

# --- Path Fixtures ---


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Base directory for test data."""
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def dicts_dir(test_data_dir: str) -> str:
    """Directory for dictionary files."""
    return os.path.join(test_data_dir, "test_dicts")


@pytest.fixture(scope="session", autouse=True)
def set_jieba_fast_dat_log_level() -> None:
    """
    Set jieba_fast_dat's internal logger level to WARNING to reduce noise during tests.
    """
    jieba_fast_dat.setLogLevel(logging.WARNING)


@pytest.fixture(scope="session")
def texts_dir(test_data_dir: str) -> str:
    """Directory for text files."""
    return os.path.join(test_data_dir, "test_texts")


@pytest.fixture(scope="session")
def dict_base_path(dicts_dir: str) -> str:
    """Path to the base dictionary."""
    return os.path.join(dicts_dir, "test_dict_base.txt")


@pytest.fixture(scope="session")
def user_dict_base_path(dicts_dir: str) -> str:
    """Path to the base user dictionary."""
    return os.path.join(dicts_dir, "test_user_dict_base.txt")


@pytest.fixture(scope="session")
def idf_base_path(dicts_dir: str) -> str:
    """Path to the base IDF file."""
    return os.path.join(dicts_dir, "text_idf_base.txt")


@pytest.fixture(scope="session")
def stop_words_path(dicts_dir: str) -> str:
    """Path to the stop words file."""
    return os.path.join(dicts_dir, "test_stop_words.txt")


@pytest.fixture(scope="session")
def main_test_text_path(texts_dir: str) -> str:
    """Path to the main test text file."""
    return os.path.join(texts_dir, "main_test_text.txt")


@pytest.fixture(scope="session")
def dict_add_path(dicts_dir: str) -> str:
    """Path to a dictionary for adding words."""
    return os.path.join(dicts_dir, "test_user_dict_base.txt")


@pytest.fixture(scope="session")
def dict_new_main_path(dicts_dir: str) -> str:
    """Path to an alternative main dictionary for testing set_dictionary behavior."""
    return os.path.join(dicts_dir, "test_dict_add.txt")


# --- Text Data Fixtures ---


@pytest.fixture(scope="session")
def main_test_text(texts_dir: str) -> str:
    """A standard Chinese text for general testing."""
    file_path = os.path.join(texts_dir, "main_test_text.txt")
    with open(file_path, encoding="utf-8") as f:
        return f.read().strip()


@pytest.fixture(scope="session")
def mixed_text() -> str:
    """A text with mixed Chinese, English, numbers, and symbols."""
    return "這是一句混合文本，包含English、數字123和符號#C++。我們來談談生成式AI。"


# --- Tokenizer Fixtures (Function-scoped for isolation) ---


@pytest.fixture(scope="function")
def orig_tokenizer(
    dict_base_path: str, user_dict_base_path: str
) -> original_jieba.Tokenizer:
    """
    Provides a fresh original jieba.Tokenizer instance for each test.
    Initialized with base and user dictionaries.
    """
    tokenizer = original_jieba.Tokenizer(dictionary=str(dict_base_path))
    tokenizer.load_userdict(str(user_dict_base_path))
    tokenizer.initialize()
    return tokenizer


@pytest.fixture(scope="function")
def fast_tokenizer(
    dict_base_path: str, user_dict_base_path: str
) -> Generator[jieba_fast_dat.Tokenizer, None, None]:
    """
    Provides a fresh jieba_fast_dat.Tokenizer instance for each test.
    Initialized with base and user dictionaries.
    """
    tokenizer = jieba_fast_dat.Tokenizer(dictionary=dict_base_path)
    tokenizer.load_userdict(user_dict_base_path)
    tokenizer.initialize()
    yield tokenizer


@pytest.fixture(scope="function")
def small_dict_tokenizer(dict_base_path: str) -> jieba_fast_dat.Tokenizer:
    """
    Provides a jieba_fast_dat.Tokenizer instance initialized with a small dictionary.
    """
    tokenizer = jieba_fast_dat.Tokenizer(dictionary=dict_base_path)
    tokenizer.initialize()
    return tokenizer


@pytest.fixture(scope="function")
def orig_pos_tokenizer(
    orig_tokenizer: original_jieba.Tokenizer,
) -> original_jieba_posseg.POSTokenizer:
    """
    Provides a fresh original jieba.posseg.POSTokenizer instance for each test.
    """
    original_jieba_posseg.initialize()  # 確保 posseg 模塊使用最新的詞典
    return original_jieba_posseg.POSTokenizer(tokenizer=orig_tokenizer)


@pytest.fixture(scope="function")
def fast_pos_tokenizer(
    fast_tokenizer: jieba_fast_dat.Tokenizer,
) -> jieba_fast_dat.posseg.POSTokenizer:
    """
    Provides a fresh jieba_fast_dat.posseg.POSTokenizer instance for each test.
    """
    jieba_fast_dat.posseg.initialize()  # 確保 posseg 模塊使用最新的詞典
    return jieba_fast_dat.posseg.POSTokenizer(tokenizer=fast_tokenizer)


@pytest.fixture(scope="function")
def pos_tokenizer(
    fast_pos_tokenizer: jieba_fast_dat.posseg.POSTokenizer,
) -> jieba_fast_dat.posseg.POSTokenizer:
    """
    Alias for fast_pos_tokenizer, used in some performance tests.
    """
    return fast_pos_tokenizer


# --- Analysis Fixtures ---


@pytest.fixture(scope="function")
def fast_tfidf_extractor(
    idf_base_path: str, fast_tokenizer: jieba_fast_dat.Tokenizer
) -> jieba_fast_dat.analyse.TFIDF:
    """
    Provides a TFIDF extractor from jieba_fast_dat, using the fast_tokenizer.
    """
    extractor = jieba_fast_dat.analyse.TFIDF(idf_path=idf_base_path)
    extractor.tokenizer = fast_tokenizer
    return extractor


@pytest.fixture(scope="function")
def orig_tfidf_extractor(
    idf_base_path: str, orig_tokenizer: original_jieba.Tokenizer
) -> original_jieba_analyse.TFIDF:
    """
    Provides a TFIDF extractor from original jieba, using the orig_tokenizer.
    """
    extractor = original_jieba_analyse.TFIDF(idf_path=str(idf_base_path))
    extractor.tokenizer = orig_tokenizer
    return extractor


@pytest.fixture(scope="function")
def fast_textrank_extractor(
    fast_pos_tokenizer: jieba_fast_dat.posseg.POSTokenizer,
) -> jieba_fast_dat.analyse.TextRank:
    """
    Provides a TextRank extractor from jieba_fast_dat.
    """
    extractor = jieba_fast_dat.analyse.TextRank()
    extractor.tokenizer = fast_pos_tokenizer
    return extractor


@pytest.fixture(scope="function")
def orig_textrank_extractor(
    orig_pos_tokenizer: original_jieba_posseg.POSTokenizer,
) -> original_jieba_analyse.TextRank:
    """
    Provides a TextRank extractor from original jieba.
    """
    extractor = original_jieba_analyse.TextRank()
    extractor.tokenizer = orig_pos_tokenizer
    return extractor
