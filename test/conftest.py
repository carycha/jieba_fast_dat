import os
import pytest
import jieba_fast_dat

# Define paths to our test resources
TEST_DICTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_dicts")
TEST_TEXTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_texts")

@pytest.fixture(scope="session")
def dicts_dir():
    return TEST_DICTS_DIR

@pytest.fixture(scope="session")
def texts_dir():
    return TEST_TEXTS_DIR

@pytest.fixture(scope="session")
def dict_base_path():
    return os.path.join(TEST_DICTS_DIR, "test_dict_base.txt")

@pytest.fixture(scope="session")
def user_dict_base_path():
    return os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")

@pytest.fixture(scope="session")
def dict_add_path():
    return os.path.join(TEST_DICTS_DIR, "test_dict_add.txt")

@pytest.fixture(scope="session")
def main_test_text_path():
    return os.path.join(TEST_TEXTS_DIR, "main_test_text.txt")

@pytest.fixture
def main_test_text(main_test_text_path):
    with open(main_test_text_path, 'r', encoding='utf-8') as f:
        return f.read()

@pytest.fixture(scope="session")
def idf_base_path():
    return os.path.join(TEST_DICTS_DIR, "text_idf_base.txt")

@pytest.fixture(scope="session")
def stop_words_path():
    return os.path.join(TEST_DICTS_DIR, "test_stop_words.txt")

@pytest.fixture(scope="module")
def tfidf_extractor(dict_base_path, idf_base_path):
    """
    Provides a TFIDF extractor instance initialized with custom dictionaries.
    """
    custom_tokenizer = jieba_fast_dat.Tokenizer(dictionary=dict_base_path)
    
    # The original test had a weird override to force HMM=False.
    # We will replicate that here inside the fixture to keep tests clean.
    original_cut = custom_tokenizer.cut
    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False) # Force HMM=False
    custom_tokenizer.cut = new_cut

    extractor = jieba_fast_dat.analyse.TFIDF(idf_path=idf_base_path)
    extractor.tokenizer = custom_tokenizer
    return extractor

@pytest.fixture
def tokenizer_base(dict_base_path, user_dict_base_path):
    """
    Provides a clean, initialized tokenizer with custom dictionaries.
    """
    tokenizer = jieba_fast_dat.Tokenizer()
    tokenizer.set_dictionary(dict_base_path)
    tokenizer.load_userdict(user_dict_base_path)
    return tokenizer

@pytest.fixture
def pos_tokenizer(dict_base_path, user_dict_base_path):
    """
    Provides a correctly initialized posseg tokenizer.
    The key is to create a Tokenizer instance, load dicts into IT, 
    and then pass THAT instance to the POSTokenizer.
    """
    # 1. Create a custom tokenizer instance.
    custom_tokenizer = jieba_fast_dat.Tokenizer()
    
    # 2. Load dictionaries directly into this instance.
    custom_tokenizer.set_dictionary(dict_base_path)
    custom_tokenizer.load_userdict(user_dict_base_path)
    
    # 3. Create the POSTokenizer from the fully configured tokenizer instance.
    pos_tokenizer = jieba_fast_dat.posseg.POSTokenizer(tokenizer=custom_tokenizer)
    return pos_tokenizer
