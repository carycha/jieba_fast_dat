# conftest.py
import pytest
import jieba_fast_dat


@pytest.fixture(autouse=True)
def clean_jieba_environment():
    """
    Ensures that each test runs in a clean jieba environment
    by re-initializing the default dictionary before each test.
    """
    # Force disable parallel mode if it was enabled by a previous test
    jieba_fast_dat.disable_parallel()

    # Re-create the default Tokenizer instance to reset its state
    jieba_fast_dat.dt = jieba_fast_dat.Tokenizer()

    # Re-bind global functions to the new Tokenizer instance's methods
    jieba_fast_dat.add_word = jieba_fast_dat.dt.add_word
    jieba_fast_dat.calc = jieba_fast_dat.dt.calc
    jieba_fast_dat.cut = jieba_fast_dat.dt.cut
    jieba_fast_dat.lcut = jieba_fast_dat.dt.lcut
    jieba_fast_dat.cut_for_search = jieba_fast_dat.dt.cut_for_search
    jieba_fast_dat.lcut_for_search = jieba_fast_dat.dt.lcut_for_search
    jieba_fast_dat.del_word = jieba_fast_dat.dt.del_word
    jieba_fast_dat.get_DAG = jieba_fast_dat.dt.get_DAG
    jieba_fast_dat.get_dict_file = jieba_fast_dat.dt.get_dict_file
    jieba_fast_dat.initialize = jieba_fast_dat.dt.initialize
    jieba_fast_dat.load_userdict = jieba_fast_dat.dt.load_userdict
    jieba_fast_dat.set_dictionary = jieba_fast_dat.dt.set_dictionary
    jieba_fast_dat.suggest_freq = jieba_fast_dat.dt.suggest_freq
    jieba_fast_dat.tokenize = jieba_fast_dat.dt.tokenize
    jieba_fast_dat.user_word_tag_tab = jieba_fast_dat.dt.user_word_tag_tab

    # Re-initialize the newly created Tokenizer with default dictionary
    jieba_fast_dat.initialize()

    # Also reset posseg's default tokenizer
    import jieba_fast_dat.posseg as posseg

    posseg.dt = posseg.POSTokenizer()
    posseg.cut = posseg.dt.cut
    posseg.lcut = posseg.dt.lcut
    posseg.initialize = posseg.dt.initialize
    posseg.initialize()


def pytest_sessionfinish(session):
    if hasattr(session.config, "_performance_summaries"):
        print("\n--- Performance Summaries ---")
        for summary in session.config._performance_summaries:
            print(summary)
        print("-----------------------------")
