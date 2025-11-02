import jieba_fast_dat
import jieba_fast_dat.posseg as posseg
import logging
import time

# system_dict_path = "extra_dict/dict.txt.big"
system_dict_path = ""
user_dict_path = ""
# user_dict_path = "extra_dict/dict.txt.big"
# user_dict_path = "extra_dict/dict.txt.big.tw_nerd.txt"
# user_dict_path = "test/test_dicts/test_user_dict_add.txt"

if system_dict_path:
    jieba_fast_dat.set_dictionary(system_dict_path)
if user_dict_path:
    jieba_fast_dat.load_userdict(user_dict_path)


time_start = time.time()
jieba_fast_dat.initialize()
time_end = time.time()
print("init time spend: {:.6f}s".format(time_end - time_start))


console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

logging.getLogger().addHandler(console)
logging.getLogger().setLevel(logging.DEBUG)

text_chinese = "我愛台灣臭豆腐跟小籠包, 覺得好吃,東北季風發威！4縣市豪大雨特報「雨下整夜」　一路濕到這天"
text_mixed = "我喜歡 Python 程式設計，版本是 3.9，價格是 $100.50。"


def test_word_segmentation():
    logging.info(f"===== Testing word segmentation for: {text_chinese}")
    result_chinese = list(jieba_fast_dat.cut(text_chinese, HMM=False))
    logging.info(f"Segmented words(no HMM): {result_chinese}")
    logging.info("-----------")
    result_chinese = list(jieba_fast_dat.cut(text_chinese, HMM=True))
    logging.info(f"Segmented words(with HMM): {result_chinese}")
    logging.info("-----------")
    assert len(result_chinese) > 0

    logging.info(f"Testing word segmentation for: {text_mixed}")
    result_mixed = list(jieba_fast_dat.cut(text_mixed))
    logging.info(f"Segmented words: {result_mixed}")
    assert len(result_mixed) > 0
    assert "Python" in result_mixed
    assert "3.9" in result_mixed
    assert "100.50" in result_mixed


def test_pos_tagging():
    logging.info(f"====== Testing POS tagging for: {text_chinese}")
    result_chinese = list(posseg.cut(text_chinese, HMM=False))
    logging.info(f"POS tagged words (no HMM): {result_chinese}")
    logging.info("-----------")
    result_chinese = list(posseg.cut(text_chinese, HMM=True))
    logging.info(f"POS tagged words (with HMM): {result_chinese}")
    logging.info("-----------")
    assert len(result_chinese) > 0
    non_x_tags_found_chinese = False
    for word, flag in result_chinese:
        assert word is not None
        assert len(flag) > 0  # Ensure flag is not empty
        if flag != "x":
            non_x_tags_found_chinese = True
    assert non_x_tags_found_chinese

    logging.info(f"Testing POS tagging for: {text_mixed}")
    result_mixed = list(posseg.cut(text_mixed, HMM=True))
    logging.info(f"POS tagged words: {result_mixed}")
    assert len(result_mixed) > 0
    # Check for specific English word and number tags
    python_found = False
    version_found = False
    price_found = False
    non_x_tags_found_mixed = False
    for word, flag in result_mixed:
        if word == "Python" and flag == "eng":
            python_found = True
        if word == "3.9" and flag == "m":
            version_found = True
        if word == "100.50" and flag == "m":
            price_found = True
        if flag != "x":
            non_x_tags_found_mixed = True
    assert python_found
    assert version_found
    assert price_found
    assert non_x_tags_found_mixed
    for word, flag in result_mixed:
        assert word is not None
        assert len(flag) > 0  # Ensure flag is not empty


if __name__ == "__main__":
    test_word_segmentation()
    test_pos_tagging()
    print("All tests passed!")
