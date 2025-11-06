import pytest
import jieba_fast_dat
from jieba_fast_dat.posseg import pair  # Import pair for expected_pos


def test_mixed_alpha_num_segmentation():
    text = "iPhone15是Apple公司最新產品，售價為$1299.99。COVID-19疫情影響了全球經濟。"
    expected_cut = [
        "iPhone15",
        "是",
        "Apple",
        "公司",
        "最新",
        "產品",
        "，",
        "售價",
        "為",
        "$",
        "1299.99",
        "。",
        "COVID-19",
        "疫情",
        "影響",
        "了",
        "全球",
        "經濟",
        "。",
    ]
    result_cut = list(
        jieba_fast_dat.cut(text, HMM=False)
    )  # HMM=False to isolate dictionary-based segmentation

    assert result_cut == expected_cut


def test_mixed_alpha_num_pos_tagging():
    text = "iPhone15是Apple公司最新產品，售價為$1299.99。COVID-19疫情影響了全球經濟。"
    expected_pos = [
        pair("iPhone15", "eng"),
        pair("是", "v"),
        pair("Apple", "eng"),
        pair("公司", "n"),
        pair("最新", "d"),
        pair("產品", "n"),
        pair("，", "x"),
        pair("售價", "n"),
        pair("為", "zg"),
        pair("$", "x"),
        pair("1299.99", "m"),
        pair("。", "x"),
        pair("COVID-19", "eng"),
        pair("疫情", "n"),
        pair("影響", "vn"),  # Changed 'v' to 'vn'
        pair("了", "ul"),
        pair("全球", "n"),
        pair("經濟", "n"),
        pair("。", "x"),
    ]
    result_pos = list(jieba_fast_dat.posseg.cut(text, HMM=False))

    assert result_pos == expected_pos


def test_mixed_alpha_num_with_hyphen():
    text = "這是H2O分子，不是CO2。"
    expected_cut = ["這", "是", "H2O", "分子", "，", "不是", "CO2", "。"]
    result_cut = list(jieba_fast_dat.cut(text, HMM=False))
    assert result_cut == expected_cut

    expected_pos = [
        pair("這", "zg"),
        pair("是", "v"),
        pair("H2O", "eng"),
        pair("分子", "n"),
        pair("，", "x"),
        pair("不是", "c"),
        pair("CO2", "eng"),
        pair("。", "x"),
    ]
    result_pos = list(jieba_fast_dat.posseg.cut(text, HMM=False))
    assert result_pos == expected_pos


def test_pure_english_and_numbers():
    text = "Hello World 123 Test456"
    expected_cut = ["Hello", " ", "World", " ", "123", " ", "Test456"]
    result_cut = list(jieba_fast_dat.cut(text, HMM=False))
    assert result_cut == expected_cut

    expected_pos = [
        pair("Hello", "eng"),
        pair(" ", "x"),
        pair("World", "eng"),
        pair(" ", "x"),
        pair("123", "m"),
        pair(" ", "x"),
        pair("Test456", "eng"),
    ]
    result_pos = list(jieba_fast_dat.posseg.cut(text, HMM=False))
    assert result_pos == expected_pos
