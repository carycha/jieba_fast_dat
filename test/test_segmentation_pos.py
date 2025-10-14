import jieba_fast_dat
import jieba_fast_dat.posseg as posseg

def test_word_segmentation():
    text_chinese = "我愛北京天安門"
    print(f"Testing word segmentation for: {text_chinese}")
    result_chinese = list(jieba_fast_dat.cut(text_chinese))
    print(f"Segmented words: {result_chinese}")
    assert len(result_chinese) > 0
    assert "我" in result_chinese
    assert "北京" in result_chinese

    text_mixed = "我喜歡 Python 程式設計，版本是 3.9，價格是 $100.50。"
    print(f"Testing word segmentation for: {text_mixed}")
    result_mixed = list(jieba_fast_dat.cut(text_mixed))
    print(f"Segmented words: {result_mixed}")
    assert len(result_mixed) > 0
    assert "Python" in result_mixed
    assert "3.9" in result_mixed
    assert "100.50" in result_mixed

def test_pos_tagging():
    text_chinese = "我愛北京天安門"
    print(f"Testing POS tagging for: {text_chinese}")
    result_chinese = list(posseg.cut(text_chinese))
    print(f"POS tagged words: {result_chinese}")
    assert len(result_chinese) > 0
    non_x_tags_found_chinese = False
    for word, flag in result_chinese:
        assert word is not None
        assert len(flag) > 0  # Ensure flag is not empty
        if flag != 'x':
            non_x_tags_found_chinese = True
    assert non_x_tags_found_chinese

    text_mixed = "我喜歡 Python 程式設計，版本是 3.9，價格是 $100.50。"
    print(f"Testing POS tagging for: {text_mixed}")
    result_mixed = list(posseg.cut(text_mixed))
    print(f"POS tagged words: {result_mixed}")
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
        if flag != 'x':
            non_x_tags_found_mixed = True
    assert python_found
    assert version_found
    assert price_found
    assert non_x_tags_found_mixed
    for word, flag in result_mixed:
        assert word is not None
        assert len(flag) > 0  # Ensure flag is not empty
