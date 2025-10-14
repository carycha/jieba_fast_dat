import jieba_fast_dat
import jieba_fast_dat.posseg as posseg

def test_word_segmentation():
    text = "我愛北京天安門"
    print(f"Testing word segmentation for: {text}")
    result = list(jieba_fast_dat.cut(text))
    print(f"Segmented words: {result}")
    assert len(result) > 0
    assert "我" in result
    assert "北京" in result

def test_pos_tagging():
    text = "我愛北京天安門"
    print(f"Testing POS tagging for: {text}")
    result = list(posseg.cut(text))
    print(f"POS tagged words: {result}")
    assert len(result) > 0
    for word, flag in result:
        assert word is not None
        assert flag != 'x' # Ensure not all tags are 'x'
        assert len(flag) > 0 # Ensure flag is not empty
