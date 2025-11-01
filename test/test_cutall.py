
def test_cutall_basic(tokenizer_base):
    """
    Tests basic cut_all functionality. The HMM parameter is explicitly set to False
    to make the test's intention clear.
    """
    test_sent = "台灣的台北是一個充滿活力的城市。"
    # We pass HMM=False explicitly, instead of hiding it in a fixture.
    words = list(tokenizer_base.cut(test_sent, cut_all=True, HMM=False))
    
    # In cut_all mode, we expect to see all possible segmentations.
    expected_words = {"台灣", "的", "台北", "是", "一個", "充滿", "活力的", "的", "城市"}
    # The result of cut_all is extensive, so we check for presence, not equality.
    for word in expected_words:
        assert word in words, f"Expected token '{word}' not found in cut_all output."

def test_cutall_with_user_dict(tokenizer_base):
    """
    Tests that cut_all correctly includes words from the user dictionary.
    """
    test_sent = "賴清德和柯文哲是台灣的政治人物。"
    words = list(tokenizer_base.cut(test_sent, cut_all=True, HMM=False))
    
    # Words from the user dictionary should be present.
    assert "賴清德" in words
    assert "柯文哲" in words
    # Words from the base dictionary should also be present.
    assert "政治人物" in words

def test_cutall_empty_sentence(tokenizer_base):
    """
    Tests cut_all with an empty sentence.
    """
    words = list(tokenizer_base.cut("", cut_all=True, HMM=False))
    assert len(words) == 0