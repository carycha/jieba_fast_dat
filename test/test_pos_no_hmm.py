

def test_pos_no_hmm_basic(pos_tokenizer):
    """
    Test basic POS tagging with HMM=False. It should correctly identify words
    from the main dictionary and user dictionary.
    """
    test_sent = "賴清德是台灣的政治人物。"
    words = list(pos_tokenizer.cut(test_sent, HMM=False))
    
    expected = [('賴清德', 'nr'), ('是', 'v'), ('台灣', 'ns'), ('的', 'uj'), ('政治人物', 'n'), ('。', 'x')]
    assert [(w.word, w.flag) for w in words] == expected, f"Failed to segment correctly. Got: {[(w.word, w.flag) for w in words]}"

def test_pos_no_hmm_with_user_dict(pos_tokenizer):
    """
    Tests that words from the user dictionary are correctly tagged.
    """
    test_sent = "柯文哲是知名的政治人物。"
    words = list(pos_tokenizer.cut(test_sent, HMM=False))

    assert ('柯文哲', 'nr') in [(w.word, w.flag) for w in words], "Word from user dict was not correctly tagged."

def test_pos_no_hmm_empty_sentence(pos_tokenizer):
    """
    Test POS tagging with HMM=False and an empty sentence.
    """
    words = list(pos_tokenizer.cut("", HMM=False))
    assert len(words) == 0

