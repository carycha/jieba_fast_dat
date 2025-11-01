
def test_default_behavior(tfidf_extractor, main_test_text):
    """
    Tests the default behavior of extract_tags, expecting a list of strings.
    """
    tags = tfidf_extractor.extract_tags(main_test_text, topK=4)
    assert tags == ["區塊鏈", "人工智慧", "台灣", "台北"]
    assert all(isinstance(t, str) for t in tags)

def test_with_weight(tfidf_extractor, main_test_text):
    """
    Tests the `withWeight=True` functionality.
    """
    tags = tfidf_extractor.extract_tags(main_test_text, topK=4, withWeight=True)
    assert len(tags) == 4
    assert all(isinstance(t, tuple) and len(t) == 2 for t in tags)
    assert all(isinstance(t[0], str) and isinstance(t[1], float) for t in tags)
    # Just check the words, as weights can be volatile
    assert [t[0] for t in tags] == ["區塊鏈", "人工智慧", "台灣", "台北"]

def test_with_stop_words(tfidf_extractor, main_test_text, stop_words_path):
    """
    Tests that stop words are correctly filtered by directly modifying the 
    extractor's stop_words set.
    """
    # Load stop words from the file
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stop_words = {line.strip() for line in f}

    # Set the stop words directly on the instance
    tfidf_extractor.stop_words = stop_words

    # The top 4 tags are normally ["區塊鏈", "人工智慧", "台灣", "台北"]
    # Our stop words file contains "台灣" and "台北".
    tags = tfidf_extractor.extract_tags(main_test_text, topK=4)

    assert "台灣" not in tags
    assert "台北" not in tags
    assert len(tags) == 4
    assert "區塊鏈" in tags
    assert "人工智慧" in tags