import jieba_fast_dat
import jieba_fast_dat.analyse

def test_extract_tags_with_custom_idf(tfidf_extractor, main_test_text):
    """
    Tests extract_tags with a custom dictionary and IDF path.
    Asserts that the top tags are the ones with the highest IDF scores.
    """
    tags = tfidf_extractor.extract_tags(main_test_text, topK=4)

    # Based on text_idf_base.txt, "區塊鏈" (1000.0) and "人工智慧" (900.0) 
    # should be the top tags. "台灣" (10.0) and "台北" (9.5) should follow.
    # The original test expected "充滿", which is wrong because it's not in the IDF file.
    expected_top_tags = ["區塊鏈", "人工智慧", "台灣", "台北"]
    
    assert isinstance(tags, list)
    assert len(tags) == 4
    assert tags == expected_top_tags, f"Expected {expected_top_tags} but got {tags}"

def test_extract_tags_with_topK(tfidf_extractor, main_test_text):
    """
    Tests that the topK parameter works as expected.
    """
    tags = tfidf_extractor.extract_tags(main_test_text, topK=2)
    expected_top_tags = ["區塊鏈", "人工智慧"]
    assert len(tags) == 2
    assert tags == expected_top_tags

def test_extract_tags_with_default_idf():
    """
    Tests that the extractor works predictably with a default IDF.
    """
    # Here we use the global default tokenizer
    jieba_fast_dat.initialize()
    extractor = jieba_fast_dat.analyse.TFIDF() # Default IDF

    test_content = "這是一個測試句子，包含區塊鏈和人工智慧。"
    tags = extractor.extract_tags(test_content, topK=2)

    assert isinstance(tags, list)
    assert len(tags) == 2
    assert all(isinstance(tag, str) for tag in tags)