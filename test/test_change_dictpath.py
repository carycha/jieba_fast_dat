
import jieba_fast_dat

def test_set_dictionary_changes_behavior(dict_base_path, dict_add_path):
    """
    Tests that jieba_fast_dat.set_dictionary() correctly changes the tokenizer's behavior.
    It checks if segmentation behavior changes after switching dictionaries.
    """
    test_sent = "程式設計師正在研究元宇宙"

    # 1. Initialize with the base dictionary.
    # In base_dict, "程式設計師" is a word, but "元宇宙" is not.
    jieba_fast_dat.set_dictionary(dict_base_path)
    jieba_fast_dat.initialize()
    
    seg_list_base = list(jieba_fast_dat.cut(test_sent))
    assert "程式設計師" in seg_list_base, "'程式設計師' should be a single token with the base dictionary"
    assert "元宇宙" not in seg_list_base, "'元宇宙' should be split with the base dictionary"

    # 2. Change to the 'add' dictionary.
    # In add_dict, "元宇宙" is a word, but "程式設計師" is not.
    jieba_fast_dat.set_dictionary(dict_add_path)
    # Re-initialization should be handled by set_dictionary implicitly.

    seg_list_add = list(jieba_fast_dat.cut(test_sent))
    assert "元宇宙" in seg_list_add, "'元宇宙' should be a single token with the add dictionary"
    assert "程式設計師" not in seg_list_add, "'程式設計師' should be split with the add dictionary"

    # 3. Restore the original dictionary to avoid side effects on other tests.
    # This is crucial for maintaining a clean state between tests.
    jieba_fast_dat.set_dictionary(dict_base_path)
    jieba_fast_dat.initialize()
