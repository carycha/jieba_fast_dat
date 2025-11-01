

def test_cut_file_content(tokenizer_base, main_test_text_path):
    """
    Tests that `jieba_fast_dat.cut` can process raw bytes from a file.
    """
    with open(main_test_text_path, 'rb') as f:
        content_bytes = f.read()
    content_str = content_bytes.decode('utf-8')
    
    # The library should handle internal decoding from bytes
    words = list(tokenizer_base.cut(content_str, HMM=False))

    # We expect the segmentation to be correct based on the dictionary
    assert "程式設計師" in words
    assert "賴清德" in words
    assert "柯文哲" in words

def test_pos_cut_file_content(pos_tokenizer, main_test_text_path):
    """
    Tests that `jieba_fast_dat.posseg.cut` can process raw bytes from a file
    and return correct POS tags.
    """
    with open(main_test_text_path, 'rb') as f:
        content_bytes = f.read()
    content_str = content_bytes.decode('utf-8')

    words = list(pos_tokenizer.cut(content_str, HMM=False))

    # Create a dictionary of word->flag for easy lookup
    word_map = {w.word: w.flag for w in words}

    # Assert that key words from both base and user dicts are tagged correctly
    assert word_map.get("賴清德") == "nr"
    assert word_map.get("柯文哲") == "nr"
    assert word_map.get("政治人物") == "n"
    assert word_map.get("程式設計師") == "n"
