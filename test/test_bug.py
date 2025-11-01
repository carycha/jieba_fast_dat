
def test_bug_repeated_char_segmentation(pos_tokenizer):
    """
    This test addresses a potential bug in the segmentation of phrases 
    with repeated characters like "又跛又啞".
    It ensures that each character is segmented correctly with its proper POS tag.
    """
    # The phrase "又跛又啞" means "lame and mute".
    # "又" is an adverb, "跛" and "啞" are adjectives.
    words = pos_tokenizer.cut("又跛又啞", HMM=False)
    result = [(w.word, w.flag) for w in words]

    # Based on common POS tagging standards, we expect something like this.
    # The exact flags might differ, but the segmentation should be ['又', '跛', '又', '啞'].
    # Let's find out the library's actual tags and lock them in.
    # After running, we find the tags are 'd' (adverb) and 'a' (adjective).
    expected = [('又', 'd'), ('跛', 'a'), ('又', 'd'), ('啞', 'a')]

    assert result == expected, f"Segmentation of '又跛又啞' failed. Expected {expected}, got {result}"