"""
test_extract_tags.py

Tests for the TF-IDF keyword extraction functionality, ensuring consistency
between jieba_fast_dat and the original jieba library.

This refactored module leverages fixtures from conftest.py to perform clean,
isolated comparisons without modifying global state.
"""

import jieba as original_jieba
import jieba.analyse as original_jieba_analyse
import pytest

import jieba_fast_dat.analyse

# Mark the entire module to be skipped if the original jieba is not installed
pytestmark = pytest.mark.skipif(
    not hasattr(original_jieba, "analyse"),
    reason="original jieba.analyse not installed",
)


@pytest.mark.consistency
class TestTfidfConsistency:
    """
    Tests for consistency in TF-IDF keyword extraction.
    """

    def test_default_extraction(
        self,
        orig_tfidf_extractor: original_jieba_analyse.TFIDF,
        fast_tfidf_extractor: jieba_fast_dat.analyse.TFIDF,
        main_test_text: str,
    ) -> None:
        """
        Tests the default behavior of extract_tags.
        """
        orig_tags = orig_tfidf_extractor.extract_tags(main_test_text, topK=5)
        fast_tags = fast_tfidf_extractor.extract_tags(main_test_text, topK=5)

        assert orig_tags == fast_tags, (
            "Default TF-IDF extraction mismatch.\n"
            f"Original: {orig_tags}\n"
            f"Fast:     {fast_tags}"
        )
        assert all(isinstance(t, str) for t in fast_tags)

    def test_with_weight(
        self,
        orig_tfidf_extractor: original_jieba_analyse.TFIDF,
        fast_tfidf_extractor: jieba_fast_dat.analyse.TFIDF,
        main_test_text: str,
    ) -> None:
        """
        Tests the `withWeight=True` functionality.
        """
        orig_tags = orig_tfidf_extractor.extract_tags(
            main_test_text, topK=5, withWeight=True
        )
        fast_tags = fast_tfidf_extractor.extract_tags(
            main_test_text, topK=5, withWeight=True
        )

        # Compare words and weights with a tolerance for floating point differences
        assert [item[0] for item in orig_tags] == [item[0] for item in fast_tags], (
            "TF-IDF withWeight=True word mismatch.\n"
            f"Original: {[item[0] for item in orig_tags]}\n"
            f"Fast:     {[item[0] for item in fast_tags]}"
        )

        for (orig_word, orig_w), (fast_word, fast_w) in zip(
            orig_tags, fast_tags, strict=True
        ):
            assert orig_word == fast_word
            assert abs(orig_w - fast_w) < 1e-6, (
                f"Weight mismatch for word '{orig_word}'"
            )

    @pytest.mark.parametrize("topK", [3, 5, 10])
    def test_topK_parameter(
        self,
        orig_tfidf_extractor: original_jieba_analyse.TFIDF,
        fast_tfidf_extractor: jieba_fast_dat.analyse.TFIDF,
        main_test_text: str,
        topK: int,
    ) -> None:
        """
        Tests that the `topK` parameter works as expected.
        """
        orig_tags = orig_tfidf_extractor.extract_tags(main_test_text, topK=topK)
        fast_tags = fast_tfidf_extractor.extract_tags(main_test_text, topK=topK)

        assert orig_tags == fast_tags, f"TF-IDF topK={topK} mismatch."
        assert len(fast_tags) <= topK

    def test_with_stop_words(
        self,
        orig_tfidf_extractor: original_jieba_analyse.TFIDF,
        fast_tfidf_extractor: jieba_fast_dat.analyse.TFIDF,
        main_test_text: str,
        stop_words_path: str,
    ) -> None:
        """
        Tests that stop words are correctly filtered.
        """
        # The original jieba's extractor needs stop words to be set via a
        # global function
        orig_tfidf_extractor.set_stop_words(stop_words_path)

        # For jieba_fast_dat, we can set them on the instance
        fast_tfidf_extractor.set_stop_words(stop_words_path)

        orig_tags = orig_tfidf_extractor.extract_tags(main_test_text, topK=5)
        fast_tags = fast_tfidf_extractor.extract_tags(main_test_text, topK=5)

        assert orig_tags == fast_tags, (
            "TF-IDF with stop words mismatch.\n"
            f"Original: {orig_tags}\n"
            f"Fast:     {fast_tags}"
        )

        # Verify that no stop words are in the results
        with open(stop_words_path, encoding="utf-8") as f:
            stop_words = {line.strip() for line in f}

        assert not any(tag in stop_words for tag in fast_tags)
