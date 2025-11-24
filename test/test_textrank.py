"""test_textrank.py

Tests for the TextRank keyword extraction functionality, ensuring consistency
between jieba_fast_dat and the original jieba library.
"""

import jieba as original_jieba
import pytest
from jieba.analyse import TextRank as OriginalTextRank

from jieba_fast_dat.analyse import TextRank as FastTextRank

# Mark the entire module to be skipped if the original jieba is not installed
pytestmark = pytest.mark.skipif(
    not hasattr(original_jieba, "analyse"),
    reason="original jieba.analyse not installed",
)


@pytest.mark.consistency
class TestTextRankConsistency:
    """
    Tests for consistency in TextRank keyword extraction.
    """

    def test_default_extraction(
        self,
        orig_textrank_extractor: OriginalTextRank,
        fast_textrank_extractor: FastTextRank,
        main_test_text: str,
    ) -> None:
        """
        Tests the default behavior of textrank.
        """
        orig_keywords = orig_textrank_extractor.textrank(main_test_text, topK=5)
        fast_keywords = fast_textrank_extractor.textrank(main_test_text, topK=5)

        assert orig_keywords == fast_keywords, (
            f"Default TextRank extraction mismatch.\n"
            f"Original: {orig_keywords}\n"
            f"Fast:     {fast_keywords}"
        )

    def test_with_weight(
        self,
        orig_textrank_extractor: OriginalTextRank,
        fast_textrank_extractor: FastTextRank,
        main_test_text: str,
    ) -> None:
        """
        Tests the `withWeight=True` functionality.
        """
        orig_keywords = orig_textrank_extractor.textrank(
            main_test_text, topK=5, withWeight=True
        )
        fast_keywords = fast_textrank_extractor.textrank(
            main_test_text, topK=5, withWeight=True
        )

        # Compare words and weights with a tolerance
        assert [item[0] for item in orig_keywords] == [
            item[0] for item in fast_keywords
        ], (
            f"TextRank withWeight=True word mismatch.\n"
            f"Original: {[item[0] for item in orig_keywords]}\n"
            f"Fast:     {[item[0] for item in fast_keywords]}"
        )

        for (orig_word, orig_w), (fast_word, fast_w) in zip(
            orig_keywords, fast_keywords, strict=True
        ):
            assert orig_word == fast_word
            assert abs(orig_w - fast_w) < 1e-6, (
                f"Weight mismatch for word '{orig_word}'"
            )

    @pytest.mark.parametrize("topK", [3, 5, 10])
    def test_topK_parameter(
        self,
        orig_textrank_extractor: OriginalTextRank,
        fast_textrank_extractor: FastTextRank,
        main_test_text: str,
        topK: int,
    ) -> None:
        """
        Tests that the `topK` parameter works as expected.
        """
        orig_keywords = orig_textrank_extractor.textrank(main_test_text, topK=topK)
        fast_keywords = fast_textrank_extractor.textrank(main_test_text, topK=topK)

        assert orig_keywords == fast_keywords, f"TextRank topK={topK} mismatch."
        assert len(fast_keywords) <= topK

    @pytest.mark.parametrize("allowPOS", [("n", "nr", "ns"), ("v", "vn"), ()])
    def test_allowPOS_parameter(
        self,
        orig_textrank_extractor: OriginalTextRank,
        fast_textrank_extractor: FastTextRank,
        main_test_text: str,
        allowPOS: tuple[str, ...],
    ) -> None:
        """
        Tests that the `allowPOS` parameter correctly filters results.
        """
        orig_keywords = orig_textrank_extractor.textrank(
            main_test_text, topK=5, allowPOS=allowPOS
        )
        fast_keywords = fast_textrank_extractor.textrank(
            main_test_text, topK=5, allowPOS=allowPOS
        )

        assert orig_keywords == fast_keywords, (
            f"TextRank allowPOS={allowPOS} mismatch.\n"
            f"Original: {orig_keywords}\n"
            f"Fast:     {fast_keywords}"
        )
