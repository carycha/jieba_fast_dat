"""
test_hmm.py

This module contains tests specifically for the Hidden Markov Model (HMM)
components, ensuring that the behavior of `finalseg` and HMM-enabled `posseg`
is consistent with the original jieba library.

This addresses the need for direct, low-level testing of the HMM implementation,
rather than only testing it indirectly through the main tokenizer's `cut` method.
"""

import jieba as original_jieba
import jieba.finalseg as original_finalseg
import jieba.posseg as original_jieba_posseg
import pytest

import jieba_fast_dat.finalseg as fast_finalseg
import jieba_fast_dat.posseg

# Mark the entire module to be skipped if the original jieba is not installed
pytestmark = pytest.mark.skipif(
    not hasattr(original_jieba, "Tokenizer"), reason="original jieba not installed"
)


@pytest.mark.consistency
class TestHMMConsistency:
    """
    Tests for consistency in HMM-based segmentation and POS tagging.
    """

    @pytest.fixture
    def hmm_sentence(self) -> str:
        """A sentence containing out-of-vocabulary words to trigger HMM."""
        return "他来到了网易杭研大厦"

    def test_finalseg_consistency(self, hmm_sentence: str) -> None:
        """
        Directly compares the output of `finalseg.cut` between both libraries.
        `finalseg` is the core HMM implementation for new word discovery.
        """
        orig_result = list(original_finalseg.cut(hmm_sentence))
        fast_result = list(fast_finalseg.cut(hmm_sentence))

        assert orig_result == fast_result, (
            f"finalseg.cut mismatch for sentence: '{hmm_sentence}'.\n"
            f"Original: {orig_result}\n"
            f"Fast:     {fast_result}"
        )

    def test_posseg_hmm_consistency(
        self,
        orig_pos_tokenizer: original_jieba_posseg.POSTokenizer,
        fast_pos_tokenizer: jieba_fast_dat.posseg.POSTokenizer,
        hmm_sentence: str,
    ) -> None:
        """
        Compares the output of `posseg.cut` with HMM enabled, focusing on a
        sentence that requires HMM for proper segmentation and tagging.
        """
        # We use the pos_tokenizer fixtures which are pre-loaded with dictionaries.
        # The HMM model is used for words not in the dictionary.
        orig_result = orig_pos_tokenizer.lcut(hmm_sentence, HMM=True)
        fast_result = fast_pos_tokenizer.lcut(hmm_sentence, HMM=True)

        orig_pairs = [(p.word, p.flag) for p in orig_result]
        fast_pairs = [(p.word, p.flag) for p in fast_result]

        assert orig_pairs == fast_pairs, (
            f"posseg.cut with HMM=True mismatch for sentence: '{hmm_sentence}'.\n"
            f"Original: {orig_pairs}\n"
            f"Fast:     {fast_pairs}"
        )
