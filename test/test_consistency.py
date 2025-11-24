"""
test_consistency.py

This module contains comprehensive tests to ensure that jieba_fast_dat's behavior
is consistent with the original jieba library across various configurations.
It replaces several older, scattered test files with a single, parameterized
testing structure.

The tests are organized into classes for different functionalities (e.g.,
segmentation, POS tagging) and heavily use pytest.mark.parametrize to cover
multiple scenarios with minimal code duplication.
"""

import jieba as original_jieba
import pytest
from jieba import Tokenizer as OriginalTokenizer
from jieba.posseg import POSTokenizer as OriginalPOSTokenizer

from jieba_fast_dat import Tokenizer as FastTokenizer
from jieba_fast_dat.posseg import POSTokenizer as FastPOSTokenizer

# Mark the entire module to be skipped if the original jieba is not installed
pytestmark = pytest.mark.skipif(
    not hasattr(original_jieba, "Tokenizer"), reason="original jieba not installed"
)


def _get_cut_result(
    tokenizer: OriginalTokenizer | FastTokenizer,
    text: str,
    mode: str,
    hmm: bool,
) -> list[str]:
    """Helper function to get segmentation results based on mode."""
    if mode == "cut":
        return tokenizer.lcut(text, HMM=hmm)
    if mode == "cut_all":
        # The 'lcut' method in jieba does not support 'cut_all' with HMM=False.
        # It defaults to HMM=True when cut_all is True.
        return tokenizer.lcut(text, cut_all=True)
    if mode == "search":
        return tokenizer.lcut_for_search(text, HMM=hmm)
    raise ValueError(f"Unknown mode: {mode}")


@pytest.mark.consistency
class TestSegmentationConsistency:
    """
    Tests for consistency in word segmentation between jieba_fast_dat
    and original jieba.
    """

    @pytest.mark.parametrize("text_fixture_name", ["main_test_text", "mixed_text"])
    @pytest.mark.parametrize("mode", ["cut", "cut_all", "search"])
    @pytest.mark.parametrize("hmm", [True, False])
    def test_segmentation(
        self,
        orig_tokenizer: OriginalTokenizer,
        fast_tokenizer: FastTokenizer,
        text_fixture_name: str,
        mode: str,
        hmm: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Compares segmentation results for various modes and HMM settings.
        """
        # Skip cut_all with HMM=False, as it's not a valid combination
        if mode == "cut_all" and not hmm:
            pytest.skip("cut_all=True does not use HMM, skipping redundant test.")

        text: str = request.getfixturevalue(text_fixture_name)

        # Get results from both tokenizers
        orig_result = _get_cut_result(orig_tokenizer, text, mode, hmm)
        fast_result = _get_cut_result(fast_tokenizer, text, mode, hmm)

        # Assert that the results are identical
        assert orig_result == fast_result, (
            f"Segmentation mismatch for mode='{mode}', HMM={hmm}, "
            f"text='{text_fixture_name}'.\n"
            f"Original: {orig_result}\n"
            f"Fast:     {fast_result}"
        )


@pytest.mark.consistency
class TestPOSTaggingConsistency:
    """
    Tests for consistency in Part-of-Speech tagging.
    """

    @pytest.mark.parametrize("text_fixture_name", ["main_test_text", "mixed_text"])
    @pytest.mark.parametrize("hmm", [True, False])
    def test_pos_tagging(
        self,
        orig_pos_tokenizer: OriginalPOSTokenizer,
        fast_pos_tokenizer: FastPOSTokenizer,
        text_fixture_name: str,
        hmm: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Compares POS tagging results for various HMM settings.
        """
        text: str = request.getfixturevalue(text_fixture_name)

        # Get results from both POS tokenizers
        orig_result = orig_pos_tokenizer.lcut(text, HMM=hmm)
        fast_result = fast_pos_tokenizer.lcut(text, HMM=hmm)

        # Convert pairs to tuples for consistent comparison
        orig_pairs = [(p.word, p.flag) for p in orig_result]
        fast_pairs = [(p.word, p.flag) for p in fast_result]

        assert orig_pairs == fast_pairs, (
            f"POS tagging mismatch for HMM={hmm}, text='{text_fixture_name}'.\n"
            f"Original: {orig_pairs}\n"
            f"Fast:     {fast_pairs}"
        )
