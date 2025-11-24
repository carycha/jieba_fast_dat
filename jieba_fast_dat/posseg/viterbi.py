import jieba_fast_dat

# This module acts as a bridge to the C++ implementation of the Viterbi algorithm.
# The function signature is kept compatible with the original jieba's viterbi function.


def viterbi(
    sentence: str,
) -> tuple[float, list[tuple[str, str]]]:
    """
    A wrapper for the C++ Viterbi implementation provided by jieba_fast_dat.

    The parameters (char_state_tab_P, start_P, etc.) are kept for API compatibility
    with the pure Python version in the original jieba, but they are not directly
    used here because the C++ extension has its own pre-loaded models.
    """
    # The C++ function `_posseg_viterbi_cpp` is assumed to be pre-loaded with
    # the necessary probability models, so we don't need to pass them again.
    prob, pos_list = jieba_fast_dat._posseg_viterbi_cpp(sentence)
    return prob, pos_list
