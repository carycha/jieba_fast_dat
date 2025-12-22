from __future__ import annotations

from collections.abc import Iterator

import jieba_fast_dat

# Import new C++ function for HMM=False POS tagging
from jieba_fast_dat._jieba_fast_dat_functions_py3 import (
    _posseg_cut_DAG_cpp,
    _posseg_cut_DAG_NO_HMM_cpp,
    _posseg_cut_internal_cpp,
    pair,  # Import C++ implementation of pair
)

from ..utils import (
    RE_ENG_POS,
    RE_HAN_DETAIL,
    RE_NUM_POS,
    RE_SKIP_DETAIL,
    load_model_pickle,
    strdecode,
)
from .viterbi import viterbi

__all__ = ["POSTokenizer", "pair", "cut", "lcut", "dt"]

MIN_FLOAT = -3.14e100


PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"


_initialized = False


# Load models from .p files
def load_model() -> None:
    global _initialized
    if _initialized:
        return

    # Load and deserialize models
    start_p = load_model_pickle(__name__, PROB_START_P)
    trans_p = load_model_pickle(__name__, PROB_TRANS_P)
    emit_p = load_model_pickle(__name__, PROB_EMIT_P)
    char_state_tab = load_model_pickle(__name__, CHAR_STATE_TAB_P)

    # Initialize C++ HMM model
    jieba_fast_dat.load_hmm_model(start_p, trans_p, emit_p, char_state_tab)
    _initialized = True


class POSTokenizer:
    """
    Tokenizer with Part-of-Speech tagging support.
    """

    def __init__(self, tokenizer: jieba_fast_dat.Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or jieba_fast_dat.dt

    def __repr__(self) -> str:
        return f"<POSTokenizer tokenizer={self.tokenizer!r}>"

    def initialize(self, dictionary: str | None = None) -> None:
        """Initialize the underlying tokenizer."""
        self.tokenizer.initialize(dictionary)

    def makesure_userdict_loaded(self) -> None:
        """Handled by Tokenizer.load_userdict."""
        pass

    def __cut(self, sentence: str) -> Iterator[pair]:
        if not _initialized:
            load_model()
        self.tokenizer.check_initialized()
        _prob, word_pos_tags_route = viterbi(sentence)
        yield from word_pos_tags_route

    def __cut_detail(self, sentence: str) -> Iterator[pair]:
        blocks = RE_HAN_DETAIL.split(sentence)
        for blk_idx, blk in enumerate(blocks):
            if not blk:
                continue
            if blk_idx % 2 == 1:  # Matched block
                yield from self.__cut(blk)
            else:
                tmp = RE_SKIP_DETAIL.split(blk)
                for x in tmp:
                    if x:
                        if RE_NUM_POS.match(x):
                            yield pair(x, "m")
                        elif RE_ENG_POS.match(x):
                            yield pair(x, "eng")
                        else:
                            yield pair(x, "x")

    def __cut_DAG_NO_HMM(self, sentence: str) -> Iterator[pair]:
        self.tokenizer.check_initialized()
        result = _posseg_cut_DAG_NO_HMM_cpp(
            self.tokenizer.dat,
            sentence,
            float(self.tokenizer.total),
        )
        yield from result

    def __cut_DAG(self, sentence: str) -> Iterator[pair]:
        self.tokenizer.check_initialized()
        result = _posseg_cut_DAG_cpp(
            self.tokenizer.dat,
            sentence,
            float(self.tokenizer.total),
        )
        yield from result

    def __cut_internal(self, sentence: str, HMM: bool = True) -> Iterator[pair]:
        if not _initialized:
            load_model()
        self.tokenizer.check_initialized()
        sentence = strdecode(sentence)
        result = _posseg_cut_internal_cpp(
            self.tokenizer.dat, sentence, float(self.tokenizer.total), HMM
        )
        yield from result

    def _lcut_internal(self, sentence: str) -> list[pair]:
        if not _initialized:
            load_model()
        self.tokenizer.check_initialized()
        sentence = strdecode(sentence)
        return _posseg_cut_internal_cpp(
            self.tokenizer.dat, sentence, float(self.tokenizer.total), True
        )

    def _lcut_internal_no_hmm(self, sentence: str) -> list[pair]:
        if not _initialized:
            load_model()
        self.tokenizer.check_initialized()
        sentence = strdecode(sentence)
        return _posseg_cut_internal_cpp(
            self.tokenizer.dat, sentence, float(self.tokenizer.total), False
        )

    def cut(self, sentence: str, HMM: bool = True) -> Iterator[pair]:
        """Part-of-speech tagging."""
        return iter(self.lcut(sentence, HMM=HMM))

    def lcut(self, sentence: str, HMM: bool = True) -> list[pair]:
        """List-based part-of-speech tagging."""
        if not _initialized:
            load_model()
        self.tokenizer.check_initialized()
        sentence = strdecode(sentence)
        return _posseg_cut_internal_cpp(
            self.tokenizer.dat, sentence, float(self.tokenizer.total), HMM
        )


# default Tokenizer instance
dt = POSTokenizer(jieba_fast_dat.dt)

# global functions
initialize = dt.initialize


def _lcut_internal(s: str) -> list[pair]:
    return dt._lcut_internal(s)


def _lcut_internal_no_hmm(s: str) -> list[pair]:
    return dt._lcut_internal_no_hmm(s)


def cut(sentence: str, HMM: bool = True) -> Iterator[pair]:
    """Part-of-speech tagging global function."""
    global dt
    if jieba_fast_dat.pool is None:
        yield from dt.cut(sentence, HMM=HMM)
    else:
        # Parallel processing
        parts = strdecode(sentence).splitlines(True)
        if HMM:
            result = list(jieba_fast_dat.pool.map(_lcut_internal, parts))
        else:
            result = list(jieba_fast_dat.pool.map(_lcut_internal_no_hmm, parts))
        for r in result:
            yield from r


def lcut(sentence: str, HMM: bool = True) -> list[pair]:
    """List-based part-of-speech tagging global function."""
    return list(cut(sentence, HMM))
