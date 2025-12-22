from collections.abc import Iterator

import jieba_fast_dat._jieba_fast_dat_functions_py3 as _jieba_fast_dat_functions
from jieba_fast_dat.utils import (
    RE_HAN_FINALSEG,
    RE_SKIP_FINALSEG,
    load_model_pickle,
)

MIN_FLOAT = -3.14e100

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"

PrevStatus = {"B": "ES", "M": "MB", "S": "SE", "E": "BM"}
Force_Split_Words: set[str] = set()

_initialized = False


def load_model() -> None:
    global _initialized
    if _initialized:
        return

    start_P = load_model_pickle(__name__, PROB_START_P)
    trans_P = load_model_pickle(__name__, PROB_TRANS_P)
    emit_P = load_model_pickle(__name__, PROB_EMIT_P)

    # Push models to C++
    _jieba_fast_dat_functions.load_finalseg_hmm_model(start_P, trans_P, emit_P)

    _initialized = True


def viterbi(
    obs: str,
    states: str,
    start_p: dict[str, float],
    trans_p: dict[str, dict[str, float]],
    emit_p: dict[str, dict[str, float]],
) -> tuple[float, list[str]]:
    # Fallback to C++ implementation for better performance
    prob, pos_list = _jieba_fast_dat_functions._finalseg_viterbi_cpp(obs)
    return prob, list(pos_list)


def __cut(sentence: str) -> Iterator[str]:
    global start_P, trans_P, emit_P
    if not _initialized:
        load_model()
    prob, pos_list = _jieba_fast_dat_functions._finalseg_viterbi_cpp(sentence)
    words = []
    begin, nexti = 0, 0
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == "B":
            begin = i
        elif pos == "E":
            words.append(sentence[begin : i + 1])
            nexti = i + 1
        elif pos == "S":
            words.append(char)
            nexti = i + 1
    if nexti < len(sentence):
        words.append(sentence[nexti:])
    yield from words


def add_force_split(word: str) -> None:
    global Force_Split_Words
    Force_Split_Words.add(word)


def cut(sentence: str) -> Iterator[str]:
    blocks = RE_HAN_FINALSEG.split(sentence)
    for blk_idx, blk in enumerate(blocks):
        if not blk:
            continue
        if blk_idx % 2 == 1:  # Matched block
            for word in __cut(blk):
                if word not in Force_Split_Words:
                    yield word
                else:
                    yield from word
        else:
            tmp = RE_SKIP_FINALSEG.split(blk)
            for x in tmp:
                if x:
                    yield x
