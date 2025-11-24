import pickle
import re
from collections.abc import Iterator

import jieba_fast_dat._jieba_fast_dat_functions_py3 as _jieba_fast_dat_functions
from jieba_fast_dat.utils import get_module_res

MIN_FLOAT = -3.14e100

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"

PrevStatus = {"B": "ES", "M": "MB", "S": "SE", "E": "BM"}
Force_Split_Words: set[str] = set()


def load_model() -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    start_p = pickle.load(get_module_res(__name__, PROB_START_P))
    trans_p = pickle.load(get_module_res(__name__, PROB_TRANS_P))
    emit_p = pickle.load(get_module_res(__name__, PROB_EMIT_P))
    return start_p, trans_p, emit_p


start_P, trans_P, emit_P = load_model()


def viterbi(
    obs: str,
    states: str,
    start_p: dict[str, float],
    trans_p: dict[str, dict[str, float]],
    emit_p: dict[str, dict[str, float]],
) -> tuple[float, list[str]]:
    V = [{}]
    path: dict[str, list[str]] = {}
    for y in states:
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath: dict[str, list[str]] = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max(
                (V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0)
                for y0 in PrevStatus[y]
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    (prob, state) = max((V[len(obs) - 1][y], y) for y in "ES")
    return (prob, path[state])


def __cut(sentence: str) -> Iterator[str]:
    global emit_P
    prob, pos_list = _jieba_fast_dat_functions._viterbi(
        sentence, "BMES", start_P, trans_P, emit_P
    )
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


re_han = re.compile("([\u4e00-\u9fd5]+)")
re_skip = re.compile("([a-zA-Z0-9]+(?:\\.\\d+)?%?)")


def add_force_split(word: str) -> None:
    global Force_Split_Words
    Force_Split_Words.add(word)


def cut(sentence: str) -> Iterator[str]:
    blocks = re_han.split(sentence)
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
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x
