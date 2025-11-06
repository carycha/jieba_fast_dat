import pickle

from jieba_fast_dat.utils import (
    get_module_res,
    split_by_char_type,
    CHAR_TYPE_ZH,
    CHAR_TYPE_NUM,
    CHAR_TYPE_ALPHA,
)
import _jieba_fast_dat_functions_py3 as _jieba_fast_dat_functions

MIN_FLOAT = -3.14e100


PROB_START_P = "prob_start.p"

PROB_TRANS_P = "prob_trans.p"

PROB_EMIT_P = "prob_emit.p"


PrevStatus = {"B": "ES", "M": "MB", "S": "SE", "E": "BM"}


Force_Split_Words = set([])


def load_model():
    start_p = pickle.load(get_module_res(__name__, PROB_START_P))

    trans_p = pickle.load(get_module_res(__name__, PROB_TRANS_P))

    emit_p = pickle.load(get_module_res(__name__, PROB_EMIT_P))

    return start_p, trans_p, emit_p


start_P, trans_P, emit_P = load_model()

"""

start_P = _cxx_replace_start(start_P)

trans_P = _cxx_replace_other(trans_P)

emit_P = _cxx_replace_other(emit_P)

"""


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # tabular

    path = {}

    for y in states:  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)

        path[y] = [y]

    for t in range(1, len(obs)):
        V.append({})

        newpath = {}

        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)

            (prob, state) = max(
                [
                    (V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0)
                    for y0 in PrevStatus[y]
                ]
            )

            V[t][y] = prob

            newpath[y] = path[state] + [y]

        path = newpath

    (prob, state) = max((V[len(obs) - 1][y], y) for y in "ES")

    return (prob, path[state])


def __cut(sentence):
    global emit_P

    prob, pos_list = _jieba_fast_dat_functions._viterbi(
        sentence, "BMES", start_P, trans_P, emit_P
    )

    begin, nexti = 0, 0

    for i, char in enumerate(sentence):
        pos = pos_list[i]

        if pos == "B":
            begin = i

        elif pos == "E":
            yield sentence[begin : i + 1]

            nexti = i + 1

        elif pos == "S":
            yield char

            nexti = i + 1

    if nexti < len(sentence):
        yield sentence[nexti:]


def add_force_split(word):
    global Force_Split_Words

    Force_Split_Words.add(word)


def cut(sentence):
    sentence = sentence

    blocks = split_by_char_type(sentence)

    for blk_str, blk_type in blocks:
        if not blk_str:
            continue

        if blk_type == CHAR_TYPE_ZH:
            for word in __cut(blk_str):
                if word not in Force_Split_Words:
                    yield word
                else:
                    for c in word:
                        yield c
        elif blk_type == CHAR_TYPE_NUM or blk_type == CHAR_TYPE_ALPHA:
            yield blk_str
        else:
            # Other characters (punctuation, symbols) are yielded one by one
            for c in blk_str:
                yield c
