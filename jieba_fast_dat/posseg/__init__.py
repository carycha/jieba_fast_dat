import pickle
import re
from collections.abc import Iterator
from typing import IO

import jieba_fast_dat

# Import new C++ function for HMM=False POS tagging
from jieba_fast_dat._jieba_fast_dat_functions_py3 import (
    _load_word_tag_pybind,  # <--- Added this import
    _posseg_cut_DAG_NO_HMM_cpp,
)

from .._compat import strdecode
from ..utils import get_module_res
from .viterbi import viterbi

MIN_FLOAT = -3.14e100


PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"


# Load models from .p files
def _load_posseg_models() -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    start_p = pickle.load(get_module_res(__name__, PROB_START_P))
    trans_p = pickle.load(get_module_res(__name__, PROB_TRANS_P))
    emit_p = pickle.load(get_module_res(__name__, PROB_EMIT_P))
    char_state_tab_p = pickle.load(get_module_res(__name__, CHAR_STATE_TAB_P))
    return start_p, trans_p, emit_p, char_state_tab_p


_start_P_dict, _trans_P_dict, _emit_P_dict, _char_state_tab_P_dict = (
    _load_posseg_models()
)

jieba_fast_dat.load_hmm_model(
    _start_P_dict, _trans_P_dict, _emit_P_dict, _char_state_tab_P_dict
)


re_han_detail = re.compile(r"([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile(r"([\.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile(r"(\r\n|\s)")

re_eng = re.compile(r"[a-zA-Z0-9]+")
re_num = re.compile(r"[\.0-9]+")

re_eng1 = re.compile("^[a-zA-Z0-9]$", re.U)


class pair:
    __slots__ = ("word", "flag")

    def __init__(self, word: str, flag: str) -> None:
        self.word = word
        self.flag = flag

    def __str__(self) -> str:
        return f"{self.word}/{self.flag}"

    def __repr__(self) -> str:
        return f"pair({self.word!r}, {self.flag!r})"

    def __iter__(self) -> Iterator[str]:
        return iter((self.word, self.flag))

    def __lt__(self, other: "pair") -> bool:
        return self.word < other.word

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, pair)
            and self.word == other.word
            and self.flag == other.flag
        )

    def __hash__(self) -> int:
        return hash(self.word)


class POSTokenizer:
    def __init__(self, tokenizer: jieba_fast_dat.Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or jieba_fast_dat.Tokenizer()
        self.word_tag_tab: dict[str, str] = {}
        self.load_word_tag(self.tokenizer.get_dict_file())

    def __repr__(self) -> str:
        return f"<POSTokenizer tokenizer={self.tokenizer!r}>"

    def initialize(self, dictionary: str | None = None) -> None:
        self.tokenizer.initialize(dictionary)
        self.load_word_tag(self.tokenizer.get_dict_file())

    def load_word_tag(self, f: str | IO[bytes]) -> None:
        self.word_tag_tab = {}
        file_path_to_load: str

        if isinstance(f, str):
            file_path_to_load = f
        else:  # f is an IO[bytes] object
            # If f is opened from get_module_res, it will have a name attribute.
            # We prioritize getting the file path if available.
            if (
                hasattr(f, "name")
                and isinstance(f.name, str)
                and not f.name.startswith("<")
            ):
                file_path_to_load = f.name
            else:
                raise TypeError(
                    "C++ _load_word_tag_pybind requires a file path. "
                    "File-like objects are not directly supported for now, "
                    "unless they have a 'name' attribute representing a file path."
                )

        # Call the C++ function to load the word tags
        _load_word_tag_pybind(file_path_to_load, self.word_tag_tab)

    def makesure_userdict_loaded(self) -> None:
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence: str) -> Iterator[pair]:
        _prob, word_pos_tags_route = viterbi(
            sentence
        )  # prob is not used, replace with _prob
        for word, tag in word_pos_tags_route:
            yield pair(word, tag)

    def __cut_detail(self, sentence: str) -> Iterator[pair]:
        blocks = re_han_detail.split(sentence)
        for blk_idx, blk in enumerate(blocks):
            if not blk:
                continue
            if blk_idx % 2 == 1:  # Matched block
                yield from self.__cut(blk)
            else:
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        if re_num.match(x):
                            yield pair(x, "m")
                        elif re_eng.match(x):
                            yield pair(x, "eng")
                        else:
                            yield pair(x, "x")

    def __cut_DAG_NO_HMM(self, sentence: str) -> Iterator[pair]:
        result = _posseg_cut_DAG_NO_HMM_cpp(
            self.tokenizer.dat,
            sentence,
            self.word_tag_tab,
            float(self.tokenizer.total),
        )
        for word, flag in result:
            yield pair(word, flag)

    def __cut_DAG(self, sentence: str) -> Iterator[pair]:
        # Python implementation
        route: list[int] = []
        self.tokenizer.check_initialized()
        jieba_fast_dat._jieba_fast_dat_functions._get_DAG_and_calc(
            self.tokenizer.dat,
            sentence,
            route,
            float(self.tokenizer.total),
        )

        x = 0
        buf = ""
        N = len(sentence)

        # Localize lookups
        word_tag_tab_get = self.word_tag_tab.get
        tokenizer_get_freq = self.tokenizer.get_freq
        re_num_fullmatch = re_num.fullmatch
        re_eng_fullmatch = re_eng.fullmatch
        cut_detail = self.__cut_detail

        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield pair(buf, word_tag_tab_get(buf, "x"))
                    elif not tokenizer_get_freq(buf):
                        if re_num_fullmatch(buf):
                            yield pair(buf, "m")
                        elif re_eng_fullmatch(buf):
                            yield pair(buf, "eng")
                        else:
                            yield from cut_detail(buf)
                    else:
                        for elem in list(buf):
                            yield pair(elem, word_tag_tab_get(elem, "x"))
                    buf = ""
                yield pair(l_word, word_tag_tab_get(l_word, "x"))
            x = y

        if buf:
            if len(buf) == 1:
                yield pair(buf, word_tag_tab_get(buf, "x"))
            elif not tokenizer_get_freq(buf):
                if re_num_fullmatch(buf):
                    yield pair(buf, "m")
                elif re_eng_fullmatch(buf):
                    yield pair(buf, "eng")
                else:
                    yield from cut_detail(buf)
            else:
                if buf:
                    for elem in list(buf):
                        yield pair(elem, word_tag_tab_get(elem, "x"))

    def __cut_internal(self, sentence: str, HMM: bool = True) -> Iterator[pair]:
        self.makesure_userdict_loaded()
        sentence = strdecode(sentence)
        blocks = re_han_internal.split(sentence)
        if HMM:
            cut_blk = self.__cut_DAG
        else:
            cut_blk = self.__cut_DAG_NO_HMM

        for blk_idx, blk in enumerate(blocks):
            if not blk:
                continue
            if blk_idx % 2 == 1:  # Matched block
                yield from cut_blk(blk)
            else:
                tmp = re_skip_internal.split(blk)
                for x in tmp:
                    if not x:
                        continue
                    if re_skip_internal.match(x):
                        yield pair(x, "x")
                    else:
                        for xx in x:
                            if xx.isdigit():
                                yield pair(xx, "m")
                            elif xx.isalpha():
                                yield pair(xx, "eng")
                            else:
                                yield pair(xx, "x")

    def _lcut_internal(self, sentence: str) -> list[pair]:
        return list(self.__cut_internal(sentence))

    def _lcut_internal_no_hmm(self, sentence: str) -> list[pair]:
        return list(self.__cut_internal(sentence, False))

    def cut(self, sentence: str, HMM: bool = True) -> Iterator[pair]:
        yield from self.__cut_internal(sentence, HMM=HMM)

    def lcut(self, sentence: str, HMM: bool = True) -> list[pair]:
        return list(self.cut(sentence, HMM))


# default Tokenizer instance
dt = POSTokenizer(jieba_fast_dat.dt)

# global functions
initialize = dt.initialize


def _lcut_internal(s: str) -> list[pair]:
    return dt._lcut_internal(s)


def _lcut_internal_no_hmm(s: str) -> list[pair]:
    return dt._lcut_internal_no_hmm(s)


def cut(sentence: str, HMM: bool = True) -> Iterator[pair]:
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
    return list(cut(sentence, HMM))
