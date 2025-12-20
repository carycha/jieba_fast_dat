import pickle
import re
from collections.abc import Iterator
from typing import IO

import jieba_fast_dat

# Import new C++ function for HMM=False POS tagging
from jieba_fast_dat._jieba_fast_dat_functions_py3 import (
    _load_word_tag_pybind,
    _posseg_cut_DAG_cpp,
    _posseg_cut_DAG_NO_HMM_cpp,
    _posseg_cut_internal_cpp,
    pair,  # Import C++ implementation of pair
)

from .._compat import strdecode
from ..utils import get_module_res
from .viterbi import viterbi

MIN_FLOAT = -3.14e100


PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"


_initialized = False


# Load models from .p files
def load_model() -> None:
    global \
        _initialized, \
        _start_P_dict, \
        _trans_P_dict, \
        _emit_P_dict, \
        _char_state_tab_P_dict
    if _initialized:
        return
    _start_P_dict = pickle.loads(get_module_res(__name__, PROB_START_P).read())
    _trans_P_dict = pickle.loads(get_module_res(__name__, PROB_TRANS_P).read())
    _emit_P_dict = pickle.loads(get_module_res(__name__, PROB_EMIT_P).read())
    _char_state_tab_P_dict = pickle.loads(
        get_module_res(__name__, CHAR_STATE_TAB_P).read()
    )

    jieba_fast_dat.load_hmm_model(
        _start_P_dict, _trans_P_dict, _emit_P_dict, _char_state_tab_P_dict
    )
    _initialized = True


re_han_detail = re.compile(r"([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile(r"([\.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile(r"(\r\n|\s)")

re_eng = re.compile(r"[a-zA-Z0-9]+")
re_num = re.compile(r"[\.0-9]+")

re_eng1 = re.compile("^[a-zA-Z0-9]$", re.U)


class POSTokenizer:
    def __init__(self, tokenizer: jieba_fast_dat.Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or jieba_fast_dat.Tokenizer()
        self.word_tag_tab: dict[str, str] = {}
        self.word_tag_tab_loaded = False

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
        # Sync to C++
        self.tokenizer.dat.update_word_tag_tab(self.word_tag_tab)
        self.word_tag_tab_loaded = True

    def makesure_userdict_loaded(self) -> None:
        if self.tokenizer.user_word_tag_tab:
            if not self.word_tag_tab_loaded:
                self.load_word_tag(self.tokenizer.get_dict_file())
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            # Sync to C++
            self.tokenizer.dat.update_word_tag_tab(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence: str) -> Iterator[pair]:
        if not _initialized:
            load_model()
        if not self.word_tag_tab_loaded:
            self.load_word_tag(self.tokenizer.get_dict_file())
        _prob, word_pos_tags_route = viterbi(
            sentence
        )  # prob is not used, replace with _prob
        yield from word_pos_tags_route

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
        if not self.word_tag_tab_loaded:
            self.load_word_tag(self.tokenizer.get_dict_file())
        result = _posseg_cut_DAG_NO_HMM_cpp(
            self.tokenizer.dat,
            sentence,
            float(self.tokenizer.total),
        )
        yield from result

    def __cut_DAG(self, sentence: str) -> Iterator[pair]:
        if not self.word_tag_tab_loaded:
            self.load_word_tag(self.tokenizer.get_dict_file())
        result = _posseg_cut_DAG_cpp(
            self.tokenizer.dat,
            sentence,
            float(self.tokenizer.total),
        )
        yield from result

    def __cut_internal(self, sentence: str, HMM: bool = True) -> Iterator[pair]:
        if not _initialized:
            load_model()
        if not self.word_tag_tab_loaded:
            self.load_word_tag(self.tokenizer.get_dict_file())
        if self.tokenizer.user_word_tag_tab:
            self.makesure_userdict_loaded()
        sentence = strdecode(sentence)
        result = _posseg_cut_internal_cpp(
            self.tokenizer.dat, sentence, float(self.tokenizer.total), HMM
        )
        yield from result

    def _lcut_internal(self, sentence: str) -> list[pair]:
        if not _initialized:
            load_model()
        if not self.word_tag_tab_loaded:
            self.load_word_tag(self.tokenizer.get_dict_file())
        if self.tokenizer.user_word_tag_tab:
            self.makesure_userdict_loaded()
        sentence = strdecode(sentence)
        return _posseg_cut_internal_cpp(
            self.tokenizer.dat, sentence, float(self.tokenizer.total), True
        )

    def _lcut_internal_no_hmm(self, sentence: str) -> list[pair]:
        if not _initialized:
            load_model()
        if not self.word_tag_tab_loaded:
            self.load_word_tag(self.tokenizer.get_dict_file())
        if self.tokenizer.user_word_tag_tab:
            self.makesure_userdict_loaded()
        sentence = strdecode(sentence)
        return _posseg_cut_internal_cpp(
            self.tokenizer.dat, sentence, float(self.tokenizer.total), False
        )

    def cut(self, sentence: str, HMM: bool = True) -> Iterator[pair]:
        return iter(self.lcut(sentence, HMM=HMM))

    def lcut(self, sentence: str, HMM: bool = True) -> list[pair]:
        if not _initialized:
            load_model()
        if not self.word_tag_tab_loaded:
            self.load_word_tag(self.tokenizer.get_dict_file())
        if self.tokenizer.user_word_tag_tab:
            self.makesure_userdict_loaded()
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
