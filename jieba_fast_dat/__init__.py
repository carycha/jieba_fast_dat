"""
Jieba Fast DAT: A high-performance Chinese segmentation library using C++
and Double Array Trie (DAT).
"""

from __future__ import annotations

import logging
from typing import Any

import jieba_fast_dat._jieba_fast_dat_functions_py3 as _jieba_fast_dat_functions

__version__ = "0.58"

from .core.functions import (
    add_word,
    calc,
    cut,
    cut_for_search,
    del_word,
    disable_parallel,
    dt,
    enable_parallel,
    get_DAG,
    get_dict_file,
    get_FREQ,
    initialize,
    lcut,
    lcut_for_search,
    load_userdict,
    set_dictionary,
    suggest_freq,
    tokenize,
    user_word_tag_tab,
)
from .core.tokenizer import DEFAULT_DICT, DEFAULT_DICT_NAME, Tokenizer
from .utils import strdecode

__all__ = [
    "Tokenizer",
    "DEFAULT_DICT",
    "DEFAULT_DICT_NAME",
    "add_word",
    "calc",
    "cut",
    "lcut",
    "cut_for_search",
    "lcut_for_search",
    "del_word",
    "get_DAG",
    "get_dict_file",
    "initialize",
    "load_userdict",
    "set_dictionary",
    "suggest_freq",
    "tokenize",
    "user_word_tag_tab",
    "get_FREQ",
    "enable_parallel",
    "disable_parallel",
    "dt",
    "strdecode",
    "setLogLevel",
    "load_hmm_model",
]

# Re-export C++ functions for compatibility (especially for posseg)
load_hmm_model = _jieba_fast_dat_functions.load_hmm_model

# Compatibility aliases
text_type = str


def setLogLevel(log_level: int) -> None:
    """Set the logging level for the jieba_fast_dat logger."""
    logging.getLogger("jieba_fast_dat").setLevel(log_level)


# For backward compatibility with internal code using jieba_fast_dat.pool
def __getattr__(name: str) -> Any:
    if name == "pool":
        from .core import functions

        return functions.pool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
