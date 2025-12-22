from collections.abc import Iterator

from .tokenizer import Tokenizer

# default Tokenizer instance
dt = Tokenizer()

# Exported global functions delegating to dt
add_word = dt.add_word
calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def get_FREQ(k: str, d: int | float | None = None) -> int | float | None:
    return dt.get_freq(k) or d


# Parallel processing support
pool = None


def _lcut(s: str) -> list[str]:
    return dt.lcut(s)


def _lcut_no_hmm(s: str) -> list[str]:
    return dt.lcut(s, HMM=False)


def _lcut_all(s: str) -> list[str]:
    return dt.lcut(s, cut_all=True)


def _lcut_for_search(s: str) -> list[str]:
    return dt.lcut_for_search(s)


def _lcut_for_search_no_hmm(s: str) -> list[str]:
    return dt.lcut_for_search(s, HMM=False)


def _pcut(sentence: str, cut_all: bool = False, HMM: bool = True) -> Iterator[str]:
    assert pool is not None
    parts = sentence.splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        yield from r


def _pcut_for_search(sentence: str, HMM: bool = True) -> Iterator[str]:
    assert pool is not None
    parts = sentence.splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        yield from r


def enable_parallel(processnum: int | None = None) -> None:
    global pool
    from multiprocessing import Pool, cpu_count

    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    # We need to update the module level functions in jieba_fast_dat
    import jieba_fast_dat

    jieba_fast_dat.cut = _pcut
    jieba_fast_dat.cut_for_search = _pcut_for_search


def disable_parallel() -> None:
    global pool
    if pool:
        pool.close()
        pool = None
    import jieba_fast_dat

    jieba_fast_dat.cut = dt.cut
    jieba_fast_dat.cut_for_search = dt.cut_for_search
