from collections.abc import Iterator
from typing import IO, BinaryIO

__version__ = "0.53"
import io
import logging
import marshal
import os
import re
import sys
import tempfile
import threading
import time
from hashlib import md5
from pathlib import Path

import jieba_fast_dat._jieba_fast_dat_functions_py3 as _jieba_fast_dat_functions
from jieba_fast_dat._jieba_fast_dat_functions_py3 import DatTrie

from . import finalseg
from .utils import get_module_res

load_hmm_model = _jieba_fast_dat_functions.load_hmm_model
_posseg_viterbi_cpp = _jieba_fast_dat_functions._posseg_viterbi_cpp
_get_DAG = _jieba_fast_dat_functions._get_DAG
_get_freq = _jieba_fast_dat_functions._get_freq
load_userdict_pybind = _jieba_fast_dat_functions.load_userdict_pybind

_replace_file = os.rename


def _get_abs_path(path: str) -> str:
    return (
        os.path.normpath(path)
        if os.path.isabs(path)
        else os.path.normpath(os.path.join(os.getcwd(), path))
    )


DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

DICT_WRITING: dict[str | None, threading.RLock] = {}

pool = None


re_eng = re.compile(r"[a-zA-Z0-9]", re.U)

# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# Adding "-" symbol in re_han_default
re_han_default = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)", re.U)
re_skip_default = re.compile(r"(\r\n|\s)", re.U)
re_skip_default = re.compile(r"(\r\n|\s)", re.U)
re_eng_num = re.compile(r"^[a-zA-Z0-9]+(?:\.\d+)?%?$", re.U)


text_type = str


def strdecode(sentence: str) -> str:
    if not isinstance(sentence, text_type):
        try:
            sentence = sentence.decode("utf-8")
        except UnicodeDecodeError:
            sentence = sentence.decode("gbk", "ignore")
    return sentence


def setLogLevel(log_level: int) -> None:
    global default_logger
    default_logger.setLevel(log_level)


class Tokenizer:
    def __init__(self, dictionary: str | None = DEFAULT_DICT) -> None:
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            assert dictionary is not None
            self.dictionary = _get_abs_path(dictionary)
        self.dat = DatTrie()
        self.total = 0
        self.user_freq: dict[str, int] = {}
        self.user_word_tag_tab: dict[str, str] = {}
        self.initialized = False
        self.tmp_dir: str | None = None
        self.cache_file: str | None = None

    def __repr__(self) -> str:
        return f"<Tokenizer dictionary={self.dictionary!r}>"

    def get_freq(self, word: str) -> int:
        return _get_freq(self.dat, self.user_freq, word)

    def gen_dat_data(self, f: IO[bytes]) -> tuple[list[tuple[str, int]], float]:
        lfreq: dict[str, int] = {}
        ltotal = 0.0
        f_name = f.name
        for line in f:
            try:
                line = line.strip().decode("utf-8")
                # Default values
                word = ""
                freq = 1000
                tag = "x"

                parts = line.split(" ", 2)  # Split into max 3 parts

                word = parts[0]

                if len(parts) > 1:
                    # Check if second part is a number
                    if parts[1].isdigit():
                        freq = int(parts[1])
                    else:
                        tag = parts[1]

                if len(parts) == 3:
                    tag = parts[2]

                lfreq[word] = freq
                ltotal += float(freq)
                if tag:  # Store tag if present
                    self.user_word_tag_tab[word] = tag

                for ch in range(len(word)):
                    wfrag = word[: ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError as e:
                raise ValueError(
                    f"invalid dictionary entry in {f_name} at line: {line}"
                ) from e
        f.close()
        word_freqs = list(lfreq.items())
        return word_freqs, ltotal

    def initialize(
        self, dictionary: str | None = None, force_rebuild: bool = False
    ) -> None:
        with self.lock:
            current_dictionary_path = self.dictionary

            if dictionary:
                abs_path = _get_abs_path(dictionary)
                if current_dictionary_path == abs_path and self.initialized:
                    return
                else:
                    self.dictionary = abs_path
                    self.initialized = False
                    # by gen_dat_data
                    self.user_freq = {}
                    # self.user_word_tag_tab = {} # Do not clear here
            else:
                abs_path = current_dictionary_path

            if self.initialized:
                return

            default_logger.debug(
                f"Building prefix dict from {abs_path or 'the default dictionary'} ..."
            )
            t1 = time.time()
            if self.cache_file:
                cache_name = self.cache_file
            elif abs_path == DEFAULT_DICT:
                cache_name = "jieba_fast_dat.cache"
            else:
                _abs_path_for_md5 = abs_path if abs_path is not None else ""
                hexdigest = md5(
                    _abs_path_for_md5.encode("utf-8", "replace")
                ).hexdigest()
                cache_name = f"jieba_fast_dat.u{hexdigest}.cache"

            cache_file = os.path.join(self.tmp_dir or tempfile.gettempdir(), cache_name)
            dat_cache_file = cache_file + ".dat"
            tmpdir = os.path.dirname(cache_file)

            load_from_cache_fail = True
            if (
                not force_rebuild
                and os.path.isfile(cache_file)
                and os.path.isfile(dat_cache_file)
                and (
                    abs_path == DEFAULT_DICT
                    or (
                        (
                            abs_path is not None
                            and os.path.getmtime(cache_file)
                            > os.path.getmtime(abs_path)
                        )
                        and (
                            abs_path is not None
                            and os.path.getmtime(dat_cache_file)
                            > os.path.getmtime(abs_path)
                        )
                    )
                )
            ):
                default_logger.debug(f"Loading model from cache {cache_file}")
                try:
                    with open(cache_file, "rb") as cf:
                        self.total = float(marshal.load(cf))
                    self.dat.open(dat_cache_file)
                    load_from_cache_fail = False
                    # Clear user_freq when loading from cache,
                    # user_word_tag_tab will be repopulated
                    # by gen_dat_data
                    self.user_freq = {}
                    # self.user_word_tag_tab = {} # Do not clear here
                    # Repopulate user_word_tag_tab from main dictionary
                    # after loading from cache
                    # This ensures that main dictionary word tags are available
                    # even when loading from cache
                    self.gen_dat_data(self.get_dict_file())
                except Exception:
                    load_from_cache_fail = True

            if load_from_cache_fail:
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                with wlock:
                    # Call gen_dat_data to populate self.user_freq and
                    # self.user_word_tag_tab from the main dictionary
                    word_freqs_list, ltotal = self.gen_dat_data(self.get_dict_file())

                    # Update total with combined frequencies
                    self.total = ltotal + sum(self.user_freq.values())

                    # Convert back to list for DatTrie.build
                    final_word_freqs = (
                        word_freqs_list  # DatTrie only built with main dict
                    )

                    self.dat.build(final_word_freqs)
                    default_logger.debug(f"Dumping model to file cache {cache_file}")
                    try:
                        # save total
                        fd, fpath = tempfile.mkstemp(dir=tmpdir)
                        with os.fdopen(fd, "wb") as temp_cache_file:
                            marshal.dump(self.total, temp_cache_file)
                        _replace_file(fpath, cache_file)
                        # save dat
                        self.dat.save(dat_cache_file)
                    except Exception:
                        default_logger.exception("Dump cache file failed.")

                try:
                    del DICT_WRITING[abs_path]
                except KeyError:
                    pass

            self.initialized = True
            default_logger.debug(f"Loading model cost {time.time() - t1:.3f} seconds.")
            default_logger.debug("Prefix dict has been built succesfully.")

    def check_initialized(self) -> None:
        if not self.initialized:
            self.initialize()

    def calc(
        self,
        sentence: str,
        DAG: dict[int, list[int]],
        route: dict[int, tuple[float, int]],
    ) -> None:
        self.check_initialized()
        _jieba_fast_dat_functions._calc(
            self.dat,
            sentence,
            DAG,
            route,
            float(self.total),
            self.user_freq,
        )

    def get_DAG(self, sentence: str) -> dict[int, list[int]]:
        self.check_initialized()
        return _get_DAG(self.dat, sentence, self.user_freq)

    def __cut_all(self, sentence: str) -> Iterator[str]:
        dag = self.get_DAG(sentence)
        old_j = -1
        eng_scan = 0
        eng_buf = ""
        for k, L in dag.items():
            if eng_scan == 1 and not re_eng.match(sentence[k]):
                eng_scan = 0
                yield eng_buf
            if len(L) == 1 and k > old_j:
                word = sentence[k : L[0] + 1]
                if re_eng.match(word):
                    if eng_scan == 0:
                        eng_scan = 1
                        eng_buf = word
                    else:
                        eng_buf += word
                if eng_scan == 0:
                    yield word
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k : j + 1]
                        old_j = j
        if eng_scan == 1:
            yield eng_buf

    def __cut_DAG_NO_HMM(self, sentence: str) -> Iterator[str]:
        DAG = self.get_DAG(sentence)
        route: dict[int, tuple[float, int]] = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ""
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if len(l_word) == 1 and (
                "a" <= l_word <= "z" or "A" <= l_word <= "Z" or "0" <= l_word <= "9"
            ):  # If it's a single English/number char
                buf += l_word
                x = y
            else:  # If it's a multi-character word or non-English/number single char
                if buf:  # Process accumulated single English/number chars
                    yield buf
                    buf = ""
                yield l_word  # Yield the word
                x = y
        if (
            buf
        ):  # Process any remaining accumulated single English/number chars at the end
            yield buf

    def __cut_DAG(self, sentence: str) -> Iterator[str]:
        route: list[int] = []
        self.check_initialized()
        _jieba_fast_dat_functions._get_DAG_and_calc(
            self.dat, self.user_freq, sentence, route, float(self.total)
        )
        x = 0
        buf = ""
        N = len(sentence)

        # Localize lookups
        re_eng_num_match = re_eng_num.match
        finalseg_cut = finalseg.cut
        get_freq = self.get_freq

        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if y - x == 1:  # If it's a single character
                buf += l_word
            else:  # If it's a multi-character word
                if buf:  # Process accumulated single characters
                    if len(buf) == 1:
                        yield buf
                    else:
                        if not get_freq(buf):  # If buf is not a recognized word
                            if re_eng_num_match(buf):
                                yield buf
                            else:
                                yield from finalseg_cut(buf)
                        else:  # If buf is a recognized word
                            yield buf  # Yield the recognized word as a whole
                    buf = ""
                yield l_word  # Yield the multi-character word
            x = y

        if buf:  # Process any remaining accumulated single characters at the end
            if len(buf) == 1:
                yield buf
            elif not self.get_freq(buf):
                if re_eng_num.match(buf):
                    yield buf
                else:
                    yield from finalseg.cut(buf)
            else:
                yield buf

    def cut(
        self,
        sentence: str,
        cut_all: bool = False,
        HMM: bool = True,
        use_paddle: bool = False,
    ) -> Iterator[str]:
        """
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.

        Parameter:
            - sentence: The str(unicode) to be segmented.
            - cut_all: Model type. True for full pattern, False for accurate pattern.
            - HMM: Whether to use the Hidden Markov Model.
        """
        sentence = strdecode(sentence)

        re_han = re_han_default
        re_skip = re_skip_default

        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM

        blocks = re_han.split(sentence)
        for blk_idx, blk in enumerate(blocks):
            if not blk:
                continue
            if blk_idx % 2 == 1:  # Matched block
                yield from cut_block(blk)
            else:
                tmp = re_skip.split(blk)
                for x_idx, x in enumerate(tmp):
                    if x_idx % 2 == 1:
                        yield x
                    elif not cut_all:
                        yield from x
                    else:
                        yield x

    def cut_for_search(self, sentence: str, HMM: bool = True) -> Iterator[str]:
        """
        Finer segmentation for search engines.
        """
        words = self.cut(sentence, HMM=HMM)
        for w in words:
            if len(w) > 2:
                for i in range(len(w) - 1):
                    gram2 = w[i : i + 2]
                    if self.get_freq(gram2):
                        yield gram2
            if len(w) > 3:
                for i in range(len(w) - 2):
                    gram3 = w[i : i + 3]
                    if self.get_freq(gram3):
                        yield gram3
            yield w

    def lcut(
        self,
        sentence: str,
        cut_all: bool = False,
        HMM: bool = True,
        use_paddle: bool = False,
    ) -> list[str]:
        return list(self.cut(sentence, cut_all=cut_all, HMM=HMM, use_paddle=use_paddle))

    def lcut_for_search(self, sentence: str, HMM: bool = True) -> list[str]:
        return list(self.cut_for_search(sentence, HMM=HMM))

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence: str) -> list[str]:
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence: str) -> list[str]:
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence: str) -> list[str]:
        return self.lcut_for_search(sentence, False)

    def get_dict_file(self) -> IO[bytes]:
        if self.dictionary == DEFAULT_DICT:
            return get_module_res(__name__, DEFAULT_DICT_NAME)
        else:
            # Ensure self.dictionary is a string path for open()
            return open(str(self.dictionary), "rb")

    def load_userdict(self, f: str | Path | BinaryIO) -> None:
        """
        Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their ocurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        """
        self.check_initialized()

        if isinstance(f, (str, Path)):
            # Use C++ optimized loader for file paths
            try:
                load_userdict_pybind(
                    self.dat,
                    self.user_freq,
                    self.user_word_tag_tab,
                    str(f),
                    finalseg.add_force_split,
                )
                return
            except Exception as e:
                default_logger.warning(
                    f"C++ load_userdict failed, falling back to Python: {e}"
                )
                # Fallback to Python implementation below

        f_to_process: BinaryIO
        f_text_stream: IO[str]
        should_close_binary = False

        if isinstance(f, (str, Path)):
            f_to_process = open(f, "rb")
            should_close_binary = True
            f_text_stream = io.TextIOWrapper(
                f_to_process, encoding="utf-8-sig", errors="ignore"
            )
        else:  # f is already BinaryIO
            f_to_process = f
            # Wrap existing BinaryIO in TextIOWrapper
            # We assume the passed BinaryIO is still readable and seekable if needed.
            # 'errors=ignore' mimics original UnicodeDecodeError handling behavior.
            # from the try-except block, although a ValueError would have been raised.
            f_text_stream = io.TextIOWrapper(
                f_to_process, encoding="utf-8-sig", errors="ignore"
            )

        try:
            for _lineno, line in enumerate(f_text_stream):
                line = line.strip()  # Removed .lstrip("\ufeff")
                if not line:
                    continue

                # Default values
                word = ""
                freq = 1000
                tag = "x"

                parts = line.split(" ", 2)  # Split into max 3 parts

                word = parts[0]

                if len(parts) > 1:
                    # Check if second part is a number
                    if parts[1].isdigit():
                        freq = int(parts[1])
                    else:
                        tag = parts[1]

                if len(parts) == 3:
                    tag = parts[2]

                # Directly update user_freq and user_word_tag_tab
                self.user_freq[word] = freq
                if tag:
                    self.user_word_tag_tab[word] = tag

                # Add prefixes
                for ch in range(len(word)):
                    wfrag = word[: ch + 1]
                    if wfrag not in self.user_freq:
                        self.user_freq[wfrag] = 0

                if freq == 0:
                    finalseg.add_force_split(word)
        finally:
            if should_close_binary:
                f_to_process.close()

    def add_word(
        self, word: str, freq: int | None = None, tag: str | None = None
    ) -> None:
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        self.check_initialized()
        word = word
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        self.user_freq[word] = freq
        # self.total += float(freq) # total is recalculated in initialize
        if tag:
            self.user_word_tag_tab[word] = tag
        for ch in range(len(word)):
            wfrag = word[: ch + 1]
            if wfrag not in self.user_freq:
                self.user_freq[wfrag] = 0
        if freq == 0:
            finalseg.add_force_split(word)

    def del_word(self, word: str) -> None:
        """
        Convenient function for deleting a word.
        """
        self.add_word(word, 0)

    def suggest_freq(self, segment: str | tuple[str, ...], tune: bool = False) -> int:
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        """
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1
        if isinstance(segment, str):
            word = segment
            for seg in self.cut(word, HMM=False):
                freq *= self.get_freq(seg) / ftotal
            freq = max(int(freq * self.total) + 1, self.get_freq(word))
        else:
            segment = tuple(map(str, segment))
            word = "".join(segment)
            for seg in segment:
                freq *= self.get_freq(seg) / ftotal
            freq = min(int(freq * self.total), self.get_freq(word))
        if tune:
            add_word(word, freq)
        return freq

    def tokenize(
        self,
        unicode_sentence: str,
        mode: str = "default",
        HMM: bool = True,
    ) -> Iterator[tuple[str, int, int]]:
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        Parameter:
            - sentence: the str(unicode) to be segmented.
            - mode: "default" or "search", "search" is for finer segmentation.
            - HMM: whether to use the Hidden Markov Model.
        """
        if not isinstance(unicode_sentence, str):
            raise ValueError("jieba: the input parameter should be unicode.")
        start = 0
        if mode == "default":
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                if len(w) > 2:
                    for i in range(len(w) - 1):
                        gram2 = w[i : i + 2]
                        if self.get_freq(gram2):
                            yield (gram2, start + i, start + i + 2)
                if len(w) > 3:
                    for i in range(len(w) - 2):
                        gram3 = w[i : i + 3]
                        if self.get_freq(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dictionary_path: str) -> None:
        with self.lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"jieba: file does not exist: {abs_path}")
            if self.dictionary != abs_path:
                self.dictionary = abs_path
                self.initialized = False
                self.user_freq = {}  # Clear user_freq
                self.user_word_tag_tab = {}  # Clear user_word_tag_tab
                # Force rebuild DatTrie when dictionary changes
                self.initialize(force_rebuild=True)
                # The user_freq and user_word_tag_tab are already cleared
                # and repopulated by initialize
                # No need to clear them again here.


# default Tokenizer instance

dt = Tokenizer()

# global functions


def get_FREQ(k: str, d: int | float | None = None) -> int | float | None:
    return dt.get_freq(k) or d


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


def _lcut(s: str) -> list[str]:
    return dt._lcut(s)


def _lcut_no_hmm(s: str) -> list[str]:
    return dt._lcut_no_hmm(s)


def _lcut_all(s: str) -> list[str]:
    return dt._lcut_all(s)


def _lcut_for_search(s: str) -> list[str]:
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s: str) -> list[str]:
    return dt._lcut_for_search_no_hmm(s)


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
    """
    Change the module's `cut` and `cut_for_search` functions to the
    parallel version.

    Note that this only works using dt, custom Tokenizer
    instances are not supported.
    """
    global pool, dt, cut, cut_for_search
    from multiprocessing import Pool, cpu_count

    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel() -> None:
    global pool, dt, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
