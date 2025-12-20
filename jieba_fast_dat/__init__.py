from collections.abc import Iterator
from typing import IO

__version__ = "0.57"

import logging
import os
import pickle
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
from .utils import _get_abs_path, get_module_res

load_hmm_model = _jieba_fast_dat_functions.load_hmm_model
_posseg_viterbi_cpp = _jieba_fast_dat_functions._posseg_viterbi_cpp
_get_DAG = _jieba_fast_dat_functions._get_DAG
_get_freq = _jieba_fast_dat_functions._get_freq
load_userdict_pybind = _jieba_fast_dat_functions.load_userdict_pybind
_cut_internal_cpp = _jieba_fast_dat_functions._cut_internal_cpp
_cut_all_internal_cpp = _jieba_fast_dat_functions._cut_all_internal_cpp
_cut_for_search_internal_cpp = _jieba_fast_dat_functions._cut_for_search_internal_cpp

_replace_file = os.rename


# 新增自定義字典快取專用的常量
USER_DICT_CACHE_PREFIX = "jieba_fast_dat.user_dict.cache"
TMP_DIR = Path(tempfile.gettempdir())  # 統一快取存放在系統的 /tmp/ 目錄下


def _get_user_dict_cache_paths(user_dict_path: str) -> Path:
    """
    為自定義字典生成唯一的快取檔案路徑。
    命名規範：/tmp/jieba_fast_dat.user_dict.cache.{md5_hash_of_path}.cache
    """
    user_dict_path_hash = md5(user_dict_path.encode("utf-8")).hexdigest()
    cache_file = TMP_DIR / f"{USER_DICT_CACHE_PREFIX}.{user_dict_path_hash}.cache"
    return cache_file


def _load_user_dict_cache(
    user_dict_path: str,
) -> tuple[dict, float, dict[str, str], bytes] | None:
    """
    嘗試從自定義字典快取載入元數據、總頻率、user_word_tag_tab 和 DatTrie 字節。
    返回 (metadata, total_freq, user_word_tag_tab, dat_bytes) 或 None。
    """
    cache_file = _get_user_dict_cache_paths(user_dict_path)

    if not cache_file.exists():
        return None

    try:
        original_stat = os.stat(user_dict_path)
        original_mtime = int(original_stat.st_mtime)
        original_size = original_stat.st_size
    except FileNotFoundError:
        cache_file.unlink(missing_ok=True)
        return None

    try:
        with open(cache_file, "rb") as cf:
            cached_data = pickle.loads(cf.read())
            # Expecting (metadata, total_freq, user_word_tag_tab, dat_bytes)
            if not isinstance(cached_data, tuple) or len(cached_data) != 4:
                raise ValueError("Invalid cache format.")
            (
                metadata,
                total_freq,
                user_word_tag_tab_from_cache,
                dat_bytes,
            ) = cached_data

            if (
                not isinstance(metadata, dict)
                or "mtime" not in metadata
                or "size" not in metadata
            ):
                raise ValueError("Invalid metadata in cache.")

            cached_mtime = int(metadata["mtime"])
            cached_size = metadata["size"]

    except Exception as e:
        default_logger.debug(
            f"Failed to load user dict cache from {cache_file}, rebuilding: {e}"
        )
        cache_file.unlink(missing_ok=True)
        return None

    if cached_mtime != original_mtime or cached_size != original_size:
        default_logger.debug(
            f"User dict '{user_dict_path}' cache outdated. Rebuilding."
        )
        cache_file.unlink(missing_ok=True)
        return None

    default_logger.debug(
        f"User dict '{user_dict_path}' cache valid. Loading from cache."
    )
    return metadata, total_freq, user_word_tag_tab_from_cache, dat_bytes


def _save_user_dict_cache(
    user_dict_path: str,
    total_freq: float,
    user_word_tag_tab: dict[str, str],
    dat_bytes: bytes,
):
    """
    保存自定義字典的快取資訊（元數據、總頻率、user_word_tag_tab 和 DatTrie 字節）。
    """
    cache_file = _get_user_dict_cache_paths(user_dict_path)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    default_logger.debug(f"Dumping user dict cache to {cache_file}")
    try:
        original_stat = os.stat(user_dict_path)
        metadata = {"mtime": int(original_stat.st_mtime), "size": original_stat.st_size}

        with open(cache_file, "wb") as temp_cache_file:
            pickle.dump(
                (metadata, total_freq, user_word_tag_tab, dat_bytes),
                temp_cache_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        default_logger.debug(f"User dict cache dumped to {cache_file} successfully.")
    except Exception as e:
        default_logger.exception(
            f"Dump user dict cache file failed for {user_dict_path}: {e}"
        )
        cache_file.unlink(missing_ok=True)


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


def _batch_add_force_split(words: list[str]) -> None:
    for word in words:
        finalseg.add_force_split(word)


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
        self.user_word_tag_tab: dict[str, str] = {}
        self.initialized = False
        self.tmp_dir: str | None = None
        self.cache_file: str | None = None

    def __repr__(self) -> str:
        return f"<Tokenizer dictionary={self.dictionary!r}>"

    def get_freq(self, word: str) -> int:
        return _get_freq(self.dat, word)

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

            load_from_cache_fail = True
            if (
                not force_rebuild
                and os.path.isfile(cache_file)
                and (
                    abs_path == DEFAULT_DICT
                    or (
                        abs_path is not None
                        and os.path.getmtime(cache_file) > os.path.getmtime(abs_path)
                    )
                )
            ):
                default_logger.debug(f"Loading model from cache {cache_file}")
                try:
                    with open(cache_file, "rb") as cf:
                        self.total, self.user_word_tag_tab, dat_bytes = pickle.loads(
                            cf.read()
                        )
                    self.dat.load_from_bytes(dat_bytes)
                    # Sync word_tag_tab to C++
                    self.dat.update_word_tag_tab(self.user_word_tag_tab)
                    load_from_cache_fail = False
                    # Clear user_freq when loading from cache,
                    # user_word_tag_tab will be repopulated
                    # by gen_dat_data
                    # self.user_word_tag_tab = {} # Do not clear here
                    # Repopulate user_word_tag_tab from main dictionary
                    # after loading from cache
                    # This ensures that main dictionary word tags are available
                    # even when loading from cache

                except Exception:
                    load_from_cache_fail = True

            if load_from_cache_fail:
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                with wlock:
                    default_logger.debug("Parsing dictionary file (C++ based)...")
                    # Call the C++ function to load the main dictionary
                    self.total = (
                        _jieba_fast_dat_functions.load_main_dict_from_path_pybind(
                            self.dat,
                            self.get_dict_file().name,
                            self.user_word_tag_tab,  # This will be populated by C++
                        )
                    )

                    # Get the DatTrie's binary data
                    dat_bytes = self.dat.save_to_bytes()

                    default_logger.debug(f"Dumping model to file cache {cache_file}")
                    try:
                        # Save total, user_word_tag_tab, and dat_bytes to .cache file
                        with open(cache_file, "wb") as temp_cache_file:
                            pickle.dump(
                                (self.total, self.user_word_tag_tab, dat_bytes),
                                temp_cache_file,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )
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
        )

    def get_DAG(self, sentence: str) -> dict[int, list[int]]:
        self.check_initialized()
        return _get_DAG(self.dat, sentence)

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
        self.check_initialized()
        words = _jieba_fast_dat_functions._cut_DAG_NO_HMM_cpp(
            self.dat, sentence, float(self.total)
        )
        yield from words

    def __cut_DAG(self, sentence: str) -> Iterator[str]:
        self.check_initialized()
        if not finalseg._initialized:
            finalseg.load_model()
        words = _jieba_fast_dat_functions._cut_DAG_cpp(
            self.dat, sentence, float(self.total)
        )
        yield from words

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
        return iter(
            self.lcut(sentence, cut_all=cut_all, HMM=HMM, use_paddle=use_paddle)
        )

    def cut_for_search(self, sentence: str, HMM: bool = True) -> Iterator[str]:
        """
        Finer segmentation for search engines.
        """
        return iter(self.lcut_for_search(sentence, HMM=HMM))

    def lcut(
        self,
        sentence: str,
        cut_all: bool = False,
        HMM: bool = True,
        use_paddle: bool = False,
    ) -> list[str]:
        sentence = strdecode(sentence)
        if cut_all:
            return _cut_all_internal_cpp(self.dat, sentence)
        else:
            if HMM and not finalseg._initialized:
                finalseg.load_model()
            return _cut_internal_cpp(self.dat, sentence, float(self.total), HMM)

    def lcut_for_search(self, sentence: str, HMM: bool = True) -> list[str]:
        sentence = strdecode(sentence)
        if HMM and not finalseg._initialized:
            finalseg.load_model()
        return _cut_for_search_internal_cpp(self.dat, sentence, float(self.total), HMM)

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

    def load_userdict(self, f: str | Path) -> None:
        """
        Load personalized dict to improve detect rate.

        Parameter:
            - f : The path of the dictionary file, whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        """
        self.check_initialized()

        if not isinstance(f, (str, Path)):
            raise TypeError(
                "File-like objects (BinaryIO) are not supported for load_userdict; "
                "please provide a file path."
            )

        user_dict_path = str(f)
        cache_file_path = _get_user_dict_cache_paths(user_dict_path)

        # Attempt to load from binary cache
        loaded_cache = _load_user_dict_cache(user_dict_path)

        if loaded_cache:
            default_logger.debug(
                f"User dict '{user_dict_path}' cache valid. Loading from cache."
            )
            try:
                # Unpack all four items from the cache
                (
                    _,
                    total_freq_from_cache,
                    user_word_tag_tab_from_cache,
                    dat_bytes,
                ) = loaded_cache
                self.total = total_freq_from_cache
                self.user_word_tag_tab = user_word_tag_tab_from_cache

                # Load DatTrie structure from bytes
                self.dat.load_from_bytes(dat_bytes)

                default_logger.debug(
                    f"User dict '{user_dict_path}' loaded from binary cache."
                )
                return

            except Exception as e:
                default_logger.exception(
                    f"Failed to load user dict from binary cache for "
                    f"'{user_dict_path}', rebuilding: {e}"
                )
                # Fallback to loading from original file if cache failed
                cache_file_path.unlink(missing_ok=True)

        # If cache invalid/not found/failed, load from original text file
        default_logger.debug(
            f"User dict '{user_dict_path}' cache invalid/not found; loading from file."
        )

        try:
            # load_userdict_pybind will rebuild self.dat and populate user_word_tag_tab
            new_total_freq = load_userdict_pybind(
                self.dat,
                user_dict_path,  # Directly pass the file path
                self.user_word_tag_tab,  # This will be populated by C++
                _batch_add_force_split,  # Pass the new batch function
            )
            self.total = new_total_freq

            # Get the trie data as bytes
            dat_bytes = self.dat.save_to_bytes()

            # Save new binary cache
            _save_user_dict_cache(
                user_dict_path, new_total_freq, self.user_word_tag_tab, dat_bytes
            )

            default_logger.debug(
                f"User dict '{user_dict_path}' loaded from file, new cache saved."
            )
            return
        except Exception as e:
            default_logger.exception(
                f"C++ load_userdict failed for '{user_dict_path}': {e}"
            )
            raise

    def add_word(
        self, word: str, freq: int | None = None, tag: str | None = None
    ) -> None:
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        self.check_initialized()
        word = strdecode(word)
        if freq is None:
            freq = self.suggest_freq(word)
        self.dat.add_word(word, freq, tag if tag else "")
        self.user_word_tag_tab[word] = tag if tag else ""
        self.total += freq

    def del_word(self, word: str) -> None:
        """
        Convenient function for deleting a word.
        """
        self.check_initialized()
        word = strdecode(word)
        old_freq = self.get_freq(word)
        self.dat.add_word(word, 0, "")
        if word in self.user_word_tag_tab:
            del self.user_word_tag_tab[word]
        self.total -= old_freq

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
