from __future__ import absolute_import, unicode_literals
__version__ = '0.39'
__license__ = 'MIT'

import re
import os
import sys
import time
import logging
import marshal
import tempfile
import threading
from math import log
from hashlib import md5
from ._compat import *
from . import finalseg
import platform
import hashlib # Added this line

from . import _jieba_fast_functions_pybind as _jieba_fast_functions

replace_file = os.rename

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

DICT_WRITING = {}

pool = None

re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)

re_eng = re.compile('[a-zA-Z0-9]', re.U)

# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
re_han_default = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
re_skip_default = re.compile(r"(\r\n|\s)", re.U)
re_han_cut_all = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.U)

def setLogLevel(log_level):
    global logger
    default_logger.setLevel(log_level)

class Tokenizer(object):

    def __init__(self, dictionary=DEFAULT_DICT):
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = get_module_res_path(DEFAULT_DICT_NAME) # Use get_module_res_path to get the full path string
        else:
            self.dictionary = _get_abs_path(dictionary)
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    def __repr__(self):
        return '<Tokenizer dictionary=%r>' % self.dictionary

    def initialize(self, dictionary=None):
        if dictionary:
            abs_path = _get_abs_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                default_logger.debug(f"Dictionary '{abs_path}' already initialized. Skipping.")
                return
            else:
                self.dictionary = abs_path
                self.initialized = False
        else:
            abs_path = self.dictionary

        with self.lock:
            try:
                with DICT_WRITING[abs_path]:
                    pass
            except KeyError:
                pass
            if self.initialized:
                default_logger.debug(f"Dictionary '{abs_path}' already initialized. Skipping.")
                return

            default_logger.debug(f"Initializing dictionary from {abs_path or 'the default dictionary'}...")
            t1 = time.time()

            default_logger.debug(f"Determining cache path for '{abs_path}'...")
            cache_file_path = get_cache_file_path(abs_path)
            if cache_file_path is None:
                default_logger.error(f"Failed to get cache file path for {abs_path}. Proceeding without cache.")
                # Fallback to original dictionary loading
                if _jieba_fast_functions.load_dict(abs_path):
                    self.initialized = True
                    default_logger.debug(
                        f"Loading model from original file cost {time.time() - t1:.3f} seconds. "
                        f"Vocabulary size: {_jieba_fast_functions.get_word_count()} words."
                    )
                    default_logger.debug("Prefix dict has been built successfully from original file.")
                else:
                    default_logger.error(f"Failed to load dictionary from {abs_path}")
                    self.initialized = False
                return

            # Try to load from cache
            trie_cache_exists = os.path.exists(cache_file_path + ".trie")
            freq_cache_exists = os.path.exists(cache_file_path + ".freq")

            if trie_cache_exists and freq_cache_exists:
                default_logger.debug(f"Attempting to load dictionary from cache: {cache_file_path}.trie and {cache_file_path}.freq")
                t_cache_load = time.time()
                if _jieba_fast_functions.open_dat(cache_file_path):
                    self.initialized = True
                    default_logger.debug(
                        f"Loading model from cache cost {time.time() - t_cache_load:.3f} seconds. "
                        f"Total initialization cost {time.time() - t1:.3f} seconds. "
                        f"Vocabulary size: {_jieba_fast_functions.get_word_count()} words."
                    )
                    default_logger.debug("Prefix dict has been built successfully from cache.")
                    return
                else:
                    default_logger.warning(f"Failed to load dictionary from cache: {cache_file_path}. Falling back to original file.")
            else:
                default_logger.debug(f"Cache files not found for {abs_path}. Trie: {trie_cache_exists}, Freq: {freq_cache_exists}. Building from original.")

            # Fallback to original dictionary loading if cache fails or doesn't exist
            default_logger.debug(f"Loading dictionary from original file: {abs_path}")
            t_original_load = time.time()
            if _jieba_fast_functions.load_dict(abs_path):
                self.initialized = True
                default_logger.debug(
                    f"Loading model from original file cost {time.time() - t_original_load:.3f} seconds. "
                    f"Total initialization cost {time.time() - t1:.3f} seconds. "
                    f"Vocabulary size: {_jieba_fast_functions.get_word_count()} words."
                )
                default_logger.debug("Prefix dict has been built successfully from original file.")

                # Save to cache after successful loading from original
                default_logger.debug(f"Saving dictionary to cache: {cache_file_path}.trie and {cache_file_path}.freq")
                t_cache_save = time.time()
                if _jieba_fast_functions.save_dat(cache_file_path):
                    default_logger.debug(f"Saving cache cost {time.time() - t_cache_save:.3f} seconds.")
                else:
                    default_logger.error(f"Failed to save cache for {abs_path}.")
                time.sleep(0.1) # Add a small delay to ensure file is written to disk
            else:
                default_logger.error(f"Failed to load dictionary from {abs_path}")
                self.initialized = False # Ensure initialized is False on failure

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def calc(self, sentence, DAG, route):
        _jieba_fast_functions._calc(sentence, DAG, route, _jieba_fast_functions.get_total_frequency())

    def get_DAG(self, sentence):
        self.check_initialized()
        DAG = {}
        _jieba_fast_functions._get_DAG(DAG, sentence)
        return DAG

    def __cut_all(self, sentence):
        dag = self.get_DAG(sentence)
        old_j = -1
        for k, L in iteritems(dag):
            if len(L) == 1 and k > old_j:
                yield sentence[k:L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j

    def __cut_DAG_NO_HMM(self, sentence):
        self.check_initialized()
        route = []
        _jieba_fast_functions._get_DAG_and_calc(sentence, route, _jieba_fast_functions.get_total_frequency())
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''

    def __cut_DAG(self, sentence):
        self.check_initialized()
        route = []
        _jieba_fast_functions._get_DAG_and_calc(sentence, route, _jieba_fast_functions.get_total_frequency())
        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    else:
                        if not _jieba_fast_functions.get_word_frequency(buf):
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not _jieba_fast_functions.get_word_frequency(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(self, sentence, cut_all=False, HMM=True):
        '''
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.

        Parameter:
            - sentence: The str(unicode) to be segmented.
            - cut_all: Model type. True for full pattern, False for accurate pattern.
            - HMM: Whether to use the Hidden Markov Model.
        '''
        sentence = strdecode(sentence)

        if cut_all:
            re_han = re_han_cut_all
            re_skip = re_skip_cut_all
        else:
            re_han = re_han_default
            re_skip = re_skip_default
        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            if re_han.match(blk):
                for word in cut_block(blk):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x

    def cut_for_search(self, sentence, HMM=True):
        """
        Finer segmentation for search engines.
        """
        words = self.cut(sentence, HMM=HMM)
        for w in words:
            if len(w) > 2:
                for i in xrange(len(w) - 1):
                    gram2 = w[i:i + 2]
                    if _jieba_fast_functions.get_word_frequency(gram2):
                        yield gram2
            if len(w) > 3:
                for i in xrange(len(w) - 2):
                    gram3 = w[i:i + 3]
                    if _jieba_fast_functions.get_word_frequency(gram3):
                        yield gram3
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs):
        return list(self.cut_for_search(*args, **kwargs))

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence):
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence):
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence):
        return self.lcut_for_search(sentence, False)

    def get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            return open(get_module_res_path(DEFAULT_DICT_NAME), 'rb')
        else:
            return open(self.dictionary, 'rb')

    def load_userdict(self, f):
        self.set_dictionary(f)

    def add_word(self, word, freq=None, tag=None):
        pass

    def del_word(self, word):
        pass

    def suggest_freq(self, segment, tune=False):
        pass

    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        Parameter:
            - sentence: the str(unicode) to be segmented.
            - mode: "default" or "search", "search" is for finer segmentation.
            - HMM: whether to use the Hidden Markov Model.
        """
        if not isinstance(unicode_sentence, text_type):
            raise ValueError("jieba: the input parameter should be unicode.")
        start = 0
        if mode == 'default':
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                if len(w) > 2:
                    for i in xrange(len(w) - 1):
                        gram2 = w[i:i + 2]
                        if _jieba_fast_functions.get_word_frequency(gram2):
                            yield (gram2, start + i, start + i + 2)
                if len(w) > 3:
                    for i in xrange(len(w) - 2):
                        gram3 = w[i:i + 3]
                        if _jieba_fast_functions.get_word_frequency(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dictionary_path):
        with self.lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise Exception("jieba: file does not exist: " + abs_path)
            self.dictionary = abs_path
            self.initialized = False


# default Tokenizer instance

dt = Tokenizer()

# global functions

calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut(s):
    return dt._lcut(s)


def _lcut_no_hmm(s):
    return dt._lcut_no_hmm(s)


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut_for_search(s):
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s):
    return dt._lcut_for_search_no_hmm(s)


def _pcut(sentence, cut_all=False, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def _pcut_for_search(sentence, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def enable_parallel(processnum=None):
    """
    Change the module's `cut` and `cut_for_search` functions to the
    parallel version.

    Note that this only works using dt, custom Tokenizer
    instances are not supported.
    """
    global pool, dt, cut, cut_for_search
    from multiprocessing import cpu_count
    if os.name == 'nt':
        raise NotImplementedError(
            "jieba: parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel():
    global pool, dt, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search

def get_cache_file_path(dict_path):
    """
    Generates a unique cache file path based on the dictionary path and its content.
    """
    # Use a temporary directory for cache for simplicity
    cache_dir = os.path.join(tempfile.gettempdir(), "jieba_fast_dat_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Calculate MD5 hash of the dictionary content
    try:
        with open(dict_path, 'rb') as f:
            dict_hash = hashlib.md5(f.read()).hexdigest()
    except IOError:
        default_logger.error(f"Error: Could not read dictionary file for hashing: {dict_path}")
        return None # Indicate failure

    cache_file_name = f"jieba_fast_dat_{dict_hash}"
    full_cache_path = os.path.join(cache_dir, cache_file_name)
    return full_cache_path
    global pool, dt, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
