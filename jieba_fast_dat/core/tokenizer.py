from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import jieba_fast_dat._jieba_fast_dat_functions_py3 as _jieba_fast_dat_functions
from jieba_fast_dat._jieba_fast_dat_functions_py3 import DatTrie

from .. import finalseg
from ..utils import (
    CacheManager,
    _get_abs_path,
    get_module_res,
    strdecode,
)

__all__ = ["Tokenizer", "DEFAULT_DICT", "DEFAULT_DICT_NAME"]

# Constants
USER_DICT_CACHE_PREFIX = "jieba_fast_dat.user_dict.v2.cache"
DEFAULT_DICT: str | None = None
DEFAULT_DICT_NAME = "dict.txt"

# Logger
default_logger = logging.getLogger(__name__)

# Global locks for dictionary building
DICT_WRITING: dict[str | None, threading.RLock] = {}


def _batch_add_force_split(words: list[str]) -> None:
    """Helper for C++ callback to add force split words to finalseg.

    Args:
        words: List of words that should be force-split.
    """
    for word in words:
        finalseg.add_force_split(word)


class Tokenizer:
    """Main segmentation class using Double Array Trie (DAT) implemented in C++.

    This class provides the core interface for word segmentation, combining
    the speed of a C++ Double Array Trie with Python's flexibility.

    Attributes:
        lock (threading.RLock): Reentrant lock for thread-safe operations.
        dictionary (str | Path | None): Path to the main dictionary file.
        dat (DatTrie): The Double Array Trie object from C++ extension.
        total (int): Total word frequency in the dictionary.
        user_word_tag_tab (dict[str, str]): Table for user-defined word tags.
        initialized (bool): Whether the dictionary has been loaded and initialized.
        tmp_dir (str | None): Temporary directory for cache files.
        cache_file (str | None): Path to the specific cache file being used.
    """

    def __init__(self, dictionary: str | Path | None = DEFAULT_DICT) -> None:
        """Initializes the Tokenizer with an optional dictionary path.

        Args:
            dictionary: Path to the dictionary file. Defaults to the system default.
        """
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            assert dictionary is not None
            self.dictionary = _get_abs_path(str(dictionary))
        self.dat = DatTrie()
        self.total = 0
        self.user_word_tag_tab: dict[str, str] = {}
        self.initialized = False
        self.tmp_dir: str | None = None
        self.cache_file: str | None = None

    def __repr__(self) -> str:
        """Returns a string representation of the Tokenizer instance."""
        return f"<Tokenizer dictionary={self.dictionary!r}>"

    def get_freq(self, word: str) -> int:
        """Gets the frequency of a word from the DAT dictionary.

        Args:
            word: The word to look up.

        Returns:
            The frequency of the word, or 0 if not found.
        """
        return _jieba_fast_dat_functions._get_freq(self.dat, word)

    def initialize(
        self, dictionary: str | Path | None = None, force_rebuild: bool = False
    ) -> None:
        """Initializes the tokenizer by building or loading the prefix dictionary.

        This method is thread-safe and ensures that the dictionary is only loaded once
        unless a different dictionary path is provided or a rebuild is forced.

        Args:
            dictionary: Path to a new dictionary file to load.
            force_rebuild: If True, forces a rebuild of the DAT cache.
        """
        with self.lock:
            if dictionary:
                abs_path = _get_abs_path(str(dictionary))
                if self.dictionary == abs_path and self.initialized:
                    return
                self.dictionary = abs_path
                self.initialized = False
            else:
                abs_path = str(self.dictionary) if self.dictionary else None

            if self.initialized:
                return

            default_logger.debug(
                f"Building prefix dict from {abs_path or 'the default dictionary'} ..."
            )
            t1 = time.time()

            prefix = (
                "jieba_fast_dat"
                if abs_path is None or abs_path == DEFAULT_DICT
                else "jieba_fast_dat.u"
            )

            load_success = CacheManager.load_trie(
                self.dat,
                abs_path,
                prefix=prefix,
                cache_file=self.cache_file,
                logger=default_logger,
                force_rebuild=force_rebuild,
            )

            if load_success:
                self.total = self.dat.total_freq
            else:
                self._build_dict_from_file(abs_path, prefix)

            self.initialized = True
            default_logger.debug(f"Loading model cost {time.time() - t1:.3f} seconds.")
            default_logger.debug("Prefix dict has been built successfully.")

    def _build_dict_from_file(self, abs_path: str | None, prefix: str) -> None:
        """Internal method to build the DAT dictionary from a text file and save cache.

        Args:
            abs_path: Absolute path to the dictionary file.
            prefix: Cache file prefix.
        """
        wlock = DICT_WRITING.get(abs_path, threading.RLock())
        DICT_WRITING[abs_path] = wlock
        with wlock:
            default_logger.debug("Parsing dictionary file (C++ based)...")
            # Ensure we get the correct path for the default dict
            dict_file = self.get_dict_file()
            dict_path = dict_file.name
            dict_file.close()  # Close immediately as C++ will open it

            self.total = _jieba_fast_dat_functions.load_main_dict_from_path_pybind(
                self.dat,
                dict_path,
                {},
            )

            CacheManager.save_trie(
                self.dat,
                abs_path,
                prefix=prefix,
                cache_file=self.cache_file,
                logger=default_logger,
            )

        try:
            del DICT_WRITING[abs_path]
        except KeyError:
            pass

    def check_initialized(self) -> None:
        """Ensures the tokenizer is initialized before performing any operations.

        If not initialized, it triggers the default initialization process.
        """
        if not self.initialized:
            self.initialize()

    def calc(
        self,
        sentence: str,
        DAG: dict[int, list[int]],
        route: dict[int, tuple[float, int]],
    ) -> None:
        """Calculates the best route for segmentation using dynamic programming.

        Args:
            sentence: The input sentence string.
            DAG: The Directed Acyclic Graph of the sentence.
            route: A dictionary to store the calculated best route.
        """
        self.check_initialized()
        _jieba_fast_dat_functions._calc(
            self.dat,
            sentence,
            DAG,
            route,
            float(self.total),
        )

    def get_DAG(self, sentence: str) -> dict[int, list[int]]:
        """Generates a Directed Acyclic Graph (DAG) for the given sentence.

        Args:
            sentence: The input sentence string.

        Returns:
            A dictionary where keys are character positions and values are lists
            of possible ending positions for words starting at that position.
        """
        self.check_initialized()
        return _jieba_fast_dat_functions._get_DAG(self.dat, sentence)

    def lcut(
        self,
        sentence: str,
        cut_all: bool = False,
        HMM: bool = True,
        use_paddle: bool = False,
    ) -> list[str]:
        """Performs word segmentation and returns a list of words.

        Args:
            sentence: The input sentence string.
            cut_all: Whether to use full-segmentation mode.
            HMM: Whether to use Hidden Markov Model for unknown word detection.
            use_paddle: Whether to use PaddlePaddle (not supported in this version).

        Returns:
            A list of segmented words.
        """
        sentence = strdecode(sentence)
        if cut_all:
            return _jieba_fast_dat_functions._cut_all_internal_cpp(self.dat, sentence)
        else:
            if HMM and not finalseg._initialized:
                finalseg.load_model()
            return _jieba_fast_dat_functions._cut_internal_cpp(
                self.dat, sentence, float(self.total), HMM
            )

    def lcut_for_search(self, sentence: str, HMM: bool = True) -> list[str]:
        """Performs word segmentation for search engines and returns a list of words.

        This mode segments long words further into shorter words.

        Args:
            sentence: The input sentence string.
            HMM: Whether to use Hidden Markov Model for unknown word detection.

        Returns:
            A list of segmented words optimized for search indexing.
        """
        sentence = strdecode(sentence)
        if HMM and not finalseg._initialized:
            finalseg.load_model()
        return _jieba_fast_dat_functions._cut_for_search_internal_cpp(
            self.dat, sentence, float(self.total), HMM
        )

    def cut(
        self,
        sentence: str,
        cut_all: bool = False,
        HMM: bool = True,
        use_paddle: bool = False,
    ) -> Iterator[str]:
        """Main segmentation function that yields words.

        Args:
            sentence: The input sentence string.
            cut_all: Whether to use full-segmentation mode.
            HMM: Whether to use Hidden Markov Model for unknown word detection.
            use_paddle: Whether to use PaddlePaddle (not supported in this version).

        Yields:
            Segmented words as an iterator.
        """
        return iter(
            self.lcut(sentence, cut_all=cut_all, HMM=HMM, use_paddle=use_paddle)
        )

    def cut_for_search(self, sentence: str, HMM: bool = True) -> Iterator[str]:
        """Finer segmentation for search engines that yields words.

        Args:
            sentence: The input sentence string.
            HMM: Whether to use Hidden Markov Model for unknown word detection.

        Yields:
            Segmented words as an iterator.
        """
        return iter(self.lcut_for_search(sentence, HMM=HMM))

    def get_dict_file(self) -> IO[bytes]:
        """Gets the file handle for the dictionary.

        Returns:
            A binary file-like object for the dictionary.
        """
        if self.dictionary == DEFAULT_DICT:
            return get_module_res("jieba_fast_dat", DEFAULT_DICT_NAME)
        else:
            return open(str(self.dictionary), "rb")

    def load_userdict(self, f: str | Path) -> None:
        """Loads a user dictionary from a file.

        Args:
            f: Path to the user dictionary file.

        Raises:
            TypeError: If f is not a string or Path object.
            Exception: If C++ dictionary loading fails.
        """
        self.check_initialized()

        if not isinstance(f, (str, Path)):
            raise TypeError(
                "File-like objects (BinaryIO) are not supported for load_userdict; "
                "please provide a file path."
            )

        user_dict_path = str(f)

        # Attempt to load from binary cache
        dat = DatTrie()
        if CacheManager.load_trie(
            dat, user_dict_path, prefix=USER_DICT_CACHE_PREFIX, logger=default_logger
        ):
            self.dat = dat
            self.total = self.dat.total_freq
            return

        default_logger.debug(
            f"User dict '{user_dict_path}' cache invalid/not found; loading from file."
        )

        try:
            new_total_freq = _jieba_fast_dat_functions.load_userdict_pybind(
                self.dat,
                user_dict_path,
                self.user_word_tag_tab,
                _batch_add_force_split,
            )
            self.total = new_total_freq

            CacheManager.save_trie(
                self.dat,
                user_dict_path,
                prefix=USER_DICT_CACHE_PREFIX,
                logger=default_logger,
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
        """Adds a word to the dictionary at runtime.

        Args:
            word: The word to add.
            freq: Optional frequency. If None, it's suggested automatically.
            tag: Optional POS tag.
        """
        self.check_initialized()
        word = strdecode(word)
        if freq is None:
            freq = self.suggest_freq(word)
        self.dat.add_word(word, freq, tag if tag else "")
        self.user_word_tag_tab[word] = tag if tag else ""
        self.total += freq

    def del_word(self, word: str) -> None:
        """Deletes a word from the dictionary at runtime.

        Args:
            word: The word to delete.
        """
        self.check_initialized()
        word = strdecode(word)
        old_freq = self.get_freq(word)
        self.dat.add_word(word, 0, "")
        if word in self.user_word_tag_tab:
            del self.user_word_tag_tab[word]
        self.total -= old_freq

    def suggest_freq(self, segment: str | tuple[str, ...], tune: bool = False) -> int:
        """Suggests a word frequency to adjust segmentation probability.

        Args:
            segment: A word or a tuple of segments.
            tune: If True, automatically adds the word with suggested frequency.

        Returns:
            The suggested frequency.
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
            self.add_word(word, freq)
        return freq

    def tokenize(
        self,
        unicode_sentence: str,
        mode: str = "default",
        HMM: bool = True,
    ) -> Iterator[tuple[str, int, int]]:
        """Tokenizes a sentence and returns generator of (word, start, end) tuples.

        Args:
            unicode_sentence: The input sentence string.
            mode: Tokenization mode ('default' or 'search').
            HMM: Whether to use Hidden Markov Model.

        Returns:
            An iterator of tuples containing word, start index, and end index.

        Raises:
            ValueError: If the input is not a string.
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

    def set_dictionary(self, dictionary_path: str | Path) -> None:
        """Sets a custom main dictionary path and re-initializes.

        Args:
            dictionary_path: Path to the dictionary file.

        Raises:
            FileNotFoundError: If the dictionary file does not exist.
        """
        with self.lock:
            abs_path = _get_abs_path(str(dictionary_path))
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"jieba: file does not exist: {abs_path}")
            if self.dictionary != abs_path:
                self.dictionary = abs_path
                self.initialized = False
                self.user_word_tag_tab = {}
                self.initialize(force_rebuild=True)
