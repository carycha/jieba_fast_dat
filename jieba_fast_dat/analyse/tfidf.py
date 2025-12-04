import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection
from operator import itemgetter
from typing import Any

import jieba_fast_dat
import jieba_fast_dat.posseg

# Local application imports
from ..utils import _get_abs_path


def _get_module_path(path: str) -> str:
    return os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), path))


DEFAULT_IDF = _get_module_path("idf.txt")


class KeywordExtractor(ABC):
    def __init__(self) -> None:
        self.stop_words = self.STOP_WORDS.copy()

    STOP_WORDS: set[str] = {
        "the",
        "of",
        "is",
        "and",
        "to",
        "in",
        "that",
        "we",
        "for",
        "an",
        "are",
        "by",
        "be",
        "as",
        "on",
        "with",
        "can",
        "if",
        "from",
        "which",
        "you",
        "it",
        "this",
        "then",
        "at",
        "have",
        "all",
        "not",
        "one",
        "has",
        "or",
    }

    def set_stop_words(self, stop_words_path: str) -> None:
        abs_path = _get_abs_path(stop_words_path)
        if not os.path.isfile(abs_path):
            raise Exception("jieba_fast_dat: file does not exist: " + abs_path)
        content = open(abs_path, "rb").read().decode("utf-8")
        for line in content.splitlines():
            self.stop_words.add(line)

    @abstractmethod
    def extract_tags(
        self,
        sentence: str,
        topK: int | None = 20,
        withWeight: bool = False,
        allowPOS: tuple[str, ...] = (),
        withFlag: bool = False,
    ) -> list[Any]:
        raise NotImplementedError


class IDFLoader:
    def __init__(self, idf_path: str | None = None) -> None:
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path: str) -> None:
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, encoding="utf-8").read()
            self.idf_freq = {}
            for line in content.splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    word, freq = parts
                    self.idf_freq[word] = float(freq)
                else:
                    # Handle cases where line might not contain a space
                    # or has unexpected format, e.g., log a warning or skip
                    pass
            self.median_idf = sorted(self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self) -> tuple[dict[str, float], float]:
        return self.idf_freq, self.median_idf


class TFIDF(KeywordExtractor):
    def __init__(self, idf_path: str | None = None) -> None:
        self.tokenizer = jieba_fast_dat.dt
        self.postokenizer = jieba_fast_dat.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path: str) -> None:
        new_abs_path = _get_abs_path(idf_path)
        if not os.path.isfile(new_abs_path):
            raise Exception("jieba_fast_dat: file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def extract_tags(
        self,
        sentence: str,
        topK: int | None = 20,
        withWeight: bool = False,
        allowPOS: Collection[str] = (),
        withFlag: bool = False,
    ) -> list[Any]:
        """
        Extract keywords from sentence using TF-IDF algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr'].
                        if the POS of w is not in this list,it will be filtered.
            - withFlag: only work with allowPOS is not empty.
                        if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        if allowPOS:
            allowPOS = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence)
        else:
            words = self.tokenizer.cut(sentence)
        freq: defaultdict[Any, float] = defaultdict(float)
        for w in words:
            if allowPOS:
                if (
                    not isinstance(w, jieba_fast_dat.posseg.pair)
                    or w.flag not in allowPOS
                ):
                    continue
                elif not withFlag:
                    word_to_count = w.word
                else:
                    word_to_count = w
            else:
                word_to_count = w

            if isinstance(word_to_count, jieba_fast_dat.posseg.pair):
                word_str = word_to_count.word
            else:
                word_str = word_to_count

            if len(word_str.strip()) < 2 or word_str.lower() in self.stop_words:
                continue
            freq[word_to_count] += 1.0
        total = sum(freq.values())
        for k in freq:
            if isinstance(k, jieba_fast_dat.posseg.pair):
                kw = k.word
            else:
                kw = k
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        if topK:
            return tags[:topK]
        else:
            return tags
