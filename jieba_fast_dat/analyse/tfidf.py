from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection
from operator import itemgetter
from pathlib import Path
from typing import Any

import jieba_fast_dat
import jieba_fast_dat.posseg

# Local application imports
from ..utils import _get_abs_path

DEFAULT_IDF = Path(__file__).parent / "idf.txt"


class KeywordExtractor(ABC):
    """Base class for keyword extraction algorithms."""

    def __init__(self) -> None:
        self.stop_words: set[str] = self.STOP_WORDS.copy()

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

    def set_stop_words(self, stop_words_path: str | Path) -> None:
        """Set custom stop words from a file."""
        abs_path = Path(_get_abs_path(str(stop_words_path)))
        if not abs_path.is_file():
            raise FileNotFoundError(f"jieba_fast_dat: file does not exist: {abs_path}")
        content = abs_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            self.stop_words.add(line.strip())

    @abstractmethod
    def extract_tags(
        self,
        sentence: str,
        topK: int | None = 20,
        withWeight: bool = False,
        allowPOS: Collection[str] = (),
        withFlag: bool = False,
    ) -> list[Any]:
        """Abstract method for extracting keywords."""
        raise NotImplementedError


class IDFLoader:
    """Loader for Inverse Document Frequency (IDF) dictionary."""

    def __init__(self, idf_path: str | Path | None = None) -> None:
        self.path: Path | None = None
        self.idf_freq: dict[str, float] = {}
        self.median_idf = 0.0
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path: str | Path) -> None:
        """Update IDF dictionary from a new file."""
        new_path = Path(_get_abs_path(str(new_idf_path)))
        if self.path != new_path:
            self.path = new_path
            content = new_path.read_text(encoding="utf-8")
            self.idf_freq = {}
            for line in content.splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    word, freq = parts
                    self.idf_freq[word] = float(freq)

            if self.idf_freq:
                self.median_idf = sorted(self.idf_freq.values())[
                    len(self.idf_freq) // 2
                ]
            else:
                self.median_idf = 0.0

    def get_idf(self) -> tuple[dict[str, float], float]:
        """Return the IDF frequency dictionary and its median value."""
        return self.idf_freq, self.median_idf


class TFIDF(KeywordExtractor):
    """TF-IDF keyword extraction."""

    def __init__(self, idf_path: str | Path | None = None) -> None:
        super().__init__()
        self.tokenizer = jieba_fast_dat.dt
        self.postokenizer = jieba_fast_dat.posseg.dt
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path: str | Path) -> None:
        """Set a custom IDF path."""
        self.idf_loader.set_new_path(idf_path)
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

        Args:
            sentence: The input text to analyze.
            topK: Return top K keywords. None for all.
            withWeight: If True, return (word, weight) pairs.
            allowPOS: Filter words by parts of speech.
            withFlag: If True, return pair(word, weight) with POS flag.
        """
        if allowPOS:
            allowPOS_set = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence)
        else:
            words = self.tokenizer.cut(sentence)

        freq: defaultdict[Any, float] = defaultdict(float)
        for w in words:
            if allowPOS:
                if isinstance(w, jieba_fast_dat.posseg.pair):
                    if w.flag not in allowPOS_set:  # type: ignore[attr-defined]
                        continue
                    word_to_count = w if withFlag else w.word  # type: ignore[attr-defined]
                else:
                    continue
            else:
                word_to_count = w

            if isinstance(word_to_count, jieba_fast_dat.posseg.pair):
                word_str = word_to_count.word  # type: ignore[attr-defined]
            else:
                word_str = str(word_to_count)

            if len(word_str.strip()) < 2 or word_str.lower() in self.stop_words:
                continue
            freq[word_to_count] += 1.0

        total = sum(freq.values())
        for k in freq:
            if isinstance(k, jieba_fast_dat.posseg.pair):
                kw = k.word
            else:
                kw = str(k)
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)

        if topK:
            return tags[:topK]
        return tags
