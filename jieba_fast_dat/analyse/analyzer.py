import re
from collections.abc import Callable, Generator

from whoosh.analysis import LowercaseFilter, StemFilter, StopFilter, Tokenizer
from whoosh.analysis import Token as WhooshToken
from whoosh.lang.porter import stem

import jieba_fast_dat

STOP_WORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "for",
        "from",
        "have",
        "if",
        "in",
        "is",
        "it",
        "may",
        "not",
        "of",
        "on",
        "or",
        "tbd",
        "that",
        "the",
        "this",
        "to",
        "us",
        "we",
        "when",
        "will",
        "with",
        "yet",
        "you",
        "your",
        "的",
        "了",
        "和",
    )
)

accepted_chars = re.compile(r"[\u4E00-\u9FD5]+")


class _Token(WhooshToken):
    """Custom Token class to satisfy pyright's static analysis."""

    original: str = ""
    text: str = ""
    pos: int = 0
    startchar: int = 0
    endchar: int = 0


class ChineseTokenizer(Tokenizer):
    def __call__(self, text: str) -> Generator[_Token, None, None]:
        words = jieba_fast_dat.tokenize(text, mode="search")
        token = _Token()
        for w, start_pos, stop_pos in words:
            if not accepted_chars.match(w) and len(w) <= 1:
                continue
            token.original = w
            token.text = w
            token.pos = start_pos
            token.startchar = start_pos
            token.endchar = stop_pos
            yield token


def ChineseAnalyzer(
    stoplist: frozenset[str] = STOP_WORDS,
    minsize: int = 1,
    stemfn: Callable[[str], str] = stem,
    cachesize: int = 50000,
) -> Tokenizer:
    return (
        ChineseTokenizer()
        | LowercaseFilter()
        | StopFilter(stoplist=stoplist, minsize=minsize)  # type: ignore
        | StemFilter(stemfn=stemfn, ignore=None, cachesize=cachesize)
    )
