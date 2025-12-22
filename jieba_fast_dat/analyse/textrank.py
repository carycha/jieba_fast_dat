from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Hashable
from operator import itemgetter
from typing import Any

import jieba_fast_dat.posseg

from .tfidf import KeywordExtractor


class UndirectWeightedGraph:
    """Undirected weighted graph for TextRank algorithm."""

    d: float = 0.85

    def __init__(self) -> None:
        self.graph: dict[Hashable, list[tuple[Hashable, Hashable, float]]] = (
            defaultdict(list)
        )

    def add_edge(self, start: Hashable, end: Hashable, weight: float) -> None:
        """Add an undirected edge with weight."""
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    addEdge = add_edge  # Legacy alias

    def rank(self) -> dict[Any, float]:
        """Run the PageRank-like ranking algorithm."""
        ws = defaultdict(float)
        out_sum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            out_sum[n] = sum((e[2] for e in out), 0.0)

        # Build stable iteration
        sorted_keys = sorted(self.graph.keys())  # type: ignore[type-var]
        for _x in range(10):  # 10 iters
            for n in sorted_keys:
                s = sum(e[2] / out_sum[e[1]] * ws[e[1]] for e in self.graph[n])
                ws[n] = (1 - self.d) + self.d * s

        if not ws:
            return ws

        min_rank = min(ws.values())
        max_rank = max(ws.values())

        for n, w in ws.items():
            # Normalize weights
            denom = max_rank - min_rank / 10.0
            ws[n] = (w - min_rank / 10.0) / (denom or 1.0)

        return ws


class TextRank(KeywordExtractor):
    """TextRank keyword extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = self.postokenizer = jieba_fast_dat.posseg.dt
        self.pos_filt = frozenset(("ns", "n", "vn", "v"))
        self.span: int = 5

    def pair_filter(self, wp: jieba_fast_dat.posseg.pair) -> bool:
        """Filter words based on POS and stop words."""
        return (
            wp.flag in self.pos_filt
            and len(wp.word.strip()) >= 2
            and wp.word.lower() not in self.stop_words
        )

    pairfilter = pair_filter  # Legacy alias

    def textrank(
        self,
        sentence: str,
        topK: int | None = 20,
        withWeight: bool = False,
        allowPOS: Collection[str] = ("ns", "n", "vn", "v"),
        withFlag: bool = False,
    ) -> list[Any]:
        """
        Extract keywords from sentence using TextRank algorithm.

        Args:
            sentence: Input text.
            topK: Return top K keywords. None for all.
            withWeight: If True, return (word, weight) pairs.
            allowPOS: Filter words by parts of speech.
            withFlag: If True, return pair(word, weight).
        """
        self.pos_filt = frozenset(allowPOS)
        g = UndirectWeightedGraph()
        cm: dict[tuple[Any, Any], int] = defaultdict(int)
        words = tuple(self.tokenizer.cut(sentence))
        for i, wp in enumerate(words):
            if self.pair_filter(wp):
                for j in range(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pair_filter(words[j]):
                        continue
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1

        for terms, w in cm.items():
            g.add_edge(terms[0], terms[1], float(w))
        nodes_rank = g.rank()

        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        if topK:
            return tags[:topK]
        return tags

    extract_tags = textrank
