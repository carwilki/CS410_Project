"""Retrieval modes: bm25, hyde, hybrid. Fusion is Pyserini's own HybridSearcher._hybrid_results
(DECISIONS P4/P5; user decision: call Pyserini, don't copy it)."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyserini.search.faiss import FaissSearcher
from pyserini.search.hybrid import HybridSearcher
from pyserini.search.lucene import LuceneSearcher

from cfpb_hyde.config import load, path
from cfpb_hyde.hyde import Stage, hyde

MODES = ("bm25", "hyde", "hybrid")


@dataclass
class Hit:
    rank: int
    docid: str
    score: float


@dataclass
class Result:
    query: str
    mode: str
    hits: list[Hit]
    hyde_record: dict | None = None
    extra: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, cfg=None):
        self.cfg = cfg or load()
        R = self.cfg.retrieval
        self.bm25 = LuceneSearcher(str(path(self.cfg.paths.bm25_index)))
        self.bm25.set_bm25(R.bm25_k1, R.bm25_b)
        # query_encoder unused: we always pass a precomputed vector (DECISIONS P1/P2)
        self.dense = FaissSearcher(str(path(self.cfg.paths.faiss_index)), query_encoder=object())
        self._meta = None

    @property
    def meta(self) -> pd.DataFrame:
        if self._meta is None:
            self._meta = pd.read_parquet(path(self.cfg.paths.corpus_parquet)).set_index("id")
        return self._meta

    def warm(self, on_stage: Stage = None) -> None:
        """Pay first-query costs up front: metadata parquet + docid hash table (~5 s, holds the GIL),
        first Faiss scan, first Lucene query."""
        say = on_stage or (lambda _: None)
        say("Loading complaint metadata (3.85M narratives)")
        self.meta.index.get_loc(self.meta.index[0])  # builds the docid hash table
        say("Warming dense index")
        self.dense.search(np.zeros((1, self.cfg.embedding.dim), dtype="float32"), 10)
        say("Warming BM25 index")
        self.bm25.search("bank account", 10)

    def search(self, q: str, mode: str, k: int, query_id: str | None = None, k0: int | None = None,
               hyde_vec: np.ndarray | None = None, hyde_record: dict | None = None,
               on_stage: Stage = None) -> Result:
        """`hyde_vec`/`hyde_record` let callers reuse one HyDE generation across modes.
        `on_stage(msg)` is called as each processing stage starts."""
        say = on_stage or (lambda _: None)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        R = self.cfg.retrieval
        k0 = k0 or max(R.hybrid_k0, k)
        if mode == "bm25":
            say("Searching BM25 index")
            return Result(q, mode, _hits([(h.docid, h.score) for h in self.bm25.search(q, k)]))
        if hyde_vec is None:
            hyde_vec, hyde_record = hyde(q, query_id=query_id, on_stage=on_stage)
        say(f"Searching dense index ({self.dense.index.ntotal:,} vectors)")
        if mode == "hyde":
            return Result(q, mode, _hits([(h.docid, h.score) for h in self.dense.search(hyde_vec, k)]), hyde_record)
        dense = self.dense.search(hyde_vec, k0)
        say(f"Searching BM25 index (top {k0})")
        sparse = self.bm25.search(q, k0)
        say("Fusing BM25 + dense scores")
        fused = HybridSearcher._hybrid_results(
            dense, sparse,
            R.fusion_alpha, k, R.fusion_normalization, R.fusion_weight_on_dense)
        return Result(q, mode, _hits([(h.docid, h.score) for h in fused]), hyde_record)


def _hits(pairs) -> list[Hit]:
    return [Hit(i + 1, d, float(s)) for i, (d, s) in enumerate(pairs)]
