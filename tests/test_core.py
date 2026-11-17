import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cfpb_hyde import hyde as hyde_mod
from cfpb_hyde.embed import doc_text, hyde_vector, query_text
from cfpb_hyde.reasoning import strip_reasoning


def test_eq3_average():
    q = np.array([1.0, 0.0, 0.0])
    d = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 0, 0], [0, 0, 0]], dtype=float)
    v = hyde_vector(q, d)
    assert v.shape == (1, 3)
    np.testing.assert_allclose(v[0], [(1 + 0 + 0 + 1 + 2 + 0) / 6, (0 + 1 + 0 + 1 + 0 + 0) / 6, (0 + 0 + 1 + 1 + 0 + 0) / 6])


def test_eq3_no_extra_normalization():
    q = np.ones(4) / 2  # unit norm
    v = hyde_vector(q, np.stack([-q] * 5))
    np.testing.assert_allclose(v[0], -4 / 6 * q)  # (q - 5q)/6, not re-normalized


def test_prefixes():
    assert query_text("late fee") == "search_query: late fee"
    assert doc_text("I was charged") == "search_document: I was charged"


@pytest.mark.parametrize("raw,clean,removed", [
    ("Plain answer.", "Plain answer.", ""),
    ("<|channel>thought\nlet me think<channel|>Final text", "Final text", "<|channel>thought\nlet me think<channel|>"),
    ("<|channel>thought\n<channel|>Final", "Final", "<|channel>thought\n<channel|>"),
    ("Start <|channel>thought\ncut off by max tok", "Start", "<|channel>thought\ncut off by max tok"),
])
def test_strip_reasoning(raw, clean, removed):
    assert strip_reasoning(raw) == (clean, removed)


def test_fusion_hand_computed():
    from pyserini.search.hybrid import HybridSearcher
    dense = [NS(docid="a", score=0.9), NS(docid="b", score=0.5)]
    sparse = [NS(docid="b", score=10.0), NS(docid="c", score=4.0)]
    # alpha*sparse + dense; missing scores filled with list minimum
    # a: 0.1*4 + 0.9 = 1.3 ; b: 0.1*10 + 0.5 = 1.5 ; c: 0.1*4 + 0.5 = 0.9
    out = HybridSearcher._hybrid_results(dense, sparse, 0.1, 10, False, False)
    assert [(h.docid, round(h.score, 6)) for h in out] == [("b", 1.5), ("a", 1.3), ("c", 0.9)]
    # normalization: s' = (s - (min+max)/2) / (max-min)
    # dense: a=0.5, b=-0.5 ; sparse: b=0.5, c=-0.5 (a gets min sparse 4 -> -0.5; c gets min dense 0.5 -> -0.5)
    # a: 0.1*-0.5 + 0.5 = 0.45 ; b: 0.1*0.5 - 0.5 = -0.45 ; c: 0.1*0.5 - 0.5 = -0.45
    out = HybridSearcher._hybrid_results(dense, sparse, 0.1, 1, True, False)
    assert out[0].docid == "a" and round(out[0].score, 6) == 0.45


def test_hyde_log_writing(tmp_path, monkeypatch):
    fake = {"choices": [{"message": {"content": "<|channel>thought\nx<channel|>My bank charged me.",
                                     "reasoning_content": ""}, "finish_reason": "stop"}]}
    monkeypatch.setattr(hyde_mod, "chat", lambda msgs, **kw: fake)
    monkeypatch.setattr(hyde_mod, "embed", lambda texts: np.eye(6, 768, dtype=np.float32)[: len(texts)])
    v, rec = hyde_mod.hyde("overdraft fees", query_id="q1", log_dir=tmp_path)
    p = tmp_path / hyde_mod.RUN_ID / "q1.json"
    saved = json.loads(p.read_text())
    assert saved == rec
    for key in ("query", "timestamp", "generator_model", "system_prompt", "generation_params",
                "generations", "documents", "embedding_model", "embedding_dim"):
        assert key in saved
    assert saved["documents"] == ["My bank charged me."] * 5
    assert saved["generations"][0]["stripped_reasoning"] == "<|channel>thought\nx<channel|>"
    assert saved["embedding_dim"] == 768 and v.shape == (1, 768)
    np.testing.assert_allclose(v[0, :6], np.full(6, 1 / 6))
