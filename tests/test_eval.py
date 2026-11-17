import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from cfpb_hyde.eval import trec, umbrela
from cfpb_hyde.eval.stats import kappas, paired_tests, trec_eval_per_query

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# ---------- TREC I/O and pooling ----------
def test_run_and_qrels_roundtrip(tmp_path):
    runs = {"q1": [("d1", 1, 3.5), ("d2", 2, 1.25)], "q2": [("d3", 1, 0.5)]}
    trec.write_run(tmp_path / "r.trec", runs, "tag")
    assert (tmp_path / "r.trec").read_text().splitlines()[0] == "q1 Q0 d1 1 3.500000 tag"
    assert trec.read_run(tmp_path / "r.trec") == runs
    g = {("q2", "d3"): 0, ("q1", "d2"): 3}
    trec.write_qrels(tmp_path / "qrels", g)
    assert (tmp_path / "qrels").read_text() == "q1 0 d2 3\nq2 0 d3 0\n"
    assert trec.read_qrels(tmp_path / "qrels") == g


def test_read_queries(tmp_path):
    p = tmp_path / "q.tsv"
    p.write_text("# header\nq1\tlate fee on card\n\nq2\tdebt\tcollector\n")
    assert trec.read_queries(p) == [("q1", "late fee on card"), ("q2", "debt\tcollector")]
    p.write_text("q1\ta\nq1\tb\n")
    with pytest.raises(ValueError):
        trec.read_queries(p)


def test_pool_union_dedup_depth(tmp_path):
    runs = {"bm25": {"q": [("a", 1, 9), ("b", 2, 8), ("c", 3, 7)]},
            "hyde": {"q": [("b", 1, 9), ("d", 2, 8), ("e", 3, 7)]},
            "hybrid": {"q": [("a", 1, 9), ("e", 2, 8), ("f", 3, 7)]}}
    p = trec.pool(runs, depth=2)
    assert p == {"q": {"a": ["bm25", "hybrid"], "b": ["bm25", "hyde"], "d": ["hyde"], "e": ["hybrid"]}}
    trec.write_pool(tmp_path / "pool.tsv", p)
    assert trec.read_pool(tmp_path / "pool.tsv") == [("q", "a"), ("q", "b"), ("q", "d"), ("q", "e")]


# ---------- UMBRELA parsing / request ----------
@pytest.mark.parametrize("text,expected", [
    ("##final score: 1", (1, 1)),
    ("M: 2, T: 1\n##final score: 3", (3, 1)),
    ("", (0, 0)),  # UMBRELA: no match -> "0", result_status 0
    ("no digits here", (0, 0)),
    # observed failure mode at max_tokens=100: truncated step list; fallback pattern grabs the last digit ("step 2")
    ("step 1: consider the intent.\n\nstep 2: measure how well the content matches (m).\nthe passage", (2, 1)),
])
def test_parse_fewshot_response(text, expected):
    assert umbrela.parse_fewshot_response(text) == expected


def test_parse_source_appends_reasoning():
    assert umbrela.parse_source("out", None) == "out"
    assert umbrela.parse_source("out", "why") == "out\nwhy"


def test_judge_request_and_strip(monkeypatch):
    sent = {}

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "<|channel>thought\n<channel|>##final score: 2"},
                                 "finish_reason": "stop"}], "usage": {}}

    def fake_post(url, json, timeout, **kw):
        sent.update(url=url, body=json)
        return Resp()

    monkeypatch.setattr(umbrela.requests, "post", fake_post)
    r = umbrela.judge("my query", "my narrative")
    b = sent["body"]
    assert sent["url"].endswith("/chat/completions")
    assert b["model"] == umbrela.load().models.judge
    assert b["reasoning_effort"] == "none" and b["temperature"] == 0.0 and b["frequency_penalty"] == 0.5
    assert b["messages"] == umbrela.messages("my query", "my narrative")
    assert "Query: my query\nPassage: my narrative\n" in b["messages"][1]["content"]
    assert r["grade"] == 2 and r["result_status"] == 1 and r["stripped_reasoning"] and not r["truncated"]


def test_judge_4xx_scored_invalid(monkeypatch):
    class Resp:
        status_code = 400
        text = "context length exceeded"
    monkeypatch.setattr(umbrela.requests, "post", lambda *a, **k: Resp())
    r = umbrela.judge("q", "p")
    assert (r["grade"], r["result_status"]) == (0, 0) and r["error"].startswith("HTTP 400")


# ---------- ordering / sampling ----------
def test_judging_order_and_sample_deterministic():
    from human_sample import sample_pairs
    from judge import judging_order
    pairs = [(f"q{i}", f"d{j}") for i in range(5) for j in range(10)]
    o1 = judging_order(pairs, 410)
    assert o1 == judging_order(list(reversed(pairs)), 410) and sorted(o1) == sorted(pairs)
    assert o1 != judging_order(pairs, 411)
    s = sample_pairs(pairs, 7, 410)
    assert s == sample_pairs(list(reversed(pairs)), 7, 410) and len(set(s)) == 7
    with pytest.raises(ValueError):
        sample_pairs(pairs, 51, 410)


# ---------- kappa ----------
def test_kappa_toy():
    h = [0, 1, 2, 3, 3, 2, 1, 0]
    assert kappas(h, h, 2)["kappa_unweighted"] == pytest.approx(1.0)
    # hand computation: 2x2 binarized (>=2): human [0,0,1,1,1,1,0,0], judge [0,1,1,1,0,1,0,0]
    j = [0, 2, 2, 3, 1, 2, 1, 0]
    k = kappas(h, j, 2)
    po = 6 / 8                          # binary agreement
    pe = (4 / 8) * (4 / 8) + (4 / 8) * (4 / 8)
    assert k["kappa_binary_ge2"] == pytest.approx((po - pe) / (1 - pe))
    po4 = 6 / 8                         # exact agreement on 0-3
    rh, rj = np.bincount(h, minlength=4) / 8, np.bincount(j, minlength=4) / 8
    assert k["kappa_unweighted"] == pytest.approx((po4 - rh @ rj) / (1 - rh @ rj))
    W = np.array([[(a - b) ** 2 for b in range(4)] for a in range(4)])
    O = np.array(k["confusion_human_rows_judge_cols"]) / 8
    assert k["kappa_quadratic"] == pytest.approx(1 - (W * O).sum() / (W * np.outer(rh, rj)).sum())


def test_kappa_reads_filled_tsv(tmp_path):
    from kappa import read_labels
    p = tmp_path / "s.tsv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["qid", "query", "docid", "narrative", "label"])
        w.writerow(["q1", "x", "d1", "line one\nline\ttwo", "2"])
    assert read_labels(p) == {("q1", "d1"): 2}


# ---------- trec_eval (Pyserini wrapper, Anserini jar) ----------
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    (d / "qrels").write_text("q1 0 d1 1\nq1 0 d2 3\nq1 0 d3 2\nq2 0 d4 1\nq2 0 d5 0\nq3 0 r 3\n")
    lines = ["q1 Q0 d1 1 3.0 r", "q1 Q0 d3 2 2.0 r", "q1 Q0 dx 3 1.0 r", "q2 Q0 d5 1 5.0 r", "q2 Q0 d4 2 4.0 r"]
    lines += [f"q3 Q0 n{i} {i} {100 - i}.0 r" for i in range(1, 11)] + ["q3 Q0 r 11 50.0 r"]  # rel doc at rank 11
    (d / "run").write_text("\n".join(lines) + "\n")
    return d


def test_trec_eval_ndcg_graded(toy):
    r = trec_eval_per_query(toy / "qrels", toy / "run", "ndcg_cut.10", 2)
    ideal_q1 = 3 + 2 / math.log2(3) + 1 / math.log2(4)
    assert r["q1"] == pytest.approx((1 + 2 / math.log2(3)) / ideal_q1, abs=1e-4)
    assert r["q2"] == pytest.approx(1 / math.log2(3), abs=1e-4)
    assert r == trec_eval_per_query(toy / "qrels", toy / "run", "ndcg_cut.10", 1)  # -l does not affect nDCG


def test_trec_eval_binary_threshold_and_cutoff(toy):
    p = trec_eval_per_query(toy / "qrels", toy / "run", "P.10", 2)
    assert (p["q1"], p["q2"], p["q3"]) == (0.1, 0.0, 0.0)
    rr = trec_eval_per_query(toy / "qrels", toy / "run", "recip_rank", 2)
    assert (rr["q1"], rr["q2"], rr["q3"]) == (0.5, 0.0, 0.0)  # q3: relevant doc at rank 11 cut by -M 10
    rr1 = trec_eval_per_query(toy / "qrels", toy / "run", "recip_rank", 1)
    assert (rr1["q1"], rr1["q2"]) == (1.0, 0.5)


def test_metrics_report_end_to_end(toy, tmp_path):
    import metrics
    (tmp_path / "runs").mkdir()
    (tmp_path / "qrels.txt").write_text((toy / "qrels").read_text())
    for m in metrics.MODES:
        (tmp_path / "runs" / f"{m}.trec").write_text((toy / "run").read_text())
    res = metrics.compute(tmp_path, 2)
    assert res["tests"]["hybrid_vs_bm25"]["P.10"]["note"] == "all differences zero or n<2"
    md = metrics.report(res, None)
    assert "grade >= 2" in md and "## Per-query nDCG@10" in md and "| q3 |" in md


def test_paired_tests_match_scipy():
    from scipy import stats
    a = {"q1": 0.5, "q2": 0.7, "q3": 0.2, "q4": 0.9, "q5": 0.4, "all": 0}
    b = {"q1": 0.4, "q2": 0.5, "q3": 0.25, "q4": 0.6, "q5": 0.1, "all": 0}
    r = paired_tests(a, b)
    x, y = [a[q] for q in sorted(a) if q != "all"], [b[q] for q in sorted(b) if q != "all"]
    assert r["n"] == 5 and r["t_p"] == pytest.approx(stats.ttest_rel(x, y).pvalue)
    assert r["w_p"] == pytest.approx(stats.wilcoxon(x, y).pvalue)
