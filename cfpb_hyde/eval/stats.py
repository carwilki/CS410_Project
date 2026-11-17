"""Agreement (Cohen's kappa via sklearn), trec_eval metrics (Pyserini wrapper), paired significance tests (scipy)."""
import contextlib
import io
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

GRADES = [0, 1, 2, 3]
# trec_eval measure -> output key. recip_rank has no cutoff parameter, so every run is truncated with -M 10
# (Anserini trec_eval 9.0.4 -h: "-M <num>: Max number of docs per topic to use in evaluation").
MEASURES = {"ndcg_cut.10": "nDCG@10", "P.10": "P@10", "recip_rank": "MRR@10"}


def kappas(human: list[int], judge: list[int], threshold: int) -> dict:
    """Unweighted and quadratic-weighted kappa over grades 0-3, plus unweighted kappa after binarizing at
    grade >= threshold (UMBRELA's calculate_kappa binarizes at >1, common_utils.py:175-180)."""
    h, j = np.asarray(human), np.asarray(judge)
    return {
        "n": int(len(h)),
        "kappa_unweighted": float(cohen_kappa_score(h, j, labels=GRADES)),
        "kappa_quadratic": float(cohen_kappa_score(h, j, labels=GRADES, weights="quadratic")),
        f"kappa_binary_ge{threshold}": float(cohen_kappa_score(h >= threshold, j >= threshold, labels=[False, True])),
        "exact_agreement": float(np.mean(h == j)),
        "confusion_human_rows_judge_cols": [[int(np.sum((h == a) & (j == b))) for b in GRADES] for a in GRADES],
    }


def trec_eval_per_query(qrels: Path, run: Path, measure: str, threshold: int) -> dict[str, float]:
    """One measure per call: the wrapper's returned dict keeps only the last metric per qid
    (pyserini/eval/trec_eval.py:186-200). Returns {qid: value, ..., 'all': mean}."""
    from pyserini.eval.trec_eval import trec_eval
    args = ["trec_eval", "-c", "-q", "-M", "10", "-l", str(threshold), "-m", measure, str(qrels), str(run)]
    with contextlib.redirect_stdout(io.StringIO()):  # wrapper prints trec_eval stdout (trec_eval.py:162-163)
        return trec_eval(args, return_per_query_results=True)


def paired_tests(a: dict[str, float], b: dict[str, float]) -> dict:
    """Two-sided paired t-test (scipy.stats.ttest_rel) and Wilcoxon signed-rank (scipy.stats.wilcoxon,
    zero_method='wilcox' default: zero differences dropped) on per-query scores, a vs b."""
    qids = sorted((a.keys() & b.keys()) - {"all"})
    x, y = np.array([a[q] for q in qids]), np.array([b[q] for q in qids])
    d = x - y
    out = {"n": len(qids), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "mean_diff": float(d.mean()),
           "n_nonzero_diff": int(np.count_nonzero(d))}
    if len(qids) < 2 or not np.any(d):
        out.update(t=None, t_p=None, w=None, w_p=None, note="all differences zero or n<2")
        return out
    t = stats.ttest_rel(x, y)
    w = stats.wilcoxon(x, y)
    out.update(t=float(t.statistic), t_p=float(t.pvalue), w=float(w.statistic), w_p=float(w.pvalue))
    return out
