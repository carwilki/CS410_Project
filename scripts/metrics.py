"""nDCG@10, P@10, MRR@10 per mode via Pyserini's trec_eval wrapper + paired significance tests -> eval/RESULTS.md.

  uv run python scripts/metrics.py [--eval-dir eval]

Reads <eval-dir>/qrels.txt and <eval-dir>/runs/{bm25,hyde,hybrid}.trec (and kappa.json if present).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfpb_hyde.config import ROOT, load  # noqa: E402
from cfpb_hyde.eval.stats import MEASURES, paired_tests, trec_eval_per_query  # noqa: E402

MODES = ("bm25", "hyde", "hybrid")
COMPARISONS = (("hybrid", "bm25"), ("hyde", "bm25"))


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def compute(eval_dir: Path, threshold: int) -> dict:
    qrels = eval_dir / "qrels.txt"
    scores = {m: {meas: trec_eval_per_query(qrels, eval_dir / "runs" / f"{m}.trec", meas, threshold)
                  for meas in MEASURES} for m in MODES}
    tests = {f"{a}_vs_{b}": {meas: paired_tests(scores[a][meas], scores[b][meas]) for meas in MEASURES}
             for a, b in COMPARISONS}
    return {"threshold": threshold, "scores": scores, "tests": tests}


def report(res: dict, kappa: dict | None) -> str:
    th, S = res["threshold"], res["scores"]
    names = list(MEASURES.values())
    L = ["# Results", "",
         f"Relevance threshold for binary metrics (P@10, MRR@10): grade >= {th} (trec_eval `-l {th}`). "
         "nDCG@10 uses the graded 0-3 judgments as gains (trec_eval ndcg_cut: \"Gain values are the relevance "
         "values in the qrels file\"; `-l` does not affect it). Runs truncated to 10 (`-M 10`); `-c` set. "
         "Unjudged documents count as non-relevant.", "",
         "## Mean over queries", "", "| mode | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for m in MODES:
        L.append(f"| {m} | " + " | ".join(fmt(S[m][meas]["all"]) for meas in MEASURES) + " |")
    L += ["", "## Paired significance tests (per-query, two-sided)", "",
          "Paired t-test (`scipy.stats.ttest_rel`) and Wilcoxon signed-rank (`scipy.stats.wilcoxon`, "
          "zero_method='wilcox': zero differences dropped). No multiple-comparison correction.", "",
          "| comparison | metric | n | mean diff | t | p (t) | W | p (Wilcoxon) | non-zero diffs |",
          "|---|---|---|---|---|---|---|---|---|"]
    for comp, per in res["tests"].items():
        for meas, t in per.items():
            L.append(f"| {comp.replace('_vs_', ' vs ')} | {MEASURES[meas]} | {t['n']} | {fmt(t['mean_diff'])} | "
                     f"{fmt(t['t'], 3)} | {fmt(t['t_p'])} | {fmt(t['w'], 1)} | {fmt(t['w_p'])} | {t['n_nonzero_diff']} |")
    if kappa:
        L += ["", "## Judge vs human agreement", "",
              f"n = {kappa['n']}; Cohen's kappa unweighted = {kappa['kappa_unweighted']:.4f}; "
              f"quadratic-weighted = {kappa['kappa_quadratic']:.4f}; binary (grade >= {th}) = "
              f"{kappa[f'kappa_binary_ge{th}']:.4f}; exact agreement = {kappa['exact_agreement']:.4f}."]
    for meas, name in MEASURES.items():
        qids = sorted(set().union(*(S[m][meas].keys() for m in MODES)) - {"all"})
        L += ["", f"## Per-query {name}", "", "| qid | " + " | ".join(MODES) + " |", "|---" * (len(MODES) + 1) + "|"]
        for q in qids:
            L.append(f"| {q} | " + " | ".join(fmt(S[m][meas].get(q)) for m in MODES) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "eval")
    a = ap.parse_args()
    res = compute(a.eval_dir, load().eval.relevance_threshold)
    kp = a.eval_dir / "kappa.json"
    kappa = json.loads(kp.read_text()) if kp.exists() else None
    (a.eval_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    (a.eval_dir / "RESULTS.md").write_text(report(res, kappa))
    print((a.eval_dir / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
