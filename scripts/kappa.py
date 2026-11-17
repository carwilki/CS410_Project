"""Cohen's kappa between human labels (filled eval/human_sample.tsv) and judge grades (eval/qrels.txt).

  uv run python scripts/kappa.py [--sample eval/human_sample.tsv] [--qrels eval/qrels.txt]

Writes eval/kappa.json. Reports unweighted kappa, quadratic-weighted kappa (both over grades 0-3) and
unweighted kappa on grades binarized at eval.relevance_threshold.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfpb_hyde.config import ROOT, load  # noqa: E402
from cfpb_hyde.eval.stats import kappas  # noqa: E402
from cfpb_hyde.eval.trec import read_qrels  # noqa: E402


def read_labels(p: Path) -> dict[tuple[str, str], int]:
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lab = (r["label"] or "").strip()
            if lab not in {"0", "1", "2", "3"}:
                raise ValueError(f"row qid={r['qid']} docid={r['docid']}: label must be 0-3, got {lab!r}")
            out[(r["qid"], r["docid"])] = int(lab)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=ROOT / "eval" / "human_sample.tsv")
    ap.add_argument("--qrels", type=Path, default=ROOT / "eval" / "qrels.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "kappa.json")
    a = ap.parse_args()
    labels, qrels = read_labels(a.sample), read_qrels(a.qrels)
    keys = sorted(labels)
    missing = [k for k in keys if k not in qrels]
    if missing:
        raise KeyError(f"{len(missing)} sampled pairs not in qrels, e.g. {missing[:3]}")
    res = kappas([labels[k] for k in keys], [qrels[k] for k in keys], load().eval.relevance_threshold)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
