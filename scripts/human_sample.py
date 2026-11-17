"""Export a seeded random sample of judged (qid, docid) pairs for human labelling (judge grade NOT included).

  uv run python scripts/human_sample.py QUERIES.tsv [--qrels eval/qrels.txt] [--out eval/human_sample.tsv]

Output is a tab-separated file written with Python's csv module (fields containing tabs/newlines/quotes are
double-quoted). Fill the `label` column with 0-3 using the same UMBRELA scale the judge used.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfpb_hyde.config import ROOT, load  # noqa: E402
from cfpb_hyde.eval.trec import load_narratives, read_qrels, read_queries  # noqa: E402


def sample_pairs(pairs, n: int, seed: int) -> list[tuple[str, str]]:
    pairs = sorted(pairs)
    if n > len(pairs):
        raise ValueError(f"sample size {n} > {len(pairs)} judged pairs")
    return random.Random(seed).sample(pairs, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", type=Path)
    ap.add_argument("--qrels", type=Path, default=ROOT / "eval" / "qrels.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "human_sample.tsv")
    a = ap.parse_args()
    cfg = load()
    queries = dict(read_queries(a.queries))
    sample = sample_pairs(read_qrels(a.qrels).keys(), cfg.eval.human_sample_size, cfg.eval.seed)
    texts = load_narratives(sorted({d for _, d in sample}))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["qid", "query", "docid", "narrative", "label"])
        for qid, d in sample:
            w.writerow([qid, queries[qid], d, texts[d], ""])
    print(f"wrote {len(sample)} rows -> {a.out}")


if __name__ == "__main__":
    main()
