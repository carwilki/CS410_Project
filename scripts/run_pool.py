"""Run bm25/hyde/hybrid for every query (one HyDE generation per query, reused), write TREC runs + pool.

  uv run python scripts/run_pool.py QUERIES.tsv [--depth 100] [--out eval]

Resumable: per-query results are cached in <out>/runs/per_query/<qid>.json; rerun skips cached queries.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfpb_hyde.config import ROOT, load  # noqa: E402
from cfpb_hyde.eval.trec import pool, read_queries, write_pool, write_run  # noqa: E402
from cfpb_hyde.search import MODES  # noqa: E402


def assemble(out: Path, qids: list[str], pool_depth: int) -> None:
    """Build <out>/runs/<mode>.trec and <out>/pool.tsv from per-query caches (in query-file order)."""
    runs = {m: {} for m in MODES}
    for qid in qids:
        rec = json.loads((out / "runs" / "per_query" / f"{qid}.json").read_text())
        for m in MODES:
            runs[m][qid] = [(h["docid"], h["rank"], h["score"]) for h in rec["hits"][m]]
    for m in MODES:
        write_run(out / "runs" / f"{m}.trec", runs[m], tag=f"cfpb_{m}")
    write_pool(out / "pool.tsv", pool(runs, pool_depth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", type=Path)
    ap.add_argument("--depth", type=int, default=100, help="run depth written to TREC files")
    ap.add_argument("--out", type=Path, default=ROOT / "eval")
    a = ap.parse_args()
    cfg = load()
    depth = max(a.depth, cfg.eval.pool_depth, 10)
    queries = read_queries(a.queries)
    cache = a.out / "runs" / "per_query"
    cache.mkdir(parents=True, exist_ok=True)

    todo = [(qid, q) for qid, q in queries if not (cache / f"{qid}.json").exists()]
    print(f"{len(queries)} queries, {len(todo)} to run (depth {depth})", flush=True)
    if todo:
        from cfpb_hyde.hyde import hyde
        from cfpb_hyde.search import Retriever
        ret = Retriever(cfg)
        for i, (qid, q) in enumerate(todo, 1):
            vec, record = hyde(q, query_id=qid)  # once per query; logged under hyde_logs/
            hits = {m: [h.__dict__ for h in ret.search(q, m, depth, query_id=qid, hyde_vec=vec, hyde_record=record).hits]
                    for m in MODES}
            tmp = cache / f"{qid}.json.tmp"
            tmp.write_text(json.dumps({"qid": qid, "query": q, "depth": depth, "hits": hits}))
            tmp.rename(cache / f"{qid}.json")
            print(f"[{i}/{len(todo)}] {qid}", flush=True)

    assemble(a.out, [qid for qid, _ in queries], cfg.eval.pool_depth)
    print(f"wrote {a.out}/runs/{{{','.join(MODES)}}}.trec and {a.out}/pool.tsv")


if __name__ == "__main__":
    main()
