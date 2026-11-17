"""UMBRELA-judge every pooled (qid, docid) with the LM Studio judge model; write TREC qrels.

  uv run python scripts/judge.py QUERIES.tsv [--pool eval/pool.tsv] [--out eval] [--limit N]
  uv run python scripts/judge.py --smoke          # one call; checks reasoning is off for the judge

Order: all pool pairs shuffled with random.Random(eval.seed). Resumable: <out>/judgments.jsonl is appended per
judgment and cached pairs are skipped. The judge sees only the query text and the narrative.
"""
import argparse
import json
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfpb_hyde.config import ROOT, load  # noqa: E402
from cfpb_hyde.eval import umbrela  # noqa: E402
from cfpb_hyde.eval.trec import load_narratives, read_pool, read_queries, write_qrels  # noqa: E402


def load_judgments(p: Path) -> dict[tuple[str, str], dict]:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["qid"], r["docid"])] = r
    return out


def judging_order(pairs: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    order = sorted(set(pairs))
    random.Random(seed).shuffle(order)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", type=Path, nargs="?")
    ap.add_argument("--pool", type=Path, default=ROOT / "eval" / "pool.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "eval")
    ap.add_argument("--limit", type=int, default=None, help="judge at most N new pairs this run")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="concurrent judge requests (LM Studio parallel slots)")
    a = ap.parse_args()
    cfg = load()

    if a.smoke:
        r = umbrela.judge("debt collector keeps calling about a debt I already paid",
                          "I paid off my medical bill in full in March. A collection agency still calls me "
                          "several times a week saying I owe the balance.", cfg)
        print(json.dumps(r, indent=1))
        ok = not r["reasoning_content"] and not r["stripped_reasoning"]
        print("reasoning OFF" if ok else "REASONING PRESENT")
        return

    queries = dict(read_queries(a.queries))
    pairs = judging_order(read_pool(a.pool), cfg.eval.seed)
    jpath = a.out / "judgments.jsonl"
    done = load_judgments(jpath)
    todo = [p for p in pairs if p not in done][: a.limit]
    print(f"{len(pairs)} pool pairs, {len(done)} cached, judging {len(todo)}", flush=True)

    texts = load_narratives(sorted({d for _, d in todo})) if todo else {}
    lock, n = threading.Lock(), [0]

    def one(pair):
        qid, d = pair
        r = umbrela.judge(queries[qid], texts[d], cfg)
        r.update(qid=qid, docid=d, timestamp=datetime.now(timezone.utc).isoformat())
        with lock:
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[(qid, d)] = r
            n[0] += 1
            print(f"[{n[0]}/{len(todo)}] {qid} {d} -> {r['grade']}{'' if r['result_status'] else ' (INVALID)'}"
                  f"{' (TRUNCATED)' if r['truncated'] else ''}", flush=True)

    # Pairs are submitted in the shuffled order; with workers > 1 completion order may interleave.
    with open(jpath, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))

    judged = {k: v for k, v in done.items() if k in set(pairs)}
    write_qrels(a.out / "qrels.txt", {k: v["grade"] for k, v in judged.items()})
    rs = list(judged.values())
    print(f"qrels: {len(judged)}/{len(pairs)} pairs -> {a.out}/qrels.txt")
    print("grades:", dict(sorted(Counter(r["grade"] for r in rs).items())),
          "| invalid:", sum(not r["result_status"] for r in rs),
          "| truncated:", sum(bool(r.get("truncated")) for r in rs),
          "| errors:", sum(bool(r["error"]) for r in rs),
          "| reasoning present:", sum(bool(r["reasoning_content"] or r["stripped_reasoning"]) for r in rs))


if __name__ == "__main__":
    main()
