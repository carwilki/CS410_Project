"""TREC file I/O: queries TSV, run files, qrels, pool."""
import csv
from pathlib import Path


def read_queries(p: Path) -> list[tuple[str, str]]:
    """qid<TAB>query per line; blank lines and lines starting with '#' skipped."""
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        qid, q = line.split("\t", 1)
        out.append((qid.strip(), q.strip()))
    if len({q for q, _ in out}) != len(out):
        raise ValueError("duplicate qids in queries file")
    return out


def run_lines(qid: str, hits: list[tuple[str, int, float]], tag: str) -> list[str]:
    """hits = [(docid, rank, score)]. Note trec_eval ignores the rank column and re-sorts by score
    (ties broken by docno) — see Anserini trec_eval -h, 'Results_file format'."""
    return [f"{qid} Q0 {d} {r} {s:.6f} {tag}" for d, r, s in hits]


def write_run(p: Path, runs: dict[str, list[tuple[str, int, float]]], tag: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for qid, hits in runs.items():
            for line in run_lines(qid, hits, tag):
                f.write(line + "\n")


def read_run(p: Path) -> dict[str, list[tuple[str, int, float]]]:
    out: dict[str, list] = {}
    for line in Path(p).read_text().splitlines():
        qid, _, d, r, s, _tag = line.split()
        out.setdefault(qid, []).append((d, int(r), float(s)))
    return out


def write_qrels(p: Path, grades: dict[tuple[str, str], int]) -> None:
    """TREC qrels: 'qid 0 docid grade', sorted by qid then docid."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for (qid, d), g in sorted(grades.items()):
            f.write(f"{qid} 0 {d} {int(g)}\n")


def read_qrels(p: Path) -> dict[tuple[str, str], int]:
    out = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            qid, _, d, g = line.split()
            out[(qid, d)] = int(g)
    return out


def pool(runs_by_mode: dict[str, dict[str, list[tuple[str, int, float]]]], depth: int) -> dict[str, dict[str, list[str]]]:
    """Union of top-`depth` docids per mode, deduplicated. Returns {qid: {docid: [modes...]}} (insertion-ordered)."""
    out: dict[str, dict[str, list[str]]] = {}
    for mode, runs in runs_by_mode.items():
        for qid, hits in runs.items():
            q = out.setdefault(qid, {})
            for d, _r, _s in sorted(hits, key=lambda h: h[1])[:depth]:
                q.setdefault(d, []).append(mode)
    return out


def write_pool(p: Path, pooled: dict[str, dict[str, list[str]]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["qid", "docid", "modes"])
        for qid, docs in pooled.items():
            for d, modes in docs.items():
                w.writerow([qid, d, ",".join(modes)])


def read_pool(p: Path) -> list[tuple[str, str]]:
    with open(p, newline="") as f:
        return [(r["qid"], r["docid"]) for r in csv.DictReader(f, delimiter="\t")]


def load_narratives(ids: list[str]) -> dict[str, str]:
    """Narrative text for docids from the corpus side store (reads only the needed rows)."""
    import pandas as pd
    from cfpb_hyde.config import load, path
    df = pd.read_parquet(path(load().paths.corpus_parquet), columns=["id", "narrative"], filters=[("id", "in", ids)])
    out = dict(zip(df["id"], df["narrative"]))
    missing = set(ids) - out.keys()
    if missing:
        raise KeyError(f"{len(missing)} docids not in corpus, e.g. {sorted(missing)[:3]}")
    return out
