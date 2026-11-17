"""Phase 1c: embed every narrative ("search_document: " prefix) via the configured provider, resumably.

Work is split into chunks; each finished chunk is saved atomically as chunk_XXXXX.npy, so a rerun
skips completed chunks. Failed docs are logged to failures.jsonl and left out of the index;
docs over the embedder context are logged to truncated.tsv. `--assemble` writes the Pyserini
Faiss layout (<dir>/index + <dir>/docid, see DECISIONS P3).

A manifest (corpus id hash, chunk size, embedder, prefix, dim) is written to the work dir on first run;
a later run against a different corpus or settings stops instead of mixing stale chunks.

Usage: python -m scripts.embed_corpus [--limit N] [--workers W] [--instances a,b] [--assemble] [--check-manifest]
"""
import argparse
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import pandas as pd

from cfpb_hyde.config import load, path
from cfpb_hyde.embed import doc_text
from cfpb_hyde.provider import embed

cfg = load()
E = cfg.embedding
CHUNK = 64 * E.batch_size  # docs per checkpoint file
work = path(cfg.paths.embed_work)
work.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(work / "progress.log"), logging.StreamHandler()])
log = logging.getLogger("embed")


def embed_batch(texts, ids, model):
    """Embed a batch; on failure fall back to one-by-one and report failed ids."""
    try:
        return embed(texts, model), []
    except Exception as e:
        log.warning("batch failed (%s); retrying singly", e)
    vecs, failed = np.full((len(texts), E.dim), np.nan, dtype=np.float32), []
    for i, (t, d) in enumerate(zip(texts, ids)):
        try:
            vecs[i] = embed([t], model)[0]
        except Exception as e:
            failed.append({"id": d, "error": str(e)[:500]})
    return vecs, failed


INSTANCES = list(E.index_instances)


def check_manifest(ids) -> None:
    """Write the manifest on first use; exit if existing chunks were built from something else."""
    m = {"n_docs": len(ids), "ids_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest(), "chunk": CHUNK,
         "embedder": cfg.models.embedder, "document_prefix": E.document_prefix, "dim": E.dim}
    f = work / "manifest.json"
    if f.exists():
        old = json.loads(f.read_text())
        if old != m:
            diff = {k: (old.get(k), v) for k, v in m.items() if old.get(k) != v}
            raise SystemExit(f"{work} holds embeddings from a different corpus or embedder (changed: "
                             f"{', '.join(diff)}). They cannot be reused; delete that directory (rm -rf "
                             f"{work.relative_to(path('.'))}) and rerun to re-embed.")
    else:
        f.write_text(json.dumps(m, indent=1))


def run_chunk(c, ids, texts):
    model = INSTANCES[c % len(INSTANCES)]
    out = work / f"chunk_{c:05d}.npy"
    if out.exists():
        return 0
    vecs, failed = [], []
    for i in range(0, len(texts), E.batch_size):
        v, f = embed_batch(texts[i:i + E.batch_size], ids[i:i + E.batch_size], model)
        vecs.append(v)
        failed += f
    vecs = np.concatenate(vecs)
    assert vecs.shape == (len(ids), E.dim), vecs.shape
    if failed:
        with open(work / "failures.jsonl", "a") as fh:
            for f in failed:
                fh.write(json.dumps({"chunk": c, **f}) + "\n")
    tmp = work / f"chunk_{c:05d}.tmp.npy"
    np.save(tmp, vecs)
    tmp.rename(out)
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N docs (testing)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--instances", help="comma-separated provider model ids (default: config index_instances)")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--check-manifest", action="store_true", help="verify/write the manifest and exit")
    a = ap.parse_args()

    global work, INSTANCES
    if a.instances:
        INSTANCES = a.instances.split(",")
    df = pd.read_parquet(path(cfg.paths.corpus_parquet), columns=["id", "narrative", "n_tokens"])
    if a.limit:  # keep test chunks out of the real checkpoint dir
        df = df.iloc[:a.limit]
        work = work / f"test_limit{a.limit}_w{a.workers}"
        work.mkdir(parents=True, exist_ok=True)
    ids, texts = df["id"].tolist(), [doc_text(t) for t in df["narrative"]]
    n_chunks = (len(ids) + CHUNK - 1) // CHUNK
    check_manifest(ids)
    if a.check_manifest:
        return log.info("manifest ok: %d docs, %d chunks", len(ids), n_chunks)

    if a.assemble:
        return assemble(ids, n_chunks, work / "faiss" if a.limit else path(cfg.paths.faiss_index))

    over = df[df["n_tokens"] > E.context_tokens]
    over[["id", "n_tokens"]].to_csv(work / "truncated.tsv", sep="\t", index=False)
    log.info("docs=%d chunks=%d truncated(>%d tok)=%d workers=%d instances=%s", len(ids), n_chunks,
             E.context_tokens, len(over), a.workers, ",".join(INSTANCES))

    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_chunk, c, ids[c * CHUNK:(c + 1) * CHUNK], texts[c * CHUNK:(c + 1) * CHUNK])
                for c in range(n_chunks)]
        for k, f in enumerate(futs):
            done += f.result()
            el = time.time() - t0
            rate = done / el if el and done else 0
            log.info("chunk %d/%d done | new docs %d | %.1f docs/s | eta %.2f h", k + 1, n_chunks, done, rate,
                     max(0, len(ids) - (k + 1) * CHUNK) / rate / 3600 if rate else float("nan"))


def assemble(ids, n_chunks, out):
    out.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(E.dim)
    kept, dropped = [], 0
    for c in range(n_chunks):
        v = np.load(work / f"chunk_{c:05d}.npy")
        cid = ids[c * CHUNK:(c + 1) * CHUNK]
        ok = ~np.isnan(v).any(axis=1)
        dropped += int((~ok).sum())
        index.add(np.ascontiguousarray(v[ok]))
        kept += [d for d, k in zip(cid, ok) if k]
    faiss.write_index(index, str(out / "index"))
    (out / "docid").write_text("\n".join(kept) + "\n")
    log.info("assembled %d vectors (%d dropped as failed) -> %s", index.ntotal, dropped, out)


if __name__ == "__main__":
    main()
