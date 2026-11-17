"""Bootstrap a fresh clone: data/*.zip + code -> corpus, BM25 index, embeddings, Faiss index.

    uv run python main.py bootstrap           # build whatever is missing (resumable)
    uv run python main.py bootstrap --check   # report only

Each step records a marker in artifacts/bootstrap_state.json only after its outputs are verified.
Markers are chained: the corpus marker stores a fingerprint of data/*.zip and a hash of the corpus
ids, and every later step stores the corpus hash it was built from. Changing the data therefore
invalidates everything downstream. Outputs that already exist but have no marker (for example a
build that predates this module) are verified and adopted instead of rebuilt.
Progress lines start with "[k/N]"; the TUI's :bootstrap command shows them in its status bar.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from cfpb_hyde import provider
from cfpb_hyde.config import ROOT, load, path

cfg = load()
STATE = path("artifacts/bootstrap_state.json")
STEPS = ["corpus", "bm25", "embed", "faiss"]
TOTAL = len(STEPS) + 1  # + preflight
JAVA_MAJOR = 21  # Pyserini 2.4.0 (DECISIONS P0)


class BootstrapError(RuntimeError):
    pass


def say(k: int, msg: str):
    print(f"[{k}/{TOTAL}] {msg}", flush=True)


# --- state ---
def read_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def write_state(state: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


def data_fingerprint() -> list:
    return [[z.name, z.stat().st_size] for z in sorted(path(cfg.paths.data_dir).glob("*.zip"))]


def step_status(state: dict | None = None) -> dict[str, bool]:
    """Which steps have a valid marker (cheap: reads the state file and lists data/)."""
    s = state if state is not None else read_state()
    c = s.get("corpus")
    ok = {"corpus": bool(c) and c.get("data") == data_fingerprint()}
    sha = c.get("corpus_sha") if c else None
    for step in STEPS[1:]:
        ok[step] = ok["corpus"] and s.get(step, {}).get("corpus_sha") == sha
    return ok


def missing_steps() -> list[str]:
    return [k for k, v in step_status().items() if not v]


def mark(state: dict, step: str, **info):
    state[step] = {**info, "ts": datetime.now(timezone.utc).isoformat()}
    if step != "corpus":
        state[step]["corpus_sha"] = state["corpus"]["corpus_sha"]
    write_state(state)


# --- helpers ---
def run(k: int, label: str, cmd: list[str], progress=None):
    """Run a build command, streaming its output; `progress(line)` may turn a line into a status message."""
    say(k, label)
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    last = ""
    for line in p.stdout:
        line = line.rstrip()
        last = line or last
        msg = progress(line) if progress else None
        say(k, msg) if msg else print("    " + line, flush=True)
    if p.wait():
        raise BootstrapError(f"{label.split(':')[0]} step failed (exit {p.returncode}): {last}")


def corpus_ids() -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_table(path(cfg.paths.corpus_parquet), columns=["id"]).column("id").to_pylist()


def count_lines(files) -> int:
    n = 0
    for f in files:
        with open(f, "rb") as fh:
            while b := fh.read(1 << 26):
                n += b.count(b"\n")
    return n


def lms_loaded() -> set[str] | None:
    """Identifiers of models loaded in LM Studio, or None if not LM Studio / the `lms` CLI is unavailable."""
    if cfg.provider.name != "lmstudio" or not shutil.which("lms"):
        return None
    out = subprocess.run(["lms", "ps", "--json"], capture_output=True, text=True)
    return {m["identifier"] for m in json.loads(out.stdout)} if out.returncode == 0 else None


# --- steps ---
def preflight(k: int, need_build: bool) -> list[str]:
    """Check data, Java, provider and models. Returns warnings; raises BootstrapError on blockers."""
    say(k, f"Preflight: data, Java, provider ({cfg.provider.name} at {cfg.provider.base_url}), models")
    errors, warnings = [], []
    zips = data_fingerprint()
    if not zips and need_build:
        errors.append(f"no CFPB zip files in {path(cfg.paths.data_dir)} (expected CCDB_Export_*.zip)")
    print(f"    data: {len(zips)} zip files", flush=True)
    try:
        v = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
        found = re.search(r'version "(\d+)', v)
        if not found:
            raise AttributeError
        major = int(found.group(1))
        print(f"    java: {v.splitlines()[0]}", flush=True)
        if major < JAVA_MAJOR:
            errors.append(f"Java {major} found; Pyserini needs JDK {JAVA_MAJOR}+")
    except (FileNotFoundError, AttributeError):
        errors.append(f"Java not found; install JDK {JAVA_MAJOR} (Pyserini requirement)")
    try:
        ids = provider.list_models()
    except Exception as e:
        raise BootstrapError("; ".join(errors + [f"provider {cfg.provider.name!r} not reachable at "
                                                 f"{cfg.provider.base_url} ({e}). Start the server or fix "
                                                 "provider.base_url in config.yaml."]))
    M = cfg.models
    for role, mid in (("embedder", M.embedder), ("generator", M.generator), ("judge", M.judge)):
        if mid not in ids:
            msg = (f"model {mid!r} ({role}) is not listed by the provider at {cfg.provider.base_url}/models; "
                   "download/serve it or change models.{role} in config.yaml").replace("{role}", role)
            (errors if role == "embedder" and need_build else warnings).append(msg)
        else:
            print(f"    {role}: {mid} available", flush=True)
    mismatch = index_embedder_mismatch()
    if mismatch:
        warnings.append(mismatch)
    if errors:
        raise BootstrapError("; ".join(errors))
    return warnings


def index_embedder_mismatch() -> str | None:
    """Warn if the configured embedder is not the one the existing embeddings were built with."""
    f = path(cfg.paths.embed_work) / "manifest.json"
    if not f.exists():
        return None
    built = json.loads(f.read_text()).get("embedder")
    if built != cfg.models.embedder:
        return (f"index was embedded with {built!r} but models.embedder is {cfg.models.embedder!r}: query "
                f"vectors only match the index if this is the same model with the same weights. Otherwise "
                f"delete {cfg.paths.embed_work} and {cfg.paths.faiss_index} and rerun bootstrap.")
    return None


def build_corpus(k: int, state: dict):
    n_zip = len(data_fingerprint())
    seen = [0]

    def prog(line):
        if line.startswith("CCDB_Export"):
            seen[0] += 1
            return f"Corpus: read {line.split()[0]} ({seen[0]}/{n_zip} zips)"
        return None

    if not (legacy_corpus(state) and adopt_corpus(k, state)):
        run(k, "Corpus: reading zips, tokenizing narratives, writing parquet + JsonCollection",
            [sys.executable, "-m", "scripts.build_corpus"], prog)
        if not adopt_corpus(k, state):
            raise BootstrapError("corpus outputs failed verification after build")


def legacy_corpus(state: dict) -> bool:
    """Existing corpus outputs may be adopted only when no marker exists (pre-bootstrap build) and they are
    newer than every data zip. A marker with a different data fingerprint means the data changed: rebuild."""
    pq_path = path(cfg.paths.corpus_parquet)
    zips = list(path(cfg.paths.data_dir).glob("*.zip"))
    return ("corpus" not in state and pq_path.exists() and bool(zips)
            and pq_path.stat().st_mtime > max(z.stat().st_mtime for z in zips))


def adoptable(state: dict, step: str) -> bool:
    """Outputs of `step` may be reused only if they predate markers or were built from the current corpus."""
    return step not in state or state[step].get("corpus_sha") == state["corpus"]["corpus_sha"]


def adopt_corpus(k: int, state: dict) -> bool:
    pq_path, jc = path(cfg.paths.corpus_parquet), path(cfg.paths.json_collection)
    files = sorted(jc.glob("docs*.jsonl")) if jc.exists() else []
    if not pq_path.exists() or not files:
        return False
    say(k, "Corpus: verifying parquet vs JsonCollection")
    ids = corpus_ids()
    n_json = count_lines(files)
    if n_json != len(ids) or len(set(ids)) != len(ids):
        print(f"    mismatch: parquet {len(ids)} rows, JsonCollection {n_json} lines", flush=True)
        return False
    sha = hashlib.sha1("\n".join(ids).encode()).hexdigest()
    mark(state, "corpus", n_docs=len(ids), corpus_sha=sha, data=data_fingerprint())
    say(k, f"Corpus: {len(ids):,} narratives verified")
    return True


def bm25_docs() -> int | None:
    idx = path(cfg.paths.bm25_index)
    if not idx.exists() or not any(idx.glob("segments_*")):
        return None
    from pyserini.search.lucene import LuceneSearcher
    return LuceneSearcher(str(idx)).num_docs


def build_bm25(k: int, state: dict):
    n = state["corpus"]["n_docs"]
    if not adoptable(state, "bm25") or bm25_docs() != n:
        idx = path(cfg.paths.bm25_index)
        if idx.exists():
            shutil.rmtree(idx)  # partial or stale index: Lucene would append to it
        run(k, f"BM25: indexing {n:,} narratives with Lucene ({os.cpu_count()} threads)",
            [sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
             "--input", str(path(cfg.paths.json_collection)), "--index", str(idx),
             "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(os.cpu_count())])
        got = bm25_docs()
        if got != n:
            raise BootstrapError(f"BM25 index has {got} docs, expected {n:,}")
    mark(state, "bm25", n_docs=n)
    say(k, f"BM25: {n:,} documents verified")


def embed_instances(k: int) -> list[str]:
    """Extra identical embedder instances from config (throughput, DECISIONS I1-I3). LM Studio: load missing
    ones with `lms`. Other providers: keep only ids the server lists at /models."""
    E, base = cfg.embedding, cfg.models.embedder
    loaded = lms_loaded()
    if loaded is None:
        served = provider.list_models()
        use = [i for i in E.index_instances if i == base or i in served] or [base]
        print(f"    embedding with {len(use)} instance(s): {', '.join(use)}", flush=True)
        return use
    use = []
    for ident in E.index_instances:
        if ident == base or ident in loaded:
            use.append(ident)
            continue
        say(k, f"Embeddings: loading embedder instance {ident} with lms")
        r = subprocess.run(["lms", "load", base, "--identifier", ident, "-c", str(E.context_tokens),
                            "--gpu", "max", "-y"], capture_output=True, text=True)
        if r.returncode == 0:
            use.append(ident)
        else:
            print(f"    could not load {ident}: {(r.stderr or r.stdout).strip()[:300]}", flush=True)
    return use or [base]


def embed_progress(line: str) -> str | None:
    m = re.search(r"chunk (\d+)/(\d+) done \| new docs (\d+) \| ([\d.]+) docs/s \| eta (-?[\d.]+|nan) h", line)
    if m:
        c, n, _, rate, eta = m.groups()
        return f"Embeddings: chunk {c}/{n} ({100 * int(c) / int(n):.0f}%), {float(rate):.0f} docs/s, eta {eta} h"
    return None


def build_embed(k: int, state: dict):
    n_chunks = embed_chunks(state)
    work = path(cfg.paths.embed_work)
    # refuse chunks embedded from a different corpus/embedder before counting them as done
    run(k, "Embeddings: checking existing chunks belong to this corpus",
        [sys.executable, "-m", "scripts.embed_corpus", "--check-manifest"])
    done = len(list(work.glob("chunk_[0-9][0-9][0-9][0-9][0-9].npy"))) if work.exists() else 0
    if done < n_chunks:
        inst = embed_instances(k)
        run(k, f"Embeddings: {done}/{n_chunks} chunks already done; embedding the rest with {len(inst)} "
               f"instance(s), resumable (stop any time, rerun to continue)",
            [sys.executable, "-m", "scripts.embed_corpus", "--workers", str(len(inst)),
             "--instances", ",".join(inst)], embed_progress)
    missing = [c for c in range(n_chunks) if not (work / f"chunk_{c:05d}.npy").exists()]
    if missing:
        raise BootstrapError(f"{len(missing)} embedding chunks missing (first: {missing[0]}); rerun to resume")
    mark(state, "embed", n_chunks=n_chunks)
    say(k, f"Embeddings: {n_chunks} chunks complete")


def embed_chunks(state: dict) -> int:
    chunk = 64 * cfg.embedding.batch_size  # must match scripts/embed_corpus.py CHUNK
    return (state["corpus"]["n_docs"] + chunk - 1) // chunk


def failed_ids() -> int:
    f = path(cfg.paths.embed_work) / "failures.jsonl"
    return len({json.loads(line)["id"] for line in f.read_text().splitlines()}) if f.exists() else 0


def faiss_ok(expected: int) -> bool:
    out = path(cfg.paths.faiss_index)
    idx, docid = out / "index", out / "docid"
    if not idx.exists() or not docid.exists():
        return False
    n = count_lines([docid])
    # IndexFlatIP file = small header + n * dim float32 (observed: 45-byte header on the full build)
    body = n * cfg.embedding.dim * 4
    return n == expected and 0 <= idx.stat().st_size - body < 4096


def build_faiss(k: int, state: dict):
    expected = state["corpus"]["n_docs"] - failed_ids()
    if not adoptable(state, "faiss") or not faiss_ok(expected):
        run(k, f"Faiss: assembling IndexFlatIP from chunks ({expected:,} vectors, ~"
               f"{expected * cfg.embedding.dim * 4 / 2**30:.1f} GiB)",
            [sys.executable, "-m", "scripts.embed_corpus", "--assemble"])
        if not faiss_ok(expected):
            raise BootstrapError("Faiss index failed verification after assembly")
    mark(state, "faiss", n_vectors=expected)
    say(k, f"Faiss: {expected:,} vectors verified")


BUILDERS = {"corpus": build_corpus, "bm25": build_bm25, "embed": build_embed, "faiss": build_faiss}


def bootstrap(check_only: bool = False) -> int:
    t0 = time.time()
    state = read_state()
    status = step_status(state)
    todo = [s for s in STEPS if not status[s]]
    try:
        for w in preflight(1, bool(todo)):
            print(f"    warning: {w}", flush=True)
    except BootstrapError as e:
        print(f"ERROR: {e}", flush=True)
        return 1
    print("    steps: " + ", ".join(f"{s} {'done' if status[s] else 'TODO'}" for s in STEPS), flush=True)
    if check_only or not todo:
        print("READY" if not todo else f"NOT READY: {', '.join(todo)}", flush=True)
        return 0 if not todo else 2
    try:
        for i, s in enumerate(STEPS, 2):
            if not step_status(state)[s]:
                BUILDERS[s](i, state)
    except BootstrapError as e:
        print(f"ERROR: {e}", flush=True)
        return 1
    extra = [x for x in cfg.embedding.index_instances if x != cfg.models.embedder]
    print(f"READY: bootstrap finished in {(time.time() - t0) / 60:.1f} min." +
          (f" Extra embedder instances ({', '.join(extra)}) can be unloaded: lms unload <identifier>"
           if extra and lms_loaded() is not None else ""), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="main.py bootstrap", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="report status only, build nothing")
    sys.exit(bootstrap(ap.parse_args(argv).check))


if __name__ == "__main__":
    main()
