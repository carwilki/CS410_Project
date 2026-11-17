# CFPB complaint search: HyDE + BM25 hybrid retrieval

Zero-shot retrieval over 3.85M CFPB consumer complaint narratives. You describe your situation in
plain language and get the most similar real complaints. Retrieval is HyDE (Gao et al., ACL 2023)
fused with BM25 through Pyserini. Every model/library claim is cited in `docs/DECISIONS.md`.

## Setup (fresh clone)

A clone contains only code. You add the data, then one command builds everything else.

| Requirement | Version used | Notes |
|---|---|---|
| Python | 3.12 | via [uv](https://docs.astral.sh/uv/) |
| Java (JDK) | 21 | required by Pyserini 2.4.0 (`java -version`) |
| Model provider | LM Studio, server on `http://localhost:1234` | any OpenAI-compatible server, see below |
| RAM | 64 GB used | Faiss index ~11 GiB + metadata ~4 GB |
| Disk | ~35 GB | corpus, BM25, embedding chunks, Faiss |

```bash
uv sync
# put the CFPB CCDB_Export_*.zip files in ./data (one CSV per zip)
uv run python main.py bootstrap --check   # preflight: data, Java, provider, models; lists missing steps
uv run python main.py bootstrap           # builds whatever is missing (~4-5 h for the full data)
```

You can also run `:bootstrap` inside the TUI. The status bar shows each step's progress, and the full
output goes to the log panel (Ctrl+G). The bootstrap steps are:

1. **Preflight:** checks for data zips, JDK 21+, a reachable provider, and that the configured models
   are listed at `GET /models`.
2. **Corpus:** narratives + metadata go to `artifacts/corpus.parquet` and a Pyserini JsonCollection
   (`scripts/build_corpus.py`, ~2 min). The nomic tokenizer is downloaded from the Hugging Face hub for
   token counts.
3. **BM25:** a Lucene index in `indexes/bm25` (`pyserini.index.lucene`, all cores, ~30 s).
4. **Embeddings:** `scripts/embed_corpus.py` writes checkpoints every 4,096 docs and can be resumed.
   With LM Studio, the extra embedder instances in `embedding.index_instances` are loaded via
   `lms load` for throughput.
5. **Faiss:** `indexes/faiss_nomic768/{index,docid}`.

Each step is verified (row counts, Lucene doc count, chunk count, vector count) before it is marked
done in `artifacts/bootstrap_state.json`. Rerunning skips finished steps and resumes embedding. New or
changed zips in `data/` invalidate all steps. Outputs from an earlier manual build are verified and
reused. Stop with Ctrl+C (or `:bootstrap stop`) at any point.

### Model provider

`config.yaml` → `provider:` selects the server. LM Studio is the default, and any OpenAI-compatible
server that serves `/models`, `/chat/completions` and `/embeddings` can be used:

```yaml
provider:
  name: lmstudio                 # anything else = generic OpenAI-compatible (no `lms` extras)
  base_url: http://localhost:1234/v1
  api_key_env: null              # env var with a bearer token, if the server needs one
  chat_extra: {reasoning_effort: "none"}   # extra request fields; provider-specific
models:                          # ids exactly as the provider lists them at GET /models
  generator: google/gemma-4-e4b
  embedder: text-embedding-nomic-embed-text-v1.5
  judge: google/gemma-4-26b-a4b
```

Models used with LM Studio:
- `google/gemma-4-e4b`: generates the hypothetical documents
- `text-embedding-nomic-embed-text-v1.5`: embedder, context 2048
- `google/gemma-4-26b-a4b`: relevance judge for `:judge`

**The Faiss index belongs to the embedder that built it.** You can switch the generator or judge
freely. If you switch the embedder, the query vectors no longer match the index unless it is the same
model with the same weights. The TUI and `bootstrap --check` warn about this. To re-embed, delete
`artifacts/embed_work` and `indexes/faiss_nomic768`, then rerun bootstrap. `reasoning_effort: "none"`
is how LM Studio turns off Gemma thinking (DECISIONS R5). Remove it for servers that reject unknown
fields.

### Manual build steps (what bootstrap runs)

```bash
uv run python -m scripts.build_corpus
uv run python -m pyserini.index.lucene --collection JsonCollection \
  --input artifacts/json_collection --index indexes/bm25 \
  --generator DefaultLuceneDocumentGenerator --threads 16
for i in 2 3 4; do lms load text-embedding-nomic-embed-text-v1.5 --identifier nomic-$i -c 2048 --gpu max -y; done
uv run python -m scripts.embed_corpus --workers 4      # resumable
uv run python -m scripts.embed_corpus --assemble
```

The embedding job checkpoints every 4,096 docs (`artifacts/embed_work/chunk_*.npy`) and logs to
`artifacts/embed_work/progress.log`. Docs longer than the embedder's 2048-token context are listed
in `truncated.tsv`; requests that fail go to `failures.jsonl` and those docs are left out of the index.

## Run the interface

```bash
uv run python main.py
```

Type a description of your situation and press Enter. Results show rank, score, complaint ID, date,
product, issue, company, company response, and a snippet. Search is zero-shot: HyDE needs no training
data or relevance labels.

The UI opens immediately and loads the indexes in the background (about 17 s). Searches typed before
loading finishes wait for it. The status bar above the input shows what is running at each stage, for
example "Generating hypothetical documents (3/5 done)" or "Judging 10 results (4/10 done)", with
elapsed time. The input stays usable during a search, so `:open` and `:rate` still work.

| Input | Action |
|---|---|
| `:open N` (or click a result header) | full narrative with the query shown; grade it with keys `0`-`3` or by clicking a grade (saved as your human grade, then moves to the next result); Left/Right (or `p`/`n`) step through results; Esc closes |
| `:hyde` | the hypothetical documents generated for the last query |
| `:mode bm25\|hyde\|hybrid` | switch retrieval mode (default hybrid, normalized fusion) |
| `:k N` | set top-n (default 10) |
| `:rate N G` | admin: grade result N of the last query, G = 0 not relevant, 1 related but a different situation, 2 similar situation, 3 same situation |
| `:judge` | admin: grade every result of the last query with the LLM judge (`google/gemma-4-26b-a4b`, UMBRELA prompt), shown next to each result |
| `:review` | admin: saved searches with their human and LLM ratings |
| `:bootstrap` / `:bootstrap check` / `:bootstrap stop` | setup: build missing indexes from `data/` (resumable), report status, or stop |
| Ctrl+S | settings: mode, top-n, fusion alpha, fusion normalization (this session) |
| `:help` / `exit`, Ctrl+G / Ctrl+L / Ctrl+Q | help / quit, log panel / clear / quit |
| Up / Down | query history |

Every search (query, mode, settings, ranked results) and every rating is stored in
`admin/relevance.sqlite`. Each HyDE query also writes `hyde_logs/<run_id>/<query_id>.json`
(prompt, parameters, raw and cleaned outputs).

The batch evaluation scripts (`scripts/run_pool.py`, `judge.py`, `human_sample.py`, `kappa.py`,
`metrics.py`) remain in the repo but are not part of the application.

## Tests

```bash
uv run pytest -q
```

## Configuration

All URLs, model ids, N, top-n, fusion weight, pool depth and thresholds are in `config.yaml`.
