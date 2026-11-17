# Decisions and verified facts

All entries checked 2026-11-15 unless noted. Probe outputs: `docs/probe_out/`. Probe scripts: `scripts/probes/`.

## Environment
| # | Fact | Source |
|---|---|---|
| E1 | Apple M5 Max, 64 GiB RAM (68719476736 B), macOS Darwin 25.6 | `sysctl hw.memsize machdep.cpu.brand_string` |
| E2 | Java: Zulu OpenJDK 21.0.12.1 | `java -version` |
| E3 | Pyserini 2.4.0 requires Java 21 | `.venv/.../pyserini-2.4.0.dist-info/METADATA:60` ("built on Python 3.12 … and Java 21") |
| E4 | Pyserini bundles `resources/jars/anserini-2.3.0-fatjar.jar`; JVM starts under Java 21 | observed: toy FaissSearcher test ran |
| E5 | faiss-cpu 1.16.0, torch 2.15.0 (pyserini dependency) | `uv pip show` |

## Models (LM Studio, http://localhost:1234)
| # | Fact | Source |
|---|---|---|
| M1 | Generator = `google/gemma-4-e4b` (user changed from Nemotron 2026-11-15); GGUF `lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf`; loaded ctx 8192 | user; `GET /v1/models`, `GET /api/v0/models`, `lms ps` |
| M2 | Judge = `google/gemma-4-26b-a4b`, loaded ctx 8192 | user; same |
| M3 | Embedder = `text-embedding-nomic-embed-text-v1.5`, loaded ctx **2048** (= GGUF max per LM Studio) | `lms ps`; `/api/v0/models` max_context_length 2048 |

## Gemma 4 reasoning control
| # | Fact | Source |
|---|---|---|
| R1 | Thinking is enabled by `<\|think\|>` at start of system prompt; removing it disables | HF card https://huggingface.co/google/gemma-4-E4B-it ("Thinking Mode Configuration"); https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4 |
| R2 | Thinking output wrapped `<\|channel>thought\n … <channel\|>`; E2B/E4B emit no channel tags when thinking is off | HF card (same) |
| R3 | Shipped GGUF template inserts `<\|think\|>` only when `enable_thinking` is true | `docs/probe_out/gemma4_e4b_chat_template.jinja:179-186` (extracted from GGUF `tokenizer.chat_template`) |
| R4 | **Observed:** LM Studio `/v1/chat/completions` with no reasoning param returns non-empty `reasoning_content` on 5/5 narrative prompts → LM Studio turns thinking ON by default for this model | `docs/probe_out/p3_reasoning.json`, `p3b_reasoning_params.json` (baseline) |
| R5 | **Observed:** on `/v1/chat/completions`, `reasoning_effort: "none"` → reasoning_content empty 5/5. `chat_template_kwargs.enable_thinking=false` and `reasoning: {effort: "none"}` did NOT disable it (5/5 still reasoned) | `p3b_reasoning_params.json` |
| R6 | `reasoning_effort` on `/v1/chat/completions` is **not documented** by LM Studio (supported-params list omits it) | https://lmstudio.ai/docs/developer/openai-compat/chat-completions |
| R7 | Documented mechanism: native `POST /api/v1/chat` field `reasoning: "off"\|"low"\|"medium"\|"high"\|"on"`. Observed with `reasoning:"off"`: `stats.reasoning_output_tokens = 0`, output contains only a `message` item | https://lmstudio.ai/docs/developer/rest/chat; observed 3 calls |
| R8 | Gemma recommended sampling: temperature 1.0, top_p 0.95, top_k 64 | HF card "Sampling Parameters" |
| R9 | LM Studio returns reasoning in `message.reasoning_content` (setting under App Settings > Developer; documented as experimental) | https://lmstudio.ai/blog/lmstudio-v0.3.9 |

**Open:** the documented switch lives on the native endpoint, not `/v1/chat/completions`. Needs user decision (see STATUS).

## Embedder (nomic-embed-text-v1.5)
| # | Fact | Source |
|---|---|---|
| N1 | Output dim 768; L2 norm = 1.0000 (LM Studio returns normalized vectors) | observed, `p4_embed.json` |
| N2 | LM Studio ignores `dimensions` (sent 256, got 768); not documented | observed; https://lmstudio.ai/docs/developer/openai-compat/embeddings |
| N3 | Prefixes required: `search_query: `, `search_document: ` | https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 |
| N4 | Matryoshka evaluated at 768/512/256/128/64 (MTEB 62.28/61.96/61.04/59.34/56.10). **Decision (user): keep 768** | HF card |
| N5 | Over-length input (6,497 tokens > 2048 ctx) returns HTTP 200 with a vector — **truncation is silent**; must be detected client-side | observed, `p4_embed.json` |
| N6 | Token lengths use the HF `nomic-ai/nomic-embed-text-v1.5` `tokenizer.json` incl. prefix + special tokens. Caveat: llama.cpp tokenization inside LM Studio may differ slightly (not verified) | `scripts/probes/p4_embed.py` |

## Pyserini 2.4.0 (paths relative to `.venv/lib/python3.12/site-packages/pyserini/`)
| # | Fact | Source |
|---|---|---|
| P1 | (a) `FaissSearcher.search(query: Union[str, np.ndarray, Dict], k=10, …)`; non-str/non-dict query is used directly as the vector (`emb_q = query`), cast to float32 | `search/faiss/_searcher.py:165-172, 193-202` |
| P2 | `FaissSearcher.__init__` needs a `query_encoder`; any non-str object is accepted as-is | `search/faiss/_searcher.py:68-82`; observed with `object()` |
| P3 | (c) Index dir layout: `<dir>/index` (read with `faiss.read_index`) + `<dir>/docid` (one docid per line); asserts `ntotal == len(docids)` | `search/faiss/_searcher.py:315-332, 367-371, 88` |
| P4 | (b) `HybridSearcher.search(query: str, k0=10, k=10, alpha=0.1, normalization=False, weight_on_dense=False)` passes the **same `query`** to both dense and sparse searchers → cannot take a precomputed vector (LuceneSearcher needs a string) | `search/hybrid/_searcher.py:40-43` |
| P5 | Fusion `_hybrid_results` (staticmethod): union of docids; missing dense score := min dense score of the list, missing sparse := min sparse; if `normalization`: `s' = (s − (min+max)/2) / (max − min)` per list; score = `alpha·sparse + dense` (or `sparse + alpha·dense` if `weight_on_dense`); sort desc, top k | `search/hybrid/_searcher.py:56-82` |
| P6 | `pyserini.eval.trec_eval` wraps Anserini's Java trec_eval (`java -cp <jar> trec_eval`) | `eval/trec_eval.py:62-65` |

## HyDE reference (texttron/HyDE @ a2fd8734307612cb0225d71ffbf26e0d225986b8, main, 2024-12-06)
Line numbers counted from fetched raw text (may be off by one).
| # | Fact | Source |
|---|---|---|
| H1 | Prompt templates, e.g. WEB_SEARCH "Please write a passage to answer the question.\nQuestion: {}\nPassage:"; FIQA "Please write a financial article passage to answer the question.\nQuestion: {}\nPassage:"; TREC_NEWS "Please write a news passage about the topic.\nTopic: {}\nPassage:" | `src/hyde/promptor.py:1-38` |
| H2 | Generator defaults: n=8, max_tokens=512, temperature=0.7, top_p=1, frequency/presence_penalty=0, stop=['\n\n\n'] | `src/hyde/generator.py:15` |
| H3 | Demo: `text-davinci-003`, task 'web search', defaults (n=8), contriever, k=10 | `hyde-demo.ipynb` cells 3, 15, 17 |
| H4 | `HyDE.encode`: unweighted mean of encoder outputs over [query] + hypothesis docs; no extra normalization | `src/hyde/hyde.py:19-27` |
| H5 | Spec sets N=5 (deviation from repo n=8 is the spec's choice; Eq. 3 divisor becomes 6) | spec |

## Data
| # | Fact | Source |
|---|---|---|
| D1 | 21 zips, each 1 CSV, identical 16-column header | `p1_data.json` |
| D2 | Fields: id `Complaint ID`; narrative `Consumer complaint narrative`; metadata `Date received`, `Product`, `Sub-product`, `Issue`, `Sub-issue`, `Company`, `Company response to consumer`, `Company public response`, `State`, `ZIP code`, `Tags`, `Submitted via`, `Date sent to company`, `Timely response?` | `p1_data.json` |
| D3 | 17,533,918 rows; 3,851,415 with non-empty narrative; all unique IDs | `p1_data.json` |

## Phase 1–3 decisions (2026-11-15, after user approval of Phase 0)
| # | Decision / fact | Source |
|---|---|---|
| U1 | User decisions: reasoning off via `reasoning_effort:"none"` on `/v1/chat/completions` (1A); fusion via Pyserini itself, not copied (2); Textual full-screen UI only (3); max_tokens 512 unless p50/p75 > 512 (4); full embedding run approved (5) | user, 2026-11-15 |
| D4 | Full-corpus token lengths (nomic tokenizer, with `search_document: `): p50 155, p75 288, p90 492, p95 690, p99 1,327, max 20,838. 13,334 docs (0.35%) > 2048 → listed in `artifacts/embed_work/truncated.tsv`. 9.2% > 512 | `scripts/build_corpus.py` output |
| G1 | max_tokens stays 512 (p50 155 and p75 288 < 512, per U1-4) | D4 |
| G2 | **Observed:** LM Studio ignores `n` (n=3 → 1 choice). HyDE makes N=5 independent requests (concurrent) | curl test 2026-11-15 |
| G3 | Prompt adapted from HyDE promptor.py shape: user turn "Please write a consumer complaint narrative, in the style of a CFPB submission, describing the situation.\nSituation: {}\nNarrative:". **Deviation:** HyDE uses no system turn; I add one asking for plain first-person prose with no headings/placeholders, because Gemma otherwise wrote form letters (Phase 0 §3 outputs) | `cfpb_hyde/hyde.py`; `docs/probe_out/p3_reasoning.json` |
| G4 | Hypothetical documents embedded with `search_document: ` (they stand in for documents); query with `search_query: ` | design decision per spec; N3 |
| G5 | f() outputs are L2-normalized (N1). Eq. 3 mean is **not** re-normalized (H4). Dense score = inner product (IndexFlatIP) with the mean vector. Fusion uses Pyserini defaults without normalization, so no extra normalization is required | `cfpb_hyde/embed.py`; P5 |
| F1 | Hybrid = `HybridSearcher._hybrid_results(dense_hits, sparse_hits, alpha, k, normalization, weight_on_dense)` called directly (staticmethod; `search()` can't take a vector, P4). Private API: pinned to pyserini 2.4.0 | `cfpb_hyde/search.py`; `search/hybrid/_searcher.py:56-82` |
| F2 | `hybrid_k0 = 1000` candidates per list before fusion. **My choice**, Pyserini default is k0=10, which would leave fusion only 10 candidates per side. In config | config.yaml |
| B1 | BM25 explicitly `set_bm25(0.9, 0.4)` (Pyserini defaults). Index: JsonCollection, DefaultLuceneDocumentGenerator, 16 threads: 3,851,415 indexed, 0 empty/skipped/errors, 22 s | `lucene/_searcher.py:583`; `logs/bm25_index.log` |
| I1 | Embedding throughput: 1 instance ≈155 docs/s regardless of client workers (1/4/8). 4 instances (`lms load … --identifier nomic-N -c 2048`) + 4 workers ≈270 docs/s; 8 instances ≈278 (GPU-bound). Using 4 → ≈4.0 h | observed, `artifacts/embed_work/progress.log` |
| I2 | All 4 instances return bit-identical vectors (max abs diff 0.0) | observed |
| I3 | Over-length docs are sent whole (no chunking); LM Studio truncates silently (N5); IDs logged to `truncated.tsv`. Failed docs → `failures.jsonl`, excluded from the index | `scripts/embed_corpus.py` |

## Phase 5 evaluation (2026-11-15; built by a sub-agent, key outputs re-checked by me)
| # | Fact / decision | Source |
|---|---|---|
| V1 | UMBRELA @ `e2c4e4126bf7b568d9230008b465a58445cabc30`: zero-shot "bing" prompt `src/umbrela/prompts/prompt_templates/qrel_zeroshot_bing.yaml:4-20`; system msg "You are a helpful assistant." (`gpt_judge.py:276-279`); temperature 0, top_p 1, frequency_penalty 0.5, presence_penalty 0, max_tokens default 100 (`gpt_judge.py:8,178,185-193,331`); parser `common_utils.py:63-113` (30 regexes, last-resort = last digit 0–3; no match → grade 0 invalid) | castorini/umbrela; copy in `cfpb_hyde/eval/umbrela.py` |
| V2 | Judge `google/gemma-4-26b-a4b` with `reasoning_effort:"none"`: 2/2 calls had 0 reasoning tokens, no channel tags | `docs/probe_out/p5_judge_smoke.json` (re-checked) |
| V3 | **Observed:** at max_tokens 100 the judge writes numbered steps, gets cut off (`finish_reason: length`), and the parser grabs the "2" from "Step 2" → spurious grade 2 marked valid. At 1024: 294 tokens, `##final score: 1` | same file (re-checked) |
| V4 | trec_eval (Anserini 9.0.4 port): `-c -q -M 10 -l <threshold> -m ndcg_cut.10 / P.10 / recip_rank`. `-l` = min grade for "relevant" (affects P, RR; nDCG uses graded gains). trec_eval re-sorts by score, tie-break docno | `java -cp anserini-2.3.0-fatjar.jar trec_eval -h`; toy runs; `pyserini/eval/trec_eval.py:62-65,186-200` |
| V5 | Significance: paired t-test (`scipy.stats.ttest_rel`) and Wilcoxon signed-rank (`scipy.stats.wilcoxon`, zero_method="wilcox"), two-sided, per metric, no multiple-comparison correction | `cfpb_hyde/eval/stats.py` |
| V6 | Kappa: sklearn `cohen_kappa_score`: unweighted 0–3, quadratic-weighted 0–3, and binary at grade ≥ threshold | same |
| V7 | Unverified: whether LM Studio applies frequency/presence penalties; behavior when prompt exceeds judge ctx 8192 (handled as UMBRELA 4xx → grade 0 invalid) | — |
| V8 | **Deviation (user 2026-11-15):** UMBRELA level definitions reworded from "answers the query" to "describes the same situation" (`PROMPT_EDITS` in `cfpb_hyde/eval/umbrela.py`; upstream text kept as `UPSTREAM_PROMPT`). Scale, instructions, steps, output format and parser unchanged. Reason: with the verbatim prompt, a narrative mirroring the query got 1 ("anecdotal description … rather than providing an answer") | V3 smoke |
| V9 | **Deviation (user 2026-11-15):** judge max_tokens 512 (UMBRELA 100). Re-smoke with V8 prompt: same pair → `##final score: 3`, not truncated, 0 reasoning tokens | `scripts/judge.py --smoke` |
| Q1 | Queries: 30 hand-drafted by a sub-agent from the product/issue distribution (no LLM generator used); approved as-is by user 2026-11-15 → `queries/queries.tsv` | `queries/DRAFT_NOTES.md` |

## Product direction (user, 2026-11-15)
| # | Decision | Source |
|---|---|---|
| U2 | The product is a zero-shot HyDE semantic search application, not an experiment: no qrels needed to search. The batch eval pipeline (pool → judge → metrics) is **not run**; relevance judging moves into the TUI as admin functions: human `:rate`, on-demand LLM `:judge` (UMBRELA, V8/V9), `:review`, and live fusion settings. Stored in `admin/relevance.sqlite` | user |
| U3 | Default = hybrid with `fusion_normalization: true` (Pyserini `_hybrid_results` normalization, P5) so the HyDE score carries weight. Raw-score fusion let BM25 (×0.1 ≈ 1–2) dominate dense (≈0.6–0.8) | user; smoke test |
| U4 | TUI responsiveness (user, 2026-11-16): status bar shows each stage live (index load and warm-up, per-document generation n/5, embedding, dense, BM25, fusion, judge n/10) with elapsed time. Indexes load in a background worker after the UI is up, and the input stays enabled during searches; a second search while one runs gets a warning toast. Measured stalls before the change: lazy metadata load 4–5 s plus the first `meta.loc` building a hash with a 550 ms GIL hold, both on the first query; cold Faiss scan 0.46 s vs 0.15 s warm. After: worst event-loop stall 6 ms during query and judge, 545 ms once at startup. Uses `App._thread_id` (present in Textual 8.2.8 `App`, checked via inspect) to tell UI thread from workers | user; scratchpad lag probes 2026-11-16 |
| U5 | Fresh-clone bootstrap (user, 2026-11-16: "only the data and code will be made available"). `main.py bootstrap [--check]` and TUI `:bootstrap [check|stop]` run `cfpb_hyde/bootstrap.py`, which handles preflight, corpus, BM25, embeddings and Faiss. Each step is verified (parquet rows = JsonCollection lines; Lucene `num_docs` = N; all chunks present; docid count = N − failures, and index file size = n·dim·4 + header, observed 45 B) before a marker is written to `artifacts/bootstrap_state.json`. Markers chain on a data-zip fingerprint and a corpus id hash. Pre-bootstrap outputs are adopted only without a marker (corpus: also newer than every zip). Embedding chunks carry a manifest (id hash, chunk size, embedder, prefix, dim), and stale chunks are refused, not deleted. Tested: adopting the existing full build (13 s, nothing rebuilt); a code-only copy with 1 zip, then 2 zips (data change → full rebuild, stale chunks refused, `lms load` of a new instance); the TUI flow on a code-only copy (setup notice → `:bootstrap check` → `:bootstrap` → search) | user; scratchpad tests 2026-11-16 |
| U6 | Provider configurable (user, 2026-11-16). `config.yaml provider:` takes name, base_url, api_key_env (bearer token from env) and chat_extra (provider-specific chat fields; `reasoning_effort: "none"` moved here from `hyde:`, R5). Any OpenAI-compatible server with /models, /chat/completions and /embeddings works. `lms` instance loading runs only for `name: lmstudio`; for others, embedder instances are limited to ids listed at /models. Non-unit embeddings are L2-normalized client-side; this is a no-op for LM Studio, which returns unit vectors (N2). The bootstrap and TUI warn if `models.embedder` differs from the embedder recorded in the index manifest. Only LM Studio has been run; other servers are untested | user |
| U7 | Grading in the document view (user, 2026-11-16). `DocScreen` (`:open N` or clicking a result header) shows the query, the result's metadata and narrative, and a grade bar. Keys 0–3 or a click store a human grade (same `AdminStore.rate` as `:rate`), refresh the result list tags, and advance to the next result. Left/Right/p/n step through results. Ratings are stored only; they are **not** used in retrieval (the user scoped this request to the view; feedback into retrieval is not built) | user; headless pilot test 2026-11-16 |
