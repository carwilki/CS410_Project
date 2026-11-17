# Status (2026-11-15)

Phase 0 approved. Decisions U1 and later, plus all facts, are in docs/DECISIONS.md.

## Done
- **Corpus:** `artifacts/corpus.parquet` (3,851,415 narratives + metadata + token counts) and JsonCollection.
- **max_tokens:** full-corpus p50 155 and p75 288 tokens, both under 512, so it stays at 512.
- **BM25:** `indexes/bm25`, 3,851,415 docs indexed, 0 errors.
- **HyDE path** (`cfpb_hyde/hyde.py`): 5 concurrent Gemma requests with `reasoning_effort:"none"`, Eq. 3 mean, JSON logs.
- **Retrieval** (`cfpb_hyde/search.py`): bm25 / hyde / hybrid, using Pyserini `HybridSearcher._hybrid_results`.
- **TUI** (`main.py` → `cfpb_hyde/tui.py`): MP2 layout plus `:open N`, `:hyde`, `:mode`, `:k`.
  - Smoke-tested on a partial Faiss index: all commands worked, 5/5 generations ended with `stop`, no reasoning.
- **Unit tests:** 9 passing (Eq. 3, prefixes, reasoning stripping, fusion vs hand values, HyDE log).

## Product direction (user, 2026-11-15)
- **Zero-shot search application:** the batch evaluation is cancelled. Judging happens in the TUI: `:rate` (human), `:judge` (LLM), `:review`, and live fusion settings (Ctrl+S).
- **Default mode:** hybrid with normalized fusion.
- **Smoke test:** all admin functions passed on a partial index. 28 tests pass.

## Full index (2026-11-15 18:39)
- **Faiss:** `indexes/faiss_nomic768`, 3,851,415 vectors with a matching docid count. 0 embedding failures; 13,334 docs truncated at 2048 tokens (`artifacts/embed_work/truncated.tsv`).
- **Search smoke test on the full index:**
  - Load time 1.8 s.
  - bm25 takes 0.1 s, hyde about 5–13 s, hybrid about 5 s; most of the time is the 5 Gemma generations.
  - On a medical-debt query, hyde and hybrid ranked on-topic narratives first.
- **TUI on the full index:** headless run of a hybrid query, `:rate 1 3`, `:judge` (10/10 judged, all grade 3) and `:review`, with all ratings stored.
- **Run it:** `uv run python main.py`

## Responsiveness (2026-11-16)
- **Status bar** above the input shows the live stage and elapsed time for loading, search and judge.
- **Startup:** indexes load and warm in the background, taking about 17 s; the UI appears immediately.
- **Input during search:** stays usable; a second search is refused with a toast.
- **Event-loop stall:** at most 6 ms during search and judge (DECISIONS U4). 28 tests pass.

## Fresh clone + provider (2026-11-16)
- **Bootstrap:** `uv run python main.py bootstrap` (or `:bootstrap` in the TUI) builds everything from `data/*.zip`; it is resumable and verified (DECISIONS U5). The existing build was adopted, and `bootstrap --check` reports READY.
- **Provider:** set in `config.yaml provider:` (DECISIONS U6). Only LM Studio has been tested.
- **Tests:** 33 pass.
