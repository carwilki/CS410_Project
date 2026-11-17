# MP2 terminal interface — observed notes

Source: /Users/cwilkins/Projects/school/cs410/mp2 (read 2026-11-15). Only what is in the code/README.

## Entry point and mode selection
- `main.py` (`__main__`, lines ~136–143): if `--plain` is in `sys.argv` or stdin is not a TTY → plain loop `main()`; otherwise `tui.run_tui()`.
- No argparse. The only CLI flag is `--plain`, checked by string membership in `sys.argv`.
- Run: `uv run python main.py` (TUI) / `uv run python main.py --plain`.

## Configuration
- `.env` via python-dotenv (`utils/config.py`): `PYSERINI_CNAME`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `RETRIEVER_K`, `PYSERINI_K1`, `PYSERINI_B`, `QUERY_DECOMPOSITION`, `QUERY_HISTORY_SIZE`.
- TUI settings dialog can write changes back to `.env` (`dotenv.set_key`), "Also save to .env" checkbox.

## Plain mode (`main.py: main()`)
- Loop: `input("Please enter your question (or type 'exit' to quit): ")`; `exit` (case-insensitive) quits with "Exiting the program. Goodbye!".
- No other commands.
- Output per query, in order: sub-queries (if any), `=== Retrieved evidence ===`, `=== Generated answer ===`, then a `-` × 50 separator.
- Colors: raw ANSI codes via `color(text, code)`; constants `CYAN=1;36, GREEN=1;32, YELLOW=1;33, RED=1;31, DIM=2, BOLD=1`. Disabled when `NO_COLOR` is set or stdout isn't a TTY (unless `FORCE_COLOR`).
- Result layout (`print_sources`): header line `[rank] doc <docid>  |  BM25 rank #r  |  score s.ss` (bold), then a tag (`<< used in answer` green / `(not cited)` dim), then a snippet of the first 300 chars (`SNIPPET_CHARS`), whitespace-collapsed, `textwrap.fill(…, 100)`, indented 4 spaces; a dim legend line at the end explaining the score.
- Section headers: `=== … ===` in cyan (evidence) / yellow (answer).

## TUI mode (`tui.py`, Textual 8.2.8 + Rich `Text`)
- `RagApp(App)`: `TITLE="MP2 RAG Agent"`, `SUB_TITLE` = active settings summary. Layout: `Header`, scrolling `VerticalScroll#history` of per-query blocks, hidden `RichLog#log-panel`, `QuestionInput` docked bottom, `Footer`.
- Bindings: Ctrl+Q quit, Ctrl+S settings dialog, Ctrl+G toggle log panel, Ctrl+L clear on-screen history; Ctrl+P Textual command palette. In the input: Up/Down recall history (persisted to `.query_history`, size `QUERY_HISTORY_SIZE`). Multi-line paste joined to one line. Typing `exit` quits.
- Per query: `QueryBlock` with `Q: <question>` (bold cyan "Q:"), collapsible "Sub-queries", "Debug info (N log lines, T s)", "Retrieved documents (N passages, M cited)", then the answer below a rule. Rounded border `$primary`.
- Retrieved documents render (`render_sources`): same header format as plain mode, full passage text (not truncated), cited bright / uncited dim, legend in dim italic.
- Clickable `[n]` citations open `PassageScreen` modal with rank/docid/score and full text; Esc or click-outside closes.
- Query runs in a worker thread (`@work(thread=True)`); input disabled while running; per-query logs captured into the Debug section.
- Settings dialog: model dropdown, corpus, k, k1, b, decomposition toggle, history size; Apply/Cancel; validation errors shown inline; failed apply restores previous settings.

## Not present in MP2
- No typed commands other than `exit` (no `:open`, `:mode`, etc.). Document-opening is by clicking a citation (TUI only).
- No argparse, no YAML config.
