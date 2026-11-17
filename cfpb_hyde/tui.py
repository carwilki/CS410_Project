"""Textual UI adapted from MP2's tui.py (see docs/INTERFACE_NOTES.md).

Kept from MP2: Header with active settings, scrolling per-query blocks with collapsible sections,
bottom input with Up/Down history (.query_history), modal popup for full text, Ctrl+S settings
dialog, Ctrl+G log panel, Ctrl+L clear, Ctrl+Q / 'exit' to quit.
New: search commands (`:open N`, `:hyde`, `:mode M`, `:k N`) and admin relevance functions
(`:rate N G`, `:judge`, `:review`), clickable result headers, and live fusion settings.
Every search and rating is stored in the admin SQLite store (cfpb_hyde/admin.py).
"""
import logging
import subprocess
import sys
import threading
import textwrap
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import (Button, Checkbox, Collapsible, Footer, Header, Input, Label, LoadingIndicator,
                             RichLog, Select, Static)

from cfpb_hyde import bootstrap
from cfpb_hyde.admin import GRADES, AdminStore
from cfpb_hyde.config import ROOT, load, path
from cfpb_hyde.eval import umbrela
from cfpb_hyde.search import MODES, Retriever

logger = logging.getLogger(__name__)
HISTORY_PATH = ROOT / ".query_history"
HISTORY_SIZE = 20
SNIPPET_CHARS = 300
HUMAN = "human"
HELP = ("Search:  :open N  full narrative of result N   :hyde  hypothetical documents   "
        ":mode bm25|hyde|hybrid   :k N  top-n\n"
        "Admin:   :rate N G  rate result N with grade G (0-3)   :judge  LLM-judge all results of the last query   "
        ":review  saved searches and ratings   Ctrl+S  settings (mode, top-n, fusion)\n"
        "Setup:   :bootstrap  build corpus + indexes from data/ (resumable)   :bootstrap check   :bootstrap stop\n"
        "Grades:  " + "   ".join(f"{g} = {d}" for g, d in GRADES.items()) + "   |   :help   exit")
_ids = count()
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class StatusBar(Static):
    """One line above the input: every running activity with its current stage and elapsed time.
    Workers report stages through HydeApp.status(); a 0.1 s timer redraws the spinner and clocks."""

    def on_mount(self):
        self.activities: dict[str, tuple[str, float]] = {}  # key -> (message, start)
        self.last = Text("Starting ...", style="dim")
        self.frame = 0
        self.set_interval(0.1, self.redraw)

    def set(self, key: str, msg: str):
        start = self.activities.get(key, (None, time.time()))[1]
        self.activities[key] = (msg, start)
        self.redraw()

    def done(self, key: str, msg: str, style: str = "green"):
        self.activities.pop(key, None)
        self.last = Text(msg, style=style)
        self.redraw()

    def redraw(self):
        self.frame += 1
        if not self.activities:
            self.update(self.last)
            return
        t = Text()
        for i, (msg, start) in enumerate(self.activities.values()):
            if i:
                t.append("   │   ", style="dim")
            t.append(SPINNER[self.frame % len(SPINNER)] + " ", style="bold yellow")
            t.append(msg).append(f"  {time.time() - start:.1f}s", style="dim")
        self.update(t)


class QueryHistory:
    def __init__(self, path: Path = HISTORY_PATH):
        self.path = path
        try:
            self.items = [l for l in path.read_text().splitlines() if l.strip()][-HISTORY_SIZE:]
        except OSError:
            self.items = []

    def add(self, q: str):
        if q in self.items:
            self.items.remove(q)
        self.items = (self.items + [q])[-HISTORY_SIZE:]
        try:
            self.path.write_text("\n".join(self.items) + "\n")
        except OSError:
            pass


class QuestionInput(Input):
    BINDINGS = [Binding("up", "history_prev", "Previous query", show=False),
                Binding("down", "history_next", "Next query", show=False)]

    def __init__(self, history: QueryHistory, **kw):
        super().__init__(**kw)
        self.history, self.pos, self.draft = history, None, ""

    def _show(self, text):
        self.value = text
        self.cursor_position = len(text)

    def action_history_prev(self):
        if not self.history.items:
            return
        if self.pos is None:
            self.draft, self.pos = self.value, len(self.history.items)
        self.pos = max(0, self.pos - 1)
        self._show(self.history.items[self.pos])

    def action_history_next(self):
        if self.pos is None:
            return
        self.pos += 1
        if self.pos >= len(self.history.items):
            self.pos = None
            self._show(self.draft)
        else:
            self._show(self.history.items[self.pos])

    def reset_history_position(self):
        self.pos = None

    def _on_paste(self, event: events.Paste) -> None:
        event.prevent_default()
        self.insert_text_at_cursor(" ".join(event.text.split()))


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


class AppLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines, self.app = [], None
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        line = self.format(record)
        self.lines = (self.lines + [line])[-2000:]
        if self.app:
            try:
                self.app.call_from_thread(self.app.write_log, line)
            except RuntimeError:  # called from the UI thread
                self.app.write_log(line)


def meta_lines(m) -> Text:
    t = Text()
    t.append("    date ", style="dim").append(str(m["date"]))
    t.append("  |  product ", style="dim").append(str(m["product"]) + "\n")
    t.append("    issue ", style="dim").append(str(m["issue"]))
    if m.get("sub_issue"):
        t.append(" / " + str(m["sub_issue"]))
    t.append("\n    company ", style="dim").append(str(m["company"]))
    t.append("  |  response ", style="dim").append(str(m["company_response"]) + "\n", style="green")
    return t


def rating_tags(r: dict) -> str:
    """'  [human 3 | llm 2]' from {rater: grade}."""
    tags = [f"human {r[HUMAN]}"] if HUMAN in r else []
    tags += [f"llm {g}" for k, g in r.items() if k.startswith("llm:")]
    return f"  [{' | '.join(tags)}]" if tags else ""


def render_results(result, meta, block_id: str, ratings: dict) -> Content:
    if not result.hits:
        return Content("(no results)")
    parts = []
    for h in result.hits:
        m = meta.loc[h.docid]
        parts.append(Content.from_markup(
            f"[bold underline @click=app.show_doc('{block_id}', {h.rank})]$hdr[/][bold $success]$tags[/]",
            hdr=f"[{h.rank}] complaint {h.docid}  |  {result.mode} score {h.score:.4f}",
            tags=rating_tags(ratings.get(h.docid, {}))))
        parts.append(Content("\n"))
        parts.append(Content.from_rich_text(meta_lines(m)))
        body = " ".join(str(m["narrative"]).split())
        snippet = body[:SNIPPET_CHARS] + ("..." if len(body) > SNIPPET_CHARS else "")
        parts.append(Content(textwrap.indent(textwrap.fill(snippet, 100), "    ") + "\n\n"))
    s = result.extra.get("settings", {})
    legend = {"bm25": "Score = BM25 term-match relevance.",
              "hyde": "Score = inner product with the HyDE vector (mean of query + hypothetical docs).",
              "hybrid": (f"Score = Pyserini hybrid fusion of BM25 and HyDE (alpha {s.get('fusion_alpha')}, "
                         f"normalization {'on' if s.get('fusion_normalization') else 'off'}).")}[result.mode]
    parts.append(Content.from_markup(
        "[dim italic]$t[/]", t=legend + " Click a header or :open N to read and grade (keys 0-3); :rate N G also grades."))
    return Content.assemble(*parts)


class TextScreen(ModalScreen):
    BINDINGS = [("escape", "app.pop_screen", "Close")]
    CSS = """
    TextScreen { align: center middle; }
    #popup { width: 90%; height: 85%; border: round $accent; background: $surface; padding: 1 2; }
    .popup-hint { color: $text-muted; }
    """

    def __init__(self, title: Text, body: Text):
        super().__init__()
        self.title_text, self.body = title, body

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="popup"):
            yield Static(self.title_text)
            yield Static(self.body)
            yield Static(Text("Esc or click outside to close", style="dim italic"), classes="popup-hint")

    def on_click(self, event):
        if event.widget is self:
            self.app.pop_screen()


class DocScreen(ModalScreen):
    """Full narrative of one result, with grading: keys 0-3 (or click a grade) store a human grade for
    this query and move to the next result; Left/Right step through the results."""
    BINDINGS = [("escape", "app.pop_screen", "Close"), ("left,p", "step(-1)", "Prev"), ("right,n", "step(1)", "Next"),
                ("0", "grade(0)", "0"), ("1", "grade(1)", "1"), ("2", "grade(2)", "2"), ("3", "grade(3)", "3")]
    CSS = """
    DocScreen { align: center middle; }
    #doc { width: 90%; height: 85%; border: round $accent; background: $surface; }
    #doc-head { padding: 1 2 0 2; height: auto; }
    #doc-grades { padding: 0 2; height: auto; border-bottom: solid $accent; }
    #doc-scroll { padding: 1 2; }
    """

    def __init__(self, block: "QueryBlock", n: int):
        super().__init__()
        self.block, self.n = block, n

    def compose(self) -> ComposeResult:
        with Vertical(id="doc"):
            yield Static(id="doc-head")
            yield Static(id="doc-grades")
            with VerticalScroll(id="doc-scroll"):
                yield Static(id="doc-body")

    def on_mount(self):
        self.render_doc()

    @property
    def hits(self):
        return self.block.result.hits

    def render_doc(self, note: str = ""):
        app, h = self.app, self.hits[self.n - 1]
        m = app.retriever.meta.loc[h.docid]
        r = app.store.ratings(self.block.query_id).get(h.docid, {})
        head = Text("Query: ", style="bold cyan").append(self.block.question + "\n")
        head.append(f"[{h.rank}/{len(self.hits)}] complaint {h.docid}  |  {self.block.result.mode} score "
                    f"{h.score:.4f}{rating_tags(r)}\n", style="bold")
        head.append_text(meta_lines(m))
        self.query_one("#doc-head", Static).update(head)
        mine = r.get(HUMAN)
        parts = [Content.from_markup("[bold]Your grade:[/] ")]
        for g, d in GRADES.items():
            style = "reverse bold $success" if g == mine else "bold"
            parts.append(Content.from_markup(f"[@click=screen.grade({g})][{style}] $g [/] $d[/]   ", g=str(g), d=d))
        parts.append(Content.from_markup(
            "\n[dim]0-3 grade and go to next  |  Left/Right or p/n step  |  Esc close[/]  [bold $warning]$note[/]",
            note=note))
        self.query_one("#doc-grades", Static).update(Content.assemble(*parts))
        self.query_one("#doc-body", Static).update(Text(str(m["narrative"])))
        self.query_one("#doc-scroll", VerticalScroll).scroll_home(animate=False)

    def action_step(self, d: int):
        n = self.n + d
        if 1 <= n <= len(self.hits):
            self.n = n
            self.render_doc()

    def action_grade(self, g: int):
        app, h = self.app, self.hits[self.n - 1]
        app.store.rate(self.block.query_id, h.docid, HUMAN, g)
        self.block.refresh_results(app.store.ratings(self.block.query_id))
        note = f"result {h.rank} graded {g} ({GRADES[g]})"
        if self.n < len(self.hits):
            self.n += 1
            note += ", showing next"
        self.render_doc(note)

    def on_click(self, event):
        if event.widget is self:
            self.app.pop_screen()


class SettingsScreen(ModalScreen):
    BINDINGS = [("escape", "cancel", "Cancel")]
    CSS = """
    SettingsScreen { align: center middle; }
    #settings { width: 70; height: auto; border: round $accent; background: $surface; padding: 1 2; }
    #settings-grid { grid-size: 2; grid-columns: 26 1fr; height: auto; grid-rows: 3; }
    #settings-error { color: $error; height: auto; }
    #settings-buttons { height: auto; align-horizontal: right; }
    """

    def __init__(self, mode: str, k: int, alpha: float, normalization: bool):
        super().__init__()
        self.values = (mode, k, alpha, normalization)

    def compose(self) -> ComposeResult:
        mode, k, alpha, norm = self.values
        with Vertical(id="settings"):
            yield Label(Text("Search settings (this session)", style="bold"))
            with Grid(id="settings-grid"):
                yield Label("Mode")
                yield Select([(m, m) for m in MODES], value=mode, allow_blank=False, id="set-mode")
                yield Label("Top-n results")
                yield Input(str(k), type="integer", id="set-k")
                yield Label("Fusion alpha (BM25 weight)")
                yield Input(str(alpha), type="number", id="set-alpha")
                yield Label("Fusion normalization")
                yield Checkbox("min-max center both lists", norm, id="set-norm")
            yield Static("", id="settings-error")
            with Horizontal(id="settings-buttons"):
                yield Button("Cancel", id="settings-cancel")
                yield Button("Apply", variant="primary", id="settings-apply")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "settings-cancel":
            return self.dismiss(None)
        err = self.query_one("#settings-error", Static)
        try:
            k = int(self.query_one("#set-k", Input).value)
            assert 1 <= k <= 1000
        except (ValueError, AssertionError):
            return err.update("Top-n must be an integer from 1 to 1000")
        try:
            alpha = float(self.query_one("#set-alpha", Input).value)
            assert alpha >= 0
        except (ValueError, AssertionError):
            return err.update("Alpha must be a number >= 0")
        self.dismiss({"mode": self.query_one("#set-mode", Select).value, "k": k, "alpha": alpha,
                      "normalization": self.query_one("#set-norm", Checkbox).value})

    def action_cancel(self):
        self.dismiss(None)


class QueryBlock(Vertical):
    def __init__(self, question: str, mode: str, k: int):
        super().__init__(classes="block")
        self.question, self.mode, self.k = question, mode, k
        self.block_id = f"b{next(_ids)}"
        self.query_id = uuid.uuid4().hex[:12]
        self.result, self.meta = None, None

    def compose(self) -> ComposeResult:
        yield Static(Text("Q: ", style="bold cyan") + Text(self.question)
                     + Text(f"   [{self.mode}, top {self.k}, id {self.query_id}]", style="dim"), classes="question")
        with Collapsible(title="Hypothetical documents", collapsed=True, classes="hyde"):
            yield Static("", classes="hyde-body")
        with Collapsible(title="Debug info (running...)", collapsed=True, classes="debug"):
            yield Static("", classes="debug-body")
        yield LoadingIndicator()
        yield Static("", classes="results")

    def show_result(self, result, meta, ratings: dict, logs: list, elapsed: float, error: str | None):
        self.query_one(LoadingIndicator).remove()
        self.query_one(".debug-body", Static).update(Text("\n".join(logs) or "(no log lines)", style="dim"))
        self.query_one(".debug", Collapsible).title = f"Debug info ({len(logs)} log lines, {elapsed:.1f}s)"
        hy = self.query_one(".hyde", Collapsible)
        if error:
            hy.display = False
            self.query_one(".results", Static).update(Text(f"Error: {error}", style="bold red"))
            return
        self.result, self.meta = result, meta
        if result.hyde_record:
            hy.title = f"Hypothetical documents ({len(result.hyde_record['documents'])}, type :hyde to open)"
            self.query_one(".hyde-body", Static).update(hyde_text(result.hyde_record))
        else:
            hy.display = False
        self.refresh_results(ratings)

    def refresh_results(self, ratings: dict):
        if self.result is not None:
            self.query_one(".results", Static).update(render_results(self.result, self.meta, self.block_id, ratings))


def hyde_text(rec: dict) -> Text:
    t = Text()
    for i, d in enumerate(rec["documents"], 1):
        t.append(f"[d{i}]\n", style="bold").append(d.strip() + "\n\n")
    return t


def review_text(searches: list[dict]) -> Text:
    t = Text()
    if not searches:
        return Text("(no saved searches yet)", style="dim")
    for s in searches:
        r = s["ratings"]
        n_h = sum(HUMAN in v for v in r.values())
        n_l = sum(any(k.startswith("llm:") for k in v) for v in r.values())
        t.append(f"{s['ts'][:19]}  ", style="dim").append(s["query"], style="bold cyan")
        t.append(f"\n  id {s['query_id']}  |  {s['mode']}, top {s['k']}  |  rated: {n_h} human, {n_l} llm\n",
                 style="dim")
        for rank, docid, score in s["hits"]:
            tags = rating_tags(r.get(docid, {}))
            if tags:
                t.append(f"    [{rank}] complaint {docid}  score {score:.4f}").append(tags + "\n", style="green")
        t.append("\n")
    return t


class HydeApp(App):
    TITLE = "CFPB Complaint Search"
    BINDINGS = [("ctrl+q", "quit", "Quit"), ("ctrl+s", "settings", "Settings"),
                ("ctrl+g", "toggle_logs", "Logs"), ("ctrl+l", "clear", "Clear history")]
    CSS = """
    #history { padding: 0 1; height: 1fr; }
    .block { height: auto; margin: 1 0; padding: 0 1; border: round $primary; }
    .question { padding: 0 1; }
    .results { padding: 1 1 0 1; border-top: solid $warning; }
    .notice { padding: 0 1; color: $text-muted; }
    Collapsible { margin: 0; }
    LoadingIndicator { height: 3; }
    #bottom { dock: bottom; height: auto; }
    #status { height: 1; padding: 0 2; background: $boost; }
    #question-input { margin: 0 1; }
    #log-panel { height: 12; border-top: solid $accent; display: none; }
    #log-panel.visible { display: block; }
    """

    def __init__(self, retriever: Retriever | None, store: AdminStore, log_handler: AppLogHandler | None = None,
                 cfg=None):
        """retriever=None: indexes are loaded in a background worker after the UI is up."""
        super().__init__()
        self.retriever, self.store, self.log_handler = retriever, store, log_handler
        self.cfg = retriever.cfg if retriever else cfg
        self.ready = threading.Event()
        self.load_state = "loading"  # loading | ready | missing | failed
        self.searching = False
        self.boot_proc: subprocess.Popen | None = None
        self.mode, self.k = self.cfg.retrieval.default_mode, self.cfg.retrieval.top_n
        self.history = QueryHistory()
        self.blocks: dict[str, QueryBlock] = {}
        self.last: QueryBlock | None = None
        self.judge_rater = f"llm:{self.cfg.models.judge}"

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="history")
        yield RichLog(id="log-panel", wrap=True, max_lines=2000)
        with Vertical(id="bottom"):
            yield StatusBar(id="status")
            yield QuestionInput(self.history, placeholder="Describe your situation and press Enter "
                                "(:help for search and admin commands, Up/Down: history, 'exit' to quit)",
                                id="question-input")
        yield Footer()

    def on_mount(self):
        self._update_subtitle()
        if self.log_handler:
            panel = self.query_one("#log-panel", RichLog)
            for line in self.log_handler.lines:
                panel.write(Text(line, style="dim"))
            self.log_handler.app = self
        self.query_one(Input).focus()
        self.load_indexes()

    # --- status bar (safe to call from worker threads) ---
    def _on_ui_thread(self) -> bool:
        return threading.get_ident() == self._thread_id

    def status(self, key: str, msg: str):
        bar = self.query_one(StatusBar)
        bar.set(key, msg) if self._on_ui_thread() else self.call_from_thread(bar.set, key, msg)

    def status_done(self, key: str, msg: str, style: str = "green"):
        bar = self.query_one(StatusBar)
        bar.done(key, msg, style) if self._on_ui_thread() else self.call_from_thread(bar.done, key, msg, style)

    @work(thread=True, group="load")
    def load_indexes(self):
        """Open indexes and pay first-query costs (metadata load, docid hash, first scans) before any search."""
        start = time.time()
        missing = bootstrap.missing_steps()
        if missing:
            self.load_state = "missing"
            msg = (f"Setup needed: {', '.join(missing)} not built. Type :bootstrap to build from data/ "
                   "(embedding takes hours; resumable), or :bootstrap check for details.")
            self.status_done("load", msg, "bold yellow")
            self.call_from_thread(self.call_later, self.notice, msg, "yellow")
            return
        mismatch = bootstrap.index_embedder_mismatch()
        if mismatch:
            self.call_from_thread(self.call_later, self.notice, "Warning: " + mismatch, "yellow")
        self.load_state = "loading"
        try:
            if self.retriever is None:
                self.status("load", "Opening BM25 and Faiss indexes (~12 GB)")
                self.retriever = Retriever(self.cfg)
            self.retriever.warm(lambda m: self.status("load", m))
        except Exception as e:
            logger.exception("index load failed")
            self.load_state = "failed"
            self.status_done("load", f"✗ Index load failed: {e}", "bold red")
            return
        self.load_state = "ready"
        self.ready.set()
        self.status_done("load", f"✓ Ready: {self.retriever.dense.index.ntotal:,} complaints indexed "
                                 f"(loaded in {time.time() - start:.1f}s). Describe a situation and press Enter.")

    def fusion(self) -> dict:
        R = self.cfg["retrieval"]
        return {"fusion_alpha": R["fusion_alpha"], "fusion_normalization": R["fusion_normalization"],
                "fusion_weight_on_dense": R["fusion_weight_on_dense"], "hybrid_k0": R["hybrid_k0"]}

    def _update_subtitle(self):
        f = self.fusion()
        self.sub_title = (f"mode={self.mode} | top-n={self.k} | alpha={f['fusion_alpha']} "
                          f"norm={'on' if f['fusion_normalization'] else 'off'} | gen={self.cfg.models.generator} "
                          f"| judge={self.cfg.models.judge} | provider={self.cfg.provider.name}")

    def write_log(self, line: str):
        self.query_one("#log-panel", RichLog).write(Text(line, style="dim"))

    def action_toggle_logs(self):
        self.query_one("#log-panel", RichLog).toggle_class("visible")

    async def notice(self, msg: str, style: str = ""):
        h = self.query_one("#history", VerticalScroll)
        await h.mount(Static(Text(msg, style=style), classes="notice"))
        h.scroll_end(animate=False)

    async def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if not text:
            return
        if text.lower() == "exit":
            self.exit()
            return
        self.history.add(text)
        event.input.reset_history_position()
        event.input.value = ""
        if text.startswith(":"):
            await self.command(text)
            return
        if self.searching:
            self.notify("A search is still running. Wait for it to finish; :open, :rate and other "
                        "commands still work.", severity="warning")
            return
        self.searching = True
        block = QueryBlock(text, self.mode, self.k)
        self.blocks[block.block_id] = block
        self.last = block
        h = self.query_one("#history", VerticalScroll)
        await h.mount(block)
        h.scroll_end(animate=False)
        self.run_query(block)

    def _last_result(self):
        return self.last.result if self.last and self.last.result else None

    async def command(self, text: str):
        cmd, *args = text[1:].split()
        cmd = cmd.lower()
        res = self._last_result()
        if cmd == "help":
            await self.notice(HELP)
        elif cmd == "mode" and args and args[0] in MODES:
            self.mode = args[0]
            self._update_subtitle()
            await self.notice(f"mode set to {self.mode}")
        elif cmd == "k" and args and args[0].isdigit() and 1 <= int(args[0]) <= 1000:
            self.k = int(args[0])
            self._update_subtitle()
            await self.notice(f"top-n set to {self.k}")
        elif cmd == "open" and args and args[0].isdigit():
            if not res:
                return await self.notice("no results yet", "red")
            self.show_doc(self.last.block_id, int(args[0]))
        elif cmd == "hyde":
            rec = res.hyde_record if res else None
            if not rec:
                return await self.notice("last query has no hypothetical documents (bm25 mode or no query yet)", "red")
            self.push_screen(TextScreen(Text(f"Hypothetical documents for: {rec['query']}", style="bold"),
                                        hyde_text(rec)))
        elif cmd == "rate" and len(args) == 2 and args[0].isdigit() and args[1] in ("0", "1", "2", "3"):
            if not res:
                return await self.notice("no results to rate yet", "red")
            n, g = int(args[0]), int(args[1])
            if not 1 <= n <= len(res.hits):
                return await self.notice(f"no result {n} (last query has {len(res.hits)})", "red")
            self.store.rate(self.last.query_id, res.hits[n - 1].docid, HUMAN, g)
            self.last.refresh_results(self.store.ratings(self.last.query_id))
            await self.notice(f"rated result {n} (complaint {res.hits[n - 1].docid}) = {g} ({GRADES[g]})")
        elif cmd == "judge":
            if not res:
                return await self.notice("no results to judge yet", "red")
            await self.notice(f"judging {len(res.hits)} results with {self.cfg.models.judge} ...")
            self.run_judge(self.last)
        elif cmd == "bootstrap" and args[:1] in ([], ["check"]):
            if self.boot_proc and self.boot_proc.poll() is None:
                return await self.notice("bootstrap is already running (:bootstrap stop to stop it)", "red")
            self.run_bootstrap(check=bool(args))
        elif cmd == "bootstrap" and args == ["stop"]:
            if not (self.boot_proc and self.boot_proc.poll() is None):
                return await self.notice("bootstrap is not running", "red")
            self.boot_proc.terminate()
            await self.notice("stopping bootstrap; finished work is kept and :bootstrap resumes it")
        elif cmd == "review":
            self.push_screen(TextScreen(Text("Saved searches and ratings (newest first)", style="bold"),
                                        review_text(self.store.review())))
        else:
            await self.notice(f"unknown or malformed command: {text}\n{HELP}", "red")

    @work(thread=True, group="query")
    def run_query(self, block: QueryBlock):
        capture = LogCapture()
        capture.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
        root = logging.getLogger()
        root.addHandler(capture)
        settings = self.fusion()
        logger.info("query_id=%s mode=%s top-n=%d %s query=%r", block.query_id, block.mode, block.k, settings,
                    block.question)
        start, result, error, meta, ratings = time.time(), None, None, None, {}
        say = lambda m: self.status("query", m)  # noqa: E731
        try:
            if not self.ready.is_set():
                say("Waiting for indexes to finish loading")
                while not self.ready.wait(0.2):
                    if self.load_state in ("missing", "failed"):
                        raise RuntimeError("indexes are not available. Run :bootstrap to build them "
                                           "(or :bootstrap check to see what is missing)")
            if self.retriever is None:
                raise RuntimeError("indexes failed to load (see Ctrl+G log)")
            result = self.retriever.search(block.question, block.mode, block.k, query_id=block.query_id,
                                           on_stage=say)
            say("Saving search and rendering results")
            result.extra["settings"] = settings
            self.store.log_search(block.query_id, block.question, block.mode, block.k, settings, result.hits)
            meta = self.retriever.meta
            ratings = self.store.ratings(block.query_id)
        except Exception as e:
            logger.exception("query failed")
            error = str(e)
        finally:
            root.removeHandler(capture)
        elapsed = time.time() - start
        self.call_from_thread(block.show_result, result, meta, ratings, capture.lines, elapsed, error)
        self.call_from_thread(self._search_finished)
        if error:
            self.status_done("query", f"✗ Search failed after {elapsed:.1f}s: {error}", "bold red")
        else:
            self.status_done("query", f"✓ {block.mode} search finished: {len(result.hits)} results in "
                                      f"{elapsed:.1f}s. :open N to read, :rate N G to grade, :judge to LLM-judge.")

    @work(thread=True, group="judge")
    def run_judge(self, block: QueryBlock):
        """UMBRELA judge (query + narrative only) on every result of `block`, concurrently."""
        hits, q = block.result.hits, block.question

        n, done, lock = len(hits), [0], threading.Lock()
        label = f"Judging {n} results with {self.cfg.models.judge}"
        self.status("judge", f"{label} (0/{n} done)")

        def one(h):
            r = umbrela.judge(q, str(self.retriever.meta.loc[h.docid, "narrative"]), self.cfg)
            self.store.rate(block.query_id, h.docid, self.judge_rater, r["grade"],
                            {k: r.get(k) for k in ("result_status", "truncated", "error", "output")})
            self.call_from_thread(block.refresh_results, self.store.ratings(block.query_id))
            with lock:
                done[0] += 1
                self.status("judge", f"{label} ({done[0]}/{n} done)")
            return r

        with ThreadPoolExecutor(self.cfg.admin.judge_workers) as ex:
            rs = list(ex.map(one, hits))
        bad = sum(bool(not r["result_status"] or r.get("truncated") or r.get("error")) for r in rs)
        msg = f"judged {len(rs)} results for query {block.query_id}" + (f" ({bad} invalid/truncated)" if bad else "")
        self.call_from_thread(self.call_later, self.notice, msg, "red" if bad else "")
        self.status_done("judge", ("✗ " if bad else "✓ ") + msg, "yellow" if bad else "green")

    @work(thread=True, group="bootstrap")
    def run_bootstrap(self, check: bool = False):
        """Run `main.py bootstrap` as a subprocess (keeps heavy build work off the UI process), streaming
        "[k/N] ..." progress to the status bar and all output to the log panel (Ctrl+G)."""
        cmd = [sys.executable, str(ROOT / "main.py"), "bootstrap"] + (["--check"] if check else [])
        self.status("bootstrap", "Bootstrap: checking" if check else "Bootstrap: starting")
        self.boot_proc = p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True, bufsize=1)
        assert p.stdout is not None
        lines, last = [], ""
        for line in p.stdout:
            line = line.rstrip()
            lines.append(line)
            self.call_from_thread(self.write_log, line)
            if line.startswith("[") or line.startswith(("READY", "NOT READY", "ERROR")):
                last = line
                self.status("bootstrap", "Bootstrap " + line)
        rc = p.wait()
        if check:
            body = Text("\n".join(x for x in lines if not x.startswith(("W1", "WARNING"))))
            self.call_from_thread(self.push_screen, TextScreen(Text("Bootstrap check", style="bold"), body))
            self.status_done("bootstrap", last or f"Bootstrap check exit {rc}", "green" if rc == 0 else "yellow")
        elif rc == 0:
            self.status_done("bootstrap", "✓ Bootstrap finished. Loading indexes ...")
            if self.load_state != "ready":
                self.call_from_thread(self.load_indexes)
        else:
            msg = (f"✗ Bootstrap stopped (exit {rc}). {last}. Ctrl+G shows the full log; "
                   ":bootstrap resumes from finished steps.")
            self.status_done("bootstrap", msg, "bold red")
            self.call_from_thread(self.call_later, self.notice, msg, "red")

    def on_unmount(self):
        if self.boot_proc and self.boot_proc.poll() is None:
            self.boot_proc.terminate()  # embedding checkpoints per chunk; :bootstrap resumes

    def _search_finished(self):
        self.searching = False
        self.query_one("#history", VerticalScroll).scroll_end(animate=False)

    def show_doc(self, block_id: str, n: int):
        block = self.blocks.get(block_id)
        hits = block.result.hits if block and block.result else []
        if not 1 <= n <= len(hits):
            self.call_later(self.notice, f"no result {n} (last query has {len(hits)})", "red")
            return
        self.push_screen(DocScreen(block, n))

    action_show_doc = show_doc

    def action_settings(self):
        f = self.fusion()

        def apply(res):
            if res:
                self.mode, self.k = res["mode"], res["k"]
                self.cfg["retrieval"]["fusion_alpha"] = res["alpha"]
                self.cfg["retrieval"]["fusion_normalization"] = res["normalization"]
                self._update_subtitle()
        self.push_screen(SettingsScreen(self.mode, self.k, f["fusion_alpha"], f["fusion_normalization"]), apply)

    def action_clear(self):
        self.query_one("#history", VerticalScroll).remove_children()
        self.blocks.clear()
        self.last = None


def run_tui():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    handler = AppLogHandler()
    root.addHandler(handler)
    cfg = load()
    HydeApp(None, AdminStore(path(cfg.paths.admin_db)), handler, cfg=cfg).run()
