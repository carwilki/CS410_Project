"""Admin store (SQLite): every TUI search with its ranked results, plus human and LLM relevance ratings."""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    query_id TEXT PRIMARY KEY, ts TEXT, query TEXT, mode TEXT, k INTEGER, settings TEXT);
CREATE TABLE IF NOT EXISTS results (
    query_id TEXT, rank INTEGER, docid TEXT, score REAL, PRIMARY KEY (query_id, rank));
CREATE TABLE IF NOT EXISTS ratings (
    query_id TEXT, docid TEXT, rater TEXT, grade INTEGER, ts TEXT, detail TEXT,
    PRIMARY KEY (query_id, docid, rater));
"""
GRADES = {0: "not relevant", 1: "related, different situation", 2: "similar situation", 3: "same situation"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AdminStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()  # one connection shared by the UI and worker threads

    def log_search(self, query_id: str, query: str, mode: str, k: int, settings: dict, hits) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO searches VALUES (?,?,?,?,?,?)",
                            (query_id, now(), query, mode, k, json.dumps(settings)))
            self.db.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?)",
                                [(query_id, h.rank, h.docid, h.score) for h in hits])

    def rate(self, query_id: str, docid: str, rater: str, grade: int, detail: dict | None = None) -> None:
        if grade not in GRADES:
            raise ValueError("grade must be 0-3")
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?,?)",
                            (query_id, docid, rater, grade, now(), json.dumps(detail or {})))

    def ratings(self, query_id: str) -> dict[str, dict[str, int]]:
        """{docid: {rater: grade}} for one search."""
        out: dict[str, dict[str, int]] = {}
        with self.lock:
            rows = self.db.execute("SELECT docid, rater, grade FROM ratings WHERE query_id=?", (query_id,)).fetchall()
        for d, r, g in rows:
            out.setdefault(d, {})[r] = g
        return out

    def review(self, limit: int = 50) -> list[dict]:
        """Recent searches, newest first, with results and ratings."""
        with self.lock:
            searches = self.db.execute("SELECT query_id, ts, query, mode, k, settings FROM searches "
                                       "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for qid, ts, q, mode, k, settings in searches:
                hits = self.db.execute("SELECT rank, docid, score FROM results WHERE query_id=? ORDER BY rank",
                                       (qid,)).fetchall()
                out.append({"query_id": qid, "ts": ts, "query": q, "mode": mode, "k": k,
                            "settings": json.loads(settings), "hits": hits})
        for s in out:
            s["ratings"] = self.ratings(s["query_id"])
        return out
