"""HyDE query path: generate N hypothetical complaints, embed, average (Eq. 3), log."""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cfpb_hyde.config import load, path
from cfpb_hyde.embed import doc_text, hyde_vector, query_text
from cfpb_hyde.provider import chat, chat_extra, embed
from cfpb_hyde.reasoning import strip_reasoning

cfg = load()
H = cfg.hyde

# Adapted from HyDE promptor.py templates (DECISIONS H1): same "Please write a ... \n<Field>: {}\n<Output>:" shape.
PROMPT = ("Please write a consumer complaint narrative, in the style of a CFPB submission, "
          "describing the situation.\nSituation: {}\nNarrative:")
# Style constraints in the system turn (deviation recorded in DECISIONS: Gemma otherwise writes form letters).
SYSTEM = ("You write consumer complaint narratives as submitted to the Consumer Financial Protection Bureau. "
          "Write only the narrative itself: first person, plain prose paragraphs, no title, no headings, "
          "no bullet points, no bracketed placeholders, no notes to the reader.")
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gen_params() -> dict:
    return {"temperature": H.temperature, "top_p": H.top_p, "max_tokens": H.max_tokens,
            "stop": list(H.stop)}


def logged_params() -> dict:
    """What was actually sent: generation params + provider chat_extra (chat() merges the latter)."""
    return {**chat_extra(), **gen_params()}


Stage = Callable[[str], None] | None


def generate(q: str, on_stage: Stage = None) -> list[dict]:
    """N independent requests (LM Studio ignores `n`, DECISIONS G2). Output order = request order."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT.format(q)}]
    resps: list = [None] * H.n_docs
    if on_stage:
        on_stage(f"Generating hypothetical documents with {cfg.models.generator} (0/{H.n_docs} done)")
    with ThreadPoolExecutor(H.n_docs) as ex:
        futs = {ex.submit(chat, msgs, **gen_params()): i for i in range(H.n_docs)}
        for done, f in enumerate(as_completed(futs), 1):
            resps[futs[f]] = f.result()
            if on_stage:
                on_stage(f"Generating hypothetical documents with {cfg.models.generator} ({done}/{H.n_docs} done)")
    out = []
    for r in resps:
        c = r["choices"][0]
        raw = c["message"].get("content") or ""
        clean, stripped = strip_reasoning(raw)
        out.append({"raw": raw, "reasoning_content": c["message"].get("reasoning_content") or "",
                    "stripped_reasoning": stripped, "clean": clean, "finish_reason": c.get("finish_reason")})
    return out


def write_log(log_dir: Path, query_id: str, record: dict) -> Path:
    p = log_dir / RUN_ID / f"{query_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=1))
    return p


def hyde(q: str, query_id: str | None = None, log_dir: Path | None = None,
         on_stage: Stage = None) -> tuple[np.ndarray, dict]:
    """Return (HyDE vector shape (1, dim), log record). Hypothetical docs are only logged."""
    query_id = query_id or uuid.uuid4().hex[:12]
    gens = generate(q, on_stage)
    docs = [g["clean"] for g in gens]
    if on_stage:
        on_stage(f"Embedding query + {len(docs)} hypothetical documents with {cfg.models.embedder}")
    vecs = embed([query_text(q)] + [doc_text(d) for d in docs])
    v = hyde_vector(vecs[0], vecs[1:])
    record = {
        "query_id": query_id, "query": q, "timestamp": datetime.now(timezone.utc).isoformat(),
        "generator_model": cfg.models.generator, "system_prompt": SYSTEM, "user_prompt": PROMPT.format(q),
        "generation_params": logged_params(), "provider": cfg.provider.base_url, "n_docs": H.n_docs, "generations": gens, "documents": docs,
        "embedding_model": cfg.models.embedder, "embedding_dim": int(v.shape[1]),
        "query_prefix": cfg.embedding.query_prefix, "document_prefix": cfg.embedding.document_prefix,
    }
    write_log(log_dir or path(cfg.paths.hyde_logs), query_id, record)
    return v, record
