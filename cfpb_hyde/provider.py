"""Model provider client: any OpenAI-compatible server (config.yaml `provider:`).

Default is LM Studio. Others (llama.cpp server, Ollama, vLLM, ...) work if they serve /v1/models,
/v1/chat/completions and /v1/embeddings. Provider-specific request fields go in `provider.chat_extra`.
"""
import logging
import os
import time

import numpy as np
import requests

from cfpb_hyde.config import load

_cfg = load()
P = _cfg.provider
log = logging.getLogger(__name__)
_warned_norm = False


def headers() -> dict:
    key = os.environ.get(P.api_key_env) if P.get("api_key_env") else None
    return {"Authorization": f"Bearer {key}"} if key else {}


def chat_extra() -> dict:
    return dict(P.get("chat_extra") or {})


def post(endpoint: str, body: dict) -> requests.Response:
    return requests.post(f"{P.base_url}/{endpoint}", json=body, headers=headers(), timeout=P.timeout_s)


def list_models() -> set[str]:
    r = requests.get(f"{P.base_url}/models", headers=headers(), timeout=10)
    r.raise_for_status()
    return {m["id"] for m in r.json()["data"]}


def embed(texts: list[str], model: str | None = None, retries: int = 3) -> np.ndarray:
    """POST /embeddings; float32 (len(texts), dim), rows ordered by response `index`.
    LM Studio returns unit-norm vectors (DECISIONS N2). Other servers may not, and the Faiss index
    is inner product, so non-unit rows are L2-normalized here (a no-op for LM Studio)."""
    global _warned_norm
    body = {"model": model or _cfg.models.embedder, "input": texts}
    for attempt in range(retries):
        try:
            r = post("embeddings", body)
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            if len(data) != len(texts):
                raise ValueError(f"expected {len(texts)} embeddings, got {len(data)}")
            v = np.asarray([d["embedding"] for d in data], dtype=np.float32)
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            if np.abs(norms - 1).max() > 1e-3:
                if not _warned_norm:
                    log.warning("provider returned non-unit embeddings; L2-normalizing client-side")
                    _warned_norm = True
                v = v / np.maximum(norms, 1e-12)
            return v
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def chat(messages: list[dict], model: str | None = None, **params) -> dict:
    """POST /chat/completions with `provider.chat_extra` merged in; returns the raw JSON response."""
    body = {"model": model or _cfg.models.generator, "messages": messages, **chat_extra(), **params}
    r = post("chat/completions", body)
    r.raise_for_status()
    return r.json()
