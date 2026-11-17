"""Phase 0.4-0.6: embedder dims/normalization, token-length distribution, throughput, memory."""
import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://localhost:1234/v1"
EMB = "text-embedding-nomic-embed-text-v1.5"
LOADED_CTX = 2048  # from `lms ps` / GET /api/v0/models (loaded_context_length), checked 2026-11-15
N_TOTAL = 3_851_415  # from p1_data.json
SEED = 410

# --- sample 10k narratives (reservoir over all files, seeded)
rng = random.Random(SEED)
sample, seen = [], 0
for zp in sorted((ROOT / "data").glob("*.zip")):
    with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
        s = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["Consumer complaint narrative"]).iloc[:, 0]
    for t in s[s.str.strip().ne("")]:
        seen += 1
        if len(sample) < 10_000:
            sample.append(t)
        else:
            j = rng.randrange(seen)
            if j < 10_000:
                sample[j] = t
assert seen == N_TOTAL, seen

# --- token lengths with nomic's own tokenizer, prefix included, special tokens included
tok = Tokenizer.from_file(hf_hub_download("nomic-ai/nomic-embed-text-v1.5", "tokenizer.json"))
tok.no_truncation()
lens = np.array([len(e.ids) for e in tok.encode_batch(["search_document: " + t for t in sample])])
dist = {p: int(np.percentile(lens, p)) for p in (50, 95, 99)}
dist.update(max=int(lens.max()), mean=float(lens.mean()),
            share_over_loaded_ctx=float((lens > LOADED_CTX).mean()), loaded_ctx=LOADED_CTX,
            share_over_8192=float((lens > 8192).mean()))

# --- dims + norm
v = requests.post(f"{BASE}/embeddings", json={"model": EMB, "input": "search_query: test"}, timeout=60).json()
vec = v["data"][0]["embedding"]
probe = {"dim": len(vec), "l2_norm": math.sqrt(sum(x * x for x in vec)), "model_field": v.get("model")}

# --- does an over-length input error or silently truncate?
longest = sample[int(lens.argmax())]
r = requests.post(f"{BASE}/embeddings", json={"model": EMB, "input": "search_document: " + longest}, timeout=120)
probe["overlength_status"] = r.status_code
probe["overlength_body_head"] = r.text[:300]

# --- throughput on 1,000 narratives
docs = ["search_document: " + t for t in sample[:1000]]
thr = {}
for bs in (16, 32, 64, 128):
    t0, fails = time.time(), 0
    for i in range(0, len(docs), bs):
        rr = requests.post(f"{BASE}/embeddings", json={"model": EMB, "input": docs[i:i + bs]}, timeout=600)
        if rr.status_code != 200:
            fails += 1
    dt = time.time() - t0
    thr[bs] = {"sec": round(dt, 2), "docs_per_sec": round(1000 / dt, 1),
               "projected_hours_full": round(N_TOTAL / (1000 / dt) / 3600, 2), "failed_batches": fails}
    print(bs, thr[bs], flush=True)

mem = {"N": N_TOTAL, "flat_index_bytes": N_TOTAL * 768 * 4,
       "flat_index_GiB": round(N_TOTAL * 768 * 4 / 2**30, 2), "ram_GiB": 64}
json.dump({"token_lengths": dist, "embed_probe": probe, "throughput": thr, "memory": mem},
          open(ROOT / "docs/probe_out/p4_embed.json", "w"), indent=1)
print(json.dumps({"token_lengths": dist, "embed_probe": probe, "memory": mem}, indent=1))
