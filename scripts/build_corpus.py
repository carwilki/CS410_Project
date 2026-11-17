"""Phase 1a: CFPB zips -> metadata/narrative parquet (side store) + Pyserini JsonCollection.

Also records each narrative's nomic token count (with document prefix) for truncation logging.
"""
import json
import zipfile

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from cfpb_hyde.config import load, path

cfg = load()
RENAME = {
    "Complaint ID": "id", "Date received": "date", "Product": "product", "Sub-product": "sub_product",
    "Issue": "issue", "Sub-issue": "sub_issue", "Company": "company",
    "Company response to consumer": "company_response", "Company public response": "public_response",
    "State": "state", "Tags": "tags", "Submitted via": "submitted_via", "Timely response?": "timely",
    "Consumer complaint narrative": "narrative",
}

frames = []
for zp in sorted(path(cfg.paths.data_dir).glob("*.zip")):
    with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=list(RENAME))
    df = df[df["Consumer complaint narrative"].str.strip().ne("")].rename(columns=RENAME)
    frames.append(df)
    print(zp.name, len(df), flush=True)
df = pd.concat(frames, ignore_index=True)
df = df.sort_values("id", key=lambda s: s.astype("int64"), kind="stable").reset_index(drop=True)
assert df["id"].is_unique

tok = Tokenizer.from_file(hf_hub_download(cfg.embedding.tokenizer_repo, "tokenizer.json"))
tok.no_truncation()
prefix = cfg.embedding.document_prefix
lens = np.empty(len(df), dtype=np.int32)
B = 20_000
for i in range(0, len(df), B):
    enc = tok.encode_batch([prefix + t for t in df["narrative"].iloc[i:i + B]])
    lens[i:i + B] = [len(e.ids) for e in enc]
df["n_tokens"] = lens

out = path(cfg.paths.corpus_parquet)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)

jc = path(cfg.paths.json_collection)
jc.mkdir(parents=True, exist_ok=True)
SHARD = 500_000
for s, i in enumerate(range(0, len(df), SHARD)):
    with open(jc / f"docs{s:02d}.jsonl", "w") as f:
        for did, txt in zip(df["id"].iloc[i:i + SHARD], df["narrative"].iloc[i:i + SHARD]):
            f.write(json.dumps({"id": did, "contents": txt}) + "\n")

ctx = cfg.embedding.context_tokens
pct = {p: int(np.percentile(lens, p)) for p in (50, 75, 90, 95, 99)}
print(json.dumps({"N": len(df), "token_percentiles": pct, "max": int(lens.max()),
                  "over_ctx": int((lens > ctx).sum()), "over_ctx_share": float((lens > ctx).mean()),
                  "over_512": float((lens > 512).mean())}, indent=1))
