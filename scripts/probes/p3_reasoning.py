"""Phase 0.3: generate 5 outputs with Gemma 4 E4B (no <|think|>) and check for reasoning text."""
import json
import re
import sys

import requests

BASE = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"
SYSTEM = "You write consumer complaint narratives in the style of CFPB submissions."
PROMPT = ("Please write a consumer complaint narrative, in the style of a CFPB submission, "
          "describing the situation.\nSituation: {}\nNarrative:")
QUERIES = [
    "a debt collector keeps calling me about a debt I already paid",
    "my credit report shows an account that isn't mine",
    "the bank charged me overdraft fees after reordering my transactions",
    "my mortgage servicer lost my loan modification paperwork",
    "someone used my identity to open a credit card",
]
MARKERS = re.compile(r"<\|?channel\|?>|<\|think\|>|<think>|</think>|\bthought\n", re.I)

results = []
for q in QUERIES:
    r = requests.post(f"{BASE}/chat/completions", timeout=300, json={
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT.format(q)}],
        "temperature": 0.7, "max_tokens": 512, "top_p": 1.0, "stop": ["\n\n\n"],
    })
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    results.append({"query": q, "message_keys": sorted(msg), "finish_reason": r.json()["choices"][0]["finish_reason"],
                    "reasoning_field_len": len(reasoning), "marker_hits": MARKERS.findall(content),
                    "content_head": content[:300]})
    print(json.dumps(results[-1]), file=sys.stderr)
print(json.dumps({"all_clean": all(not x["marker_hits"] and not x["reasoning_field_len"] for x in results),
                  "results": results}, indent=1))
