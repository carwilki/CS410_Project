"""Phase 0.3b: compare candidate thinking-off request params against a baseline that does reason."""
import json

import requests

BASE = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"
PROMPT = ("Please write a consumer complaint narrative, in the style of a CFPB submission, "
          "describing the situation.\nSituation: {}\nNarrative:")
QUERIES = [
    "a debt collector keeps calling me about a debt I already paid",
    "my credit report shows an account that isn't mine",
    "the bank charged me overdraft fees after reordering my transactions",
    "my mortgage servicer lost my loan modification paperwork",
    "someone used my identity to open a credit card",
]
VARIANTS = {
    "baseline": {},
    "chat_template_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
    "reasoning_effort_none": {"reasoning_effort": "none"},
    "reasoning_obj_none": {"reasoning": {"effort": "none"}},
}
out = {}
for name, extra in VARIANTS.items():
    rows = []
    for q in QUERIES:
        body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(q)}],
                "temperature": 0.7, "top_p": 1.0, "max_tokens": 512, **extra}
        r = requests.post(f"{BASE}/chat/completions", json=body, timeout=600).json()
        if "error" in r:
            rows.append({"error": r["error"]}); continue
        c = r["choices"][0]
        m = c["message"]
        rows.append({"reasoning_len": len(m.get("reasoning_content") or ""),
                     "content_len": len(m.get("content") or ""), "finish": c["finish_reason"],
                     "usage": r.get("usage")})
    out[name] = rows
    print(name, [(x.get("reasoning_len"), x.get("finish")) for x in rows], flush=True)
json.dump(out, open("docs/probe_out/p3b_reasoning_params.json", "w"), indent=1)
