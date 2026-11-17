"""UMBRELA judge (castorini/umbrela @ e2c4e4126bf7b568d9230008b465a58445cabc30), adapted for a local OpenAI-compatible provider.

Verbatim from upstream:
  UPSTREAM_PROMPT = src/umbrela/prompts/prompt_templates/qrel_zeroshot_bing.yaml:4-20 (prefix_user; system_message "" at :2).
            Default prompt_type "bing" (gpt_judge.py:20, osllm_judge.py:18); zero-shot when few_shot_count=0
            (template_loader.py:114-119).
  SYSTEM  = gpt_judge.py:277 (messages built in run_gpt, :276-279).
  params  = gpt_judge.py:178,185-193 (temperature 0.0, top_p 1, frequency_penalty 0.5, presence_penalty 0,
            max_tokens = max_new_tokens) and judge() default max_new_tokens=100 (gpt_judge.py:331).
  output lowercased: gpt_judge.py:293.
  parse_fewshot_response: src/umbrela/utils/common_utils.py:63-113 (patterns copied unchanged).
  parse source = output + "\n" + reasoning when reasoning exists: common_utils.py:141-144.
  HTTP 4xx -> "" (-> grade 0, result_status 0): gpt_judge.py run_gpt `except self._bad_request_error`;
  other errors retried 3x.
Adaptations (ours): PROMPT = UPSTREAM_PROMPT with PROMPT_EDITS (level wording); max_tokens from config; provider.chat_extra (LM Studio: reasoning_effort="none", DECISIONS R5); Gemma thought blocks stripped from content with
cfpb_hyde.reasoning.strip_reasoning and treated as `reasoning`. The judge sees only query + narrative.
"""
import re
import time

import requests  # noqa: F401  (tests patch umbrela.requests.post)

from cfpb_hyde import provider
from cfpb_hyde.config import load
from cfpb_hyde.reasoning import strip_reasoning

UMBRELA_REPO = "castorini/umbrela"
UMBRELA_SHA = "e2c4e4126bf7b568d9230008b465a58445cabc30"

UPSTREAM_PROMPT = (
    "Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n"
    "0 = represent that the passage has nothing to do with the query, \n"
    "1 = represents that the passage seems related to the query but does not answer it, \n"
    "2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, "
    "or hidden amongst extraneous information and \n"
    "3 = represents that the passage is dedicated to the query and contains the exact answer.\n"
    "\n"
    "Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, "
    "category 2 if passage presents something very important related to the entire topic but also has some extra "
    "information and category 3 if the passage only and entirely refers to the topic. If none of the above "
    "satisfies give it category 0.\n"
    "\n"
    "Query: {query}\n"
    "Passage: {passage}\n"
    "\n"
    "Split this problem into steps:\n"
    "Consider the underlying intent of the search.\n"
    "Measure how well the content matches a likely intent of the query (M).\n"
    "Measure how trustworthy the passage is (T).\n"
    "Consider the aspects above and the relative importance of each, and decide on a final score (O). "
    "Final score must be an integer value only.\n"
    "Do not provide any code in result. Provide each score in the format of: ##final score: score without "
    "providing any reasoning."
)
# Deviation (user decision 2026-11-15, DECISIONS V8): the upstream scale asks whether the passage *answers*
# the query; our queries describe a situation and relevant passages are complaints about the same situation.
# Only the three level definitions are reworded; scale, instructions, steps and output format are unchanged.
PROMPT_EDITS = [
    ("seems related to the query but does not answer it",
     "seems related to the query but does not describe the same situation"),
    ("has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information",
     "describes a situation similar to the query, but the match may be partial, or hidden amongst extraneous "
     "information"),
    ("is dedicated to the query and contains the exact answer",
     "is dedicated to the query and describes the same situation"),
]
PROMPT = UPSTREAM_PROMPT
for _old, _new in PROMPT_EDITS:
    assert PROMPT.count(_old) == 1, _old
    PROMPT = PROMPT.replace(_old, _new)
SYSTEM = "You are a helpful assistant."
UMBRELA_PARAMS = {"temperature": 0.0, "top_p": 1, "frequency_penalty": 0.5, "presence_penalty": 0, "max_tokens": 100}


def messages(query: str, passage: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(query=query, passage=passage)}]


def parse_fewshot_response(response: str) -> tuple[int, int]:
    """Verbatim logic of umbrela common_utils.parse_fewshot_response -> (grade, valid)."""
    response = response.strip().lower()
    valid_res = 1
    answer = ""
    patterns = [
        r'"o"\s*[:-=]?\s*(0|1|2|3)',
        r"\'o\'\s*[:-=]?\s*(0|1|2|3)",
        r"o\s*[:-=]?\s*(0|1|2|3)",
        r'"overall_score"\s*[:-=]?\s*(0|1|2|3)',
        r'"overall"\s*[:-=]?\s*(0|1|2|3)',
        r'"overall score"\s*[:-=]?\s*(0|1|2|3)',
        r'"final score"\s*[:-=]?\s*(0|1|2|3)',
        r"final score\s*[:-=]?\s*(0|1|2|3)",
        r"final score is (0|1|2|3)",
        r'"final_score"\s*[:-=]?\s*(0|1|2|3)',
        r'"score"\s*[:-=]?\s*(0|1|2|3)',
        r'"o_score"\s*[:-=]?\s*(0|1|2|3)',
        r"output score is (0|1|2|3)",
        r"score is (0|1|2|3)",
        r"score of\s+(0|1|2|3)",
        r"assign(?:ing)?\s+(?:it\s+)?(?:a\s+)?score of\s+(0|1|2|3)",
        r"conclude(?:s|d)?\s+with\s+(?:a\s+)?score of\s+(0|1|2|3)",
        r"give(?:n)?\s+(?:it\s+)?(?:a\s+)?score of\s+(0|1|2|3)",
        r"i (?:would )?(?:give|assign)\s+(?:it\s+)?(?:a\s+)?score of\s+(0|1|2|3)",
        r"score:\s*(0|1|2|3)",
        r"[a-zA-Z]+\s+is\s+(0|1|2|3)\s",
        r"relevance category\s*[:-=]?\s*(0|1|2|3)",
        r"relevance category\s*[:-=]?\s*(0|1|2|3)",
        r"relevance category is (0|1|2|3)",
        r"it falls into the category (0|1|2|3)",
        r"category\s*(0|1|2|3)",
        r"relevance category (0|1|2|3)",
        r"relevance category for this passage would be (0|1|2|3)",
        r"the relevance category would be (0|1|2|3)",
        r"\n*(0|1|2|3)",
    ]
    for pattern in patterns:
        matched = None
        for m in re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            matched = m
        if matched:
            answer = matched.group(1).capitalize()
            break
    if answer == "":
        return 0, 0
    return int(answer), valid_res


def parse_source(output: str, reasoning: str | None) -> str:
    """common_utils.prepare_judgments:141-143."""
    return f"{output}\n{reasoning}".strip() if reasoning else output


def judge(query: str, passage: str, cfg=None, retries: int = 3) -> dict:
    """One provider call. Returns a JSON-serializable record (raw response fields + grade)."""
    cfg = cfg or load()
    params = {**UMBRELA_PARAMS, "max_tokens": cfg.eval.get("judge_max_tokens", UMBRELA_PARAMS["max_tokens"])}
    body = {"model": cfg.models.judge, "messages": messages(query, passage), **params,
            **provider.chat_extra()}
    rec = {"model": cfg.models.judge, "params": {k: v for k, v in body.items() if k != "messages"},
           "raw_content": "", "reasoning_content": "", "stripped_reasoning": "", "finish_reason": None,
           "usage": None, "error": None}
    for attempt in range(retries):
        try:
            r = provider.post("chat/completions", body)
            if 400 <= r.status_code < 500:  # UMBRELA: bad request -> "" (scored 0, invalid)
                rec["error"] = f"HTTP {r.status_code}: {r.text[:500]}"
                break
            r.raise_for_status()
            j = r.json()
            c = j["choices"][0]
            rec.update(raw_content=c["message"].get("content") or "",
                       reasoning_content=c["message"].get("reasoning_content") or "",
                       finish_reason=c.get("finish_reason"), usage=j.get("usage"))
            rec["error"] = None
            break
        except Exception as e:  # noqa: BLE001 - mirror UMBRELA's retry-then-raise
            rec["error"] = repr(e)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    clean, stripped = strip_reasoning(rec["raw_content"])
    rec["stripped_reasoning"] = stripped
    output = clean.lower()
    reasoning = "\n".join(x for x in (rec["reasoning_content"], stripped) if x) or None
    rec["output"] = output
    rec["truncated"] = rec["finish_reason"] == "length"
    rec["grade"], rec["result_status"] = parse_fewshot_response(parse_source(output, reasoning))
    return rec
