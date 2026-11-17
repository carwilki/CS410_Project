"""Defensive removal of Gemma 4 thinking blocks.

Reasoning is disabled at request time (reasoning_effort="none", DECISIONS R5). This only guards
against leaks, using the Google-documented delimiters <|channel>thought ... <channel|> (R2).
"""
import re

_BLOCK = re.compile(r"<\|channel>.*?<channel\|>", re.S)
_UNCLOSED = re.compile(r"<\|channel>.*\Z", re.S)


def strip_reasoning(text: str) -> tuple[str, str]:
    """Return (cleaned_text, removed_reasoning)."""
    removed = _BLOCK.findall(text)
    text = _BLOCK.sub("", text)
    tail = _UNCLOSED.search(text)  # thinking cut off by max_tokens: everything after the opener is reasoning
    if tail:
        removed.append(tail.group(0))
        text = text[:tail.start()]
    return text.strip(), "\n".join(removed)
