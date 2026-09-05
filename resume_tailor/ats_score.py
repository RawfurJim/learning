"""ATS keyword coverage: what share of the JD's keywords the CV text already contains.

Pure Python. Matching is case-insensitive, respects word boundaries ("Java" does not
match "JavaScript") and is multiword aware: whitespace, hyphens and underscores
inside a keyword are interchangeable, so "fine-tuning" matches "Fine Tuning" and
"vector databases" matches "Vector  Databases".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

_SEPARATOR = re.compile(r"[\s_-]+")


@lru_cache(maxsize=4096)
def _pattern(keyword: str) -> re.Pattern[str] | None:
    parts = [re.escape(part) for part in _SEPARATOR.split(keyword.strip()) if part]
    if not parts:
        return None
    body = r"[\s_-]+".join(parts)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def contains(text: str, keyword: str) -> bool:
    """True if `keyword` appears in `text` as a whole word or phrase, ignoring case."""
    pattern = _pattern(keyword)
    return bool(pattern and pattern.search(text))


def present_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    return [keyword for keyword in keywords if contains(text, keyword)]


def missing_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    return [keyword for keyword in keywords if not contains(text, keyword)]


def coverage(text: str, ats_keywords: Iterable[str]) -> float:
    """Fraction of `ats_keywords` found in `text`, 0.0 when there are no keywords."""
    keywords = list(ats_keywords)
    if not keywords:
        return 0.0
    return len(present_keywords(text, keywords)) / len(keywords)
