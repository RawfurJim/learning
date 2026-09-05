"""Classify CV paragraphs into sections so only summary, skills and bullets get rewritten.

The classifier is a small state machine driven by the all-caps section headings in
Jim's CV. Heading lines themselves are `other`; everything before the first heading
is `header`; inside PROFESSIONAL EXPERIENCE, bullet paragraphs are `exp_bullet` and
every other line (employer/date lines, project sub-headings) is `exp_meta`.
"""

from __future__ import annotations

import re
from typing import Literal

from resume_tailor.docx_io import Para

Section = Literal["header", "summary", "exp_meta", "exp_bullet", "skills", "education", "other"]

# Heading keyword (matched against the upper-cased, whitespace-normalised line) -> section state.
HEADING_KEYWORDS: tuple[tuple[str, Section], ...] = (
    ("PROFESSIONAL SUMMARY", "summary"),
    ("PROFESSIONAL EXPERIENCE", "exp_meta"),
    ("CORE SKILLS", "skills"),
    ("EDUCATION", "education"),
)

_WS = re.compile(r"\s+")


def heading_section(text: str) -> Section | None:
    """Return the section a heading line opens, or None if the line is not a heading."""
    normalised = _WS.sub(" ", text).strip().upper().rstrip(":")
    if not normalised or len(normalised) > 40:
        return None
    for keyword, section in HEADING_KEYWORDS:
        if normalised.startswith(keyword):
            return section
    return None


def classify(paras: list[Para]) -> dict[str, Section]:
    """Map every paragraph id to its section."""
    result: dict[str, Section] = {}
    state: Section = "header"
    for para in paras:
        if not para.full_text.strip():
            result[para.id] = "other"
            continue
        heading = heading_section(para.full_text)
        if heading is not None:
            state = heading
            result[para.id] = "other"
            continue
        if state == "exp_meta":
            is_bullet = para.literal_bullet_prefix is not None or para.is_numbered
            result[para.id] = "exp_bullet" if is_bullet else "exp_meta"
        else:
            result[para.id] = state
    return result
