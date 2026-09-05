"""Knowledge base loader, allowed vocabulary and number extraction.

`knowledge/projects.md` is a flat Markdown file: one `## <Project name>` block per
project, each followed by `key: value` lines (see docs/ARCHITECTURE.md for the
field list). Everything here is deterministic Python; no LLM is involved, because
the vocabulary and number rules are what the reviewer uses to catch invented
content and must never depend on a model's judgement.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from resume_tailor.schemas import ProjectFact

LIST_FIELDS = ("stack", "metrics", "keywords")
TEXT_FIELDS = ("category", "employer", "period", "problem", "built")
EMPTY_MARKERS = {"", "-", "—", "n/a", "none"}

_BLOCK_HEADING = re.compile(r"^##\s+(?P<name>.+?)\s*$")
_FIELD_LINE = re.compile(r"^(?P<key>[a-z_]+):\s*(?P<value>.*)$")
# List separator: a comma not followed by a digit, so `2,000-character` and
# `1,500-review golden dataset` stay whole.
_LIST_SEP = re.compile(r",(?!\d)")

# A vocabulary token: starts and ends with a letter/digit, may contain the joiners
# found in tool names (scikit-learn, C++, C#, Node.js, CI/CD, Linq-Embed-Mistral).
_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9+#./_-]*[A-Za-z0-9+#])?")
_JOINERS = re.compile(r"[/_-]")
_HAS_LETTER = re.compile(r"[A-Za-z]")

# Numeric tokens: thousands separators (1,500), decimals (0.78), optional `%`,
# `k`/`K` or `x`/`X` suffix (100k, 2x). Must not be glued to a preceding letter or
# digit (`F1` is a metric name, not the number 1) and the k/x suffix must not be the
# start of a word (`2xl`).
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9.,])"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:%|[kKxX](?![A-Za-z]))?"
)


def _split_list(value: str) -> list[str]:
    if value.strip().lower() in EMPTY_MARKERS:
        return []
    return [item.strip() for item in _LIST_SEP.split(value) if item.strip()]


def _clean_text(value: str) -> str:
    return "" if value.strip().lower() in EMPTY_MARKERS else value.strip()


def _build_fact(name: str, fields: dict[str, str]) -> ProjectFact:
    data: dict[str, object] = {"name": name}
    for key in TEXT_FIELDS:
        if key in fields:
            data[key] = _clean_text(fields[key])
    for key in LIST_FIELDS:
        if key in fields:
            data[key] = _split_list(fields[key])
    data["category"] = str(data.get("category", "personal")).lower() or "personal"
    return ProjectFact.model_validate(data)


def parse_projects(text: str) -> list[ProjectFact]:
    """Parse knowledge-base Markdown into facts. Text before the first `## ` is ignored."""
    facts: list[ProjectFact] = []
    name: str | None = None
    fields: dict[str, str] = {}
    last_key: str | None = None

    def flush() -> None:
        if name is not None:
            facts.append(_build_fact(name, fields))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading = _BLOCK_HEADING.match(line)
        if heading:
            flush()
            name, fields, last_key = heading.group("name"), {}, None
            continue
        if name is None:
            continue
        field = _FIELD_LINE.match(line)
        if field:
            last_key = field.group("key")
            fields[last_key] = field.group("value").strip()
        elif line.strip() and last_key is not None:
            # Continuation of a wrapped paragraph value.
            fields[last_key] = f"{fields[last_key]} {line.strip()}".strip()
    flush()
    return facts


def load_projects(path: str | Path) -> list[ProjectFact]:
    """Load facts from a knowledge-base file. A missing file yields `[]`, never an error."""
    kb_path = Path(path)
    if not kb_path.is_file():
        return []
    return parse_projects(kb_path.read_text(encoding="utf-8"))


def raw_tokens(text: str) -> list[str]:
    """Vocabulary tokens of `text` in order, case preserved (`JudgeService`, `CI/CD`, `F1`)."""
    return [match.group(0) for match in _TOKEN.finditer(text)]


def tokens(text: str) -> set[str]:
    """Lower-cased vocabulary tokens of `text`: whole tokens plus their joiner-split parts."""
    found: set[str] = set()
    for raw in raw_tokens(text):
        token = raw.lower()
        if not _HAS_LETTER.search(token):
            continue  # bare numbers are handled by extract_numbers
        found.add(token)
        for part in _JOINERS.split(token):
            part = part.strip(".")
            if part and _HAS_LETTER.search(part):
                found.add(part)
    return found


def _phrases(items: Iterable[str]) -> set[str]:
    return {item.strip().lower() for item in items if item.strip()}


def allowed_vocabulary(cv_text: str, facts: Iterable[ProjectFact]) -> set[str]:
    """Case-insensitive (lower-cased) set of terms a rewrite may use.

    Sources: every token of the CV text, plus each fact's `stack`, `keywords` and
    `built`. Comma-separated stack/keyword items are also kept whole (e.g.
    "vector databases") so multi-word tool names can be checked as phrases.
    Membership tests should lower-case the candidate; `"PyTorch".lower() in vocab`.
    """
    vocab: set[str] = set()
    vocab |= tokens(cv_text)
    vocab |= _phrases(part.strip() for part in cv_text.split("|"))
    for fact in facts:
        vocab |= _phrases(fact.stack) | _phrases(fact.keywords)
        vocab |= tokens(" ".join(fact.stack + fact.keywords))
        vocab |= tokens(fact.built)
    return _CaseInsensitiveSet(vocab)


class _CaseInsensitiveSet(set[str]):
    """A `set[str]` whose membership test ignores case (`"PyTorch" in vocab`)."""

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return super().__contains__(item.lower())
        return super().__contains__(item)


def extract_numbers(text: str) -> list[str]:
    """Numeric tokens in order of appearance: `93%`, `100k`, `0.78`, `1,500`, `2x`."""
    return [match.group(0) for match in _NUMBER.finditer(text)]
