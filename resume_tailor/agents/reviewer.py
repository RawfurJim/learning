"""Agent 6 (Reviewer / Guard): the independent safety net over every rewrite.

`review` answers one question - may this rewritten paragraph replace the original? - and
is deliberately independent of the writer that produced the text: it re-derives every rule
from the original paragraph, the allowed vocabulary and (for a bullet that belongs to a
knowledge-base project) that project's notes. A writer that stops checking, a poisoned
recording or a hand-edited cache entry is caught here.

Three deterministic checks run first and short-circuit, cheapest and most certain first:

1. a capitalised / product-name term the CV and knowledge base do not contain
   -> `term not in CV/KB: React`
2. a number the original paragraph carried that the rewrite dropped
   -> `metric missing: 93%`
3. a word count outside +/-10% of the original
   -> `length: 34 vs budget 20-24`

Only when all three pass does one LLM call (`reviewer_llm.check`) ask whether the new text
claims anything its sources do not support ("led a team of 10"). The sources are the
original paragraph, the project notes (`fact`) and `sources` - the rest of the material the
writer was allowed to draw on, which for the professional summary is the whole CV and
knowledge base. Pass `semantic=False` to stay entirely deterministic (no LLM call, no
recording); the caller does that for the skills line, which is a delimiter-separated list
of terms rather than a set of claims and is fully covered by the vocabulary check.

The reviewer never rewrites anything: the caller keeps the original text whenever
`Verdict.accept` is false and shows `Verdict.reason` to the user.
"""

from __future__ import annotations

from collections.abc import Iterable

from resume_tailor.agents import reviewer_llm
from resume_tailor.agents.summary_skills_writer import (
    missing_numbers,
    suspicious_terms,
    within_word_budget,
    word_bounds,
    words,
)
from resume_tailor.knowledge import raw_tokens
from resume_tailor.schemas import ProjectFact, Verdict

UNCHANGED = "unchanged"
CLEAN = "no invented term, no lost metric, length within budget"


def invented_terms(new: str, original: str, vocab: Iterable[str]) -> list[str]:
    """Terms of `new` that are neither in `vocab` nor already in `original`.

    A term the original paragraph used is never an invention, whatever the vocabulary
    says, so the reviewer can only ever block something the rewrite introduced.
    """
    had = {token.lower() for token in raw_tokens(original)}
    return [term for term in suspicious_terms(new, vocab) if term.lower() not in had]


def review(
    orig: str,
    new: str,
    vocab: set[str],
    fact: ProjectFact | None = None,
    *,
    sources: str = "",
    semantic: bool = True,
    case: str | None = None,
) -> Verdict:
    """Whether `new` may replace `orig`, and why.

    `vocab` is `knowledge.allowed_vocabulary(cv_text, facts)`; `fact` is the
    knowledge-base project the paragraph is about, when there is one; `sources` is any
    further evidence the rewrite may draw on. `case` names the semantic check's recording
    (default: derived from the reviewed text itself).
    """
    text = " ".join(new.split())
    if not text:
        return Verdict(accept=False, reason="the rewrite is empty")
    if text == " ".join(orig.split()):
        return Verdict(accept=True, reason=UNCHANGED)

    terms = invented_terms(new, orig, vocab)
    if terms:
        return Verdict(accept=False, reason=f"term not in CV/KB: {', '.join(terms)}")

    lost = missing_numbers(new, orig)
    if lost:
        return Verdict(accept=False, reason=f"metric missing: {', '.join(lost)}")

    if not within_word_budget(new, orig):
        low, high = word_bounds(words(orig))
        return Verdict(accept=False, reason=f"length: {words(new)} vs budget {low}-{high}")

    if not semantic:
        return Verdict(accept=True, reason=CLEAN)

    drift = reviewer_llm.check(orig, new, fact, sources=sources, case=case)
    if not drift.supported:
        return Verdict(accept=False, reason=f"unsupported claim: {drift.reason.strip() or 'not backed by the CV or the project notes'}")
    return Verdict(accept=True, reason=drift.reason.strip() or CLEAN)
