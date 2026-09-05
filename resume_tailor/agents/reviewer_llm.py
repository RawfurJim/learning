"""The reviewer's single LLM call: does the rewrite claim anything its sources do not support?

This is the last check of Agent 6 and runs only when the deterministic ones in
`reviewer.review` have all passed, so the model is asked one narrow question about text
that is already known to use no unknown terms, to keep every number and to fit the length
budget. It cannot approve anything by itself either: `review` uses it only to *reject*.

`sources` carries the rest of what the rewrite is allowed to draw on. A bullet is grounded
in its own text and its project notes, so it needs none; the professional summary draws on
the whole CV and knowledge base, and without them every fact it borrows from another
paragraph reads as an invention. Give the reviewer exactly what the writer was allowed to
use, no more.

The recording case is derived from the reviewed text (`case_for`), not from the JD, so the
same (original, rewrite, project, sources) tuple replays one recording wherever it appears.
"""

from __future__ import annotations

import hashlib

from resume_tailor import llm
from resume_tailor.schemas import DriftVerdict, ProjectFact

AGENT = "reviewer"

PROMPT = """
You are the final reviewer of a tailored CV. You are given one paragraph of the candidate's
CV, a rewritten version of it, and the candidate's own notes about the project the paragraph
is about (when there are any).

Decide one thing only: does the REWRITE claim anything that the ORIGINAL and the NOTES do
not support? Examples of unsupported claims: leading, managing or sizing a team; owning a
system the original only says the candidate contributed to; a new employer, client, product,
qualification or date; a scale, volume or result that is not in the sources; seniority the
original does not state.

Rules for your judgement:
- Rewording, reordering, compressing and changing emphasis are all fine.
- Using a technology that already appears in the ORIGINAL or the NOTES is fine.
- Stronger verbs are fine as long as the fact underneath is the same.
- Judge only what the rewrite asserts, never its style, grammar or length.
- If in doubt, answer supported=false and say what is unsupported.

ORIGINAL:
{original}

REWRITE:
{rewrite}

NOTES ON THIS PROJECT:
{notes}

EVERYTHING ELSE THE CANDIDATE HAS EVIDENCE FOR (their CV and project notes; a fact stated
anywhere here is supported, even if this particular paragraph does not mention it):
{sources}

Return JSON:
{{
  "supported": true if every claim in the rewrite is supported by the sources, else false,
  "reason": one short sentence for the candidate; name the unsupported claim when supported is false
}}
"""


def notes_for(fact: ProjectFact | None) -> str:
    """The project notes as the reviewer sees them; also used to build the whole-CV corpus."""
    if fact is None:
        return "(none provided; the original paragraph is the only source)"
    parts = [f"Project: {fact.name} ({fact.category})"]
    for label, value in (("Employer", fact.employer), ("Period", fact.period), ("Problem", fact.problem), ("Built", fact.built)):
        if value:
            parts.append(f"{label}: {value}")
    for label, values in (("Stack", fact.stack), ("Metrics", fact.metrics), ("Keywords", fact.keywords)):
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    return "\n".join(parts)


def build_prompt(original: str, rewrite: str, fact: ProjectFact | None, sources: str = "") -> str:
    return PROMPT.format(
        original=original.strip(),
        rewrite=rewrite.strip(),
        notes=notes_for(fact),
        sources=sources.strip() or "(nothing beyond the notes above)",
    )


def case_for(original: str, rewrite: str, fact: ProjectFact | None, sources: str = "") -> str:
    """`reviewer/<hash>`: the same inputs always name the same recording."""
    digest = hashlib.sha256()
    for part in (original.strip(), rewrite.strip(), fact.name if fact else "", sources.strip()):
        digest.update(b"\x00")
        digest.update(part.encode("utf-8"))
    return f"{AGENT}/{digest.hexdigest()[:12]}"


def check(
    original: str,
    rewrite: str,
    fact: ProjectFact | None = None,
    *,
    sources: str = "",
    case: str | None = None,
) -> DriftVerdict:
    """Ask the model whether `rewrite` stays inside `original` + `fact` + `sources`.

    `case` names the recording; a bare name is placed under `reviewer/`. When it is
    omitted the name comes from `case_for`.
    """
    if case is None:
        case = case_for(original, rewrite, fact, sources)
    elif "/" not in case:
        case = f"{AGENT}/{case}"
    return llm.generate_json(build_prompt(original, rewrite, fact, sources), DriftVerdict, case=case)
