"""Agent 4 (Summary & Skills Writer): the professional summary and the skills line in the JD's words.

The model writes; Python enforces. Every rule that can be checked in code is checked
here, on the model's output, before anything reaches the document:

Summary
- word count within +/-10% of the original (`WORD_TOLERANCE`);
- every number token of the original survives verbatim, and no number appears that is
  not in the original or the knowledge-base facts (extra metrics only from the KB);
- every capitalised / product-name token is in the allowed vocabulary (CV text, KB
  stack/keywords/built, the matched JD skills). The first word of a sentence is exempt
  when it is an ordinary Title-case word such as "Delivered" (see `suspicious_terms`);
- no JD skill from `missing` or the unapproved `adjacent` bucket is mentioned.

Skills line
- every entry (or every term of a grouped entry such as "Python (NumPy, Pandas)") is in
  the inventory or in the approved adjacent skills; anything else is dropped;
- duplicates are removed; entries are stably sorted mandatory-matched first, then
  nice-to-have-matched, then the rest;
- the entry count is trimmed / padded (with original entries) to within +/-2 of the
  original count, and the joined line is fitted to +/-10% of the original word *and*
  character count by dropping the least relevant tail entries or unmatched parenthesised
  terms (never by adding anything that is not an original entry).

A rule the model breaks is fed back once (`<case>_retry` recording); if the retry
still breaks it, the paragraph falls back to its original text (the skills line falls
back to the original entries in mandatory-first order). The fallback is reported in
`SummarySkillsResult.notes` and `*_reverted`, never silently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from resume_tailor import llm
from resume_tailor.ats_score import contains
from resume_tailor.knowledge import extract_numbers, raw_tokens
from resume_tailor.matching import find_in_inventory, skills_line_terms
from resume_tailor.schemas import (
    JDIntent,
    JDKeywords,
    ProjectFact,
    SkillMatch,
    SummarySkillsResult,
    SummarySkillsRewrite,
)

AGENT = "summary_skills"
WORD_TOLERANCE = 0.10
CHAR_TOLERANCE = 0.10  # skills line only; per-paragraph words are the PRD rule, chars keep the +/-3% document rule safe
SKILL_COUNT_TOLERANCE = 2
MAX_FACTS = 4
_MAX_BUILT_CHARS = 320

PROMPT = """You are rewriting two paragraphs of a candidate's CV for one specific job, on behalf of an honest recruiter. The candidate must never appear to have a skill, tool, employer, title, date or number they do not have. You may only rephrase, reorder and re-emphasise what is already true.

THE ROLE (from the job description)
- Summary: {role_summary}
- Seniority: {seniority}
- Domain: {domain}
- Tone of the advert: {tone}
- Top priorities, most important first:
{priorities}

SKILLS THE JOB REQUIRES THAT THE CANDIDATE HAS (JD name -> the candidate's own term). Mention as many of these as fit naturally, using the JD's name:
{mandatory}

NICE-TO-HAVE SKILLS FROM THE JOB THAT THE CANDIDATE HAS (JD name -> the candidate's term):
{nice_to_have}

ADJACENT SKILLS THE CANDIDATE HAS EXPLICITLY APPROVED FOR THIS APPLICATION (you may list them):
{approved}

DO NOT MENTION (the job asks for these but the candidate does not have them):
{forbidden}

ALLOWED SKILLS - the complete list of tools, technologies and skills the candidate can claim, in the candidate's spelling. The skills line may contain nothing else:
{inventory}

PROFESSIONAL EVIDENCE - the candidate's most relevant projects for this role, most relevant first. Work projects (with an employer) outrank personal ones; lead with them. Use only facts written here or in the original summary:
{evidence}

ORIGINAL PROFESSIONAL SUMMARY ({summary_words} words):
<<<
{summary}
>>>

ORIGINAL SKILLS LINE ({skills_count} entries, {skills_words} words, delimiter "{delimiter}"):
{skills}
{feedback}
WRITE THE NEW SUMMARY
1. Between {min_words} and {max_words} words (the original has {summary_words}). Same register as the original: implied first person, no "I", no headings, 2-4 sentences, one paragraph.
2. Keep every one of these number tokens exactly as written: {numbers}. Do not add any other number unless it appears verbatim in the PROFESSIONAL EVIDENCE metrics.
3. Use the job's vocabulary for things the candidate genuinely did (for example the JD's name for a matched skill), and cover the top priorities in order of importance. Do not mention anything under DO NOT MENTION, and do not introduce any tool, product, company, title or capitalised term that is not in the original summary, the ALLOWED SKILLS or the PROFESSIONAL EVIDENCE.
4. Professional (employer) evidence first; personal projects only if there is room.

WRITE THE NEW SKILLS LINE
5. Return between {min_count} and {max_count} entries (the original has {skills_count}). An entry is normally a single skill from ALLOWED SKILLS / the approved list, in the candidate's spelling. The only grouped entries allowed are the original ones (for example "Python (NumPy, Pandas, SciPy, Keras, NLTK)"): keep such an entry as it is, drop a term inside it, or add at most one allowed skill the job asks for. Never create a new group and never put a term in parentheses that the original did not group.
6. Order: first the skills the job requires that the candidate has, then the nice-to-have skills the candidate has, then the most relevant remaining original entries. Drop the least relevant original entries to stay within the count. Never repeat a skill.
7. Length: the joined line must have between {min_skills_words} and {max_skills_words} words (the original has {skills_words}). Count before answering: "SQL" is 1 word, "Cloud Platforms (AWS, Azure)" is 4. When in doubt, use fewer parenthesised terms, not more.

Return a JSON object with exactly two fields: "summary" (string) and "skills" (list of strings, one per entry, without the delimiter).
"""

FEEDBACK = """
YOUR PREVIOUS ATTEMPT WAS REJECTED. Fix every point below and keep everything else:
{problems}
"""

_TITLE_CASE_WORD = re.compile(r"[A-Z][a-z]+(?:-[a-z]+)*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_JOINERS = re.compile(r"[/_-]")
_CASE_SAFE = re.compile(r"[^a-z0-9]+")


def words(text: str) -> int:
    return len(text.split())


def word_bounds(original_words: int, tolerance: float = WORD_TOLERANCE) -> tuple[int, int]:
    """Inclusive word range allowed for a rewrite of a paragraph with `original_words` words."""
    slack = tolerance * original_words
    return math.ceil(original_words - slack - 1e-9), math.floor(original_words + slack + 1e-9)


def char_bounds(original_chars: int, tolerance: float = CHAR_TOLERANCE) -> tuple[int, int]:
    """Inclusive character range for the skills line: words alone do not bound the document's length."""
    slack = tolerance * original_chars
    return math.ceil(original_chars - slack - 1e-9), math.floor(original_chars + slack + 1e-9)


def within_word_budget(new: str, original: str, tolerance: float = WORD_TOLERANCE) -> bool:
    return abs(words(new) - words(original)) <= tolerance * words(original)


def delimiter_of(skills_line: str) -> str:
    """The delimiter the original skills line uses, so the rewrite is joined the same way."""
    if " | " in skills_line:
        return " | "
    if "|" in skills_line:
        return "|"
    return ", "


def skills_entries(skills_line: str) -> list[str]:
    """The original line split into entries ("Python (NumPy, Pandas)" stays one entry)."""
    sep = delimiter_of(skills_line).strip() or ","
    return [entry.strip() for entry in skills_line.split(sep) if entry.strip()]


def _is_term_like(token: str) -> bool:
    """Tokens that name things: any capital letter, or letters mixed with digits (F1, 8B)."""
    has_upper = any(c.isupper() for c in token)
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    return has_upper or (has_digit and has_alpha)


def _in_vocab(token: str, vocab: Iterable[str]) -> bool:
    if token in vocab or token.lower() in vocab:
        return True
    parts = [p.strip(".") for p in _JOINERS.split(token)]
    parts = [p for p in parts if p and any(c.isalpha() for c in p)]
    return bool(parts) and all(p in vocab or p.lower() in vocab for p in parts)


def suspicious_terms(text: str, vocab: Iterable[str]) -> list[str]:
    """Capitalised / product-name tokens of `text` that are not in `vocab`.

    The first token of a sentence is exempt when it is an ordinary Title-case word
    ("Delivered", "Hands-on"), because sentence starts are capitalised by grammar, not
    because they name a technology. Acronyms and camel-case (AWS, PyTorch) are always
    checked, wherever they appear.
    """
    found: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        for index, token in enumerate(raw_tokens(sentence)):
            if index == 0 and _TITLE_CASE_WORD.fullmatch(token):
                continue
            if not _is_term_like(token):
                continue
            if _in_vocab(token, vocab):
                continue
            if token.lower() not in seen:
                seen.add(token.lower())
                found.append(token)
    return found


def missing_numbers(new: str, original: str) -> list[str]:
    """Number tokens of `original` that do not appear in `new` (order of the original)."""
    have = set(extract_numbers(new))
    out: list[str] = []
    for number in extract_numbers(original):
        if number not in have and number not in out:
            out.append(number)
    return out


def invented_numbers(new: str, allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    return sorted({n for n in extract_numbers(new) if n not in allowed_set})


def forbidden_mentions(text: str, forbidden: Iterable[str], original: str = "") -> list[str]:
    """Forbidden JD skills (`missing`, unapproved `adjacent`) that `text` mentions and `original` did not."""
    return [skill for skill in forbidden if contains(text, skill) and not contains(original, skill)]


@dataclass
class WriterInputs:
    """Everything the writer needs, resolved once so the prompt and the checks agree."""

    summary: str
    skills_line: str
    keywords: JDKeywords
    match: SkillMatch
    approved_adjacent: list[str]
    intent: JDIntent
    facts: list[ProjectFact]
    inventory: list[str]
    vocabulary: set[str]
    original_entries: list[str] = field(init=False)
    delimiter: str = field(init=False)
    allowed: dict[str, str] = field(init=False)  # lower -> canonical spelling
    bucket: dict[str, int] = field(init=False)  # lower term -> 0 mandatory, 1 nice-to-have
    forbidden: list[str] = field(init=False)
    allowed_numbers: set[str] = field(init=False)
    check_vocab: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.approved_adjacent = [s.strip() for s in self.approved_adjacent if s.strip()]
        self.original_entries = skills_entries(self.skills_line)
        self.delimiter = delimiter_of(self.skills_line)
        self.allowed = {}
        for term in [*self.inventory, *self.approved_adjacent]:
            self.allowed.setdefault(term.lower(), term)
        approved_lower = {s.lower() for s in self.approved_adjacent}
        mandatory = {s.lower() for s in self.keywords.mandatory}
        self.bucket = {}
        for skill in self.match.matched:
            level = 0 if skill.lower() in mandatory else 1
            for name in (skill, self._inventory_term(skill)):
                key = name.lower()
                self.bucket[key] = min(level, self.bucket.get(key, 2))
        self.forbidden = [*self.match.missing, *(s for s in self.match.adjacent if s.lower() not in approved_lower)]
        self.allowed_numbers = set(extract_numbers(self.summary))
        for fact in self.facts:
            self.allowed_numbers |= set(extract_numbers(" ".join([*fact.metrics, *fact.stack, fact.built, fact.problem])))
        self.check_vocab = set(self.vocabulary)
        for skill in self.match.matched:
            self.check_vocab |= {t.lower() for t in raw_tokens(skill)}
            self.check_vocab |= {p.lower() for t in raw_tokens(skill) for p in _JOINERS.split(t) if p}

    def _inventory_term(self, jd_skill: str) -> str:
        via = self.match.aliases.get(jd_skill)
        if via:
            return via
        return find_in_inventory(jd_skill, self.inventory) or jd_skill

    def entry_terms(self, entry: str) -> list[str]:
        return skills_line_terms(entry)

    def entry_bucket(self, entry: str) -> int:
        levels = [self.bucket.get(t.lower(), 2) for t in [entry, *self.entry_terms(entry)]]
        return min(levels) if levels else 2

    def entry_allowed(self, entry: str) -> bool:
        terms = self.entry_terms(entry)
        if not terms:
            return False
        if entry.strip().lower() in self.allowed:
            return True
        return all(t.lower() in self.allowed for t in terms)

    def entry_forbidden(self, entry: str) -> str | None:
        lowered = {t.lower() for t in [entry, *self.entry_terms(entry)]}
        for skill in self.forbidden:
            if skill.lower() in lowered:
                return skill
        return None

    def canonical_entry(self, entry: str) -> str:
        entry = " ".join(entry.split()).strip(" .")
        return self.allowed.get(entry.lower(), entry)


def _pairs(skills: Iterable[str], inputs: WriterInputs) -> str:
    lines = []
    for skill in skills:
        term = inputs._inventory_term(skill)
        lines.append(f"- {skill}" if term.lower() == skill.lower() else f"- {skill} -> {term}")
    return "\n".join(lines) or "- (none)"


def _evidence(facts: Sequence[ProjectFact]) -> str:
    blocks = []
    for fact in facts[:MAX_FACTS]:
        built = " ".join(fact.built.split())
        if len(built) > _MAX_BUILT_CHARS:
            built = built[: _MAX_BUILT_CHARS - 3].rstrip() + "..."
        head = f"- {fact.name}" + (f" (work, {fact.employer})" if fact.professional else " (personal project)")
        lines = [head]
        if built:
            lines.append(f"  built: {built}")
        if fact.metrics:
            lines.append(f"  metrics: {', '.join(fact.metrics)}")
        if fact.stack:
            lines.append(f"  stack: {', '.join(fact.stack)}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) or "- (no project notes available; use only the original summary)"


def build_prompt(inputs: WriterInputs, problems: Sequence[str] = ()) -> str:
    mandatory_lower = {s.lower() for s in inputs.keywords.mandatory}
    matched_mandatory = [s for s in inputs.match.matched if s.lower() in mandatory_lower]
    matched_nice = [s for s in inputs.match.matched if s.lower() not in mandatory_lower]
    min_words, max_words = word_bounds(words(inputs.summary))
    min_skills_words, max_skills_words = word_bounds(words(inputs.skills_line))
    count = len(inputs.original_entries)
    feedback = FEEDBACK.format(problems="\n".join(f"- {p}" for p in problems)) if problems else ""
    return PROMPT.format(
        role_summary=inputs.intent.role_summary.strip(),
        seniority=inputs.intent.seniority,
        domain=inputs.intent.domain.strip(),
        tone=inputs.intent.tone.strip(),
        priorities="\n".join(f"  {i}. {p}" for i, p in enumerate(inputs.intent.top_priorities, 1)),
        mandatory=_pairs(matched_mandatory, inputs),
        nice_to_have=_pairs(matched_nice, inputs),
        approved="\n".join(
            f"- {s} (the candidate has {inputs.match.aliases.get(s, 'a directly related tool')})"
            for s in inputs.approved_adjacent
        )
        or "- (none)",
        forbidden="\n".join(f"- {s}" for s in inputs.forbidden) or "- (none)",
        inventory="\n".join(f"- {t}" for t in sorted(inputs.allowed.values(), key=str.lower)),
        evidence=_evidence(inputs.facts),
        summary_words=words(inputs.summary),
        summary=inputs.summary.strip(),
        skills_count=count,
        skills_words=words(inputs.skills_line),
        delimiter=inputs.delimiter,
        skills="\n".join(f"- {e}" for e in inputs.original_entries),
        feedback=feedback,
        min_words=min_words,
        max_words=max_words,
        numbers=", ".join(extract_numbers(inputs.summary)) or "(none)",
        min_count=max(1, count - SKILL_COUNT_TOLERANCE),
        max_count=count + SKILL_COUNT_TOLERANCE,
        min_skills_words=min_skills_words,
        max_skills_words=max_skills_words,
    )


def check_summary(new: str, inputs: WriterInputs) -> list[str]:
    """Every rule the new summary breaks, phrased as feedback for the model."""
    new = " ".join(new.split())
    problems: list[str] = []
    lo, hi = word_bounds(words(inputs.summary))
    if not within_word_budget(new, inputs.summary):
        problems.append(f"The summary has {words(new)} words; it must have between {lo} and {hi}.")
    lost = missing_numbers(new, inputs.summary)
    if lost:
        problems.append(f"These numbers from the original are missing and must appear verbatim: {', '.join(lost)}.")
    invented = invented_numbers(new, inputs.allowed_numbers)
    if invented:
        problems.append(f"These numbers are not in the original or the evidence and must be removed: {', '.join(invented)}.")
    terms = suspicious_terms(new, inputs.check_vocab)
    if terms:
        problems.append(
            "These capitalised terms are not in the candidate's CV, notes or matched skills and must be removed: "
            + ", ".join(terms)
            + "."
        )
    mentioned = forbidden_mentions(new, inputs.forbidden, inputs.summary)
    if mentioned:
        problems.append(f"The summary mentions skills the candidate does not have: {', '.join(mentioned)}.")
    return problems


def clean_skills(raw: Sequence[str], inputs: WriterInputs) -> tuple[list[str], list[str]]:
    """Deterministic clean-up of the model's entries. Returns `(entries, notes)`.

    Drops entries outside the inventory / approved list or naming a forbidden skill,
    removes duplicates, sorts mandatory-first, then trims or pads the count to within
    +/-2 of the original. Never adds anything that is not an original entry.
    """
    notes: list[str] = []
    entries: list[str] = []
    covered: set[str] = set()
    for item in raw:
        entry = inputs.canonical_entry(item)
        if not entry:
            continue
        if not inputs.entry_allowed(entry):
            notes.append(f"Dropped skill '{entry}': not in your CV, knowledge base or approved skills.")
            continue
        blocked = inputs.entry_forbidden(entry)
        if blocked:
            notes.append(f"Dropped skill '{entry}': '{blocked}' is a JD skill you do not have (or have not approved).")
            continue
        keys = {t.lower() for t in inputs.entry_terms(entry)} | {entry.lower()}
        if keys <= covered:
            continue  # every term already listed
        covered |= keys
        entries.append(entry)

    entries.sort(key=inputs.entry_bucket)  # stable: keeps the model's order inside each bucket

    target = len(inputs.original_entries)
    lo, hi = max(1, target - SKILL_COUNT_TOLERANCE), target + SKILL_COUNT_TOLERANCE
    if len(entries) > hi:
        dropped = entries[hi:]
        entries = entries[:hi]
        notes.append(f"Trimmed {len(dropped)} skill entries to stay within +/-{SKILL_COUNT_TOLERANCE} of the original count.")
    if len(entries) < lo:
        added = 0
        for entry in inputs.original_entries:
            if len(entries) >= lo:
                break
            keys = {t.lower() for t in inputs.entry_terms(entry)} | {entry.lower()}
            if keys <= covered or inputs.entry_forbidden(entry):
                continue
            covered |= keys
            entries.append(entry)
            added += 1
        if added:
            notes.append(f"Restored {added} original skill entries to stay within +/-{SKILL_COUNT_TOLERANCE} of the original count.")
        entries.sort(key=inputs.entry_bucket)
    return entries, notes


_GROUP = re.compile(r"^(?P<head>[^()]+?)\s*\((?P<inner>[^()]*)\)\s*$")


def split_group(entry: str) -> tuple[str, list[str]]:
    """"Python (NumPy, Pandas)" -> ("Python", ["NumPy", "Pandas"]); a plain entry has no inner terms."""
    match = _GROUP.match(entry.strip())
    if not match:
        return entry.strip(), []
    inner = [t.strip() for t in match.group("inner").split(",") if t.strip()]
    return match.group("head").strip(), inner


def join_group(head: str, inner: Sequence[str]) -> str:
    return f"{head} ({', '.join(inner)})" if inner else head


def fit_word_budget(entries: list[str], inputs: WriterInputs) -> tuple[list[str], list[str]]:
    """Bring the joined line into the +/-10% word budget without inventing anything.

    Too long: drop the least relevant tail entries while the count allows, then remove
    unmatched parenthesised terms from the tail backwards. Too short: append unused
    original entries while the count allows. Returns `(entries, notes)`; anything still
    outside the budget is left for `check_skills` to report.
    """
    notes: list[str] = []
    entries = list(entries)
    lo_w, hi_w = word_bounds(words(inputs.skills_line))
    lo_ch, hi_ch = char_bounds(len(inputs.skills_line))
    target = len(inputs.original_entries)
    lo_c, hi_c = max(1, target - SKILL_COUNT_TOLERANCE), target + SKILL_COUNT_TOLERANCE

    def line() -> str:
        return inputs.delimiter.join(entries)

    def too_long() -> bool:
        return words(line()) > hi_w or len(line()) > hi_ch

    def too_short() -> bool:
        return words(line()) < lo_w or len(line()) < lo_ch

    dropped = 0
    while too_long() and len(entries) > lo_c:
        entries.pop()
        dropped += 1
    if dropped:
        notes.append(f"Dropped the {dropped} least relevant skill entries to keep the line within +/-10% of its length.")

    trimmed: list[str] = []
    for index in range(len(entries) - 1, -1, -1):
        if not too_long():
            break
        head, inner = split_group(entries[index])
        while inner and too_long():
            removable = [t for t in inner if inputs.bucket.get(t.lower(), 2) == 2]
            if not removable:
                break
            inner.remove(removable[-1])
            trimmed.append(removable[-1])
            entries[index] = join_group(head, inner)
    if trimmed:
        notes.append(f"Removed {len(trimmed)} grouped skill terms ({', '.join(trimmed)}) to keep the line within +/-10% of its length.")

    covered = {t.lower() for e in entries for t in [e, *inputs.entry_terms(e)]}
    added = 0
    for entry in inputs.original_entries:
        if not too_short() or len(entries) >= hi_c:
            break
        keys = {t.lower() for t in [entry, *inputs.entry_terms(entry)]}
        if keys <= covered or inputs.entry_forbidden(entry):
            continue
        covered |= keys
        entries.append(entry)
        added += 1
    if added:
        notes.append(f"Restored {added} original skill entries to keep the line within +/-10% of its length.")
        entries.sort(key=inputs.entry_bucket)
    return entries, notes


def check_skills(entries: Sequence[str], inputs: WriterInputs) -> list[str]:
    """Rules the cleaned skills line still breaks (count and word budget)."""
    problems: list[str] = []
    target = len(inputs.original_entries)
    if abs(len(entries) - target) > SKILL_COUNT_TOLERANCE:
        problems.append(f"The skills line has {len(entries)} entries; it must have between {target - 2} and {target + 2}.")
    line = inputs.delimiter.join(entries)
    lo, hi = word_bounds(words(inputs.skills_line))
    if not within_word_budget(line, inputs.skills_line):
        problems.append(
            f"The skills line has {words(line)} words; it must have between {lo} and {hi}. "
            "Use fewer parenthesised terms."
        )
    lo_ch, hi_ch = char_bounds(len(inputs.skills_line))
    if not lo_ch <= len(line) <= hi_ch:
        problems.append(
            f"The skills line has {len(line)} characters; it must have between {lo_ch} and {hi_ch} "
            "so the document keeps its length. Prefer short entries."
        )
    return problems


def fallback_skills(inputs: WriterInputs) -> list[str]:
    """The original entries, mandatory-matched first: always truthful, same length."""
    return sorted(inputs.original_entries, key=inputs.entry_bucket)


def default_case(inputs: WriterInputs) -> str:
    payload = json.dumps(
        [inputs.summary, inputs.skills_line, inputs.match.model_dump(), sorted(s.lower() for s in inputs.approved_adjacent)],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def case_for(base: str, approved_adjacent: Iterable[str]) -> str:
    """Stable recording name: `<base>` or `<base>__<approved skills>` when adjacent skills were approved."""
    approved = sorted({_CASE_SAFE.sub("_", s.lower()).strip("_") for s in approved_adjacent if s.strip()})
    return base if not approved else f"{base}__{'_'.join(approved)}"


def run(
    summary: str,
    skills_line: str,
    *,
    keywords: JDKeywords,
    match: SkillMatch,
    approved_adjacent: Iterable[str],
    intent: JDIntent,
    facts: Sequence[ProjectFact],
    inventory: Iterable[str],
    vocabulary: Iterable[str],
    case: str | None = None,
) -> SummarySkillsResult:
    """Rewrite the summary and skills line. `case` names the recording (`tests/recordings/summary_skills/<case>.json`).

    `match` must already have the approved adjacent skills applied (`matching.apply_approvals`);
    `facts` should be ranked most relevant first (only the top `MAX_FACTS` reach the prompt).
    """
    inputs = WriterInputs(
        summary=summary,
        skills_line=skills_line,
        keywords=keywords,
        match=match,
        approved_adjacent=list(approved_adjacent),
        intent=intent,
        facts=list(facts),
        inventory=list(inventory),
        vocabulary=set(vocabulary),
    )
    case = case or default_case(inputs)
    in_tokens = out_tokens = calls = 0
    notes: list[str] = []

    problems: list[str] = []
    summary_out = skills_out = None
    for attempt, name in enumerate((case, f"{case}_retry")):
        raw = llm.generate_json(build_prompt(inputs, problems), SummarySkillsRewrite, case=f"{AGENT}/{name}")
        calls += 1
        usage = llm.last_usage()
        if usage:
            in_tokens += usage.input_tokens
            out_tokens += usage.output_tokens

        new_summary = " ".join(raw.summary.split())
        summary_problems = check_summary(new_summary, inputs)
        entries, skill_notes = clean_skills(raw.skills, inputs)
        entries, fit_notes = fit_word_budget(entries, inputs)
        skill_notes += fit_notes
        skills_problems = check_skills(entries, inputs)

        if summary_out is None and not summary_problems:
            summary_out = new_summary
        if skills_out is None and not skills_problems:
            skills_out = entries
            notes.extend(skill_notes)
        if summary_out is not None and skills_out is not None:
            break
        problems = summary_problems + skills_problems
        if attempt == 0:
            notes.append("Retried once with feedback: " + " ".join(problems))

    summary_reverted = summary_out is None
    skills_reverted = skills_out is None
    if summary_reverted:
        summary_out = " ".join(summary.split())
        notes.append("Summary kept as the original: the rewrite still broke a rule after one retry (" + " ".join(check_summary(new_summary, inputs)) + ").")
    if skills_reverted:
        skills_out = fallback_skills(inputs)
        notes.append("Skills line kept as the original entries (mandatory skills first): the rewrite still broke a rule after one retry.")

    return SummarySkillsResult(
        summary=summary_out,
        skills=skills_out,
        summary_reverted=summary_reverted,
        skills_reverted=skills_reverted,
        notes=notes,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        calls=calls,
    )
