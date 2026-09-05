"""Agent 3 (Experience Writer): every experience bullet in the JD's words, at the same length.

One LLM call per *project group*: an `exp_meta` sub-heading and the `exp_bullet`
paragraphs under it. Each bullet is grounded in the knowledge-base `ProjectFact` whose
name matches the sub-heading (or, for a mixed group such as "Independent Projects", the
bullet text). Groups are rewritten in parallel; results come back in document order.

The model writes; Python enforces, per bullet, before anything reaches the document:
- word count within +/-10% of the original bullet (`WORD_TOLERANCE`);
- every number token of the original survives verbatim; a number may be *added* only when
  it comes from a metric copied verbatim from that bullet's own `ProjectFact.metrics`
  and reported in `used_kb_metrics` (a metric of another project is rejected);
- every capitalised / product-name token is in the allowed vocabulary (CV text, KB
  stack/keywords/built, the matched JD skills); no `missing` / unapproved `adjacent`
  JD skill is mentioned;
- `jd_keywords_used` is cleaned to JD keywords the candidate has (matched skills, or ATS
  phrases made only of allowed vocabulary) that really appear in the new text; a keyword
  the rewrite *introduced* must be evidenced by that bullet's own project notes (or its
  alias term must be), so "SQL" is not woven into a project whose notes never mention it;
- a personal-project bullet may change only when a *matched* JD keyword truthfully
  applies to that project (it is in the project's notes) and is not already evidenced by
  a professional project; and only once every professional bullet uses a JD keyword.
  `exp_meta` paragraphs are never touched.

A bullet that breaks a rule is fed back once (`<case>_retry` recording); if it still
breaks a rule it falls back to its original text, listed in `ExperienceResult.reverted`
and explained in `notes`, never silently.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from resume_tailor import llm
from resume_tailor.agents.summary_skills_writer import (
    forbidden_mentions,
    missing_numbers,
    suspicious_terms,
    within_word_budget,
    word_bounds,
    words,
)
from resume_tailor.ats_score import contains
from resume_tailor.docx_io import Para, split_bullet_prefix
from resume_tailor.knowledge import extract_numbers, raw_tokens, tokens
from resume_tailor.schemas import (
    BulletGroupRewrite,
    BulletRewrite,
    ExperienceResult,
    JDIntent,
    JDKeywords,
    ProjectFact,
    SkillMatch,
)
from resume_tailor.sections import Section

AGENT = "experience"
WORD_TOLERANCE = 0.10
MAX_WORKERS = 4
MIN_NAME_OVERLAP = 2  # significant name tokens a bullet must share with a project to be grounded in it
_MAX_BUILT_CHARS = 480
_SLUG_WORDS = 3

PROMPT = """You are rewriting the experience bullets of one project on a candidate's CV for one specific job, on behalf of an honest recruiter. The candidate must never appear to have a skill, tool, employer, title, date or number they do not have. You may only rephrase, reorder and re-emphasise what is already true, so the bullets speak the job description's language.

THE ROLE (from the job description)
- Summary: {role_summary}
- Seniority: {seniority}
- Domain: {domain}
- Tone of the advert: {tone}
- Top priorities, most important first:
{priorities}

JD SKILLS THE CANDIDATE HAS (JD name -> the candidate's own term). Where a bullet genuinely demonstrates one, use the JD's name for it:
{matched}

OTHER JD PHRASES you may use, but only where the bullet already truthfully demonstrates them:
{phrases}

DO NOT MENTION (the job asks for these but the candidate does not have them):
{forbidden}

PROJECT NOTES - the only source of extra detail and metrics for these bullets:
{evidence}

BULLETS TO REWRITE - they sit under the sub-heading "{heading}" ({kind}). Return exactly {count} entries, in this order, each with the para_id shown:
{bullets}
{feedback}
RULES
1. One entry per bullet, same order, same para_id. Never merge, split, add or drop bullets. Never change the sub-heading.
2. Length: each bullet must have between the minimum and maximum words shown for it. Count before answering.
3. Numbers: keep every number token listed for a bullet exactly as written (93% stays 93%, 0.78 stays 0.78, 2,000-character stays 2,000-character). You may add at most one extra metric per bullet, and only from that bullet's project "metrics you may add" list: phrase it naturally but keep its numbers exactly as listed, and report the list item in used_kb_metrics exactly as it appears there. Never write any other number, and never use a metric from a different project.
4. Vocabulary: use only tools, products, techniques and terms that appear in the original bullet, in that project's notes, or in the JD SKILLS / PHRASES lists above. A JD skill may be added to a bullet only if that project's notes show it (the skill or the candidate's own term for it); do not import a skill from another project. Do not introduce any other capitalised term, tool, company, title or date. Do not mention anything under DO NOT MENTION.
5. Work (employer) bullets: rewrite every one so it leads with what the role prioritises, in the JD's vocabulary; use the JD's name for each skill the bullet genuinely demonstrates. Every work bullet should use at least one JD skill or phrase where that is truthful. Keywords must read naturally to a hiring manager: swap the candidate's wording for the JD's term for the same thing; never bolt a generic term (such as "Python" or "Machine Learning") onto a clause where it adds no information. Do not change what was built, who did it, the scale or the outcome.
6. Personal-project bullets: return the text EXACTLY unchanged, unless a JD skill from the lists truthfully applies to that project (it appears in that project's notes) and is not already shown by the work projects; only then reword minimally to use the JD's name for it, and otherwise change nothing.
7. jd_keywords_used: for each bullet, list every JD skill or phrase from the lists above that appears in the rewritten text, spelled exactly as the JD spells it, whether you added it or it was already there. Empty list if none.
8. Style: same register as the original - starts with a past-tense action verb (personal projects keep their leading project name and link marker), implied first person, no "I", one paragraph per bullet, no bullet character, no markdown, no trailing commentary.

Return a JSON object with one field, "bullets": a list of objects with exactly the fields "para_id" (string), "text" (string), "used_kb_metrics" (list of strings) and "jd_keywords_used" (list of strings).
"""

FEEDBACK = """
YOUR PREVIOUS ATTEMPT WAS REJECTED. Fix every point below and keep everything else:
{problems}
"""

_PERSONAL_HEADING = re.compile(r"independent|personal|side|freelance|academic|university", re.IGNORECASE)
_STOPWORDS = {"and", "the", "for", "with", "system", "platform", "engine", "based", "model", "project", "projects", "data"}
_SLUG_SAFE = re.compile(r"[^a-z0-9]+")


# ---- grouping ---------------------------------------------------------------------------------------


@dataclass
class BulletGroup:
    """One `exp_meta` sub-heading with the `exp_bullet` paragraphs under it."""

    index: int  # 1-based, document order
    heading: Para
    bullets: list[Para]
    facts: dict[str, ProjectFact | None]  # bullet para_id -> the project it is grounded in

    @property
    def slug(self) -> str:
        return slugify(self.heading.text)

    @property
    def professional(self) -> bool:
        """Work group: any bullet has a work fact, or no fact matched and the heading is not personal."""
        known = [f for f in self.facts.values() if f is not None]
        if known:
            return any(f.professional for f in known)
        return not _PERSONAL_HEADING.search(self.heading.text)

    def bullet(self, para_id: str) -> Para:
        return next(p for p in self.bullets if p.id == para_id)

    def is_personal(self, para_id: str) -> bool:
        fact = self.facts.get(para_id)
        if fact is not None:
            return not fact.professional
        return not self.professional

    def distinct_facts(self) -> list[ProjectFact]:
        seen: list[ProjectFact] = []
        for para in self.bullets:
            fact = self.facts.get(para.id)
            if fact is not None and all(fact.name != s.name for s in seen):
                seen.append(fact)
        return seen


def slugify(text: str, max_words: int = _SLUG_WORDS) -> str:
    parts = [_SLUG_SAFE.sub("_", w.lower()).strip("_") for w in text.split()]
    parts = [p for p in parts if p]
    return "_".join(parts[:max_words]) or "group"


def _name_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if len(t) >= 3 and t not in _STOPWORDS}


def match_fact(heading: str, facts: Sequence[ProjectFact]) -> ProjectFact | None:
    """The project whose name the sub-heading carries (exact, contained, or sharing most name tokens)."""
    wanted = " ".join(heading.split()).lower()
    if not wanted:
        return None
    for fact in facts:
        if fact.name.strip().lower() == wanted:
            return fact
    for fact in facts:
        name = fact.name.strip().lower()
        if name and (name in wanted or wanted in name):
            return fact
    return _best_overlap(_name_tokens(heading), facts)


def match_fact_for_bullet(text: str, facts: Sequence[ProjectFact]) -> ProjectFact | None:
    """For mixed groups ("Independent Projects"): the project whose name tokens the bullet mentions most."""
    return _best_overlap(_name_tokens(text), facts)


def _best_overlap(have: set[str], facts: Sequence[ProjectFact]) -> ProjectFact | None:
    best: tuple[int, ProjectFact] | None = None
    for fact in facts:
        overlap = len(have & _name_tokens(fact.name))
        if overlap >= MIN_NAME_OVERLAP and (best is None or overlap > best[0]):
            best = (overlap, fact)
    return best[1] if best else None


def group_bullets(paras: Sequence[Para], sections: Mapping[str, Section], facts: Sequence[ProjectFact]) -> list[BulletGroup]:
    """Split the experience section into groups; an `exp_meta` line with no bullets (the employer line) is skipped."""
    groups: list[BulletGroup] = []
    heading: Para | None = None
    bullets: list[Para] = []

    def flush() -> None:
        if heading is not None and bullets:
            fact = match_fact(heading.text, facts)
            per_bullet = {b.id: fact or match_fact_for_bullet(b.text, facts) for b in bullets}
            groups.append(BulletGroup(len(groups) + 1, heading, list(bullets), per_bullet))

    for para in paras:
        section = sections.get(para.id)
        if section == "exp_meta":
            flush()
            heading, bullets = para, []
        elif section == "exp_bullet":
            if heading is None:  # bullets before any sub-heading: group them under a synthetic heading
                heading = para
            bullets.append(para)
    flush()
    return groups


# ---- inputs ------------------------------------------------------------------------------------------


@dataclass
class ExperienceInputs:
    """Everything the writer needs, resolved once so the prompt and the checks agree."""

    groups: list[BulletGroup]
    keywords: JDKeywords
    match: SkillMatch
    approved_adjacent: list[str]
    intent: JDIntent
    facts: list[ProjectFact]
    vocabulary: set[str]
    corpus: str = ""  # CV text + knowledge-base text: an ATS phrase is offered only if it literally occurs here
    forbidden: list[str] = field(init=False)
    allowed_keywords: dict[str, str] = field(init=False)  # lower -> JD spelling (matched skills + allowed ATS phrases)
    matched_lower: set[str] = field(init=False)
    check_vocab: set[str] = field(init=False)
    professional_text: str = field(init=False)

    def __post_init__(self) -> None:
        self.approved_adjacent = [s.strip() for s in self.approved_adjacent if s.strip()]
        approved_lower = {s.lower() for s in self.approved_adjacent}
        self.forbidden = [*self.match.missing, *(s for s in self.match.adjacent if s.lower() not in approved_lower)]
        forbidden_lower = {s.lower() for s in self.forbidden}
        self.matched_lower = {s.lower() for s in self.match.matched}
        self.allowed_keywords = {}
        for skill in self.match.matched:
            self.allowed_keywords.setdefault(skill.lower(), skill)
        for phrase in self.keywords.ats_keywords:
            key = phrase.strip().lower()
            if not key or key in self.allowed_keywords or key in forbidden_lower:
                continue
            if all(t in self.vocabulary for t in tokens(phrase)) and contains(self.corpus, phrase):
                self.allowed_keywords[key] = phrase.strip()
        self.check_vocab = {v.lower() for v in self.vocabulary}
        for skill in self.match.matched:
            self.check_vocab |= {t.lower() for t in raw_tokens(skill)}
            self.check_vocab |= {p.lower() for t in raw_tokens(skill) for p in re.split(r"[/_-]", t) if p}
        self.professional_text = " ".join(fact_text(f) for f in self.facts if f.professional)
        if not self.corpus:
            self.corpus = " ".join([*(b.text for b in self.bullets), *(fact_text(f) for f in self.facts)])

    def inventory_term(self, jd_skill: str) -> str:
        return self.match.aliases.get(jd_skill, jd_skill)

    def ats_phrases(self) -> list[str]:
        return [p for k, p in self.allowed_keywords.items() if k not in self.matched_lower]

    @property
    def bullets(self) -> list[Para]:
        return [b for g in self.groups for b in g.bullets]


def fact_text(fact: ProjectFact) -> str:
    return " ".join([*fact.stack, *fact.keywords, fact.built, fact.problem])


def build_inputs(
    paras: Sequence[Para],
    sections: Mapping[str, Section],
    *,
    keywords: JDKeywords,
    match: SkillMatch,
    approved_adjacent: Iterable[str],
    intent: JDIntent,
    facts: Sequence[ProjectFact],
    vocabulary: Iterable[str],
    cv_text: str = "",
) -> ExperienceInputs:
    """Resolve the writer's inputs. `match` must already have the approved adjacent skills applied.

    `cv_text` (the whole CV) widens the corpus that an ATS phrase must literally occur in
    before it is offered to the model; without it only the bullets and the facts count.
    """
    facts = list(facts)
    corpus = " ".join([cv_text, *(fact_text(f) for f in facts)]).strip()
    return ExperienceInputs(
        groups=group_bullets(paras, sections, facts),
        keywords=keywords,
        match=match,
        approved_adjacent=list(approved_adjacent),
        intent=intent,
        facts=facts,
        vocabulary=set(vocabulary),
        corpus=corpus,
    )


# ---- prompt ------------------------------------------------------------------------------------------


def group_case(case: str, group: BulletGroup) -> str:
    """Recording name of one group's call: `<case>__g<n>_<heading slug>`."""
    return f"{case}__g{group.index}_{group.slug}"


def default_case(inputs: ExperienceInputs) -> str:
    payload = json.dumps(
        [[b.text for b in inputs.bullets], inputs.match.model_dump(), sorted(s.lower() for s in inputs.approved_adjacent)],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _pairs(skills: Iterable[str], inputs: ExperienceInputs) -> str:
    lines = []
    for skill in skills:
        term = inputs.inventory_term(skill)
        lines.append(f"- {skill}" if term.lower() == skill.lower() else f"- {skill} -> {term}")
    return "\n".join(lines) or "- (none)"


def _evidence(group: BulletGroup) -> str:
    blocks = []
    for fact in group.distinct_facts():
        built = " ".join(fact.built.split())
        if len(built) > _MAX_BUILT_CHARS:
            built = built[: _MAX_BUILT_CHARS - 3].rstrip() + "..."
        head = f"- {fact.name}" + (f" (work, {fact.employer})" if fact.professional else " (personal project)")
        lines = [head]
        if built:
            lines.append(f"  built: {built}")
        lines.append(
            "  metrics you may add (copy verbatim): " + "; ".join(fact.metrics)
            if fact.metrics
            else "  metrics you may add: none - add no numbers to these bullets"
        )
        if fact.stack:
            lines.append(f"  stack: {', '.join(fact.stack)}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) or "- (no project notes for this group; add no numbers and use only the original wording's facts)"


def _bullet_lines(group: BulletGroup) -> str:
    lines = []
    for i, para in enumerate(group.bullets, 1):
        fact = group.facts.get(para.id)
        project = fact.name + (" (work)" if fact.professional else " (personal)") if fact else "(no notes)"
        lo, hi = word_bounds(para.words, WORD_TOLERANCE)
        numbers = ", ".join(extract_numbers(para.text)) or "none"
        lines.append(
            f'{i}. para_id "{para.id}" | project: {project} | {para.words} words, write between {lo} and {hi} | numbers to keep: {numbers}\n'
            f"   <<<\n   {para.text.strip()}\n   >>>"
        )
    return "\n".join(lines)


def build_prompt(group: BulletGroup, inputs: ExperienceInputs, problems: Sequence[str] = ()) -> str:
    feedback = FEEDBACK.format(problems="\n".join(f"- {p}" for p in problems)) if problems else ""
    return PROMPT.format(
        role_summary=inputs.intent.role_summary.strip(),
        seniority=inputs.intent.seniority,
        domain=inputs.intent.domain.strip(),
        tone=inputs.intent.tone.strip(),
        priorities="\n".join(f"  {i}. {p}" for i, p in enumerate(inputs.intent.top_priorities, 1)),
        matched=_pairs(inputs.match.matched, inputs),
        phrases="\n".join(f"- {p}" for p in inputs.ats_phrases()) or "- (none)",
        forbidden="\n".join(f"- {s}" for s in inputs.forbidden) or "- (none)",
        evidence=_evidence(group),
        heading=" ".join(group.heading.text.split()),
        kind="work project - rewrite every bullet" if group.professional else "personal projects - unchanged unless rule 6 applies",
        count=len(group.bullets),
        bullets=_bullet_lines(group),
        feedback=feedback,
    )


# ---- checks ------------------------------------------------------------------------------------------


def normalise_text(text: str) -> str:
    """Whitespace-collapsed body without any bullet character the model may have echoed."""
    _, body = split_bullet_prefix(text.strip())
    return " ".join(body.split())


def canonical_metric(entry: str, fact: ProjectFact | None) -> str | None:
    """The `fact.metrics` item `entry` names (exactly, or as its unique substring), else None."""
    if fact is None:
        return None
    wanted = " ".join(entry.split()).lower()
    if not wanted:
        return None
    for metric in fact.metrics:
        if metric.lower() == wanted:
            return metric
    hits = [m for m in fact.metrics if wanted in m.lower()]
    return hits[0] if len(hits) == 1 else None


def clean_keywords(text: str, keywords: Iterable[str], inputs: ExperienceInputs) -> list[str]:
    """JD keywords the candidate has, spelled as the JD spells them, that really appear in `text`.

    The model's own list is filtered; every matched JD skill present in the text is added.
    """
    out: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        key = " ".join(keyword.split()).lower()
        spelled = inputs.allowed_keywords.get(key)
        if spelled and key not in seen and contains(text, spelled):
            seen.add(key)
            out.append(spelled)
    for skill in inputs.match.matched:
        key = skill.lower()
        if key not in seen and contains(text, skill):
            seen.add(key)
            out.append(skill)
    return out


def _evidenced(keyword: str, evidence: str, inputs: ExperienceInputs) -> bool:
    """`keyword` (or the candidate's own term for it) occurs in `evidence`."""
    if contains(evidence, keyword):
        return True
    alias = inputs.match.aliases.get(keyword) or inputs.match.aliases.get(inputs.allowed_keywords.get(keyword.lower(), keyword))
    return bool(alias) and contains(evidence, alias)


def check_bullet(rewrite: BulletRewrite, group: BulletGroup, inputs: ExperienceInputs) -> list[str]:
    """Every rule the rewritten bullet breaks, phrased as feedback for the model."""
    para = group.bullet(rewrite.para_id)
    fact = group.facts.get(para.id)
    orig = para.text
    new = normalise_text(rewrite.text)
    problems: list[str] = []
    if not new:
        return ["The text is empty; return the rewritten bullet."]

    lo, hi = word_bounds(para.words, WORD_TOLERANCE)
    if not within_word_budget(new, orig, WORD_TOLERANCE):
        problems.append(f"It has {words(new)} words; it must have between {lo} and {hi}.")

    lost = missing_numbers(new, orig)
    if lost:
        problems.append(f"These numbers from the original are missing and must appear verbatim: {', '.join(lost)}.")

    allowed_numbers = set(extract_numbers(orig))
    for entry in rewrite.used_kb_metrics:
        metric = canonical_metric(entry, fact)
        if metric is None:
            available = "; ".join(fact.metrics) if fact and fact.metrics else "none"
            problems.append(
                f"used_kb_metrics entry '{entry}' is not in this project's metrics list ({available}); "
                "remove it and any number it brought into the text."
            )
        else:
            allowed_numbers |= set(extract_numbers(metric))
    invented = sorted({n for n in extract_numbers(new) if n not in allowed_numbers})
    if invented:
        problems.append(
            f"These numbers are not in the original bullet and not a metric from this project's notes reported in used_kb_metrics: "
            f"{', '.join(invented)}. Remove them."
        )

    terms = suspicious_terms(new, inputs.check_vocab)
    if terms:
        problems.append(
            "These capitalised terms are not in the candidate's CV, project notes or matched skills and must be removed: "
            + ", ".join(terms)
            + "."
        )
    mentioned = forbidden_mentions(new, inputs.forbidden, orig)
    if mentioned:
        problems.append(f"It mentions skills the candidate does not have: {', '.join(mentioned)}.")

    keywords = clean_keywords(new, rewrite.jd_keywords_used, inputs)
    # Without notes for this project, the whole CV + knowledge base is the evidence.
    evidence = " ".join([orig, fact_text(fact) if fact else inputs.corpus])
    unsupported = [k for k in keywords if not contains(orig, k) and not _evidenced(k, evidence, inputs)]
    if unsupported:
        problems.append(
            "These JD keywords are not evidenced by this bullet's project notes and must be removed "
            f"(keep the original wording for that part): {', '.join(unsupported)}."
        )

    if group.is_personal(para.id) and new != " ".join(orig.split()):
        evidence = fact_text(fact) if fact else orig
        justified = [
            k
            for k in keywords
            if k.lower() in inputs.matched_lower
            and not contains(orig, k)  # a keyword the original already had cannot justify a change
            and contains(evidence, k)
            and not contains(inputs.professional_text, k)
        ]
        if not justified:
            problems.append(
                "This is a personal-project bullet: return it unchanged. None of the JD skills it uses "
                f"({', '.join(keywords) or 'none'}) is new here and applies only to this project."
            )
    return problems


def finalize_bullet(rewrite: BulletRewrite, group: BulletGroup, inputs: ExperienceInputs) -> tuple[BulletRewrite, list[str]]:
    """The checked bullet with canonical metrics and cleaned keywords, plus notes about dropped items."""
    para = group.bullet(rewrite.para_id)
    fact = group.facts.get(para.id)
    new = normalise_text(rewrite.text)
    notes: list[str] = []
    metrics: list[str] = []
    for entry in rewrite.used_kb_metrics:
        metric = canonical_metric(entry, fact)
        if metric is None or metric in metrics:
            continue
        numbers = extract_numbers(metric)
        used = contains(new, metric) or any(n in extract_numbers(new) for n in numbers)
        if used:
            metrics.append(metric)
        else:
            notes.append(f"Dropped reported metric '{metric}' for bullet {para.id}: it does not appear in the rewritten text.")
    unchanged = new == " ".join(para.text.split())
    if unchanged:
        new = para.text  # byte-identical to the document, whatever whitespace the model echoed
    keywords = [] if unchanged and group.is_personal(para.id) else clean_keywords(new, rewrite.jd_keywords_used, inputs)
    return BulletRewrite(para_id=para.id, text=new, used_kb_metrics=metrics, jd_keywords_used=keywords), notes


def original_bullet(para: Para) -> BulletRewrite:
    return BulletRewrite(para_id=para.id, text=para.text)


# ---- run ---------------------------------------------------------------------------------------------


@dataclass
class _GroupOutcome:
    bullets: list[BulletRewrite]
    reverted: list[str]
    notes: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


def _rewrite_group(group: BulletGroup, inputs: ExperienceInputs, case: str) -> _GroupOutcome:
    name = group_case(case, group)
    heading = " ".join(group.heading.text.split())
    outcome = _GroupOutcome([], [], [])
    accepted: dict[str, BulletRewrite] = {}
    last_problems: dict[str, list[str]] = {}
    problems: list[str] = []

    for attempt, rec in enumerate((name, f"{name}_retry")):
        raw = llm.generate_json(build_prompt(group, inputs, problems), BulletGroupRewrite, case=f"{AGENT}/{rec}")
        outcome.calls += 1
        usage = llm.last_usage()
        if usage:
            outcome.input_tokens += usage.input_tokens
            outcome.output_tokens += usage.output_tokens

        by_id: dict[str, BulletRewrite] = {}
        for entry in raw.bullets:
            by_id.setdefault(entry.para_id.strip(), entry)
        problems = []
        for i, para in enumerate(group.bullets, 1):
            if para.id in accepted:
                continue
            entry = by_id.get(para.id)
            if entry is None:
                bullet_problems = [f'no entry with para_id "{para.id}" was returned; return one entry per bullet.']
            else:
                bullet_problems = check_bullet(entry, group, inputs)
            if bullet_problems:
                last_problems[para.id] = bullet_problems
                problems.extend(f"Bullet {i} ({para.id}): {p}" for p in bullet_problems)
            else:
                accepted[para.id], notes = finalize_bullet(entry, group, inputs)
                outcome.notes.extend(notes)
        if not problems:
            break
        if attempt == 0:
            outcome.notes.append(f"Retried '{heading}' once with feedback: " + " ".join(problems))

    for para in group.bullets:
        if para.id in accepted:
            outcome.bullets.append(accepted[para.id])
        else:
            outcome.reverted.append(para.id)
            outcome.bullets.append(original_bullet(para))
            outcome.notes.append(
                f"Bullet kept as the original ('{para.text[:50].rstrip()}...'): the rewrite still broke a rule after one retry "
                f"({' '.join(last_problems.get(para.id, []))})."
            )
    return outcome


def run(inputs: ExperienceInputs, *, case: str | None = None) -> ExperienceResult:
    """Rewrite every experience bullet, one LLM call per project group, groups in parallel.

    `case` names the recordings (`tests/recordings/experience/<case>__g<n>_<slug>.json`).
    Build `inputs` with `build_inputs`; `inputs.match` must already have the approved
    adjacent skills applied (`matching.apply_approvals`).
    """
    originals = {b.id: b.text for b in inputs.bullets}
    projects = {b.id: (g.facts.get(b.id).name if g.facts.get(b.id) else "") for g in inputs.groups for b in g.bullets}
    if not inputs.groups:
        return ExperienceResult(bullets=[], originals={}, projects={})
    case = case or default_case(inputs)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(inputs.groups))) as pool:
        outcomes = list(pool.map(lambda g: _rewrite_group(g, inputs, case), inputs.groups))

    bullets: list[BulletRewrite] = []
    reverted: list[str] = []
    notes: list[str] = []
    in_tokens = out_tokens = calls = 0
    for outcome in outcomes:
        bullets.extend(outcome.bullets)
        reverted.extend(outcome.reverted)
        notes.extend(outcome.notes)
        in_tokens += outcome.input_tokens
        out_tokens += outcome.output_tokens
        calls += outcome.calls

    # Professional first: personal bullets change only once every work bullet uses a JD keyword.
    personal_ids = {b.id for g in inputs.groups for b in g.bullets if g.is_personal(b.id)}
    work_all_keyworded = all(b.jd_keywords_used for b in bullets if b.para_id not in personal_ids)
    if not work_all_keyworded:
        for index, bullet in enumerate(bullets):
            if bullet.para_id in personal_ids and bullet.text != originals[bullet.para_id]:
                bullets[index] = BulletRewrite(para_id=bullet.para_id, text=originals[bullet.para_id])
                reverted.append(bullet.para_id)
                notes.append(
                    f"Personal-project bullet {bullet.para_id} kept as the original: professional bullets are rewritten first, "
                    "and not every work bullet uses a JD keyword yet."
                )

    return ExperienceResult(
        bullets=bullets,
        originals=originals,
        projects=projects,
        reverted=reverted,
        notes=notes,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        calls=calls,
    )
