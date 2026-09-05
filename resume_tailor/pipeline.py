"""Orchestration: JD analysis -> matching + ranking -> rewrites -> assembled .docx.

`analyse` runs everything that does not depend on the user's approvals (Agents 1 and 2,
inventory, matching, project ranking) and is what the app's Analyse button shows.
`run` takes the approved adjacent skills and the stages to rewrite, calls the writer
agents, applies the rewrites with the assembler and returns a `RunResult` holding the
analysis, the before/after texts, notes and the output bytes.

Length is enforced twice: per paragraph inside the writer (+/-10% words) and for the
whole document here (+/-3% characters). If the assembled document is still too long
or too short, rewrites are reverted one at a time (skills first, then summary) until it
fits; every reversion is written to `RunResult.notes`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from resume_tailor import assembler, llm, matching
from resume_tailor.agents import default_case, jd_intent, keywords, summary_skills_writer
from resume_tailor.ats_score import coverage
from resume_tailor.docx_io import Para, iter_paragraphs, load
from resume_tailor.knowledge import allowed_vocabulary, load_projects
from resume_tailor.schemas import JDIntent, JDKeywords, ProjectFact, SkillMatch, SummarySkillsResult, TokenUsage
from resume_tailor.sections import Section, classify

REWRITE_STAGES = ("summary", "skills")
ALL_STAGES = list(REWRITE_STAGES)


class Analysis(BaseModel):
    """Everything computed before the user approves adjacent skills."""

    case: str
    intent: JDIntent
    keywords: JDKeywords
    match: SkillMatch  # before approvals: adjacent skills are still suggestions
    ranked: list[tuple[ProjectFact, float]]
    inventory: list[str]
    cv_text: str
    usage: TokenUsage = TokenUsage()


class RunResult(BaseModel):
    """Output of `run`: the analysis, what changed, why, and the tailored document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case: str
    stages: list[str]
    intent: JDIntent
    keywords: JDKeywords
    match: SkillMatch  # after approvals
    ranked: list[tuple[ProjectFact, float]]
    approved_adjacent: list[str]
    sections: dict[str, str]  # para_id -> "summary" | "skills" for the editable paragraphs
    originals: dict[str, str]  # para_id -> original text
    rewrites: dict[str, str]  # para_id -> text actually applied (only changed paragraphs)
    notes: list[str]
    coverage_before: float
    coverage_after: float
    output: bytes
    usage: TokenUsage
    writer: SummarySkillsResult | None = None  # the summary & skills agent's checked output

    @property
    def changed(self) -> bool:
        return bool(self.rewrites)


class _UsageMeter:
    """Sums `llm.last_usage()` across calls; call `tick()` right after each agent call."""

    def __init__(self) -> None:
        self.total = TokenUsage()
        llm.reset_usage()

    def tick(self) -> None:
        usage = llm.last_usage()
        if usage is None:
            return
        self.total = TokenUsage(
            input_tokens=self.total.input_tokens + usage.input_tokens,
            output_tokens=self.total.output_tokens + usage.output_tokens,
            calls=self.total.calls + 1,
        )
        llm.reset_usage()

    def add(self, input_tokens: int, output_tokens: int, calls: int) -> None:
        self.total = TokenUsage(
            input_tokens=self.total.input_tokens + input_tokens,
            output_tokens=self.total.output_tokens + output_tokens,
            calls=self.total.calls + calls,
        )


def _editable(paras: Sequence[Para], sections: dict[str, Section]) -> dict[str, Para]:
    """`{"summary": Para, "skills": Para}` for the first paragraph of each rewritable section."""
    found: dict[str, Para] = {}
    for para in paras:
        section = sections[para.id]
        if section in REWRITE_STAGES and section not in found:
            found[section] = para
    return found


def analyse(
    cv_bytes: bytes,
    jd_text: str,
    kb_path: str | Path,
    *,
    case: str | None = None,
    jd_analysis: tuple[JDIntent, JDKeywords] | None = None,
) -> Analysis:
    """Agents 1 + 2 (or the given `jd_analysis`), inventory, matching and ranking for one CV + JD."""
    case = case or default_case(jd_text)
    meter = _UsageMeter()
    if jd_analysis is None:
        intent = jd_intent.run(jd_text, case=case)
        meter.tick()
        result = keywords.run(jd_text, case=case)
        meter.tick()
    else:
        intent, result = jd_analysis

    paras = iter_paragraphs(load(cv_bytes))
    facts = load_projects(kb_path)
    inventory = matching.build_inventory(paras, facts)
    match = matching.match(result, inventory, facts, case=case)
    meter.tick()
    ranked = matching.rank_projects(intent, facts, case=case)
    meter.tick()
    return Analysis(
        case=case,
        intent=intent,
        keywords=result,
        match=match,
        ranked=ranked,
        inventory=sorted(inventory, key=str.lower),
        cv_text="\n".join(p.full_text for p in paras),
        usage=meter.total,
    )


def run(
    cv_bytes: bytes,
    jd_text: str,
    kb_path: str | Path,
    approved_adjacent: Iterable[str] = (),
    stages: Iterable[str] = ALL_STAGES,
    *,
    case: str | None = None,
    analysis: Analysis | None = None,
) -> RunResult:
    """Tailor `cv_bytes` to `jd_text`. `stages` is a subset of {"summary", "skills"}.

    `analysis` (from `analyse`) is reused when given, so approving a skill and pressing
    Rewrite does not re-run the JD agents. `case` names every recording of the run.
    """
    stages = [s for s in stages if s in REWRITE_STAGES]
    approved = [s.strip() for s in approved_adjacent if s.strip()]
    if analysis is None:
        analysis = analyse(cv_bytes, jd_text, kb_path, case=case)
    case = analysis.case
    meter = _UsageMeter()
    meter.add(analysis.usage.input_tokens, analysis.usage.output_tokens, analysis.usage.calls)

    paras = iter_paragraphs(load(cv_bytes))
    sections = classify(paras)
    editable = _editable(paras, sections)
    match = matching.apply_approvals(analysis.match, approved)
    facts = load_projects(kb_path)
    ranked_facts = [fact for fact, _ in analysis.ranked]
    notes: list[str] = []
    rewrites: dict[str, str] = {}
    writer: SummarySkillsResult | None = None

    wanted = [s for s in stages if s in editable]
    for stage in stages:
        if stage not in editable:
            notes.append(f"No {stage} paragraph found in the CV; nothing to rewrite for that stage.")

    if wanted:
        summary_para, skills_para = editable.get("summary"), editable.get("skills")
        writer = summary_skills_writer.run(
            summary_para.text if summary_para else "",
            skills_para.text if skills_para else "",
            keywords=analysis.keywords,
            match=match,
            approved_adjacent=approved,
            intent=analysis.intent,
            facts=ranked_facts or facts,
            inventory=analysis.inventory,
            vocabulary=allowed_vocabulary(analysis.cv_text, facts),
            case=summary_skills_writer.case_for(case, approved),
        )
        meter.add(writer.input_tokens, writer.output_tokens, writer.calls)
        notes.extend(writer.notes)
        if "summary" in wanted and summary_para and writer.summary != summary_para.text:
            rewrites[summary_para.id] = writer.summary
        if "skills" in wanted and skills_para:
            line = assembler.join_skills(writer.skills, summary_skills_writer.delimiter_of(skills_para.text))
            if line != skills_para.text:
                rewrites[skills_para.id] = line

    output = assembler.apply(cv_bytes, rewrites) if rewrites else cv_bytes
    # Whole-document length guard: revert skills, then summary, until it fits.
    for stage in ("skills", "summary"):
        problems = assembler.check_layout(cv_bytes, output, rewrites)
        if not problems:
            break
        para = editable.get(stage)
        if para and para.id in rewrites:
            rewrites.pop(para.id)
            notes.append(f"Reverted the {stage} rewrite to keep the document within +/-3% characters ({'; '.join(problems)}).")
            output = assembler.apply(cv_bytes, rewrites) if rewrites else cv_bytes

    out_text = "\n".join(p.full_text for p in iter_paragraphs(load(output)))
    return RunResult(
        case=case,
        stages=stages,
        intent=analysis.intent,
        keywords=analysis.keywords,
        match=match,
        ranked=analysis.ranked,
        approved_adjacent=approved,
        sections={para.id: section for section, para in editable.items()},
        originals={para.id: para.text for para in editable.values()},
        rewrites=rewrites,
        notes=notes,
        coverage_before=coverage(analysis.cv_text, analysis.keywords.ats_keywords),
        coverage_after=coverage(out_text, analysis.keywords.ats_keywords),
        output=output,
        usage=meter.total,
        writer=writer,
    )
