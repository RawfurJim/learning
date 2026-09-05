"""Orchestration: JD analysis -> matching + ranking -> rewrites -> assembled .docx.

`analyse` runs everything that does not depend on the user's approvals (Agents 1 and 2,
inventory, matching, project ranking) and is what the app's Analyse button shows.
`run` takes the approved adjacent skills and the stages to rewrite, calls the writer
agents (summary & skills and experience in parallel, `ThreadPoolExecutor`), applies the
rewrites with the assembler and returns a `RunResult` holding the analysis, the
before/after texts, notes and the output bytes. `assemble` rebuilds the document from a
`RunResult` minus the rewrites the user rejected.

Caching (SCRUM-15): `analyse` and `run` key their result on
sha256(cv_bytes + jd_text + kb_text + json(settings)) and serve a hit from `.cache/`
without any LLM call (`use_cache=False` bypasses it; `LLM_MODE=record` never uses it so
recordings always come from a real call). A cached `RunResult` has `from_cache=True` and
the same output bytes as the run that produced it. `RunResult.usage` sums the tokens of
every agent call and prices them with `llm.estimate_cost` (0 in replay mode).

Length is enforced twice: per paragraph inside the writers (+/-10% words) and for the
whole document here (+/-3% characters). If the assembled document is still too long
or too short, rewrites are reverted one at a time (skills first, then summary, then the
bullets from the last one backwards) until it fits; every reversion is written to
`RunResult.notes`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from resume_tailor import assembler, cache, llm, matching
from resume_tailor.agents import default_case, experience_writer, jd_intent, keywords, summary_skills_writer
from resume_tailor.ats_score import coverage
from resume_tailor.docx_io import Para, iter_paragraphs, load
from resume_tailor.knowledge import allowed_vocabulary, load_projects
from resume_tailor.schemas import (
    ExperienceResult,
    JDIntent,
    JDKeywords,
    ProjectFact,
    SkillMatch,
    SummarySkillsResult,
    TokenUsage,
)
from resume_tailor.sections import Section, classify
from resume_tailor.settings import Settings

REWRITE_STAGES = ("summary", "skills", "experience")
ALL_STAGES = list(REWRITE_STAGES)
PARAGRAPH_STAGES = ("summary", "skills")  # one paragraph each; "experience" covers every exp_bullet


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
    from_cache: bool = False


class RunResult(BaseModel):
    """Output of `run`: the analysis, what changed, why, and the tailored document."""

    # bytes fields (source, output) are base64 in JSON so a cached result round-trips exactly.
    model_config = ConfigDict(arbitrary_types_allowed=True, ser_json_bytes="base64", val_json_bytes="base64")

    case: str
    stages: list[str]
    intent: JDIntent
    keywords: JDKeywords
    match: SkillMatch  # after approvals
    ranked: list[tuple[ProjectFact, float]]
    approved_adjacent: list[str]
    sections: dict[str, str]  # para_id -> "summary" | "skills" | "exp_bullet" for the editable paragraphs
    originals: dict[str, str]  # para_id -> original text
    rewrites: dict[str, str]  # para_id -> text actually applied (only changed paragraphs)
    notes: list[str]
    coverage_before: float
    coverage_after: float
    source: bytes  # the original CV, so `assemble` can rebuild the output after accept/reject
    output: bytes
    usage: TokenUsage
    writer: SummarySkillsResult | None = None  # the summary & skills agent's checked output
    experience: ExperienceResult | None = None  # the experience writer's checked output
    from_cache: bool = False  # True when served from `.cache/` instead of fresh agent calls

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

    def priced(self, model: str) -> TokenUsage:
        """The totals with `model` and its estimated GBP cost filled in."""
        total = self.total.model_copy(update={"model": model})
        return total.model_copy(update={"estimated_cost_gbp": llm.estimate_cost(total)})


def _kb_text(kb_path: str | Path) -> str:
    path = Path(kb_path)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


class _Cache:
    """Cache lookup for one call: the key covers the inputs plus everything that changes the output."""

    def __init__(
        self,
        kind: str,
        cv_bytes: bytes,
        jd_text: str,
        kb_path: str | Path,
        *,
        case: str | None,
        approved: Sequence[str] = (),
        stages: Sequence[str] = (),
        enabled: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        settings = Settings.from_env()
        self.enabled = enabled and settings.llm_mode != "record"
        self.dir = cache_dir or settings.cache_dir
        self.key = cache.cache_key(
            cv_bytes,
            jd_text,
            _kb_text(kb_path),
            {
                "kind": kind,
                "version": cache.CACHE_VERSION,
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "mode": settings.llm_mode,
                "case": case or "",
                "approved_adjacent": sorted(approved),
                "stages": list(stages),
            },
        )

    def get[T: BaseModel](self, model: type[T]) -> T | None:
        return cache.get(self.key, model, self.dir) if self.enabled else None

    def put(self, result: BaseModel) -> None:
        if self.enabled:
            cache.put(self.key, result, self.dir)


def _editable(paras: Sequence[Para], sections: dict[str, Section]) -> dict[str, Para]:
    """`{"summary": Para, "skills": Para}` for the first paragraph of each single-paragraph section."""
    found: dict[str, Para] = {}
    for para in paras:
        section = sections[para.id]
        if section in PARAGRAPH_STAGES and section not in found:
            found[section] = para
    return found


def _bullets(paras: Sequence[Para], sections: dict[str, Section]) -> list[Para]:
    return [para for para in paras if sections[para.id] == "exp_bullet"]


def analyse(
    cv_bytes: bytes,
    jd_text: str,
    kb_path: str | Path,
    *,
    case: str | None = None,
    jd_analysis: tuple[JDIntent, JDKeywords] | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> Analysis:
    """Agents 1 + 2 (or the given `jd_analysis`), inventory, matching and ranking for one CV + JD."""
    case = case or default_case(jd_text)
    cached = _Cache("analysis", cv_bytes, jd_text, kb_path, case=case, enabled=use_cache, cache_dir=cache_dir)
    hit = cached.get(Analysis)
    if hit is not None:
        return hit.model_copy(update={"from_cache": True})
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
    analysis = Analysis(
        case=case,
        intent=intent,
        keywords=result,
        match=match,
        ranked=ranked,
        inventory=sorted(inventory, key=str.lower),
        cv_text="\n".join(p.full_text for p in paras),
        usage=meter.priced(Settings.from_env().llm_model),
    )
    cached.put(analysis)
    return analysis


def experience_inputs(
    cv_bytes: bytes,
    analysis: Analysis,
    kb_path: str | Path,
    approved_adjacent: Iterable[str] = (),
) -> experience_writer.ExperienceInputs:
    """The experience writer's resolved inputs (groups, facts, vocabulary) for one CV + analysis."""
    approved = [s.strip() for s in approved_adjacent if s.strip()]
    paras = iter_paragraphs(load(cv_bytes))
    facts = load_projects(kb_path)
    return experience_writer.build_inputs(
        paras,
        classify(paras),
        keywords=analysis.keywords,
        match=matching.apply_approvals(analysis.match, approved),
        approved_adjacent=approved,
        intent=analysis.intent,
        facts=facts,
        vocabulary=allowed_vocabulary(analysis.cv_text, facts),
        cv_text=analysis.cv_text,
    )


def run_experience(
    cv_bytes: bytes,
    analysis: Analysis,
    kb_path: str | Path,
    approved_adjacent: Iterable[str] = (),
    *,
    case: str | None = None,
) -> ExperienceResult:
    """Only the experience writer, named exactly as `run` names it (so recordings are shared)."""
    approved = [s.strip() for s in approved_adjacent if s.strip()]
    inputs = experience_inputs(cv_bytes, analysis, kb_path, approved)
    return experience_writer.run(inputs, case=summary_skills_writer.case_for(case or analysis.case, approved))


def assemble(result: RunResult, rejected: Iterable[str] = ()) -> bytes:
    """The tailored .docx with every rewrite in `result` except the `rejected` paragraph ids."""
    rejected = set(rejected)
    chosen = {para_id: text for para_id, text in result.rewrites.items() if para_id not in rejected}
    return assembler.apply(result.source, chosen) if chosen else result.source


def run(
    cv_bytes: bytes,
    jd_text: str,
    kb_path: str | Path,
    approved_adjacent: Iterable[str] = (),
    stages: Iterable[str] = ALL_STAGES,
    *,
    case: str | None = None,
    analysis: Analysis | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> RunResult:
    """Tailor `cv_bytes` to `jd_text`. `stages` is a subset of {"summary", "skills", "experience"}.

    `analysis` (from `analyse`) is reused when given, so approving a skill and pressing
    Rewrite does not re-run the JD agents. `case` names every recording of the run.
    An identical earlier run (same CV, JD, knowledge base, settings, approvals and stages)
    is returned from the cache with no LLM call unless `use_cache=False` or `LLM_MODE=record`.
    """
    stages = [s for s in stages if s in REWRITE_STAGES]
    approved = [s.strip() for s in approved_adjacent if s.strip()]
    cached = _Cache(
        "run",
        cv_bytes,
        jd_text,
        kb_path,
        case=case or (analysis.case if analysis else None),
        approved=approved,
        stages=stages,
        enabled=use_cache,
        cache_dir=cache_dir,
    )
    hit = cached.get(RunResult)
    if hit is not None:
        return hit.model_copy(update={"from_cache": True})
    if analysis is None:
        analysis = analyse(cv_bytes, jd_text, kb_path, case=case, use_cache=use_cache, cache_dir=cache_dir)
    case = analysis.case
    meter = _UsageMeter()
    meter.add(analysis.usage.input_tokens, analysis.usage.output_tokens, analysis.usage.calls)

    paras = iter_paragraphs(load(cv_bytes))
    sections = classify(paras)
    editable = _editable(paras, sections)
    bullets = _bullets(paras, sections)
    match = matching.apply_approvals(analysis.match, approved)
    facts = load_projects(kb_path)
    ranked_facts = [fact for fact, _ in analysis.ranked]
    vocabulary = allowed_vocabulary(analysis.cv_text, facts)
    notes: list[str] = []
    rewrites: dict[str, str] = {}
    writer: SummarySkillsResult | None = None
    experience: ExperienceResult | None = None

    wanted = [s for s in stages if s in editable]
    for stage in stages:
        if stage in PARAGRAPH_STAGES and stage not in editable:
            notes.append(f"No {stage} paragraph found in the CV; nothing to rewrite for that stage.")
    want_experience = "experience" in stages
    if want_experience and not bullets:
        notes.append("No experience bullets found in the CV; nothing to rewrite for that stage.")
        want_experience = False
    writer_case = summary_skills_writer.case_for(case, approved)

    # Agents 3 and 4 in parallel; each is independent and reads only its own LLM usage.
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer_future = None
        if wanted:
            summary_para, skills_para = editable.get("summary"), editable.get("skills")
            writer_future = pool.submit(
                summary_skills_writer.run,
                summary_para.text if summary_para else "",
                skills_para.text if skills_para else "",
                keywords=analysis.keywords,
                match=match,
                approved_adjacent=approved,
                intent=analysis.intent,
                facts=ranked_facts or facts,
                inventory=analysis.inventory,
                vocabulary=vocabulary,
                case=writer_case,
            )
        experience_future = None
        if want_experience:
            experience_future = pool.submit(
                experience_writer.run,
                experience_inputs(cv_bytes, analysis, kb_path, approved),
                case=writer_case,
            )
        if writer_future is not None:
            writer = writer_future.result()
        if experience_future is not None:
            experience = experience_future.result()

    if writer is not None:
        summary_para, skills_para = editable.get("summary"), editable.get("skills")
        meter.add(writer.input_tokens, writer.output_tokens, writer.calls)
        notes.extend(writer.notes)
        if "summary" in wanted and summary_para and writer.summary != summary_para.text:
            rewrites[summary_para.id] = writer.summary
        if "skills" in wanted and skills_para:
            line = assembler.join_skills(writer.skills, summary_skills_writer.delimiter_of(skills_para.text))
            if line != skills_para.text:
                rewrites[skills_para.id] = line
    if experience is not None:
        meter.add(experience.input_tokens, experience.output_tokens, experience.calls)
        notes.extend(experience.notes)
        rewrites.update(experience.changed)

    output = assembler.apply(cv_bytes, rewrites) if rewrites else cv_bytes
    # Whole-document length guard: revert skills, then summary, then bullets (last first) until it fits.
    candidates: list[tuple[str, str]] = []
    for stage in PARAGRAPH_STAGES:
        if stage in editable:
            candidates.append((stage, editable[stage].id))
    candidates += [("experience bullet", para.id) for para in reversed(bullets)]
    for label, para_id in candidates:
        problems = assembler.check_layout(cv_bytes, output, rewrites)
        if not problems:
            break
        if para_id in rewrites:
            rewrites.pop(para_id)
            notes.append(f"Reverted the {label} rewrite ({para_id}) to keep the document within +/-3% characters ({'; '.join(problems)}).")
            output = assembler.apply(cv_bytes, rewrites) if rewrites else cv_bytes

    section_by_id = {para.id: section for section, para in editable.items()}
    originals = {para.id: para.text for para in editable.values()}
    if experience is not None:
        section_by_id.update({para.id: "exp_bullet" for para in bullets})
        originals.update(experience.originals)

    out_text = "\n".join(p.full_text for p in iter_paragraphs(load(output)))
    result = RunResult(
        case=case,
        stages=stages,
        intent=analysis.intent,
        keywords=analysis.keywords,
        match=match,
        ranked=analysis.ranked,
        approved_adjacent=approved,
        sections=section_by_id,
        originals=originals,
        rewrites=rewrites,
        notes=notes,
        coverage_before=coverage(analysis.cv_text, analysis.keywords.ats_keywords),
        coverage_after=coverage(out_text, analysis.keywords.ats_keywords),
        source=cv_bytes,
        output=output,
        usage=meter.priced(Settings.from_env().llm_model),
        writer=writer,
        experience=experience,
    )
    cached.put(result)
    return result
