"""ResumeTailor Streamlit UI.

Paste a job description to see what it wants and its ATS keywords (SCRUM-11). Upload
the CV (.docx) as well to see which JD skills are Matched, which are Adjacent
suggestions (opt-in checkboxes, never applied on their own), which are Missing (never
added), the knowledge-base projects ranked for this JD, and the ATS keyword coverage
of the current CV (SCRUM-12). Press "Rewrite CV" to get the professional summary and
the skills line rewritten in the JD's vocabulary, compare before/after and download the
tailored .docx with the layout untouched (SCRUM-13). Every experience bullet is rewritten
too, shown in an original | new | KB metrics | keywords table with an accept checkbox per
bullet; the download only contains the bullets you accepted (SCRUM-14). The sidebar picks
the LLM provider (Gemini / Groq / Ollama) and model, shows the tokens and estimated cost of
the last run and of the session, and clears the run cache; an identical Rewrite is served
from `.cache/` without any LLM call (SCRUM-15).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from resume_tailor import cache, llm, matching, pipeline
from resume_tailor.agents import jd_intent, keywords
from resume_tailor.ats_score import coverage, missing_keywords
from resume_tailor.docx_io import iter_paragraphs, load
from resume_tailor.llm import ConfigError, LLMOutputError, ProviderError, RecordingMissing
from resume_tailor.schemas import JDIntent, JDKeywords, ProjectFact, SkillMatch, TokenUsage
from resume_tailor.settings import DEFAULT_MODELS, PROVIDERS, REPO_ROOT, Settings

DEFAULT_KB_PATH = REPO_ROOT / "knowledge" / "projects.md"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
REWRITE_STAGES = ["summary", "skills", "experience"]
LLM_ERRORS = (ConfigError, LLMOutputError, ProviderError)
ACCEPT_KEY = "accept::"  # st.session_state[f"accept::{para_id}"] -> bool, one per rewritten bullet


def analyse(jd_text: str, case: str | None) -> tuple[JDIntent, JDKeywords]:
    """Agents 1 and 2 on the same JD. `case` names the recording used in record/replay mode."""
    return jd_intent.run(jd_text, case=case), keywords.run(jd_text, case=case)


def tailor(
    cv_bytes: bytes,
    jd_text: str,
    intent: JDIntent,
    result: JDKeywords,
    kb_path: Path,
    case: str | None,
) -> pipeline.Analysis:
    """Deterministic matching + the two recorded calls (alias, project ranking), reused by Rewrite."""
    return pipeline.analyse(cv_bytes, jd_text, kb_path, case=case, jd_analysis=(intent, result))


def jd_text_from_state() -> str:
    return (st.session_state.get("jd_text") or "").strip()


def cv_bytes_from_ui() -> bytes | None:
    """CV bytes from the uploader, or from the `cv_bytes` session_state seam.

    Test seam: `streamlit.testing.v1.AppTest` cannot drive `st.file_uploader`, so tests
    put the .docx bytes in `st.session_state["cv_bytes"]` (and may point
    `st.session_state["kb_path"]` at a fixture knowledge base). A real upload wins when
    both are present.
    """
    uploaded = st.file_uploader("CV (.docx)", type=["docx"], key="cv_upload")
    if uploaded is not None:
        st.session_state["cv_name"] = uploaded.name
        return uploaded.getvalue()
    seam = st.session_state.get("cv_bytes")
    return bytes(seam) if seam else None


def kb_path_from_ui() -> Path:
    return Path(st.session_state.get("kb_path") or DEFAULT_KB_PATH)


def render_intent(intent: JDIntent) -> None:
    st.subheader("What the role wants")
    st.markdown(intent.role_summary)
    st.markdown(f"**Seniority:** {intent.seniority}")
    st.markdown(f"**Domain:** {intent.domain}")
    st.markdown(f"**Tone:** {intent.tone}")
    st.markdown("**Top priorities**")
    st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(intent.top_priorities, 1)))


def render_keywords(result: JDKeywords) -> None:
    st.subheader("Skills and keywords")
    st.markdown("**Mandatory**")
    st.markdown(", ".join(result.mandatory) or "_none found_")
    st.markdown("**Nice to have**")
    st.markdown(", ".join(result.nice_to_have) or "_none found_")
    st.markdown("**Responsibilities**")
    st.markdown("\n".join(f"- {r}" for r in result.responsibilities) or "_none found_")
    with st.expander(f"All ATS keywords ({len(result.ats_keywords)})"):
        st.markdown(", ".join(result.ats_keywords))


def _bullets(items: list[str], aliases: dict[str, str], empty: str) -> str:
    if not items:
        return f"_{empty}_"
    lines = []
    for item in items:
        via = aliases.get(item)
        lines.append(f"- {item}" + (f" _(you have {via})_" if via else ""))
    return "\n".join(lines)


def approved_adjacent_from_ui(raw: SkillMatch) -> list[str]:
    """One checkbox per adjacent skill, default off. Ticked ones are the user's approvals."""
    approved: list[str] = []
    for skill in raw.adjacent:
        label = f"{skill} (you have {raw.aliases.get(skill, '?')})"
        if st.checkbox(label, key=f"adjacent::{skill}", value=False):
            approved.append(skill)
    return approved


def render_buckets(raw: SkillMatch) -> tuple[SkillMatch, list[str]]:
    """Three columns; returns the buckets after applying the ticked adjacent skills, and those skills."""
    col_matched, col_adjacent, col_missing = st.columns(3)
    with col_adjacent:
        st.subheader("Adjacent")
        st.caption("Suggestions only. Tick a skill to let the rewrite use it.")
        approved = approved_adjacent_from_ui(raw)
    shown = matching.apply_approvals(raw, approved)
    with col_matched:
        st.subheader("Matched")
        st.caption("In your CV or knowledge base, possibly under another name.")
        st.markdown(_bullets(shown.matched, shown.aliases, "nothing matched"))
    with col_missing:
        st.subheader("Missing")
        st.caption("Never added to the CV.")
        st.markdown(_bullets(shown.missing, {}, "nothing missing"))
    return shown, approved


def render_ranked(ranked: list[tuple[ProjectFact, float]]) -> None:
    st.subheader("Ranked projects")
    if not ranked:
        st.info(f"No knowledge base found at `{DEFAULT_KB_PATH}`; add it to rank your projects.")
        return
    lines = []
    for i, (fact, score) in enumerate(ranked, 1):
        kind = "work" if fact.professional else "personal, weighted x0.5"
        where = f" at {fact.employer}" if fact.employer else ""
        lines.append(f"{i}. **{fact.name}**{where} ({kind}) - relevance {score:.2f}")
    st.markdown("\n".join(lines))


def render_coverage(cv_text: str, result: JDKeywords) -> None:
    score = coverage(cv_text, result.ats_keywords)
    missing = missing_keywords(cv_text, result.ats_keywords)
    st.metric("ATS keyword coverage (current CV)", f"{score:.0%}")
    with st.expander(f"Keywords not in the CV yet ({len(missing)})"):
        st.markdown(", ".join(missing) or "_all present_")


def tailored_file_name() -> str:
    stem = Path(st.session_state.get("cv_name") or "cv.docx").stem
    return f"{stem}_tailored.docx"


def accept_key(para_id: str) -> str:
    return f"{ACCEPT_KEY}{para_id}"


def render_bullets(result: pipeline.RunResult) -> set[str]:
    """Per-bullet table (accept | original | new | KB metrics | keywords). Returns the rejected paragraph ids.

    The accept state lives in `st.session_state[accept_key(para_id)]` (default accepted), so
    it survives reruns and drives `pipeline.assemble` for the download.
    """
    rejected: set[str] = set()
    experience = result.experience
    if experience is None:
        return rejected
    st.subheader("Experience bullets")
    changed = [b for b in experience.bullets if b.para_id in result.rewrites]
    unchanged = [b for b in experience.bullets if b.para_id not in result.rewrites]
    if not changed:
        st.info("No experience bullet was changed.")
    else:
        st.caption("Untick a bullet to keep the original wording in the download.")
        widths = [0.7, 3, 3, 1.6, 1.6]
        header = st.columns(widths)
        for col, title in zip(header, ("Accept", "Original", "New", "KB metrics used", "JD keywords")):
            col.markdown(f"**{title}**")
        for bullet in changed:
            cols = st.columns(widths)
            accepted = cols[0].checkbox(
                f"Accept {bullet.para_id}", key=accept_key(bullet.para_id), value=True, label_visibility="collapsed"
            )
            cols[1].markdown(result.originals[bullet.para_id])
            cols[2].markdown(bullet.text)
            cols[3].markdown(", ".join(bullet.used_kb_metrics) or "-")
            cols[4].markdown(", ".join(bullet.jd_keywords_used) or "-")
            if not accepted:
                rejected.add(bullet.para_id)
    if unchanged:
        with st.expander(f"Bullets left as they were ({len(unchanged)})"):
            for bullet in unchanged:
                where = " (rule check failed, original kept)" if bullet.para_id in experience.reverted else ""
                st.markdown(f"- {bullet.text}{where}")
    return rejected


SECTION_LABELS = {"summary": "Professional summary", "skills": "Core skills", "exp_bullet": "Experience bullet"}


def render_reverts(result: pipeline.RunResult) -> None:
    """Agent 6's refusals: what it blocked, and why, above the rest of the notes."""
    if not result.reverts:
        return
    st.subheader(f"Reviewer reverted {len(result.reverts)} paragraph(s)")
    st.caption("The reviewer refused these rewrites, so the original wording is kept in the download.")
    for revert in result.reverts:
        label = SECTION_LABELS.get(revert.section, revert.section or "Paragraph")
        st.warning(f"**{label}** - {revert.reason}")
        with st.expander("What it wanted to write"):
            st.markdown(f"**Kept:** {revert.original}")
            st.markdown(f"**Blocked:** {revert.rejected}")


def render_run_result(result: pipeline.RunResult) -> None:
    """Before/after for every rewritten paragraph, the bullet table, notes, coverage and the download."""
    st.header("Tailored CV")
    for para_id, section in result.sections.items():
        if section not in ("summary", "skills"):
            continue
        st.subheader("Professional summary" if section == "summary" else "Core skills")
        before, after = st.columns(2)
        with before:
            st.caption("Before")
            st.markdown(result.originals[para_id])
        with after:
            st.caption("After" if para_id in result.rewrites else "After (unchanged)")
            st.markdown(result.rewrites.get(para_id, result.originals[para_id]))
    rejected = render_bullets(result)
    render_reverts(result)
    for note in result.notes:
        st.warning(note)

    output = pipeline.assemble(result, rejected)
    st.session_state["tailored_docx"] = output  # test seam: the bytes the download button offers
    out_text = "\n".join(p.full_text for p in iter_paragraphs(load(output)))
    coverage_after = coverage(out_text, result.keywords.ats_keywords)
    cols = st.columns(3)
    cols[0].metric("ATS coverage before", f"{result.coverage_before:.0%}")
    cols[1].metric("ATS coverage after", f"{coverage_after:.0%}", f"{coverage_after - result.coverage_before:+.0%}")
    cols[2].metric(
        "LLM calls / tokens" + (" (cached)" if result.from_cache else ""),
        f"{result.usage.calls} / {result.usage.total_tokens:,}",
    )
    st.download_button(
        f"Download tailored CV ({tailored_file_name()})",
        data=output,
        file_name=tailored_file_name(),
        mime=DOCX_MIME,
        key="download_docx",
        type="primary",
    )


def apply_llm_choice(provider: str, model: str) -> None:
    """Make the sidebar's provider/model the process settings (the app is single-user and local).

    Only writes the environment when the choice differs from the current settings, and
    rebuilds the provider on the next LLM call.
    """
    current = Settings.from_env()
    if provider == current.llm_provider and model == current.llm_model:
        return
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    llm.set_provider(None)


def render_sidebar_settings(settings: Settings) -> Settings:
    """Provider select + model input + Clear cache. Returns the settings after applying the choice."""
    st.header("LLM")
    index = PROVIDERS.index(settings.llm_provider) if settings.llm_provider in PROVIDERS else 0
    provider = st.selectbox("Provider", PROVIDERS, index=index, key="provider")
    default_model = settings.llm_model if provider == settings.llm_provider else DEFAULT_MODELS[provider]
    model = st.text_input("Model", value=default_model, key=f"model::{provider}").strip() or default_model
    apply_llm_choice(provider, model)
    settings = Settings.from_env()
    st.caption(f"Model `{settings.llm_model}` on {settings.llm_provider}, mode `{settings.llm_mode}`.")
    price = llm.price_for(settings.llm_model)
    if price is None:
        st.caption("No price known for this model; cost shows as £0.")
    elif price.input_gbp_per_1m == 0:
        st.caption("Local model: free.")
    else:
        st.caption(f"£{price.input_gbp_per_1m:.2f} in / £{price.output_gbp_per_1m:.2f} out per 1M tokens.")

    st.header("Cache")
    if st.button("Clear cache", key="clear_cache"):
        removed = cache.clear(settings.cache_dir)
        st.success(f"Removed {removed} cached run(s).")
    st.caption(f"{cache.size(settings.cache_dir)} cached run(s) in `{settings.cache_dir}`.")
    return settings


def _add_usage(total: TokenUsage, usage: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=total.input_tokens + usage.input_tokens,
        output_tokens=total.output_tokens + usage.output_tokens,
        calls=total.calls + usage.calls,
        model=usage.model or total.model,
        estimated_cost_gbp=total.estimated_cost_gbp + usage.estimated_cost_gbp,
    )


def track_usage(usage: TokenUsage, *, from_cache: bool) -> None:
    """Remember the last run's usage and add it to the session total (cache hits cost nothing)."""
    st.session_state["last_usage"] = usage
    if not from_cache:
        st.session_state["session_usage"] = _add_usage(st.session_state.get("session_usage") or TokenUsage(), usage)


def render_sidebar_usage(container) -> None:
    """Tokens used and estimated cost for the last run and the whole session."""
    last: TokenUsage = st.session_state.get("last_usage") or TokenUsage()
    session: TokenUsage = st.session_state.get("session_usage") or TokenUsage()
    with container:
        st.header("Usage")
        result = st.session_state.get("run_result")
        if result is not None and result.from_cache:
            st.caption("Last run was served from the cache: no LLM calls, no cost.")
        st.metric("Tokens used (last run)", f"{last.total_tokens:,}")
        st.caption(f"{last.input_tokens:,} in / {last.output_tokens:,} out over {last.calls} LLM call(s).")
        st.metric("Estimated cost (last run)", f"£{last.estimated_cost_gbp:.4f}")
        st.metric("Session tokens / cost", f"{session.total_tokens:,} / £{session.estimated_cost_gbp:.4f}")


def _forget(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def _forget_run() -> None:
    _forget("run_result", "tailored_docx")
    for key in [k for k in st.session_state if str(k).startswith(ACCEPT_KEY)]:
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title="ResumeTailor", layout="wide")
    st.title("ResumeTailor")
    st.caption(
        "Upload your CV, paste a job description and press Analyse to see what the role wants and "
        "what you can truthfully claim. Then press Rewrite to tailor the summary, skills line and "
        "experience bullets, accept or reject each bullet and download the result; the layout is never touched."
    )

    with st.sidebar:
        settings = render_sidebar_settings(Settings.from_env())
        usage_box = st.container()  # filled at the end, once this run's results are known
    cv_bytes = cv_bytes_from_ui()
    st.text_area("Job description", height=320, key="jd_text", placeholder="Paste the full job advert here")
    if st.button("Analyse", key="analyse", type="primary"):
        jd_text = jd_text_from_state()
        case = st.session_state.get("llm_case")
        _forget("analysis", "tailoring")
        _forget_run()
        if not jd_text:
            st.warning("Paste a job description first.")
        else:
            try:
                with st.spinner(f"Reading the job description with {settings.llm_model} ({settings.llm_mode})..."):
                    intent, result = analyse(jd_text, case)
                    st.session_state["analysis"] = (intent, result)
                    if cv_bytes:
                        tailoring = tailor(cv_bytes, jd_text, intent, result, kb_path_from_ui(), case)
                        st.session_state["tailoring"] = tailoring
                        track_usage(tailoring.usage, from_cache=tailoring.from_cache)
            except RecordingMissing:
                _forget("analysis", "tailoring")
                st.error(
                    "LLM_MODE is `replay` and there is no recording for this job description. "
                    "Set `LLM_MODE=live` in `.env` to call Gemini."
                )
            except LLM_ERRORS as exc:
                _forget("analysis", "tailoring")
                st.error(str(exc))

    if "analysis" not in st.session_state:
        render_sidebar_usage(usage_box)
        return
    intent, result = st.session_state["analysis"]
    left, right = st.columns(2)
    with left:
        render_intent(intent)
    with right:
        render_keywords(result)

    if "tailoring" not in st.session_state:
        st.info("Upload your CV (.docx) and press Analyse again to see matched skills, ranked projects and coverage.")
        render_sidebar_usage(usage_box)
        return
    tailoring: pipeline.Analysis = st.session_state["tailoring"]
    st.header("What you can claim")
    _, approved = render_buckets(tailoring.match)
    lower, upper = st.columns(2)
    with lower:
        render_ranked(tailoring.ranked)
    with upper:
        render_coverage(tailoring.cv_text, result)

    if st.button("Rewrite CV", key="rewrite", type="primary", disabled=not cv_bytes):
        _forget_run()
        try:
            with st.spinner(f"Rewriting with {settings.llm_model} ({settings.llm_mode})..."):
                run_result = pipeline.run(
                    cv_bytes,
                    jd_text_from_state(),
                    kb_path_from_ui(),
                    approved,
                    REWRITE_STAGES,
                    analysis=tailoring,
                )
                st.session_state["run_result"] = run_result
                track_usage(run_result.usage, from_cache=run_result.from_cache)
        except RecordingMissing:
            st.error("LLM_MODE is `replay` and there is no recording for this rewrite. Set `LLM_MODE=live` in `.env`.")
        except LLM_ERRORS as exc:
            st.error(str(exc))
    if "run_result" in st.session_state:
        render_run_result(st.session_state["run_result"])
    render_sidebar_usage(usage_box)


main()
