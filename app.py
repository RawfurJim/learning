"""ResumeTailor Streamlit UI.

Paste a job description to see what it wants and its ATS keywords (SCRUM-11). Upload
the CV (.docx) as well to see which JD skills are Matched, which are Adjacent
suggestions (opt-in checkboxes, never applied on their own), which are Missing (never
added), the knowledge-base projects ranked for this JD, and the ATS keyword coverage
of the current CV (SCRUM-12).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from resume_tailor import matching
from resume_tailor.agents import jd_intent, keywords
from resume_tailor.ats_score import coverage, missing_keywords
from resume_tailor.docx_io import iter_paragraphs, load
from resume_tailor.knowledge import load_projects
from resume_tailor.llm import LLMOutputError, ProviderConfigError, RecordingMissing
from resume_tailor.schemas import JDIntent, JDKeywords, ProjectFact, SkillMatch
from resume_tailor.settings import REPO_ROOT, Settings

DEFAULT_KB_PATH = REPO_ROOT / "knowledge" / "projects.md"


def analyse(jd_text: str, case: str | None) -> tuple[JDIntent, JDKeywords]:
    """Agents 1 and 2 on the same JD. `case` names the recording used in record/replay mode."""
    return jd_intent.run(jd_text, case=case), keywords.run(jd_text, case=case)


def tailor(
    cv_bytes: bytes,
    intent: JDIntent,
    result: JDKeywords,
    facts: list[ProjectFact],
    case: str | None,
) -> dict:
    """Deterministic matching + the two recorded calls (alias, project ranking) + coverage."""
    paras = iter_paragraphs(load(cv_bytes))
    inventory = matching.build_inventory(paras, facts)
    cv_text = "\n".join(p.full_text for p in paras)
    return {
        "match": matching.match(result, inventory, facts, case=case),
        "ranked": matching.rank_projects(intent, facts, case=case),
        "cv_text": cv_text,
        "inventory": inventory,
    }


def cv_bytes_from_ui() -> bytes | None:
    """CV bytes from the uploader, or from the `cv_bytes` session_state seam.

    Test seam: `streamlit.testing.v1.AppTest` cannot drive `st.file_uploader`, so tests
    put the .docx bytes in `st.session_state["cv_bytes"]` (and may point
    `st.session_state["kb_path"]` at a fixture knowledge base). A real upload wins when
    both are present.
    """
    uploaded = st.file_uploader("CV (.docx)", type=["docx"], key="cv_upload")
    if uploaded is not None:
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


def render_buckets(raw: SkillMatch) -> SkillMatch:
    """Three columns; returns the buckets after applying the ticked adjacent skills."""
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
    return shown


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


def main() -> None:
    st.set_page_config(page_title="ResumeTailor", layout="wide")
    st.title("ResumeTailor")
    st.caption(
        "Upload your CV, paste a job description and press Analyse to see what the role wants, "
        "what you can truthfully claim, and your ATS coverage. Rewrites arrive in later tickets."
    )

    settings = Settings.from_env()
    cv_bytes = cv_bytes_from_ui()
    st.text_area("Job description", height=320, key="jd_text", placeholder="Paste the full job advert here")
    if st.button("Analyse", key="analyse", type="primary"):
        jd_text = (st.session_state.get("jd_text") or "").strip()
        case = st.session_state.get("llm_case")
        if not jd_text:
            st.warning("Paste a job description first.")
        else:
            try:
                with st.spinner(f"Reading the job description with {settings.llm_model} ({settings.llm_mode})..."):
                    intent, result = analyse(jd_text, case)
                    st.session_state["analysis"] = (intent, result)
                    st.session_state.pop("tailoring", None)
                    if cv_bytes:
                        facts = load_projects(kb_path_from_ui())
                        st.session_state["tailoring"] = tailor(cv_bytes, intent, result, facts, case)
            except RecordingMissing:
                st.session_state.pop("analysis", None)
                st.session_state.pop("tailoring", None)
                st.error(
                    "LLM_MODE is `replay` and there is no recording for this job description. "
                    "Set `LLM_MODE=live` in `.env` to call Gemini."
                )
            except (ProviderConfigError, LLMOutputError) as exc:
                st.session_state.pop("analysis", None)
                st.session_state.pop("tailoring", None)
                st.error(str(exc))

    if "analysis" in st.session_state:
        intent, result = st.session_state["analysis"]
        left, right = st.columns(2)
        with left:
            render_intent(intent)
        with right:
            render_keywords(result)

        if "tailoring" in st.session_state:
            tailoring = st.session_state["tailoring"]
            st.header("What you can claim")
            render_buckets(tailoring["match"])
            lower, upper = st.columns(2)
            with lower:
                render_ranked(tailoring["ranked"])
            with upper:
                render_coverage(tailoring["cv_text"], result)
        else:
            st.info("Upload your CV (.docx) and press Analyse again to see matched skills, ranked projects and coverage.")


main()
