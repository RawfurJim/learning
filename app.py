"""ResumeTailor Streamlit UI, v0: paste a job description, see what it wants and its ATS keywords."""

from __future__ import annotations

import streamlit as st

from resume_tailor.agents import jd_intent, keywords
from resume_tailor.llm import LLMOutputError, ProviderConfigError, RecordingMissing
from resume_tailor.schemas import JDIntent, JDKeywords
from resume_tailor.settings import Settings


def analyse(jd_text: str, case: str | None) -> tuple[JDIntent, JDKeywords]:
    """Agents 1 and 2 on the same JD. `case` names the recording used in record/replay mode."""
    return jd_intent.run(jd_text, case=case), keywords.run(jd_text, case=case)


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


def main() -> None:
    st.set_page_config(page_title="ResumeTailor", layout="wide")
    st.title("ResumeTailor")
    st.caption("Paste a job description and press Analyse to see what it really wants. CV tailoring arrives in later tickets.")

    settings = Settings.from_env()
    st.text_area("Job description", height=320, key="jd_text", placeholder="Paste the full job advert here")
    if st.button("Analyse", key="analyse", type="primary"):
        jd_text = (st.session_state.get("jd_text") or "").strip()
        if not jd_text:
            st.warning("Paste a job description first.")
        else:
            try:
                with st.spinner(f"Reading the job description with {settings.llm_model} ({settings.llm_mode})..."):
                    st.session_state["analysis"] = analyse(jd_text, st.session_state.get("llm_case"))
            except RecordingMissing:
                st.session_state.pop("analysis", None)
                st.error(
                    "LLM_MODE is `replay` and there is no recording for this job description. "
                    "Set `LLM_MODE=live` in `.env` to call Gemini."
                )
            except (ProviderConfigError, LLMOutputError) as exc:
                st.session_state.pop("analysis", None)
                st.error(str(exc))

    if "analysis" in st.session_state:
        intent, result = st.session_state["analysis"]
        left, right = st.columns(2)
        with left:
            render_intent(intent)
        with right:
            render_keywords(result)


main()
