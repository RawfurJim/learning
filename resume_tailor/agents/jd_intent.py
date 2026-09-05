"""Agent 1 (JD Intent): role summary, seniority, tone, top-5 priorities and domain of a JD."""

from __future__ import annotations

from resume_tailor import llm
from resume_tailor.agents import default_case
from resume_tailor.schemas import JDIntent

AGENT = "jd_intent"

PROMPT = """You are an experienced technical recruiter reading a job description (JD) on behalf of a candidate.
Read the JD below and describe what the hiring manager really wants. Base everything on the JD text only.

Return a JSON object with exactly these fields:
- role_summary: 2-3 plain-English sentences describing the role: what the person will build or own, for whom, and what success looks like.
- seniority: exactly one of "junior", "mid", "senior", "lead". Decide from the whole advert, not the title alone:
  * "junior": 0-2 years, graduate/entry level, "early in your career", works under close support or mentoring.
  * "mid": roughly 2-5 years, works independently on well-defined problems, no ownership of design or people.
  * "senior": 5+ years or explicitly senior; owns systems end to end, leads technical design, mentors others but is still hands-on.
  * "lead": manages or leads a team or function (Lead, Principal, Head of, Staff), sets direction for others.
- tone: 3-8 words describing the voice and culture the advert projects (for example "pragmatic, hands-on, delivery-focused, mentoring").
- top_priorities: exactly 5 short phrases (4-12 words each), most important first, naming the capabilities the employer most wants evidence of. Cover the core technical work, how quality is measured or evaluated, production/operational expectations, and any collaboration or leadership expectation the JD stresses. Use the JD's own vocabulary.
- domain: one line naming the industry or business domain and the technical domain, for example "Water utility customer insights - NLP / text analytics" or "Legal and financial document intelligence - LLM applications".

Job description:
<<<
{jd_text}
>>>
"""


def build_prompt(jd_text: str) -> str:
    return PROMPT.format(jd_text=jd_text.strip())


def run(jd_text: str, *, case: str | None = None) -> JDIntent:
    """Analyse `jd_text`. `case` names the recording (`tests/recordings/jd_intent/<case>.json`)."""
    case = case or default_case(jd_text)
    return llm.generate_json(build_prompt(jd_text), JDIntent, case=f"{AGENT}/{case}")
