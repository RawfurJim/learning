# ResumeTailor — Product Requirements

Jira Epic: https://rawfurjim12.atlassian.net/browse/SCRUM-7

## Problem

Tailoring a CV per application is slow. Generic LLM rewrites hallucinate skills, push the document past one page and destroy Word formatting. Jim (AI Engineer / Data Science applicant) needs a repeatable, honest, cheap tailoring step that ATS parsers score well and that keeps the CV visually identical.

## Solution

A Streamlit app. Upload CV (.docx) + paste a job description (JD). Six Gemini 2.5 Flash agents analyse the JD, decide what Jim can truthfully claim, rewrite the summary, skills line and experience bullets in the JD's vocabulary at the original length, and reassemble a .docx with identical layout. An independent reviewer reverts anything invented. Jim accepts or rejects each change and downloads the result.

### Agents

| # | Agent | Input | Output |
|---|-------|-------|--------|
| 1 | JD Intent | JD text | role summary, seniority, tone, top-5 priorities, domain |
| 2 | Keyword Extractor | JD text | mandatory[], nice_to_have[], ats_keywords[], responsibilities[] |
| 3 | Experience Writer | bullets + word budgets + JD keywords + project facts | rewritten bullets, metrics used, keywords used |
| 4 | Summary & Skills Writer | summary + skills line + budgets + matched skills + intent | new summary, new skills list |
| 5 | Assembler (code) | original docx + accepted rewrites | tailored docx, layout checks, page count |
| 6 | Reviewer / Guard | original vs proposed text, allowed vocabulary, facts | accept / revert with reason |

Pipeline: 1 and 2 in parallel -> matching + project ranking (code) -> 3 and 4 in parallel -> 6 -> user accept/reject -> 5.

### Knowledge base

`knowledge/projects.md` holds Jim's detailed project descriptions (work and personal). Agents draw truthful detail and metrics from it, rank projects by relevance to the JD, and use it plus the CV as the **allowed vocabulary**. It is personal data and is gitignored.

## Hard rules

- Never claim a skill, tool, employer, title, date or number that is not in the CV or knowledge base.
- Adjacent skills (e.g. FastAPI -> Flask) are shown as suggestions and used only when the user ticks them.
- Extra metrics may be added only if they exist in the knowledge base. The model never invents numbers.
- Every rewritten paragraph stays within +/-10% of the original word count; total document within +/-3% characters; page count unchanged.
- Professional (JudgeService) evidence outranks personal projects in ranking, summary and bullet rewrites.
- Header/contact, employer lines, dates, project sub-headings and education are read-only.

## Goals

1. Understand the JD precisely (intent + keywords).
2. Classify JD skills as matched / adjacent / missing; rank Jim's projects by JD relevance.
3. Rewrite bullets, summary and skills in the JD's vocabulary at the original length.
4. Reassemble into an identical-looking .docx.
5. Independent reviewer blocks invented content; user accepts/rejects each change.
6. Cost under £0.01 per run on Gemini 2.5 Flash; works on the free tier; provider swappable to Groq / Ollama.

## Non-goals

PDF input, cover letters, multi-CV management, hosting/auth, adding new CV sections.

## User stories

1. Upload CV + paste JD -> see a plain-English summary of what the role wants.
2. See mandatory vs nice-to-have skills and ATS keywords from the JD.
3. See matched / adjacent (opt-in) / missing (never added) skills and my highest-ranked projects for this JD.
4. Get reworded experience bullets, same length, every metric kept, optional extra metrics from my knowledge base only, per-bullet accept/reject.
5. Get a rewritten summary and reordered skills line using only matched/approved skills, same length.
6. Download a .docx that opens in Word looking identical to the original (fonts, bullets, spacing, one page).
7. See a warning when the reviewer reverts a paragraph and why.
8. See before/after ATS keyword coverage, tokens and cost; repeat runs are cached.
9. Maintain my knowledge base as a simple Markdown file.

## Success metrics

- Zero invented tech terms and zero lost numbers across a 3-JD regression suite (asserted by tests).
- Docx structure diff shows only paragraph text changes.
- Every rewritten paragraph within +/-10% words; document within +/-3% chars; page count unchanged.
- ATS keyword coverage rises on every regression JD.

## Open data questions (Jim to confirm; knowledge base keeps both until then)

- Golden set size for the sentiment platform: **500** or **1,500** reviews?
- Self-hosted batch model: **Gemma 27B** or **Gemma 4 31B**?
