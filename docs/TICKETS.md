# ResumeTailor — Ticket map

Source of truth for each ticket's scope and test cases is **Jira**. Fetch the ticket with the Atlassian MCP (`getJiraIssue`) before starting. This file only gives the order, dependencies and status.

Epic: [SCRUM-7](https://rawfurjim12.atlassian.net/browse/SCRUM-7) — PRD

| Order | Key | Title | Type | Blocked by | Status |
|------:|-----|-------|------|-----------|--------|
| 1 | [SCRUM-8](https://rawfurjim12.atlassian.net/browse/SCRUM-8) | Scaffold project, Gemini JSON wrapper, record/replay test harness | Task | — | Done |
| 2 | [SCRUM-9](https://rawfurjim12.atlassian.net/browse/SCRUM-9) | DOCX round-trip with section classification and formatting-preserving edits | Story | SCRUM-8 | Done |
| 3 | [SCRUM-10](https://rawfurjim12.atlassian.net/browse/SCRUM-10) | Project knowledge base loader and allowed vocabulary | Task | SCRUM-8 | Done |
| 4 | [SCRUM-11](https://rawfurjim12.atlassian.net/browse/SCRUM-11) | Agent 1 (JD Intent) and Agent 2 (Keyword Extractor) with Streamlit v0 | Story | SCRUM-8 | Done |
| 5 | [SCRUM-12](https://rawfurjim12.atlassian.net/browse/SCRUM-12) | Skill matching, project relevance ranking, ATS coverage score | Story | SCRUM-9, 10, 11 | Done |
| 6 | [SCRUM-13](https://rawfurjim12.atlassian.net/browse/SCRUM-13) | Agent 4 (Summary & Skills Writer) + Assembler -> first downloadable .docx | Story | SCRUM-12 | Done |
| 7 | [SCRUM-14](https://rawfurjim12.atlassian.net/browse/SCRUM-14) | Agent 3 (Experience Writer) with per-bullet accept/reject | Story | SCRUM-13 | Done |
| 8 | [SCRUM-15](https://rawfurjim12.atlassian.net/browse/SCRUM-15) | Cache, token/cost display, provider switch (Groq / Ollama) | Story | SCRUM-13 | Done |
| 9 | [SCRUM-16](https://rawfurjim12.atlassian.net/browse/SCRUM-16) | Agent 6 (Reviewer/Guard), layout checks, 3-JD regression suite | Task | SCRUM-14 | Done |
| 10 | [SCRUM-17](https://rawfurjim12.atlassian.net/browse/SCRUM-17) | Manual QA checklist (human) | Task | SCRUM-15, 16 | Ready for Jim — see [QA_CHECKLIST.md](QA_CHECKLIST.md) |

Tickets 2, 3, 4 can be done in any order after 1. Tickets 7 and 8 can be done in any order after 6.

## Inputs already in the repo

- `tests/fixtures/cv_text.txt` — Jim's CV text, used by SCRUM-9 to build `sample_cv.docx`.
- `knowledge/projects.md` — Jim's project knowledge base (gitignored), used by SCRUM-10. It contains **3 work and 6 personal** projects.

## Per-ticket session checklist

1. Fetch ticket, confirm blockers Done.
2. Transition to In Progress.
3. Write the listed tests, implement, `uv run pytest` green.
4. Commit `SCRUM-<n>: <title>`.
5. Transition to Done, add a Jira comment, update the Status column above.
