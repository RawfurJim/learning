# ResumeTailor — project instructions for Claude

You are working on **ResumeTailor**, a Streamlit app that tailors Jim's CV (.docx) to a job description using six Gemini agents while keeping the exact document layout and never inventing skills.

Read these before doing anything:

1. `docs/PRD.md` — what we are building and the hard rules.
2. `docs/ARCHITECTURE.md` — code layout, agent contracts, testing conventions.
3. `docs/TICKETS.md` — the Jira ticket map and order.

## How work is done: one Jira ticket per session

Work is tracked in Jira project **SCRUM** (site `rawfurjim12.atlassian.net`, cloudId `774de3e3-7a56-4e82-9b23-541001ab14b8`). Epic **SCRUM-7** holds the PRD; tickets SCRUM-8 to SCRUM-17 are the work.

When Jim says "do SCRUM-<n>" (or just gives a ticket number):

1. **Fetch the ticket** with the Atlassian MCP tool `getJiraIssue` and read it fully. The ticket lists the files to create, the **test cases that define done**, acceptance criteria, blockers and the commit message.
2. **Check blockers**: every ticket listed under "Blocked by" must be Done in Jira. If not, stop and tell Jim.
3. **Transition the ticket to In Progress** (`getTransitionsForJiraIssue` then `transitionJiraIssue`).
4. **Write the tests first** exactly as named in the ticket, then implement until `uv run pytest` is green. Do not weaken or delete a listed test to make it pass; if a test is wrong, say so and propose the change to Jim.
5. **Live checks** (`uv run pytest -m live`) need `GEMINI_API_KEY` in `.env`. If no key is available, say so; do not fake recordings by hand. Recordings are produced with `LLM_MODE=record`.
6. **Commit** with the message given in the ticket, format `SCRUM-<n>: <title>`. One commit per ticket (squash fix-ups). Do not push unless Jim asks.
7. **Transition the ticket to Done** and add a short Jira comment: what was built, test count, anything Jim should verify manually.
8. **Update `docs/TICKETS.md`** status column for that ticket.

Do not start the next ticket in the same session unless Jim asks.

## Hard rules (never break these in code or prompts)

- The tool must **never add a skill, tool, employer, title, date or number** that is not in the CV or `knowledge/projects.md`.
- Adjacent skills (e.g. FastAPI -> Flask) are suggestions only and are used only when the user explicitly approves them.
- Extra metrics may be added only if they exist in the knowledge base.
- Rewritten paragraphs stay within +/-10% of the original word count; whole document within +/-3% characters; page count unchanged.
- Professional (JudgeService) evidence outranks personal projects.
- Header/contact, employer lines, dates, project sub-headings and education are read-only.

## Environment

- Python 3.14, `uv`. Run everything through `uv run ...`.
- Tests: `uv run pytest` (deterministic, replays recorded LLM JSON, no API key). `uv run pytest -m live` calls Gemini.
- App: `uv run streamlit run app.py`.
- Secrets in `.env` (gitignored). Template: `.env.example`.
- `knowledge/projects.md` holds Jim's personal data and is gitignored. Never paste its contents into Jira or commit it.

## Style

- Small modules with one job each (see `docs/ARCHITECTURE.md`). Pydantic models for every LLM output.
- Prompts live next to their agent as module-level constants. Every agent has exactly one `run(...)` function.
- Prefer deterministic code over LLM calls wherever a rule can be checked in Python (vocabulary, numbers, length).
