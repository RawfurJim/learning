# ResumeTailor

Tailor a `.docx` CV to a specific job description with a small pipeline of LLM agents —
**without inventing skills and without touching the layout**.

Upload your CV, paste a job advert, and ResumeTailor tells you what the role actually wants,
which of its requirements you can honestly claim, and then rewrites only the professional
summary, the core-skills line and the experience bullets in the advert's own vocabulary —
at the original length. An independent reviewer agent re-checks every rewritten paragraph
and reverts anything it cannot ground in your CV or knowledge base. You accept or reject each
bullet and download a `.docx` that opens in Word looking identical to the one you uploaded.

> Built as a learning project against a written PRD and a Jira backlog (SCRUM-8 … SCRUM-17),
> one ticket per session, tests first. See [`docs/`](docs/).

---

## Table of contents

- [Why](#why)
- [What it does](#what-it-does)
- [The hard rules](#the-hard-rules)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [The knowledge base](#the-knowledge-base)
- [Using the app](#using-the-app)
- [Architecture](#architecture)
  - [Pipeline](#pipeline)
  - [The six agents](#the-six-agents)
  - [Module map](#module-map)
  - [The reviewer (Agent 6)](#the-reviewer-agent-6)
  - [Layout preservation](#layout-preservation)
- [LLM access: providers, record/replay, cost](#llm-access-providers-recordreplay-cost)
- [Caching](#caching)
- [Testing](#testing)
- [Using it as a library](#using-it-as-a-library)
- [Project status](#project-status)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Privacy](#privacy)

---

## Why

Tailoring a CV per application is slow, and handing the whole document to a general-purpose
chatbot goes wrong in three predictable ways:

1. **It hallucinates.** New frameworks, invented team sizes, borrowed metrics.
2. **It breaks the layout.** Fonts, bullet glyphs, spacing and the one-page limit all go.
3. **It drifts in length.** A "tightened" bullet comes back 40% longer and the CV spills onto page two.

ResumeTailor fixes each one with code rather than trust: the vocabulary, the numbers and the
word counts are checked in Python, the document is edited paragraph-by-paragraph in place, and
the model is only ever asked to do the part that genuinely needs language.

## What it does

- **Reads the JD** — a plain-English summary of the role, its seniority, tone, domain and top five priorities.
- **Extracts keywords** — mandatory vs nice-to-have skills, ATS keywords, responsibilities.
- **Tells you where you stand** — every JD skill bucketed as **matched** (you have it),
  **adjacent** (close enough to claim only if *you* tick it, e.g. FastAPI → Flask) or
  **missing** (never added to the document, ever).
- **Ranks your projects** against the JD, weighting professional work above personal projects.
- **Rewrites** the summary, the skills line and each experience bullet in the JD's vocabulary,
  keeping every original metric and staying within ±10% of the original word count.
- **Guards the output** — an independent reviewer reverts any paragraph with an unsupported
  term, a dropped number, a bad length or an unsupported claim, and shows you why.
- **Reassembles** a `.docx` whose structure diff contains nothing but paragraph text changes.
- **Shows the cost** — tokens in/out and an estimated GBP figure per run; identical runs are cached.

## The hard rules

These are enforced in code and asserted by the test suite, not merely requested in a prompt:

| Rule | Enforced by |
|------|-------------|
| Never claim a skill, tool, employer, title, date or number that is not in the CV or knowledge base | `knowledge.allowed_vocabulary` + `reviewer.review` |
| Adjacent skills are suggestions; used only when the user explicitly approves them | `matching.apply_approvals`, checkbox per skill in the UI |
| Extra metrics only if they exist in the knowledge base; the model never invents numbers | `knowledge.extract_numbers` + reviewer's `metric missing:` check |
| Every rewritten paragraph within ±10% of the original word count | `summary_skills_writer.within_word_budget` |
| Whole document within ±3% characters; page count unchanged | `assembler.check_layout` (page count needs LibreOffice) |
| Professional (employer) evidence outranks personal projects | `matching.weighted_ranking` (personal × 0.5) |
| Header/contact, employer lines, dates, project sub-headings and education are read-only | `sections.classify` — only `summary`, `skills`, `exp_bullet` are editable |

## Quick start

Requirements: **Python 3.14** and [`uv`](https://docs.astral.sh/uv/). LibreOffice (`soffice`) is
optional and only used to verify page count.

```bash
git clone https://github.com/RawfurJim/learning.git
cd learning

uv sync                                   # create .venv and install dependencies

cp .env.example .env                      # then put your GEMINI_API_KEY in it
cp knowledge/projects.example.md knowledge/projects.md   # then write your real projects

uv run pytest                             # 141 tests, offline, no API key needed
uv run streamlit run app.py               # the app, on http://localhost:8501
```

To actually call the model, set `LLM_MODE=live` in `.env`. With the default `LLM_MODE=replay`
the app answers only from the recorded fixtures in `tests/recordings/` and shows a clear
"no recording for this job description" message for anything new — that is expected, not a bug.

## Configuration

All settings come from `.env` (see `.env.example`), overridable by real environment variables,
and are parsed by `resume_tailor/settings.py`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `groq` \| `ollama` — also switchable in the sidebar |
| `LLM_MODEL` | `gemini-3.6-flash` | Model id. Per-provider defaults: `llama-3.3-70b-versatile` (Groq), `llama3.1` (Ollama) |
| `LLM_MODE` | `replay` | `replay` (read recordings, no network) \| `record` (call and save) \| `live` (call, save nothing) |
| `GEMINI_API_KEY` | — | Required for `gemini` in `record`/`live` mode |
| `GROQ_API_KEY` | — | Required for `groq` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server |
| `LLM_RECORDINGS_DIR` | `tests/recordings` | Where recordings are read/written |
| `CACHE_DIR` | `.cache` | Where identical runs are cached (gitignored) |
| `REVIEWER_SEMANTIC` | `1` | `0` skips the reviewer's one LLM drift check per paragraph (deterministic checks always run) |

> **Model note.** The PRD specifies Gemini 2.5 Flash. On 2026-09-05 the API rejected
> `gemini-2.5-flash` as "no longer available to new users" and recommended `gemini-3.6-flash`,
> which is therefore the default.

## The knowledge base

`knowledge/projects.md` is your private detail store — the CV is a one-page summary, this file
holds the facts behind it. Agents may draw on it for extra metrics and phrasing, and it forms
half of the **allowed vocabulary** (the CV is the other half). It is **gitignored**;
`knowledge/projects.example.md` is the committed, anonymised template.

One block per project:

```markdown
## Support Ticket Triage Assistant
category: work
employer: Example Analytics Ltd
period: Mar 2023 to Present
problem: Support agents spent hours a day classifying and routing inbound tickets by hand.
built: A fine-tuned small language model that classifies tickets by product area and urgency,
  served through a FastAPI endpoint with a human review queue for low-confidence cases.
stack: Python, FastAPI, Hugging Face Transformers, LoRA, PostgreSQL, Docker
metrics: 85% of tickets routed without human correction, 40k tickets/month
keywords: text classification, fine-tuning, human-in-the-loop, MLOps
```

`category: work` marks a project professional (`ProjectFact.professional`), which is what lifts it
above personal projects in ranking. Use `-` for an empty field. **Every number in here must be true** —
it is exactly the set of numbers the writers are allowed to introduce.

## Using the app

```bash
uv run streamlit run app.py
```

1. **Upload** your CV (`.docx`) and **paste** the full job advert.
2. **Analyse** → what the role wants, mandatory vs nice-to-have keywords, your matched /
   adjacent / missing buckets, your ranked projects, and current ATS keyword coverage.
3. **Tick** any adjacent skills you are genuinely happy to claim. Nothing under *Missing* is
   ever used, ticked or not.
4. **Rewrite CV** → new summary, reordered skills line and rewritten bullets, each shown
   before/after, each with an **accept/reject** checkbox. Reverted paragraphs appear in a
   warning block with the reviewer's reason.
5. **Download** the tailored `.docx`. Rejected bullets are simply left as the original.

The sidebar carries the provider/model switch, the token and cost readout for the run, and a
**Clear cache** button.

## Architecture

Full detail lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the product spec is in
[`docs/PRD.md`](docs/PRD.md).

### Pipeline

```
                    ┌── Agent 1  JD Intent ──┐
   JD text ─────────┤                        ├──▶  matching + project ranking   (code)
                    └── Agent 2  Keywords ───┘            │
                                                          │  matched / adjacent / missing
   CV .docx ──▶ docx_io.iter_paragraphs ──▶ sections.classify        ranked ProjectFacts
                                                          │
                          ┌───────────────────────────────┴───────────────┐
                          │                                               │
              Agent 3  Experience Writer                    Agent 4  Summary & Skills Writer
                          │                                               │
                          └───────────────────┬───────────────────────────┘
                                              ▼
                               Agent 6  Reviewer / Guard   (every paragraph, in parallel)
                                              │  accept  ▼  revert → Revert(reason) shown to user
                                              ▼
                                    user accept / reject per bullet
                                              ▼
                            Agent 5  Assembler (code) ──▶ tailored .docx + layout checks
```

Agents 1 and 2 run in parallel, as do 3 and 4. `pipeline.analyse()` covers everything up to the
approval step; `pipeline.run()` does the rewriting, review and assembly; `pipeline.assemble()`
rebuilds the document after the user rejects individual bullets — with no further LLM calls.

### The six agents

| # | Agent | Module | Input | Output |
|---|-------|--------|-------|--------|
| 1 | JD Intent | `agents/jd_intent.py` | JD text | role summary, seniority, tone, 5 priorities, domain |
| 2 | Keyword Extractor | `agents/keywords.py` | JD text | `mandatory[]`, `nice_to_have[]`, `ats_keywords[]`, `responsibilities[]` |
| 3 | Experience Writer | `agents/experience_writer.py` | bullets + word budgets + JD keywords + project facts | rewritten bullets, metrics used, keywords used |
| 4 | Summary & Skills Writer | `agents/summary_skills_writer.py` | summary + skills line + budgets + matched skills + intent | new summary, reordered skills |
| 5 | Assembler *(pure code)* | `assembler.py` | original docx + accepted rewrites | tailored docx, layout checks, page count |
| 6 | Reviewer / Guard | `agents/reviewer.py`, `agents/reviewer_llm.py` | original vs proposed, vocabulary, facts | `Verdict(accept, reason)` |

Two supporting LLM helpers sit beside them: `agents/alias.py` (is a JD skill an alias for, or
merely adjacent to, something in your inventory?) and `agents/project_rank.py` (relevance scores).

Every agent module exposes exactly one `run(...)` returning a Pydantic model, keeps its prompt as a
module-level constant next to it, and never touches the document — agents work on text and return
structured data keyed by paragraph id.

### Module map

```
app.py                        Streamlit UI (upload, analyse, approve, rewrite, accept/reject, download)
resume_tailor/
  settings.py                 Settings.from_env() — .env + environment, pydantic
  llm.py                      generate_json(prompt, schema, case); providers; usage; record/replay; pricing
  schemas.py                  every LLM output as a Pydantic model (JDIntent, JDKeywords, SkillMatch,
                              ProjectFact, BulletRewrite, Verdict, Revert, TokenUsage, …)
  docx_io.py                  load / iter_paragraphs (incl. table cells) / Para / set_text / save / structure_diff
  sections.py                 classify() → header | summary | exp_meta | exp_bullet | skills | education | other
  knowledge.py                load_projects, allowed_vocabulary, extract_numbers
  matching.py                 build_inventory, match (matched/adjacent/missing), apply_approvals, rank_projects
  ats_score.py                coverage(text, keywords), present_keywords, missing_keywords
  assembler.py                apply, check_layout, char_change, page_count (LibreOffice), text_diffs
  cache.py                    sha256(cv + jd + kb + settings) → .cache/*.json
  pipeline.py                 analyse(), run(), run_experience(), assemble() — the orchestration
  agents/                     the six agents + alias.py and project_rank.py
```

### The reviewer (Agent 6)

`reviewer.review(orig, new, vocab, fact, *, sources, semantic, case) -> Verdict` is the last thing
between a rewrite and the document, and it deliberately **re-derives every rule from the sources**
instead of trusting the writer that produced the text. A writer that stops checking, a poisoned
recording or a hand-edited cache entry is caught here. Four checks, cheapest and most certain first,
short-circuiting:

1. `term not in CV/KB: React` — a capitalised or product-name token in the rewrite that is in
   neither the allowed vocabulary nor the original paragraph.
2. `metric missing: 93%` — a number the original carried that the rewrite dropped.
3. `length: 34 vs budget 20-24` — outside ±10% words.
4. Only if all three pass: **one** LLM call asking whether the new text claims anything the
   original, the project notes and the permitted sources do not support ("led a team of 10").

Each paragraph is shown exactly the sources its writer was allowed to use: a bullet sees its own
project's notes, the summary sees the whole CV plus the knowledge base. The skills line is a
delimiter-separated list of terms rather than a set of claims, so it is reviewed deterministically
only. `REVIEWER_SEMANTIC=0` disables step 4 everywhere.

The reviewer never rewrites: when `Verdict.accept` is false the original text stays, and a
`Revert(para_id, section, reason, original, rejected)` lands in `RunResult.reverts` for the UI.

### Layout preservation

- Every paragraph is iterated, **including table cells**; each `Para` carries
  `id, text, words, chars, style, is_numbered, literal_bullet_prefix, location`.
- `set_text` keeps run[0]'s formatting (font, size, bold, italic), drops the other runs and
  re-applies a literal bullet prefix (`•    `) when the original had one.
- Only `summary`, `skills` and `exp_bullet` paragraphs are ever written to. Everything else is
  byte-identical after assembly.
- `assembler.check_layout(original, output, allowed_ids, pages=…)` returns a list of problems and
  must come back empty: only `text` diffs, only on the rewritten paragraph ids, total characters
  within ±3%, and — when LibreOffice is installed — an unchanged page count.

## LLM access: providers, record/replay, cost

Everything goes through one function:

```python
llm.generate_json(prompt: str, schema: type[BaseModel], *, case: str) -> BaseModel
```

**Providers** (`LLM_PROVIDER`, or the sidebar):

- `gemini` — via the `google-genai` SDK in JSON mode with `response_schema`.
- `groq` — OpenAI-compatible HTTPS endpoint, `GROQ_API_KEY`.
- `ollama` — local `/api/chat` with `format=json`, `OLLAMA_BASE_URL`.

Groq and Ollama use `urllib` directly — no extra SDK. Anything else raises `ConfigError`.
Invalid JSON gets one retry, then raises `LLMOutputError`; provider failures are turned into a
readable message rather than a traceback.

**Record / replay** (`LLM_MODE`) is what makes the test suite deterministic and free:

| Mode | Behaviour |
|------|-----------|
| `replay` | Read `tests/recordings/<agent>/<case>.json` and validate against the schema. Missing file → `RecordingMissing`. No network, no key. |
| `record` | Call the provider and write the recording. Bypasses the cache. |
| `live` | Call the provider, write nothing. |

**Cost.** `llm.last_usage()` reports input/output tokens per call and the pipeline sums them;
`PRICES_GBP_PER_1M` holds per-model prices (converted from USD at `USD_TO_GBP`) and
`estimate_cost()` turns usage into pounds. `RunResult.usage` carries
`total_input_tokens`, `total_output_tokens` and `estimated_cost_gbp`, which the sidebar displays.
The target is **under £0.01 per run** on Gemini Flash.

## Caching

`cache.py` keys on `sha256(cv_bytes + jd_text + kb_text + json(settings))`, plus the case,
approvals and stages for a run. Both `analyse()` and `run()` results are stored as JSON under
`CACHE_DIR` (default `.cache/`, gitignored) and served with **no LLM calls** on identical inputs;
the result comes back with `from_cache=True`. Pass `use_cache=False` to bypass it; `LLM_MODE=record`
always bypasses it. Tests get an isolated cache directory via an autouse fixture. The sidebar has a
**Clear cache** button.

## Testing

```bash
uv run pytest              # 141 passed, 1 skipped — deterministic, offline, no API key
uv run pytest -m live      # calls the real provider; skipped without GEMINI_API_KEY
```

`pyproject.toml` sets `addopts = "-m 'not live'"`, so the plain command never touches the network.

- Recorded model outputs live in `tests/recordings/<agent>/<case>.json`.
- Reviewer recording cases are a hash of `(original, rewrite, project, sources)`, so identical
  reviews share one recording. `uv run python tests/fixtures/record_reviewer.py` runs the normal
  replay suite and makes a real call **only** where a reviewer recording is missing; repeat until green.
- The fixture CV is built from `tests/fixtures/cv_text.txt` by `tests/fixtures/build_fixture.py`.
- Three job descriptions in `tests/fixtures/jds/` drive `tests/test_regression.py`, which asserts
  the end-to-end promises: no forbidden skill reaches the document, no metric is lost, every revert
  carries a reason, a professional project ranks top, and the document stays within ±3% characters.
- The Streamlit UI is tested with `streamlit.testing.v1.AppTest`.

Never fake a recording by hand — produce them with `LLM_MODE=record`.

## Using it as a library

The pipeline is usable without Streamlit:

```python
from pathlib import Path
from resume_tailor import pipeline

cv_bytes = Path("my_cv.docx").read_bytes()
jd_text = Path("job.txt").read_text()

analysis = pipeline.analyse(cv_bytes, jd_text, "knowledge/projects.md")
print(analysis.intent.role_summary)
print("matched:", analysis.match.matched)
print("adjacent (need approval):", analysis.match.adjacent)
print("missing (never used):", analysis.match.missing)

result = pipeline.run(
    cv_bytes,
    jd_text,
    "knowledge/projects.md",
    approved_adjacent=["Flask"],          # only what you are happy to claim
    stages=("summary", "skills", "experience"),
    analysis=analysis,                    # reuse it: no second JD analysis
)

for revert in result.reverts:
    print("reverted", revert.para_id, "-", revert.reason)
print(f"ATS coverage {result.coverage_before:.0%} -> {result.coverage_after:.0%}")
print(f"cost ~ GBP {result.usage.estimated_cost_gbp:.4f}")

# drop a bullet you did not like and rebuild — no LLM calls
Path("tailored.docx").write_bytes(pipeline.assemble(result, rejected=["p12"]))
```

## Project status

All build tickets are **Done**; the remaining work is a human QA pass.

| Ticket | Title | Status |
|--------|-------|--------|
| SCRUM-8 | Scaffold, Gemini JSON wrapper, record/replay harness | Done |
| SCRUM-9 | DOCX round-trip, section classification, formatting-preserving edits | Done |
| SCRUM-10 | Knowledge base loader and allowed vocabulary | Done |
| SCRUM-11 | Agent 1 (JD Intent) + Agent 2 (Keywords) with Streamlit v0 | Done |
| SCRUM-12 | Skill matching, project ranking, ATS coverage | Done |
| SCRUM-13 | Agent 4 (Summary & Skills) + Assembler → first downloadable .docx | Done |
| SCRUM-14 | Agent 3 (Experience Writer) with per-bullet accept/reject | Done |
| SCRUM-15 | Cache, token/cost display, provider switch (Groq / Ollama) | Done |
| SCRUM-16 | Agent 6 (Reviewer/Guard), layout checks, 3-JD regression suite | Done |
| SCRUM-17 | Manual QA checklist | Ready for review — [`docs/QA_CHECKLIST.md`](docs/QA_CHECKLIST.md) |

Most of the QA checklist is already proven by automated tests. What genuinely needs a human:
read the rewritten summary aloud for tone, click one bullet reject in the UI, and open the
downloaded file in Word to confirm it is still one page — the one check nothing automated can do.

**Not in scope** (deliberately): PDF input, cover letters, multi-CV management, hosting/auth,
and adding new sections to the CV.

## Troubleshooting

**"LLM_MODE is `replay` and there is no recording for this job description."**
Working as designed — replay mode only knows the recorded fixtures. Set `LLM_MODE=live` in `.env`
(with a valid key) to analyse a fresh advert.

**HTTP 429 / `RESOURCE_EXHAUSTED` from Gemini.**
The key is over its quota or monthly spending cap. Raise the cap, or switch provider:
`LLM_PROVIDER=groq` with a `GROQ_API_KEY`, or run locally with `LLM_PROVIDER=ollama`.

**Page count is never checked.**
`assembler.page_count` shells out to LibreOffice (`soffice`) and returns `None` when it is not
installed, so the page check is skipped silently. Install LibreOffice to enable it.

**Nothing changed after a rewrite.**
Check `RunResult.reverts` (the UI shows them as a warning block): the reviewer refused every
rewrite. The reason names the exact rule — an unsupported term, a dropped metric, or a length breach.

**A stale result keeps coming back.**
Identical inputs are served from `.cache/`. Use the sidebar's **Clear cache**, or
`use_cache=False`.

## Repository layout

```
learning/
  app.py                     Streamlit UI
  resume_tailor/             the package (see the module map above)
  knowledge/
    projects.md              your real project facts — GITIGNORED
    projects.example.md      anonymised template (committed)
  tests/
    conftest.py              llm_replay fixture, isolated cache, `live` marker
    fixtures/                cv_text.txt, build_fixture.py, sample_cv.docx, kb_sample.md, jds/
    recordings/<agent>/      recorded model outputs, one JSON per case
    test_*.py
  docs/
    PRD.md                   product requirements and hard rules
    ARCHITECTURE.md          code layout, agent contracts, conventions
    TICKETS.md               ticket map, order, dependencies, status
    QA_CHECKLIST.md          the human verification pass
  CLAUDE.md                  working agreement for AI-assisted sessions
  .env.example  pyproject.toml  uv.lock  .gitignore
```

## Privacy

`.env` (API keys) and `knowledge/projects.md` (personal project data) are gitignored and must
never be committed, pasted into a ticket, or shared. `.cache/` holds run results derived from your
CV and is gitignored too. The CV you upload is processed in memory and sent only to the LLM
provider you configure — pick `ollama` if you want it to stay on your machine.
