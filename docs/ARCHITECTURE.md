# ResumeTailor — Architecture and conventions

## Code layout (target state after all tickets)

```
learning/
  app.py                          # Streamlit UI
  resume_tailor/
    settings.py                   # pydantic Settings from .env
    llm.py                        # generate_json(prompt, schema, *, case) ; providers ; usage ; record/replay
    schemas.py                    # pydantic models: JDIntent, JDKeywords, SkillMatch, ProjectFact, rewrites, Verdict
    docx_io.py                    # load / iter_paragraphs (incl. tables) / Para / set_text / save / structure_diff
    sections.py                   # classify paragraphs -> header|summary|exp_meta|exp_bullet|skills|education|other
    knowledge.py                  # load_projects, allowed_vocabulary, extract_numbers
    matching.py                   # build_inventory, match (matched/adjacent/missing), rank_projects
    ats_score.py                  # coverage(text, keywords)
    agents/
      jd_intent.py                # Agent 1
      keywords.py                 # Agent 2
      alias.py                    # LLM alias / adjacency resolution used by matching
      experience_writer.py        # Agent 3
      summary_skills_writer.py    # Agent 4
      reviewer.py  reviewer_llm.py# Agent 6 (deterministic checks first, LLM drift check last)
    assembler.py                  # Agent 5: apply rewrites, check_layout, page_count (LibreOffice optional)
    pipeline.py                   # orchestration: run(cv_bytes, jd_text, kb_path, approved_adjacent, stages)
    cache.py                      # sha256(cv + jd + kb + settings) -> .cache/*.json
  knowledge/
    projects.md                   # Jim's real project facts (GITIGNORED)
    projects.example.md           # anonymised example (committed)
  tests/
    conftest.py                   # llm_replay fixture, `live` marker
    fixtures/cv_text.txt          # Jim's CV text (source for the fixture docx)
    fixtures/build_fixture.py     # builds fixtures/sample_cv.docx from cv_text.txt
    fixtures/sample_cv.docx
    fixtures/kb_sample.md         # copy of the knowledge base used by tests
    fixtures/jds/                 # senior_ai_engineer.txt, ds_nlp.txt, ml_platform.txt
    recordings/<agent>/<case>.json# recorded Gemini outputs for deterministic tests
    test_*.py
  docs/                           # PRD.md, ARCHITECTURE.md, TICKETS.md
  .env.example  pyproject.toml  .gitignore
```

## LLM access

- One entry point: `resume_tailor.llm.generate_json(prompt: str, schema: type[BaseModel], *, case: str) -> BaseModel`.
- Default provider Gemini via `google-genai`, JSON mode with `response_schema=schema`. Default model `gemini-3.6-flash` (`LLM_MODEL`): the PRD names Gemini 2.5 Flash, but on 2026-09-05 the API rejected `gemini-2.5-flash` as "no longer available to new users" and recommended `gemini-3.6-flash`.
- `LLM_MODE`:
  - `replay` (default in tests): read `tests/recordings/<agent>/<case>.json`, validate with schema. Missing file -> `RecordingMissing`.
  - `record`: call the provider and write the recording.
  - `live`: call the provider, write nothing.
- Invalid JSON -> one retry -> `LLMOutputError`.
- `llm.last_usage()` returns input/output tokens and model; pipeline sums them.
- `LLM_PROVIDER` = `gemini | groq | ollama` (Groq/Ollama arrive in SCRUM-15).

## Agent contract

Each agent module exposes `run(...) -> <PydanticModel>` and keeps its prompt as a module constant. Agents never touch the docx; they work on text and return structured data keyed by paragraph id. Every rule that can be checked in Python (vocabulary, numbers, length) is checked in Python, not delegated to the model.

## Docx editing rules

- Iterate every paragraph, including table cells. Each `Para` has `id, text, words, chars, style, is_numbered, literal_bullet_prefix, location`.
- `set_text` keeps run[0] formatting (font name, size, bold, italic), removes other runs, re-applies a literal `•    ` prefix if the original had one.
- Only `summary`, `skills` and `exp_bullet` paragraphs are ever rewritten. Everything else is byte-identical after assembly.
- `structure_diff(original, output)` must report only `text` changes.
- Length: per paragraph +/-10% words, whole document +/-3% chars. When LibreOffice (`soffice`) is available, page count must equal the original.

## Jim's CV specifics (fixture)

- Header: name line, then contact line with `|` separators.
- Sections: PROFESSIONAL SUMMARY (1 paragraph), PROFESSIONAL EXPERIENCE (employer/date line, three project sub-headings with 3/4/3 bullets, then "Independent Projects" with 2 bullets), CORE SKILLS & KNOWLEDGE (one `|`-delimited paragraph), EDUCATION (2 lines).
- Bullets start with the literal prefix `•    ` (bullet + 4 spaces).
- Metrics that must always survive: 93%, 100k, 0.78, 0.96, 0.98, 20%, 90%, 2,000-character, 500 (golden set).

## Knowledge base format (`knowledge/projects.md`)

```
## <Project name>
category: work | personal
employer: <text or ->
period: <text>
problem: <one paragraph>
built: <one paragraph>
stack: <comma-separated tools>
metrics: <comma-separated metric tokens, e.g. 93%, 100k reviews/month>
keywords: <comma-separated>
```

`ProjectFact.professional = (category == "work")`. Ranking multiplies personal-project scores by 0.5.

## Testing conventions

- `uv run pytest` must pass with no network and no API key.
- `@pytest.mark.live` tests call Gemini and are skipped when `GEMINI_API_KEY` is absent.
- Streamlit UI is tested with `streamlit.testing.v1.AppTest`.
- Test names in the Jira ticket are the contract; implement them with those exact names.

## Git

- One commit per ticket: `SCRUM-<n>: <title>`. Do not push unless asked.
- `.gitignore`: `.env`, `knowledge/projects.md`, `.cache/`, `.venv/`, `__pycache__/`.
