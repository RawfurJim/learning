# SCRUM-17 — Manual QA checklist (human)

Human verification of the whole pipeline now that SCRUM-8..16 are done. Tick each item and
paste evidence (screenshots / test output) into the Jira ticket as comments.

Run the app with `uv run streamlit run app.py`, upload the **real** CV `.docx`, and work
through the three fixture JDs in `tests/fixtures/jds/` plus one fresh JD from LinkedIn.

## Status legend

- **auto** — an automated test already proves this; the named test is the evidence. Re-run
  `uv run pytest` and paste the output.
- **manual** — needs Jim, Word, or the real CV. Only these need a human pass.
- **blocked** — cannot be checked right now; see the note.

| # | Item | Status | Evidence / note |
|--:|------|--------|-----------------|
| 1 | `uv run pytest` green | auto | 141 passed, 1 skipped (5 Sep 2026) |
| 1b | `uv run pytest -m live` green | **blocked** | Gemini key is over its **monthly spending cap** (HTTP 429 `RESOURCE_EXHAUSTED`). Raise the cap at https://ai.studio/spend, or set `LLM_PROVIDER=groq` with a `GROQ_API_KEY`, then re-run. |
| 2 | Upload real CV; run 3 fixture JDs + 1 fresh LinkedIn JD | manual | Fixture JDs pass end-to-end in replay. A **fresh** JD needs a working key (see 1b) — in `replay` mode it stops with "no recording for this JD", which is expected, not a bug. |
| 3 | Intent summary reads correctly; mandatory vs nice-to-have split plausible | manual | Judgement call — read the Analyse panel. |
| 4 | No skill Jim lacks under Matched; adjacent sensible; React + Kubernetes under Missing for `senior_ai_engineer` | auto | `tests/test_matching.py`; confirmed 5 Sep: `senior_ai_engineer` Missing contains `Kubernetes`, `React`. |
| 5 | JudgeService projects rank top for every AI Engineer JD | auto | `tests/test_regression.py::test_regression_three_jds`; top project is a JudgeService one for all 3 JDs. |
| 6 | Summary & skills truthful, JD vocabulary present, no invented numbers, same length | auto + manual | Vocabulary/number/length rules enforced by Agent 6 (`tests/test_reviewer.py`). **Read it aloud** to judge tone. |
| 7 | Reject one bullet, accept the rest, download | manual | Per-bullet accept/reject is covered by `tests/test_agent_experience.py`; click through it once in the UI. |
| 8 | Open the download in Word beside the original: same fonts, bullets, spacing, **one page**; header/contact, employer lines, dates, education untouched | auto + **manual** | Paragraph count and read-only sections are asserted (`tests/test_regression.py`); document length stayed within +/-3% on all 3 JDs (+2.37%, +1.69%, +1.40%). **Page count can only be confirmed in Word — this is the one check nothing automated can do.** |
| 9 | Every revert has an understandable reason | auto | `tests/test_regression.py::test_regression_reverts_carry_a_reason`. Skim the reasons in the UI once. |
| 10 | Sidebar cost under GBP 0.01 per run; second identical run served from cache | auto + manual | `tests/test_cache.py::test_second_run_hits_cache`. Cost shows 0 in replay mode, so the **money figure needs a live run** (blocked by 1b). |
| 11 | Adversarial JD (React / Kubernetes / Java only): output contains none of those terms | auto | `tests/test_regression.py::test_regression_no_forbidden_skill_reaches_the_document`; re-confirmed 5 Sep across all 3 JDs — no forbidden term reached any output. |

## Acceptance criteria

- [ ] All items above ticked with evidence
- [ ] Items 1b, 2 (fresh JD) and 10 (cost) re-run once the Gemini spend cap is lifted
- [ ] Close Epic SCRUM-7 when passed

## What is left for a human

Everything except **3, 6 (tone), 7, 8 (page count in Word)** is already proven by the test
suite. The genuinely manual pass is: read the summary aloud, click one bullet reject, and
open the downloaded file in Word to confirm it is still one page.
