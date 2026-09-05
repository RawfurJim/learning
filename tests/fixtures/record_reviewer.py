"""Record the reviewer verdicts the replayed tests need (`uv run python tests/fixtures/record_reviewer.py`).

Agent 6 asks one semantic-drift question per rewritten paragraph, so every replayed
pipeline run needs a `tests/recordings/reviewer/<hash>.json` for each rewrite it produces.
Those hashes come from the reviewed text itself (`reviewer_llm.case_for`), so they are only
knowable by running the tests; and a plain `LLM_MODE=record` run would call Gemini for
*every* agent and rewrite the other recordings too.

This script therefore runs the normal (replay) test suite with one change: when a reviewer
recording is missing, that single call is made against the real API in record mode and the
verdict is written to `tests/recordings/reviewer/`. Nothing else is re-recorded, and no
verdict is invented - every file comes from a real Gemini answer. Needs `GEMINI_API_KEY`.

Repeat until the suite is green: each run gets further and uncovers the next rewrite.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from resume_tailor import llm  # noqa: E402
from resume_tailor.agents import reviewer_llm  # noqa: E402
from resume_tailor.schemas import DriftVerdict, ProjectFact  # noqa: E402

_lock = threading.Lock()  # the pipeline reviews in parallel; LLM_MODE is process-wide
_original_check = reviewer_llm.check
recorded: list[str] = []


def check(
    original: str,
    rewrite: str,
    fact: ProjectFact | None = None,
    *,
    sources: str = "",
    case: str | None = None,
) -> DriftVerdict:
    name = case or reviewer_llm.case_for(original, rewrite, fact, sources)
    if "/" not in name:
        name = f"{reviewer_llm.AGENT}/{name}"
    with _lock:
        if llm.recording_path(name, REPO_ROOT / "tests" / "recordings").exists():
            return _original_check(original, rewrite, fact, sources=sources, case=name)
        saved = {key: os.environ.get(key) for key in ("LLM_MODE", "LLM_RECORDINGS_DIR")}
        os.environ["LLM_MODE"] = "record"
        os.environ.pop("LLM_RECORDINGS_DIR", None)  # always write to tests/recordings/
        llm.set_provider(None)
        try:
            verdict = _original_check(original, rewrite, fact, sources=sources, case=name)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            llm.set_provider(None)
        recorded.append(name)
        print(f"  recorded {name}: supported={verdict.supported}")
        return verdict


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set; nothing can be recorded.", file=sys.stderr)
        return 2

    reviewer_llm.check = check
    code = pytest.main(["-q", *(sys.argv[1:] or [str(REPO_ROOT / "tests")])])
    print(f"\n{len(recorded)} reviewer recording(s) written.")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
