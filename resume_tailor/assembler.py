"""Agent 5 (Assembler, pure code): put accepted rewrites into the original .docx.

`apply` changes paragraph text only, through `docx_io.set_text`, so fonts, spacing,
bullets and every read-only paragraph stay byte-for-byte as they were. `check_layout`
proves it afterwards: the structure diff may contain nothing but `text` entries (on the
rewritten ids, when they are given), the whole document must stay within +/-3% characters
and, with `pages=True`, the page count must be unchanged.

Page counting needs LibreOffice: `page_count` converts the document to PDF with
`soffice --headless --convert-to pdf` and counts the pages of the result. It returns None
when LibreOffice is not installed, and callers treat None as "cannot check" rather than
as a failure - so the check is skipped, never faked. It is off by default because one
conversion takes seconds and `check_layout` runs in the pipeline's revert loop.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from resume_tailor.docx_io import Diff, iter_paragraphs, load, save, set_text, structure_diff

CHAR_TOLERANCE = 0.03
SKILLS_DELIMITER = " | "
SOFFICE_NAMES = ("soffice", "libreoffice")
SOFFICE_TIMEOUT = 180
# "/Type /Page" as an object type, not the "/Pages" tree node: the next character cannot be "s".
_PDF_PAGE = re.compile(rb"/Type\s*/Page(?![s/\w])")


def join_skills(entries: Iterable[str], delimiter: str = SKILLS_DELIMITER) -> str:
    """One skills-line paragraph from its entries, joined with the original delimiter."""
    return delimiter.join(entry.strip() for entry in entries if entry.strip())


def apply(docx_bytes: bytes, rewrites: Mapping[str, str]) -> bytes:
    """Return a new .docx with `rewrites` (`{para_id: new_text}`) applied, nothing else touched.

    Ids come from `docx_io.iter_paragraphs` on the same bytes. Unknown ids raise `KeyError`.
    """
    doc = load(docx_bytes)
    by_id = {para.id: para for para in iter_paragraphs(doc)}
    for para_id, text in rewrites.items():
        if para_id not in by_id:
            raise KeyError(f"paragraph {para_id!r} is not in the document")
        set_text(by_id[para_id], text)
    return save(doc)


def total_chars(docx_bytes: bytes) -> int:
    return sum(para.chars for para in iter_paragraphs(load(docx_bytes)))


def char_change(original: bytes, output: bytes) -> float:
    """Relative change in total paragraph characters, e.g. 0.012 for +1.2%."""
    before = total_chars(original)
    if before == 0:
        return 0.0
    return (total_chars(output) - before) / before


def soffice() -> str | None:
    """Path to the LibreOffice binary, or None when it is not installed."""
    for name in SOFFICE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def page_count(docx_bytes: bytes) -> int | None:
    """Pages the document prints to, via LibreOffice; None when LibreOffice is unavailable."""
    exe = soffice()
    if exe is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source = work / "cv.docx"
        source.write_bytes(docx_bytes)
        try:
            subprocess.run(
                [exe, "--headless", "--convert-to", "pdf", "--outdir", str(work), str(source)],
                capture_output=True,
                timeout=SOFFICE_TIMEOUT,
                check=True,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        pdfs = list(work.glob("*.pdf"))
        if not pdfs:
            return None
        pages = len(_PDF_PAGE.findall(pdfs[0].read_bytes()))
    return pages or None


def check_layout(
    original: bytes,
    output: bytes,
    allowed_ids: Iterable[str] | None = None,
    *,
    pages: bool = False,
) -> list[str]:
    """Problems with `output` relative to `original`; empty when the assembly is clean.

    Clean means: only `text` diffs (fonts, styles, numbering, spacing and table shapes all
    identical), only on `allowed_ids` when those are given, and total characters within
    +/-`CHAR_TOLERANCE`. With `pages=True` the page count must match as well, which is
    checked only when LibreOffice is installed (`page_count` returns None otherwise).
    """
    allowed = None if allowed_ids is None else set(allowed_ids)
    problems: list[str] = []
    for diff in structure_diff(original, output):
        if diff.kind != "text":
            problems.append(f"{diff.kind} changed at {diff.para_id}: {diff.detail}")
        elif allowed is not None and diff.para_id not in allowed:
            problems.append(f"text changed outside the rewritten paragraphs at {diff.para_id}")
    change = char_change(original, output)
    if abs(change) > CHAR_TOLERANCE:
        problems.append(f"document length changed by {change:+.1%} (limit +/-{CHAR_TOLERANCE:.0%})")
    if pages:
        before, after = page_count(original), page_count(output)
        if before is not None and after is not None and before != after:
            problems.append(f"page count changed from {before} to {after}")
    return problems


def text_diffs(original: bytes, output: bytes) -> list[Diff]:
    return [d for d in structure_diff(original, output) if d.kind == "text"]
