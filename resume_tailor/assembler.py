"""Agent 5 (Assembler, pure code): put accepted rewrites into the original .docx.

`apply` changes paragraph text only, through `docx_io.set_text`, so fonts, spacing,
bullets and every read-only paragraph stay byte-for-byte as they were. `check_layout`
proves it afterwards: the structure diff may contain nothing but `text` entries for the
rewritten ids, and the whole document must stay within +/-3% characters. Page counting
(LibreOffice) arrives with the reviewer ticket.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from resume_tailor.docx_io import Diff, iter_paragraphs, load, save, set_text, structure_diff

CHAR_TOLERANCE = 0.03
SKILLS_DELIMITER = " | "


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


def check_layout(original: bytes, output: bytes, allowed_ids: Iterable[str]) -> list[str]:
    """Problems with `output` relative to `original`; empty when the assembly is clean.

    Clean means: only `text` diffs, only on `allowed_ids`, and total characters within
    +/-`CHAR_TOLERANCE`.
    """
    allowed = set(allowed_ids)
    problems: list[str] = []
    for diff in structure_diff(original, output):
        if diff.kind != "text":
            problems.append(f"{diff.kind} changed at {diff.para_id}: {diff.detail}")
        elif diff.para_id not in allowed:
            problems.append(f"text changed outside the rewritten paragraphs at {diff.para_id}")
    change = char_change(original, output)
    if abs(change) > CHAR_TOLERANCE:
        problems.append(f"document length changed by {change:+.1%} (limit +/-{CHAR_TOLERANCE:.0%})")
    return problems


def text_diffs(original: bytes, output: bytes) -> list[Diff]:
    return [d for d in structure_diff(original, output) if d.kind == "text"]
