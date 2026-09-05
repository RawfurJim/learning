"""Load, walk, edit and save .docx files without disturbing their formatting.

Only paragraph *text* is ever changed. `set_text` keeps the first run's formatting,
drops the other runs and re-applies the literal `•    ` bullet prefix Jim's CV uses.
`structure_diff` proves that nothing but text changed between two documents.
"""

from __future__ import annotations

import copy
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

from docx import Document as _open_document
from docx.document import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

DiffKind = Literal["text", "style", "font", "numbering", "table"]

# Jim's CV writes bullets as literal text: the bullet character followed by spaces.
_BULLET_PREFIX_RE = re.compile(r"^(•[ \t ]*)")


@dataclass
class Para:
    """One paragraph of the document, wherever it lives (body or table cell).

    `text` is the editable content *without* the literal bullet prefix; the prefix,
    if any, is kept in `literal_bullet_prefix` and re-applied by `set_text`.
    `words`/`chars` are budgets computed on `text`.
    """

    id: str
    text: str
    words: int
    chars: int
    style: str
    is_numbered: bool
    literal_bullet_prefix: str | None
    location: str
    paragraph: Paragraph = field(repr=False, compare=False)

    @property
    def full_text(self) -> str:
        """Text as it appears in the document, prefix included."""
        return (self.literal_bullet_prefix or "") + self.text


@dataclass(frozen=True)
class Diff:
    kind: DiffKind
    para_id: str | None
    detail: str


def load(src: bytes | str | Path | BinaryIO) -> Document:
    """Open a .docx from bytes, a path, or a binary file object."""
    if isinstance(src, bytes):
        return _open_document(io.BytesIO(src))
    if isinstance(src, (str, Path)):
        return _open_document(str(src))
    return _open_document(src)


def save(doc: Document) -> bytes:
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def split_bullet_prefix(text: str) -> tuple[str | None, str]:
    """Return `(prefix, body)` where prefix is the literal bullet run, if present."""
    match = _BULLET_PREFIX_RE.match(text)
    if not match:
        return None, text
    prefix = match.group(1)
    return prefix, text[len(prefix):]


def _is_numbered(paragraph: Paragraph) -> bool:
    p_pr = paragraph._p.pPr
    return p_pr is not None and p_pr.find(qn("w:numPr")) is not None


def _walk(parent_element, doc: Document, location: str):
    """Yield `(Paragraph, location)` in document order, descending into tables."""
    for child in parent_element.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc), location
        elif child.tag == qn("w:tbl"):
            table = Table(child, doc)
            table_index = _table_counter(doc)
            for r, row in enumerate(table.rows):
                for c, cell in enumerate(row.cells):
                    cell_location = f"{location}/table:{table_index}/row:{r}/col:{c}"
                    yield from _walk(cell._tc, doc, cell_location)


def _table_counter(doc: Document) -> int:
    counter = getattr(doc, "_rt_table_counter", 0)
    doc._rt_table_counter = counter + 1  # type: ignore[attr-defined]
    return counter


def iter_paragraphs(doc: Document) -> list[Para]:
    """Every paragraph in the body, including those inside table cells, in order."""
    doc._rt_table_counter = 0  # type: ignore[attr-defined]
    paras: list[Para] = []
    for index, (paragraph, location) in enumerate(_walk(doc.element.body, doc, "body")):
        prefix, body = split_bullet_prefix(paragraph.text)
        paras.append(
            Para(
                id=f"p{index}",
                text=body,
                words=len(body.split()),
                chars=len(body),
                style=paragraph.style.name if paragraph.style is not None else "",
                is_numbered=_is_numbered(paragraph),
                literal_bullet_prefix=prefix,
                location=location,
                paragraph=paragraph,
            )
        )
    return paras


def set_text(para: Para | Paragraph, new_text: str) -> Para | Paragraph:
    """Replace a paragraph's text, keeping run[0] formatting and the bullet prefix.

    Any leading bullet in `new_text` is stripped so the prefix is never doubled.
    When given a `Para`, its `text`/`words`/`chars` are updated in place.
    """
    paragraph = para.paragraph if isinstance(para, Para) else para
    if isinstance(para, Para):
        prefix = para.literal_bullet_prefix
    else:
        prefix, _ = split_bullet_prefix(paragraph.text)

    _, body = split_bullet_prefix(new_text)
    body = body.strip()

    runs = paragraph.runs
    if runs:
        keep = runs[0]
        for extra in runs[1:]:
            extra._r.getparent().remove(extra._r)
    else:
        keep = paragraph.add_run()
    keep.text = (prefix or "") + body

    if isinstance(para, Para):
        para.text = body
        para.words = len(body.split())
        para.chars = len(body)
    return para


def _run_font_signature(paragraph: Paragraph) -> tuple:
    runs = paragraph.runs
    if not runs:
        return ()
    font = runs[0].font
    return (font.name, font.size, font.bold, font.italic, font.underline)


def _p_pr_parts(paragraph: Paragraph) -> tuple[str, str, str]:
    """Return `(pStyle xml, numPr xml, remaining pPr xml)` for comparison."""
    p_pr = paragraph._p.pPr
    if p_pr is None:
        return "", "", ""
    clone = copy.deepcopy(p_pr)
    style_el = clone.find(qn("w:pStyle"))
    num_el = clone.find(qn("w:numPr"))
    style_xml = _xml(style_el)
    num_xml = _xml(num_el)
    for el in (style_el, num_el):
        if el is not None:
            clone.remove(el)
    return style_xml, num_xml, _xml(clone)


def _xml(element) -> str:
    if element is None:
        return ""
    from lxml import etree

    return etree.tostring(element, encoding="unicode")


def _table_shapes(doc: Document) -> list[tuple[int, int]]:
    return [(len(t.rows), len(t.columns)) for t in doc.tables]


def structure_diff(a: Document | bytes, b: Document | bytes) -> list[Diff]:
    """List every difference between two documents, classified by kind.

    A clean tailoring run reports only `text` diffs for the rewritten paragraphs.
    """
    doc_a = load(a) if isinstance(a, bytes) else a
    doc_b = load(b) if isinstance(b, bytes) else b
    diffs: list[Diff] = []

    shapes_a, shapes_b = _table_shapes(doc_a), _table_shapes(doc_b)
    if shapes_a != shapes_b:
        diffs.append(Diff("table", None, f"tables {shapes_a} -> {shapes_b}"))

    paras_a, paras_b = iter_paragraphs(doc_a), iter_paragraphs(doc_b)
    for pa, pb in zip(paras_a, paras_b):
        if pa.full_text != pb.full_text:
            diffs.append(Diff("text", pa.id, f"{pa.full_text!r} -> {pb.full_text!r}"))
        style_a, num_a, rest_a = _p_pr_parts(pa.paragraph)
        style_b, num_b, rest_b = _p_pr_parts(pb.paragraph)
        if pa.style != pb.style or style_a != style_b or rest_a != rest_b:
            diffs.append(Diff("style", pa.id, f"{pa.style!r} -> {pb.style!r}"))
        if num_a != num_b or pa.is_numbered != pb.is_numbered:
            diffs.append(Diff("numbering", pa.id, "numbering changed"))
        font_a, font_b = _run_font_signature(pa.paragraph), _run_font_signature(pb.paragraph)
        if font_a != font_b:
            diffs.append(Diff("font", pa.id, f"{font_a} -> {font_b}"))
        if pa.location != pb.location:
            diffs.append(Diff("table", pa.id, f"{pa.location} -> {pb.location}"))

    for extra in paras_a[len(paras_b):]:
        diffs.append(Diff("text", extra.id, "paragraph removed"))
    for extra in paras_b[len(paras_a):]:
        diffs.append(Diff("text", extra.id, "paragraph added"))
    return diffs
