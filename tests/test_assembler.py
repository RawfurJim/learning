"""SCRUM-13: the assembler changes paragraph text and nothing else. Pure Python, no LLM."""

from __future__ import annotations

import pytest

from resume_tailor import assembler
from resume_tailor.docx_io import iter_paragraphs, load, structure_diff
from resume_tailor.sections import classify

SKILLS_DELIMITER = " | "


def _editable_ids(cv_bytes: bytes) -> tuple[str, str]:
    paras = iter_paragraphs(load(cv_bytes))
    sections = classify(paras)
    summary = next(p for p in paras if sections[p.id] == "summary")
    skills = next(p for p in paras if sections[p.id] == "skills")
    return summary.id, skills.id


def test_assembler_only_text_changes(sample_cv_path) -> None:
    original = sample_cv_path.read_bytes()
    summary_id, skills_id = _editable_ids(original)
    paras = {p.id: p for p in iter_paragraphs(load(original))}

    # Same-length rewrites: the summary's words reversed, the skills entries reversed.
    new_summary = " ".join(reversed(paras[summary_id].text.split()))
    entries = [e.strip() for e in paras[skills_id].text.split("|")]
    new_skills = assembler.join_skills(reversed(entries), SKILLS_DELIMITER)
    assert new_skills != paras[skills_id].text and new_skills.count(" | ") == paras[skills_id].text.count(" | ")

    output = assembler.apply(original, {summary_id: new_summary, skills_id: new_skills})
    assert output != original

    diffs = structure_diff(original, output)
    assert diffs, "the rewrites must show up as text diffs"
    assert {d.kind for d in diffs} == {"text"}, diffs
    assert {d.para_id for d in diffs} == {summary_id, skills_id}, diffs

    out_paras = {p.id: p for p in iter_paragraphs(load(output))}
    assert out_paras[summary_id].text == new_summary
    assert out_paras[skills_id].text == new_skills
    assert abs(assembler.char_change(original, output)) <= assembler.CHAR_TOLERANCE
    assert assembler.check_layout(original, output, {summary_id, skills_id}) == []


def test_check_layout_flags_length_and_unexpected_paragraphs(sample_cv_path) -> None:
    original = sample_cv_path.read_bytes()
    summary_id, skills_id = _editable_ids(original)
    paras = {p.id: p for p in iter_paragraphs(load(original))}

    # A change on a paragraph that was not rewritten is reported.
    other_id = next(p.id for p in iter_paragraphs(load(original)) if p.id not in (summary_id, skills_id) and p.text)
    output = assembler.apply(original, {other_id: paras[other_id].text + " extra"})
    problems = assembler.check_layout(original, output, {summary_id, skills_id})
    assert any("outside" in p for p in problems), problems

    # A rewrite that blows the +/-3% character budget is reported.
    output = assembler.apply(original, {summary_id: paras[summary_id].text * 2})
    problems = assembler.check_layout(original, output, {summary_id})
    assert any("length" in p for p in problems), problems


def test_apply_unknown_paragraph_raises(sample_cv_path) -> None:
    with pytest.raises(KeyError):
        assembler.apply(sample_cv_path.read_bytes(), {"p999": "nope"})


def test_apply_keeps_bullet_prefix(sample_cv_path) -> None:
    original = sample_cv_path.read_bytes()
    bullet = next(p for p in iter_paragraphs(load(original)) if p.literal_bullet_prefix)
    output = assembler.apply(original, {bullet.id: "• Rewritten bullet text."})
    out = next(p for p in iter_paragraphs(load(output)) if p.id == bullet.id)
    assert out.literal_bullet_prefix == bullet.literal_bullet_prefix
    assert out.text == "Rewritten bullet text."
