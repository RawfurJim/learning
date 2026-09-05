"""SCRUM-9: section classification of Jim's CV fixture."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from resume_tailor.docx_io import iter_paragraphs, load
from resume_tailor.sections import classify, heading_section


def test_classify_fixture(sample_cv_path: Path) -> None:
    paras = iter_paragraphs(load(sample_cv_path))
    sections = classify(paras)
    assert set(sections) == {p.id for p in paras}

    counts = Counter(sections.values())
    assert counts["summary"] == 1
    assert counts["skills"] == 1
    assert counts["education"] == 2
    assert counts["exp_bullet"] == 12

    by_text = {p.text: sections[p.id] for p in paras}
    # Employer line with dates, the three project sub-headings, and Independent Projects.
    assert by_text["AI Engineer — JudgeService Research Ltd     Jan 2024 to Present"] == "exp_meta"
    assert by_text["AI-Powered Review Response System — Live Across Major UK Dealer Groups"] == "exp_meta"
    assert by_text["Customer Insight AI Agent & Sentiment Platform"] == "exp_meta"
    assert by_text["RAG Documentation Engine, QA Automation & Mentoring"] == "exp_meta"
    assert by_text["Independent Projects     Jul 2022 to Dec 2023"] == "exp_meta"
    assert counts["exp_meta"] == 5

    # Name and contact line are header; heading lines are neither editable nor header.
    assert by_text["MD RAWFUR MONZUR JIM"] == "header"
    contact = next(p for p in paras if "rawfurjim12@gmail.com" in p.text)
    assert sections[contact.id] == "header"
    for heading in ("PROFESSIONAL SUMMARY", "PROFESSIONAL EXPERIENCE", "CORE SKILLS & KNOWLEDGE", "EDUCATION"):
        assert by_text[heading] == "other"

    # Every bullet is an exp_bullet and every exp_bullet is a bullet.
    for p in paras:
        assert (sections[p.id] == "exp_bullet") == (p.literal_bullet_prefix is not None), p.id

    # Skills is the single pipe-delimited paragraph.
    skills = next(p for p in paras if sections[p.id] == "skills")
    assert skills.text.count("|") >= 20


def test_heading_keywords_are_case_and_spacing_tolerant() -> None:
    assert heading_section("Professional   Summary:") == "summary"
    assert heading_section("CORE SKILLS & KNOWLEDGE") == "skills"
    assert heading_section("Education") == "education"
    assert heading_section("Built an LLM service that drafts three reply options") is None
