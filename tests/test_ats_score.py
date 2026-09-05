"""SCRUM-12: ATS keyword coverage is plain Python, case-insensitive and multiword aware."""

from __future__ import annotations

from resume_tailor.ats_score import coverage, missing_keywords, present_keywords

TEXT = (
    "AI Engineer building production LLM systems. Hands-on across the full stack: "
    "Fine-Tuning, evaluation design, FastAPI services, Docker, and GPU serving on AWS. "
    "Skills: Python | Vector Databases | CI/CD (GitHub Actions)"
)


def test_coverage_math() -> None:
    keywords = ["python", "fine-tuning", "Docker", "aws", "React", "Kubernetes", "Terraform", "TypeScript"]
    assert coverage(TEXT, keywords) == 0.5  # 4 of 8, case-insensitive, multiword "fine-tuning" counted
    assert present_keywords(TEXT, keywords) == ["python", "fine-tuning", "Docker", "aws"]
    assert missing_keywords(TEXT, keywords) == ["React", "Kubernetes", "Terraform", "TypeScript"]


def test_coverage_multiword_and_separators() -> None:
    assert coverage(TEXT, ["vector databases"]) == 1.0
    assert coverage(TEXT, ["fine tuning"]) == 1.0  # hyphen and space are the same separator
    assert coverage(TEXT, ["GitHub Actions", "CI/CD"]) == 1.0


def test_coverage_respects_word_boundaries() -> None:
    assert coverage("We use JavaScript here", ["Java"]) == 0.0
    assert coverage("Trained on 100k reviews", ["100k"]) == 1.0


def test_coverage_empty_keywords_is_zero() -> None:
    assert coverage(TEXT, []) == 0.0
    assert coverage("", ["Python"]) == 0.0
