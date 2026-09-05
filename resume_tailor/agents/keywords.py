"""Agent 2 (Keyword Extractor): mandatory / nice-to-have skills, ATS keywords and responsibilities.

The model does the reading; Python does the rules. `normalise` strips, deduplicates
case-insensitively, applies canonical casing, removes nice-to-have items that are
also mandatory, and makes `ats_keywords` a superset of `mandatory`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from resume_tailor import llm
from resume_tailor.agents import default_case
from resume_tailor.schemas import JDKeywords

AGENT = "keywords"

PROMPT = """You are an ATS (applicant tracking system) specialist. Extract the skills and keywords a recruiter's ATS would scan a CV for, from the job description (JD) below. Use only what the JD says.

Return a JSON object with exactly these fields:
- mandatory: skills, tools, technologies and qualifications the JD marks as required (sections such as "must-have", "essential", "requirements", "about you", or phrased as "you have", "strong", "expert", "experience with"). Include the specific technologies named there (for example "Python", "PyTorch", "AWS"). Do not include items the JD lists as desirable.
- nice_to_have: skills and technologies the JD marks as optional ("nice-to-have", "desirable", "bonus", "ideally", "familiarity with", "a plus"). Never repeat an item from mandatory.
- ats_keywords: every skill, tool, technology, method and domain term from the whole JD that an ATS could match on. This list must contain every item of mandatory and nice_to_have, plus terms that appear only in the responsibilities or company description.
- responsibilities: 5-10 short phrases (3-10 words each) for what the person will actually do, in the JD's own words.

Rules for skill items:
- Write each skill as a short name as it appears in the JD, not a sentence: "Kubernetes", not "Kubernetes experience for model serving".
- Split combined items: "Docker and CI/CD" becomes two items "Docker" and "CI/CD".
- Keep well-known acronyms and product names in their usual casing (PyTorch, AWS, SQL, RAG, scikit-learn).
- No duplicates within or across mandatory and nice_to_have. Do not invent anything that is not in the JD.

Job description:
<<<
{jd_text}
>>>
"""

# Casing an ATS and a human both expect, regardless of how the JD or the model wrote it.
CANONICAL = {
    "python": "Python",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "rag": "RAG",
    "retrieval-augmented generation": "Retrieval-Augmented Generation",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "llm": "LLM",
    "llms": "LLMs",
    "nlp": "NLP",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "docker": "Docker",
    "react": "React",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "sql": "SQL",
    "nosql": "NoSQL",
    "postgresql": "PostgreSQL",
    "ci/cd": "CI/CD",
    "github actions": "GitHub Actions",
    "mlflow": "MLflow",
    "mlops": "MLOps",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
    "spacy": "spaCy",
    "nltk": "NLTK",
    "pandas": "pandas",
    "numpy": "NumPy",
    "terraform": "Terraform",
    "helm": "Helm",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "streamlit": "Streamlit",
    "power bi": "Power BI",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "vllm": "vLLM",
    "sagemaker": "SageMaker",
    "lambda": "Lambda",
    "ecs": "ECS",
    "eks": "EKS",
    "s3": "S3",
    "iam": "IAM",
    "openai": "OpenAI",
    "langchain": "LangChain",
    "a/b testing": "A/B testing",
}

_WS = re.compile(r"\s+")


def _key(item: str) -> str:
    """Case-insensitive, whitespace-collapsed identity of a keyword."""
    return _WS.sub(" ", item).strip().lower()


def _jd_casing(item: str, jd_text: str) -> str | None:
    """The most common casing of `item` in the JD text, if it appears there at all."""
    pattern = r"\s+".join(re.escape(part) for part in item.split())
    found = re.findall(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", jd_text, flags=re.IGNORECASE)
    if not found:
        return None
    variants = Counter(_WS.sub(" ", match) for match in found)
    return max(variants, key=lambda v: (variants[v], -list(variants).index(v)))


def canonical(item: str, jd_text: str = "") -> str:
    """Stripped, single-spaced, canonically cased form of one keyword."""
    clean = _WS.sub(" ", item).strip()
    key = clean.lower()
    if key in CANONICAL:
        return CANONICAL[key]
    return _jd_casing(clean, jd_text) or clean


def _dedupe(items: Iterable[str], jd_text: str, exclude: Iterable[str] = ()) -> list[str]:
    seen = {_key(x) for x in exclude}
    out: list[str] = []
    for item in items:
        value = canonical(item, jd_text)
        if not value or _key(value) in seen:
            continue
        seen.add(_key(value))
        out.append(value)
    return out


def normalise(raw: JDKeywords, jd_text: str = "") -> JDKeywords:
    """Deterministic clean-up of the model output; see module docstring."""
    mandatory = _dedupe(raw.mandatory, jd_text)
    nice_to_have = _dedupe(raw.nice_to_have, jd_text, exclude=mandatory)
    ats_keywords = _dedupe([*raw.ats_keywords, *mandatory, *nice_to_have], jd_text)
    responsibilities = _dedupe(raw.responsibilities, jd_text)
    return JDKeywords(
        mandatory=mandatory,
        nice_to_have=nice_to_have,
        ats_keywords=ats_keywords,
        responsibilities=responsibilities,
    )


def build_prompt(jd_text: str) -> str:
    return PROMPT.format(jd_text=jd_text.strip())


def run(jd_text: str, *, case: str | None = None) -> JDKeywords:
    """Extract keywords from `jd_text`. `case` names the recording (`tests/recordings/keywords/<case>.json`)."""
    case = case or default_case(jd_text)
    raw = llm.generate_json(build_prompt(jd_text), JDKeywords, case=f"{AGENT}/{case}")
    return normalise(raw, jd_text)
