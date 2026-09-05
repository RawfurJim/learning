"""Run cache: sha256(cv_bytes + jd_text + kb_text + json(settings)) -> `.cache/<key>.json`.

Pipeline results are pydantic models, stored as JSON (bytes fields base64-encoded), so a hit
reproduces the exact output bytes of the first run. The directory comes from `CACHE_DIR`
(default `<repo>/.cache/`, gitignored). `get`, `put` and `clear` take an explicit
`cache_dir` so tests never touch the real cache. Record mode bypasses the cache in the
pipeline so recordings are always produced by a real call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ValidationError

from resume_tailor.settings import Settings

# Bump when the pipeline's output for identical inputs changes shape (new fields, new stages).
CACHE_VERSION = 2


def default_dir() -> Path:
    return Settings.from_env().cache_dir


def cache_key(cv_bytes: bytes, jd_text: str, kb_text: str, settings: Mapping[str, object]) -> str:
    """Hex sha256 over the four inputs; `settings` is serialised as sorted, compact JSON."""
    digest = hashlib.sha256()
    digest.update(cv_bytes)
    for part in (jd_text.strip(), kb_text.strip(), json.dumps(settings, sort_keys=True, default=str)):
        digest.update(b"\x00")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def path_for(key: str, cache_dir: Path | None = None) -> Path:
    return (cache_dir or default_dir()) / f"{key}.json"


def get[T: BaseModel](key: str, model: type[T], cache_dir: Path | None = None) -> T | None:
    """The cached `model` for `key`, or None when absent or unreadable (stale entries are deleted)."""
    path = path_for(key, cache_dir)
    if not path.exists():
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError, OSError):
        path.unlink(missing_ok=True)
        return None


def put(key: str, result: BaseModel, cache_dir: Path | None = None) -> Path:
    path = path_for(key, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(result.model_dump_json(), encoding="utf-8")
    tmp.replace(path)
    return path


def clear(cache_dir: Path | None = None) -> int:
    """Delete every cached result; returns how many were removed."""
    directory = cache_dir or default_dir()
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.glob("*.json"):
        path.unlink(missing_ok=True)
        removed += 1
    for path in directory.glob("*.json.tmp"):
        path.unlink(missing_ok=True)
    return removed


def size(cache_dir: Path | None = None) -> int:
    """How many results are cached."""
    directory = cache_dir or default_dir()
    return sum(1 for _ in directory.glob("*.json")) if directory.exists() else 0
