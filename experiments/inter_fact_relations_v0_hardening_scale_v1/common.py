from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
FROZEN_SOURCE_COMMIT = "ad19cbb701e7c9d6bc2426756a252039c3119601"
EXPERIMENT_BRANCH = "maintenance/inter-fact-relations-v0-hardening-scale-v1"

PROTECTED_TREE_HASHES = {
    "src/generation_relation_core": "03fbdce13249f84abe9d8fb605da31cdc36eda27",
    "protocol/core_v3": "0b4a2608864e771ebca7cdbfad95aabaed2d0723",
    "compat/v2": "7bbb49d18daf7ea99d7633b40c6df5bc002824ca",
    "tests/core": "280cb44d592ae48d986719638980c11e57aab1f9",
    "experiments/inter_fact_relations_v0": (
        "fccb595dfc0a8c7272f3e6e2af6937a57f8168b7"
    ),
}


class ExperimentError(ValueError):
    """Fail-closed experiment error with a registered reason code."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def content_id(prefix: str, value: Any) -> str:
    return f"{prefix}{canonical_sha256(value)}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
