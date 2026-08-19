from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
REPAIRED_BASE_HEAD = "bd5354bc7a91327839b53600349490c621b6804c"
BRANCH = "experiment/order-refund-freeze-inter-fact-relations-v1"
SCENARIOS = (
    "CONCURRENT_REFUND_WINS",
    "CONCURRENT_FREEZE_WINS",
    "LATE_REFUND_AFTER_FREEZE",
    "IDEMPOTENT_DUPLICATE_REFUND",
)
TIMEOUT_SECONDS = 20


class ExperimentError(RuntimeError):
    """Fail-closed experimental error carrying a stable reason code."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def content_id(prefix: str, value: Any) -> str:
    return f"{prefix}{canonical_sha256(value)}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def load_json(path: str | Path) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
