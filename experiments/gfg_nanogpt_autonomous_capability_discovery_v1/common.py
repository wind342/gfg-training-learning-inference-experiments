from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


BASELINE_COMMIT = "f8a9df00dcd4b240d2a911d431553d1b9cc84b7a"
NANOGPT_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
EXPERIMENT_NAME = "gfg-nanogpt-autonomous-capability-discovery-v1"
FINAL_READY = "GFG_AI_AUTONOMOUS_SCIENTIFIC_DISCOVERY_FEASIBLE"
FINAL_NOT_ESTABLISHED = (
    "GFG_AI_AUTONOMOUS_SCIENTIFIC_DISCOVERY_NOT_ESTABLISHED"
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def directory_commitment(root: Path) -> str:
    return payload_sha256(relative_file_manifest(root))


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)
