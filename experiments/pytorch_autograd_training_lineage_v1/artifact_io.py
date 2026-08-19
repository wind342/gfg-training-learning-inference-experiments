from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def build_artifact_manifest(experiment_root: Path) -> dict[str, Any]:
    artifact_root = experiment_root / "artifacts"
    rows = []
    for path in sorted(artifact_root.rglob("*.json")):
        if path.name == "artifact_manifest.json":
            continue
        data = path.read_bytes()
        rows.append({
            "path": path.relative_to(experiment_root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        })
    payload = {
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_rule": "all JSON artifacts recursively, excluding artifact_manifest.json itself",
    }
    payload["manifest_rows_sha256"] = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    return payload


def verify_artifact_manifest(experiment_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    seen = set()
    for row in manifest["artifacts"]:
        relative = row["path"]
        if relative in seen:
            mismatches.append({"path": relative, "reason": "duplicate_manifest_path"})
            continue
        seen.add(relative)
        path = experiment_root / relative
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        data = path.read_bytes()
        if len(data) != row["size_bytes"]:
            mismatches.append({"path": relative, "reason": "size"})
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            mismatches.append({"path": relative, "reason": "sha256"})
    return {
        "checked_count": len(manifest["artifacts"]),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "verified": not mismatches and len(seen) == manifest["artifact_count"],
    }
