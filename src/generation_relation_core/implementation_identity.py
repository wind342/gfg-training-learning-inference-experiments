from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from .errors import CoreV3Error


IDENTITY_SCHEME = "tracked_repository_explicit_content_v1"

# This whitelist is deliberately explicit. Tests, examples, experiment code,
# compatibility projections, tools, results, and documentation cannot change
# the runtime Core identity.
IMPLEMENTATION_FILES: tuple[tuple[str, str], ...] = (
    ("src/generation_relation_core/__init__.py", "implementation_module"),
    ("src/generation_relation_core/errors.py", "implementation_module"),
    ("src/generation_relation_core/canonical.py", "implementation_module"),
    ("src/generation_relation_core/schema_registry.py", "implementation_module"),
    ("src/generation_relation_core/entities.py", "implementation_module"),
    ("src/generation_relation_core/predicate_registry.py", "implementation_module"),
    ("src/generation_relation_core/relation_evidence.py", "implementation_module"),
    ("src/generation_relation_core/snapshots.py", "implementation_module"),
    ("src/generation_relation_core/query_engine.py", "implementation_module"),
    ("src/generation_relation_core/implementation_identity.py", "implementation_module"),
    ("protocol/core_v3/core_v3_entities.schema.json", "formal_protocol"),
    ("protocol/core_v3/core_v3_protocol.json", "formal_protocol"),
    ("protocol/core_v3/canonical_serialization_v3.json", "formal_protocol"),
    ("protocol/core_v3/test_vectors/canonical_positive.json", "formal_protocol"),
    ("protocol/core_v3/test_vectors/canonical_negative.json", "formal_protocol"),
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(repo_root: Path, *args: str, binary: bool = False) -> bytes | str:
    try:
        result = subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CoreV3Error("IMPLEMENTATION_IDENTITY_UNAVAILABLE", "GIT_COMMAND_FAILED") from exc
    return result if binary else result.strip()


def _status_paths(repo_root: Path, paths: Iterable[str]) -> list[str]:
    output = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *paths,
    )
    assert isinstance(output, str)
    return [line for line in output.splitlines() if line]


def _index_entries(repo_root: Path, paths: Iterable[str]) -> dict[str, tuple[str, str]]:
    output = _git(repo_root, "ls-files", "--stage", "--", *paths)
    assert isinstance(output, str)
    result: dict[str, tuple[str, str]] = {}
    for line in output.splitlines():
        if not line:
            continue
        metadata, relative = line.split("\t", 1)
        mode, blob_oid, stage = metadata.split()
        if stage != "0" or relative in result:
            raise CoreV3Error(
                "IMPLEMENTATION_IDENTITY_UNAVAILABLE",
                "UNMERGED_OR_DUPLICATE_INDEX_ENTRY",
            )
        result[relative] = (mode, blob_oid)
    return result


def _blob_record(
    repo_root: Path,
    relative: str,
    mode: str,
    blob_oid: str,
    role: str,
) -> dict:
    data = _git(repo_root, "cat-file", "blob", blob_oid, binary=True)
    assert isinstance(data, bytes)
    return {
        "relative_path": relative,
        "artifact_role": role,
        "git_file_mode": mode,
        "size_bytes": len(data),
        "sha256": _sha256(data),
        "git_blob_oid": blob_oid,
    }


def build_implementation_content_identity(repo_root: Path | None = None) -> dict:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    paths = [relative for relative, _ in IMPLEMENTATION_FILES]
    dirty = _status_paths(root, paths)
    if dirty:
        raise CoreV3Error("IMPLEMENTATION_IDENTITY_DIRTY", "|".join(dirty))

    index = _index_entries(root, paths)
    missing = [relative for relative in paths if relative not in index]
    if missing:
        raise CoreV3Error(
            "IMPLEMENTATION_IDENTITY_UNAVAILABLE",
            "MISSING_TRACKED_BLOB:" + "|".join(missing),
        )

    files = []
    for relative, role in IMPLEMENTATION_FILES:
        mode, blob_oid = index[relative]
        files.append(_blob_record(root, relative, mode, blob_oid, role))
    manifest = {
        "identity_scheme": IDENTITY_SCHEME,
        "implementation_scope": paths,
        "file_count": len(files),
        "files": files,
    }
    manifest_sha = _sha256(_canonical(manifest))
    return {
        "implementation_identity_scheme": IDENTITY_SCHEME,
        "implementation_content_manifest": manifest,
        "implementation_content_manifest_sha256": manifest_sha,
        "implementation_hashes": {
            row["relative_path"]: row["sha256"] for row in files
        },
    }


def tracked_blob_records(
    repo_root: Path,
    paths: Iterable[str],
    *,
    artifact_role: str,
) -> list[dict]:
    requested = sorted(paths)
    index = _index_entries(repo_root.resolve(), requested)
    records = []
    for relative in requested:
        entry = index.get(relative)
        if entry is None:
            raise CoreV3Error(
                "IMPLEMENTATION_IDENTITY_UNAVAILABLE",
                f"MISSING_TRACKED_BLOB:{relative}",
            )
        mode, blob_oid = entry
        records.append(
            _blob_record(repo_root.resolve(), relative, mode, blob_oid, artifact_role)
        )
    return records
