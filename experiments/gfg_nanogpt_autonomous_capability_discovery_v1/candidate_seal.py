from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from .common import file_sha256, payload_sha256, write_json


def seal_candidate(
    *,
    submission: Path,
    validation: dict[str, Any],
    participant_repository: Path,
    participant_gfg_id: str,
    task_contract_hash: str,
    time_alignment_hash: str,
    session_attestation_hash: str,
) -> dict[str, Any]:
    if validation["status"] != "PASS":
        raise RuntimeError("CANDIDATE_NOT_COMPLIANT")
    files = []
    for path in sorted(submission.rglob("*")):
        if not path.is_file() or path.name == "candidate_seal.json":
            continue
        name = path.relative_to(submission).as_posix()
        files.append(
            {
                "path": name,
                "byte_count": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    repository_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=participant_repository,
        text=True,
    ).strip()
    material = {
        "candidate_files": files,
        "candidate_validation_sha256": validation["validation_sha256"],
        "participant_gfg_id": participant_gfg_id,
        "participant_repository_commit": repository_commit,
        "schema": "sealed-capability-mechanism-candidate-v1",
        "session_attestation_sha256": session_attestation_hash,
        "task_contract_sha256": task_contract_hash,
        "time_alignment_sha256": time_alignment_hash,
    }
    seal = {
        **material,
        "candidate_seal_sha256": payload_sha256(material),
        "status": "SEALED",
    }
    write_json(submission / "candidate_seal.json", seal)
    for path in submission.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return seal


def verify_candidate_seal(
    *,
    submission: Path,
    seal: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    expected = {
        row["path"]: row for row in seal["candidate_files"]
    }
    actual = {
        path.relative_to(submission).as_posix(): path
        for path in submission.rglob("*")
        if path.is_file() and path.name != "candidate_seal.json"
    }
    if set(actual) != set(expected):
        failures.append("CANDIDATE_FILE_SET_CHANGED")
    for name, row in expected.items():
        path = actual.get(name)
        if path is None:
            continue
        if path.stat().st_size != row["byte_count"]:
            failures.append("CANDIDATE_BYTE_COUNT_CHANGED:" + name)
        if file_sha256(path) != row["sha256"]:
            failures.append("CANDIDATE_FILE_HASH_CHANGED:" + name)
    material = {
        "candidate_seal_sha256": seal["candidate_seal_sha256"],
        "failures": sorted(failures),
        "schema": "candidate-seal-verification-v1",
        "status": "PASS" if not failures else "FAIL",
    }
    material["verification_sha256"] = payload_sha256(material)
    return material
