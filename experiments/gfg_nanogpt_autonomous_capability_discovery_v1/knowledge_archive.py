from __future__ import annotations

import os
import re
import shutil
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .common import (
    file_sha256,
    payload_sha256,
    read_json,
    relative_file_manifest,
    require,
    write_json,
)


ARCHIVE_SCHEMA = "nanogpt-gfg-mechanism-discovery-archive-v1"
ARCHIVE_RELATIVE_ROOT = Path(
    "experiments/gfg_nanogpt_autonomous_capability_discovery_v1/"
    "research_archive"
)
LOCAL_GFG_RELATIVE_ROOT = Path(
    "data_private/gfg_nanogpt_mechanism_discovery_archive/gfg"
)

SUBMISSION_EXCLUDED_SUFFIXES = {".pyc"}
ROOT_EVIDENCE_NAMES = {
    "branch_only_difference_audit.json",
    "candidate_seal.json",
    "candidate_validation.json",
    "checkpoint_fork_audit.json",
    "instance_attestation.json",
    "sealed_forecast.json",
}
SESSION_EVIDENCE_NAMES = {
    "orientation_gate_receipt.json",
    "orientation_receipt.json",
    "session_attestation.json",
}
EVALUATION_NAMES = {
    "branch_only_difference_audit.json",
    "causal_validation.json",
    "checkpoint_fork_audit.json",
    "forecast_validation.json",
    "intervention_runtime_receipt.json",
    "runtime_repair_attestation.json",
}


def _safe_component(value: str) -> str:
    material = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    require(bool(material), "ARCHIVE_EMPTY_PATH_COMPONENT")
    return material


def _entry_id(run_name: str, instance_name: str) -> str:
    identity = payload_sha256(
        {"instance_name": instance_name, "run_name": run_name}
    )
    return f"entry-{identity[:20]}"


def _copy_immutable(source: Path, destination: Path) -> None:
    require(source.is_file(), f"ARCHIVE_SOURCE_FILE_MISSING:{source}")
    if destination.exists():
        require(destination.is_file(), f"ARCHIVE_DESTINATION_NOT_FILE:{destination}")
        require(
            file_sha256(source) == file_sha256(destination),
            f"ARCHIVE_IMMUTABLE_FILE_CONFLICT:{destination}",
        )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree_files(
    source: Path,
    destination: Path,
    files: Iterable[Path],
) -> None:
    for path in sorted(files):
        _copy_immutable(path, destination / path.relative_to(source))


def _submission_files(submission: Path) -> list[Path]:
    return [
        path
        for path in submission.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in SUBMISSION_EXCLUDED_SUFFIXES
        and "__pycache__" not in path.parts
    ]


def _participant_bundle_identity(gfg_root: Path) -> tuple[str, dict[str, Any]]:
    bundle_manifest_path = gfg_root / "manifest.json"
    capture_manifest_path = gfg_root / "capture_manifest.json"
    require(bundle_manifest_path.is_file(), "ARCHIVE_GFG_BUNDLE_MANIFEST_MISSING")
    require(capture_manifest_path.is_file(), "ARCHIVE_GFG_CAPTURE_MANIFEST_MISSING")
    bundle = read_json(bundle_manifest_path)
    capture = read_json(capture_manifest_path)
    identity = bundle.get("bundle_manifest_sha256")
    require(
        isinstance(identity, str) and len(identity) == 64,
        "ARCHIVE_GFG_BUNDLE_ID_INVALID",
    )
    return identity, {"bundle": bundle, "capture": capture}


def _gfg_manifest(gfg_root: Path) -> dict[str, Any]:
    bundle_id, source = _participant_bundle_identity(gfg_root)
    capture = source["capture"]
    bundle = source["bundle"]
    validation_path = gfg_root / "gfg_validation.json"
    validation = read_json(validation_path)
    require(validation.get("status") == "PASS", "ARCHIVE_GFG_NOT_VALIDATED")

    tensor_root = gfg_root / "tensor-objects"
    tensor_files = sorted(path for path in tensor_root.rglob("*") if path.is_file())
    tensor_names_and_sizes = [
        {
            "path": path.relative_to(gfg_root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in tensor_files
    ]
    all_files = sorted(path for path in gfg_root.rglob("*") if path.is_file())
    metadata_files = {
        path.relative_to(gfg_root).as_posix(): {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in all_files
        if tensor_root not in path.parents
        and path.name != capture.get("database", "participant_gfg.sqlite3")
    }
    database_name = capture.get("database", "participant_gfg.sqlite3")
    database_path = gfg_root / database_name
    require(database_path.is_file(), "ARCHIVE_GFG_DATABASE_MISSING")
    require(
        capture.get("database_sha256") == bundle.get("gfg_database_sha256"),
        "ARCHIVE_GFG_DATABASE_COMMITMENT_CONFLICT",
    )
    require(
        len(tensor_files) == int(bundle.get("tensor_object_count", -1)),
        "ARCHIVE_GFG_TENSOR_COUNT_CONFLICT",
    )

    material = {
        "bundle_content_sha256": bundle.get("bundle_content_sha256"),
        "capture_counts": capture.get("counts"),
        "capture_manifest_id": capture.get("manifest_sha256"),
        "capture_manifest_sha256": file_sha256(
            gfg_root / "capture_manifest.json"
        ),
        "database": database_name,
        "database_sha256": capture.get("database_sha256"),
        "database_size_bytes": database_path.stat().st_size,
        "gfg_bundle_id": bundle_id,
        "gfg_validation_sha256": validation.get("validation_sha256"),
        "gfg_validation_status": validation.get("status"),
        "local_repository_path": (
            LOCAL_GFG_RELATIVE_ROOT / bundle_id
        ).as_posix(),
        "logical_file_count": len(all_files),
        "logical_size_bytes": sum(path.stat().st_size for path in all_files),
        "metadata_files": metadata_files,
        "participant_safe": bundle.get("participant_safe"),
        "schema": "archived-participant-gfg-commitment-v1",
        "tensor_object_count": len(tensor_files),
        "tensor_path_size_commitment": payload_sha256(tensor_names_and_sizes),
    }
    material["archive_manifest_sha256"] = payload_sha256(material)
    return material


def _preserve_gfg_locally(
    *,
    source: Path,
    repository: Path,
    manifest: dict[str, Any],
) -> dict[str, int]:
    destination = repository / Path(manifest["local_repository_path"])
    stats = {"copied": 0, "existing": 0, "hardlinked": 0}
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        target = destination / path.relative_to(source)
        if target.exists():
            require(target.is_file(), f"ARCHIVE_LOCAL_GFG_CONFLICT:{target}")
            require(
                target.stat().st_size == path.stat().st_size,
                f"ARCHIVE_LOCAL_GFG_SIZE_CONFLICT:{target}",
            )
            stats["existing"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
            stats["hardlinked"] += 1
        except FileExistsError:
            require(
                target.is_file()
                and target.stat().st_size == path.stat().st_size,
                f"ARCHIVE_LOCAL_GFG_RACE_CONFLICT:{target}",
            )
            stats["existing"] += 1
        except OSError:
            try:
                shutil.copy2(path, target)
                stats["copied"] += 1
            except FileExistsError:
                require(
                    target.is_file()
                    and target.stat().st_size == path.stat().st_size,
                    f"ARCHIVE_LOCAL_GFG_RACE_CONFLICT:{target}",
                )
                stats["existing"] += 1

    require(
        sum(stats.values()) == manifest["logical_file_count"],
        "ARCHIVE_LOCAL_GFG_FILE_COUNT_MISMATCH",
    )
    database = destination / manifest["database"]
    require(
        database.stat().st_size == manifest["database_size_bytes"],
        "ARCHIVE_LOCAL_GFG_DATABASE_SIZE_MISMATCH",
    )
    return stats


def _evaluation_sources(instance_root: Path) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    if any((instance_root / name).is_file() for name in EVALUATION_NAMES):
        sources.append(("original", instance_root))
    for path in sorted(instance_root.glob("runtime-repair-*")):
        if path.is_dir() and any((path / name).is_file() for name in EVALUATION_NAMES):
            sources.append((path.name, path))
    return sources


def _archive_evaluations(
    *,
    instance_root: Path,
    entry_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for label, source in _evaluation_sources(instance_root):
        files = [source / name for name in sorted(EVALUATION_NAMES) if (source / name).is_file()]
        manifest = {
            path.name: {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        }
        evaluation_id = payload_sha256(manifest)
        directory = f"eval-{evaluation_id[:20]}"
        destination = entry_root / "evaluations" / directory
        for path in files:
            _copy_immutable(path, destination / path.name)
        forecast = (
            read_json(source / "forecast_validation.json")
            if (source / "forecast_validation.json").is_file()
            else None
        )
        causal = (
            read_json(source / "causal_validation.json")
            if (source / "causal_validation.json").is_file()
            else None
        )
        rows.append(
            {
                "causal_status": causal.get("status") if causal else None,
                "directory": directory,
                "evaluation_id": evaluation_id,
                "files": manifest,
                "forecast_status": forecast.get("status") if forecast else None,
                "label": label,
            }
        )
    return rows


def archive_instance(
    *,
    repository: Path,
    instance_root: Path,
    run_name: str,
) -> dict[str, Any]:
    repository = repository.resolve()
    instance_root = instance_root.resolve()
    submission = instance_root / "ai-session" / "result" / "submission"
    gfg_root = instance_root / "discovery-participant-gfg"
    require((submission / "discovery_report.md").is_file(), "ARCHIVE_REPORT_MISSING")
    require(gfg_root.is_dir(), "ARCHIVE_PARTICIPANT_GFG_MISSING")

    _safe_component(run_name)
    _safe_component(instance_root.name)
    entry_id = _entry_id(run_name, instance_root.name)
    archive_root = repository / ARCHIVE_RELATIVE_ROOT
    entry_root = archive_root / "entries" / entry_id

    submission_files = _submission_files(submission)
    _copy_tree_files(submission, entry_root / "submission", submission_files)
    submission_manifest = {
        path.relative_to(submission).as_posix(): {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(submission_files)
    }
    write_json(entry_root / "submission_manifest.json", submission_manifest)

    participant_repository = instance_root / "participant-repository"
    if participant_repository.is_dir():
        contract_files = [
            path for path in participant_repository.iterdir() if path.is_file()
        ]
        _copy_tree_files(
            participant_repository,
            entry_root / "contract",
            contract_files,
        )

    for name in sorted(ROOT_EVIDENCE_NAMES):
        source = instance_root / name
        if source.is_file():
            _copy_immutable(source, entry_root / "evidence" / name)
    session_root = instance_root / "ai-session"
    for name in sorted(SESSION_EVIDENCE_NAMES):
        source = session_root / name
        if source.is_file():
            _copy_immutable(source, entry_root / "evidence" / name)

    gfg_manifest = _gfg_manifest(gfg_root)
    local_stats = _preserve_gfg_locally(
        source=gfg_root,
        repository=repository,
        manifest=gfg_manifest,
    )
    gfg_manifest_path = entry_root / "gfg_manifest.json"
    if gfg_manifest_path.exists():
        require(
            read_json(gfg_manifest_path) == gfg_manifest,
            "ARCHIVE_IMMUTABLE_GFG_MANIFEST_CONFLICT",
        )
    else:
        write_json(gfg_manifest_path, gfg_manifest)

    evaluations = _archive_evaluations(
        instance_root=instance_root,
        entry_root=entry_root,
    )
    session_attestation_path = session_root / "session_attestation.json"
    session_attestation = (
        read_json(session_attestation_path)
        if session_attestation_path.is_file()
        else {}
    )
    candidate_seal_path = submission / "candidate_seal.json"
    if not candidate_seal_path.is_file():
        candidate_seal_path = instance_root / "candidate_seal.json"
    sealed = candidate_seal_path.is_file()
    preferred = evaluations[-1] if evaluations else None
    record = {
        "ai_model": session_attestation.get("model"),
        "candidate_sealed": sealed,
        "completed_at": session_attestation.get("completed_at"),
        "entry_id": entry_id,
        "entry_status": (
            "SEALED"
            if sealed
            else (
                "UNSEALED_PLATFORM_ABORTED_SUPERSEDED"
                if "platform-aborted" in instance_root.name
                else "UNSEALED"
            )
        ),
        "evaluations": evaluations,
        "formal_work_seconds": session_attestation.get("formal_work_seconds"),
        "gfg_archive_manifest_sha256": gfg_manifest[
            "archive_manifest_sha256"
        ],
        "gfg_bundle_id": gfg_manifest["gfg_bundle_id"],
        "gfg_validation_status": gfg_manifest["gfg_validation_status"],
        "instance_id": instance_root.name,
        "preferred_evaluation_id": (
            preferred["evaluation_id"] if preferred else None
        ),
        "run_name": run_name,
        "schema": "nanogpt-gfg-mechanism-discovery-archive-entry-v1",
        "submission_manifest_sha256": payload_sha256(submission_manifest),
    }
    write_json(entry_root / "entry_record.json", record)
    return {"entry": record, "local_gfg_storage": local_stats}


def _entry_records(repository: Path) -> list[dict[str, Any]]:
    entries_root = repository / ARCHIVE_RELATIVE_ROOT / "entries"
    if not entries_root.is_dir():
        return []
    return [
        read_json(path)
        for path in sorted(entries_root.glob("*/entry_record.json"))
    ]


def rebuild_archive_index(repository: Path) -> dict[str, Any]:
    entries = sorted(_entry_records(repository), key=lambda row: row["entry_id"])
    gfg_ids = sorted({row["gfg_bundle_id"] for row in entries})
    material = {
        "entry_count": len(entries),
        "entries": entries,
        "gfg_bundle_count": len(gfg_ids),
        "gfg_bundle_ids": gfg_ids,
        "schema": ARCHIVE_SCHEMA,
        "sealed_candidate_count": sum(
            bool(row["candidate_sealed"]) for row in entries
        ),
    }
    material["archive_index_sha256"] = payload_sha256(material)
    write_json(
        repository / ARCHIVE_RELATIVE_ROOT / "archive_index.json",
        material,
    )
    return material


def archive_runs_root(*, repository: Path, runs_root: Path) -> dict[str, Any]:
    reports = sorted(
        path
        for path in runs_root.rglob("discovery_report.md")
        if "gfg-nanogpt" in path.as_posix().lower()
        and path.parent.name == "submission"
    )
    require(bool(reports), "ARCHIVE_NO_DISCOVERY_REPORTS_FOUND")
    results = []
    for report in reports:
        submission = report.parent
        instance_root = submission.parents[2]
        relative = instance_root.relative_to(runs_root)
        require(len(relative.parts) >= 3, "ARCHIVE_INSTANCE_LAYOUT_INVALID")
        results.append(
            archive_instance(
                repository=repository,
                instance_root=instance_root,
                run_name=relative.parts[0],
            )
        )
    index = rebuild_archive_index(repository)
    return {
        "archive_index_sha256": index["archive_index_sha256"],
        "entry_count": len(results),
        "local_gfg_storage": {
            key: sum(result["local_gfg_storage"][key] for result in results)
            for key in ("copied", "existing", "hardlinked")
        },
        "status": "PASS",
    }


def verify_archive(*, repository: Path, deep: bool = False) -> dict[str, Any]:
    index_path = repository / ARCHIVE_RELATIVE_ROOT / "archive_index.json"
    require(index_path.is_file(), "ARCHIVE_INDEX_MISSING")
    index = read_json(index_path)
    expected_index_sha256 = index.pop("archive_index_sha256")
    require(
        payload_sha256(index) == expected_index_sha256,
        "ARCHIVE_INDEX_HASH_MISMATCH",
    )
    index["archive_index_sha256"] = expected_index_sha256

    checked_files = 0
    checked_gfg_files = 0
    for entry in index["entries"]:
        entry_root = (
            repository
            / ARCHIVE_RELATIVE_ROOT
            / "entries"
            / entry["entry_id"]
        )
        submission_manifest = read_json(entry_root / "submission_manifest.json")
        require(
            payload_sha256(submission_manifest)
            == entry["submission_manifest_sha256"],
            f"ARCHIVE_SUBMISSION_MANIFEST_HASH_MISMATCH:{entry['entry_id']}",
        )
        for relative, expected in submission_manifest.items():
            path = entry_root / "submission" / relative
            require(path.is_file(), f"ARCHIVE_SUBMISSION_FILE_MISSING:{path}")
            require(
                file_sha256(path) == expected["sha256"],
                f"ARCHIVE_SUBMISSION_FILE_HASH_MISMATCH:{path}",
            )
            checked_files += 1

        for evaluation in entry["evaluations"]:
            evaluation_root = (
                entry_root / "evaluations" / evaluation["directory"]
            )
            require(
                payload_sha256(evaluation["files"])
                == evaluation["evaluation_id"],
                f"ARCHIVE_EVALUATION_MANIFEST_HASH_MISMATCH:{evaluation_root}",
            )
            for relative, expected in evaluation["files"].items():
                path = evaluation_root / relative
                require(path.is_file(), f"ARCHIVE_EVALUATION_FILE_MISSING:{path}")
                require(
                    path.stat().st_size == expected["size_bytes"]
                    and file_sha256(path) == expected["sha256"],
                    f"ARCHIVE_EVALUATION_FILE_HASH_MISMATCH:{path}",
                )
                checked_files += 1

        gfg_manifest = read_json(entry_root / "gfg_manifest.json")
        archive_sha = gfg_manifest.pop("archive_manifest_sha256")
        require(
            payload_sha256(gfg_manifest) == archive_sha,
            f"ARCHIVE_GFG_MANIFEST_HASH_MISMATCH:{entry['entry_id']}",
        )
        gfg_manifest["archive_manifest_sha256"] = archive_sha
        gfg_root = repository / Path(gfg_manifest["local_repository_path"])
        require(gfg_root.is_dir(), f"ARCHIVE_LOCAL_GFG_MISSING:{gfg_root}")
        database = gfg_root / gfg_manifest["database"]
        require(database.is_file(), f"ARCHIVE_LOCAL_GFG_DATABASE_MISSING:{database}")
        require(
            database.stat().st_size == gfg_manifest["database_size_bytes"],
            f"ARCHIVE_LOCAL_GFG_DATABASE_SIZE_MISMATCH:{database}",
        )
        if deep:
            import numpy as np

            require(
                file_sha256(database) == gfg_manifest["database_sha256"],
                f"ARCHIVE_LOCAL_GFG_DATABASE_HASH_MISMATCH:{database}",
            )
            tensor_files = sorted(
                path
                for path in (gfg_root / "tensor-objects").rglob("*")
                if path.is_file()
            )
            for path in tensor_files:
                array = np.load(path, allow_pickle=False, mmap_mode="r")
                content_sha256 = hashlib.sha256(
                    array.tobytes(order="C")
                ).hexdigest()
                require(
                    path.stem == content_sha256,
                    f"ARCHIVE_TENSOR_CONTENT_ADDRESS_MISMATCH:{path}",
                )
            checked_gfg_files += 1 + len(tensor_files)

    return {
        "checked_gfg_files": checked_gfg_files,
        "checked_submission_files": checked_files,
        "deep": deep,
        "entry_count": len(index["entries"]),
        "status": "PASS",
    }
