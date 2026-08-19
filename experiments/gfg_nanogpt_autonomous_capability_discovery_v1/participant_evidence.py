from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from .common import (
    file_sha256,
    payload_sha256,
    read_json,
    relative_file_manifest,
    write_json,
)


def _copy_without_duplication(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_participant_evidence_bundle(
    *,
    captured_directory: Path,
    bundle_directory: Path,
) -> dict[str, Any]:
    """Expose the complete GFG while excluding checkpoints and receipts.

    The bundle contains the validated SQLite graph, its content-addressed
    tensor objects and validation metadata. It deliberately excludes the
    native checkpoint, RNG state, segment result and any hidden future.
    """

    if bundle_directory.exists():
        raise FileExistsError("PARTICIPANT_EVIDENCE_BUNDLE_EXISTS")
    bundle_directory.mkdir(parents=True)
    required = (
        "participant_gfg.sqlite3",
        "capture_manifest.json",
        "gfg_validation.json",
        "core_representative_validation.json",
    )
    for name in required:
        source = captured_directory / name
        if not source.is_file():
            raise FileNotFoundError("PARTICIPANT_EVIDENCE_MISSING:" + name)
        _copy_without_duplication(source, bundle_directory / name)
    tensor_source = captured_directory / "tensor-objects"
    tensor_destination = bundle_directory / "tensor-objects"
    tensor_destination.mkdir()
    for source in sorted(tensor_source.glob("*.npy")):
        _copy_without_duplication(source, tensor_destination / source.name)
    validation = read_json(bundle_directory / "gfg_validation.json")
    if validation["status"] != "PASS":
        raise ValueError("PARTICIPANT_GFG_NOT_VALIDATED")
    material = {
        "capture_manifest_sha256": file_sha256(
            bundle_directory / "capture_manifest.json"
        ),
        "excluded_native_files": [
            "checkpoint.pt",
            "checkpoint.receipt.json",
            "segment_result.json",
        ],
        "gfg_database_sha256": file_sha256(
            bundle_directory / "participant_gfg.sqlite3"
        ),
        "gfg_validation_sha256": validation["validation_sha256"],
        "participant_safe": True,
        "schema": "participant-safe-training-gfg-bundle-v1",
        "tensor_object_count": len(
            list(tensor_destination.glob("*.npy"))
        ),
    }
    manifest_material = {
        **material,
        "bundle_content_sha256": payload_sha256(
            relative_file_manifest(bundle_directory)
        ),
    }
    manifest = {
        **manifest_material,
        "bundle_manifest_sha256": payload_sha256(manifest_material),
    }
    write_json(bundle_directory / "manifest.json", manifest)
    return {
        **manifest,
        "file_manifest": relative_file_manifest(bundle_directory),
    }
