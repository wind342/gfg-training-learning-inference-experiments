from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)


def _link_or_copy(source: Path, destination: Path, *, allow_copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        require(os.path.samefile(source, destination), "SST_ARCHIVE_HARDLINK_IDENTITY_MISMATCH")
    except OSError:
        require(allow_copy, f"SST_ARCHIVE_CROSS_VOLUME_COPY_NOT_AUTHORIZED:{source}")
        shutil.copy2(source, destination)
        require(file_sha256(source) == file_sha256(destination), "SST_ARCHIVE_COPY_HASH_MISMATCH")


def _junction(logical: Path, physical: Path) -> None:
    logical.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(logical), str(physical)],
            capture_output=True,
            text=True,
            check=False,
        )
        require(completed.returncode == 0, f"SST_ARCHIVE_JUNCTION_FAILED:{completed.stderr or completed.stdout}")
    else:
        logical.symlink_to(physical, target_is_directory=True)


def _graph_files(
    *,
    source: Path,
    destination: Path,
    database_name: str,
    manifest_name: str,
    validation_name: str,
    allow_copy: bool,
) -> dict[str, Any]:
    validation = read_json(source / validation_name)
    require(validation["status"] == "PASS", f"SST_ARCHIVE_GRAPH_NOT_VALIDATED:{source}")
    manifest = read_json(source / manifest_name)
    require(file_sha256(source / database_name) == manifest["database_sha256"], "SST_ARCHIVE_DATABASE_HASH_MISMATCH")
    for filename in (database_name, manifest_name, validation_name):
        _link_or_copy(source / filename, destination / filename, allow_copy=allow_copy)
    tensor_count = 0
    for tensor in sorted((source / "tensor-objects").glob("*.npy")):
        _link_or_copy(tensor, destination / "tensor-objects" / tensor.name, allow_copy=allow_copy)
        tensor_count += 1
    return {
        "database": database_name,
        "database_sha256": manifest["database_sha256"],
        "manifest": manifest_name,
        "manifest_sha256": manifest["manifest_sha256"],
        "validation": validation_name,
        "validation_sha256": validation["validation_sha256"],
        "tensor_payload_count": tensor_count,
    }


def build_stepwise_archive(
    *,
    formal_root: Path,
    branch_root: Path,
    participant_root: Path,
    query_source: Path,
    independent_validation_paths: list[Path],
    machine_receipts: dict[str, Path],
    secondary_root: Path | None = None,
    allow_copy: bool = False,
) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    branch_root = branch_root.resolve()
    participant_root = participant_root.resolve()
    require(not participant_root.exists(), "SST_ARCHIVE_ROOT_ALREADY_EXISTS")
    staging = participant_root.with_name(participant_root.name + ".building")
    require(not staging.exists(), "SST_ARCHIVE_STAGING_ALREADY_EXISTS")
    staging.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    try:
        source_entries = sorted(path for path in formal_root.glob("entry-*") if (path / "stepwise_support_transition_gfg.sqlite3").is_file())
        require(len(source_entries) == 13, "SST_ARCHIVE_MAIN_ENTRY_COUNT_INVALID")
        for source_entry in source_entries:
            entry_id = source_entry.name
            logical_entry = staging / "entries" / entry_id
            source_drive = source_entry.resolve().drive.lower()
            staging_drive = staging.drive.lower()
            if secondary_root is not None and source_drive != staging_drive:
                physical_entry = secondary_root.resolve() / "entries" / entry_id
                require(not physical_entry.exists(), f"SST_ARCHIVE_SECONDARY_ENTRY_ALREADY_EXISTS:{entry_id}")
                physical_entry.mkdir(parents=True)
                _junction(logical_entry, physical_entry)
            else:
                physical_entry = logical_entry
                physical_entry.mkdir(parents=True)
            main = _graph_files(
                source=source_entry,
                destination=physical_entry / "stepwise",
                database_name="stepwise_support_transition_gfg.sqlite3",
                manifest_name="stepwise_support_transition_gfg_manifest.json",
                validation_name="stepwise_support_transition_gfg_validation.json",
                allow_copy=allow_copy,
            )
            branch_source = branch_root / entry_id
            branch = None
            if (branch_source / "stepwise_causal_branch_gfg.sqlite3").is_file():
                branch = _graph_files(
                    source=branch_source,
                    destination=physical_entry / "causal-branch",
                    database_name="stepwise_causal_branch_gfg.sqlite3",
                    manifest_name="stepwise_causal_branch_gfg_manifest.json",
                    validation_name="stepwise_causal_branch_gfg_validation.json",
                    allow_copy=allow_copy,
                )
            entries.append({"entry_id": entry_id, "stepwise": main, "causal_branch": branch})

        independent = []
        for path in independent_validation_paths:
            value = read_json(path)
            require(value["status"] == "PASS", f"SST_ARCHIVE_INDEPENDENT_VALIDATION_NOT_PASS:{path}")
            hash_field = "validation_sha256"
            require(payload_sha256({key: child for key, child in value.items() if key != hash_field}) == value[hash_field], f"SST_ARCHIVE_INDEPENDENT_VALIDATION_HASH_INVALID:{path}")
            destination = staging / "machine-receipts" / path.name
            _link_or_copy(path, destination, allow_copy=True)
            independent.append({"name": path.name, "file_sha256": file_sha256(path), "validation_sha256": value[hash_field]})
        independent_material = {
            "schema": "nanogpt-stepwise-independent-replay-validation-bundle-v1",
            "status": "PASS",
            "validations": independent,
        }
        independent_bundle = {**independent_material, "validation_sha256": payload_sha256(independent_material)}
        write_json(staging / "independent_replay_validation_v1.json", independent_bundle)

        receipt_rows = []
        for name, path in sorted(machine_receipts.items()):
            require(name.endswith(".json") and Path(name).name == name, f"SST_ARCHIVE_RECEIPT_NAME_INVALID:{name}")
            _link_or_copy(path, staging / "machine-receipts" / name, allow_copy=True)
            receipt_rows.append({"name": name, "file_sha256": file_sha256(path)})
        (staging / "participant_query.py").write_bytes(query_source.read_bytes())
        readme = (
            "# Stepwise Support-Transition GFG evidence\n\n"
            "This read-only package contains externally executed and independently validated "
            "stepwise training GFGs and matched four-branch causal GFGs. The persistent AI did "
            "not select windows, execute training, build graphs, validate evidence, or select "
            "independent replay cases.\n\n"
            "Use `python3 /support-transition-evidence/participant_query.py --root "
            "/support-transition-evidence list-entries`. Select `--graph-kind stepwise` or "
            "`--graph-kind causal-branch` for graph queries. Machine receipts are available "
            "through `list-receipts` and `receipt --name ...`.\n"
        )
        (staging / "README.md").write_text(readme, encoding="utf-8", newline="\n")
        material = {
            "schema": "nanogpt-stepwise-support-transition-participant-archive-v1",
            "status": "PASS",
            "entry_count": len(entries),
            "entries": entries,
            "independent_replay_validation_bundle_sha256": independent_bundle["validation_sha256"],
            "machine_receipts": receipt_rows,
            "participant_query_sha256": file_sha256(staging / "participant_query.py"),
            "participant_visibility": {
                "validated_stepwise_gfg": True,
                "validated_four_branch_causal_gfg": True,
                "raw_unvalidated_working_files": False,
                "writable_evidence": False,
            },
        }
        result = {**material, "archive_sha256": payload_sha256(material)}
        write_json(staging / "archive_manifest.json", result)
        staging.replace(participant_root)
        return result
    except Exception:
        # Preserve staging evidence for diagnosis. Never erase a partially built archive.
        raise
