from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)


def build_archive(
    *,
    formal_root: Path,
    participant_root: Path,
    contract_path: Path,
    selection_path: Path,
    replay_path: Path,
    query_source: Path,
) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    participant_root = participant_root.resolve()
    existing_manifest = participant_root / "archive_manifest.json"
    if existing_manifest.is_file():
        existing = read_json(existing_manifest)
        require(existing["status"] == "PASS", "CST_EXISTING_PARTICIPANT_ARCHIVE_NOT_PASS")
        return existing
    require(not participant_root.exists(), "CST_PARTIAL_PARTICIPANT_ARCHIVE_ALREADY_EXISTS")
    staging = participant_root.with_name(participant_root.name + ".building")
    require(not staging.exists(), "CST_PARTICIPANT_ARCHIVE_STAGING_ALREADY_EXISTS")
    staging.mkdir(parents=True)

    def link_or_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            require(os.path.samefile(source, destination), "CST_PARTICIPANT_HARDLINK_IDENTITY_MISMATCH")
        except OSError:
            shutil.copy2(source, destination)
            require(file_sha256(source) == file_sha256(destination), "CST_PARTICIPANT_COPY_HASH_MISMATCH")

    replay = read_json(replay_path)
    require(replay["status"] == "PASS" and replay["replay_count"] == 13, "CST_ARCHIVE_REPLAY_NOT_PASS")
    entries = []
    total_scan = 0
    total_anchors = 0
    total_historical_comparable = 0
    total_historical_exact = 0
    for entry_directory in sorted(formal_root.glob("entry-*")):
        result_validation = read_json(entry_directory / "result_validation.json")
        graph_validation = read_json(entry_directory / "support_transition_gfg_validation.json")
        scan_receipt = read_json(entry_directory / "scan_receipt.json")
        anchor_receipt = read_json(entry_directory / "anchor_receipt.json")
        require(result_validation["status"] == graph_validation["status"] == "PASS", "CST_ARCHIVE_ENTRY_NOT_PASS")
        require(scan_receipt["status"] == anchor_receipt["status"] == "PASS", "CST_ARCHIVE_ENTRY_RECEIPT_NOT_PASS")
        total_scan += int(result_validation["scan_result_count"])
        total_anchors += int(result_validation["anchor_result_count"])
        total_historical_comparable += int(result_validation["historical_comparable_scan_count"])
        total_historical_exact += int(result_validation["historical_exact_scan_count"])
        participant_entry = staging / entry_directory.name
        participant_entry.mkdir()
        for filename in (
            "support_transition_gfg.sqlite3",
            "support_transition_gfg_manifest.json",
            "support_transition_gfg_validation.json",
        ):
            link_or_copy(entry_directory / filename, participant_entry / filename)
        for tensor_path in sorted((entry_directory / "tensor-objects").glob("*.npy")):
            link_or_copy(tensor_path, participant_entry / "tensor-objects" / tensor_path.name)
        entries.append(
            {
                "anchor_receipt_sha256": anchor_receipt["result_sha256"],
                "entry_id": entry_directory.name,
                "gfg_database_sha256": file_sha256(participant_entry / "support_transition_gfg.sqlite3"),
                "gfg_manifest_sha256": file_sha256(entry_directory / "support_transition_gfg_manifest.json"),
                "gfg_validation_sha256": graph_validation["validation_sha256"],
                "result_validation_sha256": result_validation["validation_sha256"],
                "scan_receipt_sha256": scan_receipt["result_sha256"],
            }
        )
    require(len(entries) == 13, "CST_ARCHIVE_ENTRY_COUNT_INVALID")
    require(total_scan == 1300, "CST_ARCHIVE_SCAN_COUNT_INVALID")
    require(total_anchors == 52, "CST_ARCHIVE_ANCHOR_COUNT_INVALID")
    query_destination = staging / "participant_query.py"
    query_destination.write_bytes(query_source.read_bytes())
    contract_destination = staging / "capture_contract_v1.json"
    contract_destination.write_bytes(contract_path.read_bytes())
    link_or_copy(replay_path, staging / "independent_replay_validation_v1.json")
    readme = (
        "# Support-Transition GFG v1\n\n"
        "This package contains 13 externally generated and independently validated "
        "Support-Transition GFGs. The persistent AI did not select anchors, execute GPU "
        "branches, build facts, validate graphs, or perform independent replay.\n\n"
        "The package is development evidence. It contains 1,300 full-versus-skip scans "
        "and 52 frozen four-branch anchors at h={1,5,20,100}. Historical source outcomes "
        "are adjudication-only and are not transition-model features. Anchor identities "
        "are opaque in this participant package; category mappings, selection rationales "
        "and evaluator-side raw result files are deliberately excluded.\n\n"
        "Use `python3 /support-transition-evidence/participant_query.py --root "
        "/support-transition-evidence list-entries` and then `summary`, "
        "`find-occurrences`, `occurrence`, `node`, `traverse`, or `tensor --mode stats|full`.\n"
    )
    (staging / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    material = {
        "anchor_count": total_anchors,
        "contract_sha256": file_sha256(contract_path),
        "entries": entries,
        "entry_count": len(entries),
        "historical_comparable_scan_count": total_historical_comparable,
        "historical_exact_scan_count": total_historical_exact,
        "independent_replay_sha256": replay["replay_sha256"],
        "participant_query_sha256": file_sha256(query_destination),
        "scan_count": total_scan,
        "schema": "nanogpt-support-transition-archive-v1",
        "selection_sha256": read_json(selection_path)["selection_sha256"],
        "status": "PASS",
        "unseen_training_run_started": False,
        "participant_visibility": {
            "anchor_category_mapping_included": False,
            "evaluator_raw_results_included": False,
            "selection_receipt_included": False,
            "support_transition_gfg_included": True,
        },
    }
    result = {**material, "archive_sha256": payload_sha256(material)}
    write_json(staging / "archive_manifest.json", result)
    root_files = {path.name for path in staging.iterdir() if path.is_file()}
    require(
        root_files
        == {
            "README.md",
            "archive_manifest.json",
            "capture_contract_v1.json",
            "independent_replay_validation_v1.json",
            "participant_query.py",
        },
        "CST_PARTICIPANT_ROOT_FILE_SET_INVALID",
    )
    for entry in entries:
        participant_entry = staging / entry["entry_id"]
        require(
            {path.name for path in participant_entry.iterdir() if path.is_file()}
            == {
                "support_transition_gfg.sqlite3",
                "support_transition_gfg_manifest.json",
                "support_transition_gfg_validation.json",
            },
            "CST_PARTICIPANT_ENTRY_FILE_SET_INVALID",
        )
        require(
            all(path.suffix == ".npy" for path in (participant_entry / "tensor-objects").iterdir()),
            "CST_PARTICIPANT_TENSOR_FILE_SET_INVALID",
        )
    staging.replace(participant_root)
    return result
