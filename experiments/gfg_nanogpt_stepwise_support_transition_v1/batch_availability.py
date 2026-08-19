from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import TrainingGFG
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import objects_for_stage


def audit_batch_identity_availability(
    *,
    source_root: Path,
    source_archive_manifest_path: Path,
    selection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    archive = read_json(source_archive_manifest_path)
    selection = read_json(selection_path)
    bundles = {str(row["entry_id"]): str(row["gfg_bundle_id"]) for row in archive["support_bundles"]}
    steps_by_entry: dict[str, set[int]] = {}
    for window in selection["windows"]:
        entry_id = str(window["entry_id"])
        start = int(window.get("restore_optimizer_step", window["start_optimizer_step"]))
        end = int(window.get("capture_end_optimizer_step", window["end_optimizer_step"]))
        steps_by_entry.setdefault(entry_id, set()).update(range(start, end))
    results: list[dict[str, Any]] = []
    for entry_id, required_steps in sorted(steps_by_entry.items()):
        require(entry_id in bundles, f"SST_BATCH_AUDIT_BUNDLE_MISSING:{entry_id}")
        bundle = source_root / bundles[entry_id]
        graph = TrainingGFG(bundle / "participant_gfg.sqlite3")
        counts = {
            "training_batch_inputs": {"present": 0, "absent": 0, "multiple": 0},
            "training_batch_targets": {"present": 0, "absent": 0, "multiple": 0},
            "batch_selection_order": {"present": 0, "absent": 0, "multiple": 0},
        }
        try:
            for optimizer_step in sorted(required_steps):
                rows = objects_for_stage(graph, optimizer_step, "before_batch")
                for role in counts:
                    match_count = sum(row["role"] == role for row in rows)
                    bucket = "absent" if match_count == 0 else "present" if match_count == 1 else "multiple"
                    counts[role][bucket] += 1
        finally:
            graph.close()
        require(counts["training_batch_inputs"]["present"] == len(required_steps), f"SST_BATCH_AUDIT_INPUT_COVERAGE_FAILED:{entry_id}")
        require(counts["training_batch_targets"]["present"] == len(required_steps), f"SST_BATCH_AUDIT_TARGET_COVERAGE_FAILED:{entry_id}")
        require(counts["batch_selection_order"]["multiple"] == 0, f"SST_BATCH_AUDIT_ORDER_MULTIPLE:{entry_id}")
        results.append(
            {
                "entry_id": entry_id,
                "gfg_bundle_id": bundles[entry_id],
                "required_training_opportunity_count": len(required_steps),
                "minimum_required_optimizer_step": min(required_steps),
                "maximum_required_optimizer_step": max(required_steps),
                "role_counts": counts,
                "selection_order_policy": (
                    "CAPTURED_EXACTLY"
                    if counts["batch_selection_order"]["absent"] == 0
                    else "EXPLICIT_DISPOSITION_REQUIRED"
                ),
            }
        )
    material = {
        "schema": "nanogpt-stepwise-batch-identity-availability-audit-v1",
        "status": "PASS",
        "selection_sha256": selection["selection_sha256"],
        "source_archive_sha256": archive["archive_sha256"],
        "entry_count": len(results),
        "entries_with_exact_selection_order": sum(
            row["selection_order_policy"] == "CAPTURED_EXACTLY" for row in results
        ),
        "entries_requiring_explicit_disposition": sum(
            row["selection_order_policy"] == "EXPLICIT_DISPOSITION_REQUIRED" for row in results
        ),
        "training_replay_uses_captured_ordered_input_and_target_tensors": True,
        "selection_order_reconstruction_or_guess_used": False,
        "results": results,
    }
    result = {**material, "audit_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result
