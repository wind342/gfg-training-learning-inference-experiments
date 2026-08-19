from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)


def _verify_hash(value: dict[str, Any], field: str, error: str) -> None:
    material = {key: child for key, child in value.items() if key != field}
    require(payload_sha256(material) == value[field], error)


def _tensor_references(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"locator", "file_sha256"} <= set(value) and str(value["locator"]).startswith("tensor-objects/"):
            yield value
        for child in value.values():
            yield from _tensor_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _tensor_references(child)


def _verify_payload_files(entry_root: Path, value: dict[str, Any], prefix: str) -> int:
    count = 0
    seen: set[tuple[str, str]] = set()
    for reference in _tensor_references(value):
        key = (str(reference["locator"]), str(reference["file_sha256"]))
        if key in seen:
            continue
        seen.add(key)
        path = entry_root / str(reference["locator"])
        require(path.is_file(), f"{prefix}_PAYLOAD_MISSING:{path}")
        require(file_sha256(path) == reference["file_sha256"], f"{prefix}_PAYLOAD_HASH_MISMATCH:{path}")
        count += 1
    return count


def _exact_window_record(
    parent_path: Path,
    replay_path: Path,
    *,
    parent_entry_root: Path,
    replay_entry_root: Path,
    hash_field: str,
    error_prefix: str,
) -> tuple[dict[str, Any], int]:
    parent = read_json(parent_path)
    replay = read_json(replay_path)
    _verify_hash(parent, hash_field, f"{error_prefix}_PARENT_RECORD_HASH_INVALID:{parent_path}")
    _verify_hash(replay, hash_field, f"{error_prefix}_REPLAY_RECORD_HASH_INVALID:{replay_path}")
    require(parent == replay, f"{error_prefix}_REPLAY_DRIFT:{replay_path}")
    parent_count = _verify_payload_files(parent_entry_root, parent, f"{error_prefix}_PARENT")
    replay_count = _verify_payload_files(replay_entry_root, replay, f"{error_prefix}_REPLAY")
    require(parent_count == replay_count, f"{error_prefix}_PAYLOAD_COUNT_MISMATCH")
    return replay, replay_count


def validate_independent_window_replay(
    *,
    parent_root: Path,
    replay_root: Path,
    replay_selection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    selection = read_json(replay_selection_path)
    selection_material = {key: value for key, value in selection.items() if key != "selection_sha256"}
    require(payload_sha256(selection_material) == selection["selection_sha256"], "SST_INDEPENDENT_SELECTION_HASH_INVALID")
    exact_window_receipts = 0
    exact_state_records = 0
    exact_transition_records = 0
    exact_probe_records = 0
    verified_payload_count = 0
    per_entry: dict[str, list[dict[str, str]]] = {}
    for window in selection["windows"]:
        entry_id = str(window["entry_id"])
        window_id = str(window["window_id"])
        parent_window = parent_root / entry_id / "windows" / window_id
        replay_window = replay_root / entry_id / "windows" / window_id
        parent_entry_root = parent_root / entry_id
        replay_entry_root = replay_root / entry_id
        receipt, count = _exact_window_record(
            parent_window / "window_receipt.json",
            replay_window / "window_receipt.json",
            parent_entry_root=parent_entry_root,
            replay_entry_root=replay_entry_root,
            hash_field="result_sha256",
            error_prefix="SST_INDEPENDENT_WINDOW_RECEIPT",
        )
        verified_payload_count += count
        exact_window_receipts += 1
        for state in receipt["completed_states"]:
            step = int(state["optimizer_step"])
            state_record, count = _exact_window_record(
                parent_window / "states" / f"step-{step:05d}.json",
                replay_window / "states" / f"step-{step:05d}.json",
                parent_entry_root=parent_entry_root,
                replay_entry_root=replay_entry_root,
                hash_field="result_sha256",
                error_prefix="SST_INDEPENDENT_STATE",
            )
            verified_payload_count += count
            exact_state_records += 1
            probe_id = str(state_record["state"]["state_id"])
            probe_contract_id = str(state["probe_contract_id"])
            _probe, count = _exact_window_record(
                parent_root / entry_id / "probe-observations" / probe_contract_id / f"{probe_id}.json",
                replay_root / entry_id / "probe-observations" / probe_contract_id / f"{probe_id}.json",
                parent_entry_root=parent_entry_root,
                replay_entry_root=replay_entry_root,
                hash_field="result_sha256",
                error_prefix="SST_INDEPENDENT_PROBE",
            )
            verified_payload_count += count
            exact_probe_records += 1
        for transition in receipt["completed_transitions"]:
            step = int(transition["optimizer_step"])
            _transition, count = _exact_window_record(
                parent_window / "transitions" / f"step-{step:05d}-to-{step + 1:05d}.json",
                replay_window / "transitions" / f"step-{step:05d}-to-{step + 1:05d}.json",
                parent_entry_root=parent_entry_root,
                replay_entry_root=replay_entry_root,
                hash_field="result_sha256",
                error_prefix="SST_INDEPENDENT_TRANSITION",
            )
            verified_payload_count += count
            exact_transition_records += 1
        per_entry.setdefault(entry_id, []).append(
            {"window_id": window_id, "window_receipt_sha256": str(receipt["result_sha256"])}
        )

    for entry_id, windows in per_entry.items():
        replay_receipt = read_json(replay_root / entry_id / "entry_receipt.json")
        _verify_hash(replay_receipt, "result_sha256", f"SST_INDEPENDENT_ENTRY_RECEIPT_HASH_INVALID:{entry_id}")
        require(replay_receipt["selection_sha256"] == selection["selection_sha256"], f"SST_INDEPENDENT_ENTRY_SELECTION_MISMATCH:{entry_id}")
        expected = {row["window_id"]: row["window_receipt_sha256"] for row in windows}
        actual = {row["window_id"]: row["result_sha256"] for row in replay_receipt["completed_windows"]}
        require(actual == expected, f"SST_INDEPENDENT_ENTRY_WINDOW_COVERAGE_MISMATCH:{entry_id}")

    material = {
        "schema": "nanogpt-stepwise-independent-window-replay-validation-v1",
        "status": "PASS",
        "parent_root": str(parent_root.resolve()),
        "replay_root": str(replay_root.resolve()),
        "replay_selection_sha256": selection["selection_sha256"],
        "entry_count": len(per_entry),
        "exact_window_receipt_count": exact_window_receipts,
        "exact_state_record_count": exact_state_records,
        "exact_transition_record_count": exact_transition_records,
        "exact_probe_record_count": exact_probe_records,
        "verified_content_addressed_payload_reference_count": verified_payload_count,
        "native_step_replay": "EXACT",
        "probe_replay": "EXACT",
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result


def validate_independent_phase_replay(
    *,
    parent_root: Path,
    replay_root: Path,
    replay_selection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    selection = read_json(replay_selection_path)
    exact_phase_state_count = 0
    exact_phase_receipt_count = 0
    verified_payload_count = 0
    for window in selection["windows"]:
        entry_id = str(window["entry_id"])
        window_id = str(window["window_id"])
        relative = Path("derived") / "support-phase-finite-difference-v1" / window_id
        parent_phase = parent_root / entry_id / relative
        replay_phase = replay_root / entry_id / relative
        parent_receipt = read_json(parent_phase / "phase_window_receipt.json")
        replay_receipt = read_json(replay_phase / "phase_window_receipt.json")
        _verify_hash(parent_receipt, "receipt_sha256", "SST_INDEPENDENT_PHASE_PARENT_RECEIPT_HASH_INVALID")
        _verify_hash(replay_receipt, "receipt_sha256", "SST_INDEPENDENT_PHASE_REPLAY_RECEIPT_HASH_INVALID")
        require(parent_receipt == replay_receipt, f"SST_INDEPENDENT_PHASE_RECEIPT_DRIFT:{window_id}")
        exact_phase_receipt_count += 1
        for record in replay_receipt["records"]:
            step = int(record["optimizer_step"])
            parent_record = read_json(parent_phase / "states" / f"step-{step:05d}.json")
            replay_record = read_json(replay_phase / "states" / f"step-{step:05d}.json")
            _verify_hash(parent_record, "phase_state_sha256", "SST_INDEPENDENT_PHASE_PARENT_STATE_HASH_INVALID")
            _verify_hash(replay_record, "phase_state_sha256", "SST_INDEPENDENT_PHASE_REPLAY_STATE_HASH_INVALID")
            require(parent_record == replay_record, f"SST_INDEPENDENT_PHASE_STATE_DRIFT:{window_id}:{step}")
            parent_count = _verify_payload_files(parent_root / entry_id, parent_record, "SST_INDEPENDENT_PHASE_PARENT")
            replay_count = _verify_payload_files(replay_root / entry_id, replay_record, "SST_INDEPENDENT_PHASE_REPLAY")
            require(parent_count == replay_count, "SST_INDEPENDENT_PHASE_PAYLOAD_COUNT_MISMATCH")
            verified_payload_count += replay_count
            exact_phase_state_count += 1
    material = {
        "schema": "nanogpt-stepwise-independent-phase-replay-validation-v1",
        "status": "PASS",
        "parent_root": str(parent_root.resolve()),
        "replay_root": str(replay_root.resolve()),
        "replay_selection_sha256": selection["selection_sha256"],
        "exact_phase_window_receipt_count": exact_phase_receipt_count,
        "exact_phase_state_count": exact_phase_state_count,
        "verified_content_addressed_payload_reference_count": verified_payload_count,
        "finite_difference_replay": "EXACT",
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result


def validate_independent_branch_replay(
    *,
    parent_branch_root: Path,
    replay_branch_root: Path,
    replay_selection_path: Path,
    replay_branch_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    selection = read_json(replay_selection_path)
    audit = read_json(replay_branch_audit_path)
    _verify_hash(audit, "audit_sha256", "SST_INDEPENDENT_BRANCH_AUDIT_HASH_INVALID")
    windows = {str(row["window_id"]): row for row in selection["windows"]}
    expected_requests = {
        (
            str(windows[str(row[window_key])]["entry_id"]),
            str(row[window_key]),
            int(relative_h),
        )
        for row in audit["results"]
        for window_key in ("decline_window_id", "stable_window_id")
        for relative_h in row["causal_seed_relative_h"]
    }
    replay_all = read_json(replay_branch_root / "all_branches_receipt.json")
    parent_all = read_json(parent_branch_root / "all_branches_receipt.json")
    _verify_hash(replay_all, "result_sha256", "SST_INDEPENDENT_BRANCH_REPLAY_ALL_RECEIPT_HASH_INVALID")
    _verify_hash(parent_all, "result_sha256", "SST_INDEPENDENT_BRANCH_PARENT_ALL_RECEIPT_HASH_INVALID")
    require(replay_all["divergence_audit_sha256"] == audit["audit_sha256"], "SST_INDEPENDENT_BRANCH_AUDIT_REFERENCE_MISMATCH")
    require(int(replay_all["unique_seed_count"]) == len(expected_requests), "SST_INDEPENDENT_BRANCH_REQUEST_COUNT_MISMATCH")
    require(
        replay_all["branch_profile_sha256"] == parent_all["branch_profile_sha256"]
        and replay_all["main_protocol_sha256"] == parent_all["main_protocol_sha256"]
        and replay_all["component_registry_sha256"] == parent_all["component_registry_sha256"]
        and replay_all["probe_contract_sha256"] == parent_all["probe_contract_sha256"],
        "SST_INDEPENDENT_BRANCH_CONTRACT_DRIFT",
    )

    parent_seed_paths = {
        path.name: path
        for path in parent_branch_root.glob("entry-*/branch-seeds/branch-seed-*")
        if path.is_dir()
    }
    replay_seed_paths = {
        path.name: path
        for path in replay_branch_root.glob("entry-*/branch-seeds/branch-seed-*")
        if path.is_dir()
    }
    receipt_seed_ids = {str(row["seed_id"]) for row in replay_all["completed"]}
    require(set(replay_seed_paths) == receipt_seed_ids, "SST_INDEPENDENT_BRANCH_REPLAY_SEED_COVERAGE_MISMATCH")
    require(receipt_seed_ids <= set(parent_seed_paths), "SST_INDEPENDENT_BRANCH_PARENT_SEED_MISSING")

    exact_seed_count = 0
    exact_continuation_count = 0
    exact_horizon_state_count = 0
    exact_effect_count = 0
    exact_probe_count = 0
    verified_payload_count = 0
    entries: set[str] = set()
    for seed_id in sorted(receipt_seed_ids):
        parent_seed_root = parent_seed_paths[seed_id]
        replay_seed_root = replay_seed_paths[seed_id]
        parent_entry_root = parent_seed_root.parents[1]
        replay_entry_root = replay_seed_root.parents[1]
        seed, count = _exact_window_record(
            parent_seed_root / "seed_result.json",
            replay_seed_root / "seed_result.json",
            parent_entry_root=parent_entry_root,
            replay_entry_root=replay_entry_root,
            hash_field="result_sha256",
            error_prefix="SST_INDEPENDENT_BRANCH_SEED",
        )
        verified_payload_count += count
        receipt, count = _exact_window_record(
            parent_seed_root / "branch_receipt.json",
            replay_seed_root / "branch_receipt.json",
            parent_entry_root=parent_entry_root,
            replay_entry_root=replay_entry_root,
            hash_field="result_sha256",
            error_prefix="SST_INDEPENDENT_BRANCH_RECEIPT",
        )
        verified_payload_count += count
        entries.add(str(seed["entry_id"]))
        request = (str(seed["entry_id"]), str(seed["window_id"]), int(seed["relative_h"]))
        require(request in expected_requests, f"SST_INDEPENDENT_BRANCH_UNPLANNED_SEED:{seed_id}")
        for row in receipt["continuation_results"]:
            step = int(row["physical_optimizer_step"])
            _continuation, count = _exact_window_record(
                parent_seed_root / "continuations" / f"step-{step:05d}-to-{step + 1:05d}.json",
                replay_seed_root / "continuations" / f"step-{step:05d}-to-{step + 1:05d}.json",
                parent_entry_root=parent_entry_root,
                replay_entry_root=replay_entry_root,
                hash_field="result_sha256",
                error_prefix="SST_INDEPENDENT_BRANCH_CONTINUATION",
            )
            verified_payload_count += count
            exact_continuation_count += 1
        for horizon_row in receipt["horizon_results"]:
            horizon = int(horizon_row["horizon"])
            parent_horizon = parent_seed_root / "horizons" / f"h-{horizon:03d}"
            replay_horizon = replay_seed_root / "horizons" / f"h-{horizon:03d}"
            _effects, count = _exact_window_record(
                parent_horizon / "effects.json",
                replay_horizon / "effects.json",
                parent_entry_root=parent_entry_root,
                replay_entry_root=replay_entry_root,
                hash_field="result_sha256",
                error_prefix="SST_INDEPENDENT_BRANCH_EFFECT",
            )
            verified_payload_count += count
            exact_effect_count += 1
            for branch, state_row in horizon_row["states"].items():
                state, count = _exact_window_record(
                    parent_horizon / f"{branch}-state.json",
                    replay_horizon / f"{branch}-state.json",
                    parent_entry_root=parent_entry_root,
                    replay_entry_root=replay_entry_root,
                    hash_field="result_sha256",
                    error_prefix="SST_INDEPENDENT_BRANCH_HORIZON_STATE",
                )
                verified_payload_count += count
                exact_horizon_state_count += 1
                state_id = str(state["state"]["state_id"])
                # The directory is identified by the versioned contract id, not its hash.
                probe_contract_candidates = list(
                    (replay_entry_root / "probe-observations").glob(f"*/{state_id}.json")
                )
                require(len(probe_contract_candidates) == 1, f"SST_INDEPENDENT_BRANCH_PROBE_NOT_UNIQUE:{state_id}")
                replay_probe = probe_contract_candidates[0]
                parent_probe = parent_entry_root / "probe-observations" / replay_probe.parent.name / replay_probe.name
                probe, count = _exact_window_record(
                    parent_probe,
                    replay_probe,
                    parent_entry_root=parent_entry_root,
                    replay_entry_root=replay_entry_root,
                    hash_field="result_sha256",
                    error_prefix="SST_INDEPENDENT_BRANCH_PROBE",
                )
                require(probe["result_sha256"] == state_row["probe_result_sha256"], "SST_INDEPENDENT_BRANCH_PROBE_RECEIPT_MISMATCH")
                verified_payload_count += count
                exact_probe_count += 1
        exact_seed_count += 1

    require(exact_seed_count == len(expected_requests), "SST_INDEPENDENT_BRANCH_SEED_COUNT_MISMATCH")
    for entry_id in entries:
        for root, prefix in (
            (parent_branch_root, "PARENT"),
            (replay_branch_root, "REPLAY"),
        ):
            validation_path = root / entry_id / "stepwise_causal_branch_gfg_validation.json"
            validation = read_json(validation_path)
            _verify_hash(
                validation,
                "validation_sha256",
                f"SST_INDEPENDENT_BRANCH_{prefix}_GFG_VALIDATION_HASH_INVALID:{entry_id}",
            )
            require(validation["status"] == "PASS", f"SST_INDEPENDENT_BRANCH_{prefix}_GFG_NOT_PASS:{entry_id}")

    material = {
        "schema": "nanogpt-stepwise-independent-branch-replay-validation-v1",
        "status": "PASS",
        "parent_branch_root": str(parent_branch_root.resolve()),
        "replay_branch_root": str(replay_branch_root.resolve()),
        "replay_branch_audit_sha256": audit["audit_sha256"],
        "exact_seed_count": exact_seed_count,
        "exact_continuation_count": exact_continuation_count,
        "exact_horizon_state_count": exact_horizon_state_count,
        "exact_effect_count": exact_effect_count,
        "exact_probe_count": exact_probe_count,
        "verified_content_addressed_payload_reference_count": verified_payload_count,
        "four_branch_replay": "EXACT",
        "branch_gfg_independent_validation": "PASS",
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result
