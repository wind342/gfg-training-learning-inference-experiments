from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_support_transition_v1.selection import LEGACY_MATCH_FEATURES


def _row_map(value: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["optimizer_step"]): row for row in value["rows"]}


def _inside_interval(step: int, interval: dict[str, Any]) -> bool:
    end = int(interval.get("recovery") or interval["end"])
    return int(interval["start"]) <= step <= end


def _vector(
    legacy_row: dict[str, Any],
    csrg_row: dict[str, Any],
    csrg_names: tuple[str, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in LEGACY_MATCH_FEATURES:
        value = legacy_row["features"].get(name)
        require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"SST_LEGACY_FEATURE_INVALID:{name}")
        result["legacy:" + name] = float(value)
    for name in csrg_names:
        value = csrg_row["features"].get(name)
        require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"SST_CSRG_FEATURE_INVALID:{name}")
        result["csrg:" + name] = float(value)
    return result


def _window_id(source_bundle_id: str, entry_id: str, start: int, end: int) -> str:
    return "window-" + payload_sha256(
        {
            "entry_id": entry_id,
            "end_optimizer_step": int(end),
            "source_bundle_id": source_bundle_id,
            "start_optimizer_step": int(start),
        }
    )[:24]


def freeze_stepwise_selection(
    *,
    stability_feature_cache: Path,
    csrg_feature_index: Path,
    source_archive_manifest: Path,
    protocol_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    window_output: Path,
    pairing_output: Path,
    budget_output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Freeze all windows and matches without reading any stepwise result."""

    from .contracts import ComponentRegistry, ProbeContract

    component_registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, component_registry)
    probe_forward_count = probe_contract.baseline_repetitions + len(probe_contract.gate_sets)

    legacy = read_json(stability_feature_cache)
    csrg = read_json(csrg_feature_index)
    archive = read_json(source_archive_manifest)
    protocol_sha = file_sha256(protocol_path)
    source_bundles = {
        str(row["entry_id"]): str(row["gfg_bundle_id"])
        for row in archive["support_bundles"]
    }
    require(len(source_bundles) == 13, "SST_SOURCE_BUNDLE_COUNT_INVALID")
    csrg_by_entry = {str(row["entry_id"]): row for row in csrg["entries"]}
    legacy_by_entry = {str(row["entry_id"]): row for row in legacy["runs"]}
    require(set(source_bundles) == set(csrg_by_entry) == set(legacy_by_entry), "SST_ENTRY_SET_MISMATCH")
    csrg_names = tuple(sorted(csrg["feature_definitions"]["all_derived_feature_names"]))

    merged: dict[str, dict[int, dict[str, Any]]] = {}
    standardization_rows: list[dict[str, float]] = []
    for entry_id in sorted(legacy_by_entry):
        run = legacy_by_entry[entry_id]
        legacy_rows = _row_map(run)
        csrg_rows = _row_map(csrg_by_entry[entry_id])
        require(set(legacy_rows) == set(csrg_rows) == set(range(100, 10001, 100)), f"SST_GRID_MISMATCH:{entry_id}")
        merged[entry_id] = {}
        for step in sorted(legacy_rows):
            values = _vector(legacy_rows[step], csrg_rows[step], csrg_names)
            merged[entry_id][step] = {
                "legacy": legacy_rows[step],
                "csrg": csrg_rows[step],
                "values": values,
            }
            if step >= int(run["formation_transition_step"]):
                standardization_rows.append(values)

    feature_names = tuple(sorted(standardization_rows[0]))
    matrix = np.asarray([[row[name] for name in feature_names] for row in standardization_rows], dtype=np.float64)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    admitted = scales > 1e-12
    admitted_names = tuple(name for name, keep in zip(feature_names, admitted) if keep)
    means = means[admitted]
    scales = scales[admitted]
    require(len(admitted_names) >= 20, "SST_MATCH_FEATURES_DEGENERATE")

    def standardized(row: dict[str, float]) -> np.ndarray:
        return (np.asarray([row[name] for name in admitted_names], dtype=np.float64) - means) / scales

    stable_candidates: list[dict[str, Any]] = []
    for entry_id in sorted(legacy_by_entry):
        run = legacy_by_entry[entry_id]
        formation = int(run["formation_transition_step"])
        intervals = list(run["stability_intervals"])
        for step, row in sorted(merged[entry_id].items()):
            if step < formation or step >= 10000:
                continue
            current_accuracy = float(row["legacy"]["features"]["validation_accuracy"])
            next_accuracy = float(merged[entry_id][step + 100]["legacy"]["features"]["validation_accuracy"])
            if current_accuracy < 0.90 or next_accuracy < 0.90:
                continue
            if any(_inside_interval(step, value) or int(value["start"]) == step + 100 for value in intervals):
                continue
            stable_candidates.append(
                {
                    "entry_id": entry_id,
                    "end_optimizer_step": step + 100,
                    "start_optimizer_step": step,
                    "vector": standardized(row["values"]),
                }
            )
    require(len(stable_candidates) >= 72, "SST_STABLE_CANDIDATE_POOL_TOO_SMALL")

    decline_events: list[dict[str, Any]] = []
    for entry_id in sorted(legacy_by_entry):
        run = legacy_by_entry[entry_id]
        formation = int(run["formation_transition_step"])
        for ordinal, interval in enumerate(run["stability_intervals"]):
            start = int(interval["start"])
            pre = start - 100
            if pre < formation or pre not in merged[entry_id] or start not in merged[entry_id]:
                continue
            decline_events.append(
                {
                    "decline_event_id": "decline-" + payload_sha256(
                        {"entry_id": entry_id, "interval": interval, "ordinal": ordinal}
                    )[:24],
                    "entry_id": entry_id,
                    "interval": interval,
                    "pre_step": pre,
                    "source_interval_ordinal": ordinal,
                    "vector": standardized(merged[entry_id][pre]["values"]),
                }
            )
    require(len(decline_events) == 72, f"SST_LEGAL_DECLINE_COUNT_INVALID:{len(decline_events)}")

    # Greedy use balancing is frozen; future labels only define the admitted stable pool.
    use_counts: dict[tuple[str, int], int] = {}
    pairings: list[dict[str, Any]] = []
    window_roles: dict[tuple[str, int, int], set[str]] = {}
    for event in sorted(decline_events, key=lambda row: row["decline_event_id"]):
        decline_key = (event["entry_id"], int(event["pre_step"]), int(event["pre_step"]) + 100)
        window_roles.setdefault(decline_key, set()).add("pre_decline")
        ranked: list[tuple[int, float, str, int, dict[str, Any]]] = []
        for candidate in stable_candidates:
            key = (str(candidate["entry_id"]), int(candidate["start_optimizer_step"]))
            if key == (event["entry_id"], int(event["pre_step"])):
                continue
            distance = float(np.sqrt(np.mean((candidate["vector"] - event["vector"]) ** 2)))
            ranked.append((use_counts.get(key, 0), distance, key[0], key[1], candidate))
        require(bool(ranked), f"SST_STABLE_MATCH_MISSING:{event['decline_event_id']}")
        reuse_count, distance, _entry, _step, selected = min(ranked)
        selected_key = (str(selected["entry_id"]), int(selected["start_optimizer_step"]))
        use_counts[selected_key] = reuse_count + 1
        stable_window_key = (
            str(selected["entry_id"]),
            int(selected["start_optimizer_step"]),
            int(selected["end_optimizer_step"]),
        )
        window_roles.setdefault(stable_window_key, set()).add("matched_stable")
        decline_window_id = _window_id(source_bundles[decline_key[0]], *decline_key)
        stable_window_id = _window_id(source_bundles[stable_window_key[0]], *stable_window_key)
        pair_material = {
            "decline_event_id": event["decline_event_id"],
            "decline_window_id": decline_window_id,
            "stable_window_id": stable_window_id,
        }
        pairings.append(
            {
                **pair_material,
                "pair_id": "pair-" + payload_sha256(pair_material)[:24],
                "matching_distance": distance,
                "stable_candidate_prior_use_count": reuse_count,
                "future_outcome_used_only_to_define_window_categories": True,
            }
        )

    recovery_rows: list[dict[str, Any]] = []
    for entry_id in sorted(legacy_by_entry):
        complete = [row for row in legacy_by_entry[entry_id]["stability_intervals"] if row.get("recovery") is not None]
        require(bool(complete), f"SST_RECOVERY_WINDOW_MISSING:{entry_id}")
        selected = min(complete, key=lambda row: (float(row["minimum"]), int(row["start"])))
        key = (entry_id, int(selected["start"]), int(selected["recovery"]))
        window_roles.setdefault(key, set()).add("recovery")
        recovery_rows.append(
            {
                "entry_id": entry_id,
                "interval": selected,
                "window_id": _window_id(source_bundles[entry_id], *key),
            }
        )

    windows = []
    for entry_id, start, end in sorted(window_roles):
        require(0 <= start < end <= 10000, f"SST_WINDOW_BOUNDARY_INVALID:{entry_id}:{start}:{end}")
        windows.append(
            {
                "categories": sorted(window_roles[(entry_id, start, end)]),
                "entry_id": entry_id,
                "source_bundle_id": source_bundles[entry_id],
                "start_optimizer_step": start,
                "end_optimizer_step": end,
                "training_transition_count": end - start,
                "state_count": end - start + 1,
                "window_id": _window_id(source_bundles[entry_id], entry_id, start, end),
            }
        )
    require(sum("pre_decline" in row["categories"] for row in windows) == 72, "SST_DECLINE_WINDOWS_DEDUPED_UNEXPECTEDLY")
    require(sum("recovery" in row["categories"] for row in windows) == 13, "SST_RECOVERY_WINDOW_COUNT_INVALID")

    replay_pairs = []
    for entry_id in sorted(source_bundles):
        candidates = [row for row in pairings if next(e for e in decline_events if e["decline_event_id"] == row["decline_event_id"])["entry_id"] == entry_id]
        require(bool(candidates), f"SST_REPLAY_PAIR_MISSING:{entry_id}")
        selected = min(
            candidates,
            key=lambda row: hashlib.sha256(f"{protocol_sha}\0{entry_id}\0{row['pair_id']}".encode("utf-8")).hexdigest(),
        )
        replay_pairs.append({"entry_id": entry_id, "pair_id": selected["pair_id"]})

    window_material = {
        "schema": "nanogpt-stepwise-window-selection-v1",
        "status": "FROZEN_BEFORE_STEPWISE_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "source_archive_manifest_sha256": file_sha256(source_archive_manifest),
        "stability_feature_cache_sha256": file_sha256(stability_feature_cache),
        "csrg_feature_index_sha256": file_sha256(csrg_feature_index),
        "selection_rules_read_stepwise_results": False,
        "future_outcome_admitted_as_model_input": False,
        "windows": windows,
        "recovery_selections": recovery_rows,
    }
    window_result = {**window_material, "selection_sha256": payload_sha256(window_material)}
    pairing_material = {
        "schema": "nanogpt-stepwise-counterexample-pairing-v1",
        "status": "FROZEN_BEFORE_STEPWISE_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "window_selection_sha256": window_result["selection_sha256"],
        "distance_feature_names": list(admitted_names),
        "distance_standardization": "global post-formation population mean/std frozen before stepwise execution",
        "stable_candidate_use_cap": max(use_counts.values()),
        "pairings": sorted(pairings, key=lambda row: row["pair_id"]),
        "independent_replay_pairs": replay_pairs,
    }
    pairing_result = {**pairing_material, "pairing_sha256": payload_sha256(pairing_material)}

    total_states = sum(int(row["state_count"]) for row in windows)
    total_steps = sum(int(row["training_transition_count"]) for row in windows)
    probe_forwards = total_states * probe_forward_count
    branch_seed_ceiling = len(pairings) * 2 * 3
    budget_material = {
        "schema": "nanogpt-stepwise-compute-budget-v1",
        "status": "FROZEN_CEILING_BEFORE_STEPWISE_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "window_selection_sha256": window_result["selection_sha256"],
        "unique_window_count": len(windows),
        "main_training_transition_count": total_steps,
        "main_restorable_state_count": total_states,
        "main_probe_forward_count": probe_forwards,
        "probe_forward_count_per_state": probe_forward_count,
        "component_registry_id": component_registry.registry_id,
        "component_registry_sha256": component_registry.source_sha256,
        "probe_contract_id": probe_contract.probe_contract_id,
        "probe_contract_sha256": probe_contract.source_sha256,
        "key_step_branch_seed_ceiling": branch_seed_ceiling,
        "key_step_branch_training_opportunity_ceiling": branch_seed_ceiling * 4 * 100,
        "key_step_branch_probe_forward_ceiling": branch_seed_ceiling * 4 * 7 * probe_forward_count,
        "independent_replay_pair_count": len(replay_pairs),
        "budget_policy": "The pipeline may execute fewer key-step branches after boundary deduplication, but never more than this frozen ceiling. Main windows are exact, not a ceiling.",
    }
    budget_result = {**budget_material, "budget_sha256": payload_sha256(budget_material)}
    write_json(window_output, window_result)
    write_json(pairing_output, pairing_result)
    write_json(budget_output, budget_result)
    return window_result, pairing_result, budget_result


def freeze_phase_selection(
    *,
    parent_window_selection_path: Path,
    parent_pairing_path: Path,
    protocol_path: Path,
    finite_difference_protocol_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    window_output: Path,
    pairing_output: Path,
    budget_output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extend frozen v1 scientific windows with pre-cut history, without reselection."""

    from .contracts import ComponentRegistry, ProbeContract

    parent_windows = read_json(parent_window_selection_path)
    parent_pairing = read_json(parent_pairing_path)
    protocol = read_json(protocol_path)
    finite_difference_protocol = read_json(finite_difference_protocol_path)
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    require(
        parent_windows["schema"] == "nanogpt-stepwise-window-selection-v1",
        "SST_PHASE_PARENT_WINDOW_SCHEMA_INVALID",
    )
    require(
        parent_pairing["schema"] == "nanogpt-stepwise-counterexample-pairing-v1",
        "SST_PHASE_PARENT_PAIRING_SCHEMA_INVALID",
    )
    require(
        parent_pairing["window_selection_sha256"] == parent_windows["selection_sha256"],
        "SST_PHASE_PARENT_SELECTION_DRIFT",
    )
    require(protocol["schema"] == "nanogpt-stepwise-support-transition-protocol-v2", "SST_PHASE_PROTOCOL_INVALID")
    require(
        finite_difference_protocol["parent_protocol_id"] == protocol["protocol_id"],
        "SST_PHASE_FINITE_DIFFERENCE_PROTOCOL_DRIFT",
    )
    lookbacks = tuple(int(value) for value in protocol["window_policy"]["lookback_scales"])
    require(lookbacks == (1, 2, 5, 10), "SST_PHASE_LOOKBACK_SCALES_INVALID")
    prehistory = int(protocol["window_policy"]["captured_prehistory_steps"])
    require(prehistory == max(lookbacks), "SST_PHASE_PREHISTORY_INSUFFICIENT")
    protocol_sha = file_sha256(protocol_path)
    fd_protocol_sha = file_sha256(finite_difference_protocol_path)

    def phase_window_id(parent: dict[str, Any], capture_start: int) -> str:
        return "phase-window-" + payload_sha256(
            {
                "capture_start_optimizer_step": capture_start,
                "entry_id": parent["entry_id"],
                "parent_window_id": parent["window_id"],
                "protocol_sha256": protocol_sha,
                "scientific_end_optimizer_step": int(parent["end_optimizer_step"]),
                "scientific_start_optimizer_step": int(parent["start_optimizer_step"]),
            }
        )[:24]

    windows: list[dict[str, Any]] = []
    window_id_map: dict[str, str] = {}
    for parent in parent_windows["windows"]:
        scientific_start = int(parent["start_optimizer_step"])
        scientific_end = int(parent["end_optimizer_step"])
        capture_start = scientific_start - prehistory
        restore_step = scientific_start - 100
        require(scientific_start % 100 == 0, f"SST_PHASE_SCIENTIFIC_START_UNALIGNED:{parent['window_id']}")
        require(scientific_end > scientific_start, f"SST_PHASE_SCIENTIFIC_WINDOW_EMPTY:{parent['window_id']}")
        require(restore_step >= 0 and restore_step < capture_start, f"SST_PHASE_RESTORE_BOUNDARY_INVALID:{parent['window_id']}")
        new_id = phase_window_id(parent, capture_start)
        window_id_map[str(parent["window_id"])] = new_id
        windows.append(
            {
                "categories": list(parent["categories"]),
                "entry_id": parent["entry_id"],
                "source_bundle_id": parent["source_bundle_id"],
                "parent_window_id": parent["window_id"],
                "window_id": new_id,
                "restore_optimizer_step": restore_step,
                "capture_start_optimizer_step": capture_start,
                "scientific_start_optimizer_step": scientific_start,
                "scientific_end_optimizer_step": scientific_end,
                "capture_end_optimizer_step": scientific_end,
                "start_optimizer_step": scientific_start,
                "end_optimizer_step": scientific_end,
                "lookback_scales": list(lookbacks),
                "captured_prehistory_steps": prehistory,
                "replay_warmup_transition_count": capture_start - restore_step,
                "training_transition_count": scientific_end - capture_start,
                "scientific_transition_count": scientific_end - scientific_start,
                "state_count": scientific_end - capture_start + 1,
            }
        )
    require(len(windows) == len(parent_windows["windows"]), "SST_PHASE_WINDOW_COUNT_DRIFT")

    recovery_selections: list[dict[str, Any]] = []
    for row in parent_windows.get("recovery_selections", []):
        recovery_selections.append({**row, "parent_window_id": row["window_id"], "window_id": window_id_map[str(row["window_id"])]})

    window_material = {
        "schema": "nanogpt-stepwise-phase-window-selection-v2",
        "status": "FROZEN_BEFORE_FORMAL_V2_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "finite_difference_protocol_sha256": fd_protocol_sha,
        "parent_window_selection_sha256": parent_windows["selection_sha256"],
        "parent_window_selection_file_sha256": file_sha256(parent_window_selection_path),
        "selection_rules_read_formal_v2_results": False,
        "future_outcome_admitted_as_model_input": False,
        "scientific_windows_reselected": False,
        "windows": sorted(windows, key=lambda row: (row["entry_id"], row["scientific_start_optimizer_step"], row["window_id"])),
        "recovery_selections": recovery_selections,
    }
    window_result = {**window_material, "selection_sha256": payload_sha256(window_material)}

    pair_id_map: dict[str, str] = {}
    pairings: list[dict[str, Any]] = []
    for row in parent_pairing["pairings"]:
        material = {
            "parent_pair_id": row["pair_id"],
            "decline_window_id": window_id_map[str(row["decline_window_id"])],
            "stable_window_id": window_id_map[str(row["stable_window_id"])],
        }
        new_pair_id = "phase-pair-" + payload_sha256(material)[:24]
        pair_id_map[str(row["pair_id"])] = new_pair_id
        pairings.append(
            {
                **row,
                **material,
                "pair_id": new_pair_id,
                "matching_recomputed_from_formal_v2_results": False,
            }
        )
    replay_pairs = [
        {**row, "parent_pair_id": row["pair_id"], "pair_id": pair_id_map[str(row["pair_id"])]}
        for row in parent_pairing["independent_replay_pairs"]
    ]
    pairing_material = {
        "schema": "nanogpt-stepwise-phase-counterexample-pairing-v2",
        "status": "FROZEN_BEFORE_FORMAL_V2_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "finite_difference_protocol_sha256": fd_protocol_sha,
        "window_selection_sha256": window_result["selection_sha256"],
        "parent_pairing_sha256": parent_pairing["pairing_sha256"],
        "distance_feature_names": parent_pairing["distance_feature_names"],
        "distance_standardization": parent_pairing["distance_standardization"],
        "stable_candidate_use_cap": parent_pairing["stable_candidate_use_cap"],
        "pairings": sorted(pairings, key=lambda row: row["pair_id"]),
        "independent_replay_pairs": replay_pairs,
    }
    pairing_result = {**pairing_material, "pairing_sha256": payload_sha256(pairing_material)}

    probe_forward_count = probe_contract.baseline_repetitions + len(probe_contract.gate_sets)
    total_states = sum(int(row["state_count"]) for row in windows)
    total_captured_steps = sum(int(row["training_transition_count"]) for row in windows)
    total_warmup_steps = sum(int(row["replay_warmup_transition_count"]) for row in windows)
    branch_seed_ceiling = len(pairings) * 2 * 3
    budget_material = {
        "schema": "nanogpt-stepwise-phase-compute-budget-v2",
        "status": "FROZEN_CEILING_BEFORE_FORMAL_V2_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "finite_difference_protocol_sha256": fd_protocol_sha,
        "window_selection_sha256": window_result["selection_sha256"],
        "unique_window_count": len(windows),
        "replay_warmup_training_transition_count": total_warmup_steps,
        "captured_training_transition_count": total_captured_steps,
        "scientific_training_transition_count": sum(int(row["scientific_transition_count"]) for row in windows),
        "main_restorable_state_count": total_states,
        "main_probe_forward_count": total_states * probe_forward_count,
        "probe_forward_count_per_state": probe_forward_count,
        "component_registry_id": registry.registry_id,
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_id": probe_contract.probe_contract_id,
        "probe_contract_sha256": probe_contract.source_sha256,
        "lookback_scales": list(lookbacks),
        "key_step_branch_seed_ceiling": branch_seed_ceiling,
        "key_step_branch_training_opportunity_ceiling": branch_seed_ceiling * 4 * 100,
        "key_step_branch_probe_forward_ceiling": branch_seed_ceiling * 4 * 7 * probe_forward_count,
        "independent_replay_pair_count": len(replay_pairs),
        "budget_policy": "The scientific windows and pairings are inherited exactly from v1. The 10-step prehistory and 90-step replay warm-up are mandatory for every formal-v2 window.",
    }
    budget_result = {**budget_material, "budget_sha256": payload_sha256(budget_material)}
    write_json(window_output, window_result)
    write_json(pairing_output, pairing_result)
    write_json(budget_output, budget_result)
    return window_result, pairing_result, budget_result


def rebind_phase_selection_protocol(
    *,
    parent_window_selection_path: Path,
    parent_pairing_path: Path,
    parent_budget_path: Path,
    protocol_path: Path,
    window_output: Path,
    pairing_output: Path,
    budget_output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebind frozen V2 windows to V3 after a source-scope failure, without reselection."""

    parent_windows = read_json(parent_window_selection_path)
    parent_pairing = read_json(parent_pairing_path)
    parent_budget = read_json(parent_budget_path)
    protocol = read_json(protocol_path)
    require(parent_windows["schema"] == "nanogpt-stepwise-phase-window-selection-v2", "SST_V3_PARENT_WINDOW_SCHEMA_INVALID")
    require(parent_pairing["schema"] == "nanogpt-stepwise-phase-counterexample-pairing-v2", "SST_V3_PARENT_PAIRING_SCHEMA_INVALID")
    require(parent_budget["schema"] == "nanogpt-stepwise-phase-compute-budget-v2", "SST_V3_PARENT_BUDGET_SCHEMA_INVALID")
    require(protocol["schema"] == "nanogpt-stepwise-support-transition-protocol-v3", "SST_V3_PROTOCOL_INVALID")
    require(
        protocol["inherits_protocol"]["file_sha256"] == parent_windows["protocol_sha256"],
        "SST_V3_PROTOCOL_INHERITANCE_DRIFT",
    )
    require(parent_pairing["window_selection_sha256"] == parent_windows["selection_sha256"], "SST_V3_PARENT_PAIRING_DRIFT")
    require(parent_budget["window_selection_sha256"] == parent_windows["selection_sha256"], "SST_V3_PARENT_BUDGET_DRIFT")
    protocol_sha = file_sha256(protocol_path)
    window_id_map: dict[str, str] = {}
    windows: list[dict[str, Any]] = []
    for parent in parent_windows["windows"]:
        parent_id = str(parent["window_id"])
        new_id = "phase-v3-window-" + payload_sha256(
            {
                "parent_phase_v2_window_id": parent_id,
                "protocol_sha256": protocol_sha,
                "source_scope_change_only": True,
            }
        )[:24]
        window_id_map[parent_id] = new_id
        windows.append({**parent, "parent_phase_v2_window_id": parent_id, "window_id": new_id})
    recovery = [
        {
            **row,
            "parent_phase_v2_window_id": row["window_id"],
            "window_id": window_id_map[str(row["window_id"])],
        }
        for row in parent_windows["recovery_selections"]
    ]
    window_material = {
        "schema": "nanogpt-stepwise-phase-window-selection-v3",
        "status": "FROZEN_BEFORE_FORMAL_V3_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "finite_difference_protocol_sha256": parent_windows["finite_difference_protocol_sha256"],
        "parent_phase_v2_selection_sha256": parent_windows["selection_sha256"],
        "parent_phase_v2_selection_file_sha256": file_sha256(parent_window_selection_path),
        "source_scope_change_only": True,
        "selection_rules_read_formal_v2_or_v3_results": False,
        "future_outcome_admitted_as_model_input": False,
        "scientific_windows_reselected": False,
        "windows": sorted(windows, key=lambda row: (row["entry_id"], row["scientific_start_optimizer_step"], row["window_id"])),
        "recovery_selections": recovery,
    }
    window_result = {**window_material, "selection_sha256": payload_sha256(window_material)}

    pair_id_map: dict[str, str] = {}
    pairings: list[dict[str, Any]] = []
    for parent in parent_pairing["pairings"]:
        parent_id = str(parent["pair_id"])
        material = {
            "parent_phase_v2_pair_id": parent_id,
            "decline_window_id": window_id_map[str(parent["decline_window_id"])],
            "stable_window_id": window_id_map[str(parent["stable_window_id"])],
            "protocol_sha256": protocol_sha,
        }
        new_id = "phase-v3-pair-" + payload_sha256(material)[:24]
        pair_id_map[parent_id] = new_id
        pairings.append(
            {
                **parent,
                **material,
                "pair_id": new_id,
                "matching_recomputed_from_formal_v2_or_v3_results": False,
            }
        )
    replay_pairs = [
        {
            **row,
            "parent_phase_v2_pair_id": row["pair_id"],
            "pair_id": pair_id_map[str(row["pair_id"])],
        }
        for row in parent_pairing["independent_replay_pairs"]
    ]
    pairing_material = {
        "schema": "nanogpt-stepwise-phase-counterexample-pairing-v3",
        "status": "FROZEN_BEFORE_FORMAL_V3_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "finite_difference_protocol_sha256": parent_pairing["finite_difference_protocol_sha256"],
        "window_selection_sha256": window_result["selection_sha256"],
        "parent_phase_v2_pairing_sha256": parent_pairing["pairing_sha256"],
        "distance_feature_names": parent_pairing["distance_feature_names"],
        "distance_standardization": parent_pairing["distance_standardization"],
        "stable_candidate_use_cap": parent_pairing["stable_candidate_use_cap"],
        "pairings": sorted(pairings, key=lambda row: row["pair_id"]),
        "independent_replay_pairs": replay_pairs,
    }
    pairing_result = {**pairing_material, "pairing_sha256": payload_sha256(pairing_material)}

    excluded = {"schema", "status", "protocol_sha256", "window_selection_sha256", "budget_sha256"}
    inherited_budget = {key: value for key, value in parent_budget.items() if key not in excluded}
    budget_material = {
        "schema": "nanogpt-stepwise-phase-compute-budget-v3",
        "status": "FROZEN_CEILING_BEFORE_FORMAL_V3_RESULT_GENERATION",
        "protocol_sha256": protocol_sha,
        "window_selection_sha256": window_result["selection_sha256"],
        "parent_phase_v2_budget_sha256": parent_budget["budget_sha256"],
        **inherited_budget,
        "budget_policy": "All V2 scientific windows, pairs and numerical ceilings are inherited unchanged; only source batch-selection-order availability is handled explicitly.",
    }
    budget_result = {**budget_material, "budget_sha256": payload_sha256(budget_material)}
    write_json(window_output, window_result)
    write_json(pairing_output, pairing_result)
    write_json(budget_output, budget_result)
    return window_result, pairing_result, budget_result
